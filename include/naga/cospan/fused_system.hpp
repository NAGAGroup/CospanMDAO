#pragma once

// =============================================================================
// naga/cospan/fused_system.hpp — FusedSystem, sequential and parallel mode (r01 §5.4-§5.7, §7.5)
// =============================================================================
//
// CHILDREN ARE COMPONENTS ONLY (as ParallelSystem). Their
// bodies run as the stages of ONE outer kernel, the "wave": a chained executor
// walks a call sequence of the children's stage functions (kb_fn<KB>) with a
// shared record cursor, so every stage function reads its own StageRecord by
// position (§5.4). Sequential mode: every stage has group_offset 0 and the
// stages run one after the other inside each work-group.
//
// compile(): children compile; custom-op and host children are rejected (they
// have no stage function); all children must have the same kernel rank (the
// wave's rank); every entry with a value-driven size (FromInput) must be driven
// by a value that does not come from another stage of this fuse (wave_rules.hpp,
// check_value_driven_sizes: the value is read on the host before the launch).
// The dynamic_function_config is built ONCE, as a call sequence of the
// children's definition ids in child order. THERE ARE NO FRAMEWORK BARRIERS: a
// stage that reads, across work-items, what an earlier stage wrote calls its own
// group barrier first.
//
// One forward: pre-compute ALL; compute ALL (the descriptions stay
// alive to the end); validate; pack records and the three byte streams (16-byte
// aligned) into member USM buffers; ONE outer launch through the stream;
// post-compute ALL. Groups are laid along dimension 0.
//
// SEQUENTIAL MODE HAS NO INTER-STAGE ACCESS CHECKS. Stage ranges may differ, and
// a stage may read anything an earlier stage wrote: the framework runs the
// stages in child order inside each work-group and checks nothing about who
// reads what. The same work-item reads its own earlier writes; a read of another
// work-item's write needs the reader's own group barrier; across groups nothing
// is ordered.
//
// LOCAL MEMORY. Sequential: every scratch stage gets its OWN slice of the local
// buffer (StageRecord::local_off = the running sum of the earlier stages'
// scratch_bytes, each rounded up to 16) and the launch's local size is the sum.
// Parallel: the stages own disjoint groups, so each uses offset 0 and the local
// size is the max. Every forward, a local size above info::device::local_mem_size
// throws, naming the scratch stages and their byte counts.
//
// PARALLEL MODE. The stages are independent: each gets its OWN slice of groups,
// group_offset = the running sum of the previous stages' group counts, and the
// launch has the SUM of the group counts. Every forward, validate() applies the
// rules of a wave whose stages run unordered (wave_rules.hpp): NO CONNECTIONS
// between stages (an input whose root is an output of another stage throws,
// naming both; no override), and NO TIED OUTPUTS (two or more stages whose
// outputs share one root, e.g. through tie_outputs, throw, naming them) unless
// allow_tied_outputs() was called.
//
// Include order: this header includes system.hpp, whose helper form
// (property::fused) needs FusedSystem complete when it is instantiated, so
// include naga/cospan/cospan.hpp (or this header).

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "naga/cospan/system.hpp"
#include "naga/cospan/wave_rules.hpp"

namespace naga::cospan {

class FusedSystem : public System {
public:
    using Mode = property::fused::Mode;

    /// `wg`: the wave's work-group shape; absent, it is the stages' common
    /// requested shape, else the default for the wave's rank.
    explicit FusedSystem(Mode mode = Mode::sequential, std::optional<sycl::range<3>> wg = {})
        : mode_{mode}, wg_{wg} {}
    ~FusedSystem() override = default;

    /// Parallel mode: assert that stages whose outputs share one root (tie_outputs)
    /// do not race (atomic writers, say); the framework then skips that check.
    /// The no-connections rule has no override. No effect in sequential mode.
    void allow_tied_outputs() { allow_tied_ = true; }

    // ---- adding children: Components only, at compile time ----------------------------
    template <class T, class... Args>
        requires(std::derived_from<T, ComponentBase> && !detail::leads_with_props_v<Args...>)
    T& add_subsystem(std::string name, Args&&... ctor_args) {
        return System::add_subsystem<T>(std::move(name), std::forward<Args>(ctor_args)...);
    }

    /// A FusedSystem has no untyped children to add, and no property overload.
    System& add_subsystem(std::string, const property_list& = {}) = delete;

    // ---- compile -----------------------------------------------------------------------------
    void compile() override {
        System::compile();  // the children compile first
        const std::size_t n = children_.size();
        if (n == 0) throw std::invalid_argument("naga::cospan: " + describe() + ": a FusedSystem has no children");

        rank_ = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const ComponentBase& c = stage(i);
            if (c.is_custom_op() || c.is_host())
                throw std::invalid_argument("naga::cospan: " + describe() + ": child '" + c.path() +
                                            "' is a " + (c.is_host() ? "host" : "custom-op") +
                                            " body; those have no stage function and cannot be fused");
            if (child_ctx_[i].device != ctx_.device)
                throw std::invalid_argument("naga::cospan: " + describe() + ": child '" + c.path() +
                                            "' is placed on another device; a fused wave is one kernel on one device");
            if (i == 0)
                rank_ = c.kernel_rank();
            else if (c.kernel_rank() != rank_)
                throw std::invalid_argument("naga::cospan: " + describe() + ": child '" + c.path() + "' has rank " +
                                            std::to_string(c.kernel_rank()) + " but '" + stage(0).path() +
                                            "' has rank " + std::to_string(rank_) +
                                            "; the stages of a wave share one rank");
        }

        // no ragged ports in a fuse yet.
        detail::check_no_ragged_ports(describe(), children());

        // (h) a value-driven size may not be driven by a stage of this fuse (both modes).
        detail::check_value_driven_sizes(describe(), children());

        // The call sequence, built ONCE: the definition ids in child order. No barriers.
        std::vector<detail::JitId> defs;
        for (std::size_t i = 0; i < n; ++i) defs.push_back(stage(i).definition_id());
        cfg_ = std::make_unique<sycl::AdaptiveCpp_jit::dynamic_function_config>();
        cfg_->define_as_call_sequence(detail::wave_entry_id(rank_), defs);
    }

    // ---- the runner --------------------------------------------------------------------------
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        const std::size_t n = children_.size();
        for (std::size_t i = 0; i < n; ++i) {
            materialize_slot(i);
            children_[i]->run_pre_compute(child_ctx_[i]);
        }

        std::vector<std::unique_ptr<LaunchDescription>> ds;  // alive until the wave is staged
        ds.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto d = children_[i]->run_compute_forward();
            if (!d)
                throw std::logic_error("naga::cospan: " + describe() + ": forward: child '" + children_[i]->path() +
                                       "' returned no launch description; a FusedSystem child must be a Component "
                                       "that describes its work");
            ds.push_back(std::move(d));
        }

        const std::vector<const KernelParts*> parts = validate(ds);
        submit_wave(parts);

        for (std::size_t i = 0; i < n; ++i) children_[i]->run_post_compute(child_ctx_[i]);
        return nullptr;  // "I submitted"
    }

private:
    const ComponentBase& stage(std::size_t i) const {
        const auto* c = dynamic_cast<const ComponentBase*>(children_[i].get());
        if (c == nullptr)
            throw std::logic_error("naga::cospan: " + describe() + ": child '" + children_[i]->path() +
                                   "' is not a Component");
        return *c;
    }

    /// Every description carries kernel parts of the wave's rank; in parallel mode
    /// the stages are not connected and share no output root (file header).
    std::vector<const KernelParts*> validate(const std::vector<std::unique_ptr<LaunchDescription>>& ds) const {
        const std::size_t               n = children_.size();
        std::vector<const KernelParts*> parts;
        for (std::size_t i = 0; i < n; ++i) {
            const KernelParts* p =
                (ds[i]->is_custom_op() || ds[i]->is_host()) ? nullptr : ds[i]->kernel();
            if (p == nullptr)
                throw std::logic_error("naga::cospan: " + describe() + ": forward: child '" + children_[i]->path() +
                                       "' described no kernel; a fused stage must be a kernel body");
            if (p->rank != rank_)
                throw std::logic_error("naga::cospan: " + describe() + ": forward: child '" + children_[i]->path() +
                                       "' described a rank-" + std::to_string(p->rank) + " launch; the wave is rank " +
                                       std::to_string(rank_));
            parts.push_back(p);
        }

        if (mode_ == Mode::parallel) {
            // (c) no connections between stages (no override), (d) no tied outputs unless allowed.
            detail::check_no_stage_connections(describe(), "parallel-mode wave", children());
            if (!allow_tied_) detail::check_tied_outputs(describe(), "parallel-mode wave", children());
        }
        // Sequential mode: no inter-stage access checks (rule (e)).
        return parts;
    }

    static std::uint32_t append(std::vector<std::byte>& v, std::span<const std::byte> s) {
        const std::size_t off = (v.size() + 15) / 16 * 16;  // 16-byte aligned; padding is zero
        v.resize(off + s.size());
        if (!s.empty()) std::memcpy(v.data() + off, s.data(), s.size());
        return static_cast<std::uint32_t>(off);
    }

    sycl::range<3> wave_wg(const std::vector<const KernelParts*>& parts) const {
        std::optional<sycl::range<3>> wg = wg_;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            if (!parts[i]->wg) continue;
            if (!wg)
                wg = parts[i]->wg;
            else if (*wg != *parts[i]->wg)
                throw std::invalid_argument("naga::cospan: " + describe() + ": forward: child '" +
                                            children_[i]->path() +
                                            "' asks for a work-group shape other than the wave's; the stages of a "
                                            "wave share one");
        }
        if (!wg) {
            switch (rank_) {
                case 1: wg = detail::default_wg<1>(); break;
                case 2: wg = detail::default_wg<2>(); break;
                default: wg = detail::default_wg<3>(); break;
            }
        }
        sycl::range<3> w = *wg;
        for (int d = rank_; d < 3; ++d) w[d] = 1;  // dimensions past the rank do not exist
        return w;
    }

    template <class T>
    sk::allocator<T> staging_allocator() const {
        if (ctx_.substrate != nullptr && ctx_.substrate->pool) return sk::allocator<T>{*ctx_.queue, *ctx_.substrate->pool};
        return sk::allocator<T>{*ctx_.queue, sk::kind::device, ctx_.device};
    }

    template <class T>
    void upload(std::optional<sk::usm_vector<T>>& v, const std::vector<T>& src) {
        if (!v) v.emplace(staging_allocator<T>());
        diag::count_host_tasks();
        v->from_host(std::span<const T>{src});  // a routed, stream-ordered copy of a task-owned snapshot
    }

    void submit_wave(const std::vector<const KernelParts*>& parts) {
        const sycl::range<3> w       = wave_wg(parts);
        const std::size_t    wg_size = w[0] * w[1] * w[2];

        std::vector<StageRecord> recs;
        std::vector<std::byte>   in, out, body;
        std::size_t              total_groups = 0, local_bytes = 0;
        for (const KernelParts* p : parts) {
            StageRecord r  = p->record;
            // group_offset is stamped below, once this stage's group_count is known
            if (p->group_body) {
                for (int d = 0; d < 3; ++d) r.group_range[d] = (r.range[d] + w[d] - 1) / w[d];
                r.group_count = r.group_range[0] * r.group_range[1] * r.group_range[2];
            } else {
                r.group_count = (r.flat_extent + wg_size - 1) / wg_size;
                r.group_range = sycl::range<3>{r.group_count, 1, 1};
            }
            r.input_off   = append(in, p->inputs);
            r.output_off  = append(out, p->outputs);
            r.body_off    = append(body, p->body);
            r.scratch_off = p->has_scratch ? r.body_off + p->record.scratch_off : 0;  // record's is 0-relative to the body
            if (mode_ == Mode::parallel) {
                r.group_offset = total_groups;  // this stage's slice starts after all previous slices
                total_groups += r.group_count;  // the launch is the sum of the slices
            } else {
                r.group_offset = 0;  // sequential: every stage starts at group 0
                total_groups   = std::max(total_groups, r.group_count);
            }
            if (mode_ == Mode::sequential) {
                // Every scratch stage has its own slice: the running sum, each rounded up to 16.
                r.local_off = static_cast<std::uint32_t>(local_bytes);
                local_bytes += (std::size_t{r.scratch_bytes} + 15) / 16 * 16;
            } else {
                // Parallel stages own disjoint groups: they share one slice at offset 0, the launch needs the max.
                r.local_off = 0;
                local_bytes = std::max<std::size_t>(local_bytes, r.scratch_bytes);
            }
            recs.push_back(r);
        }
        const std::size_t local_limit = ctx_.device.get_info<sycl::info::device::local_mem_size>();
        if (local_bytes > local_limit) {
            std::string scratch_stages;
            for (std::size_t i = 0; i < recs.size(); ++i)
                if (recs[i].scratch_bytes != 0)
                    scratch_stages += (scratch_stages.empty() ? "'" : ", '") + children_[i]->path() + "' (" +
                                      std::to_string(recs[i].scratch_bytes) + " bytes)";
            throw std::invalid_argument(
                "naga::cospan: " + describe() + ": forward: the wave needs " + std::to_string(local_bytes) +
                " bytes of local memory (" +
                (mode_ == Mode::sequential ? "the sum of its stages' slices" : "the largest stage's slice, shared") +
                ") but the device offers " + std::to_string(local_limit) + "; scratch stages: " + scratch_stages);
        }
        if (total_groups == 0) return;  // every stage is empty

        upload(recs_, recs);
        upload(in_, in);
        upload(out_, out);
        upload(body_, body);

        SubmitTarget       t  = ctx_.target();
        const StageRecord* rp = recs_->data();
        const std::byte*   ip = in_->data();
        std::byte*         op = out_->data();
        const std::byte*   bp = body_->data();
        switch (rank_) {
            case 1:
                detail::launch_wave<1>(t, *cfg_, detail::wave_nd_range<1>(total_groups, w), rp, ip, op, bp, local_bytes);
                break;
            case 2:
                detail::launch_wave<2>(t, *cfg_, detail::wave_nd_range<2>(total_groups, w), rp, ip, op, bp, local_bytes);
                break;
            default:
                detail::launch_wave<3>(t, *cfg_, detail::wave_nd_range<3>(total_groups, w), rp, ip, op, bp, local_bytes);
                break;
        }
    }

    Mode                          mode_;
    bool                          allow_tied_ = false;  // parallel mode: stages may share an output root
    std::optional<sycl::range<3>> wg_;
    int                           rank_ = 0;
    std::unique_ptr<sycl::AdaptiveCpp_jit::dynamic_function_config> cfg_;

    // Staging buffers, kept for the System's life; each forward overwrites them
    // (stream-ordered after the previous wave's kernel, SlabKit's contract).
    std::optional<sk::usm_vector<StageRecord>> recs_;
    std::optional<sk::usm_vector<std::byte>>   in_, out_, body_;
};

}  // namespace naga::cospan