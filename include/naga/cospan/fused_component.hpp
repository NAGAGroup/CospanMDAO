#pragma once

// =============================================================================
// naga/cospan/fused_component.hpp — fused components and the descriptor helpers
// =============================================================================
//
// A FUSED COMPONENT is semantically a component: one kernel, one submission
// (docs/design-review-2026-10-03.md, section 4). Its stages are components whose bodies run as the stages of ONE
// outer kernel, the "wave": a chained executor walks a call sequence of the stages' functions (kb_fn<KB>) with a
// shared record cursor, so every stage function reads its own StageRecord by position.
//   precompute   runs each stage's precompute;
//   compute      collects each stage's complete launch (describe; the stage's own submission is bypassed), checks that
//                the stages' group shapes agree (only knowable here, since a component may choose its shape at run
//                time), packs the records and the three byte streams, submits ONE launch, and gives its event to every
//                stage, so each stage's variables get their event lists exactly as if it had submitted alone;
//   postcompute  runs each stage's postcompute.
// The call sequence (the JIT configuration) is built at submission and cached once per distinct sequence of stage
// functions (call_sequence_config). The packing buffers are rewritten each forward; that rewrite depends on the
// previous wave's event.
//
// STRATEGIES. seq: the stages run one after the other inside each work-group (every stage has group_offset 0).
// THERE ARE NO FRAMEWORK BARRIERS: a stage that reads, across work-items, what an earlier stage wrote calls its own
// group barrier first; stage ranges may differ; across groups nothing is ordered. par: the stages own disjoint slices
// of the work-groups and run unordered, so connects between them are rejected (in the call). Tied outputs follow the
// ordinary tie policy and nothing else (design review, section 9): tie_outputs requires atomic members, or its
// allow_plain flag says the writers are disjoint; the fused component adds no check of its own.
//
// LOCAL MEMORY. seq: every scratch stage gets its OWN slice (StageRecord::local_off = the running sum of the
// earlier stages' scratch_bytes, each rounded up to 16) and the launch's local size is the sum. par: the stages own
// disjoint groups, so each uses offset 0 and the local size is the max. Each forward, a local size above
// info::device::local_mem_size throws, naming the scratch stages and their byte counts.
//
// THE HELPERS (fuse_components, system_from) are for the simple case only: no per-child run policies, no promotions,
// no device splits (write your own System for those). Their result is a descriptor, so it goes straight into a
// parent's add_subsystem([props], [device], ...): the device comes from THAT call, and set_device applies to the whole
// fused component, so one device per fused component holds by construction. Ports of the children are reached by
// path, e.g. "my_fused_comp.a.x". Inheriting from FusedBase is not the user-facing way to fuse (section 4).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <concepts>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/component.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/kernel.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/system.hpp"
#include "naga/cospan/variable.hpp"
#include "naga/cospan/wiring.hpp"

namespace naga::cospan {

enum class fuse_strategy : uint8_t { seq, par };

/// The connections of a helper-built group, as (source path, target path) pairs in the group's namespace:
/// connects_desc({{"a.y", "b.x"}, {"b.z", "c.w"}}). The same connect, with the same rules, scoped to the group.
struct ConnectsDesc {
    std::vector<std::pair<std::string, std::string>> pairs;
};

inline ConnectsDesc connects_desc(std::initializer_list<std::pair<std::string_view, std::string_view>> pairs) {
    ConnectsDesc c;
    for (const auto& [a, b] : pairs) c.pairs.emplace_back(std::string{a}, std::string{b});
    return c;
}

namespace detail {

/// The call-sequence configuration for a wave of rank `rank` over the stage functions `defs`: built ONCE per distinct
/// sequence, kept in a static map for the life of the program. STATIC LIFETIME, because the runtime reads a
/// configuration on a worker thread after the submission (section 4).
///
/// THE KEY. dynamic_function_id's comparison operators could not be confirmed (the AdaptiveCpp headers are not in the
/// dev environment), so the key is NOT the ids: it is (rank, the dynamic types of the stage components). A component
/// type fixes its kernel-body type and so its stage function; two component types sharing one body get two entries,
/// which only costs one more configuration. `defs` is what a miss builds from.
inline sycl::AdaptiveCpp_jit::dynamic_function_config& call_sequence_config(int rank, const std::vector<JitId>& defs,
                                                                            const std::vector<std::type_index>& key) {
    using Config = sycl::AdaptiveCpp_jit::dynamic_function_config;
    static std::mutex                                                                  mu;
    static std::map<std::pair<int, std::vector<std::type_index>>, std::unique_ptr<Config>> cache;
    const std::lock_guard<std::mutex> lock{mu};
    std::unique_ptr<Config>&          slot = cache[{rank, key}];
    if (!slot) {
        slot = std::make_unique<Config>();
        slot->define_as_call_sequence(wave_entry_id(rank), defs);
    }
    return *slot;
}

/// A fused component's machinery: not the user path (section 4), a base for FusedComponent.
class FusedBase : public System {
public:
    explicit FusedBase(fuse_strategy s) : strategy_{s} {}
    ~FusedBase() override = default;

    /// Setup, then the setup-end check: no ragged ports.
    void run_setup() override {
        System::run_setup();
        check_no_ragged_ports(stages());
    }

    void run_precompute(Runtime& rt) override {
        for (ComponentBase* s : stages()) s->run_precompute(rt);
    }

    /// One kernel, one submission: describe every stage, pack, submit once, record the event on every stage.
    void run_compute(Runtime& rt) override {
        const std::vector<ComponentBase*> st = stages();
        if (st.empty()) throw std::invalid_argument("naga::cospan: " + describe() + ": a fused component has no stages");

        // 1. Every stage's complete launch, kept alive until the wave is staged.
        std::vector<std::unique_ptr<LaunchDescription>> ds;
        std::vector<const KernelParts*>                  parts;
        for (std::size_t i = 0; i < st.size(); ++i) {
            std::unique_ptr<LaunchDescription> d = ComponentAccess::describe(*st[i]);
            if (!d)
                throw std::logic_error("naga::cospan: " + describe() + ": forward: stage '" + st[i]->path() +
                                       "' returned no launch description; a fused component's stage must describe its work");
            const KernelParts* p = (d->is_custom_op() || d->is_host()) ? nullptr : d->kernel();
            if (p == nullptr)
                throw std::logic_error("naga::cospan: " + describe() + ": forward: stage '" + st[i]->path() +
                                       "' described no kernel; a fused stage must be a kernel body");
            if (i == 0)
                rank_ = p->rank;
            else if (p->rank != rank_)
                throw std::logic_error("naga::cospan: " + describe() + ": forward: stage '" + st[i]->path() + "' described a rank-" +
                                       std::to_string(p->rank) + " launch; the fused component is rank " + std::to_string(rank_));
            parts.push_back(p);
            ds.push_back(std::move(d));
        }

        // 2. The wave's work-group shape: the stages' common requested one, else the default for the rank.
        const sycl::range<3> w       = wave_wg(st, parts);
        const std::size_t    wg_size = w[0] * w[1] * w[2];

        // 3. (par ties follow the ordinary tie policy, checked in tie_outputs: nothing to check here, section 9)

        // 4. Records, offsets and local memory.
        const sycl::device dev = ComponentAccess::device_of(*st.front());
        std::vector<StageRecord> recs;
        std::vector<std::byte>   in, out, body;
        std::size_t              total_groups = 0, local_bytes = 0;
        for (const KernelParts* p : parts) {
            StageRecord r = p->record;
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
            if (strategy_ == fuse_strategy::par) {
                r.group_offset = total_groups;  // this stage's slice starts after all previous slices
                total_groups += r.group_count;  // the launch is the sum of the slices
            } else {
                r.group_offset = 0;  // seq: every stage starts at group 0
                total_groups   = std::max(total_groups, r.group_count);
            }
            if (strategy_ == fuse_strategy::seq) {
                // Every scratch stage has its own slice: the running sum, each rounded up to 16.
                r.local_off = static_cast<std::uint32_t>(local_bytes);
                local_bytes += (std::size_t{r.scratch_bytes} + 15) / 16 * 16;
            } else {
                // par stages own disjoint groups: they share one slice at offset 0, the launch needs the max.
                r.local_off = 0;
                local_bytes = std::max<std::size_t>(local_bytes, r.scratch_bytes);
            }
            recs.push_back(r);
        }
        const std::size_t local_limit = dev.get_info<sycl::info::device::local_mem_size>();
        if (local_bytes > local_limit) {
            std::string scratch_stages;
            for (std::size_t i = 0; i < recs.size(); ++i)
                if (recs[i].scratch_bytes != 0)
                    scratch_stages += (scratch_stages.empty() ? "'" : ", '") + st[i]->path() + "' (" +
                                      std::to_string(recs[i].scratch_bytes) + " bytes)";
            throw std::invalid_argument("naga::cospan: " + describe() + ": forward: the wave needs " + std::to_string(local_bytes) +
                                        " bytes of local memory (" +
                                        (strategy_ == fuse_strategy::seq ? "the sum of its stages' slices" : "the largest stage's slice, shared") +
                                        ") but the device offers " + std::to_string(local_limit) + "; scratch stages: " + scratch_stages);
        }
        if (total_groups == 0) {  // every stage is empty: nothing launched, and no stale events are kept
            for (ComponentBase* s : st) {
                (void)ComponentAccess::gather_deps(*s, rt);  // depends_on events last one forward, launched or not
                ComponentAccess::record_launch(*s, {});
            }
            last_wave_.clear();
            return;
        }

        // 5. The packing buffers, rewritten after the previous wave (section 4).
        upload(rt, dev, recs_, recs);
        upload(rt, dev, in_, in);
        upload(rt, dev, out_, out);
        upload(rt, dev, body_, body);

        // 6. What the wave must come after: every stage's dependencies, and the packing buffers' own events.
        std::vector<sycl::event> deps;
        for (ComponentBase* s : st) {
            const std::vector<sycl::event> d = ComponentAccess::gather_deps(*s, rt);
            deps.insert(deps.end(), d.begin(), d.end());
        }
        // The wave READS the packing buffers: it waits on their writes (the uploads above).
        for (const std::vector<sycl::event>& e : {recs_->read_deps(), in_->read_deps(), out_->read_deps(), body_->read_deps()})
            deps.insert(deps.end(), e.begin(), e.end());

        // 7. ONE outer launch, with the cached call sequence of the stages' functions in stage order.
        std::vector<JitId>          defs;
        std::vector<std::type_index> key;
        for (const ComponentBase* s : st) {
            defs.push_back(s->definition_id());
            key.emplace_back(typeid(*s));
        }
        sycl::AdaptiveCpp_jit::dynamic_function_config& cfg = call_sequence_config(rank_, defs, key);
        const StageRecord* rp = recs_->data();
        const std::byte*   ip = in_->data();
        std::byte*         op = out_->data();
        const std::byte*   bp = body_->data();
        sycl::event        ev;
        switch (rank_) {
            case 1: ev = launch_wave<1>(rt.uq, dev, deps, cfg, wave_nd_range<1>(total_groups, w), rp, ip, op, bp, local_bytes); break;
            case 2: ev = launch_wave<2>(rt.uq, dev, deps, cfg, wave_nd_range<2>(total_groups, w), rp, ip, op, bp, local_bytes); break;
            default: ev = launch_wave<3>(rt.uq, dev, deps, cfg, wave_nd_range<3>(total_groups, w), rp, ip, op, bp, local_bytes); break;
        }

        // 8. The wave's event: a read of the packing buffers (the next rewrite waits), and on every stage, so each
        // stage's containers get their reads (here) and writes (in the stage's postcompute) as if it had submitted alone.
        recs_->register_read(ev);
        in_->register_read(ev);
        out_->register_read(ev);
        body_->register_read(ev);
        last_wave_ = {ev};
        for (ComponentBase* s : st) ComponentAccess::record_launch(*s, last_wave_);
    }

    void run_postcompute(Runtime& rt) override {
        for (ComponentBase* s : stages()) s->run_postcompute(rt);
    }

    std::vector<sycl::event> events() const override { return last_wave_; }

    /// depends_on: every stage's submission (all of it is one wave) adds them for one forward.
    void add_dependencies(std::span<const sycl::event> deps) override {
        for (ComponentBase* s : stages()) s->add_dependencies(deps);
    }

private:
    /// The live children, as components: a fused component's children are components only.
    std::vector<ComponentBase*> stages() const {
        std::vector<ComponentBase*> out;
        for (Subsystem* c : child_ptrs_) {
            if (c == nullptr) continue;
            auto* s = dynamic_cast<ComponentBase*>(c);
            if (s == nullptr)
                throw std::logic_error("naga::cospan: " + describe() + ": child '" + c->path() + "' is not a Component");
            out.push_back(s);
        }
        return out;
    }

    /// A child with any ragged port (RaggedInput / RaggedOutput, or a per-index dimension) is a setup error.
    void check_no_ragged_ports(const std::vector<ComponentBase*>& st) const {
        for (const ComponentBase* s : st)
            if (const std::shared_ptr<Instance> e = s->first_ragged_port())
                throw std::invalid_argument("naga::cospan: " + describe() + ": '" + e->name + "' of '" + s->path() +
                                            "' is a ragged port: ragged ports are not supported in fused components yet");
    }

    static std::uint32_t append(std::vector<std::byte>& v, std::span<const std::byte> s) {
        const std::size_t off = (v.size() + 15) / 16 * 16;  // 16-byte aligned; padding is zero
        v.resize(off + s.size());
        if (!s.empty()) std::memcpy(v.data() + off, s.data(), s.size());
        return static_cast<std::uint32_t>(off);
    }

    /// The wave's work-group shape: every stage that asks for one must agree (group shape belongs to the component);
    /// none asking, the default for the rank.
    sycl::range<3> wave_wg(const std::vector<ComponentBase*>& st, const std::vector<const KernelParts*>& parts) const {
        std::optional<sycl::range<3>> wg;
        std::size_t                   first = 0;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            if (!parts[i]->wg) continue;
            if (!wg) {
                wg    = parts[i]->wg;
                first = i;
            } else if (*wg != *parts[i]->wg) {
                throw std::invalid_argument("naga::cospan: " + describe() + ": forward: '" + st[first]->path() + "' and '" +
                                            st[i]->path() +
                                            "' ask for different work-group shapes; the stages of a fused component share one "
                                            "(section 4: group shape belongs to the component)");
            }
        }
        if (!wg) {
            switch (rank_) {
                case 1: wg = default_wg<1>(); break;
                case 2: wg = default_wg<2>(); break;
                default: wg = default_wg<3>(); break;
            }
        }
        sycl::range<3> w = *wg;
        for (int d = rank_; d < 3; ++d) w[d] = 1;  // dimensions past the rank do not exist
        return w;
    }

    /// Rewrites a packing buffer, after the previous wave (which read it): the copy waits on `last_wave_`.
    template <class T>
    void upload(Runtime& rt, const sycl::device& dev, std::optional<sk::usm_vector<T>>& v, const std::vector<T>& src) {
        if (!v) v.emplace(rt.allocator_for<T>(dev, sk::kind::device));
        diag::count_host_tasks();
        v->from_host(std::span<const T>{src}, std::span<const sycl::event>{last_wave_});
    }

    fuse_strategy strategy_;
    int           rank_ = 0;

    std::vector<sycl::event> last_wave_;  ///< the previous wave's event: the packing buffers' rewrite depends on it

    // Packing buffers, kept while the fused component exists; each forward rewrites them.
    std::optional<sk::usm_vector<StageRecord>> recs_;
    std::optional<sk::usm_vector<std::byte>>   in_, out_, body_;
};

/// The component type of a SubsystemDesc.
template <class D>
struct desc_component;
template <class T, class... A>
struct desc_component<SubsystemDesc<T, A...>> {
    using type = T;
};
template <class D>
using desc_component_t = typename desc_component<std::remove_cvref_t<D>>::type;
template <class D>
using desc_body_t = typename desc_component_t<D>::body_type;

/// Whether the first of `Ds` is a ConnectsDesc (false for none).
template <class... Ds>
inline constexpr bool leads_with_connects_v = false;
template <class D, class... Ds>
inline constexpr bool leads_with_connects_v<D, Ds...> = std::is_same_v<std::remove_cvref_t<D>, ConnectsDesc>;

/// A plain System of the given children in order, with the given connections.
template <class... Descs>
class ComposedSystem : public System {
public:
    ComposedSystem(ConnectsDesc c, Descs... d) : connects_{std::move(c)}, descs_{std::move(d)...} {}

    void setup() override {
        std::apply([this](auto&... d) { (this->add_subsystem(d), ...); }, descs_);
        for (const auto& [a, b] : connects_.pairs) connect(a, b);
    }

private:
    ConnectsDesc        connects_;
    std::tuple<Descs...> descs_;
};

}  // namespace detail

/// One kernel, one submission: the stages `Descs` (descriptors of components), added in order, run as one wave.
/// Connections (seq only) are made after the stages are added and their variables are marked `internal`. Tied outputs of
/// its stages follow the ordinary tie policy (section 9).
template <fuse_strategy S, class... Descs>
class FusedComponent final : public detail::FusedBase {
    static_assert(sizeof...(Descs) >= 1, "naga::cospan: a fused component needs at least one stage");
    static_assert((std::derived_from<detail::desc_component_t<Descs>, ComponentBase> && ...),
                  "naga::cospan: the stages of a fused component are components (subsystem_desc<T> with T a Component)");
    static_assert(((!CustomOpBody<detail::desc_body_t<Descs>> && !HostBody<detail::desc_body_t<Descs>>) && ...),
                  "naga::cospan: custom-op and host bodies cannot be fused");
    static_assert((GroupBody<detail::desc_body_t<Descs>> && ...) || (!GroupBody<detail::desc_body_t<Descs>> && ...),
                  "naga::cospan: a fused component's stages are all work-group components or all plain");
    static_assert(((kernel_rank_v<detail::desc_body_t<Descs>> == kernel_rank_v<detail::desc_body_t<std::tuple_element_t<0, std::tuple<Descs...>>>>) && ...),
                  "naga::cospan: the stages of a fused component share one kernel rank");

public:
    FusedComponent(ConnectsDesc c, Descs... d) : detail::FusedBase{S}, connects_{std::move(c)}, descs_{std::move(d)...} {}

    void setup() override {
        std::apply([this](auto&... d) { (this->add_subsystem(d), ...); }, descs_);
        for (const auto& [a, b] : connects_.pairs) {
            if constexpr (S == fuse_strategy::par) {
                detail::wiring_fail(detail::WiringSite{describe()}, "connect",
                                    "a par fused component's stages run unordered: connect is rejected ('" + a + "' -> '" + b + "')");
            } else {
                connect(a, b);
                instance_at(b)->meta->internal = true;  // connected inside one fused component
            }
        }
    }

private:
    ConnectsDesc        connects_;
    std::tuple<Descs...> descs_;
};

// ---- helpers: the simple case only -------------------------------------------------

/// A fused component of the given stages (in order) with the given connections, as a descriptor for add_subsystem.
template <fuse_strategy S, class... Descs>
auto fuse_components(std::string name, ConnectsDesc c, Descs... d) {
    return subsystem_desc<FusedComponent<S, Descs...>>(std::move(name), std::move(c), std::move(d)...);
}
/// The same with no connections.
template <fuse_strategy S, class... Descs>
    requires(sizeof...(Descs) >= 1 && !detail::leads_with_connects_v<Descs...>)
auto fuse_components(std::string name, Descs... d) {
    return subsystem_desc<FusedComponent<S, Descs...>>(std::move(name), ConnectsDesc{}, std::move(d)...);
}

/// A plain System of the given children (in order) with the given connections, as a descriptor for add_subsystem.
template <class... Descs>
auto system_from(std::string name, ConnectsDesc c, Descs... d) {
    return subsystem_desc<detail::ComposedSystem<Descs...>>(std::move(name), std::move(c), std::move(d)...);
}
template <class... Descs>
    requires(sizeof...(Descs) >= 1 && !detail::leads_with_connects_v<Descs...>)
auto system_from(std::string name, Descs... d) {
    return subsystem_desc<detail::ComposedSystem<Descs...>>(std::move(name), ConnectsDesc{}, std::move(d)...);
}

}  // namespace naga::cospan
