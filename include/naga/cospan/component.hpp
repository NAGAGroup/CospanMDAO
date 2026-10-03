#pragma once

// =============================================================================
// naga/cospan/component.hpp — ComponentBase and Component<KB>
// =============================================================================
//
// A component is the leaf that does work, and it SUBMITS ITSELF (docs/design-review-2026-10-03.md, sections 3 and
// 4). Its forward is three framework-owned steps that call user hooks:
//
//   precompute   (1) the expected extents of each input's own dimensions are worked out (a derived dimension
//                    reads its sibling input; a value-driven one asks the driving variable for its value);
//                (2) every distinct input variable is materialized;
//                (3) each output's shape is built from its dimensions and given to its variable (a tied output's
//                    shape belongs to its tie's System);
//                (4) the user's pre_forward() runs;
//                (5) every distinct output variable is materialized;
//                (6) the reactive checks: what each dimension declared against what its variable has;
//                (7) every member wrapper is pointed at its data.
//   compute      the user's compute_forward() returns the launch (describe); the component submits it on the
//                unordered queue, depending on its variables' events and nothing else (submit), and records
//                its own event on them.
//   postcompute  the pushes to the variable's other placements, then the user's post_forward().
//
// A component OWNS its instances by shared_ptr (section 1) and holds no pointer to its owner's data. It holds no
// queue and no device of its own: the device of each instance is a field of the instance (section 2), and a
// Runtime is given for the duration of a forward step.
//
// Component<KB> adds the typed layer. ONE KERNEL BODY PER COMPONENT: the body's signature fixes the kind (plain:
// sycl::id<R>; group: GroupItem<R>; custom-op: CustomOp&; host: Host) and the kernel rank at COMPILE TIME, so there
// is no run-time branching on rank. Custom-op and host bodies cannot be fused (section 4): the fuse helper enforces
// it on the traits below.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/diagnostics.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/init.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/kernel.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/variable.hpp"
#include "naga/cospan/wiring.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

/// What a Component's compute turn yields (describe): the complete launch, configuration plus the user's setup. The
/// component submits it itself; a fused owner (phase 7) collects the descriptions of its stages instead.
class LaunchDescription {
public:
    virtual ~LaunchDescription() = default;
    virtual bool is_custom_op() const = 0;
    virtual bool is_host() const      = 0;
    /// Submits on `q` for `dev`, ordered ONLY by `deps` (section 3). Returns the events of what it submitted: one for a
    /// kernel, custom-op or host launch, none when nothing was submitted (an empty range).
    virtual std::vector<sycl::event> submit(sk::unordered_queue& q, const sycl::device& dev,
                                            const std::vector<sycl::event>& deps) const = 0;
    /// Null for custom-op and host descriptions.
    virtual const KernelParts* kernel() const { return nullptr; }
};

namespace detail {

/// RAII flag: set for a scope, cleared on every exit.
struct FlagScope {
    explicit FlagScope(bool& f) : f_{f} { f_ = true; }
    ~FlagScope() { f_ = false; }
    FlagScope(const FlagScope&)            = delete;
    FlagScope& operator=(const FlagScope&) = delete;
    bool& f_;
};

struct ComponentAccess;

}  // namespace detail

class ComponentBase : public Subsystem {
public:
    ComponentBase(const ComponentBase&)            = delete;
    ComponentBase& operator=(const ComponentBase&) = delete;
    ~ComponentBase() override = default;

    // ---- HOOKS (user-implemented) ------------------------------------------
    /// Declare I/O. Runs once, during setup. Zero allocation.
    virtual void setup() = 0;
    /// There is no init hook: a free input's initial value is declared in setup() with init_input, and outputs are
    /// never initialized.
    /// Per-forward manual shape control, after automatic derivation (reshape is legal here only).
    virtual void pre_forward() {}
    /// At the end of postcompute, after the pushes to the variables' other placements.
    virtual void post_forward() {}

    // ---- declaration API (legal inside setup() only) ------------------------
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_input(Input<DT, Rank>& m, std::string name, Specs... specs) {
        add_input<DT, Rank>(m, std::move(name),
                            std::array<detail::DimRecord, Rank>{detail::make_dim_record(specs)...});
    }
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_output(Output<DT, Rank>& m, std::string name, Specs... specs) {
        add_output<DT, Rank>(m, std::move(name),
                             std::array<detail::DimRecord, Rank>{detail::make_dim_record(specs)...});
    }

    /// The array forms: unspecified_shape<R>() and from_input_shape<R>(name).
    template <DType DT, int Rank, DimSpec S>
    void declare_input(Input<DT, Rank>& m, std::string name,
                       const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        add_input<DT, Rank>(m, std::move(name), records_of<Rank>(specs));
    }
    template <DType DT, int Rank, DimSpec S>
    void declare_output(Output<DT, Rank>& m, std::string name,
                        const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        add_output<DT, Rank>(m, std::move(name), records_of<Rank>(specs));
    }

    /// The whole-shape value-driven marker.
    template <DType DT, int Rank>
    void declare_input(Input<DT, Rank>& m, std::string name, FromInputArray marker) {
        add_input<DT, Rank>(m, std::move(name), detail::expand_from_input_array<Rank>(marker));
    }
    template <DType DT, int Rank>
    void declare_output(Output<DT, Rank>& m, std::string name, FromInputArray marker) {
        add_output<DT, Rank>(m, std::move(name), detail::expand_from_input_array<Rank>(marker));
    }

    /// Ragged ports take the array form too: cs::unspecified_shape<R>() (every dimension supplied from outside).
    template <DType DT, int Rank, DimSpec S>
    void declare_input(RaggedInput<DT, Rank>& m, std::string name, const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        add_ragged_input<DT, Rank>(m, std::move(name), records_of<Rank>(specs));
    }
    template <DType DT, int Rank, DimSpec S>
    void declare_output(RaggedOutput<DT, Rank>& m, std::string name, const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        add_ragged_output<DT, Rank>(m, std::move(name), records_of<Rank>(specs));
    }

    /// Ragged ports: RaggedInput / RaggedOutput members, Rank 2..3, with the same DimSpec
    /// arguments (scalar and per-index mixed: PerIndex, FromInputValues on dims >= 1).
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_input(RaggedInput<DT, Rank>& m, std::string name, Specs... specs) {
        add_ragged_input<DT, Rank>(m, std::move(name), std::array<detail::DimRecord, Rank>{detail::make_dim_record(specs)...});
    }
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_output(RaggedOutput<DT, Rank>& m, std::string name, Specs... specs) {
        add_ragged_output<DT, Rank>(m, std::move(name), std::array<detail::DimRecord, Rank>{detail::make_dim_record(specs)...});
    }

    /// The pre_forward-only resize of an OUTPUT (the same setter as the automatic shape): it gives the output's
    /// variable this shape. An input's shape is its variable's: it cannot be reshaped here.
    template <DType DT, int Rank>
    void reshape(Output<DT, Rank>& m, sycl::range<Rank> shape) {
        if (!in_pre_forward_)
            throw std::logic_error("naga::cospan: reshape is legal inside pre_forward() only");
        const auto it = by_member_.find(&m);
        if (it == by_member_.end())
            throw std::invalid_argument("naga::cospan: " + describe() + ": reshape of a member that was never declared");
        if (rt_ == nullptr) throw std::logic_error("naga::cospan: " + describe() + ": reshape outside a forward");
        detail::ResolvedShape s;
        s.known = true;
        for (int d = 0; d < Rank; ++d) s.extents[static_cast<std::size_t>(d)] = shape[d];
        detail::VarMeta* meta = it->second->meta.get();
        meta->set_shape(*rt_, s);
        reshaped_.push_back(meta);
    }

    // ---- initial values (legal inside setup() only) -----------------------
    /// Declare the initial value of one of this Component's OWN inputs, by name. Metadata on the input's variable
    /// (section 1); the declaration of the highest level wins (decided in this call), it is ignored when the input
    /// ends up with a source, and it is an error to name an output.
    void init_input(std::string_view name, InitSpec init) {
        if (!declaring_)
            throw std::logic_error("naga::cospan: init_input is legal inside setup() only (" + describe() + ")");
        const std::shared_ptr<detail::Instance> inst = find_kind(name, IOKind::input);  // an output throws
        detail::declare_init(detail::WiringSite{describe()}, inst, std::move(init), detail::depth_of(*this), path());
    }

    // ---- Subsystem verbs ---------------------------------------------------
    void run_setup() override {
        detail::FlagScope scope{declaring_};
        setup();
    }

    /// Section 2: each of this component's instances goes to `d`. The outer call wins.
    void set_device(const sycl::device& d) override {
        for (const auto& inst : order_) inst->device = d;
    }

    std::shared_ptr<detail::Instance> find_instance(std::string_view name) const override {
        const auto it = by_name_.find(std::string{name});
        return it == by_name_.end() ? nullptr : it->second;
    }
    void collect_instances(std::vector<std::shared_ptr<detail::Instance>>& out) const override {
        for (const auto& i : order_) out.push_back(i);
    }

    /// The last launch's event(s): what deleting this component waits on, and what depends_on hands on.
    std::vector<sycl::event> events() const override { return last_events_; }

    /// depends_on: these events are added to this component's submission for ONE forward.
    void add_dependencies(std::span<const sycl::event> deps) override {
        pending_deps_.insert(pending_deps_.end(), deps.begin(), deps.end());
    }

    // ---- the forward (section 3) ---------------------------------------------
    void run_precompute(detail::Runtime& rt) override {
        PtrScope rt_scope{rt_, &rt};
        reshaped_.clear();

        // The tie alias rules of the variables this component touches, once each (a no-op without a rule, and once per
        // forward per variable): a tied variable's shape is reactive.
        {
            std::vector<detail::VarMeta*> refreshed;
            for (const auto& inst : order_) {
                detail::VarMeta* m = inst->meta.get();
                if (std::find(refreshed.begin(), refreshed.end(), m) != refreshed.end()) continue;
                refreshed.push_back(m);
                m->refresh_alias_shape(rt);
            }
        }

        // (1) What each input's own dimensions say, as numbers: constraints, checked in (6).
        std::vector<std::pair<detail::Instance*, Expected>> expects;
        for (const auto& inst : order_)
            if (inst->kind == IOKind::input) expects.emplace_back(inst.get(), expected_of(*inst, rt));

        // (2) Every distinct input variable is materialized (a free input takes its initializer here).
        for (detail::VarMeta* m : metas_of(IOKind::input)) m->materialize(rt);

        // (3) Each output's shape, from its dimensions, unless it belongs to a tie (that System resolves it).
        for (const auto& inst : order_) {
            if (inst->kind != IOKind::output) continue;
            Expected ex = expected_of(*inst, rt);
            if (!inst->meta->alias_shape) {
                detail::ResolvedShape s;
                s.known = true;
                for (int d = 0; d < inst->meta->rank; ++d) {
                    const auto u = static_cast<std::size_t>(d);
                    if (!ex.has[u])
                        throw std::invalid_argument("naga::cospan: output '" + inst->full_path() +
                                                    "' has an unspecified dimension but is not a member of a tie_outputs: "
                                                    "nothing supplies its shape (declare the dimension, or tie it)");
                    s.extents[u] = ex.extent[u];
                    if (d >= 1 && ex.per[u]) {
                        s.per[u]       = true;
                        s.per_index[u] = ex.per_index[u];
                    }
                }
                check_dense(*inst, s);
                inst->meta->set_shape(rt, s);
            }
            expects.emplace_back(inst.get(), std::move(ex));
        }

        // (4) Manual shape control.
        {
            detail::FlagScope scope{in_pre_forward_};
            pre_forward();
        }

        // (5) Every distinct output variable is materialized.
        for (detail::VarMeta* m : metas_of(IOKind::output)) m->materialize(rt);

        // (6) Reactive checks: what each dimension declared against what its variable has. An output that was reshaped
        // in pre_forward is the user's decision and is not checked against its declaration.
        for (const auto& [inst, ex] : expects) {
            if (std::find(reshaped_.begin(), reshaped_.end(), inst->meta.get()) != reshaped_.end()) continue;
            check_expected(*inst, ex);
        }
        for (const auto& inst : order_) check_dense(*inst, inst->meta->shape);

        // (7) The member wrappers.
        for (const auto& inst : order_) inst->point_wrapper();
    }

    /// Submits the launch and records its event. Writers in one forward never wait on each other (write_deps returns the
    /// PREVIOUS generation), so tied outputs run concurrently; there is no queue-wide wait anywhere (section 3). The
    /// submission depends on its inputs' write events, its outputs' read events, the events of the data it touches,
    /// and any depends_on events.
    void run_compute(detail::Runtime& rt) override {
        const std::unique_ptr<LaunchDescription> desc = compute_forward_impl();
        if (!desc) throw std::logic_error("naga::cospan: " + describe() + ": compute_forward returned no launch");
        // the component's device: its first instance's (set_device moves them all together), else the default
        const sycl::device device = order_.empty() ? detail::default_device() : order_.front()->device;

        const std::vector<detail::VarMeta*> ins  = metas_of(IOKind::input);
        const std::vector<detail::VarMeta*> outs = metas_of(IOKind::output);
        std::vector<sycl::event>            deps;
        const auto                          add = [&](const std::vector<sycl::event>& v) { deps.insert(deps.end(), v.begin(), v.end()); };
        for (detail::VarMeta* m : ins) add(m->events.read_deps());
        for (detail::VarMeta* m : outs) add(m->events.write_deps(rt.forward));
        std::vector<detail::DeviceDataBase*> datas;
        for (const auto& inst : order_)
            if (inst->data && std::find(datas.begin(), datas.end(), inst->data.get()) == datas.end()) datas.push_back(inst->data.get());
        for (detail::DeviceDataBase* d : datas) add(d->events());
        add(pending_deps_);

        const std::vector<sycl::event> ev = desc->submit(rt.uq, device, deps);
        for (const sycl::event& e : ev) {
            for (detail::VarMeta* m : outs) m->events.record_write(e);
            for (detail::VarMeta* m : ins) m->events.record_read(e);
            for (detail::DeviceDataBase* d : datas) d->register_event(e);
        }
        last_events_ = ev;
        pending_deps_.clear();
    }

    /// Pushes each output variable this component wrote to the variable's OTHER placements, ordered after the launch
    /// and the data's own events. A tied output is pushed by the tie's System (phase 6). Then the user's hook.
    void run_postcompute(detail::Runtime& rt) override {
        (void)rt;
        std::vector<detail::VarMeta*> seen;
        for (const auto& inst : order_) {
            if (inst->kind != IOKind::output) continue;
            detail::VarMeta* m = inst->meta.get();
            if (std::find(seen.begin(), seen.end(), m) != seen.end()) continue;
            seen.push_back(m);
            if (m->alias_shape || !inst->data) continue;
            const std::shared_ptr<detail::DeviceDataBase> own = inst->data;
            for (const auto& entry : m->data) {
                const std::shared_ptr<detail::DeviceDataBase> dst = entry.second.lock();
                if (!dst || dst == own || entry.first == own->key) continue;
                std::vector<sycl::event> deps = last_events_;
                const std::vector<sycl::event> own_ev = own->events();
                deps.insert(deps.end(), own_ev.begin(), own_ev.end());
                const sycl::event e = dst->copy_from(*own, deps);
                own->register_event(e);  // a later writer of this copy waits for the read
            }
        }
        // Phase 8: producer-side shape work for shape-driving variables goes here.
        post_forward();
    }

    // ---- what a fused owner needs to know, without knowing the KernelBody type --
    // Answered by Component<KB> from its body's signature. An owner reads
    // these; nothing here knows what fusing is.
    virtual int  kernel_rank() const { return 0; }       ///< 0 for custom-op and host bodies
    virtual bool is_group_body() const { return false; }
    virtual bool has_scratch() const { return false; }
    virtual bool is_custom_op() const { return false; }
    virtual bool is_host() const { return false; }
    /// The type-erased id of this body's stage function (kb_fn<KB>), for a
    /// call sequence. Throws for a custom-op or host body (nothing is compiled).
    virtual detail::JitId definition_id() const {
        throw std::logic_error("naga::cospan: " + describe() + " has no stage function");
    }

protected:
    ComponentBase() = default;

    /// The storage kind of this Component's I/O: device, or host for a host body.
    virtual sk::kind default_storage_kind() const { return sk::kind::device; }

    /// The launch of this forward (describe): implemented by Component<KB> from the user's compute_forward().
    virtual std::unique_ptr<LaunchDescription> compute_forward_impl() = 0;

    std::string describe() const { return path().empty() ? std::string{"component"} : "'" + path() + "'"; }

    std::vector<sycl::event> last_events_;  ///< the last launch's events
    std::vector<sycl::event> pending_deps_; ///< depends_on events, for the next submission only

private:
    friend struct detail::ComponentAccess;

    /// The expected extents of one instance's own dimensions, as numbers.
    struct Expected {
        std::array<bool, 3>                     has{false, false, false};
        std::array<std::size_t, 3>              extent{1, 1, 1};
        std::array<bool, 3>                     per{false, false, false};
        std::array<std::vector<std::size_t>, 3> per_index{};
    };

    /// Sets a pointer for a scope, clears it on every exit.
    struct PtrScope {
        PtrScope(detail::Runtime*& p, detail::Runtime* v) : p_{p} { p_ = v; }
        ~PtrScope() { p_ = nullptr; }
        PtrScope(const PtrScope&)            = delete;
        PtrScope& operator=(const PtrScope&) = delete;
        detail::Runtime*& p_;
    };

    // ---- declaration -------------------------------------------------------
    template <int Rank, class S>
    static std::array<detail::DimRecord, Rank> records_of(
        const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        std::array<detail::DimRecord, Rank> out{};
        for (int d = 0; d < Rank; ++d) out[d] = detail::make_dim_record(specs[d]);
        return out;
    }

    template <DType DT, int Rank>
    void add_input(Input<DT, Rank>& m, std::string name,
                   const std::array<detail::DimRecord, Rank>& dims) {
        reject_per_index(name, dims);
        register_instance(detail::make_input<DT, Rank>(std::move(name), this, dims, &m), &m, "declare_input", false);
    }
    template <DType DT, int Rank>
    void add_output(Output<DT, Rank>& m, std::string name,
                    const std::array<detail::DimRecord, Rank>& dims) {
        reject_per_index(name, dims);
        register_instance(detail::make_output<DT, Rank>(std::move(name), this, dims, &m), &m, "declare_output", false);
    }
    template <DType DT, int Rank>
    void add_ragged_input(RaggedInput<DT, Rank>& m, std::string name, const std::array<detail::DimRecord, Rank>& dims) {
        register_instance(detail::make_ragged_input<DT, Rank>(std::move(name), this, dims, &m), &m, "declare_input", true);
    }
    template <DType DT, int Rank>
    void add_ragged_output(RaggedOutput<DT, Rank>& m, std::string name, const std::array<detail::DimRecord, Rank>& dims) {
        register_instance(detail::make_ragged_output<DT, Rank>(std::move(name), this, dims, &m), &m, "declare_output", true);
    }

    /// a DENSE wrapper cannot hold per-index dims (PerIndex, FromInputValues).
    template <std::size_t N>
    void reject_per_index(const std::string& name, const std::array<detail::DimRecord, N>& dims) const {
        for (const detail::DimRecord& r : dims)
            if (r.how == detail::DimRecord::How::per_index_host || r.how == detail::DimRecord::How::per_index_from_input)
                throw std::invalid_argument("naga::cospan: " + describe() + ": '" + name +
                                            "': per-index dims need RaggedInput/RaggedOutput");
    }

    /// The component owns its instances (section 1); their device is the default until set_device says otherwise.
    void register_instance(std::shared_ptr<detail::Instance> inst, const void* member, const char* what, bool ragged) {
        if (!declaring_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside setup() only");
        const std::string key = inst->name;
        if (by_name_.contains(key))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + key + "' is declared twice");
        if (by_member_.contains(member))
            throw std::invalid_argument("naga::cospan: " + describe() + ": member wrapper of '" + key +
                                        "' is already declared");
        inst->storage_kind = default_storage_kind();
        if (ragged) ragged_ports_.insert(inst.get());
        order_.push_back(inst);
        by_member_.emplace(member, inst.get());
        by_name_.emplace(key, std::move(inst));
    }

    // ---- lookup -----------------------------------------------------------
    std::shared_ptr<detail::Instance> find_kind(std::string_view name, IOKind k) const {
        const auto it = by_name_.find(std::string{name});
        if (it == by_name_.end())
            throw std::out_of_range("naga::cospan: " + describe() + " declares no '" + std::string{name} + "'");
        if (it->second->kind != k)
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{name} +
                                        "' is an " + (k == IOKind::input ? "output" : "input"));
        return it->second;
    }

    /// The distinct variables of this component's instances of kind `k`, in declaration order.
    std::vector<detail::VarMeta*> metas_of(IOKind k) const {
        std::vector<detail::VarMeta*> out;
        for (const auto& inst : order_) {
            if (inst->kind != k) continue;
            detail::VarMeta* m = inst->meta.get();
            if (std::find(out.begin(), out.end(), m) == out.end()) out.push_back(m);
        }
        return out;
    }

    // ---- shape derivation ----------------------------------------------------
    detail::Instance& sibling_input(const detail::Instance& of, const std::string& name) const {
        const auto it = by_name_.find(name);
        if (it == by_name_.end() || it->second->kind != IOKind::input)
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + of.name + "' derives its shape from '" + name +
                                        "', which is not an input of this component");
        return *it->second;
    }

    /// A sibling's shape for dimension `d` of `of`. A free input that is not materialized yet is materialized first
    /// (that is where its initializer's shape is applied); anything else whose shape is unknown is an error.
    const detail::ResolvedShape& source_shape(const detail::Instance& of, const detail::Instance& src, int d,
                                              detail::Runtime& rt) const {
        detail::VarMeta& m = *src.meta;
        if (!m.shape.known && !detail::first_of_kind(m, IOKind::output)) m.materialize(rt);
        if (!m.shape.known)
            throw std::logic_error("naga::cospan: '" + of.full_path() + "': dimension " + std::to_string(d) + " comes from '" +
                                   src.full_path() + "', whose shape is not known yet");
        return m.shape;
    }

    /// The numbers one instance's own dimensions stand for NOW. Derivation sources are inputs only.
    Expected expected_of(const detail::Instance& inst, detail::Runtime& rt) const {
        Expected ex;
        for (int d = 0; d < inst.meta->rank; ++d) {
            const auto               u = static_cast<std::size_t>(d);
            const detail::DimRecord& r = inst.dims[u];
            switch (r.how) {
                case detail::DimRecord::How::fixed:
                    ex.has[u]    = true;
                    ex.extent[u] = r.extent;
                    break;
                case detail::DimRecord::How::unspecified: break;
                case detail::DimRecord::How::from_input_dim: {
                    const detail::Instance& src = sibling_input(inst, r.input);
                    if (r.dim >= src.meta->rank)
                        throw std::out_of_range("naga::cospan: '" + inst.name + "' derives from a dimension past the rank of '" +
                                                r.input + "'");
                    const detail::ResolvedShape& sh = source_shape(inst, src, d, rt);
                    ex.has[u]    = true;
                    ex.extent[u] = sh.extents[r.dim];
                    // a per-index source dimension passes its array on (dimension 0 is never per-index)
                    if (d >= 1 && sh.per[r.dim]) {
                        ex.per[u]       = true;
                        ex.per_index[u] = sh.per_index[r.dim];
                        ex.extent[u]    = 0;
                    }
                    break;
                }
                case detail::DimRecord::How::from_input: {
                    const detail::Instance& src = sibling_input(inst, r.input);
                    ex.has[u]    = true;
                    ex.extent[u] = src.meta->driving_value(rt, r.value_index);
                    break;
                }
                case detail::DimRecord::How::per_index_from_input: {
                    const detail::Instance& src = sibling_input(inst, r.input);
                    ex.has[u]       = true;
                    ex.per[u]       = true;
                    ex.per_index[u] = src.meta->driving_values(rt, r.value_index);
                    ex.extent[u]    = 0;
                    break;
                }
                case detail::DimRecord::How::per_index_host:
                    ex.has[u]       = true;
                    ex.per[u]       = true;
                    ex.per_index[u] = r.per_index;
                    ex.extent[u]    = 0;
                    break;
            }
        }
        return ex;
    }

    static std::string rule_text(const detail::DimRecord& r) {
        switch (r.how) {
            case detail::DimRecord::How::fixed: return "fixed";
            case detail::DimRecord::How::unspecified: return "unspecified";
            case detail::DimRecord::How::from_input_dim: return "FromInputDim{'" + r.input + "', " + std::to_string(r.dim) + "}";
            case detail::DimRecord::How::from_input: return "FromInput{'" + r.input + "', " + std::to_string(r.value_index) + "}";
            case detail::DimRecord::How::per_index_host: return "PerIndex";
            case detail::DimRecord::How::per_index_from_input:
                return "FromInputValues{'" + r.input + "', " + std::to_string(r.value_index) + "}";
        }
        return "?";
    }

    /// Every dimension an instance declared (or derived) against what its variable has.
    void check_expected(const detail::Instance& inst, const Expected& ex) const {
        const detail::ResolvedShape& s = inst.meta->shape;
        for (int d = 0; d < inst.meta->rank; ++d) {
            const auto u = static_cast<std::size_t>(d);
            if (!ex.has[u]) continue;
            const bool        ep   = d >= 1 && ex.per[u];
            const bool        vp   = d >= 1 && s.per[u];
            const std::string head = "naga::cospan: shape mismatch: dimension " + std::to_string(d) + " of '" + inst.full_path() +
                                     "' is " + rule_text(inst.dims[u]) + " = ";
            if (ep != vp)
                throw std::invalid_argument(head + (ep ? "per-index" : std::to_string(ex.extent[u])) + " but the variable's dimension " +
                                            std::to_string(d) + " is " + (vp ? "per-index" : std::to_string(s.extents[u])));
            if (ep) {
                if (ex.per_index[u] != s.per_index[u])
                    throw std::invalid_argument(head + detail::extents_str(ex.per_index[u]) + " but the variable's dimension " +
                                                std::to_string(d) + " is " + detail::extents_str(s.per_index[u]));
            } else if (ex.extent[u] != s.extents[u]) {
                throw std::invalid_argument(head + std::to_string(ex.extent[u]) + " but the variable's dimension " + std::to_string(d) +
                                            " is " + std::to_string(s.extents[u]));
            }
        }
    }

    /// A dense port cannot see per-index extents: a variable with a per-index dimension needs RaggedInput/RaggedOutput.
    void check_dense(const detail::Instance& inst, const detail::ResolvedShape& s) const {
        if (s.known && s.ragged() && !ragged_ports_.contains(&inst))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + inst.name +
                                        "' is ragged; declare it with RaggedInput/RaggedOutput");
    }

    std::unordered_map<std::string, std::shared_ptr<detail::Instance>> by_name_;
    std::unordered_map<const void*, detail::Instance*>                 by_member_;
    std::vector<std::shared_ptr<detail::Instance>>                     order_;  // declaration order
    std::unordered_set<const detail::Instance*>                        ragged_ports_;
    std::vector<detail::VarMeta*>                                      reshaped_;  ///< reshaped in this forward's pre_forward

    bool                declaring_      = false;
    bool                in_pre_forward_ = false;
    detail::Runtime*    rt_             = nullptr;  ///< set during run_precompute only
};

namespace detail {

/// INTERNAL: what a fused owner (phase 7) calls to get a stage's complete launch without submitting it.
struct ComponentAccess {
    static std::unique_ptr<LaunchDescription> describe(ComponentBase& c) { return c.compute_forward_impl(); }
};

}  // namespace detail

// -----------------------------------------------------------------------------
// detail::KernelLaunch<KB> — the individual description of a kernel body
// -----------------------------------------------------------------------------

namespace detail {

struct NoScratch {};

template <class KB>
struct scratch_of {
    using type = NoScratch;
};
template <class KB>
    requires HasScratch<KB>
struct scratch_of<KB> {
    using type = typename KB::Scratch;
};
template <class KB>
using scratch_t = typename scratch_of<KB>::type;

/// The Scratch argument of a body that has none.
template <class KB>
scratch_t<KB> NoScratchValue() {
    return scratch_t<KB>{};
}

template <int R>
sycl::range<3> pad3(const sycl::range<R>& r) {
    sycl::range<3> out{1, 1, 1};
    for (int d = 0; d < R; ++d) out[d] = r[d];
    return out;
}

/// A kernel-body launch submitted on its own: the record and the typed
/// Inputs/Outputs/body/Scratch are captured BY VALUE in the outer kernel; there
/// is no staging allocation (fused waves stage through device buffers instead). The JIT
/// configuration is the body type's static one (jit_config<KB>()), read at submit.
template <class KB>
class KernelLaunch final : public LaunchDescription {
public:
    static constexpr int R = kernel_rank_v<KB>;
    using Scratch          = scratch_t<KB>;

    KernelLaunch(sycl::range<R> range, std::optional<sycl::range<R>> wg,
                 typename KB::Inputs in, typename KB::Outputs out, KB body, Scratch scratch)
        : payload_{StageRecord{}, in, out, BodyPack{body, scratch}} {
        wg3_ = wg ? pad3<R>(*wg) : default_wg<R>();
        StageRecord& r = payload_.rec;
        r.range        = pad3<R>(range);
        r.flat_extent  = range.size();
        r.rank         = static_cast<uint8_t>(R);
        const std::size_t wg_size = wg3_[0] * wg3_[1] * wg3_[2];
        if constexpr (GroupBody<KB>) {
            r.group_range = sycl::range<3>{(r.range[0] + wg3_[0] - 1) / wg3_[0],
                                           (r.range[1] + wg3_[1] - 1) / wg3_[1],
                                           (r.range[2] + wg3_[2] - 1) / wg3_[2]};
            r.group_count = r.group_range[0] * r.group_range[1] * r.group_range[2];
        } else {
            r.group_count = (r.flat_extent + wg_size - 1) / wg_size;
            r.group_range = sycl::range<3>{r.group_count, 1, 1};
        }
        if constexpr (HasScratch<KB>) {
            scratch_bytes_  = layout_scratch(payload_.body.scratch);
            r.scratch_bytes = scratch_bytes_;
            r.scratch_off   = static_cast<uint32_t>(
                reinterpret_cast<const std::byte*>(&payload_.body.scratch) -
                reinterpret_cast<const std::byte*>(&payload_.body));
        }
        parts_.rank        = R;
        parts_.group_body  = GroupBody<KB>;
        parts_.has_scratch = HasScratch<KB>;
        parts_.record      = r;  // offsets are 0-relative to the spans below
        parts_.inputs      = {reinterpret_cast<const std::byte*>(&payload_.in), sizeof(payload_.in)};
        parts_.outputs     = {reinterpret_cast<const std::byte*>(&payload_.out), sizeof(payload_.out)};
        parts_.body        = {reinterpret_cast<const std::byte*>(&payload_.body), sizeof(payload_.body)};
        if constexpr (HasScratch<KB>)
            parts_.scratch = {reinterpret_cast<const std::byte*>(&payload_.body.scratch),
                              sizeof(Scratch)};
        if (wg) parts_.wg = wg3_;
    }

    KernelLaunch(const KernelLaunch&)            = delete;
    KernelLaunch& operator=(const KernelLaunch&) = delete;

    bool is_custom_op() const override { return false; }
    bool is_host() const override { return false; }
    const KernelParts* kernel() const override { return &parts_; }

    std::vector<sycl::event> submit(sk::unordered_queue& q, const sycl::device& dev,
                                    const std::vector<sycl::event>& deps) const override {
        if (payload_.rec.flat_extent == 0) return {};  // nothing to run for an empty range
        return {launch_individual<R>(q, dev, deps, jit_config<KB>(), wave_nd_range<R>(payload_.rec.group_count, wg3_),
                                     payload_, scratch_bytes_)};
    }

private:
    struct BodyPack {
        KB      body;
        Scratch scratch;
    };
    struct Payload {
        StageRecord           rec;
        typename KB::Inputs   in;
        typename KB::Outputs  out;
        BodyPack              body;
    };
    static_assert(std::is_trivially_copyable_v<Payload>);

    Payload        payload_;
    sycl::range<3> wg3_{1, 1, 1};
    uint32_t       scratch_bytes_ = 0;
    KernelParts    parts_;
};

/// A custom-operation launch (D1): the body runs at submission as an
/// AdaptiveCpp custom operation with an interop_handle; it enqueues native
/// backend work on the native stream. Inputs/Outputs/body are captured by
/// value; nothing is compiled.
template <class KB>
class CustomOpLaunch final : public LaunchDescription {
public:
    CustomOpLaunch(typename KB::Inputs in, typename KB::Outputs out, KB body)
        : in_{in}, out_{out}, body_{body} {}

    bool is_custom_op() const override { return true; }
    bool is_host() const override { return false; }

    std::vector<sycl::event> submit(sk::unordered_queue& q, const sycl::device& dev,
                                    const std::vector<sycl::event>& deps) const override {
        const typename KB::Inputs  in   = in_;
        const typename KB::Outputs out  = out_;
        const KB                   body = body_;
        auto cgf = [=](sycl::handler& h) {
            h.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih) {
                CustomOp op{ih};
                body(op, in, out);
            });
        };
        return {q.submit(dev, deps, cgf)};
    }

private:
    typename KB::Inputs  in_;
    typename KB::Outputs out_;
    KB                   body_;
};

/// A host launch (D3): the body runs on the host as a host task, ordered only by `deps`.
/// Its Inputs/Outputs point at host-kind USM on the component's device.
template <class KB>
class HostLaunch final : public LaunchDescription {
public:
    HostLaunch(typename KB::Inputs in, typename KB::Outputs out, KB body)
        : in_{in}, out_{out}, body_{body} {}

    bool is_custom_op() const override { return false; }
    bool is_host() const override { return true; }

    std::vector<sycl::event> submit(sk::unordered_queue& q, const sycl::device& dev,
                                    const std::vector<sycl::event>& deps) const override {
        const typename KB::Inputs  in   = in_;
        const typename KB::Outputs out  = out_;
        const KB                   body = body_;
        auto f = [=]() { body(Host{}, in, out); };
        diag::count_host_tasks();
        return {q.async_host(dev, deps, f)};
    }

private:
    typename KB::Inputs  in_;
    typename KB::Outputs out_;
    KB                   body_;
};

}  // namespace detail

// -----------------------------------------------------------------------------
// Component<KB>
// -----------------------------------------------------------------------------

/// One KernelBody type per component: the one-kernel rule, by type.
template <KernelBodyC KB>
class Component : public ComponentBase {
public:
    /// USER, every forward: assemble Inputs/Outputs (and body config, and
    /// Scratch shapes) from the already-synced members; state the launch
    /// parameters. Returns the description the component submits (section 3).
    virtual std::unique_ptr<LaunchDescription> compute_forward() = 0;

    // ---- read off the body's signature, for a fused owner --------------------
    int  kernel_rank() const override { return kernel_rank_v<KB>; }
    bool is_group_body() const override { return GroupBody<KB>; }
    bool has_scratch() const override { return HasScratch<KB>; }
    bool is_custom_op() const override { return CustomOpBody<KB>; }
    bool is_host() const override { return HostBody<KB>; }
    detail::JitId definition_id() const override {
        if constexpr (CustomOpBody<KB> || HostBody<KB>)
            throw std::logic_error("naga::cospan: " + describe() +
                                   " is a custom-op or host body: it has no stage function");
        else
            return detail::stage_definition_id<KB>();
    }

protected:
    static constexpr int LR = launch_rank_v<KB>;

    std::unique_ptr<LaunchDescription> compute_forward_impl() override { return compute_forward(); }

    /// Kernel bodies: range, Inputs, Outputs, body (default work-group).
    std::unique_ptr<LaunchDescription> create_launch_config(sycl::range<LR> range,
                                                            typename KB::Inputs in,
                                                            typename KB::Outputs out, KB body = {})
        requires((PlainBody<KB> || GroupBody<KB>) && !HasScratch<KB>)
    {
        return make_launch(range, std::nullopt, in, out, body, detail::NoScratchValue<KB>());
    }

    /// Group bodies without Scratch, with an explicit work-group shape.
    std::unique_ptr<LaunchDescription> create_launch_config(sycl::range<LR> range, sycl::range<LR> wg,
                                                            typename KB::Inputs in,
                                                            typename KB::Outputs out, KB body = {})
        requires(GroupBody<KB> && !HasScratch<KB>)
    {
        return make_launch(range, wg, in, out, body, detail::NoScratchValue<KB>());
    }

    /// Group bodies with Scratch: the Scratch is laid out here.
    std::unique_ptr<LaunchDescription> create_launch_config(sycl::range<LR> range, sycl::range<LR> wg,
                                                            typename KB::Inputs in,
                                                            typename KB::Outputs out,
                                                            detail::scratch_t<KB> scratch,
                                                            KB body = {})
        requires HasScratch<KB>
    {
        return make_launch(range, wg, in, out, body, scratch);
    }

    /// Custom-op bodies (D1): no range; the body enqueues native work.
    std::unique_ptr<LaunchDescription> create_launch_config(typename KB::Inputs in,
                                                            typename KB::Outputs out, KB body = {})
        requires CustomOpBody<KB>
    {
        return std::make_unique<detail::CustomOpLaunch<KB>>(in, out, body);
    }

    /// Host bodies (D3): no range; the body runs on the host over host-kind USM.
    std::unique_ptr<LaunchDescription> create_launch_config(typename KB::Inputs in,
                                                            typename KB::Outputs out, KB body = {})
        requires HostBody<KB>
    {
        return std::make_unique<detail::HostLaunch<KB>>(in, out, body);
    }

    sk::kind default_storage_kind() const override {
        return HostBody<KB> ? sk::kind::host : sk::kind::device;
    }

private:
    std::unique_ptr<LaunchDescription> make_launch(sycl::range<LR> range,
                                                   std::optional<sycl::range<LR>> wg,
                                                   typename KB::Inputs in, typename KB::Outputs out,
                                                   KB body, detail::scratch_t<KB> scratch) {
        return std::make_unique<detail::KernelLaunch<KB>>(range, wg, in, out, body, scratch);
    }
};

}  // namespace naga::cospan
