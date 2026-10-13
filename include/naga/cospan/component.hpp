#pragma once

// =============================================================================
// naga/cospan/component.hpp — ComponentBase, Component<X>, CustomComponent<X> and HostComponent<X>
// =============================================================================
//
// A component is the leaf that does work, and it SUBMITS ITSELF (docs/design-review-2026-10-03.md, sections 3 and
// 4). Its forward is three framework-owned steps that call user hooks:
//
//   precompute   (1) every distinct input variable is materialized (shape hooks of the enclosing System have run);
//                (2) the user's prepare_forward() runs: an output is reshaped there with set_shape, within its definition
//                    (a tied output's shape belongs to its tie's System);
//                (3) every distinct output variable is materialized (an unknown dimension is an error);
//                (4) every member wrapper is pointed at its data.
//                Shapes (design review, section 10): a declaration says only specified or unspecified per dimension; every
//                run-time shape change is checked against the variable's definition (VarMeta::set_dims).
//   compute      the launch is described (define_forward() for a kernel component); the component submits it on the
//                unordered queue, depending on its variables' events and nothing else (submit), and records
//                its own event on them.
//   postcompute  the pushes to the variable's other placements (VarMeta::push_from).
//
// A component OWNS its instances by shared_ptr (section 1) and holds no pointer to its owner's data. It holds no
// queue and no device of its own: the device of each instance is a field of the instance (section 2), and a
// Runtime is given for the duration of a forward step.
//
// r05: a component is NOT a tree node. A driver (component_driver.hpp) owns it, is the leaf, and calls its three steps.
//
// The typed layer is by CRTP, in three types: Component<X> (kernel bodies: sycl::id<R> or GroupItem<R>, launch stated in
// define_forward()), CustomComponent<X> (static compute_forward(CustomOp&, const Inputs&, const Outputs&)) and HostComponent<X>
// (static compute_forward(const Inputs&, const Outputs&)). ONE BODY PER COMPONENT: the static compute_forward's signature
// fixes the kind and the kernel rank at COMPILE TIME, so there is no run-time branching on rank. Custom-op and host
// components cannot be fused (section 4): the fuse helper refuses them by type.

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/diagnostics.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/group_launch.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/init.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/kernel.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/model_meta.hpp"
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
    /// Asks for `bytes` of local memory in this stage's slice, for GroupItem::local<T> (kernel launches only).
    virtual void request_local_bytes(std::size_t) { throw std::logic_error("naga::cospan: request_local_bytes is for plain and group bodies"); }
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

/// cs::as_is: in set_shape, "leave this dimension alone" (it is whatever the definition, a hook or an earlier set made it).
struct AsIsTag {};
inline constexpr AsIsTag as_is{};

namespace detail {
/// What set_shape takes per dimension: an integer, or cs::as_is.
template <class E>
concept DimEntry = std::integral<std::remove_cvref_t<E>> || std::same_as<std::remove_cvref_t<E>, AsIsTag>;

template <DimEntry E>
std::optional<std::size_t> dim_entry_value(const E& e) {
    if constexpr (std::same_as<std::remove_cvref_t<E>, AsIsTag>) {
        return std::nullopt;
    } else {
        if constexpr (std::is_signed_v<E>)
            if (e < 0) throw std::invalid_argument("naga::cospan: set_shape: an extent cannot be negative");
        return static_cast<std::size_t>(e);
    }
}
}  // namespace detail

/// The shape of a ragged port (rank 2: N indices and one count per index), as a copyable handle: what a component's
/// shape_of returns for a RaggedInput / RaggedOutput, and what its set_shape takes. The counts are on the host or, for a
/// hooked ragged variable, on the device (`column`: they never come down to the host unless asked for).
struct RaggedShape {
    std::size_t                                 n = 0;      ///< the number of indices (dimension 0)
    std::uint64_t                               total = 0;  ///< the number of elements (the sum of the counts)
    std::vector<std::size_t>                    counts;     ///< the host counts (empty when `column` is set)
    std::shared_ptr<const detail::DeviceColumn> column;     ///< the device counts and offsets
    std::uint64_t                               column_version = 0;

    bool on_device() const noexcept { return column != nullptr; }

    /// The counts on the host: the host array, or the device column copied down (waiting on it).
    std::vector<std::size_t> counts_on_host(detail::Runtime& rt) const {
        detail::ResolvedShape s = to_resolved();
        return s.per_index_on_host(rt);
    }

    /// From a variable's resolved shape (a dense shape reads as constant rows).
    static RaggedShape from(const detail::ResolvedShape& s) {
        RaggedShape r;
        r.n = s.extents[0];
        if (s.per) {
            r.counts         = s.per_index;
            r.column         = s.dev_per;
            r.column_version = s.dev_version;
            r.total          = s.total;
        } else {
            r.counts.assign(r.n, s.extents[1] * s.extents[2]);
            r.total = static_cast<std::uint64_t>(r.n) * s.extents[1] * s.extents[2];
        }
        return r;
    }

    /// As a resolved shape (rank 2, ragged); the host total is the sum of the counts.
    detail::ResolvedShape to_resolved() const {
        detail::ResolvedShape s;
        s.known       = true;
        s.extents     = {n, 0, 1};
        s.per         = true;
        s.per_index   = counts;
        s.dev_per     = column;
        s.dev_version = column_version;
        s.settle_total();
        return s;
    }
};

/// What does the work, apart from where it sits in the tree (r05). A component has no name of its own and is never a tree node:
/// a DRIVER (component_driver.hpp) owns it, is the leaf, and gives it its full name (bind) before declare() runs. The
/// full name is the driver's for a lone component (`c`), and `<group>.<stage>` for a stage of a fused group.
class ComponentBase {
public:
    ComponentBase(const ComponentBase&)            = delete;
    ComponentBase& operator=(const ComponentBase&) = delete;
    virtual ~ComponentBase()                       = default;

    // ---- HOOKS (user-implemented) ------------------------------------------
    /// Declare I/O. Runs once, during setup. Zero allocation.
    virtual void declare() = 0;
    /// There is no init hook: a free input's initial value is declared in declare() with init_input, and outputs are
    /// never initialized.
    /// Per-forward shape control: shape_of / set_shape are legal here only.
    virtual void prepare_forward() {}

    // ---- declaration API (legal inside declare() only) ------------------------
    // A declaration says, per dimension, only SPECIFIED (an integer) or UNSPECIFIED (cs::unspecified); it is the
    // component's contract (docs/design-review-2026-10-03.md, section 10). Specified forces that dimension on anything
    // joined to it; unspecified places no constraint. There are no derived specs.
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

    /// The array forms: cs::unspecified_shape<R>() (or any std::array of DimSpecs).
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

    /// Ragged ports are RANK 2 ONLY (a dense leading dimension N and one per-index dimension). A declaration is one of:
    ///   (m, name)                    both unspecified
    ///   (m, name, n)                 the leading dimension specified, the counts unspecified
    ///   (m, name, cs::counts(v))     a host array of counts; it defines the leading dimension (v.size()) too
    ///   (m, name, cs::offsets(v))    the same as N + 1 offsets
    template <DType DT, int Rank>
    void declare_input(RaggedInput<DT, Rank>& m, std::string name) {
        add_ragged_input<DT, Rank>(m, std::move(name), ragged_records(std::nullopt, false), std::nullopt);
    }
    template <DType DT, int Rank, std::integral N>
    void declare_input(RaggedInput<DT, Rank>& m, std::string name, N n) {
        add_ragged_input<DT, Rank>(m, std::move(name), ragged_records(leading_of(n), false), std::nullopt);
    }
    template <DType DT, int Rank>
    void declare_input(RaggedInput<DT, Rank>& m, std::string name, Counts c) {
        const std::size_t n = c.v.size();
        add_ragged_input<DT, Rank>(m, std::move(name), ragged_records(n, true), std::move(c.v));
    }
    template <DType DT, int Rank>
    void declare_input(RaggedInput<DT, Rank>& m, std::string name, const Offsets& o) {
        declare_input<DT, Rank>(m, std::move(name), counts_of(o));
    }
    template <DType DT, int Rank>
    void declare_output(RaggedOutput<DT, Rank>& m, std::string name) {
        add_ragged_output<DT, Rank>(m, std::move(name), ragged_records(std::nullopt, false), std::nullopt);
    }
    template <DType DT, int Rank, std::integral N>
    void declare_output(RaggedOutput<DT, Rank>& m, std::string name, N n) {
        add_ragged_output<DT, Rank>(m, std::move(name), ragged_records(leading_of(n), false), std::nullopt);
    }
    template <DType DT, int Rank>
    void declare_output(RaggedOutput<DT, Rank>& m, std::string name, Counts c) {
        const std::size_t n = c.v.size();
        add_ragged_output<DT, Rank>(m, std::move(name), ragged_records(n, true), std::move(c.v));
    }
    template <DType DT, int Rank>
    void declare_output(RaggedOutput<DT, Rank>& m, std::string name, const Offsets& o) {
        declare_output<DT, Rank>(m, std::move(name), counts_of(o));
    }

    // ---- shapes in prepare_forward (legal inside prepare_forward() only) -------------
    // Inputs are never reshaped: an input's shape comes from its definition, its initializer, its connection's source, or
    // a hook. An OUTPUT is reshaped only here, by its producing component, within its definition. An output that is a
    // member of a tie_outputs cannot be reshaped by a member: its tie's System sets it with a shape hook.

    /// The shape of one of this component's own ports NOW (its variable's shape). An input's is known here; an output's
    /// only after an earlier forward or when its definition fixes it.
    template <DType DT, int Rank>
    sycl::range<Rank> shape_of(const Input<DT, Rank>& m) const {
        return dense_shape_of<Rank>(&m);
    }
    template <DType DT, int Rank>
    sycl::range<Rank> shape_of(const Output<DT, Rank>& m) const {
        return dense_shape_of<Rank>(&m);
    }
    template <DType DT, int Rank>
    RaggedShape shape_of(const RaggedInput<DT, Rank>& m) const {
        return ragged_shape_of(&m);
    }
    template <DType DT, int Rank>
    RaggedShape shape_of(const RaggedOutput<DT, Rank>& m) const {
        return ragged_shape_of(&m);
    }

    /// set_shape(out, e0, e1, ...): one entry per dimension, each an integer or cs::as_is (the number of entries must equal
    /// the rank). A number is allowed only on an UNSPECIFIED dimension: on a specified or hook-owned one it is an error at
    /// the call, even when the value matches, naming the dimension and its owner. as_is leaves a dimension alone; one that
    /// is still unknown at materialization is the shape-not-known error.
    template <DType DT, int Rank, detail::DimEntry... E>
    void set_shape(Output<DT, Rank>& m, E... entries) {
        detail::Instance& inst = settable_output(&m, "set_shape");
        if (sizeof...(E) != static_cast<std::size_t>(Rank))
            throw std::invalid_argument("naga::cospan: set_shape('" + inst.full_path() + "') takes one entry per dimension (" +
                                        std::to_string(Rank) + ") but was given " + std::to_string(sizeof...(E)));
        std::array<std::optional<std::size_t>, 3> given{};  // the count check above makes sizeof...(E) == Rank <= 3
        std::size_t                               k = 0;
        ((given[k++] = detail::dim_entry_value(entries)), ...);
        inst.meta->set_by_producer(*rt_, given, inst.full_path());
    }
    /// A ragged output's shape from a RaggedShape (read from an input's shape_of, or built): N and the counts, each only on a
    /// dimension that is unspecified.
    template <DType DT, int Rank>
    void set_shape(RaggedOutput<DT, Rank>& m, const RaggedShape& shape) {
        static_assert(Rank == 2, "cospan: a ragged port is rank 2");
        detail::Instance& inst = settable_output(&m, "set_shape");
        inst.meta->set_ragged_by_producer(*rt_, shape.to_resolved(), /*set_n=*/true, /*set_counts=*/true, inst.full_path());
    }
    /// A ragged output's shape from N (a number, or cs::as_is) and its host counts (counts.size() == n).
    template <DType DT, int Rank, detail::DimEntry N>
    void set_shape(RaggedOutput<DT, Rank>& m, N n, std::vector<std::size_t> counts) {
        static_assert(Rank == 2, "cospan: a ragged port is rank 2");
        detail::Instance&            inst = settable_output(&m, "set_shape");
        const std::optional<std::size_t> nn = detail::dim_entry_value(n);
        detail::ResolvedShape s;
        s.known     = true;
        s.per       = true;
        s.per_index = std::move(counts);
        s.extents   = {nn ? *nn : s.per_index.size(), 0, 1};
        s.settle_total();
        inst.meta->set_ragged_by_producer(*rt_, s, nn.has_value(), /*set_counts=*/true, inst.full_path());
    }
    /// A ragged output's N alone (the counts are left to their definition or a hook).
    template <DType DT, int Rank, detail::DimEntry N>
    void set_shape(RaggedOutput<DT, Rank>& m, N n, AsIsTag) {
        static_assert(Rank == 2, "cospan: a ragged port is rank 2");
        detail::Instance&            inst = settable_output(&m, "set_shape");
        const std::optional<std::size_t> nn = detail::dim_entry_value(n);
        if (!nn) return;
        detail::ResolvedShape s = inst.meta->shape;
        s.extents[0]            = *nn;
        inst.meta->set_ragged_by_producer(*rt_, s, /*set_n=*/true, /*set_counts=*/false, inst.full_path());
    }

    // ---- initial values (legal inside declare() only) -----------------------
    /// Declare the initial value of one of this Component's OWN inputs, by name. Metadata on the input's variable
    /// (section 1); the last declaration wins (a parent's setup runs after its children's, section 2), it is ignored when
    /// the input ends up with a source, and it is an error to name an output.
    void init_input(std::string_view name, InitSpec init) {
        if (!declaring_)
            throw std::logic_error("naga::cospan: init_input is legal inside declare() only (" + describe() + ")");
        const std::shared_ptr<detail::Instance> inst = find_kind(name, IOKind::input);  // an output throws
        detail::declare_init(detail::WiringSite{describe()}, inst, std::move(init));
    }

    /// The full name this component was bound to (empty before bind).
    const std::string& path() const noexcept { return full_name_; }

    /// The last launch's event(s): what depends_on hands on. (Deleting a component waits on nothing: the containers
    /// it held free themselves after their own events.)
    std::vector<sycl::event> events() const { return last_events_; }

    /// depends_on: these events are added to this component's submission for ONE forward.
    void add_dependencies(std::span<const sycl::event> deps) {
        pending_deps_.insert(pending_deps_.end(), deps.begin(), deps.end());
    }

    // ---- the forward ---------------------------------------------------------
    // ONE forward (r04) is the three steps below, called by the component's driver (a lone component: SoloDriver) or, for the
    // stages of a fused group, in pieces around the group's single launch.

    /// The pre steps below; a kernel component that launches by itself (not as a stage of a fused group) then keeps its
    /// argument array and, at the first forward, its launcher's pointer array.
    void step_pre(detail::Runtime& rt) {
        precompute_steps(rt);
        if (is_kernel_stage()) {
            stage_args_ensure(rt);
            solo_.prepare(rt, std::vector<ComponentBase*>{this});
        }
    }

    /// Steps (1) to (4) of the precompute (what a fused group runs for each of its stages).
    void precompute_steps(detail::Runtime& rt) {
        PtrScope rt_scope{rt_, &rt};

        // (1) Every input variable is materialized (a free input takes its initializer, and its initializer's shape,
        // here), by the holder that asks: the first materialization is global, a later one checks only this holder's
        // container. Shape hooks of the enclosing System have already run (docs/design-review-2026-10-03.md, section 10).
        for (const auto& inst : order_)
            if (inst->kind == IOKind::input) inst->meta->materialize(rt, inst.get());

        // (2) The user's prepare_forward(): where an output is reshaped (set_shape) within its definition.
        {
            detail::FlagScope scope{in_prepare_forward_};
            prepare_forward();
        }

        // (3) Every distinct output variable is materialized: an unknown dimension is the not-known error, naming who
        // should have set it.
        for (const auto& inst : order_)
            if (inst->kind == IOKind::output) inst->meta->materialize(rt, inst.get());

        // (4) The member wrappers (a dense port on a ragged variable is an error).
        for (const auto& inst : order_) check_dense(*inst, inst->meta->shape);
        for (const auto& inst : order_) inst->point_wrapper();
    }

    /// Submits the launch and records its event: describe, submit, record. The launch waits on each input container's
    /// read_deps (its writes) and each output container's write_deps (its reads), and writes never wait on writes, so
    /// tied outputs run concurrently; there is no queue-wide wait anywhere (section 3). The pieces are protected so a
    /// fused component (section 4) can submit ONE launch for its stages and still give each stage the event as if it had
    /// submitted alone.
    void step_submit(detail::Runtime& rt) {
        if (is_kernel_stage()) {
            // A plain or group body launches through the one path: a group of one stage in one default segment.
            (void)solo_.launch(rt, std::vector<ComponentBase*>{this}, std::vector<detail::SegmentSpec>{detail::SegmentSpec{}}, describe());
            return;
        }
        // Custom-op and host bodies keep their own path.
        const std::unique_ptr<LaunchDescription> desc = describe_launch();
        if (!desc) throw std::logic_error("naga::cospan: " + describe() + ": the component described no launch");
        const std::vector<sycl::event> ev = desc->submit(rt.uq, device_of(), gather_deps(rt));
        record_launch(ev);
    }

    /// Pushes each output variable this component wrote to the variable's OTHER live placements (host views included)
    /// by container assignment, which waits on the output's writes and registers the read, unless the variable's push is
    /// suppressed (VarMeta::push_from). A tied output is pushed by the tie's System. (The launch's events were
    /// registered on the containers at submit, in record_launch.)
    void step_post(detail::Runtime& rt) {
        (void)rt;
        std::vector<detail::VarMeta*> seen;
        for (const auto& inst : order_) {
            if (inst->kind != IOKind::output) continue;
            detail::VarMeta* m = inst->meta.get();
            if (std::find(seen.begin(), seen.end(), m) != seen.end()) continue;
            seen.push_back(m);
            if (m->is_tied_output() || m->push_suppressed || !inst->data || last_events_.empty()) continue;
            // a holder not materialized yet fills itself (push_from skips unbound data)
            (void)m->push_from(inst->data);
        }
    }

    /// The first instance that is a ragged port (declared RaggedInput / RaggedOutput) or whose variable is ragged, else
    /// null. A fused component refuses such ports (section 4).
    std::shared_ptr<detail::Instance> first_ragged_port() const {
        for (const auto& inst : order_)
            if (ragged_ports_.contains(inst.get()) || inst->meta->is_ragged()) return inst;
        return nullptr;
    }

    // ---- what a fused owner needs to know, without knowing the KernelBody type --
    // Answered by Component<X> from its body's signature. An owner reads
    // these; nothing here knows what fusing is.
    virtual int  kernel_rank() const { return 0; }       ///< 0 for custom-op and host bodies
    virtual bool is_group_body() const { return false; }
    virtual bool has_scratch() const { return false; }
    virtual bool is_custom_op() const { return false; }
    virtual bool is_host() const { return false; }
    /// A plain or group body: it launches through a GroupLauncher (alone, or as a stage of a fused group).
    virtual bool is_kernel_stage() const { return false; }
    /// The type-erased id of this body's stage function (stage_fn<KB>), for a
    /// call sequence. Throws for a custom-op or host body (nothing is compiled).
    virtual detail::JitId definition_id() const {
        throw std::logic_error("naga::cospan: " + describe() + " has no stage function");
    }

    // ---- the stage interface of a group launcher (kernel bodies only; implemented by Component<X>) ----
    /// The component's one-element argument array on its device: created (capacity 1, so its address is final) the first
    /// time.
    virtual void stage_args_ensure(detail::Runtime&) { throw_no_stage(); }
    /// Its device address, as the type-erased pointer the group's pointer array holds.
    virtual const void* stage_args_ptr() const { throw_no_stage(); }
    /// What a launch reading the array must wait on (its write list), and the hand-over of that launch's event.
    virtual std::vector<sycl::event> stage_args_deps() const { throw_no_stage(); }
    virtual void                     stage_args_register_read(const sycl::event&) { throw_no_stage(); }
    /// Finishes this forward's description for the group's work-group shape `wg`, copies its arguments into the array
    /// (a container write) and returns the stage's group range: per dimension (trailing dimensions, leading 1s) the groups it
    /// needs, all 0 when the stage is empty.
    virtual sycl::range<3> stage_upload(detail::Runtime&, LaunchDescription&, const sycl::range<3>&) { throw_no_stage(); }

protected:
    ComponentBase() = default;

    /// A host view of one of this component's OWN inputs, made with the runtime its precompute received: legal inside
    /// prepare_forward() only. It is a host holder like any other (the host is just another device of the variable): the
    /// variable is pushed to it every forward while it lives, so keep it as a member (an std::optional, made on the first
    /// forward) and call sync() before reading. The typical use is to read values in prepare_forward and set_shape an
    /// output from them. There is no equivalent for outputs. The input is already materialized here (precompute step 1).
    template <DType DT, int Rank>
    HostView<DT, Rank> view_input(const Input<DT, Rank>& m) {
        const detail::Instance& inst = member_instance(&m, "view_input");
        return detail::HostViewAccess::wrap<HostView<DT, Rank>>(
            detail::make_view_instance(*rt_, inst, DT, Rank, /*dense=*/true));
    }
    /// The same for a ragged input: a HostRagged view.
    template <DType DT, int Rank>
    HostRagged<DT, Rank> view_input(const RaggedInput<DT, Rank>& m) {
        const detail::Instance& inst = member_instance(&m, "view_input");
        return detail::HostViewAccess::wrap<HostRagged<DT, Rank>>(
            detail::make_view_instance(*rt_, inst, DT, Rank, /*dense=*/false));
    }

    /// The storage kind of this Component's I/O: device, or host for a host body.
    virtual sk::kind default_storage_kind() const { return sk::kind::device; }

    /// The launch of this forward: implemented by Component<X> (from define_forward()), CustomComponent<X> and HostComponent<X>.
    virtual std::unique_ptr<LaunchDescription> describe_launch() = 0;

    /// "'<full name>'": the prefix of every error.
    std::string describe() const { return "'" + full_name_ + "'"; }

    /// What a submission of this component must come after: each input container's read_deps (its writes), each output
    /// container's write_deps (its reads) together with its variable's pending resize events (VarMeta::resize_events: a
    /// resize allocates uninitialized, and its copy of the old elements must land before a kernel writes), and any
    /// depends_on events (which last one forward: they are cleared here). Outside writes never wait on each other.
    /// Bare components and every stage of a fused group come through here; a tied output's members share the variable.
    std::vector<sycl::event> gather_deps(detail::Runtime& rt) {
        (void)rt;
        std::vector<sycl::event> deps;
        const auto               add = [&](const std::vector<sycl::event>& v) { deps.insert(deps.end(), v.begin(), v.end()); };
        for (const auto& inst : order_) {
            if (!inst->data) continue;
            if (inst->kind == IOKind::input) {
                add(inst->data->read_deps());
            } else {
                add(inst->data->write_deps());
                add(inst->meta->resize_events());
            }
        }
        add(pending_deps_);
        pending_deps_.clear();
        return deps;
    }

    /// Records the event(s) of this component's launch THE MOMENT IT IS SUBMITTED: a READ on every input container, a
    /// WRITE on every output container, and the component's last event (what depends_on hands on). Registered at submit,
    /// so nothing that frees or reallocates a container before postcompute can miss the kernel still using it.
    void record_launch(const std::vector<sycl::event>& ev) {
        std::vector<detail::DeviceDataBase*> read, written;  // a container may be both: once in each list
        for (const auto& inst : order_) {
            if (!inst->data) continue;
            std::vector<detail::DeviceDataBase*>& done = inst->kind == IOKind::input ? read : written;
            if (std::find(done.begin(), done.end(), inst->data.get()) != done.end()) continue;
            done.push_back(inst->data.get());
            if (inst->kind == IOKind::input) inst->data->register_read(std::span<const sycl::event>{ev});
            else inst->data->register_write(std::span<const sycl::event>{ev});  // the kernel writes it
        }
        last_events_ = ev;
    }

    /// The component's device: its first instance's (set_device moves them all together), else the default.
    sycl::device device_of() const { return order_.empty() ? detail::default_device() : order_.front()->device; }

    std::vector<sycl::event> last_events_;  ///< the last launch's events
    std::vector<sycl::event> pending_deps_; ///< depends_on events, for the next submission only

    [[noreturn]] void throw_no_stage() const {
        throw std::logic_error("naga::cospan: " + describe() + " is not a plain or group body: it has no stage interface");
    }

private:
    friend struct detail::ComponentAccess;

    /// Gives the component its full name, its driver (the leaf in the tree: holders answer to its node) and the model's
    /// metadata. Called by the driver (ComponentAccess) before declare() runs.
    void bind(detail::ModelMeta* m, std::string full_name, const Subsystem* driver) {
        meta_      = m;
        full_name_ = std::move(full_name);
        driver_    = driver;
        if (m != nullptr) m->components[full_name_] = this;
    }

    /// Runs declare() with the declaring flag and the metadata's setup flag up.
    void run_declare() {
        detail::FlagScope             scope{declaring_};
        detail::ModelMeta::SetupScope setup_scope{meta_};
        declare();
    }

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

    /// A ragged port's two records: the leading dimension (`lead`, else unspecified) and the per-index dimension (fixed when
    /// its counts are declared, else unspecified).
    static std::array<detail::DimRecord, 2> ragged_records(const std::optional<std::size_t>& lead, bool counts_declared) {
        std::array<detail::DimRecord, 2> out{};
        out[0].how    = lead ? detail::DimRecord::How::fixed : detail::DimRecord::How::unspecified;
        out[0].extent = lead.value_or(0);
        out[1].how    = counts_declared ? detail::DimRecord::How::fixed : detail::DimRecord::How::unspecified;
        out[1].extent = 0;
        return out;
    }
    template <std::integral N>
    static std::size_t leading_of(N n) {
        if constexpr (std::is_signed_v<N>) {
            if (n < 0) throw std::invalid_argument("naga::cospan: a fixed extent is negative");
        }
        return static_cast<std::size_t>(n);
    }
    static Counts counts_of(const Offsets& o) {
        Counts c;
        c.v.resize(o.v.size() - 1);
        for (std::size_t i = 0; i + 1 < o.v.size(); ++i) c.v[i] = o.v[i + 1] - o.v[i];
        return c;
    }

    template <DType DT, int Rank>
    void add_input(Input<DT, Rank>& m, std::string name,
                   const std::array<detail::DimRecord, Rank>& dims) {
        register_instance(detail::make_input<DT, Rank>(std::move(name), full_name_, driver_, dims, &m), &m, "declare_input", false);
    }
    template <DType DT, int Rank>
    void add_output(Output<DT, Rank>& m, std::string name,
                    const std::array<detail::DimRecord, Rank>& dims) {
        register_instance(detail::make_output<DT, Rank>(std::move(name), full_name_, driver_, dims, &m), &m, "declare_output", false);
    }
    template <DType DT, int Rank>
    void add_ragged_input(RaggedInput<DT, Rank>& m, std::string name, const std::array<detail::DimRecord, Rank>& dims,
                          std::optional<std::vector<std::size_t>> counts) {
        static_assert(Rank == 2, "cospan: a ragged port is rank 2");
        register_instance(detail::make_ragged_input<DT, Rank>(std::move(name), full_name_, driver_, dims, &m, std::move(counts)), &m,
                          "declare_input", true);
    }
    template <DType DT, int Rank>
    void add_ragged_output(RaggedOutput<DT, Rank>& m, std::string name, const std::array<detail::DimRecord, Rank>& dims,
                           std::optional<std::vector<std::size_t>> counts) {
        static_assert(Rank == 2, "cospan: a ragged port is rank 2");
        register_instance(detail::make_ragged_output<DT, Rank>(std::move(name), full_name_, driver_, dims, &m, std::move(counts)), &m,
                          "declare_output", true);
    }

    // ---- shapes in prepare_forward -----------------------------------------
    /// The declared instance of a member, inside prepare_forward only.
    detail::Instance& member_instance(const void* member, const char* what) const {
        if (!in_prepare_forward_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside prepare_forward() only (" + describe() + ")");
        const auto it = by_member_.find(member);
        if (it == by_member_.end())
            throw std::invalid_argument("naga::cospan: " + describe() + ": " + what + " of a member that was never declared");
        if (rt_ == nullptr) throw std::logic_error("naga::cospan: " + describe() + ": " + what + " outside a forward");
        return *it->second;
    }

    /// The output instance a set_shape is for: not a member of a tie_outputs (the tie call marked it: its tie's System owns
    /// the shape).
    detail::Instance& settable_output(const void* member, const char* what) {
        detail::Instance& inst = member_instance(member, what);
        // (a member reached through a promotion is not marked itself, but its variable is a tied output)
        if (inst.tied_member || inst.meta->tied_output)
            throw std::invalid_argument("naga::cospan: '" + inst.full_path() +
                                        "' is tied; its shape is set by its tie's System (a shape hook), not by a member");
        return inst;
    }

    template <int Rank>
    sycl::range<Rank> dense_shape_of(const void* member) const {
        const detail::Instance&      inst = member_instance(member, "shape_of");
        const detail::ResolvedShape& s    = inst.meta->shape;
        if (!s.known) throw inst.meta->shape_unknown();
        if (s.ragged())
            throw std::invalid_argument("naga::cospan: '" + inst.full_path() + "' is ragged; read it with a ragged port");
        return detail::make_range<Rank>(s.extents);
    }
    RaggedShape ragged_shape_of(const void* member) const {
        const detail::Instance&      inst = member_instance(member, "shape_of");
        const detail::ResolvedShape& s    = inst.meta->shape;
        if (!s.known) throw inst.meta->shape_unknown();
        return RaggedShape::from(s);
    }

    /// The component owns its instances (section 1); their device is the default until set_device says otherwise.
    void register_instance(std::shared_ptr<detail::Instance> inst, const void* member, const char* what, bool ragged) {
        if (!declaring_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside declare() only (" + describe() + ")");
        const std::string key = inst->name;
        if (by_name_.contains(key))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + key + "' is declared twice");
        if (by_member_.contains(member))
            throw std::invalid_argument("naga::cospan: " + describe() + ": member wrapper of '" + key +
                                        "' is already declared");
        if (meta_ == nullptr)
            throw std::logic_error("naga::cospan: " + describe() + " is not part of a model yet (it is registered by add_subsystem)");
        inst->storage_kind = default_storage_kind();
        inst->refresh_container();  // the container of its placement, empty until the first materialization
        // The holder is registered in the model-wide map under the component's full name + "." + its own name, and answers to
        // its driver's node (a fused group's stage: the group's).
        meta_->register_holder(detail::ModelMeta::child_name(full_name_, key), inst, meta_->node_for(driver_), describe());
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

    /// The distinct data this component's instances hold (null ones skipped).
    std::vector<detail::DeviceDataBase*> distinct_data() const {
        std::vector<detail::DeviceDataBase*> out;
        for (const auto& inst : order_)
            if (inst->data && std::find(out.begin(), out.end(), inst->data.get()) == out.end()) out.push_back(inst->data.get());
        return out;
    }

    /// A dense port cannot see per-index extents: a variable with a per-index dimension needs RaggedInput/RaggedOutput.
    void check_dense(const detail::Instance& inst, const detail::ResolvedShape& s) const {
        if (s.known && s.ragged() && !ragged_ports_.contains(&inst))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + inst.name +
                                        "' is ragged; declare it with RaggedInput/RaggedOutput");
    }

    detail::GroupLauncher solo_;  ///< the launcher of a kernel component that is not a stage of a fused group

    std::unordered_map<std::string, std::shared_ptr<detail::Instance>> by_name_;
    std::unordered_map<const void*, detail::Instance*>                 by_member_;
    std::vector<std::shared_ptr<detail::Instance>>                     order_;  // declaration order
    std::unordered_set<const detail::Instance*>                        ragged_ports_;

    detail::ModelMeta*  meta_           = nullptr;  ///< the model's metadata (null before bind)
    std::string         full_name_;                     ///< bound by the driver; the owner_path of this component's holders
    const Subsystem*    driver_         = nullptr;      ///< the leaf node this component belongs to
    bool                declaring_         = false;
    bool                in_prepare_forward_ = false;
    detail::Runtime*    rt_                = nullptr;  ///< set during the pre steps of a forward only
};

namespace detail {

/// INTERNAL: what a fused owner (phase 7) calls to get a stage's complete launch without submitting it.
struct ComponentAccess {
    static std::unique_ptr<LaunchDescription> describe(ComponentBase& c) { return c.describe_launch(); }
    static void bind(ComponentBase& c, ModelMeta* m, std::string full_name, const Subsystem* driver) {
        c.bind(m, std::move(full_name), driver);
    }
    static void run_declare(ComponentBase& c) { c.run_declare(); }
    static std::vector<sycl::event>            gather_deps(ComponentBase& c, Runtime& rt) { return c.gather_deps(rt); }
    static void                                record_launch(ComponentBase& c, const std::vector<sycl::event>& ev) { c.record_launch(ev); }
    static sycl::device                        device_of(const ComponentBase& c) { return c.device_of(); }
};

// ---- GroupLauncher (declared in group_launch.hpp) ----------------------------------------------------------------------------

inline void GroupLauncher::prepare(Runtime& rt, const std::vector<ComponentBase*>& stages) {
    if (ptrs_ || stages.empty()) return;  // once: redone only when the group is rebuilt (a new launcher)
    const sycl::device       dev = ComponentAccess::device_of(*stages.front());
    std::vector<const void*> p;
    p.reserve(stages.size());
    for (ComponentBase* s : stages) p.push_back(s->stage_args_ptr());
    ptrs_.emplace(rt.allocator_for<const void*>(dev, sk::kind::device));
    ptrs_->from_host(std::span<const void* const>{p});
    sycl::event::wait(ptrs_->read_deps());  // filled and waited on: the array is final
    geo_.emplace(rt.allocator_for<StageGeometry>(dev, sk::kind::device));
}

inline std::vector<sycl::event> GroupLauncher::launch(Runtime& rt, const std::vector<ComponentBase*>& st,
                                                      const std::vector<SegmentSpec>& segs, const std::string& who, bool group_waits,
                                                      Split split) {
    if (st.empty()) throw std::invalid_argument("naga::cospan: " + who + ": a group has no stages");
    if (!ptrs_) throw std::logic_error("naga::cospan: " + who + ": forward before the first precompute");
    const std::size_t n = st.size();
    {
        std::size_t covered = 0;
        for (const SegmentSpec& s : segs) covered += s.count;
        if (covered != n) throw std::logic_error("naga::cospan: " + who + ": the segments do not cover the stages");
    }

    // 1. Every stage's description of this forward, kept alive until the arguments are uploaded.
    std::vector<std::unique_ptr<LaunchDescription>> ds;
    std::vector<const KernelParts*>                  parts;
    int                                              rank = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::unique_ptr<LaunchDescription> d = ComponentAccess::describe(*st[i]);
        if (!d)
            throw std::logic_error("naga::cospan: " + who + ": forward: '" + st[i]->path() + "' returned no launch description; a stage must describe its work");
        const KernelParts* p = (d->is_custom_op() || d->is_host()) ? nullptr : d->kernel();
        if (p == nullptr)
            throw std::logic_error("naga::cospan: " + who + ": forward: '" + st[i]->path() + "' described no kernel; a stage must be a kernel body");
        if (i == 0)
            rank = p->rank;
        else if (p->rank != rank)
            throw std::logic_error("naga::cospan: " + who + ": forward: '" + st[i]->path() + "' described a rank-" + std::to_string(p->rank) +
                                   " launch; the group is rank " + std::to_string(rank));
        parts.push_back(p);
        ds.push_back(std::move(d));
    }

    // 2. The group's work-group shape: every stage that asks for one must agree (checked here, every forward, before any
    // launch: a component may choose its shape at run time); none asking, the default for the rank.
    std::optional<sycl::range<3>> wg_opt;
    std::size_t                   first = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!parts[i]->wg) continue;
        if (!wg_opt) {
            wg_opt = parts[i]->wg;
            first  = i;
        } else if (*wg_opt != *parts[i]->wg) {
            throw std::invalid_argument("naga::cospan: " + who + ": forward: '" + st[first]->path() + "' and '" + st[i]->path() +
                                        "' ask for different work-group shapes; the stages of a group share one");
        }
    }
    if (!wg_opt) {
        switch (rank) {
            case 1: wg_opt = default_wg<1>(); break;
            case 2: wg_opt = default_wg<2>(); break;
            default: wg_opt = default_wg<3>(); break;
        }
    }
    sycl::range<3> wg = *wg_opt;
    for (int d = 0; d < 3 - rank; ++d) wg[d] = 1;  // the leading dimensions (before the rank's trailing ones) do not exist
    if (split == Split::padded && (rank > 2 || wg[0] != 1))
        throw std::invalid_argument("naga::cospan: " + who + ": cs::range_split::padded needs stages of rank 2 or lower");

    // 3. Every stage's arguments into its own array (a container write), and its group count.
    std::vector<sycl::range<3>> counts(n, sycl::range<3>{0, 0, 0});
    for (std::size_t i = 0; i < n; ++i) counts[i] = st[i]->stage_upload(rt, *ds[i], wg);

    // 4. The geometry: per segment the block (the largest count), the groups before it, the local slices.
    // The segment table after the n entries: the S block sizes, then the S boundary kinds.
    const std::size_t nseg  = segs.size();
    const std::size_t table = (nseg * sizeof(SegmentInfo) + sizeof(StageGeometry) - 1) / sizeof(StageGeometry);
    host_geo_.assign(n + table, StageGeometry{});
    std::vector<SegmentInfo>   infos(nseg);
    std::uint64_t              pad_offset = 0;  // pad_and_split: where the next segment starts along the last dimension
    bool                       ticket_mode = group_waits;  // the group's property; a global_barrier below also switches it on
    std::uint64_t              base = 0;
    std::size_t                local_bytes = 0;
    for (std::size_t s = 0; s < segs.size(); ++s) {
        const SegmentSpec& sp = segs[s];
        std::array<std::uint64_t, 3> block{0, 0, 0};  // the per-dimension maximum of the stages' group ranges
        for (std::size_t j = sp.first; j < sp.first + sp.count; ++j)
            for (int d = 0; d < 3; ++d) block[d] = std::max<std::uint64_t>(block[d], counts[j][d]);
        const std::uint64_t block_count = block[0] * block[1] * block[2];
        std::size_t seg_local = 0;
        for (std::size_t j = sp.first; j < sp.first + sp.count; ++j) {
            StageGeometry& g = host_geo_[j];
            g.segment        = static_cast<std::uint32_t>(s);
            g.split          = static_cast<std::uint32_t>(split);
            g.segments       = static_cast<std::uint32_t>(segs.size());
            g.stages         = static_cast<std::uint32_t>(n);
            g.range_mode     = static_cast<std::uint32_t>(sp.range);
            g.local_bytes    = static_cast<std::uint32_t>(parts[j]->local_bytes);
            if (sp.local == LocalMem::partitioned) {
                g.local_offset = static_cast<std::uint32_t>(seg_local);
                seg_local += parts[j]->local_bytes;  // each slice is a multiple of 16
            } else {
                g.local_offset = 0;
                seg_local      = std::max(seg_local, parts[j]->local_bytes);
            }
        }
        SegmentInfo& si = infos[s];
        for (int d = 0; d < 3; ++d) si.range[d] = block[d];
        si.count    = block_count;
        si.first    = base;
        si.offset   = pad_offset;
        si.boundary = static_cast<std::uint64_t>(sp.boundary);
        pad_offset += block[2];
        if (sp.boundary == Boundary::global_barrier) ticket_mode = true;
        base += block_count;
        local_bytes = std::max(local_bytes, seg_local);
    }
    const sycl::device dev         = ComponentAccess::device_of(*st.front());
    const std::size_t  local_limit = dev.get_info<sycl::info::device::local_mem_size>();
    if (local_bytes > local_limit) {
        std::string scratch_stages;
        for (std::size_t i = 0; i < n; ++i)
            if (parts[i]->local_bytes != 0)
                scratch_stages += (scratch_stages.empty() ? "'" : ", '") + st[i]->path() + "' (" + std::to_string(parts[i]->local_bytes) + " bytes)";
        throw std::invalid_argument("naga::cospan: " + who + ": forward: the launch needs " + std::to_string(local_bytes) +
                                    " bytes of local memory (the largest segment's) but the device offers " + std::to_string(local_limit) +
                                    "; stages with local memory: " + scratch_stages);
    }
    if (base == 0) {  // every stage is empty: nothing launched, and no stale events are kept
        for (ComponentBase* s : st) {
            (void)ComponentAccess::gather_deps(*s, rt);  // depends_on events last one forward, launched or not
            ComponentAccess::record_launch(*s, {});
        }
        last_.clear();
        return {};
    }
    for (std::size_t i = 0; i < n; ++i) host_geo_[i].total_groups = base;
    std::memcpy(static_cast<void*>(&host_geo_[n]), infos.data(), infos.size() * sizeof(SegmentInfo));

    // Group waits (ticket mode): the ticket state (counter, per-segment completed counters, per-ticket flags) is sized for this launch and
    // reset with a container fill, ordered after the previous launch (an outside writer of it). The launch then waits on
    // the fill (read_deps). Its address goes into every geometry entry.
    if (ticket_mode) {
        if (!ticket_) ticket_.emplace(rt.allocator_for<std::uint32_t>(dev, sk::kind::device));
        const std::size_t       words = 1 + nseg + static_cast<std::size_t>(base);
        const std::vector<sycl::event> prior = ticket_->read_deps();  // the previous launch, registered as a write
        if (ticket_->size() != words) (void)ticket_->resize(words, std::span<const sycl::event>{prior});
        (void)ticket_->fill(std::uint32_t{0}, std::span<const sycl::event>{prior});
        for (std::size_t i = 0; i < n; ++i) host_geo_[i].ticket_state = ticket_->data();
    }
    if (geo_->capacity() < host_geo_.size()) geo_->reserve(host_geo_.size());
    geo_->from_host(std::span<const StageGeometry>{host_geo_});  // a container write, ordered after the last launch's read

    // 5. What the launch must come after: every stage's dependencies (as for one component) and the arrays it reads.
    std::vector<sycl::event> deps;
    const auto               add = [&](const std::vector<sycl::event>& v) { deps.insert(deps.end(), v.begin(), v.end()); };
    for (ComponentBase* s : st) {
        add(ComponentAccess::gather_deps(*s, rt));
        add(s->stage_args_deps());
    }
    add(geo_->read_deps());
    add(ptrs_->read_deps());
    if (ticket_mode) add(ticket_->read_deps());

    // 6. ONE launch, with the cached call sequence of the stages' functions in stage order.
    std::vector<JitId> defs;
    defs.reserve(n);
    for (const ComponentBase* s : st) defs.push_back(s->definition_id());
    sycl::AdaptiveCpp_jit::dynamic_function_config& cfg = group_config(defs);
    const StageGeometry*                            gp  = geo_->data();
    const void* const*                              ap  = ptrs_->data();

    // The physical launch: always rank 3, the groups laid out by the range_split strategy.
    std::array<std::uint64_t, 3> phys{1, 1, 1};
    switch (split) {
        case Split::flatten: phys = {1, 1, base}; break;
        case Split::pad:
            phys = {1, 1, 0};
            for (const SegmentInfo& si : infos) {
                phys[0] = std::max(phys[0], si.range[0]);
                phys[1] = std::max(phys[1], si.range[1]);
                phys[2] += si.range[2];
            }
            break;
        case Split::padded:
            phys = {nseg, 1, 1};
            for (const SegmentInfo& si : infos) {
                phys[1] = std::max(phys[1], si.range[1]);
                phys[2] = std::max(phys[2], si.range[2]);
            }
            break;
    }
    const sycl::nd_range<3> nd{sycl::range<3>{phys[0] * wg[0], phys[1] * wg[1], phys[2] * wg[2]}, wg};
    const sycl::event       ev = launch_group(rt.uq, dev, deps, cfg, nd, gp, ap, local_bytes);

    // 7. The launch's event: a read of the geometry, the pointer array and every argument array, and on every stage (its
    // inputs' reads and its outputs' writes), registered here, at submit.
    geo_->register_read(ev);
    ptrs_->register_read(ev);
    if (ticket_mode) ticket_->register_write(ev);  // the launch writes the ticket state: the next reset and launch come after it
    for (ComponentBase* s : st) s->stage_args_register_read(ev);
    last_ = {ev};
    for (ComponentBase* s : st) ComponentAccess::record_launch(*s, last_);
    return last_;
}

}  // namespace detail

// -----------------------------------------------------------------------------
// detail::KernelLaunch<KB> — the individual description of a kernel body
// -----------------------------------------------------------------------------

namespace detail {

/// A kernel-body description of one forward: the stage's argument struct (the HOST MEMBER the component copies into its
/// argument array) and the facts a group launcher reads (KernelParts). It submits nothing itself: a plain or group body
/// launches through the group launcher, whose stage function reads the copy.
template <class KB>
class KernelLaunch final : public LaunchDescription {
public:
    static constexpr int R = kernel_rank_v<KB>;
    using Scratch          = scratch_t<KB>;

    KernelLaunch(sycl::range<R> range, std::optional<sycl::range<R>> wg,
                 typename KB::Inputs in, typename KB::Outputs out, KB body, Scratch scratch)
        : args_{in, out, body, scratch} {
        args_.range       = pad3<R>(range);  // in the trailing dimensions
        args_.flat_extent = range.size();
        if constexpr (HasScratch<KB>) args_.scratch_bytes = layout_scratch(args_.scratch);
        parts_.rank        = R;
        parts_.group_body  = GroupBody<KB>;
        parts_.flat_extent = args_.flat_extent;
        parts_.local_bytes = args_.scratch_bytes;
        if (wg) parts_.wg = pad3<R>(*wg);
    }

    KernelLaunch(const KernelLaunch&)            = delete;
    KernelLaunch& operator=(const KernelLaunch&) = delete;

    bool is_custom_op() const override { return false; }
    bool is_host() const override { return false; }
    const KernelParts* kernel() const override { return &parts_; }

    /// The stage's local slice is its Scratch plus these bytes (rounded up to 16), for GroupItem::local<T>.
    void request_local_bytes(std::size_t bytes) override {
        parts_.local_bytes = args_.scratch_bytes + (bytes + 15) / 16 * 16;
    }

    std::vector<sycl::event> submit(sk::unordered_queue&, const sycl::device&, const std::vector<sycl::event>&) const override {
        throw std::logic_error("naga::cospan: a plain or group body launches through its group launcher, not through submit");
    }

    /// This forward's arguments for a group of work-group shape `wg`: the stage's group grid and count.
    const StageArgs<KB>& finalize(const sycl::range<3>& wg) {
        // ceil(range / wg) per dimension, for plain and group bodies alike; an empty stage has no groups in any dimension
        args_.group_range = sycl::range<3>{(args_.range[0] + wg[0] - 1) / wg[0], (args_.range[1] + wg[1] - 1) / wg[1],
                                           (args_.range[2] + wg[2] - 1) / wg[2]};
        if (args_.flat_extent == 0) args_.group_range = sycl::range<3>{0, 0, 0};
        return args_;
    }
    sycl::range<3> group_range() const { return args_.group_range; }

private:
    static_assert(std::is_trivially_copyable_v<StageArgs<KB>>);

    StageArgs<KB> args_;
    KernelParts   parts_;
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
// Launch descriptors and the synthesized body (the component form)
// -----------------------------------------------------------------------------

/// The launch descriptors of create_launch_config: each its own type, in any order. range_desc is required for plain and
/// group bodies; wg_desc is optional; scratch_desc is optional and only for a body with a Scratch. Custom-op and host bodies
/// take none.
template <int R>
struct RangeDesc {
    sycl::range<R> value;
};
template <int R>
struct WgDesc {
    sycl::range<R> value;
};
template <class S>
struct ScratchDesc {
    S value;
};

template <int R>
RangeDesc<R> range_desc(const sycl::range<R>& r) {
    return RangeDesc<R>{r};
}
template <int R>
WgDesc<R> wg_desc(const sycl::range<R>& r) {
    return WgDesc<R>{r};
}
template <class S>
ScratchDesc<S> scratch_desc(const S& s) {
    return ScratchDesc<S>{s};
}

namespace detail {

/// kind: 0 range, 1 work-group shape, 2 scratch; -1 not a descriptor.
template <class D>
struct desc_info {
    static constexpr int kind = -1;
    static constexpr int rank = 0;
};
template <int R>
struct desc_info<RangeDesc<R>> {
    static constexpr int kind = 0;
    static constexpr int rank = R;
};
template <int R>
struct desc_info<WgDesc<R>> {
    static constexpr int kind = 1;
    static constexpr int rank = R;
};
template <class S>
struct desc_info<ScratchDesc<S>> {
    static constexpr int kind = 2;
    static constexpr int rank = 0;
};

template <class KB, class D>
constexpr bool scratch_desc_type_ok() {
    if constexpr (desc_info<D>::kind != 2)
        return true;
    else
        return std::is_same_v<D, ScratchDesc<scratch_t<KB>>>;
}

/// Whether the descriptors `Ds...` are a valid launch for the body KB: a range (of the body's rank) for plain and group bodies,
/// at most one of each descriptor, a scratch_desc only for a body with Scratch (of its type). Only a Component calls
/// create_launch_config (custom-op and host components build their launch themselves).
template <class KB, class... Ds>
constexpr bool launch_descs_ok() {
    if constexpr (!((desc_info<Ds>::kind >= 0) && ...)) {
        return false;
    } else {
        constexpr int nr = ((desc_info<Ds>::kind == 0 ? 1 : 0) + ... + 0);
        constexpr int nw = ((desc_info<Ds>::kind == 1 ? 1 : 0) + ... + 0);
        constexpr int ns = ((desc_info<Ds>::kind == 2 ? 1 : 0) + ... + 0);
        if constexpr (PlainBody<KB> || GroupBody<KB>) {
            constexpr int  R        = kernel_rank_v<KB>;
            constexpr bool ranks_ok = ((desc_info<Ds>::kind == 2 || desc_info<Ds>::rank == R) && ...);
            constexpr bool types_ok = (scratch_desc_type_ok<KB, Ds>() && ...);
            constexpr bool scr_ok   = HasScratch<KB> ? (ns <= 1) : (ns == 0);
            return nr == 1 && nw <= 1 && ranks_ok && types_ok && scr_ok;
        } else {
            return false;
        }
    }
}
template <class KB, class... Ds>
inline constexpr bool launch_descs_ok_v = launch_descs_ok<KB, Ds...>();

template <class T, class... Ds>
inline constexpr bool has_desc_v = (std::is_same_v<T, Ds> || ...);

/// The call signature of X::compute_forward (a static member function): its first four parameters.
template <class F>
struct impl_call {
    static constexpr bool        valid = false;
    static constexpr std::size_t nargs = 0;
    using a0 = None; using a1 = None; using a2 = None; using a3 = None;
};
template <class R, class... A>
struct impl_call<R (*)(A...)> {
    static constexpr bool        valid = true;
    static constexpr std::size_t nargs = sizeof...(A);
    using a0 = typename arg_or<0, None, A...>::type;
    using a1 = typename arg_or<1, None, A...>::type;
    using a2 = typename arg_or<2, None, A...>::type;
    using a3 = typename arg_or<3, None, A...>::type;
};
template <class R, class... A>
struct impl_call<R (*)(A...) noexcept> : impl_call<R (*)(A...)> {};

template <class X, class = void>
struct impl_traits : impl_call<void> {};
template <class X>
struct impl_traits<X, std::void_t<decltype(&X::compute_forward)>> : impl_call<decltype(&X::compute_forward)> {};

/// The body kind of X (the numbering of first_traits): 0 plain, 1 group, 2 custom op, 3 host. A host body is the one with no
/// tag: compute_forward(const Inputs&, const Outputs&), two parameters.
template <class X>
inline constexpr int body_kind_v = impl_traits<X>::nargs == 2 ? 3 : first_traits<typename impl_traits<X>::a0>::kind;

/// The Inputs and Outputs a synthesized body carries (a copy of the component's `inputs` and `outputs`).
template <class X>
struct SynthData {
    using Inputs  = typename X::Inputs;
    using Outputs = typename X::Outputs;

    SynthData(const Inputs& i, const Outputs& o) : in{i}, out{o} {}

    Inputs  in;
    Outputs out;
};

/// The kernel body of a component X: ONE distinct type per component, trivially copyable. Its call operator has the FIXED
/// signature the KernelBody concepts read (first argument, Inputs, Outputs[, Scratch]) and forwards to X::compute_forward; the
/// signature is chosen by specialization on the kind (the first parameter of compute_forward; none for a host body) and on
/// Scratch. A host body keeps the `Host` tag HERE, in the operator, and drops it for the user.
template <class X, int Kind, bool WithScratch>
struct SynthImpl : SynthData<X> {
    using SynthData<X>::SynthData;  // no call operator: not a body (the concepts fail, so the component's constructor is unavailable)
};

template <class X>
struct SynthImpl<X, 0, false> : SynthData<X> {
    using SynthData<X>::SynthData;
    void operator()(sycl::id<first_traits<typename impl_traits<X>::a0>::rank> i, typename X::Inputs in,
                    typename X::Outputs out) const {
        X::compute_forward(i, in, out);
    }
};

template <class X>
struct SynthImpl<X, 1, false> : SynthData<X> {
    using SynthData<X>::SynthData;
    void operator()(GroupItem<first_traits<typename impl_traits<X>::a0>::rank> it, typename X::Inputs in,
                    typename X::Outputs out) const {
        X::compute_forward(it, in, out);
    }
};

template <class X>
struct SynthImpl<X, 1, true> : SynthData<X> {
    using SynthData<X>::SynthData;
    using Scratch = typename X::Scratch;
    void operator()(GroupItem<first_traits<typename impl_traits<X>::a0>::rank> it, typename X::Inputs in,
                    typename X::Outputs out, Scratch scratch) const {
        X::compute_forward(it, in, out, scratch);
    }
};

template <class X>
struct SynthImpl<X, 2, false> : SynthData<X> {
    using SynthData<X>::SynthData;
    void operator()(CustomOp& op, typename X::Inputs in, typename X::Outputs out) const { X::compute_forward(op, in, out); }
};

template <class X>
struct SynthImpl<X, 3, false> : SynthData<X> {
    using SynthData<X>::SynthData;
    void operator()(Host, typename X::Inputs in, typename X::Outputs out) const { X::compute_forward(in, out); }
};

/// The name Component<X> uses. Its base (which reads X::compute_forward) is chosen when SynthBody<X> is instantiated, not when it
/// is named: Component<X> names it while X is still incomplete.
template <class X>
struct SynthBody : SynthImpl<X, body_kind_v<X>, (impl_traits<X>::nargs == 4)> {
    using Base = SynthImpl<X, body_kind_v<X>, (impl_traits<X>::nargs == 4)>;
    SynthBody(const typename X::Inputs& i, const typename X::Outputs& o) : Base{i, o} {}
};

}  // namespace detail

// -----------------------------------------------------------------------------
// Component<X>
// -----------------------------------------------------------------------------

/// The body concepts the component types constrain their default constructor with. X is complete where they are evaluated (the
/// constructor of the user's class), so SynthBody<X> can read X::compute_forward.
namespace detail {
template <class KB>
concept KernelStageBody = KernelBodyC<KB> && (PlainBody<KB> || GroupBody<KB>);
template <class KB>
concept CustomStageBody = KernelBodyC<KB> && CustomOpBody<KB>;
template <class KB>
concept HostStageBody = KernelBodyC<KB> && HostBody<KB>;

/// The typed layer common to the three component types (CRTP on the user's class X): the facts read off the body's signature
/// and the stage interface of a group launcher. Not the user path: derive Component, CustomComponent or HostComponent.
///
/// CRTP rule: the members of X (Inputs, Outputs, Scratch, compute_forward, inputs, outputs) are named only inside member-function
/// bodies and lazily instantiated templates here, never at class scope: X is incomplete while this class is instantiated.
template <class X>
class TypedComponent : public ComponentBase {
public:
    using body_type = SynthBody<X>;

    // ---- read off the body's signature, for a fused owner --------------------
    int  kernel_rank() const override { return kernel_rank_v<body_type>; }
    bool is_group_body() const override { return GroupBody<body_type>; }
    bool has_scratch() const override { return HasScratch<body_type>; }
    bool is_custom_op() const override { return CustomOpBody<body_type>; }
    bool is_host() const override { return HostBody<body_type>; }
    bool is_kernel_stage() const override { return PlainBody<body_type> || GroupBody<body_type>; }
    JitId definition_id() const override {
        if constexpr (CustomOpBody<body_type> || HostBody<body_type>)
            throw std::logic_error("naga::cospan: " + describe() + " is a custom-op or host body: it has no stage function");
        else
            return stage_definition_id<body_type>();  // a function-local static, one per body type
    }

    // ---- the stage interface (plain and group bodies) ------------------------------
    void stage_args_ensure(Runtime& rt) override {
        if constexpr (PlainBody<body_type> || GroupBody<body_type>) {
            if (!args_) {
                auto v = std::make_shared<ArgsVec>(rt.allocator_for<StageArgs<body_type>>(this->device_of(), sk::kind::device));
                v->reserve(1);  // capacity 1 from the start: the address the pointer array holds is final
                args_ = std::move(v);
            }
        } else {
            this->throw_no_stage();
        }
    }
    const void* stage_args_ptr() const override {
        if constexpr (PlainBody<body_type> || GroupBody<body_type>) {
            if (!args_) throw std::logic_error("naga::cospan: " + this->describe() + ": the argument array does not exist yet (precompute first)");
            return args()->data();
        } else {
            this->throw_no_stage();
        }
    }
    std::vector<sycl::event> stage_args_deps() const override {
        if constexpr (PlainBody<body_type> || GroupBody<body_type>) {
            return args()->read_deps();
        } else {
            this->throw_no_stage();
        }
    }
    void stage_args_register_read(const sycl::event& e) override {
        if constexpr (PlainBody<body_type> || GroupBody<body_type>) {
            args()->register_read(e);
        } else {
            this->throw_no_stage();
        }
    }
    sycl::range<3> stage_upload([[maybe_unused]] Runtime& rt, [[maybe_unused]] LaunchDescription& d,
                                [[maybe_unused]] const sycl::range<3>& wg) override {
        if constexpr (PlainBody<body_type> || GroupBody<body_type>) {
            auto&                     kl = dynamic_cast<KernelLaunch<body_type>&>(d);
            const StageArgs<body_type>& a  = kl.finalize(wg);
            args()->from_host(std::span<const StageArgs<body_type>>{&a, 1});  // the host member into the array: a container write
            return kl.group_range();
        } else {
            this->throw_no_stage();
        }
    }

protected:
    sk::kind default_storage_kind() const override { return HostBody<body_type> ? sk::kind::host : sk::kind::device; }

private:
    /// The component's argument array (plain and group bodies only), type-erased: this class is instantiated while X is still
    /// incomplete, so its type cannot be named in a member declaration.
    using ArgsVec = sk::usm_vector<StageArgs<SynthBody<X>>>;
    std::shared_ptr<void> args_;
    ArgsVec*              args() const { return static_cast<ArgsVec*>(args_.get()); }
};

}  // namespace detail

/// A KERNEL component, by CRTP: `class Axpb : public cs::Component<Axpb>`. The component declares nested `Inputs` and `Outputs`
/// structs of ports, a STATIC `compute_forward` (its first parameter fixes the kind and rank: sycl::id<R> plain, GroupItem<R>
/// group; a group body with a nested `Scratch` takes it as the 4th) and PUBLIC members `inputs` and `outputs`. The hooks are
/// `declare()` (ports), `prepare_forward()` (per-forward shapes) and `define_forward()`, which returns
/// create_launch_config(descriptors...) made from the members. ONE KERNEL BODY PER COMPONENT, by type. A custom-op or host
/// body is not a Component: derive CustomComponent or HostComponent.
template <class X>
class Component : public detail::TypedComponent<X> {
public:
    /// The constructor is a template over X (`Self`) so that its constraint is checked when a derived class constructs it, with X
    /// complete, and never while Component<X> is being instantiated as the base of the still-incomplete X.
    template <class Self = X>
        requires detail::KernelStageBody<detail::SynthBody<Self>>
    Component() {}

    /// USER, every forward: state the launch with create_launch_config(descriptors...). Returns the description the component
    /// submits.
    virtual std::unique_ptr<LaunchDescription> define_forward() = 0;

protected:
    std::unique_ptr<LaunchDescription> describe_launch() override { return define_forward(); }

    /// The launch of this component, from its `inputs` and `outputs` members and the descriptors, in any order:
    /// range_desc(range) (required), wg_desc(range) (optional), scratch_desc(value) (optional, a body with Scratch only).
    template <class... Ds>
        requires(detail::launch_descs_ok_v<detail::SynthBody<X>, Ds...>)
    std::unique_ptr<LaunchDescription> create_launch_config(Ds... ds) {
        using KB   = detail::SynthBody<X>;
        X&   self  = static_cast<X&>(*this);
        KB   body{self.inputs, self.outputs};
        constexpr int LR = launch_rank_v<KB>;
        const std::tuple<Ds...> t{ds...};
        const sycl::range<LR>   range = std::get<RangeDesc<LR>>(t).value;
        std::optional<sycl::range<LR>> wg;
        if constexpr (detail::has_desc_v<WgDesc<LR>, Ds...>) wg = std::get<WgDesc<LR>>(t).value;
        detail::scratch_t<KB> scratch = detail::NoScratchValue<KB>();
        if constexpr (detail::has_desc_v<ScratchDesc<detail::scratch_t<KB>>, Ds...>)
            scratch = std::get<ScratchDesc<detail::scratch_t<KB>>>(t).value;
        return std::make_unique<detail::KernelLaunch<KB>>(range, wg, self.inputs, self.outputs, body, scratch);
    }
};

/// A CUSTOM-OP component: the body gets a CustomOp (the native interop handle) and enqueues native backend work.
/// `static void compute_forward(CustomOp&, const Inputs&, const Outputs&)`. There is no define_forward: the launch is built from
/// the members. Not fusible.
template <class X>
class CustomComponent : public detail::TypedComponent<X> {
public:
    template <class Self = X>
        requires detail::CustomStageBody<detail::SynthBody<Self>>
    CustomComponent() {}

protected:
    std::unique_ptr<LaunchDescription> describe_launch() override {
        using KB  = detail::SynthBody<X>;
        X&   self = static_cast<X&>(*this);
        KB   body{self.inputs, self.outputs};
        return std::make_unique<detail::CustomOpLaunch<KB>>(self.inputs, self.outputs, body);
    }
};

/// A HOST component: the body runs on the host as a host task, over host USM (the ports are host-kind). The type fixes the
/// kind, so the body takes no tag: `static void compute_forward(const Inputs&, const Outputs&)`. There is no define_forward.
/// Not fusible.
template <class X>
class HostComponent : public detail::TypedComponent<X> {
public:
    template <class Self = X>
        requires detail::HostStageBody<detail::SynthBody<Self>>
    HostComponent() {}

protected:
    std::unique_ptr<LaunchDescription> describe_launch() override {
        using KB  = detail::SynthBody<X>;
        X&   self = static_cast<X&>(*this);
        KB   body{self.inputs, self.outputs};
        return std::make_unique<detail::HostLaunch<KB>>(self.inputs, self.outputs, body);
    }
};

}  // namespace naga::cospan
