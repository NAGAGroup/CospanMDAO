#pragma once

// =============================================================================
// naga/cospan/component.hpp — ComponentBase and Component<KB>
// =============================================================================
//
// A component is the leaf that does work, and it SUBMITS ITSELF (docs/design-review-2026-10-03.md, sections 3 and
// 4). Its forward is three framework-owned steps that call user hooks:
//
//   precompute   (1) every distinct input variable is materialized (shape hooks of the enclosing System have run);
//                (2) the user's pre_forward() runs: an output is reshaped there with set_shape, within its definition
//                    (a tied output's shape belongs to its tie's System);
//                (3) every distinct output variable is materialized (an unknown dimension is an error);
//                (4) every member wrapper is pointed at its data.
//                Shapes (design review, section 10): a declaration says only specified or unspecified per dimension; every
//                run-time shape change is checked against the variable's definition (VarMeta::set_dims).
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
    /// Per-forward shape control: shape_of / set_shape are legal here only.
    virtual void pre_forward() {}
    /// At the end of postcompute, after the pushes to the variables' other placements.
    virtual void post_forward() {}

    // ---- declaration API (legal inside setup() only) ------------------------
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

    // ---- shapes in pre_forward (legal inside pre_forward() only) -------------
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

    // ---- initial values (legal inside setup() only) -----------------------
    /// Declare the initial value of one of this Component's OWN inputs, by name. Metadata on the input's variable
    /// (section 1); the last declaration wins (a parent's setup runs after its children's, section 2), it is ignored when
    /// the input ends up with a source, and it is an error to name an output.
    void init_input(std::string_view name, InitSpec init) {
        if (!declaring_)
            throw std::logic_error("naga::cospan: init_input is legal inside setup() only (" + describe() + ")");
        const std::shared_ptr<detail::Instance> inst = find_kind(name, IOKind::input);  // an output throws
        detail::declare_init(detail::WiringSite{describe()}, inst, std::move(init));
    }

    // ---- Subsystem verbs ---------------------------------------------------
    void run_setup() override {
        detail::FlagScope scope{declaring_};
        setup();
    }

    /// Section 2: each of this component's instances goes to `d`. The outer call wins.
    void set_device(const sycl::device& d) override {
        for (const auto& inst : order_) {
            inst->device = d;
            inst->refresh_container();  // the container of the new placement
        }
    }

    std::shared_ptr<detail::Instance> find_instance(std::string_view name) const override {
        const auto it = by_name_.find(std::string{name});
        return it == by_name_.end() ? nullptr : it->second;
    }
    void collect_instances(std::vector<std::shared_ptr<detail::Instance>>& out) const override {
        for (const auto& i : order_) out.push_back(i);
    }

    /// The last launch's event(s): what depends_on hands on. (Deleting a component waits on nothing: the containers
    /// it held free themselves after their own events.)
    std::vector<sycl::event> events() const override { return last_events_; }

    /// depends_on: these events are added to this component's submission for ONE forward.
    void add_dependencies(std::span<const sycl::event> deps) override {
        pending_deps_.insert(pending_deps_.end(), deps.begin(), deps.end());
    }

    // ---- the forward (section 3) ---------------------------------------------
    /// The precompute steps below; a kernel component that launches by itself (not as a stage of a fused group) then keeps
    /// its argument array and, at the first precompute, its launcher's pointer array.
    void run_precompute(detail::Runtime& rt) override {
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

        // (2) The user's pre_forward(): where an output is reshaped (set_shape) within its definition.
        {
            detail::FlagScope scope{in_pre_forward_};
            pre_forward();
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
    void run_compute(detail::Runtime& rt) override {
        if (is_kernel_stage()) {
            // A plain or group body launches through the one path: a group of one stage in one default segment.
            (void)solo_.launch(rt, std::vector<ComponentBase*>{this}, std::vector<detail::SegmentSpec>{detail::SegmentSpec{}}, describe());
            return;
        }
        // Custom-op and host bodies keep their own path.
        const std::unique_ptr<LaunchDescription> desc = compute_forward_impl();
        if (!desc) throw std::logic_error("naga::cospan: " + describe() + ": compute_forward returned no launch");
        const std::vector<sycl::event> ev = desc->submit(rt.uq, device_of(), gather_deps(rt));
        record_launch(ev);
    }

    /// Pushes each output variable this component wrote to the variable's OTHER live placements (host views included)
    /// by container assignment, which waits on the output's writes and registers the read, unless the variable's push is
    /// suppressed. A tied output is pushed by the tie's System (phase 6). Then the user's hook. (The launch's events were
    /// registered on the containers at submit, in record_launch.)
    void run_postcompute(detail::Runtime& rt) override {
        (void)rt;
        std::vector<detail::VarMeta*> seen;
        for (const auto& inst : order_) {
            if (inst->kind != IOKind::output) continue;
            detail::VarMeta* m = inst->meta.get();
            if (std::find(seen.begin(), seen.end(), m) != seen.end()) continue;
            seen.push_back(m);
            if (m->is_tied_output() || m->push_suppressed || !inst->data || last_events_.empty()) continue;
            const std::shared_ptr<detail::DeviceDataBase> own = inst->data;
            for (const auto& entry : m->data) {
                const std::shared_ptr<detail::DeviceDataBase> dst = entry.second.lock();
                if (!dst || dst == own || !dst->bound()) continue;  // a holder not materialized yet fills itself
                dst->copy_from(*own, std::span<const sycl::event>{});
            }
        }
        post_forward();
    }

    /// The first instance that is a ragged port (declared RaggedInput / RaggedOutput) or whose variable is ragged, else
    /// null. A fused component refuses such ports (section 4).
    std::shared_ptr<detail::Instance> first_ragged_port() const {
        for (const auto& inst : order_)
            if (ragged_ports_.contains(inst.get()) || inst->meta->is_ragged()) return inst;
        return nullptr;
    }

    // ---- what a fused owner needs to know, without knowing the KernelBody type --
    // Answered by Component<KB> from its body's signature. An owner reads
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

    // ---- the stage interface of a group launcher (kernel bodies only; implemented by Component<KB>) ----
    /// The component's one-element argument array on its device: created (capacity 1, so its address is final) the first
    /// time.
    virtual void stage_args_ensure(detail::Runtime&) { throw_no_stage(); }
    /// Its device address, as the type-erased pointer the group's pointer array holds.
    virtual const void* stage_args_ptr() const { throw_no_stage(); }
    /// What a launch reading the array must wait on (its write list), and the hand-over of that launch's event.
    virtual std::vector<sycl::event> stage_args_deps() const { throw_no_stage(); }
    virtual void                     stage_args_register_read(const sycl::event&) { throw_no_stage(); }
    /// Finishes this forward's description for the group's work-group shape `wg`, copies its arguments into the array
    /// (a container write) and returns the stage's group count.
    virtual std::size_t stage_upload(detail::Runtime&, LaunchDescription&, const sycl::range<3>&) { throw_no_stage(); }

protected:
    ComponentBase() = default;

    /// A host view of one of this component's OWN inputs, made with the runtime its precompute received: legal inside
    /// pre_forward() only. It is a host holder like any other (the host is just another device of the variable): the
    /// variable is pushed to it every forward while it lives, so keep it as a member (an std::optional, made on the first
    /// forward) and call sync() before reading. The typical use is to read values in pre_forward and set_shape an
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

    /// The launch of this forward (describe): implemented by Component<KB> from the user's compute_forward().
    virtual std::unique_ptr<LaunchDescription> compute_forward_impl() = 0;

    std::string describe() const { return path().empty() ? std::string{"component"} : "'" + path() + "'"; }

    /// What a submission of this component must come after: each input container's read_deps (its writes), each output
    /// container's write_deps (its reads, and the container's own writes such as a resize fill), and any depends_on events
    /// (which last one forward: they are cleared here). Outside writes never wait on each other.
    std::vector<sycl::event> gather_deps(detail::Runtime& rt) {
        (void)rt;
        std::vector<sycl::event> deps;
        const auto               add = [&](const std::vector<sycl::event>& v) { deps.insert(deps.end(), v.begin(), v.end()); };
        for (const auto& inst : order_) {
            if (!inst->data) continue;
            add(inst->kind == IOKind::input ? inst->data->read_deps() : inst->data->write_deps());
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
        register_instance(detail::make_input<DT, Rank>(std::move(name), this, dims, &m), &m, "declare_input", false);
    }
    template <DType DT, int Rank>
    void add_output(Output<DT, Rank>& m, std::string name,
                    const std::array<detail::DimRecord, Rank>& dims) {
        register_instance(detail::make_output<DT, Rank>(std::move(name), this, dims, &m), &m, "declare_output", false);
    }
    template <DType DT, int Rank>
    void add_ragged_input(RaggedInput<DT, Rank>& m, std::string name, const std::array<detail::DimRecord, Rank>& dims,
                          std::optional<std::vector<std::size_t>> counts) {
        static_assert(Rank == 2, "cospan: a ragged port is rank 2");
        register_instance(detail::make_ragged_input<DT, Rank>(std::move(name), this, dims, &m, std::move(counts)), &m,
                          "declare_input", true);
    }
    template <DType DT, int Rank>
    void add_ragged_output(RaggedOutput<DT, Rank>& m, std::string name, const std::array<detail::DimRecord, Rank>& dims,
                           std::optional<std::vector<std::size_t>> counts) {
        static_assert(Rank == 2, "cospan: a ragged port is rank 2");
        register_instance(detail::make_ragged_output<DT, Rank>(std::move(name), this, dims, &m, std::move(counts)), &m,
                          "declare_output", true);
    }

    // ---- shapes in pre_forward ---------------------------------------------
    /// The declared instance of a member, inside pre_forward only.
    detail::Instance& member_instance(const void* member, const char* what) const {
        if (!in_pre_forward_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside pre_forward() only (" + describe() + ")");
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
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside setup() only");
        const std::string key = inst->name;
        if (by_name_.contains(key))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + key + "' is declared twice");
        if (by_member_.contains(member))
            throw std::invalid_argument("naga::cospan: " + describe() + ": member wrapper of '" + key +
                                        "' is already declared");
        inst->storage_kind = default_storage_kind();
        inst->refresh_container();  // the container of its placement, empty until the first materialization
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

    bool                declaring_      = false;
    bool                in_pre_forward_ = false;
    detail::Runtime*    rt_             = nullptr;  ///< set during run_precompute only
};

namespace detail {

/// INTERNAL: what a fused owner (phase 7) calls to get a stage's complete launch without submitting it.
struct ComponentAccess {
    static std::unique_ptr<LaunchDescription> describe(ComponentBase& c) { return c.compute_forward_impl(); }
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
                                                      const std::vector<SegmentSpec>& segs, const std::string& who) {
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
    for (int d = rank; d < 3; ++d) wg[d] = 1;  // dimensions past the rank do not exist

    // 3. Every stage's arguments into its own array (a container write), and its group count.
    std::vector<std::size_t> counts(n);
    for (std::size_t i = 0; i < n; ++i) counts[i] = st[i]->stage_upload(rt, *ds[i], wg);

    // 4. The geometry: per segment the block (the largest count), the groups before it, the local slices.
    // The segment table after the n entries: the S block sizes, then the S boundary kinds.
    const std::size_t nseg  = segs.size();
    const std::size_t table = (2 * nseg * sizeof(std::uint64_t) + sizeof(StageGeometry) - 1) / sizeof(StageGeometry);
    host_geo_.assign(n + table, StageGeometry{});
    std::vector<std::uint64_t> blocks(2 * nseg, 0);
    bool                       ticket_mode = false;
    std::uint64_t              base = 0;
    std::size_t                local_bytes = 0;
    for (std::size_t s = 0; s < segs.size(); ++s) {
        const SegmentSpec& sp = segs[s];
        std::uint64_t      block = 0;
        for (std::size_t j = sp.first; j < sp.first + sp.count; ++j) block = std::max<std::uint64_t>(block, counts[j]);
        std::size_t seg_local = 0;
        for (std::size_t j = sp.first; j < sp.first + sp.count; ++j) {
            StageGeometry& g = host_geo_[j];
            g.group_count    = counts[j];
            g.group_base     = base;
            g.segment_groups = block;
            g.segment        = static_cast<std::uint32_t>(s);
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
        blocks[s]        = block;
        blocks[nseg + s] = static_cast<std::uint64_t>(sp.boundary);
        if (sp.boundary != Boundary::none) ticket_mode = true;
        base += block;
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
    std::memcpy(static_cast<void*>(&host_geo_[n]), blocks.data(), blocks.size() * sizeof(std::uint64_t));

    // Ticket mode: the ticket state (counter, per-segment completed counters, per-ticket flags) is sized for this launch and
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
    sycl::AdaptiveCpp_jit::dynamic_function_config& cfg = group_config(rank, defs);
    const StageGeometry*                            gp  = geo_->data();
    const void* const*                              ap  = ptrs_->data();
    sycl::event                                     ev;
    switch (rank) {
        case 1: ev = launch_group<1>(rt.uq, dev, deps, cfg, group_nd_range<1>(base, wg), gp, ap, local_bytes); break;
        case 2: ev = launch_group<2>(rt.uq, dev, deps, cfg, group_nd_range<2>(base, wg), gp, ap, local_bytes); break;
        default: ev = launch_group<3>(rt.uq, dev, deps, cfg, group_nd_range<3>(base, wg), gp, ap, local_bytes); break;
    }

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
        args_.range       = pad3<R>(range);
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
        const std::size_t wg_size = wg[0] * wg[1] * wg[2];
        if constexpr (GroupBody<KB>) {
            args_.group_range = sycl::range<3>{(args_.range[0] + wg[0] - 1) / wg[0], (args_.range[1] + wg[1] - 1) / wg[1],
                                               (args_.range[2] + wg[2] - 1) / wg[2]};
            group_count_      = args_.group_range[0] * args_.group_range[1] * args_.group_range[2];
        } else {
            group_count_      = (args_.flat_extent + wg_size - 1) / wg_size;
            args_.group_range = sycl::range<3>{group_count_, 1, 1};
        }
        if (args_.flat_extent == 0) group_count_ = 0;
        return args_;
    }
    std::size_t group_count() const { return group_count_; }

private:
    static_assert(std::is_trivially_copyable_v<StageArgs<KB>>);

    StageArgs<KB> args_;
    std::size_t   group_count_ = 0;
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
// Component<KB>
// -----------------------------------------------------------------------------

/// One KernelBody type per component: the one-kernel rule, by type.
template <KernelBodyC KB>
class Component : public ComponentBase {
public:
    using body_type = KB;

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
    bool is_kernel_stage() const override { return PlainBody<KB> || GroupBody<KB>; }
    detail::JitId definition_id() const override {
        if constexpr (CustomOpBody<KB> || HostBody<KB>)
            throw std::logic_error("naga::cospan: " + describe() +
                                   " is a custom-op or host body: it has no stage function");
        else
            return detail::stage_definition_id<KB>();  // a function-local static, one per body type
    }

    // ---- the stage interface (plain and group bodies) ------------------------------
    void stage_args_ensure(detail::Runtime& rt) override {
        if constexpr (PlainBody<KB> || GroupBody<KB>) {
            if (!args_) {
                args_.emplace(rt.allocator_for<detail::StageArgs<KB>>(this->device_of(), sk::kind::device));
                args_->reserve(1);  // capacity 1 from the start: the address the pointer array holds is final
            }
        } else {
            this->throw_no_stage();
        }
    }
    const void* stage_args_ptr() const override {
        if constexpr (PlainBody<KB> || GroupBody<KB>) {
            if (!args_) throw std::logic_error("naga::cospan: " + this->describe() + ": the argument array does not exist yet (precompute first)");
            return args_->data();
        } else {
            this->throw_no_stage();
        }
    }
    std::vector<sycl::event> stage_args_deps() const override {
        if constexpr (PlainBody<KB> || GroupBody<KB>) {
            return args_->read_deps();
        } else {
            this->throw_no_stage();
        }
    }
    void stage_args_register_read(const sycl::event& e) override {
        if constexpr (PlainBody<KB> || GroupBody<KB>) {
            args_->register_read(e);
        } else {
            this->throw_no_stage();
        }
    }
    std::size_t stage_upload([[maybe_unused]] detail::Runtime& rt, [[maybe_unused]] LaunchDescription& d,
                             [[maybe_unused]] const sycl::range<3>& wg) override {
        if constexpr (PlainBody<KB> || GroupBody<KB>) {
            auto&                        kl = dynamic_cast<detail::KernelLaunch<KB>&>(d);
            const detail::StageArgs<KB>& a  = kl.finalize(wg);
            args_->from_host(std::span<const detail::StageArgs<KB>>{&a, 1});  // the host member into the array: a container write
            return kl.group_count();
        } else {
            this->throw_no_stage();
        }
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
    /// The component's argument array (plain and group bodies only).
    using ArgsVec = sk::usm_vector<detail::StageArgs<KB>>;
    std::conditional_t<PlainBody<KB> || GroupBody<KB>, std::optional<ArgsVec>, std::monostate> args_;

    std::unique_ptr<LaunchDescription> make_launch(sycl::range<LR> range,
                                                   std::optional<sycl::range<LR>> wg,
                                                   typename KB::Inputs in, typename KB::Outputs out,
                                                   KB body, detail::scratch_t<KB> scratch) {
        return std::make_unique<detail::KernelLaunch<KB>>(range, wg, in, out, body, scratch);
    }
};

}  // namespace naga::cospan
