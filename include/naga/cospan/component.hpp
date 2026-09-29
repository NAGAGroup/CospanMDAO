#pragma once

// =============================================================================
// naga/cospan/component.hpp — ComponentBase and Component<KB> (r01 §6)
// =============================================================================
//
// ComponentBase is the user surface: hooks, declarations, retrieval,
// initializers, and every runner body that touches no KB-typed state. It is
// the friend of detail::IOEntry (friendship is not inherited), so those bodies
// live here. Component<KB> adds the typed layer: launch-config staging,
// compile() and the compute-forward runner.

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

#include "naga/cospan/dtype.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/kernel.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

namespace detail {

/// Calls f(std::type_identity<T>{}) with T the C++ element type of `dt`.
template <class F>
void dispatch_dtype(DType dt, F&& f) {
    switch (value_dtype(dt)) {
        case DType::f32: f(std::type_identity<float>{}); break;
        case DType::f64: f(std::type_identity<double>{}); break;
        case DType::i32: f(std::type_identity<int32_t>{}); break;
        case DType::u32: f(std::type_identity<uint32_t>{}); break;
        case DType::i64: f(std::type_identity<int64_t>{}); break;
        case DType::u64: f(std::type_identity<uint64_t>{}); break;
        default: f(std::type_identity<uint8_t>{}); break;
    }
}

constexpr bool is_integer_dtype(DType dt) noexcept {
    const DType v = value_dtype(dt);
    return v != DType::f32 && v != DType::f64;
}

/// RAII flag: set for a scope, cleared on every exit.
struct FlagScope {
    explicit FlagScope(bool& f) : f_{f} { f_ = true; }
    ~FlagScope() { f_ = false; }
    FlagScope(const FlagScope&)            = delete;
    FlagScope& operator=(const FlagScope&) = delete;
    bool& f_;
};

}  // namespace detail

class ComponentBase : public Subsystem {
public:
    ComponentBase(const ComponentBase&)            = delete;
    ComponentBase& operator=(const ComponentBase&) = delete;
    ~ComponentBase() override = default;

    // ---- HOOKS (user-implemented) ------------------------------------------
    /// Declare I/O. Runs once, during the declare pass. Zero allocation.
    virtual void setup() = 0;
    /// Initialize own I/O; runs once after allocation and compilation.
    virtual void init(sk::queue& q, const sycl::device& dev) { (void)q; (void)dev; }
    /// Per-forward manual shape control, after automatic derivation.
    virtual void pre_forward() {}

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

    /// The array forms: from_source_shape<R>() and from_input_shape<R>(name).
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

    /// The pre_forward-only resize (same setter as automatic derivation).
    template <DType DT, int Rank>
    void reshape(Input<DT, Rank>& m, sycl::range<Rank> shape) {
        reshape_member(&m, shape);
    }
    template <DType DT, int Rank>
    void reshape(Output<DT, Rank>& m, sycl::range<Rank> shape) {
        reshape_member(&m, shape);
    }

    // ---- retrieval & enumeration ---------------------------------------------
    template <DType DT, int Rank>
    Input<DT, Rank>& get_input(std::string_view name) {
        auto* e = dynamic_cast<detail::InputEntry<DT, Rank>*>(&find_kind(name, IOKind::input));
        if (e == nullptr) throw wrong_type(name);
        return e->member();
    }
    template <DType DT, int Rank>
    Output<DT, Rank>& get_output(std::string_view name) {
        auto* e = dynamic_cast<detail::OutputEntry<DT, Rank>*>(&find_kind(name, IOKind::output));
        if (e == nullptr) throw wrong_type(name);
        return e->member();
    }

    using EntryRefs = std::vector<std::reference_wrapper<const detail::IOEntry>>;
    EntryRefs get_inputs() const { return entries_of(IOKind::input); }
    EntryRefs get_outputs() const { return entries_of(IOKind::output); }

    // ---- initializers (legal inside init() only; else throw) -----------------
    template <DType DT, int Rank>
    Initializer<DT, Rank> get_input_initializer(std::string_view name) {
        return typed_initializer<DT, Rank>(name, IOKind::input, "get_input_initializer");
    }
    template <DType DT, int Rank>
    Initializer<DT, Rank> get_output_initializer(std::string_view name) {
        return typed_initializer<DT, Rank>(name, IOKind::output, "get_output_initializer");
    }
    template <DType DT, int Rank>
    HostInitializer<DT, Rank> get_input_host_initializer(std::string_view name) {
        require_init("get_input_host_initializer");
        return HostInitializer<DT, Rank>{find_kind(name, IOKind::input), queue_ref()};
    }
    template <DType DT, int Rank>
    HostInitializer<DT, Rank> get_output_host_initializer(std::string_view name) {
        require_init("get_output_host_initializer");
        return HostInitializer<DT, Rank>{find_kind(name, IOKind::output), queue_ref()};
    }

    template <class T>
    void fill_input(std::string_view name, T value) {
        fill_entry(find_kind(name, IOKind::input), value, "fill_input");
    }
    template <class T>
    void fill_output(std::string_view name, T value) {
        fill_entry(find_kind(name, IOKind::output), value, "fill_output");
    }
    void zero_input(std::string_view name) { zero_entry(find_kind(name, IOKind::input), "zero_input"); }
    void zero_output(std::string_view name) { zero_entry(find_kind(name, IOKind::output), "zero_output"); }

    template <class T>
    void set_input_from_host(std::string_view name, std::span<const T> values) {
        set_entry_from_host<T>(find_kind(name, IOKind::input), values, "set_input_from_host");
    }
    template <class T>
    void set_output_from_host(std::string_view name, std::span<const T> values) {
        set_entry_from_host<T>(find_kind(name, IOKind::output), values, "set_output_from_host");
    }

    // ---- Subsystem verbs whose bodies touch no KB-typed state -----------------
    void run_setup() override {
        detail::FlagScope scope{declaring_};
        setup();
    }

    std::unordered_map<sycl::device, std::size_t> get_exec_info(
        const sycl::device& parent_device) override {
        device_ = parent_device;
        std::size_t bytes = 0;
        for (detail::IOEntry* e : order_) {
            e->device_       = parent_device;
            e->storage_kind_ = default_storage_kind();
            const Placement own = e->placement();
            detail::IOEntry&        root = *e->root();
            if (e->is_root()) {
                e->set_primary_placement(own);
                bytes += e->static_bytes();
            } else if (e->kind() == IOKind::input) {
                // Host-kind placements cost no arena bytes; a device-kind one
                // costs the root's arena bytes once per device.
                if (!has_registered(root, own)) {
                    root.register_placement(own);
                    if (own.kind == sk::kind::device) bytes += root.static_bytes();
                }
            } else if (root.device_ && root.placement() != own) {
                throw std::logic_error("naga::cospan: " + describe() + ": output '" +
                                       std::string{e->name()} + "' is tied to '" +
                                       std::string{root.name()} +
                                       "', which sits on another placement (device, storage "
                                       "kind); producers cannot write one storage from two "
                                       "placements");
            }
        }
        return {{parent_device, bytes}};
    }

    void prepare_exec(sk::queue& queue, Substrates& substrates) override {
        queue_ = &queue;
        subs_  = &substrates;
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::input && !e->is_root() && has_source_dim(*e))
                e->update_shape();  // throws for an unconnected input with a FromSourceDim
        for (detail::IOEntry* e : order_)
            if (e->is_root()) allocate_if_known(*e);
        for (detail::IOEntry* e : order_) {
            if (e->kind() == IOKind::input && !e->is_root() && !e->is_dynamic() &&
                !e->root()->is_dynamic())
                e->validate_connection();
            point_if_stored(*e);
        }
    }

    void run_init(const ExecCtx& owner_ctx) override {
        detail::FlagScope scope{in_init_};
        init(*owner_ctx.queue, owner_ctx.device);
    }

    /// A Component's System pass is a no-op.
    void run_system_init() override {}

    void run_pre_compute(ExecCtx& owner_ctx) override {
        (void)owner_ctx;
        std::unordered_set<const detail::IOEntry*> done, active;
        // Inputs first: derivation sources are inputs only.
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::input) resolve(*e, done, active);
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::output) resolve(*e, done, active);
        {
            detail::FlagScope scope{in_pre_forward_};
            pre_forward();
        }
        for (detail::IOEntry* e : order_)
            if (e->is_root()) allocate_if_known(*e);
        for (detail::IOEntry* e : order_) {
            if (e->kind() == IOKind::input && !e->is_root() &&
                (e->is_dynamic() || e->root()->is_dynamic()))
                e->validate_connection();
            point_if_stored(*e);
        }
    }

    /// Refresh the other-placement copies of every output root this Component
    /// produces (routed copies: device<->host, cross-device). Ordered after
    /// the launch by the stream.
    void run_post_compute(ExecCtx& owner_ctx) override {
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::output && e->is_root() && e->storage_placements().size() > 1 &&
                e->has_storage())
                e->sync_other_placements(*owner_ctx.queue);
    }

protected:
    ComponentBase() = default;

    /// The storage kind of this Component's I/O: device, or host for a host body.
    virtual sk::kind default_storage_kind() const { return sk::kind::device; }

    const sycl::device& device_ref() const {
        if (!device_) throw std::logic_error("naga::cospan: " + describe() + " has no device yet");
        return *device_;
    }
    sk::queue& queue_ref() const {
        if (queue_ == nullptr)
            throw std::logic_error("naga::cospan: " + describe() + " has no queue yet (prepare_exec)");
        return *queue_;
    }

    std::string describe() const { return path().empty() ? std::string{"component"} : "'" + path() + "'"; }

    std::optional<sycl::device> device_;
    sk::queue*                  queue_ = nullptr;
    Substrates*                 subs_  = nullptr;

private:
    /// System resolves paths into a Component's namespace and walks its
    /// entries (transient marking); nothing else reaches these.
    friend class System;

    detail::IOEntry* lookup_entry(std::string_view name) const {
        const auto it = by_name_.find(std::string{name});
        return it == by_name_.end() ? nullptr : it->second.get();
    }

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
        register_entry(std::make_unique<detail::InputEntry<DT, Rank>>(name, dims, &m), &m, "declare_input");
    }
    template <DType DT, int Rank>
    void add_output(Output<DT, Rank>& m, std::string name,
                    const std::array<detail::DimRecord, Rank>& dims) {
        register_entry(std::make_unique<detail::OutputEntry<DT, Rank>>(name, dims, &m), &m, "declare_output");
    }

    void register_entry(std::unique_ptr<detail::IOEntry> e, const void* member, const char* what) {
        if (!declaring_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside setup() only");
        const std::string key{e->name()};
        if (by_name_.contains(key))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + key + "' is declared twice");
        if (by_member_.contains(member))
            throw std::invalid_argument("naga::cospan: " + describe() + ": member wrapper of '" + key +
                                        "' is already declared");
        e->declarer_ = this;  // System's transient marking walks declarer -> owner (§4.4)
        order_.push_back(e.get());
        by_member_.emplace(member, e.get());
        by_name_.emplace(key, std::move(e));
    }

    template <DType DT, int Rank>
    Initializer<DT, Rank> typed_initializer(std::string_view name, IOKind k, const char* what) {
        require_init(what);
        auto* e = dynamic_cast<detail::TypedEntry<DT, Rank>*>(&find_kind(name, k));
        if (e == nullptr) throw wrong_type(name);
        return e->make_initializer();
    }

    // ---- lookup -----------------------------------------------------------
    detail::IOEntry& find_kind(std::string_view name, IOKind k) {
        const auto it = by_name_.find(std::string{name});
        if (it == by_name_.end())
            throw std::out_of_range("naga::cospan: " + describe() + " declares no '" + std::string{name} + "'");
        if (it->second->kind() != k)
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{name} +
                                        "' is an " + (k == IOKind::input ? "output" : "input"));
        return *it->second;
    }

    std::invalid_argument wrong_type(std::string_view name) const {
        return std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{name} +
                                     "' was asked for with the wrong dtype or rank");
    }

    EntryRefs entries_of(IOKind k) const {
        EntryRefs out;
        for (const detail::IOEntry* e : order_)
            if (e->kind() == k) out.emplace_back(*e);
        return out;
    }

    // ---- phase guards ------------------------------------------------------
    void require_init(const char* what) const {
        if (!in_init_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside init() only");
    }

    template <class W>
    void reshape_member(const W* member, sycl::range<W::rank> shape) {
        if (!in_pre_forward_)
            throw std::logic_error("naga::cospan: reshape is legal inside pre_forward() only");
        const auto it = by_member_.find(member);
        if (it == by_member_.end())
            throw std::invalid_argument("naga::cospan: " + describe() + ": reshape of a member that was never declared");
        std::array<std::size_t, 3> ext{1, 1, 1};
        for (int d = 0; d < W::rank; ++d) ext[static_cast<std::size_t>(d)] = shape[d];
        it->second->reshape(sycl::range<3>{ext[0], ext[1], ext[2]});
    }

    // ---- storage helpers ---------------------------------------------------
    static bool has_registered(const detail::IOEntry& root, const Placement& p) {
        for (const Placement& e : root.storage_placements())
            if (e == p) return true;
        return false;
    }

    static bool has_source_dim(const detail::IOEntry& e) {
        for (int d = 0; d < e.rank(); ++d)
            if (e.dim(d).how == detail::DimRecord::How::from_source_dim) return true;
        return false;
    }

    /// Allocate a root whose shape is known (every extent > 0) and that is not
    /// allocated yet: fixed roots at prepare_exec, the rest at first need.
    void allocate_if_known(detail::IOEntry& root) {
        if (root.has_storage() || root.flat_size() == 0) return;
        root.allocate(queue_ref(), *subs_);
    }

    void point_if_stored(detail::IOEntry& e) {
        if (e.root()->has_storage()) e.point_wrapper();
    }

    // ---- shape derivation (r01 §6.5 step 1) ----------------------------------
    detail::IOEntry& sibling_input(const detail::IOEntry& of, const std::string& name) {
        const auto it = by_name_.find(name);
        if (it == by_name_.end() || it->second->kind() != IOKind::input)
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{of.name()} +
                                        "' derives its shape from '" + name + "', which is not an input of this component");
        return *it->second;
    }

    /// Resolve `e`'s dimensions: its sources first (recursively), then apply.
    void resolve(detail::IOEntry& e, std::unordered_set<const detail::IOEntry*>& done,
                 std::unordered_set<const detail::IOEntry*>& active) {
        if (done.contains(&e)) return;
        if (!active.insert(&e).second)
            throw std::logic_error("naga::cospan: " + describe() + ": shape derivation cycle through '" +
                                   std::string{e.name()} + "'");
        for (int d = 0; d < e.rank(); ++d) {
            detail::DimRecord& r = e.dims_[static_cast<std::size_t>(d)];
            if (r.how == detail::DimRecord::How::from_input_dim) {
                detail::IOEntry& src = sibling_input(e, r.input);
                resolve(src, done, active);
                if (r.dim >= src.rank())
                    throw std::out_of_range("naga::cospan: '" + std::string{e.name()} +
                                            "' derives from a dimension past the rank of '" + r.input + "'");
                r.extent = src.extents()[r.dim];
            } else if (r.how == detail::DimRecord::How::from_input) {
                detail::IOEntry& src = sibling_input(e, r.input);
                resolve(src, done, active);
                r.extent = read_value(src, r.value_index);
            }
        }
        e.update_shape();
        active.erase(&e);
        done.insert(&e);
    }

    /// The VALUE of element `index` of an integer-dtype input, on the host.
    std::size_t read_value(detail::IOEntry& src, std::size_t index) {
        if (!detail::is_integer_dtype(src.dtype()))
            throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                        "' drives a shape by value and must have an integer dtype");
        allocate_if_known(*src.root());
        if (!src.root()->has_storage())
            throw std::logic_error("naga::cospan: '" + std::string{src.name()} +
                                   "' drives a shape by value but has no storage yet");
        std::size_t out = 0;
        detail::dispatch_dtype(src.dtype(), [&](auto tag) {
            using T   = typename decltype(tag)::type;
            auto snap = detail::take_snapshot<T>(src, queue_ref());
            if (index >= snap.data.size())
                throw std::out_of_range("naga::cospan: '" + std::string{src.name()} + "' has no element " +
                                        std::to_string(index));
            const T v = snap.data[index];
            if constexpr (std::is_signed_v<T>)
                if (v < 0)
                    throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                                "' holds a negative extent");
            out = static_cast<std::size_t>(v);
        });
        return out;
    }

    // ---- initializer helpers -------------------------------------------------
    /// Runs op(root, primary device, primary pointer) on the root's primary
    /// storage, then refreshes the other devices. A no-op on an unallocated
    /// EMPTY root (a shape still to be resolved).
    template <class Op>
    void write_root(detail::IOEntry& e, const char* what, Op&& op) {
        require_init(what);
        detail::IOEntry& root = *e.root();
        if (!root.has_storage()) {
            if (root.flat_size() == 0) return;
            throw std::logic_error("naga::cospan: " + describe() + ": '" + std::string{e.name()} +
                                   "' has no storage to initialize");
        }
        const Placement pl = root.storage_placements().front();
        op(root, pl.device, root.storage_on(pl));
        root.sync_other_placements(queue_ref());
    }

    template <class T>
    void fill_entry(detail::IOEntry& e, T value, const char* what) {
        write_root(e, what, [&](detail::IOEntry& root, const sycl::device& d, void* p) {
            detail::dispatch_dtype(root.dtype(), [&](auto tag) {
                using E = typename decltype(tag)::type;
                const E v = static_cast<E>(value);
                queue_ref().fill_in_order(d, p, v, root.flat_size());
            });
        });
    }

    void zero_entry(detail::IOEntry& e, const char* what) {
        write_root(e, what, [&](detail::IOEntry& root, const sycl::device& d, void* p) {
            queue_ref().memset_in_order(d, p, 0, root.flat_size() * dtype_size(root.dtype()));
        });
    }

    template <class T>
    void set_entry_from_host(detail::IOEntry& e, std::span<const T> values, const char* what) {
        require_init(what);
        detail::write_back<T>(e, queue_ref(), values);
    }

    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>> by_name_;
    std::unordered_map<const void*, detail::IOEntry*>                 by_member_;
    std::vector<detail::IOEntry*>                                     order_;  // declaration order

    bool declaring_       = false;
    bool in_init_         = false;
    bool in_pre_forward_  = false;
};

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
/// is no staging allocation (fused waves stage through device buffers instead).
template <class KB>
class KernelLaunch final : public LaunchDescription {
public:
    static constexpr int R = kernel_rank_v<KB>;
    using Scratch          = scratch_t<KB>;
    using Config           = sycl::AdaptiveCpp_jit::dynamic_function_config;

    KernelLaunch(Config* cfg, sycl::range<R> range, std::optional<sycl::range<R>> wg,
                 typename KB::Inputs in, typename KB::Outputs out, KB body, Scratch scratch)
        : cfg_{cfg}, payload_{StageRecord{}, in, out, BodyPack{body, scratch}} {
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

    void submit(SubmitTarget& t) const override {
        if (payload_.rec.flat_extent == 0) return;  // nothing to run for an empty range
        launch_individual<R>(t, *cfg_, wave_nd_range<R>(payload_.rec.group_count, wg3_), payload_,
                             scratch_bytes_);
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

    Config*        cfg_;
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

    void submit(SubmitTarget& t) const override {
        const typename KB::Inputs  in   = in_;
        const typename KB::Outputs out  = out_;
        const KB                   body = body_;
        auto cgf = [=](sycl::handler& h) {
            h.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih) {
                CustomOp op{ih};
                body(op, in, out);
            });
        };
        if (t.group != nullptr)
            t.group->submit(t.device, cgf);
        else
            t.queue->submit(t.device, cgf);
    }

private:
    typename KB::Inputs  in_;
    typename KB::Outputs out_;
    KB                   body_;
};

/// A host launch (D3): the body runs on the host as an ordered host task.
/// Its Inputs/Outputs point at host-kind USM on the component's device.
/// Without a group the task joins the stream's kernel order
/// (async_host_in_order); under a group, group.async_host (a group has no
/// in-order variant; ordering against kernel tasks is the group's).
template <class KB>
class HostLaunch final : public LaunchDescription {
public:
    HostLaunch(typename KB::Inputs in, typename KB::Outputs out, KB body)
        : in_{in}, out_{out}, body_{body} {}

    bool is_custom_op() const override { return false; }
    bool is_host() const override { return true; }

    void submit(SubmitTarget& t) const override {
        const typename KB::Inputs  in   = in_;
        const typename KB::Outputs out  = out_;
        const KB                   body = body_;
        auto f = [=]() { body(Host{}, in, out); };
        if (t.group != nullptr)
            t.group->async_host(t.device, f);
        else
            t.queue->async_host_in_order(t.device, f);
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
    /// parameters. Returns the description the owner will submit.
    virtual std::unique_ptr<LaunchDescription> compute_forward() = 0;

    /// The individual dynamic_function_config: built once, here. A custom-op
    /// or host body has nothing to compile.
    void compile() override {
        if constexpr (CustomOpBody<KB> || HostBody<KB>) {
            // nothing: no kernel is JIT-specialised for these bodies
        } else {
            constexpr int R = kernel_rank_v<KB>;
            cfg_            = std::make_unique<Config>();
            if constexpr (R == 1)
                cfg_->define(&kb_entry_r1, &kb_fn<KB>);
            else if constexpr (R == 2)
                cfg_->define(&kb_entry_r2, &kb_fn<KB>);
            else
                cfg_->define(&kb_entry_r3, &kb_fn<KB>);
        }
    }

    std::unique_ptr<LaunchDescription> run_compute_forward() override { return compute_forward(); }

protected:
    static constexpr int LR = launch_rank_v<KB>;

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
    using Config = sycl::AdaptiveCpp_jit::dynamic_function_config;

    std::unique_ptr<LaunchDescription> make_launch(sycl::range<LR> range,
                                                   std::optional<sycl::range<LR>> wg,
                                                   typename KB::Inputs in, typename KB::Outputs out,
                                                   KB body, detail::scratch_t<KB> scratch) {
        if (!cfg_)
            throw std::logic_error("naga::cospan: " + describe() +
                                   ": create_launch_config before compile()");
        return std::make_unique<detail::KernelLaunch<KB>>(cfg_.get(), range, wg, in, out, body,
                                                          scratch);
    }

    std::unique_ptr<Config> cfg_;
};

}  // namespace naga::cospan
