#pragma once

// =============================================================================
// naga/cospan/shape_hook.hpp — shape hooks: a System's contract for the shape of an input or a tie alias
// =============================================================================
//
// docs/design-review-2026-10-03.md, section 10. A hook is a lightweight type-erased STEP (not a component) that a System
// declares with System::shape_hook(target, <one descriptor per dimension>). It sets the dimensions of its target that
// are not as_is; those dimensions are unspecified in the definition and owned by this hook alone. It runs on the main
// thread immediately before the first child that has an instance of the target, so every source's producer must be in an
// earlier slot (System checks this at setup).
//
// DESCRIPTORS
//   cs::dim::as_is()                    leave this dimension alone
//   cs::dim::from_shape(path, d)        dimension d of the shape of variable `path`
//   cs::dim::from_value(path, index)    the VALUE of element `index` (flat) of the integer variable `path`
//   cs::dim::custom(fn)                 fn(ShapeHookCtx&) -> size_t
//   cs::dim::fixed_at_run(v)            a value decided at setup: it just makes the dimension specified (v)
//   cs::ragged::counts_from(path)       a ragged target (rank 2): the per-index counts are the VALUES of the integer
//   cs::ragged::offsets_from(path)      variable `path` (column 0 of a rank-2 [N, k]), or the N + 1 block offsets in it.
//                                       One descriptor stands for both dimensions.
//
// THE WORK. from_value reads are batched: every one is queued as a small copy into a host buffer, and the host waits ONCE.
// counts_from / offsets_from build a DeviceColumn on the SOURCE's device with a parallel exclusive scan (device_scan.hpp;
// offsets_from copies and validates the offsets instead): only [total, negative / invalid flag] comes back to the host.

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/device_scan.hpp"
#include "naga/cospan/dispatch.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/variable.hpp"

namespace naga::cospan {

class ShapeHookCtx;

// -----------------------------------------------------------------------------
// Descriptors
// -----------------------------------------------------------------------------

namespace dim {
struct AsIs {};
struct FromShape { std::string path; std::size_t d = 0; };
struct FromValue { std::string path; std::size_t index = 0; };
struct Custom { std::function<std::size_t(ShapeHookCtx&)> fn; };
struct FixedAtRun { std::size_t v = 0; };

inline AsIs       as_is() { return {}; }
inline FromShape  from_shape(std::string path, std::size_t d) { return FromShape{std::move(path), d}; }
inline FromValue  from_value(std::string path, std::size_t index) { return FromValue{std::move(path), index}; }
inline Custom     custom(std::function<std::size_t(ShapeHookCtx&)> fn) { return Custom{std::move(fn)}; }
inline FixedAtRun fixed_at_run(std::size_t v) { return FixedAtRun{v}; }
}  // namespace dim

namespace ragged {
struct CountsFrom { std::string path; };
struct OffsetsFrom { std::string path; };

inline CountsFrom  counts_from(std::string path) { return CountsFrom{std::move(path)}; }
inline OffsetsFrom offsets_from(std::string path) { return OffsetsFrom{std::move(path)}; }
}  // namespace ragged

namespace detail {

/// One descriptor, type-erased. `src` is the source's instance, resolved by the System at setup.
struct HookDim {
    enum class Kind : uint8_t { as_is, from_shape, from_value, custom, fixed_at_run, counts_from, offsets_from };
    Kind                                      kind = Kind::as_is;
    std::string                               path;    ///< from_shape / from_value / counts_from / offsets_from
    std::size_t                               index = 0;  ///< from_shape: the dimension; from_value: the element; fixed_at_run: the value
    std::function<std::size_t(ShapeHookCtx&)> fn;      ///< custom
    std::weak_ptr<Instance>                   src;     ///< the named source, resolved at setup (not for custom)

    bool is_ragged() const noexcept { return kind == Kind::counts_from || kind == Kind::offsets_from; }
    bool names_source() const noexcept { return kind == Kind::from_shape || kind == Kind::from_value || is_ragged(); }
};

inline HookDim make_hook_dim(const dim::AsIs&) { return HookDim{}; }
inline HookDim make_hook_dim(const dim::FromShape& s) {
    HookDim h;
    h.kind  = HookDim::Kind::from_shape;
    h.path  = s.path;
    h.index = s.d;
    return h;
}
inline HookDim make_hook_dim(const dim::FromValue& s) {
    HookDim h;
    h.kind  = HookDim::Kind::from_value;
    h.path  = s.path;
    h.index = s.index;
    return h;
}
inline HookDim make_hook_dim(const dim::Custom& s) {
    HookDim h;
    h.kind = HookDim::Kind::custom;
    h.fn   = s.fn;
    return h;
}
inline HookDim make_hook_dim(const dim::FixedAtRun& s) {
    HookDim h;
    h.kind  = HookDim::Kind::fixed_at_run;
    h.index = s.v;
    return h;
}
inline HookDim make_hook_dim(const ragged::CountsFrom& s) {
    HookDim h;
    h.kind = HookDim::Kind::counts_from;
    h.path = s.path;
    return h;
}
inline HookDim make_hook_dim(const ragged::OffsetsFrom& s) {
    HookDim h;
    h.kind = HookDim::Kind::offsets_from;
    h.path = s.path;
    return h;
}

}  // namespace detail

/// What a descriptor type is: one of the above.
template <class D>
concept HookDescriptor = requires(const D& d) { { detail::make_hook_dim(d) } -> std::same_as<detail::HookDim>; };

// -----------------------------------------------------------------------------
// The data a hook reads
// -----------------------------------------------------------------------------

namespace detail {

using InstanceResolver = std::function<std::shared_ptr<Instance>(std::string_view)>;

/// The data a read of `m` goes through: a live device-kind data if there is one, else any live data.
inline std::shared_ptr<DeviceDataBase> source_data(VarMeta& m) {
    std::shared_ptr<DeviceDataBase> src;
    for (const auto& p : m.data) {
        const std::shared_ptr<DeviceDataBase> b = p.second.lock();
        if (!b) continue;
        if (!src || (b->key.kind == sk::kind::device && src->key.kind != sk::kind::device)) src = b;
    }
    return src;
}

/// Queues the copy of element `index` (flat) of the integer variable `m` to a small host buffer and returns what reads it
/// once the event (pushed on `evs`) is done. A stale or not yet materialized variable (a free input) is materialized first.
inline std::function<std::size_t()> queue_value_read(Runtime& rt, VarMeta& m, std::size_t index, std::vector<sycl::event>& evs) {
    const std::string path = m.display_path();
    if (!is_integer_value_dtype(m.dtype))
        throw std::invalid_argument("naga::cospan: shape hook: '" + path + "' drives a size by value and must have an integer dtype");
    if (m.is_stale()) m.materialize(rt);
    if (!m.shape.known) throw m.shape_unknown();
    if (index >= m.shape.elements())
        throw std::invalid_argument("naga::cospan: shape hook: '" + path + "' drives a size from element " + std::to_string(index) +
                                    " but has only " + std::to_string(m.shape.elements()) + " elements");
    const std::shared_ptr<DeviceDataBase> src = source_data(m);
    if (!src) throw std::logic_error("naga::cospan: shape hook: '" + path + "' has no data");
    std::function<std::size_t()> getter;
    dispatch_dtype(m.dtype, [&](auto tag) {
        using E   = typename decltype(tag)::type;
        auto host = std::make_shared<sk::usm_vector<E>>(1, rt.allocator_for<E>(src->key.dev, sk::kind::host));
        std::vector<sycl::event> deps = m.events.read_deps();
        for (const std::vector<sycl::event>& v : {src->events(), host->events()}) deps.insert(deps.end(), v.begin(), v.end());
        const sycl::event e = rt.q.copy(src->key.dev, deps, static_cast<const E*>(src->raw()) + index, host->data(), 1);
        src->register_event(e);  // a later writer of the source waits for the read
        host->register_event(e);
        m.events.record_read(e);
        evs.push_back(e);
        getter = [host, path, index]() -> std::size_t {
            const E v = host->data()[0];
            if constexpr (std::is_signed_v<E>) {
                if (v < E{0})
                    throw std::invalid_argument("naga::cospan: shape hook: element " + std::to_string(index) + " of '" + path + "' is " +
                                                std::to_string(v) + "; a size cannot be negative");
            }
            return static_cast<std::size_t>(v);
        };
    });
    return getter;
}

/// Builds (or rebuilds) `col` from the integer variable `m`, ON THE SOURCE'S DEVICE, in parallel (section 10).
///   counts mode:  m is rank 1 [N] or rank 2 [N, k] (column 0): extents[i] = max(value, 0), offs = their exclusive scan
///                 (detail::exclusive_scan_u64); a negative value sets the flag.
///   offsets mode: m is rank 1 with N + 1 entries: offs = the values, extents[i] = offs[i + 1] - offs[i]; the flag is set by
///                 a negative value, offs[0] != 0, or a decrease.
/// A tiny task then copies [total, flag] to the host buffer. Returns the event of that task (also col.ev). The column's
/// vectors are reused across forwards; `version` is bumped.
inline sycl::event build_column(Runtime& rt, VarMeta& m, DeviceColumn& col, bool offsets_mode) {
    const std::string path = m.display_path();
    if (!is_integer_value_dtype(m.dtype))
        throw std::invalid_argument("naga::cospan: shape hook: '" + path + "' drives per-index extents by value and must have an integer dtype");
    if (m.is_stale()) m.materialize(rt);
    if (!m.shape.known) throw m.shape_unknown();
    if (offsets_mode ? m.rank != 1 : m.rank > 2)
        throw std::invalid_argument("naga::cospan: shape hook: '" + path + "' drives per-index extents by value and must be rank 1" +
                                    (offsets_mode ? std::string{" (N + 1 offsets)"} : std::string{" [N] or rank 2 [N, k]"}));
    const std::size_t count_src = m.shape.extents[0];
    const std::size_t k         = m.rank == 2 ? m.shape.extents[1] : 1;
    if (offsets_mode && count_src == 0)
        throw std::invalid_argument("naga::cospan: shape hook: '" + path + "' holds no offsets (it needs N + 1 >= 1 entries)");
    if (!offsets_mode && m.rank == 2 && k == 0)
        throw std::invalid_argument("naga::cospan: shape hook: '" + path + "' has no column 0");
    const std::size_t n = offsets_mode ? count_src - 1 : count_src;

    const std::shared_ptr<DeviceDataBase> src = source_data(m);
    if (!src) throw std::logic_error("naga::cospan: shape hook: '" + path + "' has no data");
    const sycl::device dev = src->key.dev;
    // The vectors are reused while they live on the same device; their size follows n (a resize within capacity).
    if (!(col.key.dev == dev) || !col.extents || !col.offs || !col.host_total || !col.neg_flag) {
        col.key = DataKey{dev, sk::kind::device};
        col.extents.emplace(n, rt.allocator_for<std::uint64_t>(dev, sk::kind::device));
        col.offs.emplace(n + 1, rt.allocator_for<std::uint64_t>(dev, sk::kind::device));
        col.neg_flag.emplace(1, rt.allocator_for<std::uint64_t>(dev, sk::kind::device));
        col.host_total.emplace(2, rt.allocator_for<std::uint64_t>(dev, sk::kind::host));
    } else {
        if (col.extents->size() != n) col.extents->resize(n, std::span<const sycl::event>{col.extents->events()});
        if (col.offs->size() != n + 1) col.offs->resize(n + 1, std::span<const sycl::event>{col.offs->events()});
    }
    col.n = n;

    std::vector<sycl::event> deps = m.events.read_deps();
    for (const std::vector<sycl::event>& v :
         {src->events(), col.extents->events(), col.offs->events(), col.host_total->events(), col.neg_flag->events()})
        deps.insert(deps.end(), v.begin(), v.end());

    std::uint64_t* ex   = col.extents->data();
    std::uint64_t* of   = col.offs->data();
    std::uint64_t* flag = col.neg_flag->data();
    std::uint64_t* tot  = col.host_total->data();
    const std::size_t nn = n, kk = k, mm = count_src;

    sycl::event ev_read;   // the last kernel that reads the source
    sycl::event ev_scan;   // the last kernel that writes the offsets and extents
    const sycl::event ev_zero = rt.uq.single_task(dev, deps, [=]() { flag[0] = 0; });
    dispatch_dtype(m.dtype, [&](auto tag) {
        using E     = typename decltype(tag)::type;
        const E* in = static_cast<const E*>(src->raw());
        if (!offsets_mode) {
            // one parallel_for: extents[i] = max(data[i * k], 0), flag |= negative; then the parallel exclusive scan
            ev_read = ev_zero;
            if (nn != 0) {
                const std::vector<sycl::event> d1{ev_zero};
                ev_read = rt.uq.parallel_for(dev, d1, sycl::range<1>{nn}, [=](sycl::id<1> id) {
                    const std::size_t i = id[0];
                    const E           v = in[i * kk];
                    std::uint64_t     x = 0;
                    if constexpr (std::is_signed_v<E>) {
                        if (v < E{0}) {
                            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                f{flag[0]};
                            f.fetch_or(std::uint64_t{1});
                        } else {
                            x = static_cast<std::uint64_t>(v);
                        }
                    } else {
                        x = static_cast<std::uint64_t>(v);
                    }
                    ex[i] = x;
                });
            }
            const std::vector<sycl::event> d2{ev_read};
            ev_scan = exclusive_scan_u64(rt, dev, d2, ex, of, nn);
        } else {
            // kernel 1 over the N + 1 entries: offs[i] = value (a negative one, or offs[0] != 0, sets the flag);
            // kernel 2 over the N blocks: extents[i] = offs[i + 1] - offs[i] (a decrease sets the flag)
            const std::vector<sycl::event> d1{ev_zero};
            ev_read = rt.uq.parallel_for(dev, d1, sycl::range<1>{mm}, [=](sycl::id<1> id) {
                const std::size_t i = id[0];
                const E           v = in[i];
                std::uint64_t     x = 0;
                bool              bad = false;
                if constexpr (std::is_signed_v<E>) {
                    if (v < E{0}) bad = true;
                    else x = static_cast<std::uint64_t>(v);
                } else {
                    x = static_cast<std::uint64_t>(v);
                }
                if (i == 0 && x != 0) bad = true;
                of[i] = x;
                if (bad) {
                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        f{flag[0]};
                    f.fetch_or(std::uint64_t{1});
                }
            });
            ev_scan = ev_read;
            if (nn != 0) {
                const std::vector<sycl::event> d2{ev_read};
                ev_scan = rt.uq.parallel_for(dev, d2, sycl::range<1>{nn}, [=](sycl::id<1> id) {
                    const std::size_t i = id[0];
                    if (of[i + 1] < of[i]) {
                        ex[i] = 0;
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            f{flag[0]};
                        f.fetch_or(std::uint64_t{1});
                    } else {
                        ex[i] = of[i + 1] - of[i];
                    }
                });
            }
        }
    });
    // a tiny task copies [total, flag] to the host buffer
    const std::vector<sycl::event> dtot{ev_scan, ev_read};
    const sycl::event              e = rt.uq.single_task(dev, dtot, [=]() {
        tot[0] = of[nn];
        tot[1] = flag[0];
    });
    src->register_event(ev_read);  // the first pass is the only read of the source
    col.extents->register_event(e);
    col.offs->register_event(e);
    col.neg_flag->register_event(e);
    col.host_total->register_event(e);
    m.events.record_read(ev_read);  // a later writer of the source waits until it has been read
    col.ev = e;
    ++col.version;
    return e;
}

// -----------------------------------------------------------------------------
// The context of a custom descriptor
// -----------------------------------------------------------------------------

}  // namespace detail

/// Given to a cs::dim::custom function: reads of any variable by path, in the namespace of the System that declared the
/// hook. Only the sources a hook NAMES (from_shape, from_value, counts_from, offsets_from) are checked to be produced
/// before it runs; whatever a custom function reads here is the author's responsibility.
class ShapeHookCtx {
public:
    ShapeHookCtx(detail::Runtime& rt, detail::InstanceResolver resolve, std::string target)
        : rt_{&rt}, resolve_{std::move(resolve)}, target_{std::move(target)} {}  // framework only

    /// The shape of the variable `path` (its producer has run). A free variable that is not materialized yet is
    /// materialized first (that is where its initializer's shape is applied).
    detail::ResolvedShape shape_of(std::string_view path) {
        const std::shared_ptr<detail::Instance> inst = resolve_(path);
        if (!inst) throw std::invalid_argument("naga::cospan: shape hook on '" + target_ + "': no variable '" + std::string{path} + "'");
        detail::VarMeta& m = *inst->meta;
        if (!m.shape.known && !m.has_producer()) m.materialize(*rt_);
        if (!m.shape.known)
            throw std::logic_error("naga::cospan: shape hook on '" + target_ + "': the shape of '" + std::string{path} + "' is not known yet");
        return m.shape;
    }

    /// The VALUE of element `index` (flat) of the integer variable `path`: reads it down from the device and waits.
    std::size_t value_of(std::string_view path, std::size_t index) {
        const std::shared_ptr<detail::Instance> inst = resolve_(path);
        if (!inst) throw std::invalid_argument("naga::cospan: shape hook on '" + target_ + "': no variable '" + std::string{path} + "'");
        std::vector<sycl::event> evs;
        const auto               get = detail::queue_value_read(*rt_, *inst->meta, index, evs);
        sycl::event::wait(evs);
        return get();
    }

private:
    detail::Runtime*          rt_;
    detail::InstanceResolver  resolve_;
    std::string               target_;
};

namespace detail {

/// A declared hook: the target, what to do per dimension, who owns the dimensions, and where it runs.
struct HookStep {
    std::weak_ptr<Instance>   target;
    std::vector<HookDim>      dims;     ///< one per dimension of the target, or ONE ragged descriptor for a ragged target
    std::uint32_t             owner = 0;
    std::string               path;     ///< the target's path in the declaring System's namespace, for messages
    std::size_t               run_slot = 0;  ///< the child the hook runs before: the first with an instance of the target's variable
    std::shared_ptr<DeviceColumn> column;    ///< a ragged hook's device counts and offsets, reused across forwards
    InstanceResolver          resolve;       ///< for the context of custom descriptors

    /// Computes the dimensions this hook owns and sets them on the target's variable (checked against its definition). All
    /// from_value copies are batched and waited for once; events of device work are appended to `events_out`.
    void run(Runtime& rt, std::vector<sycl::event>& events_out) {
        const std::shared_ptr<Instance> inst = target.lock();
        if (!inst) return;  // the target's child is gone: nothing to set
        VarMeta&      tm  = *inst->meta;
        const std::string who = "the shape hook on '" + path + "'";
        // REVIEW: section 10 does not say what a hook does when a named source's instance is gone (its child was deleted, e.g.
        // a do_once producer). Section 9's "a gone shape source keeps its last shape" is carried over: the hook does nothing
        // while the target's shape is known, and is the "does not exist right now" error when it never was.
        if (tm.shape.known)
            for (const HookDim& h : dims)
                if (h.names_source() && h.src.expired()) return;
        const auto source = [&](const HookDim& h) -> std::shared_ptr<VarMeta> {
            const std::shared_ptr<Instance> up = h.src.lock();
            if (!up) throw std::logic_error("naga::cospan: " + who + ": its source '" + h.path + "' does not exist right now");
            return up->meta;
        };

        ResolvedShape s = tm.shape;  // dimensions not owned by the hook are not touched (set_dims takes the mask)
        DimMask       mask{};
        std::vector<sycl::event> evs;
        std::vector<std::pair<std::size_t, std::function<std::size_t()>>> reads;  // from_value: dimension, its getter

        if (dims.size() == 1 && dims[0].is_ragged()) {
            const std::shared_ptr<VarMeta> m = source(dims[0]);
            if (!column) column = std::make_shared<DeviceColumn>();
            evs.push_back(build_column(rt, *m, *column, dims[0].kind == HookDim::Kind::offsets_from));
            mask[0] = mask[1] = true;
        } else {
            for (std::size_t d = 0; d < dims.size(); ++d) {
                const HookDim& h = dims[d];
                switch (h.kind) {
                    case HookDim::Kind::as_is:
                    case HookDim::Kind::fixed_at_run: break;  // fixed_at_run made the dimension specified at setup
                    case HookDim::Kind::from_shape: {
                        const std::shared_ptr<VarMeta> m = source(h);
                        if (!m->shape.known && !m->has_producer()) m->materialize(rt);
                        if (!m->shape.known)
                            throw std::logic_error("naga::cospan: " + who + ": the shape of '" + h.path + "' is not known yet");
                        if (h.index >= static_cast<std::size_t>(m->rank))
                            throw std::out_of_range("naga::cospan: " + who + ": '" + h.path + "' has rank " + std::to_string(m->rank) +
                                                    ", no dimension " + std::to_string(h.index));
                        if (h.index == 1 && m->rank == 2 && m->shape.per)
                            throw std::invalid_argument("naga::cospan: " + who + ": dimension 1 of '" + h.path +
                                                        "' is per-index and cannot become a dense dimension (use cs::ragged::counts_from)");
                        s.extents[d] = m->shape.extents[h.index];
                        mask[d]      = true;
                        break;
                    }
                    case HookDim::Kind::from_value: {
                        const std::shared_ptr<VarMeta> m = source(h);
                        reads.emplace_back(d, queue_value_read(rt, *m, h.index, evs));
                        mask[d] = true;
                        break;
                    }
                    case HookDim::Kind::custom: {
                        ShapeHookCtx ctx{rt, resolve, path};
                        s.extents[d] = h.fn(ctx);
                        mask[d]      = true;
                        break;
                    }
                    case HookDim::Kind::counts_from:
                    case HookDim::Kind::offsets_from:
                        throw std::invalid_argument("naga::cospan: " + who + ": a ragged descriptor is for a ragged target, as its only descriptor");
                }
            }
        }
        sycl::event::wait(evs);  // ONE wait for every read this hook queued
        for (const auto& [d, get] : reads) s.extents[d] = get();
        if (column && dims.size() == 1 && dims[0].is_ragged()) {
            column->total = column->host_total->data()[0];
            if (column->host_total->data()[1] != 0)
                throw std::invalid_argument("naga::cospan: " + who + ": '" + dims[0].path + "' holds " +
                                            (dims[0].kind == HookDim::Kind::offsets_from
                                                 ? std::string{"offsets that are negative, do not start at 0, or decrease"}
                                                 : std::string{"a negative value"}) +
                                            "; a size cannot be negative");
            s.per         = true;
            s.per_index.clear();
            s.dev_per     = column;
            s.dev_version = column->version;
            s.extents[0]  = column->n;
            s.extents[1]  = 0;
            s.total       = column->total;
        }
        events_out.insert(events_out.end(), evs.begin(), evs.end());
        tm.set_dims(rt, s, mask, SetterId{SetterId::Kind::hook, owner, who});
    }
};

}  // namespace detail
}  // namespace naga::cospan
