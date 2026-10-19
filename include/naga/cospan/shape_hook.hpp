#pragma once

// =============================================================================
// naga/cospan/shape_hook.hpp — shape hooks: a System's contract for the shape of an input or a tie alias
// =============================================================================
//
// internal/design-review-2026-10-03.md, sections 10 and 11.5. Each System::shape_hook(target, <one descriptor per dimension>)
// call creates ONE templated pseudo-component (ShapeHookStep<descriptor types...>) that has only a forward. It sets the
// dimensions of its target that are not as_is; those dimensions are unspecified in the definition and owned by this hook
// alone. The target is an input, an input or output tie alias, or an ordinary untied output (for which the hook runs before
// the producer, which then sets only the dimensions it owns: it passes cs::as_is for the hooked ones). It runs on the main
// thread every forward, immediately before the first child that has an instance of the target, found when the System's loop
// reaches it. There is no source-ordering check: a source that is not produced yet gives whatever is there, and a gone
// source keeps the last shape.
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

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
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
template <class D, class... Ts>
inline constexpr bool is_one_of_v = (std::is_same_v<D, Ts> || ...);
}  // namespace detail

/// What a descriptor type is: one of the above.
template <class D>
concept HookDescriptor =
    detail::is_one_of_v<D, dim::AsIs, dim::FromShape, dim::FromValue, dim::Custom, dim::FixedAtRun, ragged::CountsFrom, ragged::OffsetsFrom>;

namespace detail {

/// What a descriptor type is, as plain data: the facts the System needs when the hook is DECLARED (validating it, marking
/// the target's definition, resolving the sources it names, replaying it). The hook itself never interprets this list: the
/// step (ShapeHookStep below) evaluates each dimension by the descriptor's TYPE.
struct HookInfo {
    enum class Kind : uint8_t { as_is, from_shape, from_value, custom, fixed_at_run, counts_from, offsets_from };
    Kind        kind = Kind::as_is;
    std::string path;       ///< from_shape / from_value / counts_from / offsets_from
    std::size_t index = 0;  ///< from_shape: the dimension; from_value: the element; fixed_at_run: the value

    bool is_ragged() const noexcept { return kind == Kind::counts_from || kind == Kind::offsets_from; }
    bool names_source() const noexcept { return kind == Kind::from_shape || kind == Kind::from_value || is_ragged(); }
};

inline HookInfo hook_info(const dim::AsIs&) { return HookInfo{}; }
inline HookInfo hook_info(const dim::FromShape& s) { return HookInfo{HookInfo::Kind::from_shape, s.path, s.d}; }
inline HookInfo hook_info(const dim::FromValue& s) { return HookInfo{HookInfo::Kind::from_value, s.path, s.index}; }
inline HookInfo hook_info(const dim::Custom&) { return HookInfo{HookInfo::Kind::custom, {}, 0}; }
inline HookInfo hook_info(const dim::FixedAtRun& s) { return HookInfo{HookInfo::Kind::fixed_at_run, {}, s.v}; }
inline HookInfo hook_info(const ragged::CountsFrom& s) { return HookInfo{HookInfo::Kind::counts_from, s.path, 0}; }
inline HookInfo hook_info(const ragged::OffsetsFrom& s) { return HookInfo{HookInfo::Kind::offsets_from, s.path, 0}; }

}  // namespace detail

// -----------------------------------------------------------------------------
// The data a hook reads
// -----------------------------------------------------------------------------

namespace detail {

using InstanceResolver = std::function<std::shared_ptr<Instance>(std::string_view)>;

/// The data a read of `m` goes through: a live device-kind data if there is one, else any live data (one that holds
/// elements: a container no materialization has reached yet is empty).
inline std::shared_ptr<DeviceDataBase> source_data(VarMeta& m) {
    std::shared_ptr<DeviceDataBase> src;
    for (const auto& p : m.data) {
        const std::shared_ptr<DeviceDataBase> b = p.second.lock();
        if (!b || !b->bound()) continue;
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
        // The copy READS the source (it waits on the source's writes) and WRITES the private host buffer, right after
        // the buffer's own fill (so it waits on both of the buffer's lists).
        std::vector<sycl::event> deps = src->read_deps();
        for (const std::vector<sycl::event>& v : {host->read_deps(), host->write_deps()}) deps.insert(deps.end(), v.begin(), v.end());
        const sycl::event e = rt.q.copy(src->key.dev, deps, static_cast<const E*>(src->raw()) + index, host->data(), 1);
        src->register_read(e);  // a later writer of the source waits for the read
        host->register_write(e);
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
/// `target` is the variable the hook shapes: every resize of the column's containers, and the task that writes them, is
/// noted in the TARGET's resize-event slot (the target may be an output, so every component that writes it waits on them),
/// besides being in the containers' own write lists.
inline sycl::event build_column(Runtime& rt, VarMeta& m, VarMeta& target, DeviceColumn& col, bool offsets_mode) {
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
        // A resize that reallocates copies the old elements: its event is the container's own write (in its write list)
        // and goes into the target's slot.
        if (col.extents->size() != n) target.note_hook_resize(col.extents->resize(n));
        if (col.offs->size() != n + 1) target.note_hook_resize(col.offs->resize(n + 1));
    }
    col.n = n;

    // The tasks below READ the source and WRITE the column's containers (right after their own fills, and after the
    // previous rebuild's readers): they wait on the source's writes and on both lists of every column container.
    std::vector<sycl::event> deps = src->read_deps();
    for (const sk::usm_vector<std::uint64_t>* c : {&*col.extents, &*col.offs, &*col.host_total, &*col.neg_flag}) {
        for (const std::vector<sycl::event>& v : {c->read_deps(), c->write_deps()}) deps.insert(deps.end(), v.begin(), v.end());
    }

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
    src->register_read(ev_read);  // the first pass is the only read of the source; a later writer of it waits
    col.extents->register_write(e);
    col.offs->register_write(e);
    col.neg_flag->register_write(e);
    col.host_total->register_write(e);
    target.note_resize(e);  // the task that wrote the column (it follows every kernel above): a writer of the target waits on it
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

/// What the model metadata stores of a hook: only a forward (finalize wraps it in an Operation placed by the sources). (The concrete step is the template below.)
class HookBase {
public:
    virtual ~HookBase()                  = default;
    virtual void forward(Runtime& rt) = 0;
};

template <class D>
inline constexpr bool names_source_v = is_one_of_v<D, dim::FromShape, dim::FromValue, ragged::CountsFrom, ragged::OffsetsFrom>;

/// The pseudo-component of ONE shape_hook call: its template parameters are the descriptor types, one per dimension (or one
/// ragged descriptor), and forward() evaluates every dimension by compile-time dispatch on them. It computes the dimensions
/// its target's variable gives it, from the named sources, and sets them (checked against the definition: it owns them).
/// from_value copies are batched and waited for once; counts_from / offsets_from build a device column with the parallel scan
/// and only the total comes back to the host. It runs as an Operation of the model's tree walk, placed at finalize (directly
/// after the last producer of its sources, else just before the first holder of its target).
template <HookDescriptor... D>
class ShapeHookStep final : public HookBase {
    static_assert(sizeof...(D) >= 1, "cospan: a shape hook takes one descriptor per dimension");

public:
    static constexpr std::size_t N       = sizeof...(D);
    static constexpr bool        kRagged = N == 1 && (is_one_of_v<D, ragged::CountsFrom, ragged::OffsetsFrom> && ...);

    ShapeHookStep(std::tuple<D...> descriptors, std::string path, std::uint32_t owner, InstanceResolver resolve)
        : ds_{std::move(descriptors)}, path_{std::move(path)}, owner_{owner}, resolve_{std::move(resolve)} {}

    /// The System points the step at its target and at the sources the descriptors name (at declaration).
    void set_target(std::shared_ptr<Instance> t) { target_ = std::move(t); }
    void set_source(std::size_t i, std::shared_ptr<Instance> s) { src_[i] = std::move(s); }

    void forward(Runtime& rt) override {
        const std::shared_ptr<Instance> inst = target_.lock();
        if (!inst) return;  // the target's child is gone: nothing to set
        VarMeta&          tm  = *inst->meta;
        const std::string who = "the shape hook on '" + path_ + "'";
        // Per dimension: a dimension whose named source is gone is not set while the target's shape is known (it keeps its last
        // extent; the other dimensions still follow their sources). With the shape never known, a gone source throws (source()).

        ResolvedShape            s = tm.shape;  // dimensions the hook does not own are not touched (set_dims takes the mask)
        DimMask                  mask{};
        std::vector<sycl::event> evs;
        std::vector<std::pair<std::size_t, std::function<std::size_t()>>> reads;  // from_value: dimension, its getter

        if constexpr (kRagged) {
            using R = std::tuple_element_t<0, std::tuple<D...>>;
            if (tm.shape.known && gone(src_[0])) return;  // its one source is gone
            const std::shared_ptr<VarMeta> m = source(0, who, std::get<0>(ds_).path);
            if (!column_) column_ = std::make_shared<DeviceColumn>();
            evs.push_back(build_column(rt, *m, tm, *column_, std::is_same_v<R, ragged::OffsetsFrom>));
            // dimension 0 is the hook's, or specified (set_dims then checks the source's count against the declared N)
            mask[0] = mask[1] = true;
        } else {
            [&]<std::size_t... I>(std::index_sequence<I...>) { (eval_dim<I>(rt, who, tm, s, mask, evs, reads), ...); }(std::index_sequence_for<D...>{});
        }
        sycl::event::wait(evs);  // ONE wait for every read this hook queued
        for (const auto& [d, get] : reads) s.extents[d] = get();
        if constexpr (kRagged) {
            const auto& desc = std::get<0>(ds_);
            column_->total   = column_->host_total->data()[0];
            if (column_->host_total->data()[1] != 0)
                throw std::invalid_argument("naga::cospan: " + who + ": '" + desc.path + "' holds " +
                                            (std::is_same_v<std::tuple_element_t<0, std::tuple<D...>>, ragged::OffsetsFrom>
                                                 ? std::string{"offsets that are negative, do not start at 0, or decrease"}
                                                 : std::string{"a negative value"}) +
                                            "; a size cannot be negative");
            s.per         = true;
            s.per_index.clear();
            s.dev_per     = column_;
            s.dev_version = column_->version;
            s.extents[0]  = column_->n;
            s.extents[1]  = 0;
            s.total       = column_->total;
        }
        tm.set_dims(rt, s, mask, SetterId{SetterId::Kind::hook, owner_, who});
    }

private:
    /// A named source is gone when its instance expired, or (r04) when its variable was materialized and all of its data was
    /// freed since (its producer went out of scope and nothing else holds it): it can never change again.
    static bool gone(const std::weak_ptr<Instance>& w) {
        const std::shared_ptr<Instance> i = w.lock();
        return !i || (i->meta->materialized && !i->meta->has_live_data());
    }

    std::shared_ptr<VarMeta> source(std::size_t i, const std::string& who, const std::string& path) const {
        const std::shared_ptr<Instance> up = src_[i].lock();
        if (!up) throw std::logic_error("naga::cospan: " + who + ": its source '" + path + "' does not exist right now");
        return up->meta;
    }

    /// Dimension I, by the descriptor's type.
    template <std::size_t I>
    void eval_dim(Runtime& rt, const std::string& who, VarMeta& tm, ResolvedShape& s, DimMask& mask, std::vector<sycl::event>& evs,
                  std::vector<std::pair<std::size_t, std::function<std::size_t()>>>& reads) {
        using Dsc         = std::tuple_element_t<I, std::tuple<D...>>;
        const Dsc& d = std::get<I>(ds_);
        (void)tm, (void)d;
        if constexpr (std::is_same_v<Dsc, dim::AsIs> || std::is_same_v<Dsc, dim::FixedAtRun>) {
            (void)rt, (void)who, (void)s, (void)mask, (void)evs, (void)reads;  // fixed_at_run made the dimension specified at the call
        } else if constexpr (std::is_same_v<Dsc, dim::FromShape>) {
            if (tm.shape.known && gone(src_[I])) return;  // this dimension's source is gone: left as it is
            const std::shared_ptr<VarMeta> m = source(I, who, d.path);
            if (!m->shape.known && !m->has_producer()) m->materialize(rt);
            if (!m->shape.known) throw std::logic_error("naga::cospan: " + who + ": the shape of '" + d.path + "' is not known yet");
            if (d.d >= static_cast<std::size_t>(m->rank))
                throw std::out_of_range("naga::cospan: " + who + ": '" + d.path + "' has rank " + std::to_string(m->rank) + ", no dimension " +
                                        std::to_string(d.d));
            if (d.d == 1 && m->rank == 2 && m->shape.per)
                throw std::invalid_argument("naga::cospan: " + who + ": dimension 1 of '" + d.path +
                                            "' is per-index and cannot become a dense dimension (use cs::ragged::counts_from)");
            s.extents[I] = m->shape.extents[d.d];
            mask[I]      = true;
        } else if constexpr (std::is_same_v<Dsc, dim::FromValue>) {
            if (tm.shape.known && gone(src_[I])) return;  // this dimension's source is gone: left as it is
            const std::shared_ptr<VarMeta> m = source(I, who, d.path);
            reads.emplace_back(I, queue_value_read(rt, *m, d.index, evs));
            mask[I] = true;
        } else if constexpr (std::is_same_v<Dsc, dim::Custom>) {
            ShapeHookCtx ctx{rt, resolve_, path_};
            s.extents[I] = d.fn(ctx);
            mask[I]      = true;
        } else {
            throw std::invalid_argument("naga::cospan: " + who + ": a ragged descriptor is for a ragged target, as its only descriptor");
        }
    }

    std::tuple<D...>                     ds_;
    std::string                          path_;  ///< the target's path in the declaring System's namespace, for messages
    std::uint32_t                        owner_;
    InstanceResolver                     resolve_;  ///< for the context of custom descriptors
    std::weak_ptr<Instance>              target_;
    std::array<std::weak_ptr<Instance>, N> src_;  ///< the named sources, resolved by the System at declaration
    std::shared_ptr<DeviceColumn>        column_;   ///< a ragged hook's device counts and offsets, reused across forwards
};

}  // namespace detail
}  // namespace naga::cospan
