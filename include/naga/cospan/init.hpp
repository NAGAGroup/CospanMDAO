#pragma once

// =============================================================================
// naga/cospan/init.hpp — initial values of free input roots (r01 §4.4, D138)
// =============================================================================
//
// An initial value is METADATA on a free input root (an input root with no
// source), declared during setup() with ComponentBase::init_input(name, init)
// or System::init_input(path, init), or at any time after setup with
// Model::init_input(path, init) (D139), which wins over both. The highest level
// wins; the root's owner applies it when it allocates the root, and again when
// Model::init_input leaves one pending. There is no host write API: outputs are
// never initialized (§6.0).
//
//   cs::init::fill(v)           every element = v (converted to the root's dtype)
//   cs::init::zero()            every element = 0
//   cs::init::from_host(vec)    element i = vec[i]; vec.size() = the root's flat size
//   cs::init::custom<DT,R>(fn)  fn(queue&, Initializer<DT,R>): the function enqueues
//                               its own kernel on the queue
//   cs::init::custom_host<DT,R>(fn)  fn(HostViewMut<DT,R>&): host-side, committed on return
//                               (HostViewMut is only this callback's argument type)
//
// A shape (cs::init::Shape) is written as a list of extents, {n, m}, or as a ragged one:
//   cs::init::ragged(n, d1[, d2])  each di is one extent for every index or a vector of n extents, one per index
// An initializer that carries a shape supplies a free input's unspecified dimensions, and EVERY dimension of it is
// checked against the resolved root when it is applied (at the full range): a mismatch throws, naming both sides.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "naga/cospan/dtype.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/metadata.hpp"
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

/// Depth of a subsystem in the tree: the root System is 0, its children 1, and so on.
/// The declaring subsystem's depth decides which initial value wins (smallest depth).
inline std::size_t depth_of(const Subsystem& s) {
    if (s.path().empty()) return 0;
    std::size_t d = 1;
    for (const char c : s.path())
        if (c == '.') ++d;
    return d;
}

/// Zero a root's primary storage (ordered in the stream).
inline void zero_primary(IOEntry& root, sk::queue& q) {
    if (root.flat_size() == 0) return;
    const Placement pl = root.storage_placements().front();
    q.memset_in_order(pl.device, root.storage_on(pl), 0, root.flat_size() * dtype_size(root.dtype()));
}

inline std::string extents_str(const std::vector<std::size_t>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + std::to_string(v[i]);
    return out + "]";
}

/// The shape an init carries against the root's resolved shape: the rank and EVERY dimension (fixed, inherited,
/// value-driven, per-index element by element). A scalar extent c equals a per-index array of c's. Bare data
/// (from_host(vec)) describes only a rank-1 root; its size is checked when it is written.
inline void check_init_shape(const IOEntry& root, const InitShape& s) {
    if (s.flat && root.rank() != 1) return;
    const std::string name = root.full_path();
    if (s.rank != root.rank())
        throw std::invalid_argument("naga::cospan: init for '" + name + "' has rank " + std::to_string(s.rank) + " but '" + name +
                                    "' resolved rank " + std::to_string(root.rank()));
    const std::size_t n = root.extents()[0];
    for (int d = 0; d < root.rank(); ++d) {
        const auto        u         = static_cast<std::size_t>(d);
        const bool        init_per  = d >= 1 && s.per[u];
        const bool        root_per  = d >= 1 && root.dim_is_per(d);
        std::vector<std::size_t> init_v, root_v;
        if (d == 0) {
            init_v = {s.extents[0]};
            root_v = {n};
        } else {
            init_v = init_per ? s.per_index[u] : std::vector<std::size_t>(s.extents[0], s.extents[u]);
            root_v = root_per ? root.per_index_extents(d) : std::vector<std::size_t>(n, root.extents()[u]);
        }
        if (init_v == root_v) continue;
        const std::string init_str = init_per ? extents_str(init_v) : std::to_string(s.extents[u]);
        const std::string root_str = root_per ? extents_str(root_v) : std::to_string(root.extents()[u]);
        throw std::invalid_argument("naga::cospan: init for '" + name + "' has dimension " + std::to_string(d) + " = " + init_str +
                                    " but '" + name + "' resolved dimension " + std::to_string(d) + " (" + root.declared_rule(d) +
                                    ") = " + root_str);
    }
}

/// A free input root that its owner has just allocated from nothing: its init, else zeros;
/// then the other placements follow the primary. `full_range`: the root holds the whole run (not one chunk of a
/// ChunkedSystem's replay), so the init's shape is checked against it.
inline void apply_root_init(IOEntry& root, sk::queue& q, bool full_range = true) {
    if (full_range)
        if (const auto& s = root.init_spec(); s && s->shape()) check_init_shape(root, *s->shape());
    if (root.flat_size() == 0) return;
    root.bump_write();  // the init (fill, zero, from_host, custom) rewrites the whole root
    if (const auto& s = root.init_spec())
        s->apply(root, q);
    else
        zero_primary(root, q);
    if (root.storage_placements().size() > 1) root.sync_other_placements(q);
}

}  // namespace detail

namespace init {

namespace impl {
namespace detail = ::naga::cospan::detail;
/// A shape given as a list of extents (rank 1..3) as an initializer's shape.
inline InitShape init_shape_of(const std::vector<std::size_t>& extents) {
    if (extents.empty() || extents.size() > 3)
        throw std::invalid_argument("naga::cospan: an initializer's shape must have 1..3 extents");
    InitShape s;
    s.rank = static_cast<int>(extents.size());
    for (std::size_t d = 0; d < extents.size(); ++d) s.extents[d] = extents[d];
    return s;
}
}  // namespace impl

/// The shape of an initializer: a list of extents, {n, m} or a std::vector<std::size_t> (the spelling of the API), or a
/// ragged one made by ragged().
class Shape {
public:
    Shape(const std::vector<std::size_t>& extents) : s_{impl::init_shape_of(extents)} {}
    Shape(std::initializer_list<std::size_t> extents) : s_{impl::init_shape_of(std::vector<std::size_t>(extents))} {}
    static Shape of(InitShape s) {
        Shape out{std::vector<std::size_t>{1}};
        out.s_ = std::move(s);
        return out;
    }
    const InitShape& get() const noexcept { return s_; }

private:
    InitShape s_;
};

namespace impl {
/// One dimension (>= 1) given to ragged(): one extent for every index, or one per index.
struct RaggedDim {
    bool                     per = false;
    std::size_t              scalar = 0;
    std::vector<std::size_t> values;
    static RaggedDim one(std::size_t c) { return {false, c, {}}; }
    static RaggedDim each(const std::vector<std::size_t>& v) { return {true, 0, v}; }
};
inline Shape ragged_shape(std::size_t n, const std::vector<RaggedDim>& dims) {
    InitShape s;
    s.rank       = 1 + static_cast<int>(dims.size());
    s.extents[0] = n;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const std::size_t d = i + 1;
        if (dims[i].per) {
            if (dims[i].values.size() != n)
                throw std::invalid_argument("naga::cospan: init::ragged: dimension " + std::to_string(d) + " has " +
                                            std::to_string(dims[i].values.size()) + " extents but dimension 0 has " +
                                            std::to_string(n) + " indices");
            s.per[d]       = true;
            s.per_index[d] = dims[i].values;
            s.extents[d]   = 0;
        } else {
            s.extents[d] = dims[i].scalar;
        }
    }
    return Shape::of(std::move(s));
}
}  // namespace impl

/// A ragged shape [n, d1] or [n, d1, d2]: dimension 0 is n; each di is one extent (a std::size_t) or a vector of n extents,
/// one per index. A vector of the wrong length throws here; n = 0 is legal.
inline Shape ragged(std::size_t n, std::size_t d1) { return impl::ragged_shape(n, {impl::RaggedDim::one(d1)}); }
inline Shape ragged(std::size_t n, const std::vector<std::size_t>& d1) { return impl::ragged_shape(n, {impl::RaggedDim::each(d1)}); }
inline Shape ragged(std::size_t n, std::size_t d1, std::size_t d2) {
    return impl::ragged_shape(n, {impl::RaggedDim::one(d1), impl::RaggedDim::one(d2)});
}
inline Shape ragged(std::size_t n, std::size_t d1, const std::vector<std::size_t>& d2) {
    return impl::ragged_shape(n, {impl::RaggedDim::one(d1), impl::RaggedDim::each(d2)});
}
inline Shape ragged(std::size_t n, const std::vector<std::size_t>& d1, std::size_t d2) {
    return impl::ragged_shape(n, {impl::RaggedDim::each(d1), impl::RaggedDim::one(d2)});
}
inline Shape ragged(std::size_t n, const std::vector<std::size_t>& d1, const std::vector<std::size_t>& d2) {
    return impl::ragged_shape(n, {impl::RaggedDim::each(d1), impl::RaggedDim::each(d2)});
}

/// Every element = value (converted to the root's element type). The one-argument form carries no shape: it is
/// valid when every dimension of the root is specified.
namespace impl {
namespace detail = ::naga::cospan::detail;
template <class T>
InitSpec fill_impl(T value, std::optional<InitShape> shape) {
    return InitSpec{[value](detail::IOEntry& root, sk::queue& q) {
        const Placement pl = root.storage_placements().front();
        void*           p  = root.storage_on(pl);
        detail::dispatch_dtype(root.dtype(), [&](auto tag) {
            using E   = typename decltype(tag)::type;
            const E v = static_cast<E>(value);
            q.fill_in_order(pl.device, p, v, root.flat_size());
        });
    }, shape};
}
}  // namespace impl
template <class T>
InitSpec fill(T value) {
    return impl::fill_impl(value, std::nullopt);
}
/// fill(v, {n, m}): every element = v, and the shape is {n, m} (it supplies the root's unspecified dimensions).
template <class T>
InitSpec fill(T value, const Shape& shape) {
    return impl::fill_impl(value, std::optional<InitShape>{shape.get()});
}

/// Every element = 0 (no shape).
inline InitSpec zero() {
    return InitSpec{[](detail::IOEntry& root, sk::queue& q) { detail::zero_primary(root, q); }};
}
/// zero({n, m}): every element = 0, with that shape.
inline InitSpec zero(const Shape& shape) {
    return InitSpec{[](detail::IOEntry& root, sk::queue& q) { detail::zero_primary(root, q); }, shape.get()};
}

namespace impl {
namespace detail = ::naga::cospan::detail;
template <class T>
InitSpec from_host_impl(std::vector<T> values, std::optional<InitShape> shape) {
    return InitSpec{[values = std::move(values)](detail::IOEntry& root, sk::queue& q) {
        if (values.size() != root.flat_size())
            throw std::invalid_argument("naga::cospan: init::from_host has " + std::to_string(values.size()) +
                                        " values but '" + root.full_path() + "' has " +
                                        std::to_string(root.flat_size()) + " elements");
        detail::dispatch_dtype(root.dtype(), [&](auto tag) {
            using E = typename decltype(tag)::type;
            std::vector<E> converted(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) converted[i] = static_cast<E>(values[i]);
            detail::storage_of<E>(root).write_primary(q, std::span<const E>{converted.data(), converted.size()});
        });
    }, shape};
}
}  // namespace impl

/// Element i = values[i]; the size must equal the root's flat size when it is applied. The data's own shape is
/// rank 1 of its size: it supplies a rank-1 root's unspecified dimension.
template <class T>
InitSpec from_host(std::vector<T> values) {
    InitShape s;
    s.rank       = 1;
    s.extents[0] = values.size();
    s.flat       = true;
    return impl::from_host_impl(std::move(values), s);
}
/// from_host(data, shape): data with its shape, {n, m} or ragged(...). data.size() must equal the number of elements of
/// the shape: the product of the extents, or, for a ragged shape, the sum of the block sizes (the blocks end to end).
template <class T>
InitSpec from_host(std::vector<T> values, const Shape& shape) {
    const InitShape& s = shape.get();
    if (s.elements() != values.size())
        throw std::invalid_argument("naga::cospan: init::from_host has " + std::to_string(values.size()) + " values but its shape has " +
                                    std::to_string(s.elements()) + " elements" +
                                    (s.ragged() ? " (the sum of its block sizes)" : ""));
    return impl::from_host_impl(std::move(values), s);
}

/// A device-side initial value: `fn` gets the queue and the writable twin of the root and
/// enqueues its own kernel (or copy) on the queue. DT and Rank are the root's, spelled at the call:
///   cs::init::custom<DType::f32, 1>([](sk::queue& q, cs::Initializer<DType::f32, 1> out) { ... })
namespace impl {
namespace detail = ::naga::cospan::detail;
template <DType DT, int Rank>
InitSpec custom_impl(std::function<void(sk::queue&, Initializer<DT, Rank>)> fn, std::optional<InitShape> shape) {
    return InitSpec{[fn = std::move(fn)](detail::IOEntry& root, sk::queue& q) {
        auto* typed = dynamic_cast<detail::TypedEntry<DT, Rank>*>(&root);
        if (typed == nullptr)
            throw std::invalid_argument("naga::cospan: init::custom for '" + root.full_path() +
                                        "' was declared with the wrong dtype or rank");
        fn(q, typed->make_initializer());
    }, shape};
}
}  // namespace impl
template <DType DT, int Rank>
InitSpec custom(std::function<void(sk::queue&, Initializer<DT, Rank>)> fn) {
    return impl::custom_impl<DT, Rank>(std::move(fn), std::nullopt);
}
/// custom with a shape: custom<DT, R>(fn, {n, m}); the shape has Rank extents.
template <DType DT, int Rank>
InitSpec custom(std::function<void(sk::queue&, Initializer<DT, Rank>)> fn, const Shape& shape) {
    if (shape.get().rank != Rank)
        throw std::invalid_argument("naga::cospan: init::custom<..., " + std::to_string(Rank) + "> was given a shape of rank " +
                                    std::to_string(shape.get().rank));
    return impl::custom_impl<DT, Rank>(std::move(fn), std::optional<InitShape>{shape.get()});
}

/// A host-side initial value: the function receives a zero-initialised host copy of the root (its previous
/// contents are NOT read from the device); it is written back (a queued copy) on return.
namespace impl {
namespace detail = ::naga::cospan::detail;
template <DType DT, int Rank>
InitSpec custom_host_impl(std::function<void(HostViewMut<DT, Rank>&)> fn, std::optional<InitShape> shape) {
    return InitSpec{[fn = std::move(fn)](detail::IOEntry& root, sk::queue& q) {
        auto view = detail::HostViewAccess::view_mut<DT, Rank>(root, q);
        fn(view);
        view.commit();
    }, shape};
}
}  // namespace impl
template <DType DT, int Rank>
InitSpec custom_host(std::function<void(HostViewMut<DT, Rank>&)> fn) {
    return impl::custom_host_impl<DT, Rank>(std::move(fn), std::nullopt);
}
/// custom_host with a shape: custom_host<DT, R>(fn, {n, m}).
template <DType DT, int Rank>
InitSpec custom_host(std::function<void(HostViewMut<DT, Rank>&)> fn, const Shape& shape) {
    if (shape.get().rank != Rank)
        throw std::invalid_argument("naga::cospan: init::custom_host<..., " + std::to_string(Rank) + "> was given a shape of rank " +
                                    std::to_string(shape.get().rank));
    return impl::custom_host_impl<DT, Rank>(std::move(fn), std::optional<InitShape>{shape.get()});
}

}  // namespace init
}  // namespace naga::cospan
