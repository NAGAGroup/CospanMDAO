#pragma once

// =============================================================================
// naga/cospan/init.hpp — initializers: the initial value, and the value, of a free input
// =============================================================================
//
// An initializer is METADATA on a variable (docs/design-review-2026-10-03.md, sections 1, 5 and 6): it is
// declared during setup() with ComponentBase::init_input(name, init) or System::init_input(path, init), and
// replaced at run time by Model::update_input(path, init), which also marks the variable stale. Which
// declaration wins is simple: the LAST one (a parent's setup runs after its children's, so a tie's, promotion's or
// System's initializer replaces the members'; no conflict error).
// materialize (variable.hpp) runs the initializer when the variable is stale: on a variable that is not
// materialized, at its allocation; on a live one, in place. Outputs are never initialized.
//
// EVERY initializer RETURNS THE EVENT OF THE WORK IT ENQUEUED: materialize records it as the variable's
// write, and whatever reads the variable (the driving-value accessor of section 6, a consumer) waits on it.
//
//   cs::init::fill(v)           every element = v (converted to the variable's dtype)
//   cs::init::zero()            every element = 0
//   cs::init::from_host(vec)    element i = vec[i]; vec.size() = the variable's number of elements
//   cs::init::custom<DT,R>(fn)  fn(unordered_queue&, device, deps, Initializer<DT,R>) -> event: the function
//                               enqueues its own kernel (or copy) on the queue, makes it depend on `deps`, and
//                               returns the event of that work
//   cs::init::custom_host<DT,R>(fn)  fn(HostInit<DT,R>&): host-side; the buffer is copied to the device after it returns
//
// A shape (cs::init::Shape) is written as a list of extents, {n, m}, or as a ragged one:
//   cs::init::ragged(n, d1)        d1 is one extent for every index or a vector of n counts, one per index (rank 2 only)
// An initializer that carries a shape supplies a free input's unspecified dimensions, and EVERY dimension of it is
// checked against the resolved variable when it is applied: a mismatch throws, naming both sides.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/dispatch.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/init_spec.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/variable.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

namespace detail {
struct HostInitAccess;
}

/// HostInit<DT, Rank> — the zero-initialized host buffer handed to a cs::init::custom_host function: one element per
/// element of the variable. It is built by the framework only; the framework copies it to the device when the
/// function returns. Dense variables are addressed by shape() and operator()(i, j, ...) (row-major); a ragged
/// variable's buffer is its blocks end to end, addressed flat.
template <DType DT, int Rank>
class HostInit {
public:
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    HostInit(const HostInit&)            = delete;
    HostInit& operator=(const HostInit&) = delete;
    HostInit(HostInit&&) noexcept        = default;

    sycl::range<Rank>     shape() const { return shape_; }
    std::size_t           size() const { return host_.size(); }
    value_type*           data() { return host_.data(); }
    std::span<value_type> span() { return {host_.data(), host_.size()}; }

    value_type& operator[](std::size_t flat) { return host_[flat]; }
    value_type& operator()(std::size_t i)
        requires(Rank == 1)
    {
        return host_[i];
    }
    value_type& operator()(std::size_t i, std::size_t j)
        requires(Rank == 2)
    {
        return host_[i * shape_[1] + j];
    }
    value_type& operator()(std::size_t i, std::size_t j, std::size_t k)
        requires(Rank == 3)
    {
        return host_[(i * shape_[1] + j) * shape_[2] + k];
    }

private:
    explicit HostInit(const detail::ResolvedShape& s)
        : host_(s.elements(), value_type{}), shape_{detail::make_range<Rank>(s.extents)} {}

    std::vector<value_type> host_;
    sycl::range<Rank>       shape_;
    friend struct detail::HostInitAccess;
};

namespace detail {

/// INTERNAL: builds the HostInit of cs::init::custom_host.
struct HostInitAccess {
    template <DType DT, int Rank>
    static HostInit<DT, Rank> make(const ResolvedShape& s) {
        return HostInit<DT, Rank>{s};
    }
};

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

/// A RAGGED shape is rank 2 (section 10): a dense leading dimension n and one per-index dimension.
/// ragged(n, d1): n indices of d1 extents each (a rectangle); ragged(n, counts): one count per index, counts.size() == n
/// (a vector of the wrong length throws here; n = 0 is legal). There is no rank-3 ragged shape.
inline Shape ragged(std::size_t n, std::size_t d1) {
    InitShape s;
    s.rank       = 2;
    s.extents[0] = n;
    s.extents[1] = d1;
    return Shape::of(std::move(s));
}
inline Shape ragged(std::size_t n, const std::vector<std::size_t>& counts) {
    if (counts.size() != n)
        throw std::invalid_argument("naga::cospan: init::ragged: the counts have " + std::to_string(counts.size()) +
                                    " entries but dimension 0 has " + std::to_string(n) + " indices");
    InitShape s;
    s.rank       = 2;
    s.extents[0] = n;
    s.extents[1] = 0;
    s.per        = true;
    s.per_index  = counts;
    return Shape::of(std::move(s));
}

/// Every element = value (converted to the variable's element type). The one-argument form carries no shape: it is
/// valid when every dimension of the variable is specified.
namespace impl {
namespace detail = ::naga::cospan::detail;
template <class T>
InitSpec fill_impl(T value, std::optional<InitShape> shape) {
    return InitSpec{[value](detail::InitCtx& c) -> sycl::event {
        sycl::event e;
        detail::dispatch_dtype(c.var.dtype, [&](auto tag) {
            using E = typename decltype(tag)::type;
            auto& d = static_cast<detail::DeviceData<E>&>(c.data);
            e       = d.v().fill(static_cast<E>(value), std::span<const sycl::event>{c.deps});
        });
        return e;
    }, std::move(shape)};
}
inline InitSpec zero_impl(std::optional<InitShape> shape) {
    return InitSpec{[](detail::InitCtx& c) -> sycl::event {
        sycl::event e;
        detail::dispatch_dtype(c.var.dtype, [&](auto tag) {
            using E = typename decltype(tag)::type;
            auto& d = static_cast<detail::DeviceData<E>&>(c.data);
            e       = d.v().fill(E{}, std::span<const sycl::event>{c.deps});
        });
        return e;
    }, std::move(shape)};
}
}  // namespace impl
template <class T>
InitSpec fill(T value) {
    return impl::fill_impl(value, std::nullopt);
}
/// fill(v, {n, m}): every element = v, and the shape is {n, m} (it supplies the variable's unspecified dimensions).
template <class T>
InitSpec fill(T value, const Shape& shape) {
    return impl::fill_impl(value, std::optional<InitShape>{shape.get()});
}

/// Every element = 0 (no shape).
inline InitSpec zero() { return impl::zero_impl(std::nullopt); }
/// zero({n, m}): every element = 0, with that shape.
inline InitSpec zero(const Shape& shape) { return impl::zero_impl(std::optional<InitShape>{shape.get()}); }

namespace impl {
namespace detail = ::naga::cospan::detail;
template <class T>
InitSpec from_host_impl(std::vector<T> values, std::optional<InitShape> shape) {
    return InitSpec{[values = std::move(values)](detail::InitCtx& c) -> sycl::event {
        const std::size_t n = c.var.shape.elements();
        if (values.size() != n)
            throw std::invalid_argument("naga::cospan: init::from_host has " + std::to_string(values.size()) +
                                        " values but '" + c.var.display_path() + "' has " + std::to_string(n) + " elements");
        sycl::event e;
        detail::dispatch_dtype(c.var.dtype, [&](auto tag) {
            using E = typename decltype(tag)::type;
            std::vector<E> converted(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) converted[i] = static_cast<E>(values[i]);
            auto& d = static_cast<detail::DeviceData<E>&>(c.data);
            e       = d.v().from_host(std::span<const E>{converted.data(), converted.size()}, std::span<const sycl::event>{c.deps});
        });
        return e;
    }, std::move(shape)};
}
}  // namespace impl

/// Element i = values[i]; the size must equal the variable's number of elements when it is applied. The data's own shape is
/// rank 1 of its size: it supplies a rank-1 variable's unspecified dimension.
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

/// A device-side initial value. `fn` gets the unordered queue, the device the data lives on, the events its work must
/// come after, and the writable twin of the variable; it enqueues its own kernel (or copy) with a dependency on `deps`
/// and RETURNS THE EVENT OF THAT WORK (section 6). DT and Rank are the variable's, spelled at the call:
///   cs::init::custom<DType::f32, 1>([](sk::unordered_queue& q, const sycl::device& dev,
///                                      const std::vector<sycl::event>& deps, cs::Initializer<DType::f32, 1> out) {
///       return q.parallel_for(dev, deps, sycl::range<1>{out.size()}, [=](sycl::id<1> i) { out[i] = 1.0f; });
///   })
template <DType DT, int Rank>
using CustomFn = std::function<sycl::event(sk::unordered_queue&, const sycl::device&, const std::vector<sycl::event>&,
                                           Initializer<DT, Rank>)>;

namespace impl {
namespace detail = ::naga::cospan::detail;
template <DType DT, int Rank>
InitSpec custom_impl(CustomFn<DT, Rank> fn, std::optional<InitShape> shape) {
    return InitSpec{[fn = std::move(fn)](detail::InitCtx& c) -> sycl::event {
        if (!dtype_compatible(c.var.dtype, DT) || c.var.rank != Rank)
            throw std::invalid_argument("naga::cospan: init::custom for '" + c.var.display_path() +
                                        "' was declared with the wrong dtype or rank");
        return fn(c.rt.uq, c.data.key.dev, c.deps, detail::make_initializer<DT, Rank>(c.var, c.data));
    }, std::move(shape)};
}
}  // namespace impl
template <DType DT, int Rank>
InitSpec custom(CustomFn<DT, Rank> fn) {
    return impl::custom_impl<DT, Rank>(std::move(fn), std::nullopt);
}
/// custom with a shape: custom<DT, R>(fn, {n, m}); the shape has Rank extents.
template <DType DT, int Rank>
InitSpec custom(CustomFn<DT, Rank> fn, const Shape& shape) {
    if (shape.get().rank != Rank)
        throw std::invalid_argument("naga::cospan: init::custom<..., " + std::to_string(Rank) + "> was given a shape of rank " +
                                    std::to_string(shape.get().rank));
    return impl::custom_impl<DT, Rank>(std::move(fn), std::optional<InitShape>{shape.get()});
}

/// A host-side initial value: the function receives a zero-initialized host buffer of the variable (its previous
/// contents are NOT read from the device); the buffer is copied to the device (a queued copy) after it returns, and
/// that copy is the event the initializer returns.
namespace impl {
namespace detail = ::naga::cospan::detail;
template <DType DT, int Rank>
InitSpec custom_host_impl(std::function<void(HostInit<DT, Rank>&)> fn, std::optional<InitShape> shape) {
    return InitSpec{[fn = std::move(fn)](detail::InitCtx& c) -> sycl::event {
        if (!dtype_compatible(c.var.dtype, DT) || c.var.rank != Rank)
            throw std::invalid_argument("naga::cospan: init::custom_host for '" + c.var.display_path() +
                                        "' was declared with the wrong dtype or rank");
        using E = cpp_type_t<DT>;
        HostInit<DT, Rank> host = detail::HostInitAccess::make<DT, Rank>(c.var.shape);
        sycl::event::wait(c.deps);  // the host function may read what its dependencies produced
        fn(host);
        auto& d = static_cast<detail::DeviceData<E>&>(c.data);
        return d.v().from_host(std::span<const E>{host.data(), host.size()}, std::span<const sycl::event>{});
    }, std::move(shape)};
}
}  // namespace impl
template <DType DT, int Rank>
InitSpec custom_host(std::function<void(HostInit<DT, Rank>&)> fn) {
    return impl::custom_host_impl<DT, Rank>(std::move(fn), std::nullopt);
}
/// custom_host with a shape: custom_host<DT, R>(fn, {n, m}).
template <DType DT, int Rank>
InitSpec custom_host(std::function<void(HostInit<DT, Rank>&)> fn, const Shape& shape) {
    if (shape.get().rank != Rank)
        throw std::invalid_argument("naga::cospan: init::custom_host<..., " + std::to_string(Rank) + "> was given a shape of rank " +
                                    std::to_string(shape.get().rank));
    return impl::custom_host_impl<DT, Rank>(std::move(fn), std::optional<InitShape>{shape.get()});
}

}  // namespace init
}  // namespace naga::cospan
