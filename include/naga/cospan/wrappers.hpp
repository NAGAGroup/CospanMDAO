#pragma once

// =============================================================================
// naga/cospan/wrappers.hpp — the device-side data handles (r01 §2.1, §5.2)
// =============================================================================
//
// Input, Output, Initializer and Local are pure device payload: {shape,
// stride, data}, trivially copyable, no identity. Their fields are private and
// written only by detail::IOEntry (friendship): user code never sees a pointer.
// Indexing is by sycl::id<Rank> or by a flat row-major index over shape();
// stride() is the element step per dimension in the OWNING allocation.

#include <array>
#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "naga/cospan/dtype.hpp"

namespace naga::cospan {

struct CustomOp;  // the one place a raw device pointer is obtainable (r01 §5.1)

namespace detail {

class IOEntry;
struct ScratchLayout;
struct ItemFixup;

/// A range of rank R taken from the first R entries of a 1-padded triple.
template <int R>
constexpr sycl::range<R> make_range(const std::array<std::size_t, 3>& a) {
    if constexpr (R == 1)
        return sycl::range<1>{a[0]};
    else if constexpr (R == 2)
        return sycl::range<2>{a[0], a[1]};
    else
        return sycl::range<3>{a[0], a[1], a[2]};
}

/// A range of rank R with every extent `v`.
template <int R>
constexpr sycl::range<R> filled_range(std::size_t v) {
    return make_range<R>({v, v, v});
}

template <int R>
constexpr std::size_t range_size(const sycl::range<R>& r) {
    std::size_t n = 1;
    for (int d = 0; d < R; ++d) n *= r[d];
    return n;
}

/// Element offset of a multi-index under `stride`.
template <int R>
constexpr std::size_t offset_of(const sycl::id<R>& idx, const sycl::range<R>& stride) {
    std::size_t o = 0;
    for (int d = 0; d < R; ++d) o += idx[d] * stride[d];
    return o;
}

/// Element offset of a flat row-major index over `shape`, under `stride`.
template <int R>
constexpr std::size_t offset_of_flat(std::size_t flat, const sycl::range<R>& shape,
                                     const sycl::range<R>& stride) {
    std::size_t o = 0;
    for (int d = R - 1; d >= 0; --d) {
        o += (flat % shape[d]) * stride[d];
        flat /= shape[d];
    }
    return o;
}

}  // namespace detail

/// Input<DT, Rank> — read-only view. A kernel cannot write through an Input:
/// all access is const-reference. Atomic dtypes are ill-formed on Input.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
struct Input {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    const value_type& operator[](sycl::id<Rank> idx) const {
        return data_[detail::offset_of<Rank>(idx, stride_)];
    }
    const value_type& operator[](std::size_t flat) const {
        return data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)];
    }

    sycl::range<Rank> shape() const { return shape_; }
    sycl::range<Rank> stride() const { return stride_; }
    std::size_t       size() const { return detail::range_size<Rank>(shape_); }

private:
    sycl::range<Rank> shape_  = detail::filled_range<Rank>(0);
    sycl::range<Rank> stride_ = detail::filled_range<Rank>(0);
    const value_type* data_   = nullptr;
    friend class detail::IOEntry;
    friend struct CustomOp;
};

/// Output<DT, Rank> — read-write view with exactly ONE access discipline per
/// instantiation: bare mutable references for plain dtypes, sycl::atomic_ref
/// (relaxed, device scope) for atomic dtypes. No bare reference is obtainable
/// from an atomic Output.
template <DType DT, int Rank>
struct Output {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    using value_type = cpp_type_t<DT>;
    using atomic_type =
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    value_type& operator[](sycl::id<Rank> idx) const
        requires(!is_atomic_v<DT>)
    {
        return data_[detail::offset_of<Rank>(idx, stride_)];
    }
    value_type& operator[](std::size_t flat) const
        requires(!is_atomic_v<DT>)
    {
        return data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)];
    }

    atomic_type operator[](sycl::id<Rank> idx) const
        requires(is_atomic_v<DT>)
    {
        return atomic_type{data_[detail::offset_of<Rank>(idx, stride_)]};
    }
    atomic_type operator[](std::size_t flat) const
        requires(is_atomic_v<DT>)
    {
        return atomic_type{data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)]};
    }

    sycl::range<Rank> shape() const { return shape_; }
    sycl::range<Rank> stride() const { return stride_; }
    std::size_t       size() const { return detail::range_size<Rank>(shape_); }

private:
    sycl::range<Rank> shape_  = detail::filled_range<Rank>(0);
    sycl::range<Rank> stride_ = detail::filled_range<Rank>(0);
    value_type*       data_   = nullptr;
    friend class detail::IOEntry;
    friend struct CustomOp;
};

/// Initializer<DT, Rank> — the writable twin of an I/O for init(): the same
/// accessor discipline as Output, over the entry's root storage on the
/// Component's device. Obtained inside init() only (r01 §6.2).
template <DType DT, int Rank>
struct Initializer {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    using value_type = cpp_type_t<DT>;
    using atomic_type =
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::device>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    value_type& operator[](sycl::id<Rank> idx) const
        requires(!is_atomic_v<DT>)
    {
        return data_[detail::offset_of<Rank>(idx, stride_)];
    }
    value_type& operator[](std::size_t flat) const
        requires(!is_atomic_v<DT>)
    {
        return data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)];
    }

    atomic_type operator[](sycl::id<Rank> idx) const
        requires(is_atomic_v<DT>)
    {
        return atomic_type{data_[detail::offset_of<Rank>(idx, stride_)]};
    }
    atomic_type operator[](std::size_t flat) const
        requires(is_atomic_v<DT>)
    {
        return atomic_type{data_[detail::offset_of_flat<Rank>(flat, shape_, stride_)]};
    }

    sycl::range<Rank> shape() const { return shape_; }
    sycl::range<Rank> stride() const { return stride_; }
    std::size_t       size() const { return detail::range_size<Rank>(shape_); }

private:
    sycl::range<Rank> shape_  = detail::filled_range<Rank>(0);
    sycl::range<Rank> stride_ = detail::filled_range<Rank>(0);
    value_type*       data_   = nullptr;
    friend class detail::IOEntry;
    friend struct CustomOp;
};

/// Local<DT, Rank> — group-local memory with wrapper ergonomics. Plain dtypes
/// only in v1. The storage is group-local and the wrapper carries a byte
/// OFFSET instead of a pointer; the device function fixes up the pointer as
/// local_base + offset before the body runs. Contiguous row-major over shape.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
struct Local {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    using value_type = cpp_type_t<DT>;

    void reshape(sycl::range<Rank> shape) { shape_ = shape; }  // host, compute_forward

    value_type& operator[](sycl::id<Rank> idx) const {  // device, post-fixup
        std::size_t flat = 0;
        for (int d = 0; d < Rank; ++d) flat = flat * shape_[d] + idx[d];
        return ptr_[flat];
    }
    value_type&       operator[](std::size_t flat) const { return ptr_[flat]; }
    sycl::range<Rank> shape() const { return shape_; }
    std::size_t       size() const { return detail::range_size<Rank>(shape_); }

private:
    sycl::range<Rank> shape_  = detail::filled_range<Rank>(0);
    std::size_t       offset_ = 0;        // into the launch's local allocation
    value_type*       ptr_    = nullptr;  // null until device-side fixup
    friend struct detail::ScratchLayout;
};

static_assert(std::is_trivially_copyable_v<Input<DType::f32, 2>>);
static_assert(std::is_trivially_copyable_v<Output<DType::atomic_f32, 1>>);
static_assert(std::is_trivially_copyable_v<Initializer<DType::f32, 1>>);
static_assert(std::is_trivially_copyable_v<Local<DType::f32, 2>>);

}  // namespace naga::cospan
