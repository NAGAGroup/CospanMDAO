// =============================================================================
// include/naga/cospan/core/unsafe.hpp — the ONE raw-pointer escape hatch
// =============================================================================
//
// FRAMEWORK_CLEANUP S4: the default view types expose NO pointer surface.
// The only way user code obtains raw device pointers is an explicit mutation
// through naga::cospan::unsafe_cast(view), which returns a bare struct of typed
// pointers. The friction is the point: every raw-pointer site in user code is
// a deliberate, grep-able opt-in (`grep -rn unsafe_cast`).
//
// What you give up by casting: slot discipline (r/w pair mechanics), grad
// routing, and the compile-time capability tiers. What you get: pointers.
// Use it for hand-rolled nd-range/work-group kernels and probe plumbing that
// the framework primitives don't cover — nothing else.
//
// Framework internals do NOT route through unsafe_cast (they have
// detail::RawAccess); hits on unsafe_cast are user code by definition.

#pragma once

#include <cstddef>

#include "naga/cospan/core/io.hpp"

namespace naga::cospan::core {

namespace detail {

struct RawAccess {
    // Input
    template <int Dim, DType DT>
    static dtype_t<DT>* data(const Input<Dim, DT>& v) noexcept {
        return v.data_.data();
    }
    template <int Dim, DType DT>
    static dtype_t<DT>* grad(const Input<Dim, DT>& v) noexcept {
        return v.grad_.data();
    }
    template <int Dim, DType DT>
    static dtype_t<DT>* accum_grad(const Input<Dim, DT>& v) noexcept {
        return v.accum_grad_.data();
    }

    // Output
    template <int Dim, DType DT>
    static dtype_t<DT>* data(const Output<Dim, DT>& v) noexcept {
        return v.data_.data();
    }
    template <int Dim, DType DT>
    static dtype_t<DT>* grad(const Output<Dim, DT>& v) noexcept {
        return v.grad_.data();
    }

    // InternalBuffer
    template <int Dim, DType DT>
    static dtype_t<DT>* data(const InternalBuffer<Dim, DT>& v) noexcept {
        return v.data_;
    }
    template <int Dim, DType DT>
    static dtype_t<DT>* grad(const InternalBuffer<Dim, DT>& v) noexcept {
        return v.grad_;
    }

    // IOInitializer
    template <int Dim, DType DT>
    static dtype_t<DT>* data(const IOInitializer<Dim, DT>& v) noexcept {
        return v.data_;
    }
    template <int Dim, DType DT>
    static dtype_t<DT>* prev(const IOInitializer<Dim, DT>& v) noexcept {
        return v.prev_;
    }

    // StateVar
    template <int Dim, DType DT>
    static dtype_t<DT>* prev(const StateVar<Dim, DT>& v) noexcept {
        return v.prev_data_;
    }
    template <int Dim, DType DT>
    static dtype_t<DT>* current(const StateVar<Dim, DT>& v) noexcept {
        return v.current_data_;
    }
    template <int Dim, DType DT>
    static dtype_t<DT>* grad_read(const StateVar<Dim, DT>& v) noexcept {
        return v.grad_read_;
    }
    template <int Dim, DType DT>
    static dtype_t<DT>* adj_write(const StateVar<Dim, DT>& v) noexcept {
        return v.adj_write_;
    }
};

} // namespace detail

// ── Bare pointer bundles ─────────────────────────────────────────────────────

template <class T>
struct UnsafeIO {
    T* data = nullptr;
    T* grad = nullptr;        // grad curr slot (null on data tier)
    T* accum_grad = nullptr;  // grad accumulation slot (Input only; null
                              // elsewhere): curr for fresh consumers, prev for
                              // back-edge (deferred) targets
    std::size_t size = 0;
};

template <class T>
struct UnsafeStateVar {
    T* prev = nullptr;       // previous-sweep primal (prev slot)
    T* current = nullptr;    // current-sweep primal (curr slot)
    T* grad_read = nullptr;  // grad curr slot (consumed this adjoint iteration)
    T* adj_write = nullptr;  // grad prev slot (transported to next iteration)
    std::size_t size = 0;
};

template <class T>
struct UnsafeInit {
    T* data = nullptr;  // curr slot
    T* prev = nullptr;  // prev slot (== data for single-slot entries)
    std::size_t size = 0;
};

// ── unsafe_cast overload set ─────────────────────────────────────────────────

template <int Dim, DType DT>
UnsafeIO<dtype_t<DT>> unsafe_cast(const Input<Dim, DT>& v) noexcept {
    return {detail::RawAccess::data(v), detail::RawAccess::grad(v),
            detail::RawAccess::accum_grad(v), v.size()};
}

template <int Dim, DType DT>
UnsafeIO<dtype_t<DT>> unsafe_cast(const Output<Dim, DT>& v) noexcept {
    return {detail::RawAccess::data(v), detail::RawAccess::grad(v), nullptr,
            v.size()};
}

template <int Dim, DType DT>
UnsafeIO<dtype_t<DT>> unsafe_cast(const InternalBuffer<Dim, DT>& v) noexcept {
    return {detail::RawAccess::data(v), detail::RawAccess::grad(v), nullptr,
            v.size()};
}

template <int Dim, DType DT>
UnsafeInit<dtype_t<DT>> unsafe_cast(const IOInitializer<Dim, DT>& v) noexcept {
    return {detail::RawAccess::data(v), detail::RawAccess::prev(v), v.size()};
}

template <int Dim, DType DT>
UnsafeStateVar<dtype_t<DT>> unsafe_cast(const StateVar<Dim, DT>& v) noexcept {
    return {detail::RawAccess::prev(v), detail::RawAccess::current(v),
            detail::RawAccess::grad_read(v), detail::RawAccess::adj_write(v),
            v.size()};
}

} // namespace naga::cospan::core

namespace naga::cospan {
using core::unsafe_cast;
} // namespace naga::cospan
