// =============================================================================
// include/naga/core/io.hpp — dtyped device-view types for component I/O
// =============================================================================
//
// FRAMEWORK_CLEANUP P2: every view type is templated on <int Dim, DType DT>.
// DT defaults to DType::f32 so the historical spellings (Input<1>, Output<2>,
// StateVar<1>, ...) remain valid and mean exactly what they always meant.
// Dtyped access spells the enum explicitly: Input<1, DType::i32>.
//
// Capability tiers (see dtype.hpp):
//   - Differentiable (f32): gradient methods exist. f64 is differentiable-tier
//     by dtype but has no grad storage yet — widen the f32 gates when the
//     adjoint is dtyped.
//   - Data (i32, i64, u8): NO gradient methods — they are not merely
//     unsupported at runtime, they do not exist on the type (requires-gated),
//     so passing a data-tier view to a gradient-touching kernel is a compile
//     error.
//   - atomic_add exists only where sycl::atomic_ref is legal (4/8-byte
//     element types) — u8 views do not satisfy AtomicAddView.
//
// NO raw-pointer surface: data_ptr()/grad_ptr()/prev_data_ptr()/
// current_data_ptr() are gone. Bulk movement goes through the transfer
// primitives (core/transfer.hpp); deliberate raw-pointer work goes through
// naga::unsafe_cast (core/unsafe.hpp). Framework internals reach storage via
// detail::RawAccess (friended below) — user code has exactly one escape
// hatch, and it is grep-able.
//
// StateVar exposes its slot-pair storage exclusively through two windows
// (FRAMEWORK_CLEANUP S3):
//   sv.prev()    → Input window  : get() reads the prev slot,
//                                  accum_grad() writes the grad prev slot
//                                  (the self-back-edge transport, consumed
//                                  next adjoint iteration).
//   sv.current() → Output window : write()/atomic_add() the curr slot,
//                                  get_grad() reads the grad curr slot (the
//                                  total gradient this adjoint iteration).
//
// All types are trivially copyable for SYCL kernel capture.

#pragma once

#include <sycl/sycl.hpp>
#include <slabkit/array.hpp>
#include <cstddef>
#include <type_traits>

#include "naga/core/dtype.hpp"

namespace naga::core {

namespace detail {
// Sole framework-internal accessor to view storage (befriended by every view
// type). Used by core/transfer.hpp and core/unsafe.hpp. User code must go
// through naga::unsafe_cast.
struct RawAccess;

template <int Dim>
inline size_t flat_idx(sycl::id<Dim> id, sycl::range<Dim> r) noexcept {
    if constexpr (Dim == 1)      return id[0];
    else if constexpr (Dim == 2) return id[0] * r[1] + id[1];
    else                         return id[0] * r[1] * r[2] + id[1] * r[2] + id[2];
}
} // namespace detail

// =============================================================================
// IOInitializer<Dim, DT> — write-only initializer view.
//
// For recurrent outputs (StateVars and any back-edge cut source) write()
// stores to BOTH the curr slot (current sweep) and the prev slot so component
// authors never think about slot management. For single-slot entries both
// pointers are identical.
// =============================================================================

template <int Dim, DType DT = DType::f32>
struct IOInitializer {
    using value_type = dtype_t<DT>;
    static constexpr DType dtype = DT;

    IOInitializer() noexcept : data_(nullptr), prev_(nullptr), range_{} {}
    IOInitializer(value_type* data, value_type* prev,
                  sycl::range<Dim> range) noexcept
        : data_(data), prev_(prev), range_(range) {}

    void write(sycl::id<Dim> idx, value_type val) const {
        const size_t flat = detail::flat_idx<Dim>(idx, range_);
        data_[flat] = val;
        prev_[flat] = val;
    }

    void write(size_t flat, value_type val) const {
        data_[flat] = val;
        prev_[flat] = val;
    }

    sycl::range<Dim> range() const noexcept { return range_; }
    size_t size() const noexcept {
        size_t n = 1;
        for (int i = 0; i < Dim; ++i) n *= range_[i];
        return n;
    }

private:
    value_type*      data_;  // curr slot (current sweep)
    value_type*      prev_;  // prev slot — == data_ if single-slot
    sycl::range<Dim> range_;

    friend struct detail::RawAccess;
};

static_assert(std::is_trivially_copyable_v<IOInitializer<1>>);

// =============================================================================
// Input<Dim, DT> — read data; write grad (atomic accumulate) on the
// differentiable tier.
// =============================================================================

template <int Dim, DType DT = DType::f32>
struct Input {
    using value_type = dtype_t<DT>;
    static constexpr DType dtype = DT;

    Input() noexcept
        : data_{nullptr, sycl::range<Dim>{}},
          grad_{nullptr, sycl::range<Dim>{}},
          accum_grad_{nullptr, sycl::range<Dim>{}} {}

    // Full window (differentiable use — grads may still be null views when the
    // entry carries no grad storage, e.g. f64 in P1..P3).
    Input(slabkit::DeviceView<value_type, Dim> data,
          slabkit::DeviceView<value_type, Dim> read_grad,
          slabkit::DeviceView<value_type, Dim> accum_grad) noexcept
        : data_(data), grad_(read_grad), accum_grad_(accum_grad) {}

    // Data-only window (data tier).
    explicit Input(slabkit::DeviceView<value_type, Dim> data) noexcept
        : data_(data),
          grad_{nullptr, data.get_range()},
          accum_grad_{nullptr, data.get_range()} {}

    // Forward data access
    value_type get(size_t flat_idx) const { return data_.data()[flat_idx]; }
    value_type get(sycl::id<Dim> id) const { return data_[id]; }

    // Gradient access — differentiable tier only
    value_type get_grad(sycl::id<Dim> id) const noexcept
        requires(DT == DType::f32)
    { return grad_[id]; }
    value_type get_grad(int idx) const noexcept
        requires(DT == DType::f32)
    { return grad_.data()[idx]; }

    // Write gradient via atomic accumulate — differentiable tier only
    void accum_grad(sycl::id<Dim> id, value_type v) const noexcept
        requires(DT == DType::f32)
    {
        value_type* gptr = &accum_grad_[id];
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                         sycl::memory_scope::device> gref(*gptr);
        gref.fetch_add(v);
    }

    void accum_grad(int idx, value_type v) const noexcept
        requires(DT == DType::f32)
    {
        value_type* gptr = &accum_grad_.data()[idx];
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                         sycl::memory_scope::device> gref(*gptr);
        gref.fetch_add(v);
    }

    sycl::range<Dim> range() const noexcept { return data_.get_range(); }
    size_t size() const noexcept { return data_.size(); }

private:
    slabkit::DeviceView<value_type, Dim> data_;
    slabkit::DeviceView<value_type, Dim> grad_;        // grad curr slot (read)
    slabkit::DeviceView<value_type, Dim> accum_grad_;  // accumulation slot: grad
                                                       // curr for fresh consumers,
                                                       // grad prev when this input
                                                       // is a back-edge (deferred)
                                                       // target

    friend struct detail::RawAccess;
};

static_assert(std::is_trivially_copyable_v<Input<1>>);
static_assert(std::is_trivially_copyable_v<Input<2>>);
static_assert(std::is_trivially_copyable_v<Input<3>>);
static_assert(std::is_trivially_copyable_v<Input<1, DType::i32>>);

// =============================================================================
// Output<Dim, DT> — read-write data; read-only grad on the differentiable tier.
// =============================================================================

template <int Dim, DType DT = DType::f32>
struct Output {
    using value_type = dtype_t<DT>;
    static constexpr DType dtype = DT;

    Output() noexcept
        : data_{nullptr, sycl::range<Dim>{}},
          grad_{nullptr, sycl::range<Dim>{}} {}

    Output(slabkit::DeviceView<value_type, Dim> data,
           slabkit::DeviceView<value_type, Dim> read_grad) noexcept
        : data_(data), grad_(read_grad) {}

    // Data-only window (data tier).
    explicit Output(slabkit::DeviceView<value_type, Dim> data) noexcept
        : data_(data), grad_{nullptr, data.get_range()} {}

    // Forward data access
    value_type get(size_t flat_idx) const { return data_.data()[flat_idx]; }
    value_type get(sycl::id<Dim> id) const { return data_[id]; }

    // Write forward data
    void write(size_t flat_idx, value_type v) const { data_.data()[flat_idx] = v; }
    void write(sycl::id<Dim> id, value_type v) const { data_[id] = v; }

    // Atomic fetch-add on data — 4/8-byte element types only
    void atomic_add(size_t flat_idx, value_type v) const
        requires(sizeof(value_type) >= 4)
    {
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                         sycl::memory_scope::device> dref(data_.data()[flat_idx]);
        dref.fetch_add(v);
    }

    // Gradient access — differentiable tier only
    value_type get_grad(sycl::id<Dim> id) const noexcept
        requires(DT == DType::f32)
    { return grad_[id]; }
    value_type get_grad(int idx) const noexcept
        requires(DT == DType::f32)
    { return grad_.data()[idx]; }

    sycl::range<Dim> range() const noexcept { return data_.get_range(); }
    size_t size() const noexcept { return data_.size(); }

private:
    slabkit::DeviceView<value_type, Dim> data_;
    slabkit::DeviceView<value_type, Dim> grad_;  // grad curr slot (read)

    friend struct detail::RawAccess;
};

static_assert(std::is_trivially_copyable_v<Output<1>>);
static_assert(std::is_trivially_copyable_v<Output<2>>);
static_assert(std::is_trivially_copyable_v<Output<3>>);
static_assert(std::is_trivially_copyable_v<Output<1, DType::u8>>);

// =============================================================================
// StateVar<Dim, DT> — intra-component recurrent state: the degenerate
// back-edge where the owning component is also its own deferred consumer.
// current() = owner write to the curr slot; prev() = the back-edge read of
// the prev slot (and its accum_grad routes to the grad prev slot — the
// transported contribution consumed by the NEXT adjoint iteration).
// =============================================================================

template <int Dim, DType DT = DType::f32>
struct StateVar {
    using value_type = dtype_t<DT>;
    static constexpr DType dtype = DT;

    StateVar() noexcept
        : prev_data_(nullptr), current_data_(nullptr),
          grad_read_(nullptr), adj_write_(nullptr), range_{} {}

    StateVar(value_type* prev, value_type* current, value_type* grad_r,
             value_type* adj_w, sycl::range<Dim> range) noexcept
        : prev_data_(prev), current_data_(current),
          grad_read_(grad_r), adj_write_(adj_w), range_(range) {}

    // Previous-tick window: get() reads f_{t-1}; accum_grad() accumulates the
    // adjoint — semantically a normal input gradient.
    Input<Dim, DT> prev() const noexcept {
        return Input<Dim, DT>(
            slabkit::DeviceView<value_type, Dim>(prev_data_, range_),
            slabkit::DeviceView<value_type, Dim>(grad_read_, range_),
            slabkit::DeviceView<value_type, Dim>(adj_write_, range_));
    }

    // Current-tick window: write()/atomic_add() the new value; get_grad()
    // reads the accumulated gradient — semantically a normal output.
    Output<Dim, DT> current() const noexcept {
        return Output<Dim, DT>(
            slabkit::DeviceView<value_type, Dim>(current_data_, range_),
            slabkit::DeviceView<value_type, Dim>(grad_read_, range_));
    }

    sycl::range<Dim> range() const noexcept { return range_; }
    size_t size() const noexcept {
        size_t n = 1;
        for (int i = 0; i < Dim; ++i) n *= range_[i];
        return n;
    }

private:
    value_type*      prev_data_;    // previous-sweep primal (prev slot)
    value_type*      current_data_; // current-sweep primal (curr slot)
    value_type*      grad_read_;    // grad CURR slot — consumed this adjoint iteration
    value_type*      adj_write_;    // grad PREV slot — transported to the next iteration
    sycl::range<Dim> range_;

    friend struct detail::RawAccess;
};

static_assert(std::is_trivially_copyable_v<StateVar<1>>);
static_assert(std::is_trivially_copyable_v<StateVar<2>>);
static_assert(std::is_trivially_copyable_v<StateVar<3>>);

// =============================================================================
// InternalBuffer<Dim, DT> — intra-component intermediate activation buffer.
// Not connectable externally. Read/write primal; read/accum grad on the
// differentiable tier.
// =============================================================================

template <int Dim, DType DT = DType::f32>
struct InternalBuffer {
    using value_type = dtype_t<DT>;
    static constexpr DType dtype = DT;

    InternalBuffer() noexcept : data_(nullptr), grad_(nullptr), range_{} {}
    InternalBuffer(value_type* data, value_type* grad,
                   sycl::range<Dim> range) noexcept
        : data_(data), grad_(grad), range_(range) {}

    // Primal
    value_type get(sycl::id<Dim> id) const noexcept {
        return data_[detail::flat_idx<Dim>(id, range_)];
    }
    value_type get(size_t flat) const noexcept { return data_[flat]; }
    void write(sycl::id<Dim> id, value_type v) const noexcept {
        data_[detail::flat_idx<Dim>(id, range_)] = v;
    }
    void write(size_t flat, value_type v) const noexcept { data_[flat] = v; }
    void atomic_add(sycl::id<Dim> id, value_type v) const noexcept
        requires(sizeof(value_type) >= 4)
    {
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            r(data_[detail::flat_idx<Dim>(id, range_)]);
        r.fetch_add(v);
    }

    // Grad — differentiable tier only
    value_type get_grad(sycl::id<Dim> id) const noexcept
        requires(DT == DType::f32)
    { return grad_[detail::flat_idx<Dim>(id, range_)]; }
    value_type get_grad(size_t flat) const noexcept
        requires(DT == DType::f32)
    { return grad_[flat]; }
    void accum_grad(sycl::id<Dim> id, value_type v) const noexcept
        requires(DT == DType::f32)
    {
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            r(grad_[detail::flat_idx<Dim>(id, range_)]);
        r.fetch_add(v);
    }
    void accum_grad(size_t flat, value_type v) const noexcept
        requires(DT == DType::f32)
    {
        sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                         sycl::memory_scope::device> r(grad_[flat]);
        r.fetch_add(v);
    }

    sycl::range<Dim> range() const noexcept { return range_; }
    size_t size() const noexcept {
        size_t n = 1;
        for (int i = 0; i < Dim; ++i) n *= range_[i];
        return n;
    }

private:
    value_type*      data_;  // single primal buffer (read and write)
    value_type*      grad_;  // single grad buffer (read and atomic accum)
    sycl::range<Dim> range_;

    friend struct detail::RawAccess;
};

static_assert(std::is_trivially_copyable_v<InternalBuffer<1>>);
static_assert(std::is_trivially_copyable_v<InternalBuffer<2>>);
static_assert(std::is_trivially_copyable_v<InternalBuffer<3>>);

}  // namespace naga::core
