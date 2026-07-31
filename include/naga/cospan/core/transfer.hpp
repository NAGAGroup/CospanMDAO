// =============================================================================
// include/naga/cospan/core/transfer.hpp — slot-aware bulk data movement
// =============================================================================
//
// FRAMEWORK_CLEANUP S5b: fill / copy / upload / download. These absorb every
// host↔device staging pattern that used to require data_ptr() + manual slot
// fixup ("memcpy the write slot; if the read slot differs, memcpy it too").
// Slot mechanics are handled HERE — user code never sees them:
//
//   - IOInitializer targets write BOTH slots of double-buffered entries.
//   - Output / Input targets are single windows; one memcpy.
//
// All operations are synchronous (.wait()), matching the call sites they
// replaced (host-side seeding, readback, post-settle commits).
//
// Raw-pointer variants exist only where one side is genuinely outside the
// entry system (host arrays; device scratch acquired via unsafe_cast) — the
// view side stays a view.

#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>

#include "naga/cospan/core/io.hpp"
#include "naga/cospan/core/unsafe.hpp" // detail::RawAccess

namespace naga::cospan::core {

// ── fill ─────────────────────────────────────────────────────────────────────

// Fill both slots of an initializer window with a constant.
template <int Dim, DType DT>
void fill(sycl::queue& q, const IOInitializer<Dim, DT>& dst,
          dtype_t<DT> value) {
    auto* a = detail::RawAccess::data(dst);
    auto* b = detail::RawAccess::prev(dst);
    const std::size_t n = dst.size();
    if (n == 0) return;
    q.fill(a, value, n).wait();
    if (b && b != a) q.fill(b, value, n).wait();
}

// Fill an Output window with a constant.
template <int Dim, DType DT>
void fill(sycl::queue& q, const Output<Dim, DT>& dst, dtype_t<DT> value) {
    const std::size_t n = dst.size();
    if (n == 0) return;
    q.fill(detail::RawAccess::data(dst), value, n).wait();
}

// Fill an InternalBuffer's primal with a constant (fwd→bwd scratch resets).
template <int Dim, DType DT>
void fill(sycl::queue& q, const InternalBuffer<Dim, DT>& dst,
          dtype_t<DT> value) {
    const std::size_t n = dst.size();
    if (n == 0) return;
    q.fill(detail::RawAccess::data(dst), value, n).wait();
}

// ── upload (host → device) ───────────────────────────────────────────────────

// Upload into both slots of an initializer window. `offset` is the element
// offset into the destination entry (host_src is read from element 0).
template <int Dim, DType DT>
void upload(sycl::queue& q, const IOInitializer<Dim, DT>& dst,
            const dtype_t<DT>* host_src, std::size_t count,
            std::size_t offset = 0) {
    if (count == 0) return;
    auto* a = detail::RawAccess::data(dst);
    auto* b = detail::RawAccess::prev(dst);
    q.memcpy(a + offset, host_src, count * sizeof(dtype_t<DT>)).wait();
    if (b && b != a)
        q.memcpy(b + offset, host_src, count * sizeof(dtype_t<DT>)).wait();
}

// Upload into an Output window (single buffer).
template <int Dim, DType DT>
void upload(sycl::queue& q, const Output<Dim, DT>& dst,
            const dtype_t<DT>* host_src, std::size_t count,
            std::size_t offset = 0) {
    if (count == 0) return;
    q.memcpy(detail::RawAccess::data(dst) + offset, host_src,
             count * sizeof(dtype_t<DT>)).wait();
}

// ── download (device → host) ─────────────────────────────────────────────────

template <int Dim, DType DT>
void download(sycl::queue& q, dtype_t<DT>* host_dst,
              const Output<Dim, DT>& src, std::size_t count,
              std::size_t offset = 0) {
    if (count == 0) return;
    q.memcpy(host_dst, detail::RawAccess::data(src) + offset,
             count * sizeof(dtype_t<DT>)).wait();
}

template <int Dim, DType DT>
void download(sycl::queue& q, dtype_t<DT>* host_dst,
              const Input<Dim, DT>& src, std::size_t count,
              std::size_t offset = 0) {
    if (count == 0) return;
    q.memcpy(host_dst, detail::RawAccess::data(src) + offset,
             count * sizeof(dtype_t<DT>)).wait();
}

// ── copy (device → device) ───────────────────────────────────────────────────

// Raw device source (pointer typically acquired via unsafe_cast upstream —
// e.g. the post-settle commit copying a realized slot into a plain Output).
template <int Dim, DType DT>
void copy(sycl::queue& q, const Output<Dim, DT>& dst,
          const dtype_t<DT>* src_device, std::size_t count) {
    if (count == 0) return;
    q.memcpy(detail::RawAccess::data(dst), src_device,
             count * sizeof(dtype_t<DT>)).wait();
}

// Raw device source into BOTH slots of an initializer window — the sanctioned
// out-of-band mutation path for persistent (no-op-forward) outputs that may
// carry a prev/curr slot pair (e.g. the post-settle committed-buffer commit).
template <int Dim, DType DT>
void copy(sycl::queue& q, const IOInitializer<Dim, DT>& dst,
          const dtype_t<DT>* src_device, std::size_t count) {
    if (count == 0) return;
    auto* a = detail::RawAccess::data(dst);
    auto* b = detail::RawAccess::prev(dst);
    q.memcpy(a, src_device, count * sizeof(dtype_t<DT>)).wait();
    if (b && b != a)
        q.memcpy(b, src_device, count * sizeof(dtype_t<DT>)).wait();
}

} // namespace naga::cospan::core

namespace naga::cospan {
using core::fill;
using core::upload;
using core::download;
using core::copy;
} // namespace naga::cospan
