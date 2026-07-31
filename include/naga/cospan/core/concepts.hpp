// =============================================================================
// include/naga/cospan/core/concepts.hpp — kernel-contract concepts for view types
// =============================================================================
//
// FRAMEWORK_CLEANUP S2: framework kernel primitives (and user functors that
// want to be generic) constrain on these capability concepts — never on
// concrete view types. Each view type satisfies exactly the subset it
// honestly supports, so discipline is enforced at compile time:
//
//   - Data-tier views (i32/i64/u8) never declare grad methods → they can
//     never satisfy GradReadable/GradAccumView.
//   - u8 views never declare atomic_add (sycl::atomic_ref needs 4/8-byte
//     types) → AtomicAddView is unsatisfiable for u8.
//   - IOInitializer satisfies WritableView only (write-only by design —
//     WritableView deliberately does NOT imply ReadableView).
//
// Expected concept table (statically asserted at the bottom):
//
//   type                       Readable Writable AtomicAdd GradRead GradAccum
//   Input<f32>                    x                            x        x
//   Input<i32>                    x
//   Output<f32>                   x        x        x         x
//   Output<u8>                    x        x
//   InternalBuffer<f32>           x        x        x         x        x
//   IOInitializer<f32>                     x
//   StateVar — no direct kernel surface; use the prev()/current() windows.

#pragma once

#include <concepts>
#include <cstddef>

#include "naga/cospan/core/io.hpp"

namespace naga::cospan::core {

template <class V>
concept ReadableView = requires(const V v, std::size_t i) {
    typename V::value_type;
    { v.get(i) } -> std::convertible_to<typename V::value_type>;
    { v.size() } -> std::convertible_to<std::size_t>;
};

// Write-capable view. Deliberately does NOT imply ReadableView so that
// write-only views (IOInitializer) satisfy it honestly; primitives that need
// both compose: `ReadableView auto` + `WritableView auto`.
template <class V>
concept WritableView =
    requires(const V v, std::size_t i, typename V::value_type x) {
        typename V::value_type;
        v.write(i, x);
        { v.size() } -> std::convertible_to<std::size_t>;
    };

template <class V>
concept AtomicAddView =
    requires(const V v, std::size_t i, typename V::value_type x) {
        v.atomic_add(i, x);
    };

template <class V>
concept GradReadable = requires(const V v, std::size_t i) {
    { v.get_grad(i) } -> std::convertible_to<typename V::value_type>;
};

template <class V>
concept GradAccumView =
    requires(const V v, std::size_t i, typename V::value_type x) {
        v.accum_grad(i, x);
    };

// ── Static verification of the concept table ────────────────────────────────

static_assert(ReadableView<Input<1>>);
static_assert(!WritableView<Input<1>>);
static_assert(GradReadable<Input<1>>);
static_assert(GradAccumView<Input<1>>);

static_assert(ReadableView<Input<1, DType::i32>>);
static_assert(!GradReadable<Input<1, DType::i32>>);
static_assert(!GradAccumView<Input<1, DType::i32>>);

static_assert(ReadableView<Output<1>>);
static_assert(WritableView<Output<1>>);
static_assert(AtomicAddView<Output<1>>);
static_assert(GradReadable<Output<1>>);
static_assert(!GradAccumView<Output<1>>);

static_assert(ReadableView<Output<1, DType::u8>>);
static_assert(WritableView<Output<1, DType::u8>>);
static_assert(!AtomicAddView<Output<1, DType::u8>>);
static_assert(!GradReadable<Output<1, DType::u8>>);

static_assert(ReadableView<InternalBuffer<1>>);
static_assert(WritableView<InternalBuffer<1>>);
static_assert(AtomicAddView<InternalBuffer<1>>);
static_assert(GradReadable<InternalBuffer<1>>);
static_assert(GradAccumView<InternalBuffer<1>>);

static_assert(!ReadableView<IOInitializer<1>>);
static_assert(WritableView<IOInitializer<1>>);
static_assert(!GradReadable<IOInitializer<1>>);

} // namespace naga::cospan::core
