// =============================================================================
// naga/core/dtype.hpp — DType: the framework's closed universe of I/O element
// types (FRAMEWORK_CLEANUP S1/S1a).
// =============================================================================
//
// The enum IS the API surface: declaration and access APIs are templated on
// the enum value (e.g. declare_input<DType::i32, 1>), and the enum member is
// recorded in entry metadata. One glance at this enum shows the entire
// supported universe.
//
// Capability tiers:
//   - Differentiable (f32, f64): may carry gradients / participate in the
//     adjoint machinery and be design variables.
//   - Data (i32, i64, u8): connections, promotions, sharing, checkpointing —
//     but NO grad slots, ever.
//
// No unsigned arithmetic types by design (u8 is passive bit-pattern storage
// for flags/masks, not arithmetic). i64 exists for at-scale counts/offsets.
//
// Note: the adjoint/slot machinery is currently f32-only; f64 entries are
// accepted as data storage but do not receive grad slots (their views carry
// null grad windows — do not call gradient methods on them).

#pragma once

#include <cstddef>
#include <cstdint>

namespace naga::core {

enum class DType : std::uint8_t { f32, f64, i32, i64, u8 };

constexpr std::size_t dtype_size(DType dt) {
  switch (dt) {
  case DType::f32:
    return 4;
  case DType::f64:
    return 8;
  case DType::i32:
    return 4;
  case DType::i64:
    return 8;
  case DType::u8:
    return 1;
  }
  return 0;
}

constexpr const char *dtype_name(DType dt) {
  switch (dt) {
  case DType::f32:
    return "f32";
  case DType::f64:
    return "f64";
  case DType::i32:
    return "i32";
  case DType::i64:
    return "i64";
  case DType::u8:
    return "u8";
  }
  return "?";
}

// Differentiable tier — see header comment. NOTE: io.hpp grad methods gate on
// f32 specifically (no f64 grad storage exists yet); this predicate defines
// the tier itself and re-enters the gates when the adjoint is dtyped.
constexpr bool dtype_differentiable(DType dt) {
  return dt == DType::f32 || dt == DType::f64;
}

// Internal trait mapping DType → C++ type. The enum remains the user-facing
// surface; this trait exists only so framework internals can mint typed
// pointers/views from the type-erased storage.
template <DType DT> struct dtype_traits;
template <> struct dtype_traits<DType::f32> { using type = float; };
template <> struct dtype_traits<DType::f64> { using type = double; };
template <> struct dtype_traits<DType::i32> { using type = std::int32_t; };
template <> struct dtype_traits<DType::i64> { using type = std::int64_t; };
template <> struct dtype_traits<DType::u8> { using type = std::uint8_t; };

template <DType DT> using dtype_t = typename dtype_traits<DT>::type;

} // namespace naga::core
