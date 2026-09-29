#pragma once

// =============================================================================
// naga/cospan/dtype.hpp — the closed set of I/O element types (r01 §2.1)
// =============================================================================

#include <cstddef>
#include <cstdint>

namespace naga::cospan {

/// The closed set of I/O element types. Extension is a spec revision.
/// Atomic variants are distinct enum entries: declaring an I/O with an atomic
/// dtype changes the wrapper's accessor discipline (atomic references) and
/// nothing else about its storage. An atomic variant exists exactly where
/// sycl::atomic_ref is legal (4/8-byte element types); u8 has no atomic form.
enum class DType : uint8_t {
    f32, f64, i32, u32, i64, u64, u8,
    atomic_f32, atomic_f64, atomic_i32, atomic_u32, atomic_i64, atomic_u64
};

/// True for the atomic entries.
constexpr bool is_atomic(DType dt) noexcept { return dt >= DType::atomic_f32; }

/// The underlying value dtype (atomic_f32 -> f32). Connection, tie and
/// promotion compatibility is judged on this, never on the DType itself.
constexpr DType value_dtype(DType dt) noexcept {
    return is_atomic(dt) ? static_cast<DType>(static_cast<uint8_t>(dt) -
                                              static_cast<uint8_t>(DType::atomic_f32) +
                                              static_cast<uint8_t>(DType::f32))
                         : dt;
}

/// The atomic form of a dtype, or the dtype itself when it already is one.
/// u8 has none: check has_atomic_form first.
constexpr bool has_atomic_form(DType dt) noexcept { return value_dtype(dt) != DType::u8; }

constexpr DType to_atomic(DType dt) noexcept {
    return has_atomic_form(dt)
               ? static_cast<DType>(static_cast<uint8_t>(value_dtype(dt)) -
                                    static_cast<uint8_t>(DType::f32) +
                                    static_cast<uint8_t>(DType::atomic_f32))
               : dt;
}

constexpr bool is_differentiable(DType dt) noexcept {
    return value_dtype(dt) == DType::f32 || value_dtype(dt) == DType::f64;
}

/// Element size in bytes (through the atomic modifier).
constexpr std::size_t dtype_size(DType dt) noexcept {
    switch (value_dtype(dt)) {
        case DType::f32: case DType::i32: case DType::u32: return 4;
        case DType::f64: case DType::i64: case DType::u64: return 8;
        default: return 1;  // u8
    }
}

/// Whether two declarations may alias: judged on value_dtype, so the atomic
/// modifier never blocks aliasing.
constexpr bool dtype_compatible(DType a, DType b) noexcept {
    return value_dtype(a) == value_dtype(b);
}

// ---- traits ------------------------------------------------------------------

/// value_dtype_of<DT> — underlying value dtype (atomic_f32 -> f32).
template <DType DT>
inline constexpr DType value_dtype_of = value_dtype(DT);

template <DType DT>
inline constexpr bool is_atomic_v = is_atomic(DT);

/// true for f32/f64, plain or atomic.
template <DType DT>
inline constexpr bool is_differentiable_v = is_differentiable(DT);

/// The atomic form of DT; ill-formed for u8.
template <DType DT>
    requires(has_atomic_form(DT))
inline constexpr DType to_atomic_v = to_atomic(DT);

namespace detail {
template <DType V> struct cpp_type;
template <> struct cpp_type<DType::f32> { using type = float; };
template <> struct cpp_type<DType::f64> { using type = double; };
template <> struct cpp_type<DType::i32> { using type = int32_t; };
template <> struct cpp_type<DType::u32> { using type = uint32_t; };
template <> struct cpp_type<DType::i64> { using type = int64_t; };
template <> struct cpp_type<DType::u64> { using type = uint64_t; };
template <> struct cpp_type<DType::u8>  { using type = uint8_t; };
}  // namespace detail

/// C++ element type, through the atomic modifier.
template <DType DT>
using cpp_type_t = typename detail::cpp_type<value_dtype_of<DT>>::type;

static_assert(value_dtype_of<DType::atomic_f32> == DType::f32);
static_assert(value_dtype_of<DType::atomic_u64> == DType::u64);
static_assert(to_atomic_v<DType::i32> == DType::atomic_i32);
static_assert(!has_atomic_form(DType::u8));
static_assert(dtype_size(DType::atomic_f64) == 8);

}  // namespace naga::cospan
