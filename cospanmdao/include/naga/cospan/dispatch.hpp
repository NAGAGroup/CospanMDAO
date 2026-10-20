#pragma once

// =============================================================================
// naga/cospan/dispatch.hpp — from run-time dtype and rank to compile-time types
// =============================================================================
//
// A variable's dtype and an instance's rank are known at run time (the shape
// definition travels with the declaration), while its storage and its typed
// leaves are templates. These two functions are the one place that bridge is
// crossed: they switch over the closed set of dtypes (and ranks 1..3) and call
// a generic callable with the compile-time values.

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "naga/cospan/dtype.hpp"

namespace naga::cospan::detail {

/// Calls f(std::type_identity<T>{}) with T the C++ element type of `dt`. Atomic and
/// plain dtypes share a storage type, so this switches on value_dtype.
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

/// Calls f(std::integral_constant<DType, DT>{}, std::integral_constant<int, R>{}) for the 13 dtypes
/// (atomic ones included: they are distinct DTypes) and ranks 1..3. \throws std::invalid_argument for
/// any other rank.
template <class F>
void dispatch_dtype_rank(DType dt, int rank, F&& f) {
    const auto by_rank = [&](auto dtc) {
        switch (rank) {
            case 1: f(dtc, std::integral_constant<int, 1>{}); break;
            case 2: f(dtc, std::integral_constant<int, 2>{}); break;
            case 3: f(dtc, std::integral_constant<int, 3>{}); break;
            default: throw std::invalid_argument("naga::cospan: rank is 1..3");
        }
    };
    switch (dt) {
        case DType::f32: by_rank(std::integral_constant<DType, DType::f32>{}); return;
        case DType::f64: by_rank(std::integral_constant<DType, DType::f64>{}); return;
        case DType::i32: by_rank(std::integral_constant<DType, DType::i32>{}); return;
        case DType::u32: by_rank(std::integral_constant<DType, DType::u32>{}); return;
        case DType::i64: by_rank(std::integral_constant<DType, DType::i64>{}); return;
        case DType::u64: by_rank(std::integral_constant<DType, DType::u64>{}); return;
        case DType::u8: by_rank(std::integral_constant<DType, DType::u8>{}); return;
        case DType::atomic_f32: by_rank(std::integral_constant<DType, DType::atomic_f32>{}); return;
        case DType::atomic_f64: by_rank(std::integral_constant<DType, DType::atomic_f64>{}); return;
        case DType::atomic_i32: by_rank(std::integral_constant<DType, DType::atomic_i32>{}); return;
        case DType::atomic_u32: by_rank(std::integral_constant<DType, DType::atomic_u32>{}); return;
        case DType::atomic_i64: by_rank(std::integral_constant<DType, DType::atomic_i64>{}); return;
        case DType::atomic_u64: by_rank(std::integral_constant<DType, DType::atomic_u64>{}); return;
    }
    throw std::invalid_argument("naga::cospan: unknown dtype");
}

}  // namespace naga::cospan::detail
