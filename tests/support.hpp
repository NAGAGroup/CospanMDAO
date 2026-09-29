#pragma once

// =============================================================================
// tests/support.hpp — small helpers shared by the Catch2 tests
// =============================================================================
//
// Every test goes through a cospan::Model (create_model / setup / forward /
// wait / view_* / set_*). Nothing here reaches past the Model's public API.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace cst {

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;

/// The configuration every test uses: the runtime's default device, one context.
inline cs::ModelConfig default_config() { return cs::ModelConfig{}; }

// ---- host-reference comparison ------------------------------------------------
/// |got - want| <= abs_tol + rel_tol * |want| (a NaN never compares close).
inline bool close(float got, float want, float abs_tol = 1e-6f, float rel_tol = 1e-5f) {
    return std::fabs(got - want) <= abs_tol + rel_tol * std::fabs(want);
}
inline bool close(double got, double want, double abs_tol = 1e-12, double rel_tol = 1e-9) {
    return std::fabs(got - want) <= abs_tol + rel_tol * std::fabs(want);
}

/// Same size, and every element close.
template <class T>
inline bool all_close(std::span<const T> got, std::span<const T> want) {
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!close(got[i], want[i])) return false;
    return true;
}
template <class T>
inline bool all_close(const std::vector<T>& got, const std::vector<T>& want) {
    return all_close<T>(std::span<const T>{got}, std::span<const T>{want});
}

/// The elements of a host view as a vector (for comparing with ==, or with all_close).
template <class View>
inline auto to_vector(const View& v) {
    const auto s = v.span();
    return std::vector<std::remove_cvref_t<decltype(s[0])>>(s.begin(), s.end());
}

// ---- running a Model -----------------------------------------------------------
/// One forward, wait, then a read-only host copy of an output.
template <DType DT, int Rank>
inline cs::HostView<DT, Rank> wait_and_view(cs::Model& m, std::string_view path) {
    m.forward();
    m.wait();
    return m.view_output<DT, Rank>(path);
}

}  // namespace cst
