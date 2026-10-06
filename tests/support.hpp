#pragma once

// =============================================================================
// tests/support.hpp — small helpers shared by the Catch2 tests
// =============================================================================
//
// Every test goes through a cospan::Model (create_model / setup / forward /
// wait / view_* / update_input). Nothing here reaches past the Model's public API.
//
// Reading results (docs/design-review-2026-10-03.md, section 5): forward() only enqueues. A host view can be
// obtained only after the first forward, and it must be sync()ed before it is read. The helpers below do the
// forward + view + sync sequence so the cases do not repeat it.

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

/// The configuration every test uses: the runtime's default device, one context, no pool.
inline cs::ModelConfig default_config() { return cs::ModelConfig{}; }

/// default_config() plus a 1 MiB pool on every device: transient roots draw from it, so
/// pool_in_use() measures what they hold.
inline cs::ModelConfig pooled_config() {
    cs::ModelConfig cfg = cs::ModelConfig{};
    cfg.default_pool_bytes = std::size_t{1} << 20;
    return cfg;
}

/// Pool bytes handed out right now, summed over the model's devices (between runs).
inline std::size_t pool_in_use(const cs::Model& m) {
    std::size_t total = 0;
    for (const auto& [d, p] : m.memory_report().devices) total += p.pool_bytes_in_use;
    return total;
}

/// The pool's node size (ModelConfig::pool_node_size, 1024 by default).
inline constexpr std::size_t pool_node_bytes = 1024;

/// What a container of `bytes` takes from a pool: its capacity rounded up to whole nodes (0 for no storage). The pool's
/// bytes in use on a device are the sum of this over the live containers on that device.
constexpr std::size_t pool_bytes_of(std::size_t bytes) {
    return (bytes + pool_node_bytes - 1) / pool_node_bytes * pool_node_bytes;
}

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

/// The elements of a host view as a vector (for comparing with ==, or with all_close). The view must be synced.
template <class View>
inline auto to_vector(const View& v) {
    const auto s = v.span();
    return std::vector<std::remove_cvref_t<decltype(s[0])>>(s.begin(), s.end());
}

// ---- running a Model -----------------------------------------------------------
/// One forward, then wait for everything submitted ("run it and let it finish").
inline void forward_and_wait(cs::Model& m) {
    m.forward();
    m.wait();
}

/// A synced read-only host view of an output. The model must have run at least once.
template <DType DT, int Rank>
inline cs::HostView<DT, Rank> synced_output(cs::Model& m, std::string_view path) {
    auto v = m.view_output<DT, Rank>(path);
    v.sync();
    return v;
}

/// A synced read-only host view of an input. The model must have run at least once.
template <DType DT, int Rank>
inline cs::HostView<DT, Rank> synced_input(cs::Model& m, std::string_view path) {
    auto v = m.view_input<DT, Rank>(path);
    v.sync();
    return v;
}

/// One forward, then a synced host view of an output (forward + view + sync).
template <DType DT, int Rank>
inline cs::HostView<DT, Rank> wait_and_view(cs::Model& m, std::string_view path) {
    m.forward();
    return synced_output<DT, Rank>(m, path);
}

}  // namespace cst
