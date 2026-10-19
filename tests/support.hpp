#pragma once

// =============================================================================
// tests/support.hpp — small helpers shared by the Catch2 tests
// =============================================================================
//
// Every test goes through a cospan::Model (create_model / setup / run /
// wait / view_* / update_input). Nothing here reaches past the Model's public API.
//
// Reading results (internal/design-review-2026-10-03.md, section 5): run() only enqueues. A host view can be
// obtained only after the first forward, and it must be sync()ed before it is read. The helpers below do the
// forward + view + sync sequence so the cases do not repeat it.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace cst {

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;

/// The configuration every test uses: no descriptors, so no pool on any device (every allocation takes the OS route).
inline cs::model_config_desc default_config() { return cs::model_config_desc{}; }

/// default_config() plus a 1 MiB pool on every device: transient roots draw from it, so
/// pool_in_use() measures what they hold.
inline cs::model_config_desc pooled_config() {
    return cs::model_config_desc{cs::pool_desc{cs::all_devices, std::size_t{1} << 20}};
}

/// Pool bytes handed out right now, summed over the model's devices (between runs).
inline std::size_t pool_in_use(const cs::Model& m) {
    std::size_t total = 0;
    for (const auto& [d, p] : m.memory_report().devices) total += p.pool_bytes_in_use;
    return total;
}

/// The pool's node size (cs::node_size_desc, 1024 by default).
inline constexpr std::size_t pool_node_bytes = 1024;

/// What a container of `bytes` takes from a pool: its capacity rounded up to whole nodes (0 for no storage). The pool's
/// bytes in use on a device are the sum of this over the live containers on that device.
constexpr std::size_t pool_bytes_of(std::size_t bytes) {
    return (bytes + pool_node_bytes - 1) / pool_node_bytes * pool_node_bytes;
}

/// The arrays a plain or group component keeps for launching ALONE: its argument array (one element), and its launcher's
/// pointer array and geometry array. Each is far below one node, so each takes one. Custom-op and host components keep none.
constexpr std::size_t solo_launch_bytes(std::size_t kernel_components) { return 3 * kernel_components * pool_node_bytes; }
/// The same for a fused group of `stages` stages: one argument array per stage, one pointer array, one geometry array.
constexpr std::size_t group_launch_bytes(std::size_t stages) { return (stages + 2) * pool_node_bytes; }

// ---- the removed fuse strategies, as a test helper -------------------------------
/// What cs::fuse_components<seq|par> was, spelled with the segment API (P7a): seq = ONE segment holding every stage, range
/// mode own (each stage runs over its own range, so stages of different sizes are safe); par = every stage its own segment (a
/// bare descriptor is a one-stage segment).
enum class FuseMode { seq, par };

template <FuseMode S, class... D>
auto fuse(std::string name, cs::ConnectsDesc c, D... d) {
    if constexpr (S == FuseMode::seq)
        return cs::fuse_components(std::move(name), std::move(c), cs::segment<cs::range_mode::own>(std::move(d)...));
    else
        return cs::fuse_components(std::move(name), std::move(c), std::move(d)...);
}
template <FuseMode S, class... D>
    requires(sizeof...(D) >= 1 && !cs::detail::leads_with_connects_v<D...>)
auto fuse(std::string name, D... d) {
    return fuse<S>(std::move(name), cs::ConnectsDesc{}, std::move(d)...);
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
/// One run, then wait for everything submitted ("run it and let it finish").
inline void forward_and_wait(cs::Model& m) {
    m.run();
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

/// One run, then a synced host view of an output (run + view + sync).
template <DType DT, int Rank>
inline cs::HostView<DT, Rank> wait_and_view(cs::Model& m, std::string_view path) {
    m.run();
    return synced_output<DT, Rank>(m, path);
}

}  // namespace cst
