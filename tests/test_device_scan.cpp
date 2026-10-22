// The parallel exclusive scan of u32 on the device (cospanmdao/include/naga/cospan/device_scan.hpp), against std::exclusive_scan.
//
// Sizes cover: empty, one element, below / at / above one work-group (the group size is min(device max, 256)), several
// work-groups, and enough work-groups (> 256) that the block sums are themselves scanned in more than one work-group.
// The scan is exercised directly on a Runtime; nothing else of the Model is involved.

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace cs = naga::cospan;
namespace sk = naga::slabkit;

namespace {

/// Scans `values` on the default device and returns the n + 1 outputs.
std::vector<std::uint32_t> device_scan(const std::vector<std::uint32_t>& values) {
    const sycl::device dev{sycl::default_selector_v};
    sk::context        ctx(dev);
    sk::queue          q{ctx, sycl::async_handler{}};
    sk::unordered_queue uq{ctx, sycl::async_handler{}};
    cs::detail::Runtime rt{q, uq, {}};  // no pools: direct allocations

    const std::size_t n = values.size();
    // host-kind USM: readable by the kernels and by this thread
    sk::usm_vector<std::uint32_t> in(n, rt.allocator_for<std::uint32_t>(dev, sk::kind::host));
    sk::usm_vector<std::uint32_t> out(n + 1, rt.allocator_for<std::uint32_t>(dev, sk::kind::host));
    // Size-only containers are uninitialized and submit nothing; sync() is kept so the host writes below follow any event.
    in.sync();
    for (std::size_t i = 0; i < n; ++i) in.data()[i] = values[i];
    // The scan reads `in` (it waits on its writes) and writes `out` (it waits on both of its lists).
    std::vector<sycl::event> deps = in.read_deps();
    for (const sycl::event& e : out.read_deps()) deps.push_back(e);
    for (const sycl::event& e : out.write_deps()) deps.push_back(e);

    sycl::event ev = cs::detail::exclusive_scan_u32(rt, dev, deps, in.data(), out.data(), n);
    ev.wait();
    q.wait_and_throw();
    uq.wait_and_throw();
    return std::vector<std::uint32_t>(out.data(), out.data() + n + 1);
}

std::vector<std::uint32_t> reference(const std::vector<std::uint32_t>& values) {
    std::vector<std::uint32_t> out(values.size() + 1, 0);
    std::exclusive_scan(values.begin(), values.end(), out.begin(), std::uint32_t{0});
    out[values.size()] = std::accumulate(values.begin(), values.end(), std::uint32_t{0});
    return out;
}

std::vector<std::uint32_t> pattern(std::size_t n) {
    std::vector<std::uint32_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = (i * 2654435761u + 7u) % 11u;  // small, with zeros
    return v;
}

}  // namespace

TEST_CASE("device scan: exclusive scan of u32, n + 1 outputs, the last is the total", "[scan]") {
    for (const std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{37}, std::size_t{255}, std::size_t{256},
                                std::size_t{257}, std::size_t{1000}, std::size_t{4096 + 3}}) {
        INFO("n = " << n);
        const auto v = pattern(n);
        CHECK(device_scan(v) == reference(v));
    }
}

TEST_CASE("device scan: more than one work-group of block sums scans them recursively", "[scan]") {
    // 256 * 256 + 17 elements: at the largest group size the block sums need 257 entries, more than one work-group's worth
    // (a smaller device group size reaches the recursion earlier).
    const auto v = pattern(256 * 256 + 17);
    CHECK(device_scan(v) == reference(v));
}

TEST_CASE("device scan: all zeros and a single large value", "[scan]") {
    std::vector<std::uint32_t> zeros(700, 0);
    CHECK(device_scan(zeros) == reference(zeros));
    std::vector<std::uint32_t> big(700, 0);
    big[699] = std::uint32_t{1} << 30;
    CHECK(device_scan(big) == reference(big));
}
