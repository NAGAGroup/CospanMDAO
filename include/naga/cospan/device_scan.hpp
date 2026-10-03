#pragma once

// =============================================================================
// naga/cospan/device_scan.hpp — a parallel exclusive scan of u64 on the device
// =============================================================================
//
// The scan behind a value-driven column (variable.hpp, do_column_work) runs in parallel on the device, ragged dimensions
// included (docs/design-review-2026-10-03.md, section 9): no single serial task. REDUCE-THEN-SCAN over work-groups:
//
//   (a) an nd_range kernel: each work-group scans its chunk of `wg` elements (sycl::exclusive_scan_over_group), writes the
//       chunk-local exclusive results into `out`, and writes its chunk sum into a block-sums buffer;
//   (b) the block sums are scanned the same way: recursively when there is more than one work-group's worth of them,
//       else by one work-group;
//   (c) one kernel adds each block's offset to its chunk, and writes out[n], the total.
//
// With a single work-group (n <= wg) the three steps are one kernel. The temporary block-sum buffers are usm_vectors from
// the Runtime's allocator on the device; the final event is registered on them before they go out of scope, so their
// free waits for the kernels that read them. n == 0 writes out[0] = 0.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/runtime.hpp"

namespace naga::cospan::detail {

/// The scan's work-group size: the device's maximum, at most 256.
inline std::size_t scan_group_size(const sycl::device& dev) {
    return std::min<std::size_t>(dev.get_info<sycl::info::device::max_work_group_size>(), 256);
}

/// out[0..n] = the exclusive scan of in[0..n) (n + 1 outputs: out[n] is the total), after `deps`, on `dev`. `in` and
/// `out` are device-accessible USM pointers that must stay valid until the returned event completes; they must not
/// overlap. Returns the event of the last kernel.
inline sycl::event exclusive_scan_u64(Runtime& rt, const sycl::device& dev, const std::vector<sycl::event>& deps,
                                      const std::uint64_t* in, std::uint64_t* out, std::size_t n) {
    if (n == 0) return rt.uq.single_task(dev, deps, [=]() { out[0] = 0; });

    const std::size_t wg     = scan_group_size(dev);
    const std::size_t groups = (n + wg - 1) / wg;

    if (groups == 1) {  // one work-group: scan, and the total, in one kernel
        return rt.uq.parallel_for(dev, deps, sycl::nd_range<1>{sycl::range<1>{wg}, sycl::range<1>{wg}}, [=](sycl::nd_item<1> it) {
            const std::size_t   i     = it.get_local_id(0);
            const std::uint64_t v     = i < n ? in[i] : 0;
            const std::uint64_t ex    = sycl::exclusive_scan_over_group(it.get_group(), v, std::uint64_t{0}, sycl::plus<std::uint64_t>{});
            const std::uint64_t total = sycl::reduce_over_group(it.get_group(), v, sycl::plus<std::uint64_t>{});
            if (i < n) out[i] = ex;
            if (i == 0) out[n] = total;
        });
    }

    const auto alloc = rt.allocator_for<std::uint64_t>(dev, sk::kind::device);
    sk::usm_vector<std::uint64_t> sums(groups, alloc);       // one chunk sum per work-group
    sk::usm_vector<std::uint64_t> block_offs(groups + 1, alloc);  // their exclusive scan; the last is the total
    std::uint64_t*                s = sums.data();
    std::uint64_t*                b = block_offs.data();

    // (a) each work-group scans its chunk and writes its sum
    std::vector<sycl::event> da = deps;
    for (const sycl::event& e : sums.events()) da.push_back(e);
    const sycl::event ev_a = rt.uq.parallel_for(
        dev, da, sycl::nd_range<1>{sycl::range<1>{groups * wg}, sycl::range<1>{wg}}, [=](sycl::nd_item<1> it) {
            const std::size_t   i     = it.get_global_id(0);
            const std::uint64_t v     = i < n ? in[i] : 0;
            const std::uint64_t ex    = sycl::exclusive_scan_over_group(it.get_group(), v, std::uint64_t{0}, sycl::plus<std::uint64_t>{});
            const std::uint64_t chunk = sycl::reduce_over_group(it.get_group(), v, sycl::plus<std::uint64_t>{});
            if (i < n) out[i] = ex;
            if (it.get_local_id(0) == 0) s[it.get_group(0)] = chunk;
        });
    sums.register_event(ev_a);

    // (b) the block sums, scanned the same way (recursively when they need more than one work-group)
    std::vector<sycl::event> db{ev_a};
    for (const sycl::event& e : block_offs.events()) db.push_back(e);
    const sycl::event ev_b = exclusive_scan_u64(rt, dev, db, s, b, groups);
    sums.register_event(ev_b);
    block_offs.register_event(ev_b);

    // (c) each chunk gets its block's offset; out[n] is the total
    const std::vector<sycl::event> dc{ev_a, ev_b};
    const sycl::event ev_c = rt.uq.parallel_for(dev, dc, sycl::range<1>{n + 1}, [=](sycl::id<1> id) {
        const std::size_t i = id[0];
        if (i < n)
            out[i] += b[i / wg];
        else
            out[n] = b[groups];
    });
    sums.register_event(ev_c);
    block_offs.register_event(ev_c);
    return ev_c;
}

}  // namespace naga::cospan::detail
