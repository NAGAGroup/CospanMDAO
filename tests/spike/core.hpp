#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace spike {

struct StageRecord {
    std::size_t   flat_extent;
    std::size_t   group_offset;
    std::size_t   group_count;
    std::uint32_t body_off;
};

// Position-as-identity: every stage in a call sequence gets the same arguments,
// so each advances this shared cursor as its first statement.
struct FnOffsetInfo {
    const StageRecord* records;
    std::uint32_t      i;
    const StageRecord& next() { return records[i++]; }
};

// The placeholder: declaration only; the JIT replaces calls to it.
SYCL_EXTERNAL void kb_entry_r1(sycl::nd_item<1> it, FnOffsetInfo& info, const std::byte* bodies,
                               std::byte* local_base);

// The one outer kernel. Blocks until done (spike only).
void launch_r1(sycl::queue& q, sycl::AdaptiveCpp_jit::dynamic_function_config& cfg,
               std::size_t total_groups, std::size_t wg, const StageRecord* records_dev,
               const std::byte* bodies_dev, std::size_t local_bytes);

}  // namespace spike
