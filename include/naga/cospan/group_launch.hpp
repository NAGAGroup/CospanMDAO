#pragma once

// =============================================================================
// naga/cospan/group_launch.hpp — the launcher of a group of stages (design review, 11.6)
// =============================================================================
//
// EVERY plain-body and group-body component launches through a GroupLauncher: a fused group owns one for its stages, and a
// bare component owns one for itself (a group of one stage in one default segment). It owns the two device arrays the
// kernel reads besides the stages' own argument arrays:
//   - the POINTER ARRAY: one pointer per stage, to that stage's one-element argument array. Allocated, filled and
//     waited on ONCE, at the first precompute (devices are final by then, not at setup); it is redone only when the
//     group is rebuilt (a new launcher).
//   - the GEOMETRY ARRAY: one StageGeometry per stage and then the segment table, refilled from a host member EVERY
//     forward with the container's event ordering.
// The class is declared here; its methods are defined in component.hpp, after ComponentBase.
//
// SEGMENTS. A segment is a run of consecutive stages sharing groups: it has its own block of groups, the largest group
// count any of its stages asked for, and the launch is the sum of the blocks. Range mode (own: a stage runs for its first
// N groups of the block; whole: for every group of the block) and local memory (partitioned: each stage its own slice,
// the segment's size is the sum; shared: one block, the size is the max) are per segment. The launch's local block is the
// largest segment size. THERE ARE NO FRAMEWORK BARRIERS: a stage that reads across work-items what an earlier stage wrote
// calls its own group barrier; across groups nothing is ordered.
//
// GROUP WAITS (the fuse_components option cs::group_waits_enabled, or a cs::global_barrier() between two segments, of a
// group of group-body stages): either puts the WHOLE launch in ticket mode. The group then owns a small device array, the TICKET STATE (kernel.hpp), reset
// with a container fill before every launch; the launch is an outside writer of it. Each work-group takes a ticket at its
// start and the ticket, not the hardware group id, says which segment and group it is, so a group only ever waits for
// earlier tickets and no co-residency is needed. A global_barrier makes every group of the segment after it wait for every
// group of the earlier segments; GroupItem::wait_group / wait_earlier_segments wait from inside a body. The segment
// table after the stage entries of the geometry array is the blocks, then each segment's boundary kind.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/kernel.hpp"
#include "naga/cospan/runtime.hpp"

namespace naga::cospan {

class ComponentBase;

namespace detail {

/// A run of consecutive stages (`count` of them, from `first`) that share groups.
struct SegmentSpec {
    std::size_t first = 0;
    std::size_t count = 1;
    LocalMem    local = LocalMem::partitioned;
    RangeMode   range = RangeMode::own;
    Boundary    boundary = Boundary::none;  ///< what stands between the previous segment and this one
};

class GroupLauncher {
public:
    /// The first precompute: the pointer array (once). Every stage's argument array must exist.
    void prepare(Runtime& rt, const std::vector<ComponentBase*>& stages);

    /// One forward: describe every stage, check the work-group shapes, upload the stages' arguments and the geometry, submit
    /// ONE launch, register its event on every container it touches and on every stage. `who` names the owner in messages.
    /// Returns the launch's events (none when every stage is empty).
    /// `group_waits`: the group has the group_waits_enabled property (a global_barrier in `segments` also switches the wait
    /// state on). Default false: a lone component.
    /// `split`: the range_split strategy (a bare component: pad, one segment, the native launch).
    std::vector<sycl::event> launch(Runtime& rt, const std::vector<ComponentBase*>& stages, const std::vector<SegmentSpec>& segments,
                                    const std::string& who, bool group_waits = false, Split split = Split::pad);

    const std::vector<sycl::event>& events() const { return last_; }

    /// Whether the wait state exists: only after a launch of a group with group waits (property or global_barrier).
    bool has_wait_state() const { return ticket_.has_value(); }

private:
    std::optional<sk::usm_vector<const void*>>   ptrs_;
    std::optional<sk::usm_vector<StageGeometry>> geo_;
    std::optional<sk::usm_vector<std::uint32_t>> ticket_;  ///< wait state ("ticket state"): only in a group with group waits, reset every forward
    std::vector<StageGeometry>                   host_geo_;  ///< the host member the geometry array is copied from
    std::vector<sycl::event>                     last_;
};

}  // namespace detail
}  // namespace naga::cospan
