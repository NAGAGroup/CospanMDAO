#pragma once

// =============================================================================
// naga/cospan/kernel.hpp — kernel machinery (design review, sections 4 and 11.6)
// =============================================================================
//
// The KernelBody contract (concepts read off the call operator's signature), GroupItem, the Local layout, the
// per-component argument struct (StageArgs) and the per-stage geometry entry (StageGeometry), the STAGE FUNCTION
// stage_fn<KB> (one per body type), the placeholder run_stages the JIT replaces, and the group launch helper.
// All header-only: placeholders, outer kernels and stage functions instantiate in the consumer's translation unit.
//
// THE LAUNCH PATH (every plain-body and group-body component, bare or in a fused group; custom-op and host bodies keep
// their own). ONE generic rank-3 kernel calls the declared dynamic function
//     run_stages(StageItem, const StageGeometry* geometry, const void* const* args, int stage)
// whose definition is an INDEXED CALL SEQUENCE of stage functions: stage k of the sequence is called with the same
// arguments and `stage` = k (fork: define_as_indexed_call_sequence(&run_stages, {ids...}, 3)). A stage function reads
// geometry[k] and args[k] (its own StageArgs, a one-element device array every component keeps), builds the item
// wrapper (GroupItem) and calls the user's body. The configuration of a sequence lives in a process-wide cache
// (group_config), created once per distinct sequence and never destroyed: the runtime reads a configuration on a worker
// thread after the submission, and a transient group can be deleted before its kernel has even been compiled.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/dtype.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

namespace sk = ::naga::slabkit;

// -----------------------------------------------------------------------------
// CustomOp, Host: the first-parameter tags of the non-kernel bodies
// -----------------------------------------------------------------------------

/// What a custom-operation body receives (D1: native backend work only). The
/// one place a raw device pointer is obtainable (r02 section 4.1).
struct CustomOp {
    sycl::interop_handle& h;

    /// Input -> const value_type*, Output/Initializer -> value_type*.
    template <class W>
    auto device_ptr(const W& io) const {
        return io.data_;
    }
};

/// First-parameter tag of a host body (D3): `void operator()(Host, Inputs, Outputs) const`.
struct Host {};

// -----------------------------------------------------------------------------
// Segment options: cs::segment<cs::local_mem::shared, cs::range_mode::whole>(descs...)
// -----------------------------------------------------------------------------

/// Local memory of a segment. partitioned (the default): every stage gets its own slice, the segment's size is the sum.
/// shared: every stage gets the same block, the size is the max.
namespace local_mem {
struct partitioned {};
struct shared {};
}  // namespace local_mem

/// Range of a segment. own (the default): a stage runs only for its first N groups of the segment's block and skips the
/// rest. whole: every stage runs for every group of the block.
namespace range_mode {
struct own {};
struct whole {};
}  // namespace range_mode

namespace detail {
enum class LocalMem : std::uint8_t { partitioned = 0, shared = 1 };
enum class RangeMode : std::uint8_t { own = 0, whole = 1 };

/// What stands before a segment in a fused group (cs::global_barrier()). A group with a global_barrier, like a group with
/// the group_waits_enabled property, has the wait state ("ticket mode" internally); global_barrier also makes every group of
/// the segment wait for every earlier segment.
enum class Boundary : std::uint8_t { none = 0, global_barrier = 1 };

/// The range_split strategy of a fused group (a bare component launches as pad_and_split with one segment: the native launch).
enum class Split : std::uint8_t { flatten = 0, pad = 1, padded = 2 };

/// One entry per stage, in a device array the group owns and refills every forward. Followed (after `stages` entries) by
/// the segment table: one SegmentInfo per segment.
struct StageGeometry {
    void*         ticket_state = nullptr;  ///< the group's ticket state; null: the group neither has group_waits_enabled nor a global_barrier (no ticket mode)
    std::uint64_t total_groups = 0;        ///< the logical groups of the launch: the sum of the segments' counts
    std::uint32_t segment      = 0;
    std::uint32_t segments     = 0;
    std::uint32_t stages       = 0;  ///< the number of stage entries (where the segment table starts)
    std::uint32_t local_offset = 0;  ///< bytes into the launch's local block
    std::uint32_t local_bytes  = 0;  ///< this stage's slice
    std::uint32_t range_mode   = 0;  ///< RangeMode
    std::uint32_t split        = 0;  ///< Split
    std::uint32_t reserved     = 0;
};
static_assert(std::is_trivially_copyable_v<StageGeometry>);
static_assert(sizeof(StageGeometry) % 8 == 0);

/// One entry per segment: its block of groups, as a rank-3 range (the per-dimension maximum of its stages' group ranges,
/// leading 1s for lower ranks), the logical group count (the product), the first ticket (the groups of the earlier segments),
/// the offset along the last dimension (pad_and_split) and what stands before the segment (Boundary).
struct SegmentInfo {
    std::uint64_t range[3] = {0, 0, 0};
    std::uint64_t count    = 0;
    std::uint64_t first    = 0;
    std::uint64_t offset   = 0;
    std::uint64_t boundary = 0;
};
static_assert(std::is_trivially_copyable_v<SegmentInfo>);

inline const SegmentInfo* segment_table(const StageGeometry* geometry) {
    return reinterpret_cast<const SegmentInfo*>(geometry + geometry[0].stages);
}

// ---- rank-R values in the trailing physical dimensions ------------------------------------------------------------------
/// The physical launch is rank 3; a rank-R value lives in the LAST R dimensions (leading 1s for ranges, leading 0s for ids).
template <int R>
sycl::id<R> tail_id(const sycl::id<3>& v) {
    if constexpr (R == 1)
        return sycl::id<1>{v[2]};
    else if constexpr (R == 2)
        return sycl::id<2>{v[1], v[2]};
    else
        return v;
}
template <int R>
sycl::range<R> tail_range(const sycl::range<3>& v) {
    if constexpr (R == 1)
        return sycl::range<1>{v[2]};
    else if constexpr (R == 2)
        return sycl::range<2>{v[1], v[2]};
    else
        return v;
}
template <int R>
sycl::id<3> lift_id(const sycl::id<R>& v) {
    if constexpr (R == 1)
        return sycl::id<3>{0, 0, v[0]};
    else if constexpr (R == 2)
        return sycl::id<3>{0, v[0], v[1]};
    else
        return v;
}
template <int R>
sycl::range<3> lift_range(const sycl::range<R>& v) {
    if constexpr (R == 1)
        return sycl::range<3>{1, 1, v[0]};
    else if constexpr (R == 2)
        return sycl::range<3>{1, v[0], v[1]};
    else
        return v;
}

/// The native item of a stage function: the physical (rank-3) nd_item, the launch's local block (the one thing a stage
/// function cannot obtain by itself), and where the work-group sits LOGICALLY: its segment and its group coordinate in that
/// segment's range (rank 3, the stage's rank-R id is the trailing part).
struct StageItem {
    sycl::nd_item<3> item;
    std::byte*       local;
    std::uint32_t    segment;
    sycl::id<3>      coord;
};

// ---- ticket state ---------------------------------------------------------------------------------------------
// A device array of std::uint32_t the group owns (ticket mode only), reset every forward:
//   [0]            the ticket counter (fetch_add by each work-group's leader at its start)
//   [1, 1+S)       per segment, the number of its groups that have FINISHED every stage of the segment
//   [1+S, 1+S+T)   per ticket, 1 once that group has finished (T = the launch's logical groups)
// The ticket gives the segment and the group: segment-major, then row-major over the segment's range, the last dimension
// fastest, whatever the range_split strategy. Both waits are COLLECTIVE (the leader spins with acquire loads, then a group
// barrier releases the group) and are deadlock-free without co-residency only because a group waits for EARLIER tickets alone:
// every earlier ticket is held by a group that has started (resident or finished).
using TicketRef = sycl::atomic_ref<std::uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>;

/// The row-major index of the group `c` in the segment's range.
inline std::size_t segment_linear(const SegmentInfo& s, const sycl::id<3>& c) {
    return (static_cast<std::size_t>(c[0]) * s.range[1] + c[1]) * s.range[2] + c[2];
}

/// Wait for the group `g` (a rank-3 coordinate) of segment s.
inline void ticket_wait_group(std::uint32_t* ts, std::size_t segments, const SegmentInfo* tbl, std::size_t s, const sycl::id<3>& g,
                              const sycl::nd_item<3>& it) {
    if (it.get_local_linear_id() == 0) {
        const std::size_t t = tbl[s].first + segment_linear(tbl[s], g);
        TicketRef flag{ts[1 + segments + t]};
        while (flag.load(sycl::memory_order::acquire) == 0) {}
    }
    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
}

/// Wait until every group of segments 0 .. upto-1 has finished.
inline void ticket_wait_segments(std::uint32_t* ts, const SegmentInfo* tbl, std::size_t upto, const sycl::nd_item<3>& it) {
    if (it.get_local_linear_id() == 0) {
        for (std::size_t i = 0; i < upto; ++i) {
            TicketRef done{ts[1 + i]};
            while (static_cast<std::uint64_t>(done.load(sycl::memory_order::acquire)) != tbl[i].count) {}
        }
    }
    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
}

// ---- where a physical work-group sits logically ----------------------------------------------------------------------
struct LogicalGroup {
    std::uint32_t segment = 0;
    sycl::id<3>   coord{0, 0, 0};
    bool          valid = false;
};

/// The group number `lin` (a ticket, or the flattened hardware id) -> segment and coordinate: segment-major, row-major.
inline LogicalGroup locate_linear(const SegmentInfo* tbl, std::size_t segments, std::uint64_t lin, std::uint64_t total) {
    LogicalGroup r;
    if (lin >= total) return r;
    std::size_t s = 0;
    while (s + 1 < segments && lin >= tbl[s].first + tbl[s].count) ++s;
    std::uint64_t k = lin - tbl[s].first;
    r.segment       = static_cast<std::uint32_t>(s);
    r.coord[2]      = k % tbl[s].range[2];
    k /= tbl[s].range[2];
    r.coord[1] = k % tbl[s].range[1];
    r.coord[0] = k / tbl[s].range[1];
    r.valid    = true;
    return r;
}

/// The hardware-direct mapping (no ticket): flatten_and_split reads the group number along the last dimension; pad_and_split
/// finds the segment by its offset along the last dimension; padded takes the segment from dimension 0. A group outside its
/// segment's range (padding) is not valid.
inline LogicalGroup locate_physical(const StageGeometry& g0, const SegmentInfo* tbl, const sycl::nd_item<3>& it) {
    const std::size_t S = g0.segments;
    if (g0.split == static_cast<std::uint32_t>(Split::flatten)) return locate_linear(tbl, S, it.get_group(2), g0.total_groups);
    LogicalGroup r;
    const std::uint64_t d0 = it.get_group(0), d1 = it.get_group(1), d2 = it.get_group(2);
    if (g0.split == static_cast<std::uint32_t>(Split::pad)) {
        for (std::size_t s = 0; s < S; ++s) {
            if (d2 >= tbl[s].offset && d2 < tbl[s].offset + tbl[s].range[2]) {
                if (d0 < tbl[s].range[0] && d1 < tbl[s].range[1]) {
                    r.segment = static_cast<std::uint32_t>(s);
                    r.coord   = sycl::id<3>{d0, d1, d2 - tbl[s].offset};
                    r.valid   = true;
                }
                break;
            }
        }
    } else {
        if (d0 < S && tbl[d0].range[0] > 0 && d1 < tbl[d0].range[1] && d2 < tbl[d0].range[2]) {
            r.segment = static_cast<std::uint32_t>(d0);
            r.coord   = sycl::id<3>{0, d1, d2};
            r.valid   = true;
        }
    }
    return r;
}

struct ItemFixup;
}  // namespace detail

template <int R>
struct GroupItem;

/// The sub-group of the physical work-group: the SYCL type itself (it has no group coordinate to translate).
using sub_group = sycl::sub_group;

// -----------------------------------------------------------------------------
// Group<R>: the work-group a stage belongs to, with rank-R ids
// -----------------------------------------------------------------------------

/// The CoSpan wrapper of the work-group, made by GroupItem<R>::get_group(). Its group id and range are the stage's rank-R ones
/// (see GroupItem); the local ids and ranges are the trailing R physical ones. to_sycl() (and the implicit conversion) give
/// the PHYSICAL sycl::group<3>, whose ids are not the stage's: pass it to a raw SYCL algorithm, never read ids from it.
/// The cs:: group functions below take a Group<R> and translate stage-rank ids and ranges to rank 3.
template <int R>
struct Group {
    sycl::id<R>    get_group_id() const { return id_; }
    std::size_t    get_group_id(int d) const { return id_[d]; }
    sycl::range<R> get_group_range() const { return range_; }
    std::size_t    get_group_range(int d) const { return range_[d]; }
    std::size_t    get_group_linear_id() const {
        std::size_t l = 0;
        for (int d = 0; d < R; ++d) l = l * range_[d] + id_[d];
        return l;
    }
    std::size_t get_group_linear_range() const {
        std::size_t l = 1;
        for (int d = 0; d < R; ++d) l *= range_[d];
        return l;
    }

    sycl::id<R>    get_local_id() const { return detail::tail_id<R>(g_.get_local_id()); }
    std::size_t    get_local_id(int d) const { return g_.get_local_id(3 - R + d); }
    sycl::range<R> get_local_range() const { return detail::tail_range<R>(g_.get_local_range()); }
    std::size_t    get_local_range(int d) const { return g_.get_local_range(3 - R + d); }
    std::size_t    get_local_linear_id() const { return g_.get_local_linear_id(); }
    std::size_t    get_local_linear_range() const {
        const sycl::range<3> r = g_.get_local_range();
        return r[0] * r[1] * r[2];
    }
    bool leader() const { return g_.get_local_linear_id() == 0; }

    sycl::group<3>           to_sycl() const { return g_; }
    operator sycl::group<3>() const { return g_; }

private:
    Group(const sycl::group<3>& g, const sycl::id<R>& id, const sycl::range<R>& range) : g_{g}, id_{id}, range_{range} {}
    sycl::group<3>    g_;
    sycl::id<R>       id_;
    sycl::range<R>    range_;
    friend struct GroupItem<R>;
};

// -----------------------------------------------------------------------------
// GroupItem<R>
// -----------------------------------------------------------------------------

/// What a group body receives: the API of sycl::nd_item<R> over the stage's rank-R ids. It wraps the PHYSICAL
/// sycl::nd_item<3> of the launch (to_sycl(), or the implicit conversion): the physical ids are NOT the stage's, a stage reads
/// ids from GroupItem only. Every group-level index is relative to the stage's SEGMENT (group ids count from 0 within the
/// segment's range; the group range is the stage's own, or the whole segment's under range_mode::whole). Local ids and ranges
/// are the physical ones of the trailing R dimensions.
template <int R>
struct GroupItem {
    sycl::id<R>    get_local_id() const { return detail::tail_id<R>(it_.get_local_id()); }
    std::size_t    get_local_id(int d) const { return it_.get_local_id(3 - R + d); }
    sycl::range<R> get_local_range() const { return detail::tail_range<R>(it_.get_local_range()); }
    std::size_t    get_local_range(int d) const { return it_.get_local_range(3 - R + d); }
    std::size_t    get_local_linear_id() const { return it_.get_local_linear_id(); }

    sycl::id<R>    get_group_id() const { return group_id_; }
    std::size_t    get_group(int d) const { return group_id_[d]; }
    sycl::range<R> get_group_range() const { return group_range_; }
    std::size_t    get_group_range(int d) const { return group_range_[d]; }
    /// The group's index within its segment, from 0 (row-major over the group range).
    std::size_t    get_group_linear_id() const {
        std::size_t l = 0;
        for (int d = 0; d < R; ++d) l = l * group_range_[d] + group_id_[d];
        return l;
    }

    sycl::id<R> get_global_id() const {
        sycl::id<R> g;
        for (int d = 0; d < R; ++d) g[d] = group_id_[d] * get_local_range(d) + get_local_id(d);
        return g;
    }
    std::size_t    get_global_id(int d) const { return group_id_[d] * get_local_range(d) + get_local_id(d); }
    sycl::range<R> get_global_range() const {
        sycl::range<R> g = group_range_;
        for (int d = 0; d < R; ++d) g[d] = group_range_[d] * get_local_range(d);
        return g;
    }
    std::size_t get_global_linear_id() const {
        const sycl::id<R>    g = get_global_id();
        const sycl::range<R> r = get_global_range();
        std::size_t          l = 0;
        for (int d = 0; d < R; ++d) l = l * r[d] + g[d];
        return l;
    }

    /// The segment this stage belongs to, and how many segments the launch has.
    std::size_t get_segment() const { return segment_; }
    std::size_t get_segment_count() const { return segments_; }
    /// Segment `s`'s block of groups as a rank-R range (the same for every stage of the segment).
    sycl::range<R> get_segment_group_count(std::size_t s) const {
        return detail::tail_range<R>(sycl::range<3>{table_[s].range[0], table_[s].range[1], table_[s].range[2]});
    }

    /// `n` elements of T from this stage's slice of the launch's local memory (a typed bump allocator: each call hands out
    /// the next aligned bytes, after the stage's Scratch). nullptr when the slice, as the stage requested it in its launch
    /// config (request_local_bytes), is exhausted. In a `shared` segment every stage's slice is the same block, so stages
    /// that allocate the same way see the same memory.
    template <class T>
    T* local(std::size_t n) {
        const std::size_t off = (used_ + alignof(T) - 1) / alignof(T) * alignof(T);
        const std::size_t end = off + n * sizeof(T);
        if (end > cap_) return nullptr;
        used_ = static_cast<std::uint32_t>(end);
        return reinterpret_cast<T*>(base_ + off);
    }

    /// COLLECTIVE (every work-item of the group calls it): wait until group `g` (its rank-R group id in segment `s`) has
    /// finished all its stages. Needs a group with the cs::group_waits_enabled property or a cs::global_barrier(), and `s`
    /// EARLIER than this stage's segment; with no wait state the pointer is null and the behavior is undefined (no checks, as
    /// with waiting on your own or a later segment, which never returns).
    void wait_group(std::size_t s, const sycl::id<R>& g) const {
        detail::ticket_wait_group(ticket_state_, segments_, table_, s, detail::lift_id<R>(g), it_);
    }
    /// COLLECTIVE: wait until every group of every earlier segment has finished (what cs::global_barrier() does for you).
    /// The same requirements as wait_group.
    void wait_earlier_segments() const { detail::ticket_wait_segments(ticket_state_, table_, segment_, it_); }

    /// The CoSpan wrapper of the work-group, for the cs:: group functions.
    Group<R> get_group() const { return Group<R>{it_.get_group(), group_id_, group_range_}; }
    sycl::sub_group get_sub_group() const { return it_.get_sub_group(); }

    /// The PHYSICAL item (and the implicit conversion to it): for raw SYCL code; its ids are the launch's, not the stage's.
    sycl::nd_item<3> to_sycl() const { return it_; }
    operator sycl::nd_item<3>() const { return it_; }

private:
    GroupItem(const sycl::nd_item<3>& it, const sycl::id<R>& gid, const sycl::range<R>& grange, std::uint32_t segment,
              std::uint32_t segments, const detail::SegmentInfo* table, std::uint32_t* ticket_state, std::byte* base,
              std::uint32_t used, std::uint32_t cap)
        : it_{it}, group_id_{gid}, group_range_{grange}, segment_{segment}, segments_{segments}, table_{table},
          ticket_state_{ticket_state}, base_{base}, used_{used}, cap_{cap} {}

    sycl::nd_item<3>           it_;
    sycl::id<R>                group_id_;     // within the segment
    sycl::range<R>             group_range_;  // the stage's group range (the segment's under range_mode::whole)
    std::uint32_t              segment_;
    std::uint32_t              segments_;
    const detail::SegmentInfo* table_;        // the segment table
    std::uint32_t*             ticket_state_; // null: not in ticket mode
    std::byte*                 base_;         // this stage's slice of the local block
    std::uint32_t              used_;
    std::uint32_t              cap_;
    friend struct detail::ItemFixup;
};

// -----------------------------------------------------------------------------
// The SYCL 2020 group functions over Group<R>
// -----------------------------------------------------------------------------
// Each mirrors the SYCL function of the same name and arguments, with the CoSpan group wrapper in place of the SYCL group:
// the stage-rank ids and ranges are translated to the physical rank 3. The sub-group functions take the SYCL sub-group
// (cs::sub_group is an alias of it) and are forwarded unchanged.

template <int R>
void group_barrier(const Group<R>& g, sycl::memory_scope fence_scope = sycl::memory_scope::work_group) {
    sycl::group_barrier(g.to_sycl(), fence_scope);
}

template <int R, class T>
T group_broadcast(const Group<R>& g, T x) {
    return sycl::group_broadcast(g.to_sycl(), x);
}
/// By the local LINEAR id of the source work-item.
template <int R, class T>
T group_broadcast(const Group<R>& g, T x, std::size_t local_linear_id) {
    return sycl::group_broadcast(g.to_sycl(), x, local_linear_id);
}
/// By the (rank-R) local id of the source work-item.
template <int R, class T>
T group_broadcast(const Group<R>& g, T x, const sycl::id<R>& local_id) {
    return sycl::group_broadcast(g.to_sycl(), x, detail::lift_id<R>(local_id));
}

template <int R>
bool any_of_group(const Group<R>& g, bool pred) {
    return sycl::any_of_group(g.to_sycl(), pred);
}
template <int R>
bool all_of_group(const Group<R>& g, bool pred) {
    return sycl::all_of_group(g.to_sycl(), pred);
}
template <int R>
bool none_of_group(const Group<R>& g, bool pred) {
    return sycl::none_of_group(g.to_sycl(), pred);
}

template <int R, class Ptr, class Predicate>
bool joint_any_of(const Group<R>& g, Ptr first, Ptr last, Predicate pred) {
    return sycl::joint_any_of(g.to_sycl(), first, last, pred);
}
template <int R, class Ptr, class Predicate>
bool joint_all_of(const Group<R>& g, Ptr first, Ptr last, Predicate pred) {
    return sycl::joint_all_of(g.to_sycl(), first, last, pred);
}
template <int R, class Ptr, class Predicate>
bool joint_none_of(const Group<R>& g, Ptr first, Ptr last, Predicate pred) {
    return sycl::joint_none_of(g.to_sycl(), first, last, pred);
}

template <int R, class T, class Op>
auto reduce_over_group(const Group<R>& g, T x, Op op) {
    return sycl::reduce_over_group(g.to_sycl(), x, op);
}
template <int R, class V, class T, class Op>
auto reduce_over_group(const Group<R>& g, V x, T init, Op op) {
    return sycl::reduce_over_group(g.to_sycl(), x, init, op);
}
template <int R, class Ptr, class Op>
auto joint_reduce(const Group<R>& g, Ptr first, Ptr last, Op op) {
    return sycl::joint_reduce(g.to_sycl(), first, last, op);
}
template <int R, class Ptr, class T, class Op>
auto joint_reduce(const Group<R>& g, Ptr first, Ptr last, T init, Op op) {
    return sycl::joint_reduce(g.to_sycl(), first, last, init, op);
}

template <int R, class T, class Op>
auto exclusive_scan_over_group(const Group<R>& g, T x, Op op) {
    return sycl::exclusive_scan_over_group(g.to_sycl(), x, op);
}
template <int R, class V, class T, class Op>
auto exclusive_scan_over_group(const Group<R>& g, V x, T init, Op op) {
    return sycl::exclusive_scan_over_group(g.to_sycl(), x, init, op);
}
template <int R, class T, class Op>
auto inclusive_scan_over_group(const Group<R>& g, T x, Op op) {
    return sycl::inclusive_scan_over_group(g.to_sycl(), x, op);
}
/// (SYCL's order: the operation, then the initial value.)
template <int R, class V, class Op, class T>
auto inclusive_scan_over_group(const Group<R>& g, V x, Op op, T init) {
    return sycl::inclusive_scan_over_group(g.to_sycl(), x, op, init);
}

template <int R, class InPtr, class OutPtr, class Op>
auto joint_exclusive_scan(const Group<R>& g, InPtr first, InPtr last, OutPtr result, Op op) {
    return sycl::joint_exclusive_scan(g.to_sycl(), first, last, result, op);
}
template <int R, class InPtr, class OutPtr, class T, class Op>
auto joint_exclusive_scan(const Group<R>& g, InPtr first, InPtr last, OutPtr result, T init, Op op) {
    return sycl::joint_exclusive_scan(g.to_sycl(), first, last, result, init, op);
}
template <int R, class InPtr, class OutPtr, class Op>
auto joint_inclusive_scan(const Group<R>& g, InPtr first, InPtr last, OutPtr result, Op op) {
    return sycl::joint_inclusive_scan(g.to_sycl(), first, last, result, op);
}
template <int R, class InPtr, class OutPtr, class Op, class T>
auto joint_inclusive_scan(const Group<R>& g, InPtr first, InPtr last, OutPtr result, Op op, T init) {
    return sycl::joint_inclusive_scan(g.to_sycl(), first, last, result, op, init);
}

template <class T>
T shift_group_left(sub_group g, T x, sub_group::linear_id_type delta = 1) {
    return sycl::shift_group_left(g, x, delta);
}
template <class T>
T shift_group_right(sub_group g, T x, sub_group::linear_id_type delta = 1) {
    return sycl::shift_group_right(g, x, delta);
}
template <class T>
T permute_group_by_xor(sub_group g, T x, sub_group::linear_id_type mask) {
    return sycl::permute_group_by_xor(g, x, mask);
}
template <class T>
T select_from_group(sub_group g, T x, sub_group::id_type remote_local_id) {
    return sycl::select_from_group(g, x, remote_local_id);
}

// -----------------------------------------------------------------------------
// Launch facts a fused owner reads off a kernel description
// -----------------------------------------------------------------------------

/// What a group launcher reads off a kernel description.
struct KernelParts {
    int                           rank        = 0;
    bool                          group_body  = false;
    std::size_t                   flat_extent = 0;  ///< the stage's iteration space (0: nothing to run)
    std::size_t                   local_bytes = 0;  ///< the stage's slice: its Scratch plus the bytes it requested, each rounded to 16
    std::optional<sycl::range<3>> wg;               ///< the work-group shape the stage asked for, if any
};

// -----------------------------------------------------------------------------
// The KernelBody concepts: the signature is the declaration
// -----------------------------------------------------------------------------

namespace detail {

struct None {};

template <std::size_t I, class Def, class... A>
struct arg_or {
    using type = Def;
};
template <std::size_t I, class Def, class H, class... T>
struct arg_or<I, Def, H, T...> : arg_or<I - 1, Def, T...> {};
template <class Def, class H, class... T>
struct arg_or<0, Def, H, T...> {
    using type = H;
};

template <class M>
struct call_traits {
    static constexpr bool valid = false;
    using a0 = None; using a1 = None; using a2 = None; using a3 = None;
    static constexpr std::size_t nargs = 0;
};
template <class C, class R, class... A>
struct call_traits<R (C::*)(A...) const> {
    static constexpr bool        valid = true;
    static constexpr std::size_t nargs = sizeof...(A);
    using a0 = typename arg_or<0, None, A...>::type;
    using a1 = typename arg_or<1, None, A...>::type;
    using a2 = typename arg_or<2, None, A...>::type;
    using a3 = typename arg_or<3, None, A...>::type;
};

template <class KB>
struct body_traits : call_traits<void> {};
template <class KB>
    requires requires { &KB::operator(); }
struct body_traits<KB> : call_traits<decltype(&KB::operator())> {};

/// kind: 0 plain (id<R>), 1 group (GroupItem<R>), 2 custom op, 3 host.
template <class T>
struct first_traits {
    static constexpr int kind = -1;
    static constexpr int rank = 0;
};
template <int R>
struct first_traits<sycl::id<R>> {
    static constexpr int kind = 0;
    static constexpr int rank = R;
};
template <int R>
struct first_traits<GroupItem<R>> {
    static constexpr int kind = 1;
    static constexpr int rank = R;
};
template <>
struct first_traits<CustomOp&> {
    static constexpr int kind = 2;
    static constexpr int rank = 0;
};
template <>
struct first_traits<Host> {
    static constexpr int kind = 3;
    static constexpr int rank = 0;
};

template <class KB>
concept HasIO = requires {
    typename KB::Inputs;
    typename KB::Outputs;
};

template <class KB>
constexpr bool io_args_match() {
    using T = body_traits<KB>;
    return std::is_same_v<std::remove_cvref_t<typename T::a1>, typename KB::Inputs> &&
           std::is_same_v<std::remove_cvref_t<typename T::a2>, typename KB::Outputs>;
}

}  // namespace detail

template <class KB>
concept PlainBody = detail::HasIO<KB> && detail::body_traits<KB>::valid &&
                    detail::first_traits<typename detail::body_traits<KB>::a0>::kind == 0 &&
                    detail::body_traits<KB>::nargs == 3 && detail::io_args_match<KB>();

template <class KB>
concept GroupBody = detail::HasIO<KB> && detail::body_traits<KB>::valid &&
                    detail::first_traits<typename detail::body_traits<KB>::a0>::kind == 1 &&
                    (detail::body_traits<KB>::nargs == 3 || detail::body_traits<KB>::nargs == 4) &&
                    detail::io_args_match<KB>();

template <class KB>
concept HasScratch = GroupBody<KB> && detail::body_traits<KB>::nargs == 4 &&
                     requires { typename KB::Scratch; } &&
                     std::is_same_v<std::remove_cvref_t<typename detail::body_traits<KB>::a3>,
                                    typename KB::Scratch>;

template <class KB>
concept CustomOpBody = detail::HasIO<KB> && detail::body_traits<KB>::valid &&
                       detail::first_traits<typename detail::body_traits<KB>::a0>::kind == 2 &&
                       detail::body_traits<KB>::nargs == 3 && detail::io_args_match<KB>();

template <class KB>
concept HostBody = detail::HasIO<KB> && detail::body_traits<KB>::valid &&
                   detail::first_traits<typename detail::body_traits<KB>::a0>::kind == 3 &&
                   detail::body_traits<KB>::nargs == 3 && detail::io_args_match<KB>();

namespace detail {
template <class KB>
constexpr bool scratch_trivial() {
    if constexpr (HasScratch<KB>)
        return std::is_trivially_copyable_v<typename KB::Scratch>;
    else
        return true;
}
}  // namespace detail

template <class KB>
concept KernelBodyC =
    std::is_trivially_copyable_v<KB> && detail::HasIO<KB> &&
    std::is_trivially_copyable_v<typename KB::Inputs> &&
    std::is_trivially_copyable_v<typename KB::Outputs> && detail::scratch_trivial<KB>() &&
    (int(PlainBody<KB>) + int(GroupBody<KB>) + int(CustomOpBody<KB>) + int(HostBody<KB>) == 1);

/// The kernel rank, from the signature; 0 for custom-op and host bodies.
template <class KB>
inline constexpr int kernel_rank_v = detail::first_traits<typename detail::body_traits<KB>::a0>::rank;

/// The rank used for range-typed parameters: never 0.
template <class KB>
inline constexpr int launch_rank_v = kernel_rank_v<KB> > 0 ? kernel_rank_v<KB> : 1;

// -----------------------------------------------------------------------------
// Local layout (r02 section 4.1)
// -----------------------------------------------------------------------------

namespace detail {

template <class T>
struct is_local : std::false_type {};
template <DType DT, int R>
struct is_local<Local<DT, R>> : std::true_type {};

struct AnyRef {
    template <class T>
    operator T&() const;
};

template <class S, std::size_t... I>
constexpr bool init_with_n(std::index_sequence<I...>) {
    return requires { S{((void)I, AnyRef{})...}; };
}

/// The number of members of an aggregate (at most 8).
template <class S>
constexpr std::size_t field_count() {
    std::size_t best = 0;
    [&]<std::size_t... N>(std::index_sequence<N...>) {
        ((init_with_n<S>(std::make_index_sequence<N>{}) ? (best = N, 0) : 0), ...);
    }(std::make_index_sequence<9>{});
    return best;
}

template <class S, class F>
constexpr void for_each_member(S& s, F&& f) {
    constexpr std::size_t N = field_count<S>();
    static_assert(N <= 6, "cospan: a Scratch holds at most 6 Local members");
    if constexpr (N == 1) {
        auto& [a] = s;
        f(a);
    } else if constexpr (N == 2) {
        auto& [a, b] = s;
        f(a); f(b);
    } else if constexpr (N == 3) {
        auto& [a, b, c] = s;
        f(a); f(b); f(c);
    } else if constexpr (N == 4) {
        auto& [a, b, c, d] = s;
        f(a); f(b); f(c); f(d);
    } else if constexpr (N == 5) {
        auto& [a, b, c, d, e] = s;
        f(a); f(b); f(c); f(d); f(e);
    } else if constexpr (N == 6) {
        auto& [a, b, c, d, e, g] = s;
        f(a); f(b); f(c); f(d); f(e); f(g);
    }
}

/// The sole writer of Local's offset and pointer (friendship).
struct ScratchLayout {
    /// Host: lay every Local of `s` out (aligned, summed); returns the total
    /// bytes, rounded up to 16.
    template <class S>
    static uint32_t layout(S& s) {
        std::size_t cur = 0;
        for_each_member(s, [&](auto& l) {
            using L = std::remove_cvref_t<decltype(l)>;
            static_assert(is_local<L>::value, "cospan: a Scratch may hold only Local members");
            using V = typename L::value_type;
            cur     = (cur + alignof(V) - 1) / alignof(V) * alignof(V);
            l.offset_ = cur;
            cur += l.size() * sizeof(V);
        });
        return static_cast<uint32_t>((cur + 15) / 16 * 16);
    }

    /// Device: Local.ptr_ = local_base + offset_.
    template <class S>
    static void fixup(S& s, std::byte* local_base) {
        for_each_member(s, [&](auto& l) {
            using L = std::remove_cvref_t<decltype(l)>;
            using V = typename L::value_type;
            l.ptr_  = reinterpret_cast<V*>(local_base + l.offset_);
        });
    }
};

template <class S>
uint32_t layout_scratch(S& s) {
    return ScratchLayout::layout(s);
}

template <class S>
void fixup_scratch(S& s, std::byte* local_base) {
    ScratchLayout::fixup(s, local_base);
}

/// A rank-R range in the TRAILING dimensions of a rank-3 range (leading 1s).
template <int R>
sycl::range<3> pad3(const sycl::range<R>& r) {
    return lift_range<R>(r);
}

// -----------------------------------------------------------------------------
// StageArgs: the argument struct of a component (body, inputs, outputs, scratch, launch parameters)
// -----------------------------------------------------------------------------

struct NoScratch {};

template <class KB>
struct scratch_of {
    using type = NoScratch;
};
template <class KB>
    requires HasScratch<KB>
struct scratch_of<KB> {
    using type = typename KB::Scratch;
};
template <class KB>
using scratch_t = typename scratch_of<KB>::type;

/// The Scratch argument of a body that has none.
template <class KB>
scratch_t<KB> NoScratchValue() {
    return scratch_t<KB>{};
}

/// Everything a stage function reads of its component: trivially copyable, one element in a device array the component
/// keeps, copied in from a host member every forward. `group_range` (the stage's own group range, rank 3, leading 1s) is set
/// by the group launcher once the group's work-group shape is known; 0 in every dimension: the stage is empty.
template <class KB>
struct StageArgs {
    StageArgs(typename KB::Inputs i, typename KB::Outputs o, KB b, scratch_t<KB> s) : in{i}, out{o}, body{b}, scratch{s} {}

    sycl::range<3> range{1, 1, 1};        ///< the logical iteration space, in the trailing dimensions
    sycl::range<3> group_range{1, 1, 1};  ///< the stage's own group range
    std::uint64_t  flat_extent   = 0;     ///< the product of range
    std::uint32_t  scratch_bytes = 0;     ///< the laid-out Scratch, rounded to 16
    typename KB::Inputs  in;
    typename KB::Outputs out;
    KB                   body;
    scratch_t<KB>        scratch;
};

/// The stage-local view of the physical item: the logical group coordinate comes with the StageItem.
struct ItemFixup {
    template <int R>
    static GroupItem<R> make(const StageItem& si, const StageGeometry& g, const SegmentInfo* tbl, const sycl::range<3>& own,
                             std::uint32_t free_off) {
        sycl::range<3> gr = own;
        if (g.range_mode == static_cast<std::uint32_t>(RangeMode::whole))
            gr = sycl::range<3>{tbl[g.segment].range[0], tbl[g.segment].range[1], tbl[g.segment].range[2]};
        return GroupItem<R>{si.item, tail_id<R>(si.coord), tail_range<R>(gr), g.segment, g.segments, tbl,
                            static_cast<std::uint32_t*>(g.ticket_state), si.local + g.local_offset, free_off, g.local_bytes};
    }
};

}  // namespace detail

// -----------------------------------------------------------------------------
// The placeholder, the stage function and the launch (design review, 11.6)
// -----------------------------------------------------------------------------

/// The placeholder: ONE for every launch (the launch is rank 3); a declaration only, the JIT replaces it with the indexed call
/// sequence.
[[clang::noinline]] SYCL_EXTERNAL void run_stages(detail::StageItem, const detail::StageGeometry*, const void* const*, int);

/// ONE stage function per KernelBody type, at the physical rank 3. Every instantiation has the same signature: (the native
/// item, the geometry array, the argument-pointer array, the stage index). All the conditions below are group-uniform.
template <class KB>
SYCL_EXTERNAL void stage_fn(detail::StageItem si, const detail::StageGeometry* geometry, const void* const* args, int k) {
    constexpr int R = kernel_rank_v<KB>;
    const detail::StageGeometry& g = geometry[k];
    const auto&                  a = *static_cast<const detail::StageArgs<KB>*>(args[k]);

    // This stage's groups: those of its segment; an empty stage runs nothing; range_mode::own also stops at the stage's own
    // range in ANY dimension.
    if (si.segment != g.segment) return;
    const sycl::range<3>& mine = a.group_range;
    if (mine[0] * mine[1] * mine[2] == 0) return;
    if (g.range_mode == static_cast<std::uint32_t>(detail::RangeMode::own))
        for (int d = 0; d < 3; ++d)
            if (si.coord[d] >= mine[d]) return;

    auto in   = a.in;
    auto out  = a.out;
    auto body = a.body;

    if constexpr (PlainBody<KB>) {
        // The stage's global index per dimension; the tail mask is here, the body sees only its range.
        sycl::id<3> gid;
        for (int d = 0; d < 3; ++d) {
            gid[d] = si.coord[d] * si.item.get_local_range(d) + si.item.get_local_id(d);
            if (gid[d] >= a.range[d]) return;
        }
        body(detail::tail_id<R>(gid), in, out);
    } else if constexpr (HasScratch<KB>) {
        const detail::SegmentInfo* tbl = detail::segment_table(geometry);
        GroupItem<R>               gi  = detail::ItemFixup::make<R>(si, g, tbl, mine, a.scratch_bytes);
        auto                       s   = a.scratch;
        detail::fixup_scratch(s, si.local + g.local_offset);  // this stage's own slice of the local block
        body(gi, in, out, s);
    } else {
        const detail::SegmentInfo* tbl = detail::segment_table(geometry);
        GroupItem<R>               gi  = detail::ItemFixup::make<R>(si, g, tbl, mine, a.scratch_bytes);
        body(gi, in, out);
    }
}

namespace detail {

using JitId = sycl::AdaptiveCpp_jit::dynamic_function_id;

/// The signature of the placeholder and of every stage function, as the type-erased id types need it.
using run_stages_fn_t = sycl::AdaptiveCpp_jit::dynamic_function<void, StageItem, const StageGeometry*, const void* const*, int>;
using stage_def_t     = sycl::AdaptiveCpp_jit::dynamic_function_definition<void, StageItem, const StageGeometry*, const void* const*, int>;

/// The id of stage_fn<KB>: what a group puts in its call sequence. Obtained once per body type, through a function-local
/// static; the component reaches it through a virtual.
template <class KB>
JitId stage_definition_id() {
    static const JitId id = stage_def_t{&stage_fn<KB>}.id();
    return id;
}

/// The id of the placeholder.
inline JitId run_stages_id() { return run_stages_fn_t{&run_stages}.id(); }

/// The dynamic_function_config of a stage sequence: ONE per distinct sequence of stage definitions, created the first time
/// and NEVER destroyed. A config must outlive every kernel that uses it (the runtime reads it on a worker thread after the
/// submission, possibly before the kernel has been compiled), and a transient group that used it can be deleted at any
/// moment, so the cache is process-wide and leaked on purpose. The key is the definitions' symbol names (an id's handle is
/// its name).
inline sycl::AdaptiveCpp_jit::dynamic_function_config& group_config(const std::vector<JitId>& defs) {
    using Config = sycl::AdaptiveCpp_jit::dynamic_function_config;
    struct Cache {
        std::mutex                                                  mu;
        std::map<std::vector<std::string>, std::unique_ptr<Config>> map;
    };
    static Cache* const cache = new Cache;  // never destroyed

    std::vector<std::string> names;
    names.reserve(defs.size());
    for (const JitId& d : defs) names.emplace_back(reinterpret_cast<const char*>(d.get_handle()));

    const std::lock_guard<std::mutex> lock{cache->mu};
    std::unique_ptr<Config>&          slot = cache->map[std::move(names)];
    if (!slot) {
        slot = std::make_unique<Config>();
        slot->define_as_indexed_call_sequence(run_stages_id(), defs, /*index_param=*/3);
    }
    return *slot;
}

/// Default work-group shape per rank, in the trailing dimensions (implementation-tunable, r02 section 4.2).
template <int R>
sycl::range<3> default_wg() {
    if constexpr (R == 1)
        return sycl::range<3>{1, 1, 128};
    else if constexpr (R == 2)
        return sycl::range<3>{1, 8, 8};
    else
        return sycl::range<3>{4, 4, 4};
}

/// Submit the ONE kernel of a group on the unordered queue `q` for `dev`: ordered ONLY by `deps` (section 3). The launch is
/// always an nd_range<3>. `geometry` and `args` are device arrays the group owns; the kernel passes stage 0 as the index
/// placeholder, the configuration replaces it by each stage's position. Returns its event.
inline sycl::event launch_group(sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                                sycl::AdaptiveCpp_jit::dynamic_function_config& cfg, const sycl::nd_range<3>& nd,
                                const StageGeometry* geometry, const void* const* args, std::size_t local_bytes) {
    auto cgf = [&](sycl::handler& h) {
        sycl::local_accessor<std::byte, 1> local{sycl::range<1>{local_bytes ? local_bytes : 1}, h};
        h.parallel_for(nd, cfg.apply([=](sycl::nd_item<3> it) {
            std::byte* lb = local.template get_multi_ptr<sycl::access::decorated::no>().get();

            // Ticket mode (the group has a boundary or the property): the work-group's leader takes the next ticket and
            // broadcasts it; the ticket, not the hardware id, says which segment and group of the segment this work-group is
            // (segment-major, row-major, whatever the strategy). A group past the logical total exits before anything is
            // written. Without ticket state the hardware id decides (locate_physical): one load and no counters.
            const StageGeometry& g0   = geometry[0];
            std::uint32_t*       ts   = static_cast<std::uint32_t*>(g0.ticket_state);
            const SegmentInfo*   tbl  = segment_table(geometry);
            const std::size_t    nseg = g0.segments;
            LogicalGroup         lg;
            std::uint32_t        ticket = 0;
            if (ts != nullptr) {
                if (it.get_local_linear_id() == 0) {
                    TicketRef counter{ts[0]};
                    ticket = counter.fetch_add(1u, sycl::memory_order::relaxed);
                }
                ticket = sycl::group_broadcast(it.get_group(), ticket);
                lg     = locate_linear(tbl, nseg, ticket, g0.total_groups);
                if (!lg.valid) return;
                if (tbl[lg.segment].boundary == static_cast<std::uint64_t>(Boundary::global_barrier)) ticket_wait_segments(ts, tbl, lg.segment, it);
            } else {
                lg = locate_physical(g0, tbl, it);
                if (!lg.valid) return;
            }

            const StageItem si{it, lb, lg.segment, lg.coord};
            run_stages(si, geometry, args, 0);

            // Completion: every stage of this group's segment has run (the other stages skipped this group). A group
            // barrier, a device-scope release fence, then the leader's release atomics: the group's flag, and its segment's
            // completed counter.
            if (ts != nullptr) {
                sycl::group_barrier(it.get_group());
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                if (it.get_local_linear_id() == 0) {
                    TicketRef flag{ts[1 + nseg + ticket]};
                    flag.store(1u, sycl::memory_order::release);
                    TicketRef done{ts[1 + lg.segment]};
                    done.fetch_add(1u, sycl::memory_order::release);
                }
            }
        }));
    };
    return q.submit(dev, deps, cgf);
}

}  // namespace detail
}  // namespace naga::cospan
