#pragma once

// =============================================================================
// naga/cospan/kernel.hpp — kernel machinery (design review, sections 4 and 11.6)
// =============================================================================
//
// The KernelBody contract (concepts read off the call operator's signature), GroupItem, LocalMem, the per-component
// argument struct (StageArgs) and the per-stage geometry entry (StageGeometry), the STAGE FUNCTION stage_fn<KB> (one per
// body type), the two placeholders the JIT replaces (run_stages for group-range stages, run_index_stages for index-range
// stages) and the two launch helpers (launch_group, launch_range).
// All header-only: placeholders, outer kernels and stage functions instantiate in the consumer's translation unit.
//
// TWO KINDS OF KERNEL BODY. An INDEX-RANGE body takes a sycl::id<R> and runs once per index of its range. A GROUP-RANGE
// body takes a GroupItem<R> and runs once per work-item of its range, which is a whole number of work-groups (a multiple of
// the work-group shape in every dimension; the stage's group range is range / shape); it may take a LocalMem as its fourth
// parameter.
//
// THE LAUNCH PATH (every index-range and group-range component, bare or in a fused group; custom-op and host bodies keep
// their own). ONE generic rank-3 kernel calls the declared dynamic function
//     run_stages(StageItem, const StageGeometry* geometry, const void* const* args, int stage)         (group-range)
//     run_index_stages(IndexStageItem, const StageGeometry* geometry, const void* const* args, int stage)  (index-range)
// whose definition is an INDEXED CALL SEQUENCE of stage functions: stage k of the sequence is called with the same
// arguments and `stage` = k (fork: define_as_indexed_call_sequence(&run_stages, {ids...}, 3)). A stage function reads
// geometry[k] and args[k] (its own StageArgs, a one-element device array every component keeps), builds the item
// wrapper (GroupItem for a group-range stage) and calls the user's body. A group-range launch is an nd_range<3> of
// work-groups; an index-range launch is a range<3> of single indices (a launch unit of one index). The configuration of a
// sequence lives in a process-wide cache (group_config), created once per distinct sequence and never destroyed: the
// runtime reads a configuration on a worker thread after the submission, and a transient group can be deleted before its
// kernel has even been compiled.

#include <algorithm>
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
#include <tuple>
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

/// Local memory of a segment (group-range stages only). partitioned (the default): every stage gets its own slice, the
/// segment's size is the sum. shared: every stage gets the same block, the size is the max.
namespace local_mem {
struct partitioned {};
struct shared {};
}  // namespace local_mem

/// Range of a segment. own (the default): a stage runs only for its first N units (work-groups, or indices for an
/// index-range stage) of the segment's block and skips the rest. whole: every stage runs for every unit of the block.
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

/// The native item of a group-range stage function: the physical (rank-3) nd_item, the launch's local block (the one thing a
/// stage function cannot obtain by itself), and where the work-group sits LOGICALLY: its segment and its group coordinate in
/// that segment's range (rank 3, the stage's rank-R id is the trailing part).
struct StageItem {
    sycl::nd_item<3> item;
    std::byte*       local;
    std::uint32_t    segment;
    sycl::id<3>      coord;
};

/// The native item of an index-range stage function: where the work-item sits LOGICALLY, its segment and its index in that
/// segment's range (rank 3, the stage's rank-R id is the trailing part). An index-range launch has no work-group, no nd_item
/// and no local memory.
struct IndexStageItem {
    std::uint32_t segment;
    sycl::id<3>   coord;
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

/// The hardware-direct mapping (no ticket): flatten_and_split reads the unit number along the last dimension; pad_and_split
/// finds the segment by its offset along the last dimension; padded takes the segment from dimension 0. A unit outside its
/// segment's range (padding) is not valid. `phys` is the unit's physical id: a work-group's id in a group-range launch, an
/// item's id in an index-range launch (whose unit is one index).
inline LogicalGroup locate_physical(const StageGeometry& g0, const SegmentInfo* tbl, const sycl::id<3>& phys) {
    const std::size_t S = g0.segments;
    const std::uint64_t d0 = phys[0], d1 = phys[1], d2 = phys[2];
    if (g0.split == static_cast<std::uint32_t>(Split::flatten)) return locate_linear(tbl, S, d2, g0.total_groups);
    LogicalGroup r;
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

/// The same for a work-group of a group-range launch: its physical group id.
inline LogicalGroup locate_physical(const StageGeometry& g0, const SegmentInfo* tbl, const sycl::nd_item<3>& it) {
    return locate_physical(g0, tbl, sycl::id<3>{it.get_group(0), it.get_group(1), it.get_group(2)});
}

struct ItemFixup;
}  // namespace detail

template <int R>
struct GroupItem;

template <class... Arrays>
struct LocalMem;

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
              std::uint32_t segments, const detail::SegmentInfo* table, std::uint32_t* ticket_state, std::byte* base)
        : it_{it}, group_id_{gid}, group_range_{grange}, segment_{segment}, segments_{segments}, table_{table},
          ticket_state_{ticket_state}, base_{base} {}

    sycl::nd_item<3>           it_;
    sycl::id<R>                group_id_;     // within the segment
    sycl::range<R>             group_range_;  // the stage's group range (the segment's under range_mode::whole)
    std::uint32_t              segment_;
    std::uint32_t              segments_;
    const detail::SegmentInfo* table_;        // the segment table
    std::uint32_t*             ticket_state_; // null: not in ticket mode
    std::byte*                 base_;         // this stage's slice of the launch's local block
    friend struct detail::ItemFixup;
    template <class... Arrays>
    friend struct LocalMem;
};

// -----------------------------------------------------------------------------
// LocalMem: a group-range stage's local memory
// -----------------------------------------------------------------------------

namespace detail {

template <class T>
struct is_local_array : std::false_type {};
template <DType DT, int R>
struct is_local_array<LocalArray<DT, R>> : std::true_type {};

constexpr std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

/// The aggregate of views that LocalMem::get returns, one member per array (v0, v1, ...), so that
/// `auto [a, b] = lm.get(it);` works. One to six arrays.
template <class... Vs>
struct LocalViews;
template <class A>
struct LocalViews<A> {
    A v0;
};
template <class A, class B>
struct LocalViews<A, B> {
    A v0;
    B v1;
};
template <class A, class B, class C>
struct LocalViews<A, B, C> {
    A v0;
    B v1;
    C v2;
};
template <class A, class B, class C, class D>
struct LocalViews<A, B, C, D> {
    A v0;
    B v1;
    C v2;
    D v3;
};
template <class A, class B, class C, class D, class E>
struct LocalViews<A, B, C, D, E> {
    A v0;
    B v1;
    C v2;
    D v3;
    E v4;
};
template <class A, class B, class C, class D, class E, class F>
struct LocalViews<A, B, C, D, E, F> {
    A v0;
    B v1;
    C v2;
    D v3;
    E v4;
    F v5;
};

}  // namespace detail

/// The local memory of a group-range stage, declared by the arrays' dtypes and ranks and carrying their shapes (one
/// sycl::range per array, given by cs::local_desc in define_forward). It is the fourth parameter of compute_forward:
///     static void compute_forward(cs::GroupItem<1> it, const Inputs&, const Outputs&,
///                                 cs::LocalMem<cs::LocalArray<cs::DType::f32, 1>, cs::LocalArray<cs::DType::i32, 2>> lm) {
///         auto [a, b] = lm.get(it);
///     }
/// The arrays sit one after the other in declaration order in the stage's slice of the launch's local block; each starts at
/// a multiple of the alignment of its element type, and the slice is exactly as large as the arrays need. Trivially copyable.
template <class... Arrays>
struct LocalMem {
    static_assert(sizeof...(Arrays) >= 1 && sizeof...(Arrays) <= 6, "cospan: a LocalMem holds one to six arrays");
    static_assert((detail::is_local_array<Arrays>::value && ...), "cospan: a LocalMem holds cs::LocalArray types");
    static_assert(std::max({alignof(typename Arrays::value_type)...}) <= 16, "cospan: a LocalArray element is at most 16-byte aligned");

    static constexpr std::size_t count = sizeof...(Arrays);

    /// The declaration of array `I`.
    template <std::size_t I>
    using array_t = std::tuple_element_t<I, std::tuple<Arrays...>>;

    /// Every array empty.
    LocalMem() : LocalMem(detail::filled_range<Arrays::rank>(0)...) {}
    /// One shape per array, in declaration order.
    explicit LocalMem(sycl::range<Arrays::rank>... shapes) : shapes_{detail::lift_range<Arrays::rank>(shapes)...} {}

    /// The shape of array `I`.
    template <std::size_t I>
    sycl::range<array_t<I>::rank> shape() const {
        return detail::tail_range<array_t<I>::rank>(shapes_[I]);
    }

    /// The bytes of the stage's slice: the arrays' offsets rounded up to their element alignment, plus their sizes.
    std::size_t bytes() const {
        std::size_t cur = 0;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((cur = detail::align_up(cur, alignof(typename array_t<I>::value_type)) +
                    detail::range_size<array_t<I>::rank>(shape<I>()) * sizeof(typename array_t<I>::value_type)),
             ...);
        }(std::make_index_sequence<count>{});
        return cur;
    }

    /// The strictest alignment of the arrays' element types.
    static constexpr std::size_t alignment() { return std::max({alignof(typename Arrays::value_type)...}); }

    /// COLLECTIVE-FREE, per work-item: the views of the arrays in this stage's slice of the launch's local block, as an
    /// aggregate of typed views (LocalView) in declaration order. The offsets are recomputed from the carried shapes.
    template <int R>
    auto get(const GroupItem<R>& it) const {
        std::byte* const base = it.base_;
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            std::array<std::size_t, count> off{};
            std::size_t                    cur = 0;
            ((cur = detail::align_up(cur, alignof(typename array_t<I>::value_type)), off[I] = cur,
              cur += detail::range_size<array_t<I>::rank>(shape<I>()) * sizeof(typename array_t<I>::value_type)),
             ...);
            return detail::LocalViews<typename array_t<I>::view_type...>{typename array_t<I>::view_type{
                reinterpret_cast<typename array_t<I>::value_type*>(base + off[I]), shape<I>()}...};
        }(std::make_index_sequence<count>{});
    }

private:
    std::array<sycl::range<3>, count> shapes_;
};

static_assert(std::is_trivially_copyable_v<LocalMem<LocalArray<DType::f32, 1>>>);

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
    bool                          group_range = false;  ///< a group-range stage (else index-range)
    std::size_t                   flat_extent = 0;      ///< the stage's iteration space (0: nothing to run)
    std::size_t                   local_bytes = 0;      ///< the stage's slice: exactly what its LocalMem arrays need
    std::size_t                   local_align = 1;      ///< the strictest alignment among its LocalMem arrays
    std::optional<sycl::range<3>> wg;                   ///< the work-group shape of a group-range stage (set for every one)
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

/// kind: 0 index-range (id<R>), 1 group-range (GroupItem<R>), 2 custom op, 3 host.
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

namespace detail {
template <class T>
struct is_local_mem : std::false_type {};
template <class... As>
struct is_local_mem<::naga::cospan::LocalMem<As...>> : std::true_type {};
}  // namespace detail

/// An index-range body: `void operator()(sycl::id<R>, Inputs, Outputs) const`.
template <class KB>
concept IndexRangeBody = detail::HasIO<KB> && detail::body_traits<KB>::valid &&
                         detail::first_traits<typename detail::body_traits<KB>::a0>::kind == 0 &&
                         detail::body_traits<KB>::nargs == 3 && detail::io_args_match<KB>();

/// A group-range body: `void operator()(GroupItem<R>, Inputs, Outputs[, LocalMem<...>]) const`.
template <class KB>
concept GroupRangeBody =
    detail::HasIO<KB> && detail::body_traits<KB>::valid &&
    detail::first_traits<typename detail::body_traits<KB>::a0>::kind == 1 &&
    (detail::body_traits<KB>::nargs == 3 ||
     (detail::body_traits<KB>::nargs == 4 &&
      detail::is_local_mem<std::remove_cvref_t<typename detail::body_traits<KB>::a3>>::value)) &&
    detail::io_args_match<KB>();

/// A group-range body with a trailing LocalMem parameter.
template <class KB>
concept HasLocalMem = GroupRangeBody<KB> && detail::body_traits<KB>::nargs == 4;

template <class KB>
concept CustomOpBody = detail::HasIO<KB> && detail::body_traits<KB>::valid &&
                       detail::first_traits<typename detail::body_traits<KB>::a0>::kind == 2 &&
                       detail::body_traits<KB>::nargs == 3 && detail::io_args_match<KB>();

template <class KB>
concept HostBody = detail::HasIO<KB> && detail::body_traits<KB>::valid &&
                   detail::first_traits<typename detail::body_traits<KB>::a0>::kind == 3 &&
                   detail::body_traits<KB>::nargs == 3 && detail::io_args_match<KB>();

template <class KB>
concept KernelBodyC =
    std::is_trivially_copyable_v<KB> && detail::HasIO<KB> &&
    std::is_trivially_copyable_v<typename KB::Inputs> &&
    std::is_trivially_copyable_v<typename KB::Outputs> &&
    (int(IndexRangeBody<KB>) + int(GroupRangeBody<KB>) + int(CustomOpBody<KB>) + int(HostBody<KB>) == 1);

/// The kernel rank, from the signature; 0 for custom-op and host bodies.
template <class KB>
inline constexpr int kernel_rank_v = detail::first_traits<typename detail::body_traits<KB>::a0>::rank;

/// The rank used for range-typed parameters: never 0.
template <class KB>
inline constexpr int launch_rank_v = kernel_rank_v<KB> > 0 ? kernel_rank_v<KB> : 1;

// -----------------------------------------------------------------------------
// StageArgs: the argument struct of a component (body, inputs, outputs, local memory, launch parameters)
// -----------------------------------------------------------------------------

namespace detail {

/// A rank-R range in the TRAILING dimensions of a rank-3 range (leading 1s).
template <int R>
sycl::range<3> pad3(const sycl::range<R>& r) {
    return lift_range<R>(r);
}

/// The LocalMem value of a body that has none.
struct NoLocalMem {};

template <class KB>
struct local_mem_of {
    using type = NoLocalMem;
};
template <class KB>
    requires HasLocalMem<KB>
struct local_mem_of<KB> {
    using type = std::remove_cvref_t<typename body_traits<KB>::a3>;
};
template <class KB>
using local_mem_t = typename local_mem_of<KB>::type;

/// Everything a stage function reads of its component: trivially copyable, one element in a device array the component
/// keeps, copied in from a host member every forward. `units` is the stage's own range in launch units (rank 3, leading 1s):
/// work-groups for a group-range stage, indices for an index-range stage; 0 in every dimension: the stage is empty.
template <class KB>
struct StageArgs {
    StageArgs(typename KB::Inputs i, typename KB::Outputs o, KB b, local_mem_t<KB> l) : in{i}, out{o}, body{b}, local{l} {}

    sycl::range<3> range{1, 1, 1};  ///< the logical iteration space, in the trailing dimensions
    sycl::range<3> units{1, 1, 1};  ///< the stage's own range in launch units
    std::uint64_t  flat_extent = 0; ///< the product of range
    typename KB::Inputs  in;
    typename KB::Outputs out;
    KB                   body;
    local_mem_t<KB>      local;
};

/// The item a stage function receives: the group-range item (nd_item, local block, segment, group) or the index-range item
/// (segment, index).
template <class KB>
using stage_item_t = std::conditional_t<IndexRangeBody<KB>, IndexStageItem, StageItem>;

/// The stage-local view of the physical item: the logical group coordinate comes with the StageItem.
struct ItemFixup {
    template <int R>
    static GroupItem<R> make(const StageItem& si, const StageGeometry& g, const SegmentInfo* tbl, const sycl::range<3>& own) {
        sycl::range<3> gr = own;
        if (g.range_mode == static_cast<std::uint32_t>(RangeMode::whole))
            gr = sycl::range<3>{tbl[g.segment].range[0], tbl[g.segment].range[1], tbl[g.segment].range[2]};
        return GroupItem<R>{si.item, tail_id<R>(si.coord), tail_range<R>(gr), g.segment, g.segments, tbl,
                            static_cast<std::uint32_t*>(g.ticket_state), si.local + g.local_offset};
    }
};

}  // namespace detail

// -----------------------------------------------------------------------------
// The placeholders, the stage function and the launches
// -----------------------------------------------------------------------------

/// The placeholders: ONE per kind of launch (the launch is rank 3); declarations only, the JIT replaces each with the indexed
/// call sequence of its stages.
[[clang::noinline]] SYCL_EXTERNAL void run_stages(detail::StageItem, const detail::StageGeometry*, const void* const*, int);
[[clang::noinline]] SYCL_EXTERNAL void run_index_stages(detail::IndexStageItem, const detail::StageGeometry*, const void* const*, int);

/// ONE stage function per KernelBody type, at the physical rank 3. Every instantiation of a kind has the same signature: (the
/// native item, the geometry array, the argument-pointer array, the stage index). All the conditions below are
/// group-uniform (index-range stages: uniform per index).
template <class KB>
SYCL_EXTERNAL void stage_fn(detail::stage_item_t<KB> si, const detail::StageGeometry* geometry, const void* const* args, int k) {
    constexpr int R = kernel_rank_v<KB>;
    const detail::StageGeometry& g = geometry[k];
    const auto&                  a = *static_cast<const detail::StageArgs<KB>*>(args[k]);

    // This stage's units: those of its segment; an empty stage runs nothing; range_mode::own also stops at the stage's own
    // range in ANY dimension.
    if (si.segment != g.segment) return;
    const sycl::range<3>& mine = a.units;
    if (mine[0] * mine[1] * mine[2] == 0) return;
    if (g.range_mode == static_cast<std::uint32_t>(detail::RangeMode::own))
        for (int d = 0; d < 3; ++d)
            if (si.coord[d] >= mine[d]) return;

    auto in   = a.in;
    auto out  = a.out;
    auto body = a.body;

    if constexpr (IndexRangeBody<KB>) {
        body(detail::tail_id<R>(si.coord), in, out);
    } else if constexpr (HasLocalMem<KB>) {
        const detail::SegmentInfo* tbl = detail::segment_table(geometry);
        GroupItem<R>               gi  = detail::ItemFixup::make<R>(si, g, tbl, mine);
        auto                       lm  = a.local;
        body(gi, in, out, lm);
    } else {
        const detail::SegmentInfo* tbl = detail::segment_table(geometry);
        GroupItem<R>               gi  = detail::ItemFixup::make<R>(si, g, tbl, mine);
        body(gi, in, out);
    }
}

namespace detail {

using JitId = sycl::AdaptiveCpp_jit::dynamic_function_id;

/// The signatures of the placeholders and of the stage functions, as the type-erased id types need them.
using run_stages_fn_t       = sycl::AdaptiveCpp_jit::dynamic_function<void, StageItem, const StageGeometry*, const void* const*, int>;
using stage_def_t           = sycl::AdaptiveCpp_jit::dynamic_function_definition<void, StageItem, const StageGeometry*, const void* const*, int>;
using run_index_stages_fn_t = sycl::AdaptiveCpp_jit::dynamic_function<void, IndexStageItem, const StageGeometry*, const void* const*, int>;
using index_stage_def_t     = sycl::AdaptiveCpp_jit::dynamic_function_definition<void, IndexStageItem, const StageGeometry*, const void* const*, int>;

/// The id of stage_fn<KB>: what a group puts in its call sequence. Obtained once per body type, through a function-local
/// static; the component reaches it through a virtual.
template <class KB>
JitId stage_definition_id() {
    if constexpr (IndexRangeBody<KB>) {
        static const JitId id = index_stage_def_t{&stage_fn<KB>}.id();
        return id;
    } else {
        static const JitId id = stage_def_t{&stage_fn<KB>}.id();
        return id;
    }
}

/// The ids of the placeholders.
inline JitId run_stages_id() { return run_stages_fn_t{&run_stages}.id(); }
inline JitId run_index_stages_id() { return run_index_stages_fn_t{&run_index_stages}.id(); }

/// The dynamic_function_config of a stage sequence: ONE per distinct sequence of stage definitions, created the first time
/// and NEVER destroyed. A config must outlive every kernel that uses it (the runtime reads it on a worker thread after the
/// submission, possibly before the kernel has been compiled), and a transient group that used it can be deleted at any
/// moment, so the cache is process-wide and leaked on purpose. The key is the definitions' symbol names (an id's handle is
/// its name); `placeholder` is the id of the placeholder the sequence replaces (run_stages_id() or run_index_stages_id(); the
/// definitions of one kind never share a name with the other kind's).
inline sycl::AdaptiveCpp_jit::dynamic_function_config& group_config(const std::vector<JitId>& defs, const JitId& placeholder) {
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
        slot->define_as_indexed_call_sequence(placeholder, defs, /*index_param=*/3);
    }
    return *slot;
}

/// The element of a launch's local block: 16-byte aligned, so the block starts aligned for every element type a LocalMem holds.
struct alignas(16) LocalBlock {
    std::byte bytes[16];
};

/// Submit the ONE kernel of a group-range group on the unordered queue `q` for `dev`: ordered ONLY by `deps` (section 3). The
/// launch is an nd_range<3>. `geometry` and `args` are device arrays the group owns; the kernel passes stage 0 as the index
/// placeholder, the configuration replaces it by each stage's position. `local_bytes` is the size of the local block. Returns
/// its event.
inline sycl::event launch_group(sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                                sycl::AdaptiveCpp_jit::dynamic_function_config& cfg, const sycl::nd_range<3>& nd,
                                const StageGeometry* geometry, const void* const* args, std::size_t local_bytes) {
    auto cgf = [&](sycl::handler& h) {
        const std::size_t                  blocks = (local_bytes + sizeof(LocalBlock) - 1) / sizeof(LocalBlock);
        sycl::local_accessor<LocalBlock, 1> local{sycl::range<1>{blocks ? blocks : 1}, h};
        h.parallel_for(nd, cfg.apply([=](sycl::nd_item<3> it) {
            std::byte* lb = reinterpret_cast<std::byte*>(local.template get_multi_ptr<sycl::access::decorated::no>().get());

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

/// Submit the ONE kernel of an index-range group on the unordered queue `q` for `dev`: ordered ONLY by `deps` (section 3). The
/// launch is a range<3> of single indices (`phys`, laid out by the range_split strategy); there is no work-group, no local
/// memory and no ticket state. Each item is located to its segment and index by locate_physical; an item outside every
/// segment's range returns. `geometry` and `args` are device arrays the group owns. Returns its event.
inline sycl::event launch_range(sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                                sycl::AdaptiveCpp_jit::dynamic_function_config& cfg, const sycl::range<3>& phys,
                                const StageGeometry* geometry, const void* const* args) {
    auto cgf = [&](sycl::handler& h) {
        h.parallel_for(phys, cfg.apply([=](sycl::id<3> idx) {
            const StageGeometry& g0  = geometry[0];
            const SegmentInfo*   tbl = segment_table(geometry);
            const LogicalGroup   lg  = locate_physical(g0, tbl, idx);
            if (!lg.valid) return;
            const IndexStageItem si{lg.segment, lg.coord};
            run_index_stages(si, geometry, args, 0);
        }));
    };
    return q.submit(dev, deps, cgf);
}

}  // namespace detail
}  // namespace naga::cospan
