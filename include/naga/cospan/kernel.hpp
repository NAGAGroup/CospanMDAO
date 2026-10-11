#pragma once

// =============================================================================
// naga/cospan/kernel.hpp — kernel machinery (design review, sections 4 and 11.6)
// =============================================================================
//
// The KernelBody contract (concepts read off the call operator's signature), GroupItem, the Local layout, the
// per-component argument struct (StageArgs) and the per-stage geometry entry (StageGeometry), the STAGE FUNCTION
// stage_fn<KB> (one per body type), the placeholder run_stages_rN the JIT replaces, and the group launch helper.
// All header-only: placeholders, outer kernels and stage functions instantiate in the consumer's translation unit.
//
// THE LAUNCH PATH (every plain-body and group-body component, bare or in a fused group; custom-op and host bodies keep
// their own). ONE generic kernel per rank calls the declared dynamic function
//     run_stages_rN(StageItem<R>, const StageGeometry* geometry, const void* const* args, int stage)
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

/// One entry per stage, in a device array the group owns and refills every forward. Followed (after `stages` entries) by
/// the segment table: one std::uint64_t per segment, that segment's block of groups.
struct StageGeometry {
    std::uint64_t group_count    = 0;  ///< this stage's own number of groups
    std::uint64_t group_base     = 0;  ///< the groups before this stage's segment
    std::uint64_t segment_groups = 0;  ///< this segment's block (the largest group count of its stages)
    void*         ticket_state   = nullptr;  ///< the group's ticket state; null: the group neither has group_waits_enabled nor a global_barrier (no ticket mode)
    std::uint32_t segment        = 0;
    std::uint32_t segments       = 0;
    std::uint32_t stages         = 0;  ///< the number of stage entries (where the segment table starts)
    std::uint32_t local_offset   = 0;  ///< bytes into the launch's local block
    std::uint32_t local_bytes    = 0;  ///< this stage's slice
    std::uint32_t range_mode     = 0;  ///< RangeMode
};
static_assert(std::is_trivially_copyable_v<StageGeometry>);
static_assert(sizeof(StageGeometry) % 8 == 0);

/// The native item of a stage function: the nd_item, the launch's local block (the one thing a stage function cannot
/// obtain by itself) and the work-group's linear index in the launch: the hardware group id, or in ticket mode the ticket
/// the group took (tickets are handed out in increasing order, so ticket order is segment order).
template <int R>
struct StageItem {
    sycl::nd_item<R> item;
    std::byte*       local;
    std::size_t      group;
};

// ---- ticket state ---------------------------------------------------------------------------------------------
// A device array of std::uint32_t the group owns (ticket mode only), reset every forward:
//   [0]            the ticket counter (fetch_add by each work-group's leader at its start)
//   [1, 1+S)       per segment, the number of its groups that have FINISHED every stage of the segment
//   [1+S, 1+S+T)   per ticket, 1 once that group has finished (T = the launch's groups)
// Both waits are COLLECTIVE (the leader spins with acquire loads, then a group barrier releases the group) and are
// deadlock-free without co-residency only because a group waits for EARLIER tickets alone: every earlier ticket is held by
// a group that has started (resident or finished).
using TicketRef = sycl::atomic_ref<std::uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>;

/// Wait for the group with ticket (blocks[0] + ... + blocks[s-1]) + g, i.e. group g of segment s.
template <int R>
inline void ticket_wait_group(std::uint32_t* ts, std::size_t segments, const std::uint64_t* blocks, std::size_t s,
                              std::size_t g, const sycl::nd_item<R>& it) {
    if (it.get_local_linear_id() == 0) {
        std::size_t t = g;
        for (std::size_t i = 0; i < s; ++i) t += blocks[i];
        TicketRef flag{ts[1 + segments + t]};
        while (flag.load(sycl::memory_order::acquire) == 0) {}
    }
    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
}

/// Wait until every group of segments 0 .. upto-1 has finished.
template <int R>
inline void ticket_wait_segments(std::uint32_t* ts, const std::uint64_t* blocks, std::size_t upto, const sycl::nd_item<R>& it) {
    if (it.get_local_linear_id() == 0) {
        for (std::size_t i = 0; i < upto; ++i) {
            TicketRef done{ts[1 + i]};
            while (static_cast<std::uint64_t>(done.load(sycl::memory_order::acquire)) != blocks[i]) {}
        }
    }
    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
}

struct ItemFixup;
}  // namespace detail

// -----------------------------------------------------------------------------
// GroupItem<R>
// -----------------------------------------------------------------------------

/// What a group body receives: the API of sycl::nd_item<R> with every group-level index relative to the stage's SEGMENT
/// (group ids count from 0 within the segment's block). Local ids and ranges are the physical ones.
template <int R>
struct GroupItem {
    sycl::id<R>    get_local_id() const { return it_.get_local_id(); }
    std::size_t    get_local_id(int d) const { return it_.get_local_id(d); }
    sycl::range<R> get_local_range() const { return it_.get_local_range(); }
    std::size_t    get_local_range(int d) const { return it_.get_local_range(d); }
    std::size_t    get_local_linear_id() const { return it_.get_local_linear_id(); }

    sycl::id<R>    get_group_id() const { return group_id_; }
    std::size_t    get_group(int d) const { return group_id_[d]; }
    sycl::range<R> get_group_range() const { return group_range_; }
    std::size_t    get_group_range(int d) const { return group_range_[d]; }
    /// The group's index within its segment, from 0.
    std::size_t    get_group_linear_id() const {
        std::size_t l = 0;
        for (int d = 0; d < R; ++d) l = l * group_range_[d] + group_id_[d];
        return l;
    }

    sycl::id<R> get_global_id() const {
        sycl::id<R> g;
        for (int d = 0; d < R; ++d) g[d] = group_id_[d] * it_.get_local_range(d) + it_.get_local_id(d);
        return g;
    }
    std::size_t    get_global_id(int d) const {
        return group_id_[d] * it_.get_local_range(d) + it_.get_local_id(d);
    }
    sycl::range<R> get_global_range() const {
        sycl::range<R> g = group_range_;
        for (int d = 0; d < R; ++d) g[d] = group_range_[d] * it_.get_local_range(d);
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
    /// Segment `s`'s block of groups (the same for every stage of the segment).
    std::size_t get_segment_group_count(std::size_t s) const { return segment_groups_[s]; }

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

    /// COLLECTIVE (every work-item of the group calls it): wait until group `g` of segment `s` has finished all its stages.
    /// Needs a group with the cs::group_waits_enabled property or a cs::global_barrier(), and `s` EARLIER than this
    /// stage's segment; with no wait state the pointer is null and the behavior is undefined (no checks, as with
    /// waiting on your own or a later segment, which never returns).
    void wait_group(std::size_t s, std::size_t g) const {
        detail::ticket_wait_group<R>(ticket_state_, segments_, segment_groups_, s, g, it_);
    }
    /// COLLECTIVE: wait until every group of every earlier segment has finished (what cs::global_barrier() does for you).
    /// The same requirements as wait_group.
    void wait_earlier_segments() const { detail::ticket_wait_segments<R>(ticket_state_, segment_groups_, segment_, it_); }

    /// The PHYSICAL group, for group functions (barriers, reductions).
    sycl::group<R>  get_group() const { return it_.get_group(); }
    sycl::sub_group get_sub_group() const { return it_.get_sub_group(); }

private:
    GroupItem(const sycl::nd_item<R>& it, const sycl::id<R>& gid, const sycl::range<R>& grange, std::uint32_t segment,
              std::uint32_t segments, const std::uint64_t* segment_groups, std::uint32_t* ticket_state, std::byte* base,
              std::uint32_t used, std::uint32_t cap)
        : it_{it}, group_id_{gid}, group_range_{grange}, segment_{segment}, segments_{segments}, segment_groups_{segment_groups},
          ticket_state_{ticket_state}, base_{base}, used_{used}, cap_{cap} {}

    sycl::nd_item<R>     it_;
    sycl::id<R>          group_id_;     // within the segment
    sycl::range<R>       group_range_;  // this stage's group grid
    std::uint32_t        segment_;
    std::uint32_t        segments_;
    const std::uint64_t* segment_groups_;  // the segment table: the blocks, then the boundary kinds
    std::uint32_t*       ticket_state_;    // null: not in ticket mode
    std::byte*           base_;            // this stage's slice of the local block
    std::uint32_t        used_;
    std::uint32_t        cap_;
    friend struct detail::ItemFixup;
};

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

/// Row-major delinearization over the first R dims of a 1-padded range.
template <int R>
constexpr sycl::id<R> delinearize(std::size_t flat, const sycl::range<3>& r) {
    sycl::id<R> idx;
    for (int d = R - 1; d >= 0; --d) {
        idx[d] = flat % r[d];
        flat /= r[d];
    }
    return idx;
}

/// The same, but dimension 0 is not wrapped: a flat index past the range gives a dimension-0 index past it (a `whole`
/// segment runs a stage over groups beyond its own grid).
template <int R>
constexpr sycl::id<R> delinearize_open(std::size_t flat, const sycl::range<3>& r) {
    sycl::id<R> idx;
    for (int d = R - 1; d >= 1; --d) {
        idx[d] = flat % r[d];
        flat /= r[d];
    }
    idx[0] = flat;
    return idx;
}

template <int R>
constexpr sycl::range<R> first_dims(const sycl::range<3>& r) {
    if constexpr (R == 1)
        return sycl::range<1>{r[0]};
    else if constexpr (R == 2)
        return sycl::range<2>{r[0], r[1]};
    else
        return r;
}

template <int R>
sycl::range<3> pad3(const sycl::range<R>& r) {
    sycl::range<3> out{1, 1, 1};
    for (int d = 0; d < R; ++d) out[d] = r[d];
    return out;
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
/// keeps, copied in from a host member every forward. `group_range` is set by the group launcher once the group's
/// work-group shape is known.
template <class KB>
struct StageArgs {
    StageArgs(typename KB::Inputs i, typename KB::Outputs o, KB b, scratch_t<KB> s) : in{i}, out{o}, body{b}, scratch{s} {}

    sycl::range<3> range{1, 1, 1};        ///< the logical iteration space, 1-padded
    sycl::range<3> group_range{1, 1, 1};  ///< the stage's own group grid (a plain body: {groups, 1, 1})
    std::uint64_t  flat_extent   = 0;     ///< the product of range
    std::uint32_t  scratch_bytes = 0;     ///< the laid-out Scratch, rounded to 16
    typename KB::Inputs  in;
    typename KB::Outputs out;
    KB                   body;
    scratch_t<KB>        scratch;
};

/// The stage-local view of the physical item. The group id is the physical linear group minus the segment's first group,
/// delinearized over the stage's group grid (a `whole` segment widens dimension 0 to cover the block).
struct ItemFixup {
    template <int R>
    static GroupItem<R> make(const StageItem<R>& si, const StageGeometry& g, const std::uint64_t* segment_groups,
                             std::size_t segment_group, const sycl::range<3>& grange, std::uint32_t free_off) {
        sycl::range<3> gr = grange;
        if (g.range_mode == static_cast<std::uint32_t>(RangeMode::whole)) {
            const std::size_t per = gr[1] * gr[2];
            gr[0]                 = per ? (g.segment_groups + per - 1) / per : g.segment_groups;
        }
        return GroupItem<R>{si.item,       delinearize_open<R>(segment_group, gr), first_dims<R>(gr),
                            g.segment,     g.segments,                            segment_groups,
                            static_cast<std::uint32_t*>(g.ticket_state), si.local + g.local_offset, free_off, g.local_bytes};
    }
};

}  // namespace detail

// -----------------------------------------------------------------------------
// The placeholders, the stage function and the launch (design review, 11.6)
// -----------------------------------------------------------------------------

/// Placeholders: one per launch rank; declarations only, the JIT replaces them with the indexed call sequence.
[[clang::noinline]] SYCL_EXTERNAL void run_stages_r1(detail::StageItem<1>, const detail::StageGeometry*, const void* const*, int);
[[clang::noinline]] SYCL_EXTERNAL void run_stages_r2(detail::StageItem<2>, const detail::StageGeometry*, const void* const*, int);
[[clang::noinline]] SYCL_EXTERNAL void run_stages_r3(detail::StageItem<3>, const detail::StageGeometry*, const void* const*, int);

/// ONE stage function per KernelBody type, at the body's rank. Every instantiation has the same signature: (the native
/// item, the geometry array, the argument-pointer array, the stage index). All the conditions below are group-uniform.
template <class KB>
SYCL_EXTERNAL void stage_fn(detail::StageItem<kernel_rank_v<KB>> si, const detail::StageGeometry* geometry,
                            const void* const* args, int k) {
    constexpr int R = kernel_rank_v<KB>;
    const detail::StageGeometry& g = geometry[k];
    const auto&                  a = *static_cast<const detail::StageArgs<KB>*>(args[k]);
    const sycl::nd_item<R>&      it = si.item;

    // This stage's block: the segment's groups only; `own` also stops at the stage's own group count.
    const std::size_t glid = si.group;  // the hardware group, or in ticket mode the ticket
    if (g.group_count == 0 || glid < g.group_base) return;
    const std::size_t seg_group = glid - g.group_base;
    if (seg_group >= g.segment_groups) return;
    if (g.range_mode == static_cast<std::uint32_t>(detail::RangeMode::own) && seg_group >= g.group_count) return;

    auto in   = a.in;
    auto out  = a.out;
    auto body = a.body;

    if constexpr (PlainBody<KB>) {
        // The stage's flat index, contiguous under any geometry; the tail mask is here, the body sees only its range.
        const std::size_t flat = seg_group * it.get_local_range().size() + it.get_local_linear_id();
        if (flat >= a.flat_extent) return;
        const auto idx = detail::delinearize<R>(flat, a.range);
        body(idx, in, out);
    } else if constexpr (HasScratch<KB>) {
        const std::uint64_t* seg = reinterpret_cast<const std::uint64_t*>(geometry + g.stages);
        GroupItem<R>         gi  = detail::ItemFixup::make(si, g, seg, seg_group, a.group_range, a.scratch_bytes);
        auto                 s   = a.scratch;
        detail::fixup_scratch(s, si.local + g.local_offset);  // this stage's own slice of the local block
        body(gi, in, out, s);
    } else {
        const std::uint64_t* seg = reinterpret_cast<const std::uint64_t*>(geometry + g.stages);
        GroupItem<R>         gi  = detail::ItemFixup::make(si, g, seg, seg_group, a.group_range, a.scratch_bytes);
        body(gi, in, out);
    }
}

namespace detail {

using JitId = sycl::AdaptiveCpp_jit::dynamic_function_id;

/// The signature of the placeholder and of every stage function, as the type-erased id types need it.
template <int R>
using run_stages_fn_t = sycl::AdaptiveCpp_jit::dynamic_function<void, StageItem<R>, const StageGeometry*, const void* const*, int>;
template <int R>
using stage_def_t = sycl::AdaptiveCpp_jit::dynamic_function_definition<void, StageItem<R>, const StageGeometry*, const void* const*, int>;

/// The id of stage_fn<KB>: what a group puts in its call sequence. Obtained once per body type, through a function-local
/// static; the component reaches it through a virtual.
template <class KB>
JitId stage_definition_id() {
    static const JitId id = stage_def_t<kernel_rank_v<KB>>{&stage_fn<KB>}.id();
    return id;
}

/// The id of the placeholder of rank `rank` (1..3).
inline JitId run_stages_id(int rank) {
    switch (rank) {
        case 1: return run_stages_fn_t<1>{&run_stages_r1}.id();
        case 2: return run_stages_fn_t<2>{&run_stages_r2}.id();
        case 3: return run_stages_fn_t<3>{&run_stages_r3}.id();
        default: break;
    }
    throw std::logic_error("naga::cospan: no kernel of that rank");
}

/// The dynamic_function_config of a stage sequence: ONE per distinct (rank, sequence of stage definitions), created
/// the first time and NEVER destroyed. A config must outlive every kernel that uses it (the runtime reads it on a worker
/// thread after the submission, possibly before the kernel has been compiled), and a transient group that used it can be
/// deleted at any moment, so the cache is process-wide and leaked on purpose. The key is the definitions' symbol names
/// (an id's handle is its name).
inline sycl::AdaptiveCpp_jit::dynamic_function_config& group_config(int rank, const std::vector<JitId>& defs) {
    using Config = sycl::AdaptiveCpp_jit::dynamic_function_config;
    struct Cache {
        std::mutex                                                                  mu;
        std::map<std::pair<int, std::vector<std::string>>, std::unique_ptr<Config>> map;
    };
    static Cache* const cache = new Cache;  // never destroyed

    std::vector<std::string> names;
    names.reserve(defs.size());
    for (const JitId& d : defs) names.emplace_back(reinterpret_cast<const char*>(d.get_handle()));

    const std::lock_guard<std::mutex> lock{cache->mu};
    std::unique_ptr<Config>&          slot = cache->map[{rank, std::move(names)}];
    if (!slot) {
        slot = std::make_unique<Config>();
        slot->define_as_indexed_call_sequence(run_stages_id(rank), defs, /*index_param=*/3);
    }
    return *slot;
}

/// Default work-group shape per rank (implementation-tunable, r02 section 4.2).
template <int R>
sycl::range<3> default_wg() {
    if constexpr (R == 1)
        return sycl::range<3>{128, 1, 1};
    else if constexpr (R == 2)
        return sycl::range<3>{8, 8, 1};
    else
        return sycl::range<3>{4, 4, 4};
}

/// nd_range<R> with the groups laid along dimension 0 only.
template <int R>
sycl::nd_range<R> group_nd_range(std::size_t groups, const sycl::range<3>& wg) {
    sycl::range<R> local  = first_dims<R>(wg);
    sycl::range<R> global = local;
    global[0] = groups * wg[0];
    return sycl::nd_range<R>{global, local};
}

/// Submit the ONE kernel of a group, of rank R, on the unordered queue `q` for `dev`: ordered ONLY by `deps` (section 3).
/// `geometry` and `args` are device arrays the group owns; the kernel passes stage 0 as the index placeholder, the
/// configuration replaces it by each stage's position. Returns its event.
template <int R>
sycl::event launch_group(sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                         sycl::AdaptiveCpp_jit::dynamic_function_config& cfg, const sycl::nd_range<R>& nd,
                         const StageGeometry* geometry, const void* const* args, std::size_t local_bytes) {
    auto cgf = [&](sycl::handler& h) {
        sycl::local_accessor<std::byte, 1> local{sycl::range<1>{local_bytes ? local_bytes : 1}, h};
        h.parallel_for(nd, cfg.apply([=](sycl::nd_item<R> it) {
            std::byte* lb = local.template get_multi_ptr<sycl::access::decorated::no>().get();

            // Ticket mode (the group has a boundary): the work-group's leader takes the next ticket and broadcasts it; the
            // ticket, not the hardware id, says which segment and group of the segment this work-group is. Without ticket
            // state this is one load and no counters.
            std::uint32_t* ts    = static_cast<std::uint32_t*>(geometry[0].ticket_state);
            std::size_t    group = it.get_group_linear_id();
            std::size_t    seg   = 0;
            std::size_t    nseg  = 0;
            const std::uint64_t* tbl = nullptr;
            if (ts != nullptr) {
                nseg = geometry[0].segments;
                tbl  = reinterpret_cast<const std::uint64_t*>(geometry + geometry[0].stages);  // blocks[S], then kinds[S]
                std::uint32_t t = 0;
                if (it.get_local_linear_id() == 0) {
                    TicketRef counter{ts[0]};
                    t = counter.fetch_add(1u, sycl::memory_order::relaxed);
                }
                group = sycl::group_broadcast(it.get_group(), t);
                std::uint64_t first = 0;  // the segment of this ticket: the block sizes in ticket order
                while (seg + 1 < nseg && group >= first + tbl[seg]) {
                    first += tbl[seg];
                    ++seg;
                }
                if (tbl[nseg + seg] == static_cast<std::uint64_t>(Boundary::global_barrier)) ticket_wait_segments<R>(ts, tbl, seg, it);
            }

            const StageItem<R> si{it, lb, group};
            if constexpr (R == 1)
                run_stages_r1(si, geometry, args, 0);
            else if constexpr (R == 2)
                run_stages_r2(si, geometry, args, 0);
            else
                run_stages_r3(si, geometry, args, 0);

            // Completion: every stage of this group's segment has run (the other stages skipped this ticket). A group
            // barrier, a device-scope release fence, then the leader's release atomics: the group's flag, and its segment's
            // completed counter.
            if (ts != nullptr) {
                sycl::group_barrier(it.get_group());
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                if (it.get_local_linear_id() == 0) {
                    TicketRef flag{ts[1 + nseg + group]};
                    flag.store(1u, sycl::memory_order::release);
                    TicketRef done{ts[1 + seg]};
                    done.fetch_add(1u, sycl::memory_order::release);
                }
            }
        }));
    };
    return q.submit(dev, deps, cgf);
}

}  // namespace detail
}  // namespace naga::cospan
