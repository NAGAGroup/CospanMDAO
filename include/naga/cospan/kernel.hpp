#pragma once

// =============================================================================
// naga/cospan/kernel.hpp — kernel machinery (r02 sections 4 and 5)
// =============================================================================
//
// The KernelBody contract (concepts read off the call operator's signature),
// GroupItem, Local layout, the stage record and its cursor, the generated
// device function kb_fn<KB>, the placeholders the JIT replaces, and the outer
// launch helper. All header-only: placeholders, outer kernels and stages
// instantiate in the consumer's translation unit (D2, D4).

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
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
// GroupItem<R>
// -----------------------------------------------------------------------------

struct StageRecord;

namespace detail {
struct ItemFixup;
}

/// What a group body receives: the API of sycl::nd_item<R> with every
/// group-level index STAGE-LOCAL. Local ids and ranges are the physical ones.
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

    /// The PHYSICAL group, for group functions (barriers, reductions).
    sycl::group<R>  get_group() const { return it_.get_group(); }
    sycl::sub_group get_sub_group() const { return it_.get_sub_group(); }

private:
    GroupItem(const sycl::nd_item<R>& it, const sycl::id<R>& gid, const sycl::range<R>& grange)
        : it_{it}, group_id_{gid}, group_range_{grange} {}

    sycl::nd_item<R> it_;
    sycl::id<R>      group_id_;     // stage-local
    sycl::range<R>   group_range_;  // this stage's group grid
    friend struct detail::ItemFixup;
};

// -----------------------------------------------------------------------------
// StageRecord, FnOffsetInfo (r02 section 5)
// -----------------------------------------------------------------------------

struct StageRecord {
    sycl::range<3> range{1, 1, 1};        ///< logical iteration space, 1-padded
    std::size_t    flat_extent = 0;       ///< product of range
    std::size_t    group_offset = 0;      ///< System-stamped
    std::size_t    group_count = 0;       ///< System-stamped
    sycl::range<3> group_range{1, 1, 1};  ///< System-stamped; group_count is its product
    uint32_t       input_off = 0;
    uint32_t       output_off = 0;
    uint32_t       body_off = 0;
    uint32_t       scratch_off = 0;
    uint32_t       scratch_bytes = 0;
    uint32_t       local_off = 0;         ///< owner-stamped: this stage's byte offset into the launch's local buffer (individual launches: 0)
    uint8_t        rank = 0;
};

static_assert(std::is_trivially_copyable_v<StageRecord>);

/// Position-as-identity for the chained executor: every stage function's
/// FIRST, unconditional statement is info.next().
struct FnOffsetInfo {
    const StageRecord* records;
    uint32_t           i;
    const StageRecord& next() { return records[i++]; }
};

/// What a fused owner reads off a kernel description.
struct KernelParts {
    int                                 rank        = 0;
    bool                                group_body  = false;
    bool                                has_scratch = false;
    StageRecord                         record;  ///< offsets 0-relative; group_* stamped by the owner
    std::span<const std::byte>          inputs, outputs, body, scratch;
    std::optional<sycl::range<3>>       wg;
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

template <int R>
constexpr sycl::range<R> first_dims(const sycl::range<3>& r) {
    if constexpr (R == 1)
        return sycl::range<1>{r[0]};
    else if constexpr (R == 2)
        return sycl::range<2>{r[0], r[1]};
    else
        return r;
}

/// The sole constructor of GroupItem (friendship): the stage-local view of the
/// physical item. Stage-local group = physical linear group - group_offset,
/// delinearized over the stage's group grid.
struct ItemFixup {
    template <int R>
    static GroupItem<R> make(const sycl::nd_item<R>& it, const StageRecord& r) {
        const std::size_t local_group = it.get_group_linear_id() - r.group_offset;
        return GroupItem<R>{it, delinearize<R>(local_group, r.group_range),
                            first_dims<R>(r.group_range)};
    }
};

}  // namespace detail

// -----------------------------------------------------------------------------
// The generated device function and its placeholders (r02 section 5)
// -----------------------------------------------------------------------------

/// Placeholders: one per launch rank; declarations only, the JIT replaces them.
SYCL_EXTERNAL void kb_entry_r1(sycl::nd_item<1>, FnOffsetInfo&, const std::byte*, std::byte*,
                               const std::byte*, std::byte*);
SYCL_EXTERNAL void kb_entry_r2(sycl::nd_item<2>, FnOffsetInfo&, const std::byte*, std::byte*,
                               const std::byte*, std::byte*);
SYCL_EXTERNAL void kb_entry_r3(sycl::nd_item<3>, FnOffsetInfo&, const std::byte*, std::byte*,
                               const std::byte*, std::byte*);

/// One stage function per KernelBody type, at the body's rank.
template <class KB>
SYCL_EXTERNAL void kb_fn(sycl::nd_item<kernel_rank_v<KB>> it, FnOffsetInfo& info,
                         const std::byte* inputs, std::byte* outputs, const std::byte* bodies,
                         std::byte* local_base) {
    constexpr int R = kernel_rank_v<KB>;
    // LAW: the cursor advance is the first statement and unconditional.
    const StageRecord& r = info.next();

    // Whole-group slice check: group-uniform by construction.
    const std::size_t glid = it.get_group_linear_id();
    if (glid < r.group_offset || glid - r.group_offset >= r.group_count) return;

    // Stage-local flat index, contiguous per stage under any geometry.
    const std::size_t flat =
        (glid - r.group_offset) * it.get_local_range().size() + it.get_local_linear_id();

    // id-only bodies: the tail mask is here; the body sees only its range.
    if constexpr (PlainBody<KB>)
        if (flat >= r.flat_extent) return;

    auto in   = *reinterpret_cast<const typename KB::Inputs*>(inputs + r.input_off);
    auto out  = *reinterpret_cast<const typename KB::Outputs*>(outputs + r.output_off);
    auto body = *reinterpret_cast<const KB*>(bodies + r.body_off);

    if constexpr (HasScratch<KB>) {
        GroupItem<R> gi = detail::ItemFixup::make(it, r);
        auto s = *reinterpret_cast<const typename KB::Scratch*>(bodies + r.scratch_off);
        detail::fixup_scratch(s, local_base + r.local_off);  // this stage's own slice of the local buffer
        body(gi, in, out, s);
    } else if constexpr (GroupBody<KB>) {
        GroupItem<R> gi = detail::ItemFixup::make(it, r);
        body(gi, in, out);
    } else {
        const auto idx = detail::delinearize<R>(flat, r.range);
        body(idx, in, out);
    }
}

namespace detail {

/// The dynamic_function_config of a kernel-body type: ONE per KB, built once, with static lifetime. The runtime
/// reads a config on a worker thread AFTER the kernel is submitted, so it must outlive every component and every
/// launch in flight; a function-local static does, which is why no retire list is needed
/// (docs/design-review-2026-10-03.md, section 3). Only plain and group bodies have one (custom-op and host
/// bodies compile no kernel).
template <class KB>
sycl::AdaptiveCpp_jit::dynamic_function_config& jit_config() {
    constexpr int R = kernel_rank_v<KB>;
    static sycl::AdaptiveCpp_jit::dynamic_function_config cfg;
    static const bool defined = [] {
        if constexpr (R == 1)
            cfg.define(&kb_entry_r1, &kb_fn<KB>);
        else if constexpr (R == 2)
            cfg.define(&kb_entry_r2, &kb_fn<KB>);
        else
            cfg.define(&kb_entry_r3, &kb_fn<KB>);
        return true;
    }();
    (void)defined;
    return cfg;
}

}  // namespace detail

// -----------------------------------------------------------------------------
// The outer launch (individual: one-stage wave)
// -----------------------------------------------------------------------------

namespace detail {

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
sycl::nd_range<R> wave_nd_range(std::size_t groups, const sycl::range<3>& wg) {
    sycl::range<R> local  = first_dims<R>(wg);
    sycl::range<R> global = local;
    global[0] = groups * wg[0];
    return sycl::nd_range<R>{global, local};
}

template <class T>
const std::byte* bytes_of(const T* p) {
    return reinterpret_cast<const std::byte*>(p);
}

/// Submit the ONE outer kernel of rank R on the unordered queue `q`, for device `dev`. It is ordered ONLY by `deps`
/// (docs/design-review-2026-10-03.md, section 3): nothing else in the queue orders it. `payload` (a trivially
/// copyable struct with members rec, in, out, body) is captured by value; the stage function reads the private copies
/// through FnOffsetInfo and byte pointers. Returns its event.
template <int R, class Payload>
sycl::event launch_individual(sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                              sycl::AdaptiveCpp_jit::dynamic_function_config& cfg, const sycl::nd_range<R>& nd,
                              const Payload& payload, std::size_t scratch_bytes) {
    auto cgf = [&](sycl::handler& h) {
        sycl::local_accessor<std::byte, 1> local{sycl::range<1>{scratch_bytes ? scratch_bytes : 1}, h};
        h.parallel_for(nd, cfg.apply([=](sycl::nd_item<R> it) {
            FnOffsetInfo info{&payload.rec, 0};  // fresh per work-item
            const std::byte* in_b   = bytes_of(&payload.in);
            std::byte*       out_b  = const_cast<std::byte*>(bytes_of(&payload.out));
            const std::byte* body_b = bytes_of(&payload.body);
            std::byte*       lb = local.template get_multi_ptr<sycl::access::decorated::no>().get();
            if constexpr (R == 1)
                kb_entry_r1(it, info, in_b, out_b, body_b, lb);
            else if constexpr (R == 2)
                kb_entry_r2(it, info, in_b, out_b, body_b, lb);
            else
                kb_entry_r3(it, info, in_b, out_b, body_b, lb);
        }));
    };
    return q.submit(dev, deps, cgf);
}

// -----------------------------------------------------------------------------
// The chained wave (fused owners, r02 section 5)
// -----------------------------------------------------------------------------

using JitId = sycl::AdaptiveCpp_jit::dynamic_function_id;

/// The REAL six-argument signature of the entry placeholder and of every stage
/// function, as the type-erased id types need it (the spike's four-argument
/// version was illustrative only).
template <int R>
using entry_fn_t =
    sycl::AdaptiveCpp_jit::dynamic_function<void, sycl::nd_item<R>, FnOffsetInfo&, const std::byte*,
                                            std::byte*, const std::byte*, std::byte*>;
template <int R>
using stage_def_t =
    sycl::AdaptiveCpp_jit::dynamic_function_definition<void, sycl::nd_item<R>, FnOffsetInfo&,
                                                       const std::byte*, std::byte*, const std::byte*,
                                                       std::byte*>;

/// The type-erased id of kb_fn<KB>: what a fused owner puts in a call sequence.
template <class KB>
JitId stage_definition_id() {
    return stage_def_t<kernel_rank_v<KB>>{&kb_fn<KB>}.id();
}

/// The id of the entry placeholder of rank `rank` (1..3).
inline JitId wave_entry_id(int rank) {
    switch (rank) {
        case 1: return entry_fn_t<1>{&kb_entry_r1}.id();
        case 2: return entry_fn_t<2>{&kb_entry_r2}.id();
        case 3: return entry_fn_t<3>{&kb_entry_r3}.id();
        default: break;
    }
    throw std::logic_error("naga::cospan: no kernel entry of that rank");
}

/// Submit the ONE outer kernel of a fused wave: like launch_individual, but the
/// records and the three byte streams are USM the owner staged, captured as
/// raw pointers. Every work-item starts with a fresh cursor at record 0; the
/// call sequence in `cfg` walks the records in position order. Submitted on the unordered queue `q` for `dev`, ordered
/// ONLY by `deps` (section 3). Returns its event.
template <int R>
sycl::event launch_wave(sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                        sycl::AdaptiveCpp_jit::dynamic_function_config& cfg, const sycl::nd_range<R>& nd, const StageRecord* records, const std::byte* in_bytes,
                        std::byte* out_bytes, const std::byte* body_bytes, std::size_t local_bytes) {
    auto cgf = [&](sycl::handler& h) {
        sycl::local_accessor<std::byte, 1> local{sycl::range<1>{local_bytes ? local_bytes : 1}, h};
        h.parallel_for(nd, cfg.apply([=](sycl::nd_item<R> it) {
            FnOffsetInfo info{records, 0};  // fresh per work-item
            std::byte*   lb = local.template get_multi_ptr<sycl::access::decorated::no>().get();
            if constexpr (R == 1)
                kb_entry_r1(it, info, in_bytes, out_bytes, body_bytes, lb);
            else if constexpr (R == 2)
                kb_entry_r2(it, info, in_bytes, out_bytes, body_bytes, lb);
            else
                kb_entry_r3(it, info, in_bytes, out_bytes, body_bytes, lb);
        }));
    };
    return q.submit(dev, deps, cgf);
}

}  // namespace detail
}  // namespace naga::cospan
