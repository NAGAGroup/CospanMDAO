#pragma once

// =============================================================================
// examples/gemver-chain/systems.hpp — the chain as Systems, in several fusion styles
// =============================================================================
//
// GEMVER and BiCGK are reusable System classes with promoted ports; their internal
// style is a constructor argument. Every root below promotes the same names, so the
// driver reads and writes one set of paths whatever the variant:
//
//   inputs   A                     G1's matrix A0
//            u1a v1a u2a v2a ya za G1's vectors
//            u1b v1b u2b v2b       G2's copies of u1..v2 (set separately, not tied)
//   outputs  B1 x1 w1              G1
//            q s                   BiCGK
//            B2 x2 w2              G2
//
// Styles (Style):
//   unfused   every component is its own kernel
//   parallel  as unfused, but BiCGK's gemv_q and gtp_s are a system_from group (the old ParallelSystem; the two
//             stages are independent, so their concurrency now comes from data dependencies alone)
//   fused     GEMVER: fuse_components(segment<whole>[r2u, gtp]), cred, gemv;
//             BiCGK : fuse_components(gemv_q, gtp_s) (one segment each), cred_s
// and, as its own root, BarrierTop: the whole chain as ONE fuse_components segment<whole> with
// software grid barriers between the dependent stages (design review, section 4).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "components.hpp"

namespace gc {

enum class Style { unfused, parallel, fused, onepass };  // onepass: fused GEMVERs, one-pass BiCGK

// =============================================================================
// GEMVER: ports A u1 v1 u2 v2 y z -> B x w
// =============================================================================
//   B = A + u1 v1^T + u2 v2^T ;  x = beta * B^T y + z ;  w = alpha * B x
class Gemver : public cs::System {
    Style style_;
    int   n_, R_;

public:
    Gemver(Style style, int n, int R) : style_{style}, n_{n}, R_{R} {}

    void setup() override {
        const float ab = 1.0f / std::sqrt(static_cast<float>(n_));  // alpha = beta
        std::string pre;                                             // the path prefix of r2u and gtp
        if (style_ == Style::fused || style_ == Style::onepass) {
            // r2u and gtp share their (c, j) mapping, so the same work-item reads its own B writes.
            // (the work-group size 256 comes from the stages' own launch configs)
            add_subsystem(cs::fuse_components(
                "fw", cs::connects_desc({{"r2u.B", "gtp.B"}}),
                cs::segment<cs::range_mode::whole>(cs::subsystem_desc<Rank2Update>("r2u", n_, R_),
                                                   cs::subsystem_desc<GemvTPartial>("gtp", n_, R_))));
            pre = "fw.";
        } else {
            add_subsystem(cs::subsystem_desc<Rank2Update>("r2u", n_, R_));
            add_subsystem(cs::subsystem_desc<GemvTPartial>("gtp", n_, R_));
            connect("r2u.B", "gtp.B");
        }
        add_subsystem(cs::subsystem_desc<ColReduce<true>>("cred", n_, R_, ab));
        add_subsystem(cs::subsystem_desc<Gemv>("gemv", n_, ab));

        connect(pre + "gtp.P", "cred.P");
        connect(pre + "r2u.B", "gemv.B");
        connect("cred.x", "gemv.x");

        promote<DType::f32, 2>(pre + "r2u.A", "A");
        promote<DType::f32, 1>(pre + "r2u.u1", "u1");
        promote<DType::f32, 1>(pre + "r2u.v1", "v1");
        promote<DType::f32, 1>(pre + "r2u.u2", "u2");
        promote<DType::f32, 1>(pre + "r2u.v2", "v2");
        promote<DType::f32, 1>(pre + "gtp.y", "y");
        promote<DType::f32, 1>("cred.z", "z");
        promote<DType::f32, 2>(pre + "r2u.B", "B");
        promote<DType::f32, 1>("cred.x", "x");
        promote<DType::f32, 1>("gemv.y", "w");
    }
};

// =============================================================================
// BiCGK: ports B p r -> q s      q = B p ; s = B^T r
// =============================================================================
class Bicgk : public cs::System {
    Style style_;
    int   n_, R_;

public:
    Bicgk(Style style, int n, int R) : style_{style}, n_{n}, R_{R} {}

    void setup() override {
        if (style_ == Style::onepass) {
            // q and s from ONE pass over B: the one-pass kernel, then the two reducers.
            add_subsystem(cs::subsystem_desc<BicgkOnePass>("onepass", n_, R_));
            add_subsystem(cs::subsystem_desc<RowReduce>("rowred", n_));
            add_subsystem(cs::subsystem_desc<ColReduce<false>>("cred_s", n_, R_, 1.0f));
            connect("onepass.Pq", "rowred.Pq");
            connect("onepass.Ps", "cred_s.P");
            promote<DType::f32, 2>("onepass.B", "B");
            promote<DType::f32, 1>("onepass.p", "p");
            promote<DType::f32, 1>("onepass.r", "r");
            promote<DType::f32, 1>("rowred.q", "q");
            promote<DType::f32, 1>("cred_s.x", "s");
            return;
        }
        std::string pre;  // the path prefix of gemv_q and gtp_s
        if (style_ == Style::parallel) {
            // REVIEW: was a ParallelSystem (independent children, no kernel fusion). A plain group of the two children is the
            // same thing now: they share no variable, so nothing orders them and they overlap.
            add_subsystem(cs::system_from("par", cs::subsystem_desc<Gemv>("gemv_q", n_, 1.0f, std::size_t{0}, true),
                                          cs::subsystem_desc<GemvTPartial>("gtp_s", n_, R_, std::size_t{0}, true)));
            pre = "par.";
        } else if (style_ == Style::fused) {
            // Independent stages: each gets its own slice of groups.
            add_subsystem(cs::fuse_components(
                "fw", cs::subsystem_desc<Gemv>("gemv_q", n_, 1.0f, std::size_t{0}, true),
                cs::subsystem_desc<GemvTPartial>("gtp_s", n_, R_, std::size_t{0}, true)));
            pre = "fw.";
        } else {
            add_subsystem(cs::subsystem_desc<Gemv>("gemv_q", n_, 1.0f, std::size_t{0}, true));
            add_subsystem(cs::subsystem_desc<GemvTPartial>("gtp_s", n_, R_, std::size_t{0}, true));
        }
        add_subsystem(cs::subsystem_desc<ColReduce<false>>("cred_s", n_, R_, 1.0f));

        connect(pre + "gtp_s.P", "cred_s.P");
        // One B feeds both kernels; both declare B rank-only and the tie alias owns its shape.
        tie_inputs<DType::f32, 2>({pre + "gemv_q.B", pre + "gtp_s.B"}, "B", cs::shape(n_, n_));

        promote<DType::f32, 1>(pre + "gemv_q.x", "p");
        promote<DType::f32, 1>(pre + "gtp_s.y", "r");
        promote<DType::f32, 1>(pre + "gemv_q.y", "q");
        promote<DType::f32, 1>("cred_s.x", "s");
    }
};

// =============================================================================
// The chain: G1 -> BiCGK -> G2 (unfused, parallel or fused children)
// =============================================================================
class Chain : public cs::System {
    Style style_;
    int   n_, R_;

public:
    Chain(Style style, int n, int R) : style_{style}, n_{n}, R_{R} {}

    void setup() override {
        add_subsystem(cs::subsystem_desc<Gemver>("g1", style_, n_, R_));
        add_subsystem(cs::subsystem_desc<Bicgk>("bicgk", style_, n_, R_));
        add_subsystem(cs::subsystem_desc<Gemver>("g2", style_, n_, R_));

        connect("g1.B", "bicgk.B");
        connect("g1.x", "bicgk.p");  // p = x1
        connect("g1.w", "bicgk.r");  // r = w1
        connect("g1.B", "g2.A");     // G2's A is B1
        connect("bicgk.q", "g2.y");  // y = q
        connect("bicgk.s", "g2.z");  // z = s

        promote<DType::f32, 2>("g1.A", "A");
        promote<DType::f32, 1>("g1.u1", "u1a");
        promote<DType::f32, 1>("g1.v1", "v1a");
        promote<DType::f32, 1>("g1.u2", "u2a");
        promote<DType::f32, 1>("g1.v2", "v2a");
        promote<DType::f32, 1>("g1.y", "ya");
        promote<DType::f32, 1>("g1.z", "za");
        promote<DType::f32, 1>("g2.u1", "u1b");
        promote<DType::f32, 1>("g2.v1", "v1b");
        promote<DType::f32, 1>("g2.u2", "u2b");
        promote<DType::f32, 1>("g2.v2", "v2b");
        promote<DType::f32, 2>("g1.B", "B1");
        promote<DType::f32, 1>("g1.x", "x1");
        promote<DType::f32, 1>("g1.w", "w1");
        promote<DType::f32, 1>("bicgk.q", "q");
        promote<DType::f32, 1>("bicgk.s", "s");
        promote<DType::f32, 2>("g2.B", "B2");
        promote<DType::f32, 1>("g2.x", "x2");
        promote<DType::f32, 1>("g2.w", "w2");
    }
};

// =============================================================================
// The barrier variant: the whole chain as ONE sequential fused component, flat
// =============================================================================
// Every stage runs over `groups` resident work-groups (groups * 256 items, grid-stride),
// and a GridBarrier stands between every pair of stages that depend on each other across
// groups. Sequential-mode stages run one after the other inside each work-group, so the
// barrier stage is where the groups of the WHOLE grid meet.
//
// REVIEW: was a FusedSystem subclass with its own connects and promotions. The helper takes neither promotions nor a
// subclass (design review, section 4), so the connects go into connects_desc (paths relative to the group) and BarrierTop
// promotes the ports it needs by path ("chain.r2u1.A"). Whether promote may name a stage port of a fused component
// (and an output that is consumed inside the same wave) is not stated in fused_component.hpp; the tie_outputs below
// already reach "chain.bar1.count" the same way.
inline auto barrier_chain(int n, int R, std::size_t groups, std::uint32_t spin_limit) {
    const float       ab  = 1.0f / std::sqrt(static_cast<float>(n));
    const std::size_t res = groups * kWg;  // resident_items of every stage
    return cs::fuse_components(
        "chain",
        cs::connects_desc({// G1
                           {"r2u1.B", "gtp1.B"},
                           {"gtp1.P", "cred1.P"},
                           {"r2u1.B", "gemv1.B"},
                           {"cred1.x", "gemv1.x"},
                           // BiCGK on B1: p = x1, r = w1
                           {"r2u1.B", "gemv_q.B"},
                           {"cred1.x", "gemv_q.x"},
                           {"r2u1.B", "gtp_s.B"},
                           {"gemv1.y", "gtp_s.y"},
                           {"gtp_s.P", "cred_s.P"},
                           // G2: A = B1, y = q, z = s
                           {"r2u1.B", "r2u2.A"},
                           {"r2u2.B", "gtp2.B"},
                           {"gemv_q.y", "gtp2.y"},
                           {"gtp2.P", "cred2.P"},
                           {"cred_s.x", "cred2.z"},
                           {"r2u2.B", "gemv2.B"},
                           {"cred2.x", "gemv2.x"}}),
        cs::segment<cs::range_mode::whole>(
        cs::subsystem_desc<Rank2Update>("r2u1", n, R, res), cs::subsystem_desc<GemvTPartial>("gtp1", n, R, res),
        cs::subsystem_desc<GridBarrier>("bar1", groups, spin_limit), cs::subsystem_desc<ColReduce<true>>("cred1", n, R, ab, res),
        cs::subsystem_desc<GridBarrier>("bar2", groups, spin_limit), cs::subsystem_desc<Gemv>("gemv1", n, ab, res),
        cs::subsystem_desc<GridBarrier>("bar3", groups, spin_limit), cs::subsystem_desc<Gemv>("gemv_q", n, 1.0f, res),
        cs::subsystem_desc<GemvTPartial>("gtp_s", n, R, res), cs::subsystem_desc<GridBarrier>("bar4", groups, spin_limit),
        cs::subsystem_desc<ColReduce<false>>("cred_s", n, R, 1.0f, res), cs::subsystem_desc<GridBarrier>("bar5", groups, spin_limit),
        cs::subsystem_desc<Rank2Update>("r2u2", n, R, res), cs::subsystem_desc<GemvTPartial>("gtp2", n, R, res),
        cs::subsystem_desc<GridBarrier>("bar6", groups, spin_limit), cs::subsystem_desc<ColReduce<true>>("cred2", n, R, ab, res),
        cs::subsystem_desc<GridBarrier>("bar7", groups, spin_limit), cs::subsystem_desc<Gemv>("gemv2", n, ab, res)));
}

// The barrier counters are zeroed by ONE ZeroCounters launch (14 outputs: count1..7, timeout1..7)
// BEFORE the fused kernel (outputs are never initialized, D138). Inside the wave a late group
// could zero a counter after another group had incremented it, so the zeroing must never be a
// stage of the fused component.
//
// The barrier variant's root: the zeroing component first, then the chain, each zero output
// tied to its barrier's counter (tie_outputs across the fuse boundary). The chain's ports are promoted
// so the root presents the same names as the other variants; the tie aliases
// "bar<i>_count" / "bar<i>_timeout" are the barriers' counters.
class BarrierTop : public cs::System {
    int           n_, R_;
    std::size_t   groups_;
    std::uint32_t spin_limit_;

public:
    static constexpr int kBarriers = 7;

    BarrierTop(int n, int R, std::size_t groups, std::uint32_t spin_limit)
        : n_{n}, R_{R}, groups_{groups}, spin_limit_{spin_limit} {}

    void setup() override {
        add_subsystem(cs::subsystem_desc<ZeroCounters>("zero"));
        add_subsystem(barrier_chain(n_, R_, groups_, spin_limit_));
        for (int i = 1; i <= kBarriers; ++i) {
            const std::string k = std::to_string(i), b = "chain.bar" + k;
            tie_outputs<DType::atomic_u32, 1>({"zero.count" + k, b + ".count"}, "bar" + k + "_count", cs::shape(1));
            tie_outputs<DType::atomic_u32, 1>({"zero.timeout" + k, b + ".timeout"}, "bar" + k + "_timeout", cs::shape(1));
        }
        // Tied writers never wait on each other (design review, section 3): the barriers add onto what zero wrote, so the
        // edge is declared (it used to be registration order).
        depends_on("chain", "zero");

        promote<DType::f32, 2>("chain.r2u1.A", "A");
        promote<DType::f32, 1>("chain.r2u1.u1", "u1a");
        promote<DType::f32, 1>("chain.r2u1.v1", "v1a");
        promote<DType::f32, 1>("chain.r2u1.u2", "u2a");
        promote<DType::f32, 1>("chain.r2u1.v2", "v2a");
        promote<DType::f32, 1>("chain.gtp1.y", "ya");
        promote<DType::f32, 1>("chain.cred1.z", "za");
        promote<DType::f32, 1>("chain.r2u2.u1", "u1b");
        promote<DType::f32, 1>("chain.r2u2.v1", "v1b");
        promote<DType::f32, 1>("chain.r2u2.u2", "u2b");
        promote<DType::f32, 1>("chain.r2u2.v2", "v2b");
        promote<DType::f32, 2>("chain.r2u1.B", "B1");
        promote<DType::f32, 1>("chain.cred1.x", "x1");
        promote<DType::f32, 1>("chain.gemv1.y", "w1");
        promote<DType::f32, 1>("chain.gemv_q.y", "q");
        promote<DType::f32, 1>("chain.cred_s.x", "s");
        promote<DType::f32, 2>("chain.r2u2.B", "B2");
        promote<DType::f32, 1>("chain.cred2.x", "x2");
        promote<DType::f32, 1>("chain.gemv2.y", "w2");
    }
};
}  // namespace gc
