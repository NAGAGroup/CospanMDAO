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
//   parallel  as unfused, but BiCGK's gemv_q and gtp_s are a ParallelSystem
//   fused     GEMVER: FusedSystem(sequential) [r2u, gtp], cred, gemv;
//             BiCGK : FusedSystem(parallel) [gemv_q, gtp_s], cred_s
// and, as its own root, BarrierChain: the whole chain as ONE FusedSystem with
// software grid barriers between the dependent stages.

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

inline sycl::range<3> wg256() { return sycl::range<3>{kWg, 1, 1}; }

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
            add_subsystem("fw",
                          cs::property_list{cs::property::fused{cs::property::fused::Mode::sequential, wg256()}},
                          child<Rank2Update>("r2u", n_, R_), child<GemvTPartial>("gtp", n_, R_));
            pre = "fw.";
        } else {
            add_subsystem<Rank2Update>("r2u", n_, R_);
            add_subsystem<GemvTPartial>("gtp", n_, R_);
        }
        add_subsystem<ColReduce<true>>("cred", n_, R_, ab);
        add_subsystem<Gemv>("gemv", n_, ab);

        connect(pre + "r2u.B", pre + "gtp.B");
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
            add_subsystem<BicgkOnePass>("onepass", n_, R_);
            add_subsystem<RowReduce>("rowred", n_);
            add_subsystem<ColReduce<false>>("cred_s", n_, R_, 1.0f);
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
            add_subsystem("par", cs::property_list{cs::property::parallel{}},
                          child<Gemv>("gemv_q", n_, 1.0f, std::size_t{0}, true),
                          child<GemvTPartial>("gtp_s", n_, R_, std::size_t{0}, true));
            pre = "par.";
        } else if (style_ == Style::fused) {
            // Independent stages: each gets its own slice of groups.
            add_subsystem("fw",
                          cs::property_list{cs::property::fused{cs::property::fused::Mode::parallel, wg256()}},
                          child<Gemv>("gemv_q", n_, 1.0f, std::size_t{0}, true),
                          child<GemvTPartial>("gtp_s", n_, R_, std::size_t{0}, true));
            pre = "fw.";
        } else {
            add_subsystem<Gemv>("gemv_q", n_, 1.0f, std::size_t{0}, true);
            add_subsystem<GemvTPartial>("gtp_s", n_, R_, std::size_t{0}, true);
        }
        add_subsystem<ColReduce<false>>("cred_s", n_, R_, 1.0f);

        connect(pre + "gtp_s.P", "cred_s.P");
        // One B feeds both kernels; both declare B rank-only and the tie alias owns its shape.
        tie_inputs<DType::f32, 2>({pre + "gemv_q.B", pre + "gtp_s.B"}, "B", cs::alias::fixed(n_, n_));

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
        add_subsystem<Gemver>("g1", style_, n_, R_);
        add_subsystem<Bicgk>("bicgk", style_, n_, R_);
        add_subsystem<Gemver>("g2", style_, n_, R_);

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
// The barrier variant: the whole chain as ONE sequential FusedSystem, flat
// =============================================================================
// Every stage runs over `groups` resident work-groups (groups * 256 items, grid-stride),
// and a GridBarrier stands between every pair of stages that depend on each other across
// groups. Sequential-mode stages run one after the other inside each work-group, so the
// barrier stage is where the groups of the WHOLE grid meet.
class BarrierChain : public cs::FusedSystem {
    int           n_, R_;
    std::size_t   groups_;
    std::uint32_t spin_limit_;

public:
    static constexpr int kBarriers = 7;

    BarrierChain(int n, int R, std::size_t groups, std::uint32_t spin_limit)
        : cs::FusedSystem(Mode::sequential, std::optional<sycl::range<3>>{wg256()})
        , n_{n}, R_{R}, groups_{groups}, spin_limit_{spin_limit} {}

    void setup() override {
        const float       ab  = 1.0f / std::sqrt(static_cast<float>(n_));
        const std::size_t res = groups_ * kWg;  // resident_items of every stage

        add_subsystem<Rank2Update>("r2u1", n_, R_, res);
        add_subsystem<GemvTPartial>("gtp1", n_, R_, res);
        add_subsystem<GridBarrier>("bar1", groups_, spin_limit_);
        add_subsystem<ColReduce<true>>("cred1", n_, R_, ab, res);
        add_subsystem<GridBarrier>("bar2", groups_, spin_limit_);
        add_subsystem<Gemv>("gemv1", n_, ab, res);
        add_subsystem<GridBarrier>("bar3", groups_, spin_limit_);
        add_subsystem<Gemv>("gemv_q", n_, 1.0f, res);
        add_subsystem<GemvTPartial>("gtp_s", n_, R_, res);
        add_subsystem<GridBarrier>("bar4", groups_, spin_limit_);
        add_subsystem<ColReduce<false>>("cred_s", n_, R_, 1.0f, res);
        add_subsystem<GridBarrier>("bar5", groups_, spin_limit_);
        add_subsystem<Rank2Update>("r2u2", n_, R_, res);
        add_subsystem<GemvTPartial>("gtp2", n_, R_, res);
        add_subsystem<GridBarrier>("bar6", groups_, spin_limit_);
        add_subsystem<ColReduce<true>>("cred2", n_, R_, ab, res);
        add_subsystem<GridBarrier>("bar7", groups_, spin_limit_);
        add_subsystem<Gemv>("gemv2", n_, ab, res);

        // G1
        connect("r2u1.B", "gtp1.B");
        connect("gtp1.P", "cred1.P");
        connect("r2u1.B", "gemv1.B");
        connect("cred1.x", "gemv1.x");
        // BiCGK on B1: p = x1, r = w1
        connect("r2u1.B", "gemv_q.B");
        connect("cred1.x", "gemv_q.x");
        connect("r2u1.B", "gtp_s.B");
        connect("gemv1.y", "gtp_s.y");
        connect("gtp_s.P", "cred_s.P");
        // G2: A = B1, y = q, z = s
        connect("r2u1.B", "r2u2.A");
        connect("r2u2.B", "gtp2.B");
        connect("gemv_q.y", "gtp2.y");
        connect("gtp2.P", "cred2.P");
        connect("cred_s.x", "cred2.z");
        connect("r2u2.B", "gemv2.B");
        connect("cred2.x", "gemv2.x");

        promote<DType::f32, 2>("r2u1.A", "A");
        promote<DType::f32, 1>("r2u1.u1", "u1a");
        promote<DType::f32, 1>("r2u1.v1", "v1a");
        promote<DType::f32, 1>("r2u1.u2", "u2a");
        promote<DType::f32, 1>("r2u1.v2", "v2a");
        promote<DType::f32, 1>("gtp1.y", "ya");
        promote<DType::f32, 1>("cred1.z", "za");
        promote<DType::f32, 1>("r2u2.u1", "u1b");
        promote<DType::f32, 1>("r2u2.v1", "v1b");
        promote<DType::f32, 1>("r2u2.u2", "u2b");
        promote<DType::f32, 1>("r2u2.v2", "v2b");
        promote<DType::f32, 2>("r2u1.B", "B1");
        promote<DType::f32, 1>("cred1.x", "x1");
        promote<DType::f32, 1>("gemv1.y", "w1");
        promote<DType::f32, 1>("gemv_q.y", "q");
        promote<DType::f32, 1>("cred_s.x", "s");
        promote<DType::f32, 2>("r2u2.B", "B2");
        promote<DType::f32, 1>("cred2.x", "x2");
        promote<DType::f32, 1>("gemv2.y", "w2");
    }
};

// The barrier counters are zeroed by ONE ZeroCounters launch (14 outputs: count1..7, timeout1..7)
// BEFORE the fused kernel (outputs are never initialized, D138). Inside the wave a late group
// could zero a counter after another group had incremented it, so the zeroing must never be a
// stage of the FusedSystem.
//
// The barrier variant's root: the zeroing component first, then the chain, each zero output
// tied to its barrier's counter (tie_outputs across the fuse boundary). The chain's promoted names
// are promoted again so the root presents the same names as the other variants; the tie aliases
// "bar<i>_count" / "bar<i>_timeout" are the barriers' counters.
class BarrierTop : public cs::System {
    int           n_, R_;
    std::size_t   groups_;
    std::uint32_t spin_limit_;

public:
    BarrierTop(int n, int R, std::size_t groups, std::uint32_t spin_limit)
        : n_{n}, R_{R}, groups_{groups}, spin_limit_{spin_limit} {}

    void setup() override {
        add_subsystem<ZeroCounters>("zero");  // BEFORE the chain: registration order is execution order
        add_subsystem<BarrierChain>("chain", n_, R_, groups_, spin_limit_);
        for (int i = 1; i <= BarrierChain::kBarriers; ++i) {
            const std::string k = std::to_string(i), b = "chain.bar" + k;
            tie_outputs<DType::atomic_u32, 1>({"zero.count" + k, b + ".count"}, "bar" + k + "_count", cs::alias::fixed(1));
            tie_outputs<DType::atomic_u32, 1>({"zero.timeout" + k, b + ".timeout"}, "bar" + k + "_timeout",
                                              cs::alias::fixed(1));
        }
        for (const char* n : {"A", "B1", "B2"}) promote<DType::f32, 2>(std::string{"chain."} + n, n);
        for (const char* n : {"u1a", "v1a", "u2a", "v2a", "ya", "za", "u1b", "v1b", "u2b", "v2b",
                              "x1",  "w1",  "q",   "s",   "x2",  "w2"})
            promote<DType::f32, 1>(std::string{"chain."} + n, n);
    }
};

}  // namespace gc