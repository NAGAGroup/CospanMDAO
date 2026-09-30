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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "components.hpp"

namespace gc {

enum class Style { unfused, parallel, fused };

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
        if (style_ == Style::fused) {
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

        promote(pre + "r2u.A", "A");
        promote(pre + "r2u.u1", "u1");
        promote(pre + "r2u.v1", "v1");
        promote(pre + "r2u.u2", "u2");
        promote(pre + "r2u.v2", "v2");
        promote(pre + "gtp.y", "y");
        promote("cred.z", "z");
        promote(pre + "r2u.B", "B");
        promote("cred.x", "x");
        promote("gemv.y", "w");
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
        std::string pre;  // the path prefix of gemv_q and gtp_s
        if (style_ == Style::parallel) {
            add_subsystem("par", cs::property_list{cs::property::parallel{}}, child<Gemv>("gemv_q", n_, 1.0f),
                          child<GemvTPartial>("gtp_s", n_, R_));
            pre = "par.";
        } else if (style_ == Style::fused) {
            // Independent stages: each gets its own slice of groups.
            add_subsystem("fw",
                          cs::property_list{cs::property::fused{cs::property::fused::Mode::parallel, wg256()}},
                          child<Gemv>("gemv_q", n_, 1.0f), child<GemvTPartial>("gtp_s", n_, R_));
            pre = "fw.";
        } else {
            add_subsystem<Gemv>("gemv_q", n_, 1.0f);
            add_subsystem<GemvTPartial>("gtp_s", n_, R_);
        }
        add_subsystem<ColReduce<false>>("cred_s", n_, R_, 1.0f);

        connect(pre + "gtp_s.P", "cred_s.P");
        tie_inputs({pre + "gemv_q.B", pre + "gtp_s.B"}, "B");  // one B feeds both kernels

        promote(pre + "gemv_q.x", "p");
        promote(pre + "gtp_s.y", "r");
        promote(pre + "gemv_q.y", "q");
        promote("cred_s.x", "s");
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

        promote("g1.A", "A");
        promote("g1.u1", "u1a");
        promote("g1.v1", "v1a");
        promote("g1.u2", "u2a");
        promote("g1.v2", "v2a");
        promote("g1.y", "ya");
        promote("g1.z", "za");
        promote("g2.u1", "u1b");
        promote("g2.v1", "v1b");
        promote("g2.u2", "u2b");
        promote("g2.v2", "v2b");
        promote("g1.B", "B1");
        promote("g1.x", "x1");
        promote("g1.w", "w1");
        promote("bicgk.q", "q");
        promote("bicgk.s", "s");
        promote("g2.B", "B2");
        promote("g2.x", "x2");
        promote("g2.w", "w2");
    }
};

}  // namespace gc
