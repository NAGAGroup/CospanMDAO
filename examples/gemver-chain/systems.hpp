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
//   parallel  as unfused, but BiCGK's gemv_q and gtp_s are a system_from group: a plain System, TWO launches (the two
//             stages are independent, so they overlap through data dependencies alone; no fused group is involved)
//   fused     GEMVER: fuse_components(connects, segment<range_mode::whole>(r2u, gtp)), then cred, gemv;
//             BiCGK : fuse_components(gemv_q, gtp_s), each stage its own segment, no boundary: ONE launch, then cred_s
//   onepass   fused GEMVERs as above, and BiCGK from ONE pass over B (BicgkOnePass, then RowReduce and ColReduce)

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
            // A plain group of the two children: they share no variable, so nothing orders them and they overlap.
            add_subsystem(cs::system_from("par", cs::subsystem_desc<Gemv>("gemv_q", n_, 1.0f, true),
                                          cs::subsystem_desc<GemvTPartial>("gtp_s", n_, R_, true)));
            pre = "par.";
        } else if (style_ == Style::fused) {
            // Independent stages: each is a one-stage segment with its own slice of groups.
            add_subsystem(cs::fuse_components(
                "fw", cs::subsystem_desc<Gemv>("gemv_q", n_, 1.0f, true), cs::subsystem_desc<GemvTPartial>("gtp_s", n_, R_, true)));
            pre = "fw.";
        } else {
            add_subsystem(cs::subsystem_desc<Gemv>("gemv_q", n_, 1.0f, true));
            add_subsystem(cs::subsystem_desc<GemvTPartial>("gtp_s", n_, R_, true));
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

}  // namespace gc
