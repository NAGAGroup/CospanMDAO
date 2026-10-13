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
// SIZES. n and R are the constructor arguments of the root (Chain), passed down to the children; nothing else holds them.
// The components declare every data port with an unspecified shape and read their sizes from the shapes of their ports
// (components.hpp), so the Systems supply the shapes:
//   - Chain gives each FREE input (A, u1a .. za, u1b .. v2b) all its dimensions with shape_hook(.., fixed_at_run(n)). It does
//     it, not Gemver, because only Chain knows which inputs are free: G2's A, y and z are connected, and a hook and a
//     connection exclude each other.
//   - Gemver and Bicgk (and Chain, for the crafted chain) give each partial-sum OUTPUT (GemvTPartial's P, BicgkOnePass's Ps,
//     the crafted components' P and Ps) its shape with shape_hook(.., fixed_at_run(n / R), fixed_at_run(n)).
//   - Every other output takes its shape in the component's prepare_forward from the shapes of its inputs.
//   - R: no shape carries it for Rank2Update, which takes R as its constructor argument and initializes its own `R` input.
//   - scale: one tied alias per System (GEMVER: cred and gemv, 1/sqrt(n); BiCGK: gemv_q and cred_s, 1), initialized by that
//     System (BiCGK's onepass and naive styles have one user, cred_s: a tie needs two members, so they promote it). In the
//     crafted chain the three fused groups g1, mid and g2 play those roles: g1 and g2 tie their own `scale`, mid's one user
//     is promoted as `mid_scale`, and Chain initializes the three.
//
// Styles (Style):
//   unfused   every component is its own kernel
//   parallel  as unfused, but BiCGK's gemv_q and gtp_s are a system_from group: a plain System, TWO launches (the two
//             stages are independent, so they overlap through data dependencies alone; no fused group is involved)
//   fused     GEMVER: fuse_components(connects, segment<range_mode::whole>(r2u, gtp)), then cred, gemv;
//             BiCGK : fuse_components(gemv_q, gtp_s), each stage its own segment, no boundary: ONE launch, then cred_s
//   onepass   fused GEMVERs as above, and BiCGK from ONE pass over B (BicgkOnePass, then RowReduce and ColReduce)
//   naive     the SAME components, fused with global barriers into three launches for the whole chain:
//             G1  : fuse_components(segment<whole>(r2u, gtp), global_barrier(), cred, global_barrier(), gemv)
//             BiCGK: fuse_components(onepass, global_barrier(), segment(rowred, cred_s))
//             G2  : as G1
//   crafted   components written to be fused (components.hpp, bottom), B tiles kept in local memory; three launches:
//             g1 : segment<local_mem::shared>(r2u_local, gtp_local), global_barrier(), cred, global_barrier(), gemv
//             mid: segment<local_mem::shared>(bicgk_tile, r2u_from_local), global_barrier(), segment(rowred, cred_s)
//                  (BiCGK and G2's rank-2 update share ONE read of B1)
//             g2 : gtp (reads B2 from global memory), global_barrier(), cred, global_barrier(), gemv

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "components.hpp"

namespace gc {

// onepass: fused GEMVERs, one-pass BiCGK. naive: three launches with global barriers. crafted: three launches, local-memory tiles.
enum class Style { unfused, parallel, fused, onepass, naive, crafted };

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
        const float       ab     = 1.0f / std::sqrt(static_cast<float>(n_));  // alpha = beta
        const std::size_t strips = static_cast<std::size_t>(n_ / R_), n = static_cast<std::size_t>(n_);
        std::string       pre;  // the path prefix of r2u and gtp
        if (style_ == Style::naive) {
            // The whole GEMVER in ONE launch: r2u and gtp share their (c, j) mapping; cred needs every strip's partials and
            // gemv the whole x, so a global barrier stands before each. The connects are inside the group.
            add_subsystem(cs::fuse_components(
                "fw", cs::connects_desc({{"r2u.B", "gtp.B"}, {"gtp.P", "cred.P"}, {"r2u.B", "gemv.B"}, {"cred.x", "gemv.x"}}),
                cs::segment<cs::range_mode::whole>(cs::subsystem_desc<Rank2Update>("r2u", R_), cs::subsystem_desc<GemvTPartial>("gtp")),
                cs::global_barrier(), cs::subsystem_desc<ColReduce<true>>("cred"), cs::global_barrier(),
                cs::subsystem_desc<Gemv>("gemv")));
            promote<DType::f32, 2>("fw.r2u.A", "A");
            promote<DType::f32, 1>("fw.r2u.u1", "u1");
            promote<DType::f32, 1>("fw.r2u.v1", "v1");
            promote<DType::f32, 1>("fw.r2u.u2", "u2");
            promote<DType::f32, 1>("fw.r2u.v2", "v2");
            promote<DType::f32, 1>("fw.gtp.y", "y");
            promote<DType::f32, 1>("fw.cred.z", "z");
            promote<DType::f32, 2>("fw.r2u.B", "B");
            promote<DType::f32, 1>("fw.cred.x", "x");
            promote<DType::f32, 1>("fw.gemv.y", "w");
            shape_hook("fw.gtp.P", cs::dim::fixed_at_run(strips), cs::dim::fixed_at_run(n));
            tie_inputs<DType::f32, 1>({"fw.cred.scale", "fw.gemv.scale"}, "scale", cs::shape(1));
            init_input("scale", cs::init::fill(ab));
            return;
        }
        if (style_ == Style::fused || style_ == Style::onepass) {
            // r2u and gtp share their (c, j) mapping, so the same work-item reads its own B writes.
            // (the work-group size 256 comes from the stages' own launch configs)
            add_subsystem(cs::fuse_components(
                "fw", cs::connects_desc({{"r2u.B", "gtp.B"}}),
                cs::segment<cs::range_mode::whole>(cs::subsystem_desc<Rank2Update>("r2u", R_), cs::subsystem_desc<GemvTPartial>("gtp"))));
            pre = "fw.";
        } else {
            add_subsystem(cs::subsystem_desc<Rank2Update>("r2u", R_));
            add_subsystem(cs::subsystem_desc<GemvTPartial>("gtp"));
            connect("r2u.B", "gtp.B");
        }
        add_subsystem(cs::subsystem_desc<ColReduce<true>>("cred"));
        add_subsystem(cs::subsystem_desc<Gemv>("gemv"));

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

        shape_hook(pre + "gtp.P", cs::dim::fixed_at_run(strips), cs::dim::fixed_at_run(n));
        tie_inputs<DType::f32, 1>({"cred.scale", "gemv.scale"}, "scale", cs::shape(1));
        init_input("scale", cs::init::fill(ab));
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
        const std::size_t strips = static_cast<std::size_t>(n_ / R_), n = static_cast<std::size_t>(n_);
        if (style_ == Style::naive) {
            // The whole BiCGK in ONE launch: the one-pass kernel, a global barrier, then the two reducers in one segment.
            add_subsystem(cs::fuse_components(
                "fw", cs::connects_desc({{"onepass.Pq", "rowred.Pq"}, {"onepass.Ps", "cred_s.P"}}),
                cs::subsystem_desc<BicgkOnePass>("onepass"), cs::global_barrier(),
                cs::segment(cs::subsystem_desc<RowReduce>("rowred"), cs::subsystem_desc<ColReduce<false>>("cred_s"))));
            promote<DType::f32, 2>("fw.onepass.B", "B");
            promote<DType::f32, 1>("fw.onepass.p", "p");
            promote<DType::f32, 1>("fw.onepass.r", "r");
            promote<DType::f32, 1>("fw.rowred.q", "q");
            promote<DType::f32, 1>("fw.cred_s.x", "s");
            shape_hook("fw.onepass.Ps", cs::dim::fixed_at_run(strips), cs::dim::fixed_at_run(n));
            promote<DType::f32, 1>("fw.cred_s.scale", "scale");  // one user: a promotion (a tie needs two members)
            init_input("scale", cs::init::fill(1.0f, {1}));
            return;
        }
        if (style_ == Style::onepass) {
            // q and s from ONE pass over B: the one-pass kernel, then the two reducers.
            add_subsystem(cs::subsystem_desc<BicgkOnePass>("onepass"));
            add_subsystem(cs::subsystem_desc<RowReduce>("rowred"));
            add_subsystem(cs::subsystem_desc<ColReduce<false>>("cred_s"));
            connect("onepass.Pq", "rowred.Pq");
            connect("onepass.Ps", "cred_s.P");
            promote<DType::f32, 2>("onepass.B", "B");
            promote<DType::f32, 1>("onepass.p", "p");
            promote<DType::f32, 1>("onepass.r", "r");
            promote<DType::f32, 1>("rowred.q", "q");
            promote<DType::f32, 1>("cred_s.x", "s");
            shape_hook("onepass.Ps", cs::dim::fixed_at_run(strips), cs::dim::fixed_at_run(n));
            promote<DType::f32, 1>("cred_s.scale", "scale");  // one user: a promotion (a tie needs two members)
            init_input("scale", cs::init::fill(1.0f, {1}));
            return;
        }
        std::string pre;  // the path prefix of gemv_q and gtp_s
        if (style_ == Style::parallel) {
            // A plain group of the two children: they share no variable, so nothing orders them and they overlap.
            add_subsystem(cs::system_from("par", cs::subsystem_desc<Gemv>("gemv_q"), cs::subsystem_desc<GemvTPartial>("gtp_s")));
            pre = "par.";
        } else if (style_ == Style::fused) {
            // Independent stages: each is a one-stage segment with its own slice of groups.
            add_subsystem(cs::fuse_components("fw", cs::subsystem_desc<Gemv>("gemv_q"), cs::subsystem_desc<GemvTPartial>("gtp_s")));
            pre = "fw.";
        } else {
            add_subsystem(cs::subsystem_desc<Gemv>("gemv_q"));
            add_subsystem(cs::subsystem_desc<GemvTPartial>("gtp_s"));
        }
        add_subsystem(cs::subsystem_desc<ColReduce<false>>("cred_s"));

        connect(pre + "gtp_s.P", "cred_s.P");
        // One B feeds both kernels; both declare B rank-only. The tie alias is connected to the producer, which gives it its shape.
        tie_inputs<DType::f32, 2>({pre + "gemv_q.B", pre + "gtp_s.B"}, "B");

        promote<DType::f32, 1>(pre + "gemv_q.x", "p");
        promote<DType::f32, 1>(pre + "gtp_s.y", "r");
        promote<DType::f32, 1>(pre + "gemv_q.y", "q");
        promote<DType::f32, 1>("cred_s.x", "s");

        shape_hook(pre + "gtp_s.P", cs::dim::fixed_at_run(strips), cs::dim::fixed_at_run(n));
        tie_inputs<DType::f32, 1>({pre + "gemv_q.scale", "cred_s.scale"}, "scale", cs::shape(1));
        init_input("scale", cs::init::fill(1.0f));
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
        if (style_ == Style::crafted) {
            setup_crafted();
            return;
        }
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

        // The free inputs: every dimension is n. (G2's A, y and z are connected, BiCGK's inputs too.)
        const auto n = cs::dim::fixed_at_run(static_cast<std::size_t>(n_));
        shape_hook("g1.A", n, n);
        for (const char* v : {"g1.u1", "g1.v1", "g1.u2", "g1.v2", "g1.y", "g1.z", "g2.u1", "g2.v1", "g2.u2", "g2.v2"}) shape_hook(v, n);
    }

private:
    /// The crafted chain: three fused launches (g1, mid, g2) over the components written for local memory. It promotes the
    /// same names as the other styles. Tiles are kRt rows by 256 columns (R is kRt here, passed by the driver: the strips of
    /// the reducers are n / kRt).
    void setup_crafted() {
        const float       ab     = 1.0f / std::sqrt(static_cast<float>(n_));
        const std::size_t n      = static_cast<std::size_t>(n_);
        const std::size_t strips = static_cast<std::size_t>(n_ / R_);

        // G1: the B tile stays in local memory from the rank-2 update to the partial column sums; then x needs every strip's
        // partials (barrier), and w needs the whole x (barrier).
        add_subsystem(cs::fuse_components(
            "g1", cs::connects_desc({{"gtp.P", "cred.P"}, {"r2u.B", "gemv.B"}, {"cred.x", "gemv.x"}}),
            cs::ties_desc({cs::tie_inputs<DType::f32, 1>({"cred.scale", "gemv.scale"}, "scale", cs::shape(1))}),
            cs::segment<cs::local_mem::shared>(cs::subsystem_desc<Rank2UpdateLocal>("r2u"), cs::subsystem_desc<GemvTPartialLocal>("gtp")),
            cs::global_barrier(), cs::subsystem_desc<ColReduce<true>>("cred"), cs::global_barrier(),
            cs::subsystem_desc<Gemv>("gemv")));

        // The middle launch: BiCGK and G2's rank-2 update share ONE read of B1 (G2's update does not depend on q or s).
        add_subsystem(cs::fuse_components(
            "mid", cs::connects_desc({{"bk.Pq", "rowred.Pq"}, {"bk.Ps", "cred_s.P"}}),
            cs::segment<cs::local_mem::shared>(cs::subsystem_desc<BicgkTileLocal>("bk"), cs::subsystem_desc<Rank2UpdateFromLocal>("r2u")),
            cs::global_barrier(), cs::segment(cs::subsystem_desc<RowReduce>("rowred"), cs::subsystem_desc<ColReduce<false>>("cred_s"))));

        // G2: gtp needs y = q, so it reads B2 from global memory.
        add_subsystem(cs::fuse_components(
            "g2", cs::connects_desc({{"gtp.P", "cred.P"}, {"cred.x", "gemv.x"}}),
            cs::ties_desc({cs::tie_inputs<DType::f32, 1>({"cred.scale", "gemv.scale"}, "scale", cs::shape(1))}),
            cs::subsystem_desc<GemvTPartial>("gtp"), cs::global_barrier(), cs::subsystem_desc<ColReduce<true>>("cred"),
            cs::global_barrier(), cs::subsystem_desc<Gemv>("gemv")));

        connect("g1.r2u.B", "mid.bk.B");    // B1
        connect("g1.cred.x", "mid.bk.p");   // p = x1
        connect("g1.gemv.y", "mid.bk.r");   // r = w1
        connect("mid.r2u.B", "g2.gtp.B");   // B2 feeds both of G2's readers
        connect("mid.r2u.B", "g2.gemv.B");
        connect("mid.rowred.q", "g2.gtp.y");  // y = q
        connect("mid.cred_s.x", "g2.cred.z"); // z = s

        promote<DType::f32, 2>("g1.r2u.A", "A");
        promote<DType::f32, 1>("g1.r2u.u1", "u1a");
        promote<DType::f32, 1>("g1.r2u.v1", "v1a");
        promote<DType::f32, 1>("g1.r2u.u2", "u2a");
        promote<DType::f32, 1>("g1.r2u.v2", "v2a");
        promote<DType::f32, 1>("g1.gtp.y", "ya");
        promote<DType::f32, 1>("g1.cred.z", "za");
        promote<DType::f32, 1>("mid.r2u.u1", "u1b");
        promote<DType::f32, 1>("mid.r2u.v1", "v1b");
        promote<DType::f32, 1>("mid.r2u.u2", "u2b");
        promote<DType::f32, 1>("mid.r2u.v2", "v2b");
        promote<DType::f32, 2>("g1.r2u.B", "B1");
        promote<DType::f32, 1>("g1.cred.x", "x1");
        promote<DType::f32, 1>("g1.gemv.y", "w1");
        promote<DType::f32, 1>("mid.rowred.q", "q");
        promote<DType::f32, 1>("mid.cred_s.x", "s");
        promote<DType::f32, 1>("mid.cred_s.scale", "mid_scale");  // one user: a promotion (a tie needs two members)
        promote<DType::f32, 2>("mid.r2u.B", "B2");
        promote<DType::f32, 1>("g2.cred.x", "x2");
        promote<DType::f32, 1>("g2.gemv.y", "w2");

        // The free inputs: every dimension is n.
        const auto dn = cs::dim::fixed_at_run(n);
        shape_hook("g1.r2u.A", dn, dn);
        for (const char* v : {"g1.r2u.u1", "g1.r2u.v1", "g1.r2u.u2", "g1.r2u.v2", "g1.gtp.y", "g1.cred.z", "mid.r2u.u1", "mid.r2u.v1",
                              "mid.r2u.u2", "mid.r2u.v2"})
            shape_hook(v, dn);
        // The partial sums: (n / kRt, n), the strips of the tiles.
        const auto ds = cs::dim::fixed_at_run(strips);
        shape_hook("g1.gtp.P", ds, dn);
        shape_hook("mid.bk.Ps", ds, dn);
        shape_hook("g2.gtp.P", ds, dn);

        init_input("g1.scale", cs::init::fill(ab));
        init_input("mid_scale", cs::init::fill(1.0f, {1}));
        init_input("g2.scale", cs::init::fill(ab));
    }
};

}  // namespace gc
