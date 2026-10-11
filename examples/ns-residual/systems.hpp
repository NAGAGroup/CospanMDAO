#pragma once

// =============================================================================
// examples/ns-residual/systems.hpp — the residual evaluation as Systems: unfused and fused
// =============================================================================
//
// Paths the driver uses (the two roots differ only by the "ns." prefix of the fused group):
//   input   U                       the conserved state (4, ny, nx)
//   output  R                       the residual (4, ny, nx)
//           Dt Ws Sq                the ATOMIC diagnostics: min local timestep, max wave speed, the four sums of R^2
//           detn.Det detd.Det detw.Det   the DETERMINISTIC ones (norms, min dt, max wave speed)
//           err.Err                 atomic - deterministic, [dt, ws, sq0..sq3]
//
//   NsUnfused<Lim,Bc>  every stage its own kernel through global memory (padded arrays for the tile):
//                      init, pad (load + boundary), prim, grad, recon, cfx, cfy, vfx, vfy, cdiv, vdiv, dt, ws, rsq, detn,
//                      detd, detw, err
//   NsFused<Lim,Bc>    ONE launch, five segments (the README draws them):
//                        0 init (1 group)
//                        1 tile pipeline (shared local memory): load, bc, prim, grad, rec, cfx, cfy, vfx, vfy
//                        2 divergence and diagnostics (partitioned): cdiv, vdiv, dtk, wsk, rsq
//                        global_barrier()
//                        3 deterministic reductions (partitioned, own): detn (4 groups), detd (1), detw (1)
//                        4 err (1 group)

#include <naga/cospan/cospan.hpp>

#include "components.hpp"

namespace nsr {

template <class Lim, class Bc>
class NsUnfused : public cs::System {
    int nx_, ny_;

public:
    NsUnfused(int nx, int ny) : nx_{nx}, ny_{ny} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<InitDiagnostics>("init"));
        add_subsystem(cs::subsystem_desc<PadBoundary<Bc>>("pad", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ToPrimitiveG>("prim", nx_, ny_));
        add_subsystem(cs::subsystem_desc<GradientsG>("grad", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ReconstructG<Lim>>("rec", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ConvFluxG<0>>("cfx", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ConvFluxG<1>>("cfy", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ViscFluxG<0>>("vfx", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ViscFluxG<1>>("vfy", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ConvDivergence<false>>("cdiv", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ViscDivergenceChained>("vdiv", nx_, ny_));
        add_subsystem(cs::subsystem_desc<LocalTimestep<false>>("dtk", nx_, ny_));
        add_subsystem(cs::subsystem_desc<WaveSpeed<false>>("wsk", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ResidualSquares<false>>("rsq", nx_, ny_));
        add_subsystem(cs::subsystem_desc<DetNorm>("detn", nx_, ny_));
        add_subsystem(cs::subsystem_desc<DetMin>("detd", nx_, ny_));
        add_subsystem(cs::subsystem_desc<DetMax>("detw", nx_, ny_));
        add_subsystem(cs::subsystem_desc<ErrorStage<false>>("err"));

        connect("pad.P", "prim.P");
        for (const char* c : {"grad", "rec", "cfx", "cfy", "vfx", "vfy"}) connect("prim.W", std::string{c} + ".W");
        connect("rec.S", "cfx.S");
        connect("rec.S", "cfy.S");
        connect("grad.G", "vfx.G");
        connect("grad.G", "vfy.G");
        connect("cfx.F", "cdiv.Fx");
        connect("cfy.F", "cdiv.Fy");
        connect("cdiv.R", "vdiv.Rc");
        connect("vfx.F", "vdiv.Fx");
        connect("vfy.F", "vdiv.Fy");
        connect("vdiv.R", "rsq.R");
        connect("rsq.Part", "detn.Part");
        connect("dtk.Part", "detd.Part");
        connect("wsk.Part", "detw.Part");
        connect("detn.Det", "err.DetSq");
        connect("detd.Det", "err.DetDt");
        connect("detw.Det", "err.DetWs");

        // the state is read by the boundary kernel and the two per-cell diagnostics
        tie_inputs<DType::f32, 3>({"pad.U", "dtk.U", "wsk.U"}, "U", cs::shape(4, ny_, nx_));
        // the grid (nx, ny, tx, ty): read by the boundary kernel and the deterministic reductions; the driver sets it
        tie_inputs<DType::i32, 1>({"pad.geo", "detn.geo", "detd.geo", "detw.geo"}, "geo", cs::shape(4));
        // the atomic accumulators: zeroed by init, accumulated by the diagnostics, read by err
        tie_outputs<DType::atomic_f32, 1>({"init.Dt", "dtk.Dt", "err.Dt"}, "Dt", cs::shape(1));
        tie_outputs<DType::atomic_f32, 1>({"init.Ws", "wsk.Ws", "err.Ws"}, "Ws", cs::shape(1));
        tie_outputs<DType::atomic_f32, 1>({"init.Sq", "rsq.Sq", "err.Sq"}, "Sq", cs::shape(4));
        promote<DType::f32, 3>("vdiv.R", "R");
    }
};

template <class Lim, class Bc, class Split = cs::range_split::flatten_and_split>
class NsFused : public cs::System {
    int nx_, ny_;

public:
    NsFused(int nx, int ny) : nx_{nx}, ny_{ny} {}
    void setup() override {
        add_subsystem(cs::fuse_components<cs::group_waits_enabled, Split>(
            "ns",
            cs::connects_desc({{"cfx.F", "cdiv.Fx"},
                               {"cfy.F", "cdiv.Fy"},
                               {"vfx.F", "vdiv.Fx"},
                               {"vfy.F", "vdiv.Fy"},
                               {"R", "rsq.R"},  // from the tie alias: a tied member cannot be a source
                               {"rsq.Part", "detn.Part"},
                               {"dtk.Part", "detd.Part"},
                               {"wsk.Part", "detw.Part"},
                               {"detn.Det", "err.DetSq"},
                               {"detd.Det", "err.DetDt"},
                               {"detw.Det", "err.DetWs"}}),
            cs::ties_desc({cs::tie_inputs<DType::f32, 3>({"load.U", "bc.U", "dtk.U", "wsk.U"}, "U", cs::shape(4, ny_, nx_)),
                           cs::tie_inputs<DType::i32, 1>({"load.geo", "bc.geo", "cfx.geo", "cfy.geo", "vfx.geo", "vfy.geo", "cdiv.geo", "vdiv.geo",
                                                          "detn.geo", "detd.geo", "detw.geo"},
                                                         "geo", cs::shape(4)),
                           cs::tie_outputs<DType::f32, 3>({"cdiv.R", "vdiv.R"}, "R", cs::shape(4, ny_, nx_), cs::TieOutputsPolicy::allow_plain),
                           cs::tie_outputs<DType::atomic_f32, 1>({"init.Dt", "dtk.Dt", "err.Dt"}, "Dt", cs::shape(1)),
                           cs::tie_outputs<DType::atomic_f32, 1>({"init.Ws", "wsk.Ws", "err.Ws"}, "Ws", cs::shape(1)),
                           cs::tie_outputs<DType::atomic_f32, 1>({"init.Sq", "rsq.Sq", "err.Sq"}, "Sq", cs::shape(4))}),
            // segment 0: the atomic accumulators are reset by one group; NO barrier after it (the diagnostics wait on it)
            cs::segment(cs::subsystem_desc<InitDiagnostics>("init")),
            // segment 1: the tile pipeline; the 20 x 20 tile stays in local memory from stage to stage
            cs::segment<cs::local_mem::shared>(cs::subsystem_desc<LoadTile>("load", nx_, ny_), cs::subsystem_desc<ApplyBoundary<Bc>>("bc", nx_, ny_),
                                               cs::subsystem_desc<ToPrimitive>("prim", nx_, ny_), cs::subsystem_desc<Gradients>("grad", nx_, ny_),
                                               cs::subsystem_desc<Reconstruct<Lim>>("rec", nx_, ny_),
                                               cs::subsystem_desc<ConvFlux<0>>("cfx", nx_, ny_),
                                               cs::subsystem_desc<ConvFlux<1>>("cfy", nx_, ny_),
                                               cs::subsystem_desc<ViscFlux<0>>("vfx", nx_, ny_),
                                               cs::subsystem_desc<ViscFlux<1>>("vfy", nx_, ny_)),
            // segment 2: divergences (wait on the tile-pipeline groups they read) and the diagnostics (wait on init)
            cs::segment<cs::local_mem::partitioned>(cs::subsystem_desc<ConvDivergence<true>>("cdiv", nx_, ny_),
                                                    cs::subsystem_desc<ViscDivergence>("vdiv", nx_, ny_),
                                                    cs::subsystem_desc<LocalTimestep<true>>("dtk", nx_, ny_),
                                                    cs::subsystem_desc<WaveSpeed<true>>("wsk", nx_, ny_),
                                                    cs::subsystem_desc<ResidualSquares<true>>("rsq", nx_, ny_)),
            cs::global_barrier(),
            // segment 3: the deterministic reductions of the per-tile partials, fixed order
            cs::segment<cs::local_mem::partitioned, cs::range_mode::own>(cs::subsystem_desc<DetNorm>("detn", nx_, ny_),
                                                                         cs::subsystem_desc<DetMin>("detd", nx_, ny_),
                                                                         cs::subsystem_desc<DetMax>("detw", nx_, ny_)),
            // segment 4: atomic - deterministic, after waiting on exactly the segment-3 groups it compares
            cs::segment(cs::subsystem_desc<ErrorStage<true>>("err"))));
    }
};

}  // namespace nsr
