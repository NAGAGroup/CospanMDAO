#pragma once

// =============================================================================
// examples/ns-residual/components.hpp — the residual's stages as CoSpan components
// =============================================================================
//
// Array conventions: physics.hpp. Every field is f32; U, R: (4, ny, nx); x-face fluxes (4, ny, nx+1); y-face fluxes
// (4, ny+1, nx); per-tile partials (ty, tx) or (4, ty, tx).
//
// FUSED stages (all rank-2 work-group stages, work-group 16 x 16, one group per tile):
//   Segment "tile pipeline" (local_mem::shared; the 20 x 20 tile with halo stays in local memory from stage to stage):
//     LoadTile, ApplyBoundary<Bc>, ToPrimitive, Gradients, Reconstruct<Lim>, ConvFlux (dir), ViscFlux (dir)
//   Segment "divergence and diagnostics":
//     ConvDivergence, ViscDivergence (wait_group on the tile-pipeline groups they read), LocalTimestep, WaveSpeed,
//     ResidualSquares (wait_group on the init group right before their atomic)
//   Init: InitDiagnostics.  Deterministic reductions: DetNorm, DetMin, DetMax.  Error: ErrorStage.
//   The `fused` flag of a body switches its wait_group calls on: the same bodies run in the unfused variant as plain
//   launches (no wait state there, the data dependencies between the kernels order them).
//
// UNFUSED counterparts of the local-memory stages (the tile lives in padded global arrays): PadBoundary<Bc>,
// ToPrimitiveG, GradientsG, ReconstructG<Lim>, ConvFluxG (dir), ViscFluxG (dir), ViscDivergenceChained.

#include <cstddef>
#include <limits>
#include <memory>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "physics.hpp"
#include "tile.hpp"

namespace nsr {

namespace cs = naga::cospan;
using cs::DType;

using In3  = cs::Input<DType::f32, 3>;
using Out3 = cs::Output<DType::f32, 3>;
using In2  = cs::Input<DType::f32, 2>;
using Out2 = cs::Output<DType::f32, 2>;
using In1  = cs::Input<DType::f32, 1>;
using Out1 = cs::Output<DType::f32, 1>;
using Acc1 = cs::Output<DType::atomic_f32, 1>;

inline sycl::id<3> id3(int v, int j, int i) {
    return sycl::id<3>{static_cast<std::size_t>(v), static_cast<std::size_t>(j), static_cast<std::size_t>(i)};
}
inline sycl::id<2> id2(int j, int i) { return sycl::id<2>{static_cast<std::size_t>(j), static_cast<std::size_t>(i)}; }
inline sycl::id<1> id1(int i) { return sycl::id<1>{static_cast<std::size_t>(i)}; }

/// The work-item's place in its tile, from the group item of a 2D stage.
inline Th make_th(const cs::GroupItem<2>& it) {
    return Th{static_cast<int>(it.get_local_id(1)), static_cast<int>(it.get_local_id(0)), static_cast<int>(it.get_local_linear_id()),
              static_cast<int>(it.get_group(1)), static_cast<int>(it.get_group(0)), it.get_group().to_sycl()};
}

/// The segments of the fused group (indices used by wait_group).
constexpr std::size_t kSegInit = 0, kSegTile = 1, kSegDiv = 2, kSegDet = 3, kSegErr = 4;

inline sycl::range<2> tile_range(const Geo& g) {
    return sycl::range<2>{static_cast<std::size_t>(g.ny), static_cast<std::size_t>(g.nx)};
}
inline sycl::range<2> tile_wg() { return sycl::range<2>{kTile, kTile}; }

// =============================================================================
// FUSED: the tile pipeline
// =============================================================================

// ---- LoadTile -----------------------------------------------------------------------------------------------------
struct LoadTileBody {
    struct Inputs { In3 U; };
    struct Outputs {};
    Geo geo;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs) const {
        float* L = it.local<float>(static_cast<std::size_t>(kNLoc) * kLocCells);
        tile_load(make_th(it), geo, L, [&](int v, int j, int i) { return in.U[id3(v, j, i)]; });
    }
};
class LoadTile : public cs::Component<LoadTileBody> {
    In3 U_;
    Geo geo_;

public:
    LoadTile(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override { declare_input(U_, "U", cs::unspecified_shape<3>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = create_launch_config(tile_range(geo_), tile_wg(), LoadTileBody::Inputs{U_}, LoadTileBody::Outputs{}, LoadTileBody{geo_});
        d->request_local_bytes(kLocalBytes);
        return d;
    }
};

// ---- ApplyBoundary<Bc> --------------------------------------------------------------------------------------------
template <class Bc>
struct ApplyBoundaryBody {
    struct Inputs { In3 U; };
    struct Outputs {};
    Geo geo;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs) const {
        float* L = it.local<float>(static_cast<std::size_t>(kNLoc) * kLocCells);
        tile_boundary<Bc>(make_th(it), geo, L, [&](int v, int j, int i) { return in.U[id3(v, j, i)]; });
    }
};
template <class Bc>
class ApplyBoundary : public cs::Component<ApplyBoundaryBody<Bc>> {
    In3 U_;
    Geo geo_;

public:
    ApplyBoundary(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override { this->declare_input(U_, "U", cs::unspecified_shape<3>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        using B = ApplyBoundaryBody<Bc>;
        auto d = this->create_launch_config(tile_range(geo_), tile_wg(), typename B::Inputs{U_}, typename B::Outputs{}, B{geo_});
        d->request_local_bytes(kLocalBytes);
        return d;
    }
};

// ---- ToPrimitive, Gradients (no ports: they work on the local tile) ----------------------------------------------------
struct ToPrimitiveBody {
    struct Inputs {};
    struct Outputs {};
    Geo geo;
    void operator()(cs::GroupItem<2> it, Inputs, Outputs) const {
        float* L = it.local<float>(static_cast<std::size_t>(kNLoc) * kLocCells);
        tile_to_primitive(make_th(it), L);
    }
};
class ToPrimitive : public cs::Component<ToPrimitiveBody> {
    Geo geo_;

public:
    ToPrimitive(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {}
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = create_launch_config(tile_range(geo_), tile_wg(), ToPrimitiveBody::Inputs{}, ToPrimitiveBody::Outputs{}, ToPrimitiveBody{geo_});
        d->request_local_bytes(kLocalBytes);
        return d;
    }
};

struct GradientsBody {
    struct Inputs {};
    struct Outputs {};
    Geo geo;
    void operator()(cs::GroupItem<2> it, Inputs, Outputs) const {
        float* L = it.local<float>(static_cast<std::size_t>(kNLoc) * kLocCells);
        tile_gradients(make_th(it), L);
    }
};
class Gradients : public cs::Component<GradientsBody> {
    Geo geo_;

public:
    Gradients(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {}
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = create_launch_config(tile_range(geo_), tile_wg(), GradientsBody::Inputs{}, GradientsBody::Outputs{}, GradientsBody{geo_});
        d->request_local_bytes(kLocalBytes);
        return d;
    }
};

// ---- Reconstruct<Lim> ---------------------------------------------------------------------------------------------
template <class Lim>
struct ReconstructBody {
    struct Inputs {};
    struct Outputs {};
    Geo geo;
    void operator()(cs::GroupItem<2> it, Inputs, Outputs) const {
        float* L = it.local<float>(static_cast<std::size_t>(kNLoc) * kLocCells);
        tile_reconstruct<Lim>(make_th(it), L);
    }
};
template <class Lim>
class Reconstruct : public cs::Component<ReconstructBody<Lim>> {
    Geo geo_;

public:
    Reconstruct(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {}
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        using B = ReconstructBody<Lim>;
        auto d = this->create_launch_config(tile_range(geo_), tile_wg(), typename B::Inputs{}, typename B::Outputs{}, B{geo_});
        d->request_local_bytes(kLocalBytes);
        return d;
    }
};

// ---- ConvFlux / ViscFlux: ONE type each, instanced for x (dir 0) and y (dir 1) --------------------------------------
struct ConvFluxBody {
    struct Inputs {};
    struct Outputs { Out3 F; };
    Geo geo;
    int dir;
    void operator()(cs::GroupItem<2> it, Inputs, Outputs out) const {
        const float* L = it.local<float>(static_cast<std::size_t>(kNLoc) * kLocCells);
        tile_conv_flux(make_th(it), geo, L, dir, [&](int v, int j, int i, float x) { out.F[id3(v, j, i)] = x; });
    }
};
struct ViscFluxBody {
    struct Inputs {};
    struct Outputs { Out3 F; };
    Geo geo;
    int dir;
    void operator()(cs::GroupItem<2> it, Inputs, Outputs out) const {
        const float* L = it.local<float>(static_cast<std::size_t>(kNLoc) * kLocCells);
        tile_visc_flux(make_th(it), geo, L, dir, [&](int v, int j, int i, float x) { out.F[id3(v, j, i)] = x; });
    }
};
template <class Body>
class FaceFlux : public cs::Component<Body> {
    Out3 F_;
    Geo  geo_;
    int  dir_;

public:
    FaceFlux(int nx, int ny, int dir) : geo_{make_geo(nx, ny)}, dir_{dir} {}
    void setup() override {
        // x-faces: (4, ny, nx+1); y-faces: (4, ny+1, nx)
        this->declare_output(F_, "F", 4, geo_.ny + (dir_ == 1 ? 1 : 0), geo_.nx + (dir_ == 0 ? 1 : 0));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = this->create_launch_config(tile_range(geo_), tile_wg(), typename Body::Inputs{}, typename Body::Outputs{F_}, Body{geo_, dir_});
        d->request_local_bytes(kLocalBytes);
        return d;
    }
};
using ConvFlux = FaceFlux<ConvFluxBody>;
using ViscFlux = FaceFlux<ViscFluxBody>;

// =============================================================================
// Divergence and diagnostics (shared by the fused and unfused variants; `fused` turns the waits on)
// =============================================================================

/// wait_group on the tile-pipeline groups whose faces this tile's cells read: its own, the east and the north neighbour.
inline void wait_tile_neighbours(const cs::GroupItem<2>& it, const Th& t, const Geo& g) {
    const std::size_t gy = static_cast<std::size_t>(t.gy), gx = static_cast<std::size_t>(t.gx);
    it.wait_group(kSegTile, sycl::id<2>{gy, gx});
    if (t.gx + 1 < g.tx) it.wait_group(kSegTile, sycl::id<2>{gy, gx + 1});
    if (t.gy + 1 < g.ty) it.wait_group(kSegTile, sycl::id<2>{gy + 1, gx});
}

// ---- ConvDivergence: R = -div F_conv (writes R) ---------------------------------------------------------------------
struct ConvDivBody {
    struct Inputs { In3 Fx, Fy; };
    struct Outputs { Out3 R; };
    Geo  geo;
    bool fused;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out) const {
        const Th t = make_th(it);
        if (fused) wait_tile_neighbours(it, t, geo);
        cell_divergence(
            t, -1.0f, [&](int v, int j, int i) { return in.Fx[id3(v, j, i)]; }, [&](int v, int j, int i) { return in.Fy[id3(v, j, i)]; },
            [&](int v, int j, int i, float x) { out.R[id3(v, j, i)] = x; });
    }
};
class ConvDivergence : public cs::Component<ConvDivBody> {
    In3  Fx_, Fy_;
    Out3 R_;
    Geo  geo_;
    bool fused_, tied_r_;

public:
    /// tied_r: R is a member of a tie (its shape comes from the alias)
    ConvDivergence(int nx, int ny, bool fused, bool tied_r) : geo_{make_geo(nx, ny)}, fused_{fused}, tied_r_{tied_r} {}
    void setup() override {
        declare_input(Fx_, "Fx", 4, geo_.ny, geo_.nx + 1);
        declare_input(Fy_, "Fy", 4, geo_.ny + 1, geo_.nx);
        if (tied_r_) declare_output(R_, "R", cs::unspecified_shape<3>());
        else declare_output(R_, "R", 4, geo_.ny, geo_.nx);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(tile_range(geo_), tile_wg(), ConvDivBody::Inputs{Fx_, Fy_}, ConvDivBody::Outputs{R_}, ConvDivBody{geo_, fused_});
    }
};

// ---- ViscDivergence: R += div F_visc (fused: R is the tied output the convective part wrote) ------------------------
struct ViscDivBody {
    struct Inputs { In3 Fx, Fy; };
    struct Outputs { Out3 R; };
    Geo  geo;
    bool fused;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out) const {
        const Th t = make_th(it);
        if (fused) wait_tile_neighbours(it, t, geo);
        cell_divergence(
            t, 1.0f, [&](int v, int j, int i) { return in.Fx[id3(v, j, i)]; }, [&](int v, int j, int i) { return in.Fy[id3(v, j, i)]; },
            [&](int v, int j, int i, float x) { out.R[id3(v, j, i)] += x; });
    }
};
class ViscDivergence : public cs::Component<ViscDivBody> {
    In3  Fx_, Fy_;
    Out3 R_;
    Geo  geo_;
    bool fused_;

public:
    ViscDivergence(int nx, int ny, bool fused) : geo_{make_geo(nx, ny)}, fused_{fused} {}
    void setup() override {
        declare_input(Fx_, "Fx", 4, geo_.ny, geo_.nx + 1);
        declare_input(Fy_, "Fy", 4, geo_.ny + 1, geo_.nx);
        declare_output(R_, "R", cs::unspecified_shape<3>());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(tile_range(geo_), tile_wg(), ViscDivBody::Inputs{Fx_, Fy_}, ViscDivBody::Outputs{R_}, ViscDivBody{geo_, fused_});
    }
};

// ---- the unfused viscous divergence: R = Rc + div F_visc into its own output (no tie between the two kernels) ----------
struct ViscDivChainedBody {
    struct Inputs { In3 Rc, Fx, Fy; };
    struct Outputs { Out3 R; };
    Geo geo;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out) const {
        cell_divergence(
            make_th(it), 1.0f, [&](int v, int j, int i) { return in.Fx[id3(v, j, i)]; }, [&](int v, int j, int i) { return in.Fy[id3(v, j, i)]; },
            [&](int v, int j, int i, float x) { out.R[id3(v, j, i)] = in.Rc[id3(v, j, i)] + x; });
    }
};
class ViscDivergenceChained : public cs::Component<ViscDivChainedBody> {
    In3  Rc_, Fx_, Fy_;
    Out3 R_;
    Geo  geo_;

public:
    ViscDivergenceChained(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {
        declare_input(Rc_, "Rc", 4, geo_.ny, geo_.nx);
        declare_input(Fx_, "Fx", 4, geo_.ny, geo_.nx + 1);
        declare_input(Fy_, "Fy", 4, geo_.ny + 1, geo_.nx);
        declare_output(R_, "R", 4, geo_.ny, geo_.nx);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(tile_range(geo_), tile_wg(), ViscDivChainedBody::Inputs{Rc_, Fx_, Fy_}, ViscDivChainedBody::Outputs{R_},
                                    ViscDivChainedBody{geo_});
    }
};

// ---- InitDiagnostics: the atomic accumulators (min dt: +inf, max wave speed: -inf, the four sums: 0) -----------------
struct InitDiagBody {
    struct Inputs {};
    struct Outputs { Acc1 Dt, Ws, Sq; };
    void operator()(cs::GroupItem<2> it, Inputs, Outputs out) const {
        if (it.get_local_linear_id() == 0) {
            out.Dt[id1(0)].store(std::numeric_limits<float>::infinity());
            out.Ws[id1(0)].store(-std::numeric_limits<float>::infinity());
            for (int v = 0; v < 4; ++v) out.Sq[id1(v)].store(0.0f);
        }
    }
};
class InitDiagnostics : public cs::Component<InitDiagBody> {
    Acc1 Dt_, Ws_, Sq_;

public:
    void setup() override {
        declare_output(Dt_, "Dt", cs::unspecified_shape<1>());
        declare_output(Ws_, "Ws", cs::unspecified_shape<1>());
        declare_output(Sq_, "Sq", cs::unspecified_shape<1>());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(tile_wg(), tile_wg(), InitDiagBody::Inputs{}, InitDiagBody::Outputs{Dt_, Ws_, Sq_}, InitDiagBody{});
    }
};

// ---- LocalTimestep / WaveSpeed: a group reduction, then the atomic and the per-tile partial ------------------------
// Each waits (collectively, right before its atomic) on the init group when fused.
struct LocalTimestepBody {
    struct Inputs { In3 U; };
    struct Outputs { Acc1 Dt; Out2 Part; };
    Geo  geo;
    bool fused;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out) const {
        const Th t = make_th(it);
        float dt, ws;
        cell_dt_ws(t, [&](int v, int j, int i) { return in.U[id3(v, j, i)]; }, dt, ws);
        const float m = tile_min(t, dt);
        if (fused) it.wait_group(kSegInit, sycl::id<2>{0, 0});
        if (t.tid == 0) {
            out.Dt[id1(0)].fetch_min(m);
            out.Part[id2(t.gy, t.gx)] = m;
        }
    }
};
struct WaveSpeedBody {
    struct Inputs { In3 U; };
    struct Outputs { Acc1 Ws; Out2 Part; };
    Geo  geo;
    bool fused;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out) const {
        const Th t = make_th(it);
        float dt, ws;
        cell_dt_ws(t, [&](int v, int j, int i) { return in.U[id3(v, j, i)]; }, dt, ws);
        const float m = tile_max(t, ws);
        if (fused) it.wait_group(kSegInit, sycl::id<2>{0, 0});
        if (t.tid == 0) {
            out.Ws[id1(0)].fetch_max(m);
            out.Part[id2(t.gy, t.gx)] = m;
        }
    }
};
/// LocalTimestep (port Dt) and WaveSpeed (port Ws): the same component shape.
template <class Body, bool IsDt>
class TileExtreme : public cs::Component<Body> {
    In3  U_;
    Acc1 A_;
    Out2 Part_;
    Geo  geo_;
    bool fused_;

public:
    TileExtreme(int nx, int ny, bool fused) : geo_{make_geo(nx, ny)}, fused_{fused} {}
    void setup() override {
        this->declare_input(U_, "U", cs::unspecified_shape<3>());
        this->declare_output(A_, IsDt ? "Dt" : "Ws", cs::unspecified_shape<1>());
        this->declare_output(Part_, "Part", geo_.ty, geo_.tx);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(tile_range(geo_), tile_wg(), typename Body::Inputs{U_}, typename Body::Outputs{A_, Part_}, Body{geo_, fused_});
    }
};
using LocalTimestep = TileExtreme<LocalTimestepBody, true>;
using WaveSpeed     = TileExtreme<WaveSpeedBody, false>;

// ---- ResidualSquares: the four sums of R^2 over the tile ---------------------------------------------------------------
struct ResidualSquaresBody {
    struct Inputs { In3 R; };
    struct Outputs { Acc1 Sq; Out3 Part; };
    Geo  geo;
    bool fused;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out) const {
        const Th t = make_th(it);
        const int j = t.gy * kTile + t.ly, i = t.gx * kTile + t.lx;
        float s[4];
        for (int v = 0; v < 4; ++v) {
            const float r = in.R[id3(v, j, i)];
            s[v]          = tile_sum(t, r * r);
        }
        if (fused) it.wait_group(kSegInit, sycl::id<2>{0, 0});
        if (t.tid == 0) {
            for (int v = 0; v < 4; ++v) {
                out.Sq[id1(v)].fetch_add(s[v]);
                out.Part[id3(v, t.gy, t.gx)] = s[v];
            }
        }
    }
};
class ResidualSquares : public cs::Component<ResidualSquaresBody> {
    In3  R_;
    Acc1 Sq_;
    Out3 Part_;
    Geo  geo_;
    bool fused_;

public:
    ResidualSquares(int nx, int ny, bool fused) : geo_{make_geo(nx, ny)}, fused_{fused} {}
    void setup() override {
        declare_input(R_, "R", 4, geo_.ny, geo_.nx);
        declare_output(Sq_, "Sq", cs::unspecified_shape<1>());
        declare_output(Part_, "Part", 4, geo_.ty, geo_.tx);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(tile_range(geo_), tile_wg(), ResidualSquaresBody::Inputs{R_}, ResidualSquaresBody::Outputs{Sq_, Part_},
                                    ResidualSquaresBody{geo_, fused_});
    }
};

// =============================================================================
// The deterministic reductions of the per-tile partials, in fixed order, and the Error stage
// =============================================================================
struct DetNormBody {
    struct Inputs { In3 Part; };
    struct Outputs { Out1 Det; };
    Geo geo;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out) const {
        const Th t   = make_th(it);
        const int v  = static_cast<int>(it.get_group(0));  // one group per variable
        const int tx = geo.tx;
        const float s = det_sum(t, geo.tx * geo.ty, [&](int k) { return in.Part[id3(v, k / tx, k % tx)]; });
        if (t.tid == 0) out.Det[id1(v)] = s;
    }
};
class DetNorm : public cs::Component<DetNormBody> {
    In3  Part_;
    Out1 Det_;
    Geo  geo_;

public:
    DetNorm(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {
        declare_input(Part_, "Part", 4, geo_.ty, geo_.tx);
        declare_output(Det_, "Det", 4);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<2>{4 * kTile, kTile}, tile_wg(), DetNormBody::Inputs{Part_}, DetNormBody::Outputs{Det_}, DetNormBody{geo_});
    }
};

/// The min timestep (IsMin) or the max wave speed over the per-tile partials: one group.
template <bool IsMin>
struct DetExtremeBody {
    struct Inputs { In2 Part; };
    struct Outputs { Out1 Det; };
    Geo geo;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out) const {
        const Th t  = make_th(it);
        const int tx = geo.tx;
        auto      p = [&](int k) { return in.Part[id2(k / tx, k % tx)]; };
        const float s = IsMin ? det_min(t, geo.tx * geo.ty, p) : det_max(t, geo.tx * geo.ty, p);
        if (t.tid == 0) out.Det[id1(0)] = s;
    }
};
template <bool IsMin>
class DetExtreme : public cs::Component<DetExtremeBody<IsMin>> {
    In2  Part_;
    Out1 Det_;
    Geo  geo_;

public:
    DetExtreme(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {
        this->declare_input(Part_, "Part", geo_.ty, geo_.tx);
        this->declare_output(Det_, "Det", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        using B = DetExtremeBody<IsMin>;
        return this->create_launch_config(tile_wg(), tile_wg(), typename B::Inputs{Part_}, typename B::Outputs{Det_}, B{geo_});
    }
};
using DetMin = DetExtreme<true>;
using DetMax = DetExtreme<false>;

/// Err = (atomic result) - (deterministic result): [min dt, max wave speed, sum R0^2 .. sum R3^2]. Fused: it waits on the
/// four deterministic-reduction groups whose results it compares (groups 0-3 of that segment).
struct ErrorBody {
    struct Inputs { In1 DetSq, DetDt, DetWs; };
    struct Outputs { Acc1 Dt, Ws, Sq; Out1 Err; };
    bool fused;
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out) const {
        if (fused)
            for (std::size_t g = 0; g < 4; ++g) it.wait_group(kSegDet, sycl::id<2>{g, 0});
        if (it.get_local_linear_id() == 0) {
            out.Err[id1(0)] = out.Dt[id1(0)].load() - in.DetDt[id1(0)];
            out.Err[id1(1)] = out.Ws[id1(0)].load() - in.DetWs[id1(0)];
            for (int v = 0; v < 4; ++v) out.Err[id1(2 + v)] = out.Sq[id1(v)].load() - in.DetSq[id1(v)];
        }
    }
};
class ErrorStage : public cs::Component<ErrorBody> {
    In1  DetSq_, DetDt_, DetWs_;
    Acc1 Dt_, Ws_, Sq_;
    Out1 Err_;
    bool fused_;

public:
    explicit ErrorStage(bool fused) : fused_{fused} {}
    void setup() override {
        declare_input(DetSq_, "DetSq", 4);
        declare_input(DetDt_, "DetDt", 1);
        declare_input(DetWs_, "DetWs", 1);
        declare_output(Dt_, "Dt", cs::unspecified_shape<1>());
        declare_output(Ws_, "Ws", cs::unspecified_shape<1>());
        declare_output(Sq_, "Sq", cs::unspecified_shape<1>());
        declare_output(Err_, "Err", 6);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(tile_wg(), tile_wg(), ErrorBody::Inputs{DetSq_, DetDt_, DetWs_}, ErrorBody::Outputs{Dt_, Ws_, Sq_, Err_},
                                    ErrorBody{fused_});
    }
};

// =============================================================================
// UNFUSED counterparts of the local-memory stages: the padded tile lives in global memory
// =============================================================================
// Padded arrays have shape (nf, ny+4, nx+4): padded cell (b, a) is the domain cell (b-2, a-2).

/// LoadTile and ApplyBoundary in one kernel: the padded conserved state, interior copied, halo wrapped or mirrored.
template <class Bc>
struct PadBoundaryBody {
    struct Inputs { In3 U; };
    struct Outputs { Out3 P; };
    Geo geo;
    void operator()(sycl::id<2> idx, Inputs in, Outputs out) const {
        const int b = static_cast<int>(idx[0]), a = static_cast<int>(idx[1]);
        const int j = b - kHalo, i = a - kHalo;
        const int is = wrapi(i, geo.nx);
        int       js = j;
        float     sg = 1.0f;
        if (j < 0 || j >= geo.ny) {
            if constexpr (Bc::walls) {
                js = j < 0 ? -1 - j : 2 * geo.ny - 1 - j;
                sg = -1.0f;
            } else {
                js = wrapi(j, geo.ny);
            }
        }
        for (int v = 0; v < 4; ++v) {
            const float x = in.U[id3(v, js, is)];
            out.P[id3(v, b, a)] = (v == 1 || v == 2) ? x * sg : x;
        }
    }
};
template <class Bc>
class PadBoundary : public cs::Component<PadBoundaryBody<Bc>> {
    In3  U_;
    Out3 P_;
    Geo  geo_;

public:
    PadBoundary(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {
        this->declare_input(U_, "U", cs::unspecified_shape<3>());
        this->declare_output(P_, "P", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        using B = PadBoundaryBody<Bc>;
        return this->create_launch_config(sycl::range<2>{static_cast<std::size_t>(geo_.ny + 2 * kHalo), static_cast<std::size_t>(geo_.nx + 2 * kHalo)},
                                          typename B::Inputs{U_}, typename B::Outputs{P_}, B{geo_});
    }
};

struct ToPrimitiveGBody {
    struct Inputs { In3 P; };
    struct Outputs { Out3 W; };
    void operator()(sycl::id<2> idx, Inputs in, Outputs out) const {
        const int b = static_cast<int>(idx[0]), a = static_cast<int>(idx[1]);
        float u[4], w[4];
        for (int v = 0; v < 4; ++v) u[v] = in.P[id3(v, b, a)];
        to_prim<float>(u, w);
        for (int v = 0; v < 4; ++v) out.W[id3(v, b, a)] = w[v];
    }
};
struct GradientsGBody {
    struct Inputs { In3 W; };
    struct Outputs { Out3 G; };
    void operator()(sycl::id<2> idx, Inputs in, Outputs out) const {
        const int b = 1 + static_cast<int>(idx[0]), a = 1 + static_cast<int>(idx[1]);
        float xm[4], xp[4], ym[4], yp[4], g[6];
        for (int v = 0; v < 4; ++v) {
            xm[v] = in.W[id3(v, b, a - 1)];
            xp[v] = in.W[id3(v, b, a + 1)];
            ym[v] = in.W[id3(v, b - 1, a)];
            yp[v] = in.W[id3(v, b + 1, a)];
        }
        grad_cell<float>(xm, xp, ym, yp, g);
        for (int q = 0; q < 6; ++q) out.G[id3(q, b, a)] = g[q];
    }
};
template <class Lim>
struct ReconstructGBody {
    struct Inputs { In3 W; };
    struct Outputs { Out3 S; };
    void operator()(sycl::id<2> idx, Inputs in, Outputs out) const {
        const int b = 1 + static_cast<int>(idx[0]), a = 1 + static_cast<int>(idx[1]);
        for (int v = 0; v < 4; ++v) {
            const float w0 = in.W[id3(v, b, a)];
            out.S[id3(v, b, a)]     = Lim::template slope<float>(w0 - in.W[id3(v, b, a - 1)], in.W[id3(v, b, a + 1)] - w0);
            out.S[id3(4 + v, b, a)] = Lim::template slope<float>(w0 - in.W[id3(v, b - 1, a)], in.W[id3(v, b + 1, a)] - w0);
        }
    }
};

inline sycl::range<2> padded_range(const Geo& g, int trim) {
    return sycl::range<2>{static_cast<std::size_t>(g.ny + 2 * kHalo - 2 * trim), static_cast<std::size_t>(g.nx + 2 * kHalo - 2 * trim)};
}

class ToPrimitiveG : public cs::Component<ToPrimitiveGBody> {
    In3  P_;
    Out3 W_;
    Geo  geo_;

public:
    ToPrimitiveG(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {
        declare_input(P_, "P", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        declare_output(W_, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(padded_range(geo_, 0), ToPrimitiveGBody::Inputs{P_}, ToPrimitiveGBody::Outputs{W_}, ToPrimitiveGBody{});
    }
};

class GradientsG : public cs::Component<GradientsGBody> {
    In3  W_;
    Out3 G_;
    Geo  geo_;

public:
    GradientsG(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {
        declare_input(W_, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        declare_output(G_, "G", 6, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(padded_range(geo_, 1), GradientsGBody::Inputs{W_}, GradientsGBody::Outputs{G_}, GradientsGBody{});
    }
};

template <class Lim>
class ReconstructG : public cs::Component<ReconstructGBody<Lim>> {
    In3  W_;
    Out3 S_;
    Geo  geo_;

public:
    ReconstructG(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void setup() override {
        this->declare_input(W_, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        this->declare_output(S_, "S", 8, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        using B = ReconstructGBody<Lim>;
        return this->create_launch_config(padded_range(geo_, 1), typename B::Inputs{W_}, typename B::Outputs{S_}, B{});
    }
};

/// The face fluxes from the padded arrays; one plain launch over the faces. dir 0: x-faces (ny, nx+1), 1: y-faces (ny+1, nx).
struct ConvFluxGBody {
    struct Inputs { In3 W, S; };
    struct Outputs { Out3 F; };
    int dir;
    void operator()(sycl::id<2> idx, Inputs in, Outputs out) const {
        const int j = static_cast<int>(idx[0]), i = static_cast<int>(idx[1]);
        const int b = j + kHalo, a = i + kHalo;
        const int bl = dir == 0 ? b : b - 1, al = dir == 0 ? a - 1 : a;
        float wl[4], wr[4], F[4];
        for (int v = 0; v < 4; ++v) {
            wl[v] = in.W[id3(v, bl, al)] + 0.5f * in.S[id3(4 * dir + v, bl, al)];
            wr[v] = in.W[id3(v, b, a)] - 0.5f * in.S[id3(4 * dir + v, b, a)];
        }
        hllc<float>(wl, wr, dir, F);
        for (int v = 0; v < 4; ++v) out.F[id3(v, j, i)] = F[v];
    }
};
struct ViscFluxGBody {
    struct Inputs { In3 W, G; };
    struct Outputs { Out3 F; };
    int dir;
    void operator()(sycl::id<2> idx, Inputs in, Outputs out) const {
        const int j = static_cast<int>(idx[0]), i = static_cast<int>(idx[1]);
        const int b = j + kHalo, a = i + kHalo;
        const int bl = dir == 0 ? b : b - 1, al = dir == 0 ? a - 1 : a;
        float wl[4], wr[4], gl[6], gr[6], F[4];
        for (int v = 0; v < 4; ++v) {
            wl[v] = in.W[id3(v, bl, al)];
            wr[v] = in.W[id3(v, b, a)];
        }
        for (int q = 0; q < 6; ++q) {
            gl[q] = in.G[id3(q, bl, al)];
            gr[q] = in.G[id3(q, b, a)];
        }
        visc_flux<float>(wl, wr, gl, gr, dir, F);
        for (int v = 0; v < 4; ++v) out.F[id3(v, j, i)] = F[v];
    }
};
inline sycl::range<2> face_range(const Geo& g, int dir) {
    return sycl::range<2>{static_cast<std::size_t>(g.ny + (dir == 1 ? 1 : 0)), static_cast<std::size_t>(g.nx + (dir == 0 ? 1 : 0))};
}
class ConvFluxG : public cs::Component<ConvFluxGBody> {
    In3  W_, S_;
    Out3 F_;
    Geo  geo_;
    int  dir_;

public:
    ConvFluxG(int nx, int ny, int dir) : geo_{make_geo(nx, ny)}, dir_{dir} {}
    void setup() override {
        declare_input(W_, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        declare_input(S_, "S", 8, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        declare_output(F_, "F", 4, geo_.ny + (dir_ == 1 ? 1 : 0), geo_.nx + (dir_ == 0 ? 1 : 0));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(face_range(geo_, dir_), ConvFluxGBody::Inputs{W_, S_}, ConvFluxGBody::Outputs{F_}, ConvFluxGBody{dir_});
    }
};
class ViscFluxG : public cs::Component<ViscFluxGBody> {
    In3  W_, G_;
    Out3 F_;
    Geo  geo_;
    int  dir_;

public:
    ViscFluxG(int nx, int ny, int dir) : geo_{make_geo(nx, ny)}, dir_{dir} {}
    void setup() override {
        declare_input(W_, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        declare_input(G_, "G", 6, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        declare_output(F_, "F", 4, geo_.ny + (dir_ == 1 ? 1 : 0), geo_.nx + (dir_ == 0 ? 1 : 0));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(face_range(geo_, dir_), ViscFluxGBody::Inputs{W_, G_}, ViscFluxGBody::Outputs{F_}, ViscFluxGBody{dir_});
    }
};

}  // namespace nsr
