#pragma once

// =============================================================================
// examples/ns-residual/components.hpp — the residual's stages as CoSpan components
// =============================================================================
//
// Array conventions: physics.hpp. Every field is f32; U, R: (4, ny, nx); x-face fluxes (4, ny, nx+1); y-face fluxes
// (4, ny+1, nx); per-tile partials (ty, tx) or (4, ty, tx).
//
// The grid (nx, ny, tx, ty) is one i32 input "geo" of extent 4 on every component that needs it. The owning system ties
// those inputs into one alias, and the driver sets it with update_input.
//
// FUSED stages (all rank-2 work-group stages, work-group 16 x 16, one group per tile):
//   Segment "tile pipeline" (local_mem::shared; the 20 x 20 tile with halo stays in local memory from stage to stage):
//     LoadTile, ApplyBoundary<Bc>, ToPrimitive, Gradients, Reconstruct<Lim>, ConvFlux<Dir>, ViscFlux<Dir>
//   Segment "divergence and diagnostics":
//     ConvDivergence<true>, ViscDivergence<true> (wait_group on the tile-pipeline groups they read), LocalTimestep<true>,
//     WaveSpeed<true>, ResidualSquares<true> (wait_group on the init group right before their atomic)
//   Init: InitDiagnostics.  Deterministic reductions: DetNorm, DetMin, DetMax.  Error: ErrorStage<true>.
//   The Fused template parameter of a stage switches its wait_group calls on. The unfused variant (Fused = false) is a
//   different component type: the same logic as an ordinary launch (no wait state there, the data dependencies between the
//   kernels order the launches).
//
// UNFUSED counterparts of the local-memory stages (the tile lives in padded global arrays): PadBoundary<Bc>,
// ToPrimitiveG, GradientsG, ReconstructG<Lim>, ConvFluxG<Dir>, ViscFluxG<Dir>, ViscDivergenceChained.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "physics.hpp"
#include "tile.hpp"

namespace nsr {

namespace cs = naga::cospan;
using cs::DType;

using In3   = cs::Input<DType::f32, 3>;
using Out3  = cs::Output<DType::f32, 3>;
using In2   = cs::Input<DType::f32, 2>;
using Out2  = cs::Output<DType::f32, 2>;
using In1   = cs::Input<DType::f32, 1>;
using Out1  = cs::Output<DType::f32, 1>;
using Acc1  = cs::Output<DType::atomic_f32, 1>;
using GeoIn = cs::Input<DType::i32, 1>;  // (nx, ny, tx, ty)

inline sycl::id<3> id3(int v, int j, int i) {
    return sycl::id<3>{static_cast<std::size_t>(v), static_cast<std::size_t>(j), static_cast<std::size_t>(i)};
}
inline sycl::id<2> id2(int j, int i) { return sycl::id<2>{static_cast<std::size_t>(j), static_cast<std::size_t>(i)}; }
inline sycl::id<1> id1(int i) { return sycl::id<1>{static_cast<std::size_t>(i)}; }

/// The grid read from the "geo" input.
inline Geo geo_of(const GeoIn& g) {
    return Geo{static_cast<int>(g[std::size_t{0}]), static_cast<int>(g[std::size_t{1}]), static_cast<int>(g[std::size_t{2}]),
               static_cast<int>(g[std::size_t{3}])};
}
/// The values of the "geo" input, for the driver's update_input.
inline std::vector<std::int32_t> geo_values(int nx, int ny) {
    const Geo g = make_geo(nx, ny);
    return {g.nx, g.ny, g.tx, g.ty};
}

/// The work-item's place in its tile, from the group item of a 2D stage.
inline Th make_th(const cs::GroupItem<2>& it) {
    return Th{static_cast<int>(it.get_local_id(1)), static_cast<int>(it.get_local_id(0)), static_cast<int>(it.get_local_linear_id()),
              static_cast<int>(it.get_group(1)), static_cast<int>(it.get_group(0)), it.get_group().to_sycl()};
}

/// The segments of the fused group (indices used by wait_group).
constexpr std::size_t kSegInit = 0, kSegTile = 1, kSegDiv = 2, kSegDet = 3, kSegErr = 4;

/// The launch range of a tile stage: a whole number of kTile x kTile work-groups, one per tile.
inline sycl::range<2> tile_range(const Geo& g) {
    return sycl::range<2>{static_cast<std::size_t>(g.ty) * kTile, static_cast<std::size_t>(g.tx) * kTile};
}
inline sycl::range<2> tile_wg() { return sycl::range<2>{kTile, kTile}; }

/// The local memory of every tile stage: the same one array of kNLoc fields of kLocCells floats, so the stages of the
/// local_mem::shared tile pipeline see one tile.
using TileMem = cs::LocalMem<cs::LocalArray<DType::f32, 1>>;
inline sycl::range<1> tile_mem_shape() { return sycl::range<1>{static_cast<std::size_t>(kNLoc) * kLocCells}; }

/// The tile kernels in tile.hpp work on a raw pointer to the first field.
inline float* tile_ptr(const cs::LocalView<DType::f32, 1>& v) { return &v[std::size_t{0}]; }

// =============================================================================
// FUSED: the tile pipeline
// =============================================================================

// ---- LoadTile -----------------------------------------------------------------------------------------------------
class LoadTile : public cs::Component<LoadTile> {
public:
    struct Inputs { In3 U; GeoIn geo; };
    struct Outputs {};
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs&, TileMem lm) {
        auto [tile] = lm.get(it);
        float* L    = tile_ptr(tile);
        tile_load(make_th(it), geo_of(in.geo), L, [&](int v, int j, int i) { return in.U[id3(v, j, i)]; });
    }

    Inputs  inputs;
    Outputs outputs;

    LoadTile(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        declare_input(inputs.U, "U", cs::unspecified_shape<3>());
        declare_input(inputs.geo, "geo", cs::unspecified_shape<1>());  // tied
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()), cs::local_desc(tile_mem_shape()));
    }

private:
    Geo geo_;
};

// ---- ApplyBoundary<Bc> --------------------------------------------------------------------------------------------
template <class Bc>
class ApplyBoundary : public cs::Component<ApplyBoundary<Bc>> {
public:
    struct Inputs { In3 U; GeoIn geo; };
    struct Outputs {};
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs&, TileMem lm) {
        auto [tile] = lm.get(it);
        float* L    = tile_ptr(tile);
        tile_boundary<Bc>(make_th(it), geo_of(in.geo), L, [&](int v, int j, int i) { return in.U[id3(v, j, i)]; });
    }

    Inputs  inputs;
    Outputs outputs;

    ApplyBoundary(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.U, "U", cs::unspecified_shape<3>());
        this->declare_input(inputs.geo, "geo", cs::unspecified_shape<1>());  // tied
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()), cs::local_desc(tile_mem_shape()));
    }

private:
    Geo geo_;
};

// ---- ToPrimitive, Gradients (no ports: they work on the local tile) ----------------------------------------------------
class ToPrimitive : public cs::Component<ToPrimitive> {
public:
    struct Inputs {};
    struct Outputs {};
    static void compute_forward(cs::GroupItem<2> it, const Inputs&, const Outputs&, TileMem lm) {
        auto [tile] = lm.get(it);
        float* L    = tile_ptr(tile);
        tile_to_primitive(make_th(it), L);
    }

    Inputs  inputs;
    Outputs outputs;

    ToPrimitive(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {}
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()), cs::local_desc(tile_mem_shape()));
    }

private:
    Geo geo_;
};

class Gradients : public cs::Component<Gradients> {
public:
    struct Inputs {};
    struct Outputs {};
    static void compute_forward(cs::GroupItem<2> it, const Inputs&, const Outputs&, TileMem lm) {
        auto [tile] = lm.get(it);
        float* L    = tile_ptr(tile);
        tile_gradients(make_th(it), L);
    }

    Inputs  inputs;
    Outputs outputs;

    Gradients(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {}
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()), cs::local_desc(tile_mem_shape()));
    }

private:
    Geo geo_;
};

// ---- Reconstruct<Lim> ---------------------------------------------------------------------------------------------
template <class Lim>
class Reconstruct : public cs::Component<Reconstruct<Lim>> {
public:
    struct Inputs {};
    struct Outputs {};
    static void compute_forward(cs::GroupItem<2> it, const Inputs&, const Outputs&, TileMem lm) {
        auto [tile] = lm.get(it);
        float* L    = tile_ptr(tile);
        tile_reconstruct<Lim>(make_th(it), L);
    }

    Inputs  inputs;
    Outputs outputs;

    Reconstruct(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {}
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()), cs::local_desc(tile_mem_shape()));
    }

private:
    Geo geo_;
};

// ---- ConvFlux<Dir> / ViscFlux<Dir>: instanced for x (Dir 0) and y (Dir 1) ---------------------------------------------
// x-faces: (4, ny, nx+1); y-faces: (4, ny+1, nx)
template <int Dir>
class ConvFlux : public cs::Component<ConvFlux<Dir>> {
public:
    struct Inputs { GeoIn geo; };
    struct Outputs { Out3 F; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out, TileMem lm) {
        auto [tile]    = lm.get(it);
        const float* L = tile_ptr(tile);
        tile_conv_flux(make_th(it), geo_of(in.geo), L, Dir, [&](int v, int j, int i, float x) { out.F[id3(v, j, i)] = x; });
    }

    Inputs  inputs;
    Outputs outputs;

    ConvFlux(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.geo, "geo", cs::unspecified_shape<1>());  // tied
        this->declare_output(outputs.F, "F", 4, geo_.ny + (Dir == 1 ? 1 : 0), geo_.nx + (Dir == 0 ? 1 : 0));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()), cs::local_desc(tile_mem_shape()));
    }

private:
    Geo geo_;
};

template <int Dir>
class ViscFlux : public cs::Component<ViscFlux<Dir>> {
public:
    struct Inputs { GeoIn geo; };
    struct Outputs { Out3 F; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out, TileMem lm) {
        auto [tile]    = lm.get(it);
        const float* L = tile_ptr(tile);
        tile_visc_flux(make_th(it), geo_of(in.geo), L, Dir, [&](int v, int j, int i, float x) { out.F[id3(v, j, i)] = x; });
    }

    Inputs  inputs;
    Outputs outputs;

    ViscFlux(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.geo, "geo", cs::unspecified_shape<1>());  // tied
        this->declare_output(outputs.F, "F", 4, geo_.ny + (Dir == 1 ? 1 : 0), geo_.nx + (Dir == 0 ? 1 : 0));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()), cs::local_desc(tile_mem_shape()));
    }

private:
    Geo geo_;
};

// =============================================================================
// Divergence and diagnostics (the Fused parameter turns the waits on)
// =============================================================================

/// wait_group on the tile-pipeline groups whose faces this tile's cells read: its own, the east and the north neighbour.
inline void wait_tile_neighbours(const cs::GroupItem<2>& it, const Th& t, const Geo& g) {
    const std::size_t gy = static_cast<std::size_t>(t.gy), gx = static_cast<std::size_t>(t.gx);
    it.wait_group(kSegTile, sycl::id<2>{gy, gx});
    if (t.gx + 1 < g.tx) it.wait_group(kSegTile, sycl::id<2>{gy, gx + 1});
    if (t.gy + 1 < g.ty) it.wait_group(kSegTile, sycl::id<2>{gy + 1, gx});
}

// ---- ConvDivergence<Fused>: R = -div F_conv (writes R; fused: R is a member of the tie) -------------------------------
// Only the fused variant reads the grid (for its waits).
template <bool Fused>
class ConvDivergence : public cs::Component<ConvDivergence<Fused>> {
public:
    struct InputsPlain { In3 Fx, Fy; };
    struct InputsFused { In3 Fx, Fy; GeoIn geo; };
    using Inputs = std::conditional_t<Fused, InputsFused, InputsPlain>;
    struct Outputs { Out3 R; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        const Th t = make_th(it);
        if constexpr (Fused) wait_tile_neighbours(it, t, geo_of(in.geo));
        cell_divergence(
            t, -1.0f, [&](int v, int j, int i) { return in.Fx[id3(v, j, i)]; }, [&](int v, int j, int i) { return in.Fy[id3(v, j, i)]; },
            [&](int v, int j, int i, float x) { out.R[id3(v, j, i)] = x; });
    }

    Inputs  inputs;
    Outputs outputs;

    ConvDivergence(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.Fx, "Fx", 4, geo_.ny, geo_.nx + 1);
        this->declare_input(inputs.Fy, "Fy", 4, geo_.ny + 1, geo_.nx);
        if constexpr (Fused) {
            this->declare_input(inputs.geo, "geo", cs::unspecified_shape<1>());  // tied
            this->declare_output(outputs.R, "R", cs::unspecified_shape<3>());
        } else {
            this->declare_output(outputs.R, "R", 4, geo_.ny, geo_.nx);
        }
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()));
    }

private:
    Geo geo_;
};

// ---- ViscDivergence (fused only): R += div F_visc (R is the tied output the convective part wrote) --------------------
class ViscDivergence : public cs::Component<ViscDivergence> {
public:
    struct Inputs { In3 Fx, Fy; GeoIn geo; };
    struct Outputs { Out3 R; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        const Th t = make_th(it);
        wait_tile_neighbours(it, t, geo_of(in.geo));
        cell_divergence(
            t, 1.0f, [&](int v, int j, int i) { return in.Fx[id3(v, j, i)]; }, [&](int v, int j, int i) { return in.Fy[id3(v, j, i)]; },
            [&](int v, int j, int i, float x) { out.R[id3(v, j, i)] += x; });
    }

    Inputs  inputs;
    Outputs outputs;

    ViscDivergence(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        declare_input(inputs.Fx, "Fx", 4, geo_.ny, geo_.nx + 1);
        declare_input(inputs.Fy, "Fy", 4, geo_.ny + 1, geo_.nx);
        declare_input(inputs.geo, "geo", cs::unspecified_shape<1>());  // tied
        declare_output(outputs.R, "R", cs::unspecified_shape<3>());
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()));
    }

private:
    Geo geo_;
};

// ---- the unfused viscous divergence: R = Rc + div F_visc into its own output (no tie between the two kernels) ----------
class ViscDivergenceChained : public cs::Component<ViscDivergenceChained> {
public:
    struct Inputs { In3 Rc, Fx, Fy; };
    struct Outputs { Out3 R; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        cell_divergence(
            make_th(it), 1.0f, [&](int v, int j, int i) { return in.Fx[id3(v, j, i)]; }, [&](int v, int j, int i) { return in.Fy[id3(v, j, i)]; },
            [&](int v, int j, int i, float x) { out.R[id3(v, j, i)] = in.Rc[id3(v, j, i)] + x; });
    }

    Inputs  inputs;
    Outputs outputs;

    ViscDivergenceChained(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        declare_input(inputs.Rc, "Rc", 4, geo_.ny, geo_.nx);
        declare_input(inputs.Fx, "Fx", 4, geo_.ny, geo_.nx + 1);
        declare_input(inputs.Fy, "Fy", 4, geo_.ny + 1, geo_.nx);
        declare_output(outputs.R, "R", 4, geo_.ny, geo_.nx);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()));
    }

private:
    Geo geo_;
};

// ---- InitDiagnostics: the atomic accumulators (min dt: +inf, max wave speed: -inf, the four sums: 0) -----------------
class InitDiagnostics : public cs::Component<InitDiagnostics> {
public:
    struct Inputs {};
    struct Outputs { Acc1 Dt, Ws, Sq; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs&, const Outputs& out) {
        if (it.get_local_linear_id() == 0) {
            out.Dt[id1(0)].store(std::numeric_limits<float>::infinity());
            out.Ws[id1(0)].store(-std::numeric_limits<float>::infinity());
            for (int v = 0; v < 4; ++v) out.Sq[id1(v)].store(0.0f);
        }
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_output(outputs.Dt, "Dt", cs::unspecified_shape<1>());
        declare_output(outputs.Ws, "Ws", cs::unspecified_shape<1>());
        declare_output(outputs.Sq, "Sq", cs::unspecified_shape<1>());
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(tile_wg()), cs::wg_desc(tile_wg()));
    }
};

// ---- LocalTimestep<Fused> / WaveSpeed<Fused>: a group reduction, then the atomic and the per-tile partial ---------------
// Fused: each waits (collectively, right before its atomic) on the init group.
template <bool Fused>
class LocalTimestep : public cs::Component<LocalTimestep<Fused>> {
public:
    struct Inputs { In3 U; };
    struct Outputs { Acc1 Dt; Out2 Part; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        const Th t = make_th(it);
        float dt, ws;
        cell_dt_ws(t, [&](int v, int j, int i) { return in.U[id3(v, j, i)]; }, dt, ws);
        const float m = tile_min(t, dt);
        if constexpr (Fused) it.wait_group(kSegInit, sycl::id<2>{0, 0});
        if (t.tid == 0) {
            out.Dt[id1(0)].fetch_min(m);
            out.Part[id2(t.gy, t.gx)] = m;
        }
    }

    Inputs  inputs;
    Outputs outputs;

    LocalTimestep(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.U, "U", cs::unspecified_shape<3>());
        this->declare_output(outputs.Dt, "Dt", cs::unspecified_shape<1>());
        this->declare_output(outputs.Part, "Part", geo_.ty, geo_.tx);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()));
    }

private:
    Geo geo_;
};

template <bool Fused>
class WaveSpeed : public cs::Component<WaveSpeed<Fused>> {
public:
    struct Inputs { In3 U; };
    struct Outputs { Acc1 Ws; Out2 Part; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        const Th t = make_th(it);
        float dt, ws;
        cell_dt_ws(t, [&](int v, int j, int i) { return in.U[id3(v, j, i)]; }, dt, ws);
        const float m = tile_max(t, ws);
        if constexpr (Fused) it.wait_group(kSegInit, sycl::id<2>{0, 0});
        if (t.tid == 0) {
            out.Ws[id1(0)].fetch_max(m);
            out.Part[id2(t.gy, t.gx)] = m;
        }
    }

    Inputs  inputs;
    Outputs outputs;

    WaveSpeed(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.U, "U", cs::unspecified_shape<3>());
        this->declare_output(outputs.Ws, "Ws", cs::unspecified_shape<1>());
        this->declare_output(outputs.Part, "Part", geo_.ty, geo_.tx);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()));
    }

private:
    Geo geo_;
};

// ---- ResidualSquares<Fused>: the four sums of R^2 over the tile --------------------------------------------------------
template <bool Fused>
class ResidualSquares : public cs::Component<ResidualSquares<Fused>> {
public:
    struct Inputs { In3 R; };
    struct Outputs { Acc1 Sq; Out3 Part; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        const Th t = make_th(it);
        const int j = t.gy * kTile + t.ly, i = t.gx * kTile + t.lx;
        float s[4];
        for (int v = 0; v < 4; ++v) {
            const float r = in.R[id3(v, j, i)];
            s[v]          = tile_sum(t, r * r);
        }
        if constexpr (Fused) it.wait_group(kSegInit, sycl::id<2>{0, 0});
        if (t.tid == 0) {
            for (int v = 0; v < 4; ++v) {
                out.Sq[id1(v)].fetch_add(s[v]);
                out.Part[id3(v, t.gy, t.gx)] = s[v];
            }
        }
    }

    Inputs  inputs;
    Outputs outputs;

    ResidualSquares(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.R, "R", 4, geo_.ny, geo_.nx);
        this->declare_output(outputs.Sq, "Sq", cs::unspecified_shape<1>());
        this->declare_output(outputs.Part, "Part", 4, geo_.ty, geo_.tx);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_range(geo_)), cs::wg_desc(tile_wg()));
    }

private:
    Geo geo_;
};

// =============================================================================
// The deterministic reductions of the per-tile partials, in fixed order, and the Error stage
// =============================================================================
class DetNorm : public cs::Component<DetNorm> {
public:
    struct Inputs { In3 Part; GeoIn geo; };
    struct Outputs { Out1 Det; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        const Geo   geo = geo_of(in.geo);
        const Th    t   = make_th(it);
        const int   v   = static_cast<int>(it.get_group(0));  // one group per variable
        const int   tx  = geo.tx;
        const float s   = det_sum(t, geo.tx * geo.ty, [&](int k) { return in.Part[id3(v, k / tx, k % tx)]; });
        if (t.tid == 0) out.Det[id1(v)] = s;
    }

    Inputs  inputs;
    Outputs outputs;

    DetNorm(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        declare_input(inputs.Part, "Part", 4, geo_.ty, geo_.tx);
        declare_input(inputs.geo, "geo", cs::unspecified_shape<1>());  // tied
        declare_output(outputs.Det, "Det", 4);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<2>{4 * kTile, kTile}), cs::wg_desc(tile_wg()));
    }

private:
    Geo geo_;
};

/// The min timestep (IsMin) or the max wave speed over the per-tile partials: one group.
template <bool IsMin>
class DetExtreme : public cs::Component<DetExtreme<IsMin>> {
public:
    struct Inputs { In2 Part; GeoIn geo; };
    struct Outputs { Out1 Det; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        const Geo   geo = geo_of(in.geo);
        const Th    t   = make_th(it);
        const int   tx  = geo.tx;
        auto        p   = [&](int k) { return in.Part[id2(k / tx, k % tx)]; };
        const float s   = IsMin ? det_min(t, geo.tx * geo.ty, p) : det_max(t, geo.tx * geo.ty, p);
        if (t.tid == 0) out.Det[id1(0)] = s;
    }

    Inputs  inputs;
    Outputs outputs;

    DetExtreme(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.Part, "Part", geo_.ty, geo_.tx);
        this->declare_input(inputs.geo, "geo", cs::unspecified_shape<1>());  // tied
        this->declare_output(outputs.Det, "Det", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_wg()), cs::wg_desc(tile_wg()));
    }

private:
    Geo geo_;
};
using DetMin = DetExtreme<true>;
using DetMax = DetExtreme<false>;

/// Err = (atomic result) - (deterministic result): [min dt, max wave speed, sum R0^2 .. sum R3^2]. Fused: it waits on the
/// four deterministic-reduction groups whose results it compares (groups 0-3 of that segment).
template <bool Fused>
class ErrorStage : public cs::Component<ErrorStage<Fused>> {
public:
    struct Inputs { In1 DetSq, DetDt, DetWs; };
    struct Outputs { Acc1 Dt, Ws, Sq; Out1 Err; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        if constexpr (Fused)
            for (std::size_t g = 0; g < 4; ++g) it.wait_group(kSegDet, sycl::id<2>{g, 0});
        if (it.get_local_linear_id() == 0) {
            out.Err[id1(0)] = out.Dt[id1(0)].load() - in.DetDt[id1(0)];
            out.Err[id1(1)] = out.Ws[id1(0)].load() - in.DetWs[id1(0)];
            for (int v = 0; v < 4; ++v) out.Err[id1(2 + v)] = out.Sq[id1(v)].load() - in.DetSq[id1(v)];
        }
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        this->declare_input(inputs.DetSq, "DetSq", 4);
        this->declare_input(inputs.DetDt, "DetDt", 1);
        this->declare_input(inputs.DetWs, "DetWs", 1);
        this->declare_output(outputs.Dt, "Dt", cs::unspecified_shape<1>());
        this->declare_output(outputs.Ws, "Ws", cs::unspecified_shape<1>());
        this->declare_output(outputs.Sq, "Sq", cs::unspecified_shape<1>());
        this->declare_output(outputs.Err, "Err", 6);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(tile_wg()), cs::wg_desc(tile_wg()));
    }
};

// =============================================================================
// UNFUSED counterparts of the local-memory stages: the padded tile lives in global memory
// =============================================================================
// Padded arrays have shape (nf, ny+4, nx+4): padded cell (b, a) is the domain cell (b-2, a-2).

/// LoadTile and ApplyBoundary in one kernel: the padded conserved state, interior copied, halo wrapped or mirrored.
template <class Bc>
class PadBoundary : public cs::Component<PadBoundary<Bc>> {
public:
    struct Inputs { In3 U; GeoIn geo; };
    struct Outputs { Out3 P; };
    static void compute_forward(sycl::id<2> idx, const Inputs& in, const Outputs& out) {
        const Geo geo = geo_of(in.geo);
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

    Inputs  inputs;
    Outputs outputs;

    PadBoundary(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.U, "U", cs::unspecified_shape<3>());
        this->declare_input(inputs.geo, "geo", cs::unspecified_shape<1>());  // tied
        this->declare_output(outputs.P, "P", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(
            cs::range_desc(sycl::range<2>{static_cast<std::size_t>(geo_.ny + 2 * kHalo), static_cast<std::size_t>(geo_.nx + 2 * kHalo)}));
    }

private:
    Geo geo_;
};

inline sycl::range<2> padded_range(const Geo& g, int trim) {
    return sycl::range<2>{static_cast<std::size_t>(g.ny + 2 * kHalo - 2 * trim), static_cast<std::size_t>(g.nx + 2 * kHalo - 2 * trim)};
}

class ToPrimitiveG : public cs::Component<ToPrimitiveG> {
public:
    struct Inputs { In3 P; };
    struct Outputs { Out3 W; };
    static void compute_forward(sycl::id<2> idx, const Inputs& in, const Outputs& out) {
        const int b = static_cast<int>(idx[0]), a = static_cast<int>(idx[1]);
        float u[4], w[4];
        for (int v = 0; v < 4; ++v) u[v] = in.P[id3(v, b, a)];
        to_prim<float>(u, w);
        for (int v = 0; v < 4; ++v) out.W[id3(v, b, a)] = w[v];
    }

    Inputs  inputs;
    Outputs outputs;

    ToPrimitiveG(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        declare_input(inputs.P, "P", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        declare_output(outputs.W, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(padded_range(geo_, 0)));
    }

private:
    Geo geo_;
};

class GradientsG : public cs::Component<GradientsG> {
public:
    struct Inputs { In3 W; };
    struct Outputs { Out3 G; };
    static void compute_forward(sycl::id<2> idx, const Inputs& in, const Outputs& out) {
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

    Inputs  inputs;
    Outputs outputs;

    GradientsG(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        declare_input(inputs.W, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        declare_output(outputs.G, "G", 6, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(padded_range(geo_, 1)));
    }

private:
    Geo geo_;
};

template <class Lim>
class ReconstructG : public cs::Component<ReconstructG<Lim>> {
public:
    struct Inputs { In3 W; };
    struct Outputs { Out3 S; };
    static void compute_forward(sycl::id<2> idx, const Inputs& in, const Outputs& out) {
        const int b = 1 + static_cast<int>(idx[0]), a = 1 + static_cast<int>(idx[1]);
        for (int v = 0; v < 4; ++v) {
            const float w0 = in.W[id3(v, b, a)];
            out.S[id3(v, b, a)]     = Lim::template slope<float>(w0 - in.W[id3(v, b, a - 1)], in.W[id3(v, b, a + 1)] - w0);
            out.S[id3(4 + v, b, a)] = Lim::template slope<float>(w0 - in.W[id3(v, b - 1, a)], in.W[id3(v, b + 1, a)] - w0);
        }
    }

    Inputs  inputs;
    Outputs outputs;

    ReconstructG(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.W, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        this->declare_output(outputs.S, "S", 8, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(padded_range(geo_, 1)));
    }

private:
    Geo geo_;
};

/// The face fluxes from the padded arrays; one index-range launch over the faces. Dir 0: x-faces (ny, nx+1), 1: y-faces (ny+1, nx).
inline sycl::range<2> face_range(const Geo& g, int dir) {
    return sycl::range<2>{static_cast<std::size_t>(g.ny + (dir == 1 ? 1 : 0)), static_cast<std::size_t>(g.nx + (dir == 0 ? 1 : 0))};
}

template <int Dir>
class ConvFluxG : public cs::Component<ConvFluxG<Dir>> {
public:
    struct Inputs { In3 W, S; };
    struct Outputs { Out3 F; };
    static void compute_forward(sycl::id<2> idx, const Inputs& in, const Outputs& out) {
        const int j = static_cast<int>(idx[0]), i = static_cast<int>(idx[1]);
        const int b = j + kHalo, a = i + kHalo;
        const int bl = Dir == 0 ? b : b - 1, al = Dir == 0 ? a - 1 : a;
        float wl[4], wr[4], F[4];
        for (int v = 0; v < 4; ++v) {
            wl[v] = in.W[id3(v, bl, al)] + 0.5f * in.S[id3(4 * Dir + v, bl, al)];
            wr[v] = in.W[id3(v, b, a)] - 0.5f * in.S[id3(4 * Dir + v, b, a)];
        }
        hllc<float>(wl, wr, Dir, F);
        for (int v = 0; v < 4; ++v) out.F[id3(v, j, i)] = F[v];
    }

    Inputs  inputs;
    Outputs outputs;

    ConvFluxG(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.W, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        this->declare_input(inputs.S, "S", 8, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        this->declare_output(outputs.F, "F", 4, geo_.ny + (Dir == 1 ? 1 : 0), geo_.nx + (Dir == 0 ? 1 : 0));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(face_range(geo_, Dir)));
    }

private:
    Geo geo_;
};

template <int Dir>
class ViscFluxG : public cs::Component<ViscFluxG<Dir>> {
public:
    struct Inputs { In3 W, G; };
    struct Outputs { Out3 F; };
    static void compute_forward(sycl::id<2> idx, const Inputs& in, const Outputs& out) {
        const int j = static_cast<int>(idx[0]), i = static_cast<int>(idx[1]);
        const int b = j + kHalo, a = i + kHalo;
        const int bl = Dir == 0 ? b : b - 1, al = Dir == 0 ? a - 1 : a;
        float wl[4], wr[4], gl[6], gr[6], F[4];
        for (int v = 0; v < 4; ++v) {
            wl[v] = in.W[id3(v, bl, al)];
            wr[v] = in.W[id3(v, b, a)];
        }
        for (int q = 0; q < 6; ++q) {
            gl[q] = in.G[id3(q, bl, al)];
            gr[q] = in.G[id3(q, b, a)];
        }
        visc_flux<float>(wl, wr, gl, gr, Dir, F);
        for (int v = 0; v < 4; ++v) out.F[id3(v, j, i)] = F[v];
    }

    Inputs  inputs;
    Outputs outputs;

    ViscFluxG(int nx, int ny) : geo_{make_geo(nx, ny)} {}
    void declare() override {
        this->declare_input(inputs.W, "W", 4, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        this->declare_input(inputs.G, "G", 6, geo_.ny + 2 * kHalo, geo_.nx + 2 * kHalo);
        this->declare_output(outputs.F, "F", 4, geo_.ny + (Dir == 1 ? 1 : 0), geo_.nx + (Dir == 0 ? 1 : 0));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(face_range(geo_, Dir)));
    }

private:
    Geo geo_;
};

}  // namespace nsr
