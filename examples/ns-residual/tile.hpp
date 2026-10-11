#pragma once

// =============================================================================
// examples/ns-residual/tile.hpp — the per-tile device stages as plain functions
// =============================================================================
//
// One work-group (16 x 16 work-items) works on one tile of 16 x 16 cells plus a 2-cell halo, held in a local-memory
// block of kNLoc fields of 20 x 20 floats (physics.hpp lists the fields). Each function here is ONE stage of the
// algorithm and ends with a group barrier when the next stage reads what other work-items wrote.
//
// The CoSpan components (components.hpp) are thin wrappers around these functions, and the hand-written kernel
// (handwritten.hpp) calls the same functions from its own monolithic kernel, so the three variants run one arithmetic.
// The data is reached through small callables (`U(v,j,i)`, `set(v,j,i,x)`), so each caller supplies its own memory.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "physics.hpp"

namespace nsr {

/// The grid in tiles: nx, ny cells (multiples of 16); tx, ty tiles.
struct Geo {
    int nx, ny, tx, ty;
};
inline Geo make_geo(int nx, int ny) { return Geo{nx, ny, nx / kTile, ny / kTile}; }

/// The work-item's place in its tile (group (gy, gx) of the tile grid, local (ly, lx)) and its physical group.
struct Th {
    int             lx, ly, tid, gx, gy;
    sycl::group<3>  grp;  // the physical work-group: shared with the raw SYCL kernel
};

inline int wrapi(int k, int n) { return ((k % n) + n) % n; }

// ---- LoadTile ---------------------------------------------------------------------------------------------------
/// The tile and as much of its halo as lies inside the domain, into local memory (conserved variables).
template <class UGet>
inline void tile_load(const Th& t, const Geo& g, float* L, UGet U) {
    for (int idx = t.tid; idx < kLocCells; idx += kWg) {
        const int b = idx / kLoc, a = idx % kLoc;
        const int j = t.gy * kTile - kHalo + b, i = t.gx * kTile - kHalo + a;
        if (i >= 0 && i < g.nx && j >= 0 && j < g.ny)
            for (int v = 0; v < 4; ++v) L[v * kLocCells + idx] = U(v, j, i);
    }
    sycl::group_barrier(t.grp);
}

// ---- ApplyBoundary ------------------------------------------------------------------------------------------------
/// Fills the halo cells outside the domain. Interior tiles (no edge of the domain) do nothing. Bc is BcPeriodic (wrap in
/// x and y) or BcWalls (wrap in x, no-slip adiabatic mirror in y: rho and E mirrored, both momenta negated).
template <class Bc, class UGet>
inline void tile_boundary(const Th& t, const Geo& g, float* L, UGet U) {
    if (t.gx > 0 && t.gx < g.tx - 1 && t.gy > 0 && t.gy < g.ty - 1) return;  // group-uniform: all or none
    if constexpr (!Bc::walls) {
        for (int idx = t.tid; idx < kLocCells; idx += kWg) {
            const int b = idx / kLoc, a = idx % kLoc;
            const int j = t.gy * kTile - kHalo + b, i = t.gx * kTile - kHalo + a;
            if (i < 0 || i >= g.nx || j < 0 || j >= g.ny)
                for (int v = 0; v < 4; ++v) L[v * kLocCells + idx] = U(v, wrapi(j, g.ny), wrapi(i, g.nx));
        }
        sycl::group_barrier(t.grp);
    } else {
        // 1. the x halo of the rows inside the domain: wrap
        for (int idx = t.tid; idx < kLocCells; idx += kWg) {
            const int b = idx / kLoc, a = idx % kLoc;
            const int j = t.gy * kTile - kHalo + b, i = t.gx * kTile - kHalo + a;
            if ((i < 0 || i >= g.nx) && j >= 0 && j < g.ny)
                for (int v = 0; v < 4; ++v) L[v * kLocCells + idx] = U(v, j, wrapi(i, g.nx));
        }
        sycl::group_barrier(t.grp);
        // 2. the rows outside the domain: the mirror row of the tile (an interior row, complete after step 1)
        for (int idx = t.tid; idx < kLocCells; idx += kWg) {
            const int b = idx / kLoc, a = idx % kLoc;
            const int j = t.gy * kTile - kHalo + b;
            if (j < 0 || j >= g.ny) {
                const int jm = j < 0 ? -1 - j : 2 * g.ny - 1 - j;
                const int bm = jm - (t.gy * kTile - kHalo);
                const int src = bm * kLoc + a;
                L[0 * kLocCells + idx] = L[0 * kLocCells + src];
                L[1 * kLocCells + idx] = -L[1 * kLocCells + src];
                L[2 * kLocCells + idx] = -L[2 * kLocCells + src];
                L[3 * kLocCells + idx] = L[3 * kLocCells + src];
            }
        }
        sycl::group_barrier(t.grp);
    }
}

// ---- ToPrimitive (in place) -------------------------------------------------------------------------------------
inline void tile_to_primitive(const Th& t, float* L) {
    for (int idx = t.tid; idx < kLocCells; idx += kWg) {
        float u[4], w[4];
        for (int v = 0; v < 4; ++v) u[v] = L[v * kLocCells + idx];
        to_prim<float>(u, w);
        for (int v = 0; v < 4; ++v) L[(fW + v) * kLocCells + idx] = w[v];
    }
    sycl::group_barrier(t.grp);
}

// ---- Gradients --------------------------------------------------------------------------------------------------
/// Cell-centred gradients of u, v and T for the cells 1 .. 18 of the local tile.
inline void tile_gradients(const Th& t, float* L) {
    for (int k = t.tid; k < (kLoc - 2) * (kLoc - 2); k += kWg) {
        const int b = 1 + k / (kLoc - 2), a = 1 + k % (kLoc - 2);
        const int c = b * kLoc + a;
        float xm[4], xp[4], ym[4], yp[4], g[6];
        for (int v = 0; v < 4; ++v) {
            xm[v] = L[(fW + v) * kLocCells + c - 1];
            xp[v] = L[(fW + v) * kLocCells + c + 1];
            ym[v] = L[(fW + v) * kLocCells + c - kLoc];
            yp[v] = L[(fW + v) * kLocCells + c + kLoc];
        }
        grad_cell<float>(xm, xp, ym, yp, g);
        for (int q = 0; q < 6; ++q) L[(fG + q) * kLocCells + c] = g[q];
    }
    sycl::group_barrier(t.grp);
}

// ---- Reconstruct (MUSCL slopes) -----------------------------------------------------------------------------------
/// Limited slopes of rho, u, v, p in x and in y for the cells 1 .. 18. Lim is MinMod or VanLeer.
template <class Lim>
inline void tile_reconstruct(const Th& t, float* L) {
    for (int k = t.tid; k < (kLoc - 2) * (kLoc - 2); k += kWg) {
        const int b = 1 + k / (kLoc - 2), a = 1 + k % (kLoc - 2);
        const int c = b * kLoc + a;
        for (int v = 0; v < 4; ++v) {
            const float* w = L + (fW + v) * kLocCells;
            L[(fS + v) * kLocCells + c]     = Lim::template slope<float>(w[c] - w[c - 1], w[c + 1] - w[c]);
            L[(fS + 4 + v) * kLocCells + c] = Lim::template slope<float>(w[c] - w[c - kLoc], w[c + kLoc] - w[c]);
        }
    }
    sycl::group_barrier(t.grp);
}

// ---- the owned faces ------------------------------------------------------------------------------------------------
/// The faces the work-item owns in direction dir (0: x, west faces; 1: y, south faces): the face of its cell, and, for the
/// last work-item column / row of an east / north edge tile, the domain-boundary face. Calls f(fa, fb, fj, fi): the
/// local coordinates (a, b) of the face's right / upper cell and the face's global (j, i).
template <class F>
inline void for_owned_faces(const Th& t, const Geo& g, int dir, F f) {
    const int x0 = t.gx * kTile, y0 = t.gy * kTile;
    f(t.lx + kHalo, t.ly + kHalo, y0 + t.ly, x0 + t.lx);
    if (dir == 0 && t.gx == g.tx - 1 && t.lx == kTile - 1) f(kLoc - kHalo, t.ly + kHalo, y0 + t.ly, x0 + kTile);
    if (dir == 1 && t.gy == g.ty - 1 && t.ly == kTile - 1) f(t.lx + kHalo, kLoc - kHalo, y0 + kTile, x0 + t.lx);
}

// ---- ConvectiveFlux (HLLC) ----------------------------------------------------------------------------------------
/// One component, two instances: dir 0 writes the x-faces, dir 1 the y-faces. set(v, j, i, value).
template <class FSet>
inline void tile_conv_flux(const Th& t, const Geo& g, const float* L, int dir, FSet set) {
    for_owned_faces(t, g, dir, [&](int fa, int fb, int fj, int fi) {
        const int rc = fb * kLoc + fa;
        const int lc = dir == 0 ? rc - 1 : rc - kLoc;
        float wl[4], wr[4], F[4];
        for (int v = 0; v < 4; ++v) {
            wl[v] = L[(fW + v) * kLocCells + lc] + 0.5f * L[(fS + 4 * dir + v) * kLocCells + lc];
            wr[v] = L[(fW + v) * kLocCells + rc] - 0.5f * L[(fS + 4 * dir + v) * kLocCells + rc];
        }
        hllc<float>(wl, wr, dir, F);
        for (int v = 0; v < 4; ++v) set(v, fj, fi, F[v]);
    });
}

// ---- ViscousFlux --------------------------------------------------------------------------------------------------
template <class FSet>
inline void tile_visc_flux(const Th& t, const Geo& g, const float* L, int dir, FSet set) {
    for_owned_faces(t, g, dir, [&](int fa, int fb, int fj, int fi) {
        const int rc = fb * kLoc + fa;
        const int lc = dir == 0 ? rc - 1 : rc - kLoc;
        float wl[4], wr[4], gl[6], gr[6], F[4];
        for (int v = 0; v < 4; ++v) {
            wl[v] = L[(fW + v) * kLocCells + lc];
            wr[v] = L[(fW + v) * kLocCells + rc];
        }
        for (int q = 0; q < 6; ++q) {
            gl[q] = L[(fG + q) * kLocCells + lc];
            gr[q] = L[(fG + q) * kLocCells + rc];
        }
        visc_flux<float>(wl, wr, gl, gr, dir, F);
        for (int v = 0; v < 4; ++v) set(v, fj, fi, F[v]);
    });
}

// ---- divergences (global memory; one work-item per cell) ----------------------------------------------------------
/// op(v, j, i, sign * ((Fx(i+1) - Fx(i)) + (Fy(j+1) - Fy(j)))) for the work-item's cell. Fx(v,j,i), Fy(v,j,i): the face fluxes.
template <class FxGet, class FyGet, class Op>
inline void cell_divergence(const Th& t, float sign, FxGet Fx, FyGet Fy, Op op) {
    const int j = t.gy * kTile + t.ly, i = t.gx * kTile + t.lx;
    for (int v = 0; v < 4; ++v) op(v, j, i, sign * ((Fx(v, j, i + 1) - Fx(v, j, i)) + (Fy(v, j + 1, i) - Fy(v, j, i))));
}

// ---- the per-cell diagnostics, reduced over the tile ----------------------------------------------------------------
/// The conserved state of the work-item's cell -> its dt and wave speed.
template <class UGet>
inline void cell_dt_ws(const Th& t, UGet U, float& dt, float& ws) {
    const int j = t.gy * kTile + t.ly, i = t.gx * kTile + t.lx;
    float u[4], w[4];
    for (int v = 0; v < 4; ++v) u[v] = U(v, j, i);
    to_prim<float>(u, w);
    cell_diag<float>(w, dt, ws);
}

/// A tile's contribution (a group reduction) of the min / max / sum over its cells' values.
inline float tile_min(const Th& t, float x) { return sycl::reduce_over_group(t.grp, x, sycl::minimum<float>()); }
inline float tile_max(const Th& t, float x) { return sycl::reduce_over_group(t.grp, x, sycl::maximum<float>()); }
inline float tile_sum(const Th& t, float x) { return sycl::reduce_over_group(t.grp, x, sycl::plus<float>()); }

// ---- the deterministic reductions of the per-tile partials (fixed order) --------------------------------------------
/// work-item tid sums partials tid, tid + 256, ... in order, then the group tree: the same order on every run.
template <class PGet>
inline float det_sum(const Th& t, int n, PGet p) {
    float acc = 0.0f;
    for (int k = t.tid; k < n; k += kWg) acc += p(k);
    return sycl::reduce_over_group(t.grp, acc, sycl::plus<float>());
}
template <class PGet>
inline float det_min(const Th& t, int n, PGet p) {
    float acc = 3.0e38f;
    for (int k = t.tid; k < n; k += kWg) acc = tmin(acc, p(k));
    return sycl::reduce_over_group(t.grp, acc, sycl::minimum<float>());
}
template <class PGet>
inline float det_max(const Th& t, int n, PGet p) {
    float acc = -3.0e38f;
    for (int k = t.tid; k < n; k += kWg) acc = tmax(acc, p(k));
    return sycl::reduce_over_group(t.grp, acc, sycl::maximum<float>());
}

}  // namespace nsr
