#pragma once

// =============================================================================
// examples/ns-residual/reference.hpp — the host double-precision reference
// =============================================================================
//
// WRITTEN FIRST: it fixes the conventions every variant matches (physics.hpp lists them). It knows nothing of tiles: it
// pads the whole domain by 2 cells (periodic wrap or wall mirror), then evaluates face fluxes row by row with the same
// face-ownership convention (west / south faces, plus the east / north domain-boundary faces), in double precision.
// Memory: the padded primitives (4 doubles per cell) and R (4 doubles per cell); the fluxes live in row buffers.
//
// The initial field is taken as an input (the float values the device generated, read back or recomputed with the same
// formula), so the reference sees exactly the state the variants start from.

#include <cmath>
#include <cstddef>
#include <vector>

#include "physics.hpp"

namespace nsr {

struct RefResult {
    std::vector<double> R;  // (4, ny, nx)
    double              dt_min = 0.0, ws_max = 0.0;
    double              sq[4]  = {0, 0, 0, 0};
};

/// U: (4, ny, nx) float, row-major. Lim: MinMod or VanLeer; walls: no-slip adiabatic walls in y (else periodic in y).
template <class Lim>
RefResult reference_residual(int nx, int ny, bool walls, const std::vector<float>& U) {
    const std::size_t NX = static_cast<std::size_t>(nx), NY = static_cast<std::size_t>(ny);
    const std::size_t PX = NX + 4, PY = NY + 4;

    // ---- the padded primitive state (4, ny+4, nx+4) ----
    std::vector<double> Wp(4 * PX * PY);
    const auto wp = [&](int c, long b, long a) -> double& { return Wp[(static_cast<std::size_t>(c) * PY + static_cast<std::size_t>(b)) * PX + static_cast<std::size_t>(a)]; };
    const auto wrap = [](long k, long n) { return ((k % n) + n) % n; };
    for (long b = 0; b < static_cast<long>(PY); ++b) {
        for (long a = 0; a < static_cast<long>(PX); ++a) {
            const long j = b - 2, i = a - 2;
            const long is = wrap(i, nx);
            long       js = j;
            double     sgn = 1.0;
            if (j < 0 || j >= ny) {
                if (walls) {
                    js  = j < 0 ? -1 - j : 2L * ny - 1 - j;  // mirror about the wall
                    sgn = -1.0;                              // no-slip: both momenta flip
                } else {
                    js = wrap(j, ny);
                }
            }
            double c[4];
            for (int v = 0; v < 4; ++v) c[v] = static_cast<double>(U[(static_cast<std::size_t>(v) * NY + static_cast<std::size_t>(js)) * NX + static_cast<std::size_t>(is)]);
            c[1] *= sgn;
            c[2] *= sgn;
            double w[4];
            to_prim<double>(c, w);
            for (int v = 0; v < 4; ++v) wp(v, b, a) = w[v];
        }
    }

    // ---- building blocks, all in padded coordinates (b = j+2, a = i+2) ----
    const auto prim = [&](long b, long a, double* w) { for (int v = 0; v < 4; ++v) w[v] = wp(v, b, a); };
    const auto slopes = [&](long b, long a, int dir, double* s) {
        for (int v = 0; v < 4; ++v) {
            const double wm = dir == 0 ? wp(v, b, a - 1) : wp(v, b - 1, a);
            const double w0 = wp(v, b, a);
            const double wpp = dir == 0 ? wp(v, b, a + 1) : wp(v, b + 1, a);
            s[v] = Lim::template slope<double>(w0 - wm, wpp - w0);
        }
    };
    const auto grad = [&](long b, long a, double* g) {
        double xm[4], xp[4], ym[4], yp[4];
        prim(b, a - 1, xm);
        prim(b, a + 1, xp);
        prim(b - 1, a, ym);
        prim(b + 1, a, yp);
        grad_cell<double>(xm, xp, ym, yp, g);
    };
    // The faces between the cell (bl, al) and the cell (br, ar), normal direction dir.
    const auto face = [&](long bl, long al, long br, long ar, int dir, double* Fc, double* Fv) {
        double wl[4], wr[4], sl[4], sr[4], gl[6], gr[6], pl[4], pr[4];
        prim(bl, al, pl);
        prim(br, ar, pr);
        slopes(bl, al, dir, sl);
        slopes(br, ar, dir, sr);
        for (int v = 0; v < 4; ++v) {
            wl[v] = pl[v] + 0.5 * sl[v];
            wr[v] = pr[v] - 0.5 * sr[v];
        }
        hllc<double>(wl, wr, dir, Fc);
        grad(bl, al, gl);
        grad(br, ar, gr);
        visc_flux<double>(pl, pr, gl, gr, dir, Fv);
    };

    RefResult res;
    res.R.assign(4 * NY * NX, 0.0);
    res.dt_min = 1e300;
    res.ws_max = -1e300;

    // row buffers: x-faces of the row (nx+1 faces), y-faces of the lower and upper row of the cells (nx faces)
    std::vector<double> fcx(4 * (NX + 1)), fvx(4 * (NX + 1)), fcy0(4 * NX), fvy0(4 * NX), fcy1(4 * NX), fvy1(4 * NX);
    for (int i = 0; i < nx; ++i) face(2 - 1, i + 2, 2, i + 2, 1, &fcy1[4 * static_cast<std::size_t>(i)], &fvy1[4 * static_cast<std::size_t>(i)]);

    for (int j = 0; j < ny; ++j) {
        std::swap(fcy0, fcy1);
        std::swap(fvy0, fvy1);
        for (int i = 0; i < nx; ++i) face(j + 3 - 1, i + 2, j + 3, i + 2, 1, &fcy1[4 * static_cast<std::size_t>(i)], &fvy1[4 * static_cast<std::size_t>(i)]);
        for (int i = 0; i <= nx; ++i) face(j + 2, i + 2 - 1, j + 2, i + 2, 0, &fcx[4 * static_cast<std::size_t>(i)], &fvx[4 * static_cast<std::size_t>(i)]);
        for (int i = 0; i < nx; ++i) {
            double w[4];
            prim(j + 2, i + 2, w);
            double dt, ws;
            cell_diag<double>(w, dt, ws);
            res.dt_min = tmin(res.dt_min, dt);
            res.ws_max = tmax(res.ws_max, ws);
            for (int v = 0; v < 4; ++v) {
                const std::size_t x0 = 4 * static_cast<std::size_t>(i), x1 = 4 * static_cast<std::size_t>(i + 1);
                const double divc = (fcx[x1 + static_cast<std::size_t>(v)] - fcx[x0 + static_cast<std::size_t>(v)]) + (fcy1[x0 + static_cast<std::size_t>(v)] - fcy0[x0 + static_cast<std::size_t>(v)]);
                const double divv = (fvx[x1 + static_cast<std::size_t>(v)] - fvx[x0 + static_cast<std::size_t>(v)]) + (fvy1[x0 + static_cast<std::size_t>(v)] - fvy0[x0 + static_cast<std::size_t>(v)]);
                const double r    = -divc + divv;
                res.R[(static_cast<std::size_t>(v) * NY + static_cast<std::size_t>(j)) * NX + static_cast<std::size_t>(i)] = r;
                res.sq[v] += r * r;
            }
        }
    }
    return res;
}

}  // namespace nsr
