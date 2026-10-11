#pragma once

// =============================================================================
// examples/ns-residual/physics.hpp — the numerics, shared by the host reference and every device variant
// =============================================================================
//
// 2D compressible Navier-Stokes, conserved variables U = (rho, rho u, rho v, E), ideal gas, constant viscosity.
// One evaluation of R = -div F_conv + div F_visc per cell on a uniform grid with dx = dy = 1.
//
// Every function is a template over the scalar type T: the host reference instantiates them with double, the device code
// with float. They take small arrays (4 or 6 values) so that every variant can gather its inputs from its own memory
// (local memory tile, padded global arrays, ...) and still run the same arithmetic.
//
// CONVENTIONS (fixed by reference.hpp; all variants match them):
//   arrays        row-major (y, x): index [j][i], i along x (contiguous), j along y; a field is (ny, nx)
//   primitives    W = (rho, u, v, p); T = p / rho (gas constant 1)
//   faces         the x-face i of row j is the WEST face of cell (j, i); i = 0 .. nx (face nx is the east domain boundary);
//                 the y-face j of column i is the SOUTH face of cell (j, i); j = 0 .. ny.
//                 Fx has shape (4, ny, nx+1), Fy has shape (4, ny+1, nx).  R(j,i) = (Fx(i)-Fx(i+1)) + (Fy(j)-Fy(j+1)) for the
//                 convective part (with the sign flipped for the viscous part: it is ADDED as the divergence).
//   ownership     a tile (16 x 16 cells) owns the west and south faces of its cells; the tiles on the east and north domain
//                 edges also own the domain-boundary faces nx / ny.
//   halo          2 cells: a face needs its two neighbouring cells and their MUSCL slopes (cells a-2 .. a+1).
//   boundaries    periodic: wrap.  walls (y only): ghost cells mirror the interior (rho, E same; both momenta negated):
//                 no-slip, adiabatic.  In x the domain is always periodic.
//   reconstruction MUSCL on primitives, slope = limiter(w[c]-w[c-1], w[c+1]-w[c]) per cell, face states w[c] +- slope/2
//   convective    HLLC with Davis wave speed estimates
//   viscous       normal derivative from the two cells of the face, tangential derivative from the mean of the two
//                 cell-centred (central-difference) gradients; face velocity = mean of the two cells;
//                 heat flux k grad T, k = mu gamma / ((gamma-1) Pr)

#include <cstddef>

#include <sycl/sycl.hpp>

namespace nsr {

constexpr int kTile = 16;                 // the work-group is kTile x kTile cells, one group per tile
constexpr int kHalo = 2;
constexpr int kLoc  = kTile + 2 * kHalo;  // 20: the local tile edge
constexpr int kLocCells = kLoc * kLoc;    // 400
constexpr int kWg   = kTile * kTile;      // 256 work-items

constexpr double kGamma = 1.4;
constexpr double kMu    = 0.01;
constexpr double kPr    = 0.72;
constexpr double kCfl   = 0.5;
constexpr double kWavelength = 32.0;  // cells: fixed, not scaled with the grid

// local-memory field indices (each field is kLocCells floats)
constexpr int fW = 0;   // primitives rho u v p            (converted in place from the conserved tile)
constexpr int fG = 4;   // gradients ux uy vx vy Tx Ty     (cell-centred)
constexpr int fS = 10;  // slopes: x-slopes of rho u v p, then y-slopes
constexpr int kNLoc = 18;
constexpr std::size_t kLocalBytes = static_cast<std::size_t>(kNLoc) * kLocCells * sizeof(float);

template <class T> inline T tmin(T a, T b) { return a < b ? a : b; }
template <class T> inline T tmax(T a, T b) { return a > b ? a : b; }
template <class T> inline T tabs(T a) { return a < T(0) ? -a : a; }

// ---- limiters ---------------------------------------------------------------------------------------------------
// slope(a, b): the cell slope from the backward difference a and the forward difference b. Zero at an extremum, so a
// uniform field has exactly zero slopes.
struct MinMod {
    static constexpr const char* name = "minmod";
    template <class T> static T slope(T a, T b) {
        if (a * b <= T(0)) return T(0);
        return a > T(0) ? tmin(a, b) : tmax(a, b);
    }
};
struct VanLeer {
    static constexpr const char* name = "vanleer";
    template <class T> static T slope(T a, T b) {
        const T ab = a * b;
        if (ab <= T(0)) return T(0);
        return T(2) * ab / (a + b);
    }
};

// ---- boundary conditions (compile-time tags of the device components) ----------------------------------------------
struct BcPeriodic { static constexpr bool walls = false; static constexpr const char* name = "periodic"; };
struct BcWalls    { static constexpr bool walls = true;  static constexpr const char* name = "walls"; };

// ---- state conversions --------------------------------------------------------------------------------------------
template <class T> inline void to_prim(const T* U, T* W) {
    const T r = U[0];
    const T u = U[1] / r;
    const T v = U[2] / r;
    W[0] = r;
    W[1] = u;
    W[2] = v;
    W[3] = (T(kGamma) - T(1)) * (U[3] - T(0.5) * r * (u * u + v * v));
}

/// The cell-centred gradients (ux, uy, vx, vy, Tx, Ty) by central differences from the four neighbours' primitives.
template <class T> inline void grad_cell(const T* wxm, const T* wxp, const T* wym, const T* wyp, T* g) {
    g[0] = T(0.5) * (wxp[1] - wxm[1]);
    g[1] = T(0.5) * (wyp[1] - wym[1]);
    g[2] = T(0.5) * (wxp[2] - wxm[2]);
    g[3] = T(0.5) * (wyp[2] - wym[2]);
    g[4] = T(0.5) * (wxp[3] / wxp[0] - wxm[3] / wxm[0]);
    g[5] = T(0.5) * (wyp[3] / wyp[0] - wym[3] / wym[0]);
}

// ---- HLLC ---------------------------------------------------------------------------------------------------------
/// Convective flux through a face with the normal along x (dir 0) or y (dir 1), from the reconstructed primitives of the
/// two sides (left/below = wl, right/above = wr). F = (mass, x-momentum, y-momentum, energy).
template <class T> inline void hllc(const T* wl, const T* wr, int dir, T* F) {
    const T g  = T(kGamma);
    const T rl = wl[0], rr = wr[0], pl = wl[3], pr = wr[3];
    const T unl = dir == 0 ? wl[1] : wl[2], utl = dir == 0 ? wl[2] : wl[1];
    const T unr = dir == 0 ? wr[1] : wr[2], utr = dir == 0 ? wr[2] : wr[1];
    const T cl = sycl::sqrt(g * pl / rl), cr = sycl::sqrt(g * pr / rr);
    const T El = pl / (g - T(1)) + T(0.5) * rl * (unl * unl + utl * utl);
    const T Er = pr / (g - T(1)) + T(0.5) * rr * (unr * unr + utr * utr);
    const T sl = tmin(unl - cl, unr - cr);
    const T sr = tmax(unl + cl, unr + cr);
    const T sm = (pr - pl + rl * unl * (sl - unl) - rr * unr * (sr - unr)) / (rl * (sl - unl) - rr * (sr - unr));

    T f[4];  // (mass, normal momentum, tangential momentum, energy) in the face frame
    if (sl >= T(0)) {
        f[0] = rl * unl;
        f[1] = rl * unl * unl + pl;
        f[2] = rl * unl * utl;
        f[3] = (El + pl) * unl;
    } else if (sr <= T(0)) {
        f[0] = rr * unr;
        f[1] = rr * unr * unr + pr;
        f[2] = rr * unr * utr;
        f[3] = (Er + pr) * unr;
    } else if (sm >= T(0)) {
        const T k  = rl * (sl - unl) / (sl - sm);
        const T Es = k * (El / rl + (sm - unl) * (sm + pl / (rl * (sl - unl))));
        f[0] = rl * unl + sl * (k - rl);
        f[1] = rl * unl * unl + pl + sl * (k * sm - rl * unl);
        f[2] = rl * unl * utl + sl * (k * utl - rl * utl);
        f[3] = (El + pl) * unl + sl * (Es - El);
    } else {
        const T k  = rr * (sr - unr) / (sr - sm);
        const T Es = k * (Er / rr + (sm - unr) * (sm + pr / (rr * (sr - unr))));
        f[0] = rr * unr + sr * (k - rr);
        f[1] = rr * unr * unr + pr + sr * (k * sm - rr * unr);
        f[2] = rr * unr * utr + sr * (k * utr - rr * utr);
        f[3] = (Er + pr) * unr + sr * (Es - Er);
    }
    F[0] = f[0];
    F[1] = dir == 0 ? f[1] : f[2];
    F[2] = dir == 0 ? f[2] : f[1];
    F[3] = f[3];
}

// ---- viscous flux -------------------------------------------------------------------------------------------------
/// wl/wr: primitives of the two cells of the face (left/below, right/above); gl/gr: their cell-centred gradients.
template <class T> inline void visc_flux(const T* wl, const T* wr, const T* gl, const T* gr, int dir, T* F) {
    const T uf = T(0.5) * (wl[1] + wr[1]);
    const T vf = T(0.5) * (wl[2] + wr[2]);
    const T Tl = wl[3] / wl[0], Tr = wr[3] / wr[0];
    T ux, uy, vx, vy, Tx, Ty;
    if (dir == 0) {
        ux = wr[1] - wl[1];
        vx = wr[2] - wl[2];
        Tx = Tr - Tl;
        uy = T(0.5) * (gl[1] + gr[1]);
        vy = T(0.5) * (gl[3] + gr[3]);
        Ty = T(0.5) * (gl[5] + gr[5]);
    } else {
        uy = wr[1] - wl[1];
        vy = wr[2] - wl[2];
        Ty = Tr - Tl;
        ux = T(0.5) * (gl[0] + gr[0]);
        vx = T(0.5) * (gl[2] + gr[2]);
        Tx = T(0.5) * (gl[4] + gr[4]);
    }
    const T mu  = T(kMu);
    const T k   = mu * T(kGamma) / ((T(kGamma) - T(1)) * T(kPr));
    const T div = ux + vy;
    const T txx = mu * (T(2) * ux - T(2) / T(3) * div);
    const T tyy = mu * (T(2) * vy - T(2) / T(3) * div);
    const T txy = mu * (uy + vx);
    F[0] = T(0);
    if (dir == 0) {
        F[1] = txx;
        F[2] = txy;
        F[3] = uf * txx + vf * txy + k * Tx;
    } else {
        F[1] = txy;
        F[2] = tyy;
        F[3] = uf * txy + vf * tyy + k * Ty;
    }
}

// ---- per-cell diagnostics -----------------------------------------------------------------------------------------
/// The local timestep of one cell (CFL form with the viscous term) and its wave speed, from the cell's primitives.
template <class T> inline void cell_diag(const T* W, T& dt, T& ws) {
    const T c  = sycl::sqrt(T(kGamma) * W[3] / W[0]);
    const T ax = tabs(W[1]) + c, ay = tabs(W[2]) + c;
    const T nu = T(kMu) / W[0] * tmax(T(4) / T(3), T(kGamma) / T(kPr));
    dt = T(kCfl) / (ax + ay + T(4) * nu);
    ws = tmax(ax, ay);
}

// ---- the initial field --------------------------------------------------------------------------------------------
/// Uniform flow plus a smooth perturbation of amplitude `amp` (amp = 0: exactly uniform), cell (j, i), conserved variables.
inline void init_cons(int j, int i, float amp, float* U) {
    const float k  = static_cast<float>(2.0 * 3.14159265358979323846 / kWavelength);
    const float x  = static_cast<float>(i) + 0.5f, y = static_cast<float>(j) + 0.5f;
    const float sx = sycl::sin(k * x), cx = sycl::cos(k * x), sy = sycl::sin(k * y), cy = sycl::cos(k * y);
    const float r  = 1.0f + 0.1f * amp * sx * sy;
    const float u  = 0.3f + 0.05f * amp * cx * sy;
    const float v  = 0.2f + 0.05f * amp * sx * cy;
    const float p  = 1.0f + 0.05f * amp * cx * cy;
    U[0] = r;
    U[1] = r * u;
    U[2] = r * v;
    U[3] = p / (static_cast<float>(kGamma) - 1.0f) + 0.5f * r * (u * u + v * v);
}

}  // namespace nsr
