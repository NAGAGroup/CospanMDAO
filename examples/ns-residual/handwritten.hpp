#pragma once

// =============================================================================
// examples/ns-residual/handwritten.hpp — ONE monolithic SYCL kernel, outside CoSpan
// =============================================================================
//
// The same algorithm and the same schedule as the fused CoSpan group, written by hand: its own queue, its own USM
// buffers, one nd_range launch of (1 + 2T + 4 + 1) work-groups of 16 x 16 (T = tiles). The per-tile stages are the
// functions of tile.hpp (the arithmetic is shared with the CoSpan variants; the scheduling is not).
//
// SCHEDULE. A work-group takes a ticket from an atomic counter when it starts. Tickets are handed out in increasing order,
// so ticket order is segment order, and every wait targets a LOWER ticket (a group that has already started), which is why
// no co-residency is assumed:
//   ticket 0                      init: reset the atomic accumulators (+inf, -inf, 0 x 4)
//   tickets 1 .. T                tile pipeline for tile p = t - 1 (load, boundary, prim, grad, recon, the four flux kernels)
//   tickets T+1 .. 2T             tile p = t - T - 1: wait on the flags of tile p, its east and its north neighbour, then the
//                                 two divergences; the three diagnostics each wait on ticket 0's flag right before their atomic
//   tickets 2T+1 .. 2T+4          the deterministic reductions, after its OWN barrier: wait until 2T+1 groups have finished
//                                 (every group of the three earlier segments); group 0 does min dt and max wave speed too
//   ticket 2T+5                   the error: waits on the flags of tickets 2T+1 .. 2T+4, then atomic - deterministic
// A group finishes with a group barrier, a release fence, its flag (per ticket) and a global done counter.
// The ticket state is cleared by a memset before each launch (the one host-side reset).

#include <cstddef>
#include <cstdint>
#include <limits>

#include <sycl/sycl.hpp>

#include "physics.hpp"
#include "tile.hpp"

namespace nsr {

/// All device buffers of the hand-written variant.
struct HwBuffers {
    float *U, *Fcx, *Fcy, *Fvx, *Fvy, *R;
    float *PartDt, *PartWs, *PartSq;  // per tile: (ty, tx), (ty, tx), (4, ty, tx)
    float *Dt, *Ws, *Sq;              // the atomic accumulators: 1, 1, 4 (accessed through atomic_ref)
    float *DetSq, *DetDt, *DetWs;     // the deterministic reductions: 4, 1, 1
    float *Err;                       // 6
    std::uint32_t* ts;                // [0] ticket counter, [1] groups finished, [2 + t] flag of ticket t
};

inline std::size_t hw_groups(const Geo& g) { return 1 + 2 * static_cast<std::size_t>(g.tx) * g.ty + 4 + 1; }

using HwAtomicU = sycl::atomic_ref<std::uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::global_space>;
using HwAtomicF = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;

/// Launch the whole evaluation (async; in-order queue).
template <class Lim, class Bc>
sycl::event hw_forward(sycl::queue& q, const HwBuffers& b, const Geo geo) {
    const std::size_t groups = hw_groups(geo);
    q.memset(b.ts, 0, (2 + groups) * sizeof(std::uint32_t));
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> loc{sycl::range<1>{static_cast<std::size_t>(kNLoc) * kLocCells}, h};
        h.parallel_for(sycl::nd_range<3>{sycl::range<3>{1, kTile, groups * kTile}, sycl::range<3>{1, kTile, kTile}}, [=](sycl::nd_item<3> it) {
            const int tx = geo.tx, ty = geo.ty, T = tx * ty;
            sycl::group<3> grp = it.get_group();
            const int      tid = static_cast<int>(it.get_local_linear_id());

            // ---- the ticket ----
            std::uint32_t t = 0;
            if (tid == 0) {
                HwAtomicU counter{b.ts[0]};
                t = counter.fetch_add(1u, sycl::memory_order::relaxed);
            }
            t = sycl::group_broadcast(grp, t);

            // wait (collective) until the group of ticket k has finished
            const auto wait_ticket = [&](std::uint32_t k) {
                if (tid == 0) {
                    HwAtomicU flag{b.ts[2 + k]};
                    while (flag.load(sycl::memory_order::acquire) == 0) {}
                }
                sycl::group_barrier(grp);
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
            };

            Th th{static_cast<int>(it.get_local_id(2)), static_cast<int>(it.get_local_id(1)), tid, 0, 0, grp};
            const auto at = [&](int index) {  // the tile of index p = gy * tx + gx
                th.gy = index / tx;
                th.gx = index % tx;
            };
            const std::size_t NX = static_cast<std::size_t>(geo.nx), NY = static_cast<std::size_t>(geo.ny);
            const std::size_t NX1 = NX + 1;
            // flat offsets
            const auto cell  = [&](int v, int j, int i) { return (static_cast<std::size_t>(v) * NY + static_cast<std::size_t>(j)) * NX + static_cast<std::size_t>(i); };
            const auto fxoff = [&](int v, int j, int i) { return (static_cast<std::size_t>(v) * NY + static_cast<std::size_t>(j)) * NX1 + static_cast<std::size_t>(i); };
            const auto fyoff = [&](int v, int j, int i) { return (static_cast<std::size_t>(v) * (NY + 1) + static_cast<std::size_t>(j)) * NX + static_cast<std::size_t>(i); };
            const auto Uget  = [&](int v, int j, int i) { return b.U[cell(v, j, i)]; };

            if (t == 0) {
                // ---- init ----
                if (tid == 0) {
                    HwAtomicF{b.Dt[0]}.store(std::numeric_limits<float>::infinity());
                    HwAtomicF{b.Ws[0]}.store(-std::numeric_limits<float>::infinity());
                    for (int v = 0; v < 4; ++v) HwAtomicF{b.Sq[v]}.store(0.0f);
                }
            } else if (t <= static_cast<std::uint32_t>(T)) {
                // ---- tile pipeline ----
                at(static_cast<int>(t) - 1);
                float* L = loc.template get_multi_ptr<sycl::access::decorated::no>().get();
                tile_load(th, geo, L, Uget);
                tile_boundary<Bc>(th, geo, L, Uget);
                tile_to_primitive(th, L);
                tile_gradients(th, L);
                tile_reconstruct<Lim>(th, L);
                tile_conv_flux(th, geo, L, 0, [&](int v, int j, int i, float x) { b.Fcx[fxoff(v, j, i)] = x; });
                tile_conv_flux(th, geo, L, 1, [&](int v, int j, int i, float x) { b.Fcy[fyoff(v, j, i)] = x; });
                tile_visc_flux(th, geo, L, 0, [&](int v, int j, int i, float x) { b.Fvx[fxoff(v, j, i)] = x; });
                tile_visc_flux(th, geo, L, 1, [&](int v, int j, int i, float x) { b.Fvy[fyoff(v, j, i)] = x; });
            } else if (t <= static_cast<std::uint32_t>(2 * T)) {
                // ---- divergences and diagnostics of tile p ----
                const int p = static_cast<int>(t) - T - 1;
                at(p);
                wait_ticket(static_cast<std::uint32_t>(1 + p));
                if (th.gx + 1 < tx) wait_ticket(static_cast<std::uint32_t>(1 + p + 1));
                if (th.gy + 1 < ty) wait_ticket(static_cast<std::uint32_t>(1 + p + tx));
                cell_divergence(th, -1.0f, [&](int v, int j, int i) { return b.Fcx[fxoff(v, j, i)]; }, [&](int v, int j, int i) { return b.Fcy[fyoff(v, j, i)]; },
                                [&](int v, int j, int i, float x) { b.R[cell(v, j, i)] = x; });
                cell_divergence(th, 1.0f, [&](int v, int j, int i) { return b.Fvx[fxoff(v, j, i)]; }, [&](int v, int j, int i) { return b.Fvy[fyoff(v, j, i)]; },
                                [&](int v, int j, int i, float x) { b.R[cell(v, j, i)] += x; });

                float dt, ws;
                cell_dt_ws(th, Uget, dt, ws);
                const float mdt = tile_min(th, dt);
                wait_ticket(0);  // the accumulators are reset
                if (tid == 0) {
                    HwAtomicF{b.Dt[0]}.fetch_min(mdt);
                    b.PartDt[p] = mdt;
                }
                const float mws = tile_max(th, ws);
                wait_ticket(0);
                if (tid == 0) {
                    HwAtomicF{b.Ws[0]}.fetch_max(mws);
                    b.PartWs[p] = mws;
                }
                const int jj = th.gy * kTile + th.ly, ii = th.gx * kTile + th.lx;
                float     s[4];
                for (int v = 0; v < 4; ++v) {
                    const float r = b.R[cell(v, jj, ii)];
                    s[v]          = tile_sum(th, r * r);
                }
                wait_ticket(0);
                if (tid == 0)
                    for (int v = 0; v < 4; ++v) {
                        HwAtomicF{b.Sq[v]}.fetch_add(s[v]);
                        b.PartSq[static_cast<std::size_t>(v) * T + p] = s[v];
                    }
            } else if (t <= static_cast<std::uint32_t>(2 * T + 4)) {
                // ---- the deterministic reductions, after the group's own barrier ----
                if (tid == 0) {
                    HwAtomicU done{b.ts[1]};
                    while (done.load(sycl::memory_order::acquire) < static_cast<std::uint32_t>(2 * T + 1)) {}
                }
                sycl::group_barrier(grp);
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
                const int d = static_cast<int>(t) - (2 * T + 1);
                const float s = det_sum(th, T, [&](int k) { return b.PartSq[static_cast<std::size_t>(d) * T + k]; });
                if (tid == 0) b.DetSq[d] = s;
                if (d == 0) {
                    const float mn = det_min(th, T, [&](int k) { return b.PartDt[k]; });
                    const float mx = det_max(th, T, [&](int k) { return b.PartWs[k]; });
                    if (tid == 0) {
                        b.DetDt[0] = mn;
                        b.DetWs[0] = mx;
                    }
                }
            } else {
                // ---- the error: wait on the four reduction groups it compares ----
                for (int g = 0; g < 4; ++g) wait_ticket(static_cast<std::uint32_t>(2 * T + 1 + g));
                if (tid == 0) {
                    b.Err[0] = HwAtomicF{b.Dt[0]}.load() - b.DetDt[0];
                    b.Err[1] = HwAtomicF{b.Ws[0]}.load() - b.DetWs[0];
                    for (int v = 0; v < 4; ++v) b.Err[2 + v] = HwAtomicF{b.Sq[v]}.load() - b.DetSq[v];
                }
            }

            // ---- completion: barrier, release fence, the flag of this ticket and the done counter ----
            sycl::group_barrier(grp);
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
            if (tid == 0) {
                HwAtomicU flag{b.ts[2 + t]};
                flag.store(1u, sycl::memory_order::release);
                HwAtomicU done{b.ts[1]};
                done.fetch_add(1u, sycl::memory_order::release);
            }
        });
    });
}

}  // namespace nsr
