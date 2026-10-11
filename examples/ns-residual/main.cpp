// examples/ns-residual/main.cpp — one residual evaluation of the 2D compressible Navier-Stokes equations, on CoSpan
// (unfused, fused) against a hand-written single-kernel SYCL implementation.
//
//   ns-residual [--variant all|unfused|fused|handwritten] [--nx 1024] [--ny 1024] [--iters 20] [--warmup 3] [--check]
//               [--limiter minmod|vanleer] [--bc walls|periodic] [--range-split flatten|pad|padded|all]
//   --range-split picks the cs::range_split strategy of the fused variant (flatten_and_split, pad_and_split, padded; all runs the three
//   as fused-flatten, fused-pad, fused-padded).
//
// R = -div F_conv + div F_visc per cell (f32, ideal gas gamma 1.4, mu 0.01, Pr 0.72, dx = dy = 1), plus three diagnostics:
// the minimum local timestep, the maximum wave speed and the sum of R^2 per variable. Each diagnostic is computed twice:
// atomically (fetch_min / fetch_max / fetch_add into one global value) and deterministically (reductions of per-tile
// partials in a fixed order); an error stage reports atomic - deterministic. physics.hpp lists the conventions; the
// README draws the fused group. The initial field (uniform flow plus a smooth perturbation) is generated ON THE DEVICE.
//
// GB/s is the NOMINAL traffic: one read of the 4 state fields and one write of R per cell,
//   bytes = nx * ny * (4 + 4) * 4 B = 32 B per cell   (not measured: fluxes and partials are not counted).
// --check compares every variant with the host double-precision reference (reference.hpp):
//   R relative L2 <= 1e-4; deterministic diagnostics within 1e-4 relative; atomic min and max equal to the deterministic
//   ones; the gap between the atomic and deterministic norms and the error stage's values are REPORTED only;
//   with --bc periodic: uniform flow gives R exactly 0.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "handwritten.hpp"
#include "reference.hpp"
#include "systems.hpp"

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;

namespace {

struct Options {
    std::string variant = "all";
    int         nx = 1024, ny = 1024, iters = 20, warmup = 3;
    bool        check   = false;
    std::string limiter = "minmod";
    std::string bc      = "walls";
    std::string split   = "flatten";  // the range_split strategy of the fused variant: flatten | pad | padded | all
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a   = argv[i];
        const auto        val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--variant") o.variant = val();
        else if (a == "--nx") o.nx = std::stoi(val());
        else if (a == "--ny") o.ny = std::stoi(val());
        else if (a == "--iters") o.iters = std::stoi(val());
        else if (a == "--warmup") o.warmup = std::stoi(val());
        else if (a == "--limiter") o.limiter = val();
        else if (a == "--bc") o.bc = val();
        else if (a == "--range-split") o.split = val();
        else if (a == "--check") o.check = true;
        else throw std::invalid_argument("unknown argument " + a);
    }
    return o;
}

/// Everything one evaluation produced, read back to the host.
struct Snapshot {
    std::vector<float> R;  // (4, ny, nx)
    float              dt = 0, ws = 0, sq[4] = {0, 0, 0, 0};              // the atomic diagnostics
    float              det_dt = 0, det_ws = 0, det_sq[4] = {0, 0, 0, 0};  // the deterministic ones
    float              err[6] = {0, 0, 0, 0, 0, 0};                       // the error stage: atomic - deterministic
};

struct Result {
    std::string name;
    bool        ran = false, ok = true;
    double      ms = 0.0;
    std::string note;
    bool        checked = false;
    Snapshot    snap;
    double      uniform_max = -1.0;  // max |R| of the uniform-flow run (periodic only); -1: not run
};

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---- the initial field, on the device --------------------------------------------------------------------------------
auto make_init(int nx, int ny, float amp) {
    return cs::init::custom<DType::f32, 3>([nx, ny, amp](sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                                                         cs::Initializer<DType::f32, 3> out) {
        const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
        return q.parallel_for(dev, deps, sycl::range<1>{n}, [=](sycl::id<1> t) {
            const int k = static_cast<int>(t[0]);
            const int j = k / nx, i = k % nx;
            float     U[4];
            nsr::init_cons(j, i, amp, U);
            for (int v = 0; v < 4; ++v) out[sycl::id<3>{static_cast<std::size_t>(v), static_cast<std::size_t>(j), static_cast<std::size_t>(i)}] = U[v];
        });
    });
}

// ---- a CoSpan variant -------------------------------------------------------------------------------------------------
struct Paths {
    std::string geo, U, R, Dt, Ws, Sq, DetSq, DetDt, DetWs, Err;
};
Paths paths(bool fused) {
    const std::string p = fused ? "ns." : "";
    return Paths{p + "geo", p + "U", p + "R", p + "Dt", p + "Ws", p + "Sq", p + "detn.Det", p + "detd.Det", p + "detw.Det", p + "err.Err"};
}

template <int Rk>
std::vector<float> read_output(cs::Model& m, const std::string& path) {
    const auto view = m.view_output<DType::f32, Rk>(path);
    view.sync();
    const auto s = view.span();
    return std::vector<float>(s.begin(), s.end());
}

Snapshot read_snapshot(cs::Model& m, const Paths& p) {
    Snapshot   s;
    s.R            = read_output<3>(m, p.R);
    const auto dt  = read_output<1>(m, p.Dt);
    const auto ws  = read_output<1>(m, p.Ws);
    const auto sq  = read_output<1>(m, p.Sq);
    const auto dsq = read_output<1>(m, p.DetSq);
    const auto ddt = read_output<1>(m, p.DetDt);
    const auto dws = read_output<1>(m, p.DetWs);
    const auto err = read_output<1>(m, p.Err);
    s.dt           = dt.at(0);
    s.ws           = ws.at(0);
    s.det_dt       = ddt.at(0);
    s.det_ws       = dws.at(0);
    for (std::size_t v = 0; v < 4; ++v) {
        s.sq[v]     = sq.at(v);
        s.det_sq[v] = dsq.at(v);
    }
    for (std::size_t v = 0; v < 6; ++v) s.err[v] = err.at(v);
    return s;
}

double max_abs(const std::vector<float>& x) {
    double m = 0.0;
    for (float v : x) m = std::max(m, static_cast<double>(std::fabs(v)));
    return m;
}

template <class Top>
Result run_cospan(const char* name, bool fused, const Options& o) {
    Result r;
    r.name = name;
    cs::Model   model = cs::create_model<Top>(o.nx, o.ny);
    const Paths p     = paths(fused);
    try {
        model.setup();
        // the grid (nx, ny, tx, ty) is a free input of the stages that need it, tied into one alias by the system
        model.update_input(p.geo, cs::init::from_host(nsr::geo_values(o.nx, o.ny)));
        model.update_input(p.U, make_init(o.nx, o.ny, 1.0f));
        for (int i = 0; i < o.warmup; ++i) {
            model.run();
            model.wait();
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < o.iters; ++i) model.run();
        model.wait();
        const auto t1 = std::chrono::steady_clock::now();
        r.ms          = ms_between(t0, t1) / o.iters;
        r.ran         = true;
        if (o.check) {
            r.snap    = read_snapshot(model, p);
            r.checked = true;
            if (o.bc == "periodic") {  // uniform flow: R is exactly zero
                model.update_input(p.U, make_init(o.nx, o.ny, 0.0f));
                model.run();
                model.wait();
                r.uniform_max = max_abs(read_output<3>(model, p.R));
            }
        }
    } catch (const std::exception& e) {
        r.ok   = false;
        r.note = std::string("exception: ") + e.what();
    }
    try {
        model.wait();
    } catch (...) {
    }
    return r;
}

// ---- the hand-written variant -----------------------------------------------------------------------------------------
template <class Lim, class Bc>
Result run_handwritten(const sycl::device& dev, const Options& o) {
    Result r;
    r.name = "handwritten";
    const nsr::Geo    geo = nsr::make_geo(o.nx, o.ny);
    const std::size_t NX = static_cast<std::size_t>(o.nx), NY = static_cast<std::size_t>(o.ny);
    const std::size_t T = static_cast<std::size_t>(geo.tx) * static_cast<std::size_t>(geo.ty);
    sycl::queue       q{dev, sycl::property::queue::in_order{}};
    nsr::HwBuffers    b{};
    std::vector<void*> allocs;
    const auto         dev_f = [&](std::size_t n) {
        float* p = sycl::malloc_device<float>(n, q);
        if (p == nullptr) throw std::runtime_error("device allocation failed");
        allocs.push_back(p);
        return p;
    };
    try {
        b.U      = dev_f(4 * NY * NX);
        b.Fcx    = dev_f(4 * NY * (NX + 1));
        b.Fvx    = dev_f(4 * NY * (NX + 1));
        b.Fcy    = dev_f(4 * (NY + 1) * NX);
        b.Fvy    = dev_f(4 * (NY + 1) * NX);
        b.R      = dev_f(4 * NY * NX);
        b.PartDt = dev_f(T);
        b.PartWs = dev_f(T);
        b.PartSq = dev_f(4 * T);
        b.Dt     = dev_f(1);
        b.Ws     = dev_f(1);
        b.Sq     = dev_f(4);
        b.DetSq  = dev_f(4);
        b.DetDt  = dev_f(1);
        b.DetWs  = dev_f(1);
        b.Err    = dev_f(6);
        b.ts     = sycl::malloc_device<std::uint32_t>(2 + nsr::hw_groups(geo), q);
        if (b.ts == nullptr) throw std::runtime_error("device allocation failed");
        allocs.push_back(b.ts);

        const auto init = [&](float amp) {  // the same generator as the CoSpan variants
            float*            U     = b.U;
            const int         nx    = o.nx;
            const std::size_t ncell = NX * NY;
            q.parallel_for(sycl::range<1>{ncell}, [=](sycl::id<1> t) {
                const int k = static_cast<int>(t[0]);
                float     u[4];
                nsr::init_cons(k / nx, k % nx, amp, u);
                for (std::size_t v = 0; v < 4; ++v) U[v * ncell + static_cast<std::size_t>(k)] = u[v];
            }).wait();
        };
        const auto fetch = [&](const float* d, std::size_t n) {
            std::vector<float> h(n);
            q.memcpy(h.data(), d, n * sizeof(float)).wait();
            return h;
        };

        init(1.0f);
        for (int i = 0; i < o.warmup; ++i) {
            nsr::hw_forward<Lim, Bc>(q, b, geo);
            q.wait();
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < o.iters; ++i) nsr::hw_forward<Lim, Bc>(q, b, geo);
        q.wait();
        const auto t1 = std::chrono::steady_clock::now();
        r.ms          = ms_between(t0, t1) / o.iters;
        r.ran         = true;
        if (o.check) {
            r.snap.R       = fetch(b.R, 4 * NY * NX);
            const auto dt  = fetch(b.Dt, 1), ws = fetch(b.Ws, 1), sq = fetch(b.Sq, 4);
            const auto ddt = fetch(b.DetDt, 1), dws = fetch(b.DetWs, 1), dsq = fetch(b.DetSq, 4), err = fetch(b.Err, 6);
            r.snap.dt = dt[0];
            r.snap.ws = ws[0];
            r.snap.det_dt = ddt[0];
            r.snap.det_ws = dws[0];
            for (std::size_t v = 0; v < 4; ++v) {
                r.snap.sq[v]     = sq[v];
                r.snap.det_sq[v] = dsq[v];
            }
            for (std::size_t v = 0; v < 6; ++v) r.snap.err[v] = err[v];
            r.checked = true;
            if (o.bc == "periodic") {
                init(0.0f);
                nsr::hw_forward<Lim, Bc>(q, b, geo);
                q.wait();
                r.uniform_max = max_abs(fetch(b.R, 4 * NY * NX));
            }
        }
    } catch (const std::exception& e) {
        r.ok   = false;
        r.note = std::string("exception: ") + e.what();
    }
    try {
        q.wait();
    } catch (...) {
    }
    for (void* p : allocs) sycl::free(p, q);
    return r;
}

// ---- the check --------------------------------------------------------------------------------------------------------
double rel(double a, double ref) { return std::fabs(a - ref) / std::max(std::fabs(ref), 1e-300); }

/// Compare one variant with the reference; prints one line. Returns whether every criterion holds.
bool check_variant(const Result& r, const nsr::RefResult& ref, bool periodic) {
    const Snapshot& s = r.snap;
    double          num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < s.R.size(); ++k) {
        const double d = static_cast<double>(s.R[k]) - ref.R[k];
        num += d * d;
        den += ref.R[k] * ref.R[k];
    }
    const double l2 = std::sqrt(num / std::max(den, 1e-300));
    double       det = std::max(rel(s.det_dt, ref.dt_min), rel(s.det_ws, ref.ws_max));
    double       atomic_norm = 0.0;
    for (int v = 0; v < 4; ++v) {
        det         = std::max(det, rel(s.det_sq[v], ref.sq[v]));
        atomic_norm = std::max(atomic_norm, rel(s.sq[v], s.det_sq[v]));
    }
    const bool minmax_equal = (s.dt == s.det_dt) && (s.ws == s.det_ws);
    const bool uniform_ok   = !periodic || r.uniform_max == 0.0;
    const bool ok           = l2 <= 1e-4 && det <= 1e-4 && minmax_equal && uniform_ok;  // the norm gap is reported only
    std::printf("  check %s: R_relL2=%.2e det_rel=%.2e atomic_minmax_equal=%s atomic_norm_gap_rel=%.2e", r.name.c_str(), l2, det, minmax_equal ? "yes" : "NO",
                atomic_norm);
    if (periodic) std::printf(" uniform_R_max=%.1e", r.uniform_max);
    std::printf(" error[dt ws sq0 sq1 sq2 sq3]=[%.2e %.2e %.2e %.2e %.2e %.2e] %s\n", static_cast<double>(s.err[0]), static_cast<double>(s.err[1]),
                static_cast<double>(s.err[2]), static_cast<double>(s.err[3]), static_cast<double>(s.err[4]), static_cast<double>(s.err[5]),
                ok ? "OK" : "FAIL");
    return ok;
}

template <class Lim, class Bc>
int run_all(const Options& o) {
    const sycl::device dev{sycl::default_selector_v};
    const double       bytes = static_cast<double>(o.nx) * static_cast<double>(o.ny) * 8.0 * 4.0;  // 4 state fields read + R (4 fields) written
    std::printf("ns-residual: nx=%d ny=%d limiter=%s bc=%s iters=%d warmup=%d device=\"%s\"\n", o.nx, o.ny, Lim::name, Bc::name, o.iters, o.warmup,
                dev.get_info<sycl::info::device::name>().c_str());
    std::printf("GB/s uses the NOMINAL traffic nx*ny*(4 state reads + 4 R writes)*4 B = %.3f GB per evaluation (not measured)\n", bytes / 1e9);

    const auto          want = [&](const char* nm) { return o.variant == "all" || o.variant == nm; };
    std::vector<Result> results;
    if (want("unfused")) results.push_back(run_cospan<nsr::NsUnfused<Lim, Bc>>("unfused", false, o));
    const auto want_split = [&](const char* nm) { return o.split == "all" || o.split == nm; };
    if (want("fused") && want_split("flatten"))
        results.push_back(run_cospan<nsr::NsFused<Lim, Bc, cs::range_split::flatten_and_split>>("fused-flatten", true, o));
    if (want("fused") && want_split("pad"))
        results.push_back(run_cospan<nsr::NsFused<Lim, Bc, cs::range_split::pad_and_split>>("fused-pad", true, o));
    if (want("fused") && want_split("padded"))
        results.push_back(run_cospan<nsr::NsFused<Lim, Bc, cs::range_split::padded>>("fused-padded", true, o));
    if (want("handwritten")) results.push_back(run_handwritten<Lim, Bc>(dev, o));

    std::printf("\n%-14s %10s %10s\n", "variant", "ms/eval", "GB/s");
    bool all_ok = true;
    for (const Result& r : results) {
        if (!r.ran || !r.ok) {
            std::printf("%-14s %10s %10s  FAIL: %s\n", r.name.c_str(), "-", "-", r.note.c_str());
            all_ok = false;
            continue;
        }
        std::printf("%-14s %10.4f %10.1f\n", r.name.c_str(), r.ms, bytes / (r.ms * 1e6));
    }

    if (o.check) {
        // the reference, from the same initial field the devices generate (recomputed on the host)
        const std::size_t NX = static_cast<std::size_t>(o.nx), NY = static_cast<std::size_t>(o.ny);
        std::vector<float> U0(4 * NY * NX);
        for (std::size_t j = 0; j < NY; ++j)
            for (std::size_t i = 0; i < NX; ++i) {
                float u[4];
                nsr::init_cons(static_cast<int>(j), static_cast<int>(i), 1.0f, u);
                for (std::size_t v = 0; v < 4; ++v) U0[(v * NY + j) * NX + i] = u[v];
            }
        const nsr::RefResult ref = nsr::reference_residual<Lim>(o.nx, o.ny, Bc::walls, U0);
        std::printf("\ncheck vs the host double reference (R relL2 <= 1e-4, deterministic diagnostics <= 1e-4, atomic min/max equal%s; the atomic-vs-deterministic norm gap is reported only):\n",
                    Bc::walls ? "" : ", uniform flow R == 0");
        for (const Result& r : results)
            if (r.checked) all_ok = check_variant(r, ref, !Bc::walls) && all_ok;
            else if (r.ran) all_ok = false;
    }
    std::printf("\n%s\n", all_ok ? "ALL OK" : "FAILED");
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (o.nx < nsr::kTile || o.ny < nsr::kTile || o.nx % nsr::kTile != 0 || o.ny % nsr::kTile != 0)
            throw std::invalid_argument("--nx and --ny must be positive multiples of 16");
        if (o.variant != "all" && o.variant != "unfused" && o.variant != "fused" && o.variant != "handwritten")
            throw std::invalid_argument("--variant must be all|unfused|fused|handwritten");
        if (o.limiter != "minmod" && o.limiter != "vanleer") throw std::invalid_argument("--limiter must be minmod|vanleer");
        if (o.split != "flatten" && o.split != "pad" && o.split != "padded" && o.split != "all")
            throw std::invalid_argument("--range-split must be flatten|pad|padded|all");
        if (o.bc != "walls" && o.bc != "periodic") throw std::invalid_argument("--bc must be walls|periodic");

        const bool walls = o.bc == "walls";
        if (o.limiter == "minmod") return walls ? run_all<nsr::MinMod, nsr::BcWalls>(o) : run_all<nsr::MinMod, nsr::BcPeriodic>(o);
        return walls ? run_all<nsr::VanLeer, nsr::BcWalls>(o) : run_all<nsr::VanLeer, nsr::BcPeriodic>(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ns-residual: error: %s\n", e.what());
        return 2;
    }
}
