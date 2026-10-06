// examples/softmax/main.cpp — row-wise softmax on CoSpan (unfused, fused) against a hand-written SYCL kernel.
//
//   softmax [--n 65536] [--m 4096] [--iters 20] [--warmup 3] [--variant all|unfused|fused|handwritten] [--check]
//
// y[i][j] = exp(x[i][j] - max_i) / sum_j exp(x[i][j] - max_i), n rows x m columns of f32, one work-group of 256 per row.
// The input is generated ON THE DEVICE from a deterministic integer hash (every program variant uses the same generator,
// and the host reference recomputes the same floats), so a 1 GiB matrix needs no host copy.
//
//   unfused      three CoSpan components through global memory (RowMax, ExpSum, Normalize)
//   fused        ONE CoSpan launch, three stages that keep the row in local memory
//   handwritten  one plain SYCL kernel outside CoSpan (own queue, own USM buffers) doing the same algorithm with the same
//                work-group size and local memory, on the same device
//
// Per variant: ms per softmax, GB/s against the IDEAL traffic of 2 n m 4 bytes (read x once, write y once), the percentage
// of the device-to-device copy of the same bytes (the practical peak), and the time relative to the handwritten kernel.
// --check compares every variant with a host double-precision reference: max absolute error <= 1e-5.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "systems.hpp"

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;

namespace {

struct Options {
    int         n       = 65536;
    int         m       = 4096;
    int         iters   = 20;
    int         warmup  = 3;
    std::string variant = "all";
    bool        check   = false;
    double      tol     = 1e-5;  // max absolute error against the host reference
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a   = argv[i];
        const auto        val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--n") o.n = std::stoi(val());
        else if (a == "--m") o.m = std::stoi(val());
        else if (a == "--iters") o.iters = std::stoi(val());
        else if (a == "--warmup") o.warmup = std::stoi(val());
        else if (a == "--variant") o.variant = val();
        else if (a == "--tol") o.tol = std::stod(val());
        else if (a == "--check") o.check = true;
        else throw std::invalid_argument("unknown argument " + a);
    }
    return o;
}

// ---- the deterministic input ------------------------------------------------------------------------------------
// An integer hash of the flat index to a float in [-4, 4): 24 hash bits, exact in f32. Plain integer and float arithmetic,
// so the device and the host compute bit-identical values.
inline std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
inline float input_value(std::uint64_t k) {
    const std::uint32_t lo = static_cast<std::uint32_t>(k);
    const std::uint32_t hi = static_cast<std::uint32_t>(k >> 32);
    const std::uint32_t h  = mix32(mix32(0x85ebca6bu + 0x9e3779b9u) ^ mix32(lo + mix32(hi + 1u)));
    return (static_cast<float>(h >> 8) * (1.0f / 8388608.0f) - 1.0f) * 4.0f;
}

/// Max absolute error of `y` (n x m, row-major) against the double-precision softmax of the generated input.
double max_abs_error(const float* y, std::size_t n, std::size_t m) {
    double              worst = 0.0;
    std::vector<double> e(m);
    for (std::size_t i = 0; i < n; ++i) {
        double mx = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < m; ++j) mx = std::max(mx, static_cast<double>(input_value(i * m + j)));
        double sum = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            e[j] = std::exp(static_cast<double>(input_value(i * m + j)) - mx);
            sum += e[j];
        }
        for (std::size_t j = 0; j < m; ++j) {
            const double d = std::fabs(static_cast<double>(y[i * m + j]) - e[j] / sum);
            if (!(d <= worst)) worst = std::isnan(d) ? std::numeric_limits<double>::infinity() : d;
        }
    }
    return worst;
}

// ---- results ------------------------------------------------------------------------------------------------------
struct Result {
    std::string name;
    bool        ran = false, ok = true;
    double      ms  = 0.0;
    double      err = -1.0;  // max abs error, when checked
    std::string note;
};

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

/// A CoSpan model: warm up (forward + wait), time `iters` forwards followed by ONE wait; check y when asked.
template <class Top>
Result run_cospan(const char* name, const Options& o) {
    Result r;
    r.name = name;
    cs::Model   model = cs::create_model<Top>(cs::ModelConfig{}, o.n, o.m);
    const auto  n = static_cast<std::size_t>(o.n), m = static_cast<std::size_t>(o.m);
    try {
        model.setup();
        model.update_input("x", cs::init::custom<DType::f32, 2>(
                                    [n, m](sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                                           cs::Initializer<DType::f32, 2> out) {
                                        return q.parallel_for(dev, deps, sycl::range<1>{n * m}, [=](sycl::id<1> t) {
                                            const std::size_t k = t[0];
                                            out[sycl::id<2>{k / m, k % m}] = input_value(k);
                                        });
                                    }));
        for (int i = 0; i < o.warmup; ++i) {
            model.forward();
            model.wait();
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < o.iters; ++i) model.forward();
        model.wait();
        const auto t1 = std::chrono::steady_clock::now();
        r.ms          = ms_between(t0, t1) / o.iters;
        r.ran         = true;
        if (o.check) {
            const auto view = model.view_output<DType::f32, 2>("y");
            view.sync();
            const auto s = view.span();
            r.err        = max_abs_error(s.data(), n, m);
            r.ok         = r.err <= o.tol;
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

// ---- the hand-written kernel: the same algorithm, outside CoSpan --------------------------------------------------
/// One work-group of 256 per row (a grid-stride loop over rows, run with one group per row, as the fused launch is): the
/// row goes into local memory, the max is reduced, the row is exponentiated in local memory, the sum is reduced, and the
/// normalized row is written. Local memory: m floats of row and two scalars.
sycl::event handwritten_softmax(sycl::queue& q, const float* x, float* y, std::uint32_t n, std::uint32_t m) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> local{sycl::range<1>{static_cast<std::size_t>(m) + 2}, h};
        h.parallel_for(sycl::nd_range<1>{static_cast<std::size_t>(n) * sm::kWg, sm::kWg}, [=](sycl::nd_item<1> it) {
            float*              row = &local[0];
            float*              sc  = row + m;
            const std::uint32_t l   = static_cast<std::uint32_t>(it.get_local_id(0));
            const std::uint32_t wg  = static_cast<std::uint32_t>(it.get_local_range(0));
            const std::uint32_t ng  = static_cast<std::uint32_t>(it.get_group_range(0));
            for (std::uint32_t i = static_cast<std::uint32_t>(it.get_group(0)); i < n; i += ng) {
                const std::size_t base = static_cast<std::size_t>(i) * m;
                float             acc  = std::numeric_limits<float>::lowest();
                for (std::uint32_t j = l; j < m; j += wg) {
                    const float v = x[base + j];
                    row[j]        = v;
                    acc           = sycl::fmax(acc, v);
                }
                const float mx = sycl::reduce_over_group(it.get_group(), acc, sycl::maximum<float>());
                if (l == 0) sc[0] = mx;
                sycl::group_barrier(it.get_group());
                const float mxv = sc[0];
                float       sum = 0.0f;
                for (std::uint32_t j = l; j < m; j += wg) {
                    const float v = sycl::exp(row[j] - mxv);
                    row[j]        = v;
                    sum += v;
                }
                const float s = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                if (l == 0) sc[1] = s;
                sycl::group_barrier(it.get_group());
                const float sv = sc[1];
                for (std::uint32_t j = l; j < m; j += wg) y[base + j] = row[j] / sv;
                sycl::group_barrier(it.get_group());  // before the next row of the stride reuses the local row
            }
        });
    });
}

/// A plain device-to-device copy kernel: one float4 per work-item (count % 4 elements, if any, are copied by a second kernel).
void copy_kernel(sycl::queue& q, const float* x, float* y, std::size_t count) {
    const std::size_t vec = count / 4;
    q.parallel_for(sycl::range<1>{vec}, [=](sycl::id<1> t) {
        reinterpret_cast<sycl::float4*>(y)[t[0]] = reinterpret_cast<const sycl::float4*>(x)[t[0]];
    });
    if (count % 4 != 0)
        q.parallel_for(sycl::range<1>{count % 4}, [=](sycl::id<1> t) { y[vec * 4 + t[0]] = x[vec * 4 + t[0]]; });
}

/// Hand-written softmax and the device-to-device copies of the same bytes (the practical peak), on their own queue and USM.
/// Two copies are timed, the runtime's q.memcpy and a copy kernel; the peak is the faster of the two.
struct Handwritten {
    Result handwritten, copy, copy_kernel;
};
Handwritten run_handwritten(const sycl::device& dev, const Options& o, bool want_handwritten, bool want_copy) {
    Handwritten out;
    out.handwritten.name = "handwritten";
    out.copy.name        = "copy q.memcpy";
    out.copy_kernel.name = "copy kernel";
    const std::size_t n = static_cast<std::size_t>(o.n), m = static_cast<std::size_t>(o.m), count = n * m;
    sycl::queue       q{dev, sycl::property::queue::in_order{}};
    float*            x = sycl::malloc_device<float>(count, q);
    float*            y = sycl::malloc_device<float>(count, q);
    try {
        if (x == nullptr || y == nullptr) throw std::runtime_error("device allocation failed");
        q.parallel_for(sycl::range<1>{count}, [=](sycl::id<1> t) { x[t[0]] = input_value(t[0]); }).wait();
        const auto time = [&](auto&& launch) {
            for (int i = 0; i < o.warmup; ++i) launch();
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < o.iters; ++i) launch();
            q.wait();
            return ms_between(t0, std::chrono::steady_clock::now()) / o.iters;
        };
        if (want_handwritten) {
            out.handwritten.ms  = time([&] { handwritten_softmax(q, x, y, static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(m)); });
            out.handwritten.ran = true;
            if (o.check) {
                std::vector<float> host(count);
                q.memcpy(host.data(), y, count * sizeof(float)).wait();
                out.handwritten.err = max_abs_error(host.data(), n, m);
                out.handwritten.ok  = out.handwritten.err <= o.tol;
            }
        }
        if (want_copy) {
            out.copy.ms  = time([&] { q.memcpy(y, x, count * sizeof(float)); });
            out.copy.ran = true;
            out.copy_kernel.ms  = time([&] { copy_kernel(q, x, y, count); });
            out.copy_kernel.ran = true;
        }
    } catch (const std::exception& e) {
        out.handwritten.ok   = false;
        out.handwritten.note = std::string("exception: ") + e.what();
    }
    sycl::free(x, q);
    sycl::free(y, q);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (o.n <= 0 || o.m <= 0) throw std::invalid_argument("--n and --m must be positive");
        if (o.variant != "all" && o.variant != "unfused" && o.variant != "fused" && o.variant != "handwritten")
            throw std::invalid_argument("--variant must be all|unfused|fused|handwritten");
        if ((static_cast<std::size_t>(o.m) + 2) * sizeof(float) > 48 * 1024)
            throw std::invalid_argument("--m: a row (m floats) plus two scalars must fit in 48 KB of local memory (m <= 12286)");

        const sycl::device dev{sycl::default_selector_v};
        const double       bytes = 2.0 * static_cast<double>(o.n) * static_cast<double>(o.m) * 4.0;  // ideal: read x, write y
        std::printf("softmax: n=%d m=%d (%.3f GiB per matrix) iters=%d warmup=%d device=\"%s\"\n", o.n, o.m,
                    static_cast<double>(o.n) * o.m * 4.0 / (1024.0 * 1024.0 * 1024.0), o.iters, o.warmup,
                    dev.get_info<sycl::info::device::name>().c_str());
        std::printf("GB/s uses the IDEAL traffic of 2*n*m*4 bytes = %.3f GB per softmax (not measured)\n", bytes / 1e9);

        const auto         want = [&](const char* nm) { return o.variant == "all" || o.variant == nm; };
        std::vector<Result> results;
        if (want("unfused")) results.push_back(run_cospan<sm::SoftmaxUnfused>("unfused", o));
        if (want("fused")) results.push_back(run_cospan<sm::SoftmaxFused>("fused", o));
        const Handwritten hw = run_handwritten(dev, o, want("handwritten"), /*want_copy=*/true);
        if (want("handwritten")) results.push_back(hw.handwritten);

        const double hw_ms   = hw.handwritten.ran ? hw.handwritten.ms : 0.0;
        // the practical peak: the faster of the two device-to-device copies of the same bytes
        const double peak_ms = hw.copy.ran ? std::min(hw.copy.ms, hw.copy_kernel.ms) : 0.0;
        std::printf("\n%-13s %10s %10s %12s %16s\n", "variant", "ms", "GB/s", "% of copy", "time / handwritten");
        bool all_ok = true;
        for (const Result& r : results) {
            if (!r.ran) {
                std::printf("%-13s %10s %10s %12s %16s  FAIL: %s\n", r.name.c_str(), "-", "-", "-", "-", r.note.c_str());
                all_ok = false;
                continue;
            }
            const double gbs = bytes / (r.ms * 1e6);
            char         pct[32] = "-", ratio[32] = "-";
            if (peak_ms > 0.0) std::snprintf(pct, sizeof pct, "%.1f%%", 100.0 * peak_ms / r.ms);
            if (hw_ms > 0.0) std::snprintf(ratio, sizeof ratio, "%.3f", r.ms / hw_ms);
            std::printf("%-13s %10.4f %10.1f %12s %16s", r.name.c_str(), r.ms, gbs, pct, ratio);
            if (!r.ok) {
                std::printf("  FAIL%s%s", r.note.empty() ? "" : ": ", r.note.c_str());
                all_ok = false;
            }
            std::printf("\n");
        }
        for (const Result* c : {&hw.copy, &hw.copy_kernel}) {
            if (!c->ran) continue;
            char pct[32];
            std::snprintf(pct, sizeof pct, "%.1f%%", 100.0 * peak_ms / c->ms);
            std::printf("%-13s %10.4f %10.1f %12s %16s\n", c->name.c_str(), c->ms, bytes / (c->ms * 1e6), pct, "-");
        }

        if (o.check) {
            std::printf("\ncheck (max absolute error vs the host double reference, tolerance %.1e):\n", o.tol);
            for (const Result& r : results)
                if (r.err >= 0.0) std::printf("  %-13s %.3e  %s\n", r.name.c_str(), r.err, r.ok ? "OK" : "FAIL");
        }
        std::printf("\n%s\n", all_ok ? "ALL OK" : "FAILED");
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "softmax: error: %s\n", e.what());
        return 2;
    }
}
