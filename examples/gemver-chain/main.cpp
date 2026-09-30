// examples/gemver-chain/main.cpp — the GEMVER -> BiCGK -> GEMVER chain on CoSpan, in several fusion styles.
//
//   gemver-chain [--n 16384] [--R 64] [--iters 20] [--warmup 3]
//                [--variant all|unfused|parallel|fused|barrier|onepass]
//                [--blocks-per-cu 2] [--spin-limit 16777216] [--check]
//
// Per variant: create the model, set the inputs from the shared generator, warm up
// (forward + wait), then time `iters` forwards followed by ONE wait. --check compares
// the outputs with the host double-precision reference and the variants with each other.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "common/chain_data.hpp"
#include "systems.hpp"

namespace cs = naga::cospan;
namespace gd = gemver_chain;
using cs::DType;

namespace {

struct Options {
    int           n              = 16384;
    int           R              = 64;
    int           iters          = 20;
    int           warmup         = 3;
    std::string   variant        = "all";
    int           blocks_per_cu  = 2;
    std::uint32_t spin_limit     = 1u << 24;  // barrier polls before a group gives up
    bool          check          = false;
    bool          verbose        = false;
    double        tol            = 1e-4;      // relative L2 against the host reference
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto        val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--n") o.n = std::stoi(val());
        else if (a == "--R") o.R = std::stoi(val());
        else if (a == "--iters") o.iters = std::stoi(val());
        else if (a == "--warmup") o.warmup = std::stoi(val());
        else if (a == "--variant") o.variant = val();
        else if (a == "--blocks-per-cu") o.blocks_per_cu = std::stoi(val());
        else if (a == "--spin-limit") o.spin_limit = static_cast<std::uint32_t>(std::stoul(val()));
        else if (a == "--tol") o.tol = std::stod(val());
        else if (a == "--check") o.check = true;
        else if (a == "--verbose") o.verbose = true;
        else throw std::invalid_argument("unknown argument " + a);
    }
    return o;
}

// ---- the inputs and outputs of one chain ----------------------------------------------------------------
struct Data {
    std::vector<float> A0;
    gd::Vectors        v;
};

/// What a run leaves behind: the chain's outputs, read back after the timed run.
struct Outputs {
    std::vector<float> x1, w1, q, s, x2, w2, B2;
    std::vector<const std::vector<float>*> vectors() const { return {&x1, &w1, &q, &s, &x2, &w2}; }
};
constexpr const char* kVecNames[] = {"x1", "w1", "q", "s", "x2", "w2"};

void set_inputs(cs::Model& m, const Data& d) {
    const auto v1 = [&](const char* path, const std::vector<float>& v) {
        m.set_input<DType::f32, 1>(path, std::span<const float>{v});
    };
    m.set_input<DType::f32, 2>("A", std::span<const float>{d.A0});
    v1("u1a", d.v.u1); v1("v1a", d.v.v1); v1("u2a", d.v.u2); v1("v2a", d.v.v2); v1("ya", d.v.y); v1("za", d.v.z);
    v1("u1b", d.v.u1); v1("v1b", d.v.v1); v1("u2b", d.v.u2); v1("v2b", d.v.v2);  // G2's own copies, not tied
}

// A HostView owns the host snapshot its span() points into: keep the view alive while reading it.
std::vector<float> read1(cs::Model& m, const char* path) {
    const auto view = m.view_output<DType::f32, 1>(path);
    const auto s    = view.span();
    return std::vector<float>(s.begin(), s.end());
}
Outputs read_outputs(cs::Model& m, bool with_B2) {
    Outputs o{read1(m, "x1"), read1(m, "w1"), read1(m, "q"), read1(m, "s"), read1(m, "x2"), read1(m, "w2"), {}};
    if (with_B2) {
        const auto view = m.view_output<DType::f32, 2>("B2");
        const auto s    = view.span();
        o.B2.assign(s.begin(), s.end());
    }
    return o;
}

// ---- variants -------------------------------------------------------------------------------------------
enum class Variant { unfused, parallel, fused, barrier };
constexpr const char* name_of(Variant v) {
    switch (v) {
        case Variant::unfused: return "unfused";
        case Variant::parallel: return "parallel";
        case Variant::fused: return "fused";
        case Variant::barrier: return "barrier";
    }
    return "?";
}

cs::Model make_model(Variant v, const Options& o, std::size_t groups) {
    switch (v) {
        case Variant::unfused: return cs::create_model<gc::Chain>(cs::ModelConfig{}, gc::Style::unfused, o.n, o.R);
        case Variant::parallel: return cs::create_model<gc::Chain>(cs::ModelConfig{}, gc::Style::parallel, o.n, o.R);
        case Variant::fused: return cs::create_model<gc::Chain>(cs::ModelConfig{}, gc::Style::fused, o.n, o.R);
        case Variant::barrier:
            return cs::create_model<gc::BarrierChain>(cs::ModelConfig{}, o.n, o.R, groups, o.spin_limit);
    }
    throw std::logic_error("unknown variant");
}

/// The value of every barrier's timeout flag; nonzero = some group gave up waiting.
/// A view of an atomic output is read through its value dtype.
std::uint32_t barrier_timeouts(cs::Model& m) {
    std::uint32_t any = 0;
    for (int i = 1; i <= gc::BarrierChain::kBarriers; ++i) {
        const std::string path = "bar" + std::to_string(i) + ".timeout";
        const auto        view = m.view_output<DType::u32, 1>(path);  // the view owns the snapshot
        any |= view.span()[0];
    }
    return any;
}

struct Report {
    Variant     variant;
    bool        ran = false, ok = true;
    double      ms = 0.0;  // per chain
    std::string note;
    Outputs     out;
    bool        have_out = false;
};

Report run_variant(Variant v, const Options& o, const Data& d, std::size_t groups, bool need_B2) {
    Report rep;
    rep.variant = v;
    cs::Model model = make_model(v, o, groups);
    try {
        model.setup();
        set_inputs(model, d);
        for (int i = 0; i < o.warmup; ++i) {
            model.forward();
            model.wait();
        }
        bool timed_out = false;
        if (v == Variant::barrier && barrier_timeouts(model) != 0) timed_out = true;
        if (!timed_out) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < o.iters; ++i) model.forward();
            model.wait();  // ONE wait for the whole batch
            const auto t1 = std::chrono::steady_clock::now();
            rep.ms        = std::chrono::duration<double, std::milli>(t1 - t0).count() / o.iters;
            rep.ran       = true;
            if (v == Variant::barrier && barrier_timeouts(model) != 0) timed_out = true;
        }
        if (timed_out) {
            rep.ok   = false;
            rep.note = "grid not co-resident at G=" + std::to_string(groups);
        }
        if (o.check && rep.ok) {
            rep.out      = read_outputs(model, need_B2);
            rep.have_out = true;
        }
    } catch (const std::exception& e) {
        rep.ok   = false;
        rep.note = std::string("exception: ") + e.what();
    }
    try {
        model.wait();  // always, before the model is destroyed
    } catch (...) {
    }
    return rep;
}

// ---- checks -----------------------------------------------------------------------------------------------
bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

/// Against the host reference; prints one line, returns false on any error above the tolerance.
bool check_against_reference(const Report& r, const gd::ChainResult& ref, const Options& o, bool with_B2) {
    const std::vector<double>* want[] = {&ref.x1, &ref.w1, &ref.q, &ref.s, &ref.x2, &ref.w2};
    const auto                 got    = r.out.vectors();
    bool                       ok     = true;
    std::printf("  check %-8s vs reference, rel L2:", name_of(r.variant));
    for (int i = 0; i < 6; ++i) {
        const double e = gd::relative_l2(*got[i], *want[i]);
        std::printf(" %s %.2e", kVecNames[i], e);
        if (!(e <= o.tol)) {
            ok = false;
            if (o.verbose)
                std::printf(" [got %g %g %g %g | want %g %g %g %g]", (*got[i])[0], (*got[i])[1], (*got[i])[2], (*got[i])[3],
                            (*want[i])[0], (*want[i])[1], (*want[i])[2], (*want[i])[3]);
        }
    }
    if (with_B2) {
        const double e = gd::relative_l2(r.out.B2, ref.B2);
        std::printf(" B2 %.2e", e);
        if (!(e <= o.tol)) ok = false;
    }
    std::printf("  %s\n", ok ? "OK" : "FAIL");
    return ok;
}

/// Two runs against each other. bitwise: they must be bit-identical; else within the tolerance.
bool check_agree(const Report& a, const Report& b, bool bitwise, bool with_B2, const Options& o) {
    bool                    ok  = true, all_bitwise = true;
    double                  worst = 0.0;
    std::vector<const std::vector<float>*> ga = a.out.vectors(), gb = b.out.vectors();
    if (with_B2) {
        ga.push_back(&a.out.B2);
        gb.push_back(&b.out.B2);
    }
    for (std::size_t i = 0; i < ga.size(); ++i) {
        if (!bitwise_equal(*ga[i], *gb[i])) all_bitwise = false;
        worst = std::max(worst, gd::relative_l2(*ga[i], *gb[i]));
    }
    if (bitwise && !all_bitwise) ok = false;
    if (!bitwise && !(worst <= o.tol)) ok = false;
    std::printf("  agree %-8s vs %-8s: %s (worst rel L2 %.2e)  %s\n", name_of(b.variant), name_of(a.variant),
                all_bitwise ? "bitwise equal" : "NOT bitwise equal", worst, ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        gd::check_problem(static_cast<std::size_t>(o.n), static_cast<std::size_t>(o.R));
        if (o.variant != "all" && o.variant != "unfused" && o.variant != "parallel" && o.variant != "fused" &&
            o.variant != "barrier")
            throw std::invalid_argument("--variant must be all|unfused|parallel|fused|barrier");

        const sycl::device dev{sycl::default_selector_v};
        const bool         is_omp = dev.get_backend() == sycl::backend::omp;
        const std::size_t  groups = static_cast<std::size_t>(dev.get_info<sycl::info::device::max_compute_units>()) *
                                    static_cast<std::size_t>(o.blocks_per_cu);
        std::printf("gemver-chain: n=%d R=%d iters=%d warmup=%d device=\"%s\"\n", o.n, o.R, o.iters, o.warmup,
                    dev.get_info<sycl::info::device::name>().c_str());
        std::printf("effective GB/s uses a NOMINAL traffic of 8*n^2*4 bytes = %.3f GB per chain (not measured)\n",
                    gd::nominal_bytes(static_cast<std::size_t>(o.n)) / 1e9);

        std::vector<Variant> todo;
        const auto           want = [&](const char* nm) { return o.variant == "all" || o.variant == nm; };
        if (want("unfused")) todo.push_back(Variant::unfused);
        if (want("parallel")) todo.push_back(Variant::parallel);
        if (want("fused")) todo.push_back(Variant::fused);
        if (want("barrier")) {
            if (is_omp)
                std::printf("barrier: skipped on the omp backend (a software grid barrier needs co-resident groups)\n");
            else
                todo.push_back(Variant::barrier);
        }
        if (todo.empty()) throw std::invalid_argument("--variant: nothing to run for '" + o.variant + "'");

        Data data;
        data.A0 = gd::make_matrix_A0(static_cast<std::size_t>(o.n));
        data.v  = gd::make_vectors(static_cast<std::size_t>(o.n));
        const bool with_B2 = o.n <= 4096;

        std::vector<Report> reports;
        for (Variant v : todo) reports.push_back(run_variant(v, o, data, groups, with_B2));

        // ---- timing table ----
        double base = 0.0;
        for (const Report& r : reports)
            if (r.variant == Variant::unfused && r.ran) base = r.ms;
        std::printf("\n%-9s %12s %10s %10s\n", "variant", "ms/chain", "speedup", "GB/s");
        bool all_ok = true;
        for (const Report& r : reports) {
            if (r.ran) {
                const double gbs = gd::nominal_bytes(static_cast<std::size_t>(o.n)) / (r.ms * 1e6);
                if (base > 0.0)
                    std::printf("%-9s %12.4f %9.2fx %10.1f", name_of(r.variant), r.ms, base / r.ms, gbs);
                else
                    std::printf("%-9s %12.4f %10s %10.1f", name_of(r.variant), r.ms, "-", gbs);
            } else {
                std::printf("%-9s %12s %10s %10s", name_of(r.variant), "-", "-", "-");
            }
            if (!r.ok) {
                std::printf("  FAIL: %s", r.note.c_str());
                all_ok = false;
            }
            std::printf("\n");
        }

        // ---- checks ----
        if (o.check) {
            std::printf("\ncheck (tolerance %.1e, B2 %s):\n", o.tol, with_B2 ? "included" : "skipped for n > 4096");
            const gd::ChainResult ref = gd::reference_chain(static_cast<std::size_t>(o.n), data.A0, data.v, with_B2);
            const Report*         first = nullptr;
            for (const Report& r : reports) {
                if (!r.have_out) {
                    all_ok = false;
                    continue;
                }
                if (!check_against_reference(r, ref, o, with_B2)) all_ok = false;
                if (first == nullptr)
                    first = &r;
                else if (!check_agree(*first, r, /*bitwise=*/true, with_B2, o))
                    all_ok = false;
            }
        }
        std::printf("\n%s\n", all_ok ? "ALL OK" : "FAILED");
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemver-chain: error: %s\n", e.what());
        return 2;
    }
}
