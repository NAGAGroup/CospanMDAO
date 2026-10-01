// Event counters (include/naga/cospan/diagnostics.hpp): a BASELINE. These cases print what CoSpan does per
// forward (the chunk loop's window and the whole forward) and assert only that the counters are wired.
//
//   a  dense:  Prod [N,3] (outside) -> chunked{ A -> internal t [N,4]; B: t -> y } -> Cons (outside), 10 chunks
//   b  ragged: the LRPIM pattern: an internal ragged tie alias [N, c_i, c_i] from alias::input_values("sizes") of a
//              promoted input, members BuildS (reads `sizes`) and Factor (reads its own output), then SumT; chunked
//
// Each case runs two forwards with unchanged inputs; case b then re-initialises the `sizes` input (other counts)
// and runs a third time. Output lines start with "[diag]".

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace tdiag {

using Counts = std::vector<std::uint32_t>;

// ---- dense --------------------------------------------------------------------------------------------------
struct ProdBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> x;
    };
    void operator()(sycl::id<2> i, Inputs, Outputs o) const { o.x[i] = static_cast<float>((i[0] * 7 + i[1] * 3) % 11) + 0.5f; }
};
class Prod : public cs::Component<ProdBody> {
    std::size_t               n_;
    cs::Output<DType::f32, 2> x_;

public:
    explicit Prod(std::size_t n) : n_{n} {}
    void setup() override { declare_output(x_, "x", n_, 3); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(x_.shape(), {}, {x_}, {}); }
};

struct ABody {
    struct Inputs {
        cs::Input<DType::f32, 2> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> t;
    };
    void operator()(sycl::id<2> i, Inputs a, Outputs o) const {
        const float s = a.in[sycl::id<2>{i[0], 0}] + 2.0f * a.in[sycl::id<2>{i[0], 1}] + 3.0f * a.in[sycl::id<2>{i[0], 2}];
        o.t[i]        = s * static_cast<float>(i[1] + 1);
    }
};
class A : public cs::Component<ABody> {
    cs::Input<DType::f32, 2>  in_;
    cs::Output<DType::f32, 2> t_;

public:
    void setup() override {
        declare_input(in_, "in", cs::FromSourceDim{0}, cs::FromSourceDim{1});
        declare_output(t_, "t", cs::FromInputDim{"in", 0}, 4);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(t_.shape(), {in_}, {t_}, {}); }
};

struct BBody {
    struct Inputs {
        cs::Input<DType::f32, 2> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const {
        float s = 0.0f;
        for (std::size_t k = 0; k < a.in.shape()[1]; ++k) s += a.in[sycl::id<2>{i[0], k}];
        o.y[i] = s;
    }
};
class B : public cs::Component<BBody> {
    cs::Input<DType::f32, 2>  in_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(in_, "in", cs::FromSourceDim{0}, cs::FromSourceDim{1});
        declare_output(y_, "y", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {in_}, {y_}, {}); }
};

struct ConsBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.out[i] = 2.0f * a.in[i] + 1.0f; }
};
class Cons : public cs::Component<ConsBody> {
    cs::Input<DType::f32, 1>  in_;
    cs::Output<DType::f32, 1> out_;

public:
    void setup() override {
        declare_input(in_, "in", cs::FromSourceDim{0});
        declare_output(out_, "out", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(out_.shape(), {in_}, {out_}, {}); }
};

class DenseInner : public cs::ChunkedSystem {
public:
    explicit DenseInner(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override {
        add_subsystem<A>("a");
        add_subsystem<B>("b");
        connect("a.t", "b.in");
    }
};
class DenseTop : public cs::System {
    std::size_t n_, budget_;

public:
    DenseTop(std::size_t n, std::size_t budget) : n_{n}, budget_{budget} {}
    void setup() override {
        add_subsystem<Prod>("prod", n_);
        add_subsystem<DenseInner>("inner", budget_);
        add_subsystem<Cons>("cons");
        connect("prod.x", "inner.a.in");
        connect("inner.b.y", "cons.in");
    }
};

// ---- the LRPIM pattern ------------------------------------------------------------------------------------------
struct CProdBody {
    struct Inputs {
        cs::Input<DType::u32, 1> raw;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> counts;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.counts[i] = a.raw[i]; }
};
class CProd : public cs::Component<CProdBody> {
    Counts                    init_;
    cs::Input<DType::u32, 1>  raw_;
    cs::Output<DType::u32, 1> counts_;

public:
    explicit CProd(Counts init) : init_{std::move(init)} {}
    void setup() override {
        declare_input(raw_, "raw", init_.size());
        init_input("raw", cs::init::from_host(init_));
        declare_output(counts_, "counts", init_.size());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {raw_}, {counts_}, {});
    }
};

struct SrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> x;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const { o.x[i] = 0.5f * static_cast<float>(i[0]) + 1.0f; }
};
class Src : public cs::Component<SrcBody> {
    std::size_t               n_;
    cs::Output<DType::f32, 1> x_;

public:
    explicit Src(std::size_t n) : n_{n} {}
    void setup() override { declare_output(x_, "x", n_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(x_.shape(), {}, {x_}, {}); }
};

/// BuildS: t = in (q + 1), a tied member; also has the u32 input `sizes` (promoted; the alias reads its values).
struct BuildSBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
        cs::Input<DType::u32, 1> sizes;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = a.in[id] * static_cast<float>(q + 1);
    }
};
class BuildS : public cs::Component<BuildSBody> {
    cs::Input<DType::f32, 1>        in_;
    cs::Input<DType::u32, 1>        sizes_;
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override {
        declare_input(in_, "in", cs::FromSourceDim{0});
        declare_input(sizes_, "sizes", cs::FromSourceDim{0});
        declare_output(t_, "t", cs::tied);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{t_.size()}, {in_, sizes_}, {t_}, {});
    }
};

/// Factor: t = 2 t + 1 (reads its own output), a tied member.
struct FactorBody {
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = 2.0f * b[q] + 1.0f;
    }
};
class Factor : public cs::Component<FactorBody> {
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override { declare_output(t_, "t", cs::tied); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{t_.size()}, {}, {t_}, {});
    }
};

struct SumTBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 3> pts;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    void operator()(sycl::id<1> id, Inputs in, Outputs out) const {
        auto  b = in.pts.block(id[0]);
        float s = 0.0f;
        for (std::size_t q = 0; q < b.size(); ++q) s += b[q];
        out.sums[id] = s;
    }
};
class SumT : public cs::Component<SumTBody> {
    cs::RaggedInput<DType::f32, 3> pts_;
    cs::Output<DType::f32, 1>      sums_;

public:
    void setup() override {
        declare_input(pts_, "pts", cs::FromSourceDim{0}, cs::FromSourceDim{1}, cs::FromSourceDim{2});
        declare_output(sums_, "sums", cs::FromInputDim{"pts", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sums_.shape(), {pts_}, {sums_}, {});
    }
};

/// The tie system (as test_ragged_ties.cpp's GV, variant 0), a plain System inside the chunked one.
class TieG : public cs::System {
public:
    void setup() override {
        add_subsystem<BuildS>("p");
        promote<DType::u32, 1>("p.sizes", "sizes");
        add_subsystem<Factor>("q");
        add_subsystem<SumT>("s");
        tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t",
                                   cs::alias::dims(cs::alias::input_dim("sizes", 0), cs::alias::input_values("sizes"),
                                                   cs::alias::input_values("sizes")),
                                   cs::TieOutputsPolicy::allow_plain);
        connect("t", "s.pts");
    }
};
class RaggedInner : public cs::ChunkedSystem {
public:
    explicit RaggedInner(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override { add_subsystem<TieG>("g"); }
};
class RaggedTop : public cs::System {
    Counts      counts_;
    std::size_t budget_;

public:
    RaggedTop(Counts counts, std::size_t budget) : counts_{std::move(counts)}, budget_{budget} {}
    void setup() override {
        add_subsystem<CProd>("sz", counts_);
        add_subsystem<Src>("src", counts_.size());
        add_subsystem<RaggedInner>("inner", budget_);
        add_subsystem<Cons>("cons");
        connect("sz.counts", "inner.g.sizes");
        connect("src.x", "inner.g.p.in");
        connect("inner.g.s.sums", "cons.in");
    }
};

// ---- measuring a forward --------------------------------------------------------------------------------------------
struct Row {
    std::size_t           chunks = 0;
    cs::diag::EventCounts window{}, whole{};
};

Row measure(cs::Model& m) {
    const auto before = cs::diag::snapshot();
    m.forward();
    m.wait();
    Row r;
    r.whole  = cs::diag::snapshot() - before;
    r.window = cs::diag::last_window();
    r.chunks = dynamic_cast<cs::ChunkedSystem&>(m.get_subsystem("inner")).last_chunk_count();
    return r;
}

void print_header(const char* title, std::size_t budget) {
    std::printf("[diag] %s, budget %zu\n", title, budget);
    std::printf("[diag] %-34s %6s | %-8s %5s %5s %5s %5s %5s %5s %5s %5s %5s\n", "forward", "chunks", "scope", "wait", "free", "alloc",
                "hread", "hhit", "htask", "rupl", "fpass", "resol");
}

void print_row(const char* what, const Row& r) {
    const auto line = [&](const char* scope, const cs::diag::EventCounts& c, bool first) {
        std::printf("[diag] %-34s %6s | %-8s %5llu %5llu %5llu %5llu %5llu %5llu %5llu %5llu %5llu\n", first ? what : "",
                    first ? std::to_string(r.chunks).c_str() : "", scope, static_cast<unsigned long long>(c.queue_waits),
                    static_cast<unsigned long long>(c.frees), static_cast<unsigned long long>(c.allocations),
                    static_cast<unsigned long long>(c.host_reads), static_cast<unsigned long long>(c.host_read_hits),
                    static_cast<unsigned long long>(c.host_tasks),
                    static_cast<unsigned long long>(c.ragged_uploads), static_cast<unsigned long long>(c.full_passes),
                    static_cast<unsigned long long>(c.resolve_shapes));
    };
    line("window", r.window, true);
    line("whole", r.whole, false);
}

void check_wired(const Row& r) {
    CHECK(r.chunks >= 2);
    CHECK(r.whole.full_passes >= 1);
    CHECK(r.whole.resolve_shapes >= 1);
    CHECK(r.whole.queue_waits >= 1);
    CHECK(r.whole.allocations >= 1);
    CHECK(r.whole.frees >= 1);
}

}  // namespace tdiag

TEST_CASE("Diagnostics: the counters and the window API", "[diag]") {
    cs::diag::reset();
    const auto z = cs::diag::snapshot();
    CHECK(z.queue_waits == 0);
    CHECK(z.allocations == 0);
    cs::diag::count_queue_waits(3);
    cs::diag::count_full_passes();
    const auto a = cs::diag::snapshot();
    CHECK(a.queue_waits == 3);
    CHECK(a.full_passes == 1);
    cs::diag::window_begin();
    cs::diag::count_frees(2);
    cs::diag::window_end();
    cs::diag::count_frees(5);  // after the window: not in it
    const auto w = cs::diag::last_window();
    CHECK(w.frees == 2);
    CHECK(w.queue_waits == 0);
    CHECK((cs::diag::snapshot() - a).frees == 7);
}

TEST_CASE("Diagnostics baseline a: dense chunked, 10 chunks", "[diag]") {
    constexpr std::size_t n = 10, budget = 16;  // 16 bytes of internal t per index
    cs::Model             m = cs::create_model<tdiag::DenseTop>(default_config(), n, budget);
    m.setup();
    tdiag::print_header("a: dense, n = 10", budget);
    const auto r1 = tdiag::measure(m);
    const auto r2 = tdiag::measure(m);
    const auto r3 = tdiag::measure(m);
    tdiag::print_row("forward 1 (first)", r1);
    tdiag::print_row("forward 2 (inputs unchanged)", r2);
    tdiag::print_row("forward 3 (unchanged; no sizes input)", r3);
    tdiag::check_wired(r1);
    tdiag::check_wired(r2);
    tdiag::check_wired(r3);
    CHECK(r1.chunks == 10);
}

TEST_CASE("Diagnostics baseline b: LRPIM pattern (ragged tie alias from input values), chunked", "[diag]") {
    constexpr std::size_t n = 24, budget = 150;  // t costs 4 c^2 bytes per index, c in 1..6
    tdiag::Counts         first(n), second(n);
    for (std::size_t i = 0; i < n; ++i) {
        first[i]  = static_cast<std::uint32_t>(1 + (i * 5) % 6);
        second[i] = static_cast<std::uint32_t>(1 + (i * 7 + 2) % 6);
    }
    cs::Model m = cs::create_model<tdiag::RaggedTop>(default_config(), first, budget);
    m.setup();
    tdiag::print_header("b: ragged tie alias from input_values(sizes), n = 24", budget);
    const auto r1 = tdiag::measure(m);
    const auto r2 = tdiag::measure(m);
    m.init_input("sz.raw", cs::init::from_host(second));
    const auto r3 = tdiag::measure(m);
    tdiag::print_row("forward 1 (first)", r1);
    tdiag::print_row("forward 2 (inputs unchanged)", r2);
    tdiag::print_row("forward 3 (sizes re-initialised)", r3);
    tdiag::check_wired(r1);
    tdiag::check_wired(r2);
    tdiag::check_wired(r3);
}
