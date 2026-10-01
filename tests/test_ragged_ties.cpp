// Ragged tie aliases (D140): a tie alias takes the same per-index shape rules as a port.
//
//   alias::dims(spec0, spec1, ...) with an integer, per_index, output_value, output_values, output_dim;
//   from_output_shape and from_connection copy a ragged shape.
//
// Members are RaggedOutput / RaggedInput declared with cs::tied. The ragged wrappers are non-atomic, so the
// tie_outputs here use TieOutputsPolicy::allow_plain (the producers run in order, they do not race).
//
// Everything is checked against host references.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace trt {

using Sizes  = std::vector<std::size_t>;
using Counts = std::vector<std::uint32_t>;

template <class F>
void expect_throws(F&& f, const std::vector<std::string>& needles) {
    std::string what = "<no exception>";
    try {
        f();
    } catch (const std::exception& e) {
        what = e.what();
    }
    INFO("thrown: " << what);
    for (const auto& n : needles) CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}

// ---- CProd: counts = a free u32 input (init from the host), copied ----------------------------------------
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

// ---- Gen: pts [N, c_i, 3], p[i][q] = i * 1000 + q ---------------------------------------------------------------
struct GenBody {
    struct Inputs {
        cs::Input<DType::u32, 1> counts;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> pts;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        auto b = o.pts.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = static_cast<float>(id[0] * 1000 + q);
    }
};
class Gen : public cs::Component<GenBody> {
    cs::Input<DType::u32, 1>        counts_;
    cs::RaggedOutput<DType::f32, 3> pts_;

public:
    void setup() override {
        declare_input(counts_, "counts", cs::FromSourceDim{0});
        declare_output(pts_, "pts", cs::FromInputDim{"counts", 0}, cs::FromInputValues{"counts"}, 3);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {counts_}, {pts_}, {});
    }
};

// ---- tie members: a rank-3 ragged output declared cs::tied -------------------------------------------------------------
/// Op: 0 zero, 1 accumulate (t += (i+1)(q+1)), 2 build (t = (i+1)(q+1)), 3 factor (t = 2 t + 1, reads its own output).
template <int Op>
struct TieBody {
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        auto        b    = o.t.block(id[0]);
        const float base = static_cast<float>(id[0] + 1);
        for (std::size_t q = 0; q < b.size(); ++q) {
            if constexpr (Op == 0) b[q] = 0.0f;
            if constexpr (Op == 1) b[q] += base * static_cast<float>(q + 1);
            if constexpr (Op == 2) b[q] = base * static_cast<float>(q + 1);
            if constexpr (Op == 3) b[q] = 2.0f * b[q] + 1.0f;
        }
    }
};
template <int Op>
class TieMember : public cs::Component<TieBody<Op>> {
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override { this->declare_output(t_, "t", cs::tied); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(sycl::range<1>{t_.size()}, {}, {t_}, {});
    }
};
using Zero   = TieMember<0>;
using Acc    = TieMember<1>;
using Build0 = TieMember<2>;
using Factor = TieMember<3>;

// ---- SumT: sums[i] = the sum of block i of a ragged [N, a, b] input ---------------------------------------------------------
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

// ---- tie_outputs top: sz -> gen, then the tie (zero/acc or build/factor), then a sum ---------------------------------
/// shape_mode 0: alias::dims(output_dim(sz.counts, 0), output_values(sz.counts), output_values(sz.counts)) -> [N, m, m]
/// shape_mode 1: alias::from_output_shape(gen.pts) -> [N, m, 3]
template <class P, class Q>
class TieOutTop : public cs::System {
    Counts init_;
    int    shape_mode_;

public:
    TieOutTop(Counts init, int shape_mode) : init_{std::move(init)}, shape_mode_{shape_mode} {}
    void setup() override {
        add_subsystem<CProd>("sz", init_);
        add_subsystem<Gen>("gen");
        add_subsystem<P>("p");
        add_subsystem<Q>("q");
        add_subsystem<SumT>("sum");
        connect("sz.counts", "gen.counts");
        if (shape_mode_ == 0)
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t",
                                       cs::alias::dims(cs::alias::output_dim("sz.counts", 0), cs::alias::output_values("sz.counts"),
                                                       cs::alias::output_values("sz.counts")),
                                       cs::TieOutputsPolicy::allow_plain);
        else
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::from_output_shape("gen.pts"),
                                       cs::TieOutputsPolicy::allow_plain);
        connect("t", "sum.pts");
    }
};

/// The reference of the tie: block i has counts[i] rows and (shape_mode 0: counts[i], 1: 3) columns.
struct TieRef {
    std::vector<float>         data, sums;
    std::vector<std::uint64_t> offsets{0};
};
TieRef tie_ref(const Counts& counts, int shape_mode, bool factor) {
    TieRef r;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const std::size_t size = counts[i] * (shape_mode == 0 ? counts[i] : 3u);
        float             s    = 0.0f;
        for (std::size_t q = 0; q < size; ++q) {
            const float base = static_cast<float>((i + 1) * (q + 1));
            const float v    = factor ? 2.0f * base + 1.0f : base;
            r.data.push_back(v);
            s += v;
        }
        r.sums.push_back(s);
        r.offsets.push_back(r.data.size());
    }
    return r;
}

void run_tie_out(cs::Model& m, const Counts& counts, int shape_mode, bool factor) {
    m.init_input("sz.raw", cs::init::from_host(counts));
    m.forward();
    m.wait();
    const TieRef ref  = tie_ref(counts, shape_mode, factor);
    auto         sums = m.view_output<DType::f32, 1>("sum.sums");
    REQUIRE(sums.size() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) CHECK(sums.span()[i] == ref.sums[i]);
    auto pts = m.view_ragged_input<DType::f32, 3>("sum.pts");  // the alias's data, through a peer
    CHECK(pts.offsets == ref.offsets);
    CHECK(pts.data == ref.data);
    CHECK(pts.size() == counts.size());
    CHECK(pts.is_per_index(1));
    if (shape_mode == 0)
        CHECK(pts.is_per_index(2));
    else
        CHECK_FALSE(pts.is_per_index(2));
}

// ---- tie_inputs top: gen -> (two ragged inputs tied, from_connection) -----------------------------------------------
template <int K>
struct SumKBody {
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
        out.sums[id] = static_cast<float>(K) * s;
    }
};
template <int K>
class SumK : public cs::Component<SumKBody<K>> {
    cs::RaggedInput<DType::f32, 3> pts_;
    cs::Output<DType::f32, 1>      sums_;

public:
    void setup() override {
        this->declare_input(pts_, "pts", cs::tied);
        this->declare_output(sums_, "sums", cs::FromInputDim{"pts", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(sums_.shape(), {pts_}, {sums_}, {});
    }
};

class TieInTop : public cs::System {
    Counts init_;

public:
    explicit TieInTop(Counts init) : init_{std::move(init)} {}
    void setup() override {
        add_subsystem<CProd>("sz", init_);
        add_subsystem<Gen>("gen");
        add_subsystem<SumK<1>>("sa");
        add_subsystem<SumK<2>>("sb");
        connect("sz.counts", "gen.counts");
        tie_inputs<DType::f32, 3>({"sa.pts", "sb.pts"}, "pts", cs::alias::from_connection());
        connect("gen.pts", "pts");
    }
};

// ---- a plain tie inside (and outside) a ChunkedSystem ----------------------------------------------------------------------
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
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {}, {x_}, {});
    }
};

/// BuildX: t = in[i] (q + 1), a tied output; reads the outside input `in`.
struct BuildXBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = a.in[id] * static_cast<float>(q + 1);
    }
};
class BuildX : public cs::Component<BuildXBody> {
    cs::Input<DType::f32, 1>        in_;
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override {
        declare_input(in_, "in", cs::FromSourceDim{0});
        declare_output(t_, "t", cs::tied);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{t_.size()}, {in_}, {t_}, {});
    }
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
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {in_}, {out_}, {});
    }
};

/// g: build (t = in (q+1)) -> factor (t = 2 t + 1, in place) -> sum; the tie's alias is [N, per_index(m), per_index(m)].
class G : public cs::System {
    std::size_t n_;
    Sizes       sizes_;

public:
    G(std::size_t n, Sizes sizes) : n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem<BuildX>("p");
        add_subsystem<Factor>("q");
        add_subsystem<SumT>("s");
        tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(n_, cs::per_index(sizes_), cs::per_index(sizes_)),
                                   cs::TieOutputsPolicy::allow_plain);
        connect("t", "s.pts");
    }
};
class InnerPlain : public cs::System {
    std::size_t n_;
    Sizes       sizes_;

public:
    InnerPlain(std::size_t, std::size_t n, Sizes sizes) : n_{n}, sizes_{std::move(sizes)} {}
    void setup() override { add_subsystem<G>("g", n_, sizes_); }
};
class InnerChunk : public cs::ChunkedSystem {
    std::size_t n_;
    Sizes       sizes_;

public:
    InnerChunk(std::size_t budget, std::size_t n, Sizes sizes) : cs::ChunkedSystem(budget), n_{n}, sizes_{std::move(sizes)} {}
    void setup() override { add_subsystem<G>("g", n_, sizes_); }
};
template <class Inner>
class ChunkTop : public cs::System {
    std::size_t n_, budget_;
    Sizes       sizes_;

public:
    ChunkTop(std::size_t n, std::size_t budget, Sizes sizes) : n_{n}, budget_{budget}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem<Src>("src", n_);
        add_subsystem<Inner>("inner", budget_, n_, sizes_);
        add_subsystem<Cons>("cons");
        connect("src.x", "inner.g.p.in");
        connect("inner.g.s.sums", "cons.in");
    }
};

std::vector<float> forward_out(cs::Model& m) {
    m.forward();
    m.wait();
    auto v = m.view_output<DType::f32, 1>("cons.out");
    return std::vector<float>(v.span().begin(), v.span().end());
}

std::vector<float> chunk_ref(const Sizes& sizes) {
    std::vector<float> out;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const float x = 0.5f * static_cast<float>(i) + 1.0f;
        float       s = 0.0f;
        for (std::size_t q = 0; q < sizes[i] * sizes[i]; ++q) s += 2.0f * (x * static_cast<float>(q + 1)) + 1.0f;
        out.push_back(2.0f * s + 1.0f);
    }
    return out;
}

// ---- errors ----------------------------------------------------------------------------------------------------------------
class ErrTop : public cs::System {
    int variant_;

public:
    explicit ErrTop(int variant) : variant_{variant} {}
    void setup() override {
        add_subsystem<CProd>("sz", Counts{1, 2, 3, 4});
        add_subsystem<Zero>("p");
        add_subsystem<Acc>("q");
        const auto policy = cs::TieOutputsPolicy::allow_plain;
        if (variant_ == 0)  // a per-index spec on dim 0
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(cs::per_index(Sizes{1, 2}), 2, 2), policy);
        else if (variant_ == 1)  // a fixed dim 0 of 4 and a per-index array of 2
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(4, cs::per_index(Sizes{1, 2}), 2), policy);
        else  // dim 0 is 4 (from sz.counts) at run time; the per-index array has 2
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t",
                                       cs::alias::dims(cs::alias::output_dim("sz.counts", 0), cs::per_index(Sizes{1, 2}), 2), policy);
    }
};

struct SrcUBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::u32, 1> x;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const { o.x[i] = static_cast<std::uint32_t>(i[0] % 3 + 1); }
};
class SrcU : public cs::Component<SrcUBody> {
    cs::Output<DType::u32, 1> x_;

public:
    void setup() override { declare_output(x_, "x", 6); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {}, {x_}, {});
    }
};
struct CopyUBody {
    struct Inputs {
        cs::Input<DType::u32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> counts;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.counts[i] = a.in[i]; }
};
class CopyU : public cs::Component<CopyUBody> {
    cs::Input<DType::u32, 1>  in_;
    cs::Output<DType::u32, 1> counts_;

public:
    void setup() override {
        declare_input(in_, "in", cs::FromSourceDim{0});
        declare_output(counts_, "counts", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {in_}, {counts_}, {});
    }
};
/// The tie's per-index source (cc.counts) is computed INSIDE the chunked system.
class GBad : public cs::System {
public:
    void setup() override {
        add_subsystem<CopyU>("cc");
        add_subsystem<Zero>("p");
        add_subsystem<Acc>("q");
        tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(6, cs::alias::output_values("cc.counts"), 2),
                                   cs::TieOutputsPolicy::allow_plain);
    }
};
class BadInner : public cs::ChunkedSystem {
public:
    BadInner() : cs::ChunkedSystem(1000) {}
    void setup() override { add_subsystem<GBad>("g"); }
};
class BadTop : public cs::System {
public:
    void setup() override {
        add_subsystem<SrcU>("src");
        add_subsystem<BadInner>("inner");
        connect("src.x", "inner.g.cc.in");
    }
};

}  // namespace trt

TEST_CASE("Ragged ties: tie_outputs of two ragged members (zero, then accumulate), sizes from an upstream output",
          "[ragged][tie]") {
    for (const int shape_mode : {0, 1}) {
        INFO("shape_mode " << shape_mode);
        cs::Model m = cs::create_model<trt::TieOutTop<trt::Zero, trt::Acc>>(default_config(), trt::Counts{3, 0, 2, 4}, shape_mode);
        m.setup();
        trt::run_tie_out(m, {3, 0, 2, 4}, shape_mode, false);  // a zero in the middle
        trt::run_tie_out(m, {0, 5, 1, 2}, shape_mode, false);  // an empty first block, growth
        trt::run_tie_out(m, {0, 0, 0, 0}, shape_mode, false);  // all empty
        trt::run_tie_out(m, {2, 2, 2, 1}, shape_mode, false);  // and back
    }
}

TEST_CASE("Ragged ties: a plain tie whose second member transforms the first one's output in place", "[ragged][tie]") {
    for (const int shape_mode : {0, 1}) {
        INFO("shape_mode " << shape_mode);
        cs::Model m = cs::create_model<trt::TieOutTop<trt::Build0, trt::Factor>>(default_config(), trt::Counts{1, 3, 0, 2}, shape_mode);
        m.setup();
        trt::run_tie_out(m, {1, 3, 0, 2}, shape_mode, true);
        trt::run_tie_out(m, {4, 0, 2, 2}, shape_mode, true);
    }
}

TEST_CASE("Ragged ties: tie_inputs of two RaggedInputs, from_connection to a ragged output", "[ragged][tie]") {
    cs::Model m = cs::create_model<trt::TieInTop>(default_config(), trt::Counts{2, 0, 3, 1});
    m.setup();
    for (const trt::Counts& counts : {trt::Counts{2, 0, 3, 1}, trt::Counts{0, 4, 0, 2}, trt::Counts{1, 1, 1, 1}}) {
        m.init_input("sz.raw", cs::init::from_host(counts));
        m.forward();
        m.wait();
        auto sa = m.view_output<DType::f32, 1>("sa.sums");
        auto sb = m.view_output<DType::f32, 1>("sb.sums");
        REQUIRE(sa.size() == counts.size());
        REQUIRE(sb.size() == counts.size());
        for (std::size_t i = 0; i < counts.size(); ++i) {
            float want = 0.0f;
            for (std::size_t q = 0; q < counts[i] * 3u; ++q) want += static_cast<float>(i * 1000 + q);
            CHECK(sa.span()[i] == want);
            CHECK(sb.span()[i] == 2.0f * want);
        }
        auto pa = m.view_ragged_input<DType::f32, 3>("sa.pts");
        CHECK(pa.extents(1) == trt::Sizes(counts.begin(), counts.end()));
        CHECK(pa.is_per_index(1));
    }
}

TEST_CASE("Ragged ties: a tie inside a ChunkedSystem, bitwise equal to unchunked, one allocation per forward",
          "[ragged][tie][chunked]") {
    const trt::Sizes sizes{1, 3, 2, 3, 0, 3, 1, 2};
    const std::size_t n = sizes.size();
    const auto        want = trt::chunk_ref(sizes);
    cs::Model plain = cs::create_model<trt::ChunkTop<trt::InnerPlain>>(default_config(), n, std::size_t{0}, sizes);
    plain.setup();
    const auto pout = trt::forward_out(plain);
    REQUIRE(pout.size() == n);
    for (std::size_t i = 0; i < n; ++i) CHECK(pout[i] == want[i]);
    for (const std::size_t budget : {std::size_t{36}, std::size_t{50}, std::size_t{100}, std::size_t{100000}}) {
        INFO("budget " << budget);
        cs::Model m = cs::create_model<trt::ChunkTop<trt::InnerChunk>>(default_config(), n, budget, sizes);
        m.setup();
        cs::detail::IOEntry& t = m.root().entry_at("inner.g.t");  // the internal ragged tie alias root
        for (int rep = 0; rep < 2; ++rep) {
            const std::size_t before = t.allocation_count();
            const auto        out    = trt::forward_out(m);
            CHECK(out == pout);
            CHECK(t.allocation_count() - before == 1);
            CHECK_FALSE(t.held());
        }
    }
}

TEST_CASE("Ragged ties: errors", "[ragged][tie]") {
    SECTION("a per-index spec on dimension 0") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 0);
        trt::expect_throws([&] { m.setup(); }, {"alias::dims, dimension 0", "per-index spec is for dimensions >= 1"});
    }
    SECTION("a per-index array that does not match a fixed dimension 0") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 1);
        trt::expect_throws([&] { m.setup(); }, {"per_index has 2 extents", "dimension 0 is 4"});
    }
    SECTION("a per-index array that does not match dimension 0 at run time") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 2);
        m.setup();
        trt::expect_throws([&] { m.forward(); }, {"has 2 extents", "dimension 0 has 4"});
        m.wait();  // earlier components of the failed forward were submitted: wait before the Model is destroyed (D135)
    }
    SECTION("a per-index source computed inside a chunked system") {
        cs::Model m = cs::create_model<trt::BadTop>(default_config());
        m.setup();
        trt::expect_throws([&] { m.forward(); }, {"is sized by a value computed inside chunked system", "inner"});
        m.wait();
    }
}
