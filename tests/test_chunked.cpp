// ChunkedSystem. Every case builds the same subsystem twice, once as a plain System and once as
// a ChunkedSystem, and compares the outputs bitwise; host references check both.
//
//   dense:   Prod [N,3] (outside) -> chunked{ A: [N,3] -> internal t [N,4]; B: t -> y [N] } -> Cons (outside)
//   ragged:  CProd counts -> CoordGen pts [N,c_i,3] (outside) -> chunked{ S: pts -> internal ragged gram
//            [N,c_i,c_i]; R: gram -> out [N] } -> Cons (outside)

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "chunk_capture.hpp"
#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace tch {

using Sizes = std::vector<std::size_t>;

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

// ---- dense components ----------------------------------------------------------------------------------
/// Outside producer: x[i,j] = (7 i + 3 j) % 11 + 0.5, fixed [n, 3].
struct ProdBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> x;
    };
    void operator()(sycl::id<2> i, Inputs, Outputs o) const {
        o.x[i] = static_cast<float>((i[0] * 7 + i[1] * 3) % 11) + 0.5f;
    }
};
class Prod : public cs::Component<ProdBody> {
    std::size_t               n_;
    cs::Output<DType::f32, 2> x_;

public:
    explicit Prod(std::size_t n) : n_{n} {}
    void setup() override { declare_output(x_, "x", n_, 3); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {}, {x_}, {});
    }
};

/// A: t[i,k] = (in[i,0] + 2 in[i,1] + 3 in[i,2]) * (k + 1), t is [N, 4]. `in` takes its shape from its
/// connection, or is a fixed [n, 3] free input when n_fixed > 0.
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
    std::size_t               n_fixed_;
    cs::Input<DType::f32, 2>  in_;
    cs::Output<DType::f32, 2> t_;

public:
    explicit A(std::size_t n_fixed = 0) : n_fixed_{n_fixed} {}
    void setup() override {
        if (n_fixed_ > 0)
            declare_input(in_, "in", n_fixed_, 3);
        else
            declare_input(in_, "in", cs::unspecified, cs::unspecified);
        declare_output(t_, "t", cs::FromInputDim{"in", 0}, 4);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(t_.shape(), {in_}, {t_}, {});
    }
};

/// B: y[i] = sum_k in[i,k].
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
        declare_input(in_, "in", cs::unspecified, cs::unspecified);
        declare_output(y_, "y", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {in_}, {y_}, {});
    }
};

/// Cons (outside): out[i] = 2 in[i] + 1.
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
        declare_input(in_, "in", cs::unspecified);
        declare_output(out_, "out", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {in_}, {out_}, {});
    }
};

/// The same subsystem as a plain System and as a ChunkedSystem.
void fill_dense(cs::System& s, bool promote_t) {
    s.add_subsystem<A>("a");
    s.add_subsystem<B>("b");
    s.connect("a.t", "b.in");
    s.promote<DType::f32, 2>("a.in", "in");  // the interface of an opaque chunked system: its promotions
    s.promote<DType::f32, 1>("b.y", "y");
    if (promote_t) s.promote<DType::f32, 2>("a.t", "t");  // boundary AND used inside by b
}
class InnerPlain : public cs::System {
    bool promote_t_;

public:
    InnerPlain(std::size_t /*budget*/, bool promote_t) : promote_t_{promote_t} {}
    void setup() override { fill_dense(*this, promote_t_); }
};
class InnerChunk : public cs::ChunkedSystem {
    bool promote_t_;

public:
    InnerChunk(std::size_t budget, bool promote_t) : cs::ChunkedSystem(budget), promote_t_{promote_t} {}
    void setup() override { fill_dense(*this, promote_t_); }
};

template <class Inner>
class DenseTop : public cs::System {
    std::size_t n_, budget_;
    bool        promote_t_;

public:
    DenseTop(std::size_t n, std::size_t budget, bool promote_t = false) : n_{n}, budget_{budget}, promote_t_{promote_t} {}
    void setup() override {
        add_subsystem<Prod>("prod", n_);
        add_subsystem<Inner>("inner", budget_, promote_t_);
        add_subsystem<Cons>("cons");
        connect("prod.x", "inner.in");
        connect("inner.y", "cons.in");
    }
};

/// A free input of an inside component promoted out ("w"), its output promoted out ("y") and consumed outside.
template <class Inner>
class PromTop : public cs::System {
    std::size_t n_, budget_;

public:
    PromTop(std::size_t n, std::size_t budget) : n_{n}, budget_{budget} {}
    void setup() override {
        add_subsystem<Inner>("inner", n_, budget_);
        add_subsystem<Cons>("cons");
        connect("inner.y", "cons.in");
    }
    std::size_t n() const { return n_; }
};
class PromInnerPlain : public cs::System {
    std::size_t n_;

public:
    PromInnerPlain(std::size_t n, std::size_t /*budget*/) : n_{n} {}
    void setup() override {
        add_subsystem<A>("a", n_);
        add_subsystem<B>("b");
        connect("a.t", "b.in");
        promote<DType::f32, 2>("a.in", "w");
        promote<DType::f32, 1>("b.y", "y");
    }
};
class PromInnerChunk : public cs::ChunkedSystem {
    std::size_t n_;

public:
    PromInnerChunk(std::size_t n, std::size_t budget) : cs::ChunkedSystem(budget), n_{n} {}
    void setup() override {
        add_subsystem<A>("a", n_);
        add_subsystem<B>("b");
        connect("a.t", "b.in");
        promote<DType::f32, 2>("a.in", "w");
        promote<DType::f32, 1>("b.y", "y");
    }
};

// ---- ragged components -----------------------------------------------------------------------------------
/// CProd: counts = a free u32 input (init from the host), copied.
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
    std::vector<std::uint32_t> init_;
    cs::Input<DType::u32, 1>   raw_;
    cs::Output<DType::u32, 1>  counts_;

public:
    explicit CProd(std::vector<std::uint32_t> init) : init_{std::move(init)} {}
    void setup() override {
        declare_input(raw_, "raw", init_.size());
        init_input("raw", cs::init::from_host(init_));
        declare_output(counts_, "counts", init_.size());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {raw_}, {counts_}, {});
    }
};

/// CoordGen: pts [N, c_i, 3], p[j,k] = 5 (i % 3) + 2 j + k + 1.
struct CoordBody {
    struct Inputs {
        cs::Input<DType::u32, 1> counts;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> pts;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        const std::size_t i = id[0];
        auto              b = o.pts.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j)
            for (std::size_t k = 0; k < b.shape()[1]; ++k) b[sycl::id<2>{j, k}] = static_cast<float>(5 * (i % 3) + 2 * j + k + 1);
    }
};
class CoordGen : public cs::Component<CoordBody> {
    cs::Input<DType::u32, 1>        counts_;
    cs::RaggedOutput<DType::f32, 3> pts_;

public:
    void setup() override {
        declare_input(counts_, "counts", cs::unspecified);
        declare_output(pts_, "pts", cs::FromInputDim{"counts", 0}, cs::FromInputValues{"counts"}, 3);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {counts_}, {pts_}, {});
    }
};

/// S: gram.block(i)[j,m] = sum_k p[j,k] p[m,k], an internal ragged [N, c_i, c_i].
struct SBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 3> pts;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> gram;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto p = a.pts.block(id[0]);
        auto g = o.gram.block(id[0]);
        for (std::size_t j = 0; j < g.shape()[0]; ++j)
            for (std::size_t m = 0; m < g.shape()[1]; ++m) {
                float s = 0.0f;
                for (std::size_t k = 0; k < p.shape()[1]; ++k) s += p[sycl::id<2>{j, k}] * p[sycl::id<2>{m, k}];
                g[sycl::id<2>{j, m}] = s;
            }
    }
};
class S : public cs::Component<SBody> {
    cs::RaggedInput<DType::f32, 3>  pts_;
    cs::RaggedOutput<DType::f32, 3> gram_;

public:
    void setup() override {
        declare_input(pts_, "pts", cs::unspecified, cs::unspecified, cs::unspecified);
        declare_output(gram_, "gram", cs::FromInputDim{"pts", 0}, cs::FromInputDim{"pts", 1}, cs::FromInputDim{"pts", 1});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{pts_.size()}, {pts_}, {gram_}, {});
    }
};

/// R: out[i] = the sum of block i of gram.
struct RBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 3> gram;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto  g = a.gram.block(id[0]);
        float s = 0.0f;
        for (std::size_t q = 0; q < g.size(); ++q) s += g[q];
        o.out[id] = s;
    }
};
class R : public cs::Component<RBody> {
    cs::RaggedInput<DType::f32, 3> gram_;
    cs::Output<DType::f32, 1>      out_;

public:
    void setup() override {
        declare_input(gram_, "gram", cs::unspecified, cs::unspecified, cs::unspecified);
        declare_output(out_, "out", cs::FromInputDim{"gram", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {gram_}, {out_}, {});
    }
};

void fill_ragged(cs::System& s) {
    s.add_subsystem<S>("s");
    s.add_subsystem<R>("r");
    s.connect("s.gram", "r.gram");
    s.promote<DType::f32, 3>("s.pts", "pts");  // the interface of an opaque chunked system
    s.promote<DType::f32, 1>("r.out", "out");
}
class RInnerPlain : public cs::System {
public:
    explicit RInnerPlain(std::size_t) {}
    void setup() override { fill_ragged(*this); }
};
class RInnerChunk : public cs::ChunkedSystem {
public:
    explicit RInnerChunk(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override { fill_ragged(*this); }
};
template <class Inner>
class RaggedTop : public cs::System {
    std::vector<std::uint32_t> counts_;
    std::size_t                budget_;

public:
    RaggedTop(std::vector<std::uint32_t> counts, std::size_t budget) : counts_{std::move(counts)}, budget_{budget} {}
    void setup() override {
        add_subsystem<CProd>("prod", counts_);
        add_subsystem<CoordGen>("gen");
        add_subsystem<Inner>("inner", budget_);
        add_subsystem<Cons>("cons");
        connect("prod.counts", "gen.counts");
        connect("gen.pts", "inner.pts");
        connect("inner.out", "cons.in");
    }
};

// ---- references -------------------------------------------------------------------------------------------
float xval(std::size_t i, std::size_t j) { return static_cast<float>((i * 7 + j * 3) % 11) + 0.5f; }
float tval(std::size_t i, std::size_t k) {
    return (xval(i, 0) + 2.0f * xval(i, 1) + 3.0f * xval(i, 2)) * static_cast<float>(k + 1);
}
float yval(std::size_t i) { return tval(i, 0) + tval(i, 1) + tval(i, 2) + tval(i, 3); }

/// What the greedy scan of the framework gives for per-index costs.
std::size_t expected_chunks(const Sizes& cost, std::size_t budget) {
    std::size_t chunks = 0, acc = 0;
    for (std::size_t i = 0; i < cost.size(); ++i) {
        if (acc + cost[i] > budget) {
            ++chunks;
            acc = 0;
        }
        acc += cost[i];
    }
    return cost.empty() ? 0 : chunks + 1;
}

std::vector<float> forward_out(cs::Model& m, const char* path = "cons.out") {
    m.forward();
    m.wait();
    auto v = m.view_output<DType::f32, 1>(path);
    return std::vector<float>(v.span().begin(), v.span().end());
}

/// An entry INSIDE the chunked system 'inner', resolved by the chunked system itself (its own subtree is visible to it;
/// from outside only its promotions are: the Model's root.entry_at("inner.a.t") is refused).
cs::detail::IOEntry& inner_entry(cs::Model& m, const char* path) {
    return dynamic_cast<cs::ChunkedSystem&>((*m.get_subsystem("inner"))).entry_at(path);
}

std::size_t chunks_of(cs::Model& m) { return dynamic_cast<cs::ChunkedSystem&>((*m.get_subsystem("inner"))).last_chunk_count(); }

}  // namespace tch

TEST_CASE("Chunked: dense, 1 / 2 / uneven / many chunks, bitwise equal to the plain System", "[chunked]") {
    constexpr std::size_t n = 10;
    std::vector<float>    want(n);
    for (std::size_t i = 0; i < n; ++i) want[i] = 2.0f * tch::yval(i) + 1.0f;

    cs::Model plain = cs::create_model<tch::DenseTop<tch::InnerPlain>>(default_config(), n, std::size_t{0});
    plain.setup();
    const auto plain_out = tch::forward_out(plain);
    CHECK(plain_out == want);

    // the cost of one index is 16 bytes (internal t [N,4] f32)
    struct Case {
        std::size_t budget, chunks;
    };
    for (const Case c : {Case{160, 1}, Case{1000, 1}, Case{80, 2}, Case{56, 4}, Case{16, 10}}) {
        INFO("budget " << c.budget);
        cs::Model m = cs::create_model<tch::DenseTop<tch::InnerChunk>>(default_config(), n, c.budget);
        m.setup();
        const auto out = tch::forward_out(m);
        CHECK(out == plain_out);
        CHECK(tch::chunks_of(m) == c.chunks);
        CHECK(tch::expected_chunks(tch::Sizes(n, 16), c.budget) == c.chunks);
        // views from the host see whole roots after the run
        CHECK(m.view_output<DType::f32, 1>("inner.y").size() == n);
        CHECK(out == want);
    }
}

TEST_CASE("Chunked: dense, two forwards", "[chunked]") {
    constexpr std::size_t n = 10;
    cs::Model m = cs::create_model<tch::DenseTop<tch::InnerChunk>>(default_config(), n, std::size_t{56});
    m.setup();
    const auto a = tch::forward_out(m);
    const auto b = tch::forward_out(m);
    CHECK(a == b);
    CHECK(tch::chunks_of(m) == 4);
}

TEST_CASE("Chunked: an inside output both promoted out and consumed by a later inside component", "[chunked]") {
    constexpr std::size_t n = 10;
    cs::Model plain = cs::create_model<tch::DenseTop<tch::InnerPlain>>(default_config(), n, std::size_t{0}, true);
    plain.setup();
    const auto pout = tch::forward_out(plain);
    auto       pt   = plain.view_output<DType::f32, 2>("inner.t");

    for (const std::size_t budget : {std::size_t{160}, std::size_t{48}, std::size_t{16}}) {
        INFO("budget " << budget);
        cs::Model m = cs::create_model<tch::DenseTop<tch::InnerChunk>>(default_config(), n, budget, true);
        m.setup();
        CHECK(tch::forward_out(m) == pout);
        auto t = m.view_output<DType::f32, 2>("inner.t");  // the promoted name: the whole [N,4]
        REQUIRE(t.size() == n * 4);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < 4; ++k) {
                CHECK(t.span()[i * 4 + k] == tch::tval(i, k));
                CHECK(t.span()[i * 4 + k] == pt.span()[i * 4 + k]);
            }
    }
}

TEST_CASE("Chunked: a free input of an inside component, promoted out and set with init_input", "[chunked]") {
    constexpr std::size_t n = 10;
    const auto            make_w = [&](float scale) {
        std::vector<float> w(n * 3);
        for (std::size_t q = 0; q < w.size(); ++q) w[q] = scale * static_cast<float>(q % 7) + 0.25f;
        return w;
    };
    const auto ref = [&](const std::vector<float>& w) {
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = 2.0f * ((w[i * 3] + 2.0f * w[i * 3 + 1] + 3.0f * w[i * 3 + 2]) * 10.0f) + 1.0f;
        return out;
    };
    cs::Model plain = cs::create_model<tch::PromTop<tch::PromInnerPlain>>(default_config(), n, std::size_t{0});
    plain.setup();
    cs::Model m = cs::create_model<tch::PromTop<tch::PromInnerChunk>>(default_config(), n, std::size_t{48});
    m.setup();
    for (const float scale : {1.0f, 3.0f}) {  // init between forwards re-applies
        const auto w = make_w(scale);
        plain.init_input("inner.w", cs::init::from_host(w));
        m.init_input("inner.w", cs::init::from_host(w));
        const auto pout = tch::forward_out(plain);
        const auto out  = tch::forward_out(m);
        CHECK(out == pout);
        CHECK(out == ref(w));
        CHECK(dynamic_cast<cs::ChunkedSystem&>((*m.get_subsystem("inner"))).last_chunk_count() == 4);
        auto wv = m.view_input<DType::f32, 2>("inner.w");  // the promoted free input holds the whole value
        REQUIRE(wv.size() == n * 3);
        for (std::size_t q = 0; q < w.size(); ++q) CHECK(wv.span()[q] == w[q]);
    }
}

TEST_CASE("Chunked: ragged scratch, counts with zeros, several budgets, two forwards", "[chunked]") {
    const std::vector<std::vector<std::uint32_t>> counts_sets{{3, 0, 2, 4, 0, 1, 2, 3}, {0, 4, 4, 1, 0, 0, 2, 1}};
    for (const auto& counts : counts_sets) {
        const std::size_t n = counts.size();
        // the reference: out[i] = sum_k (sum_j p[j,k])^2
        std::vector<float> ref(n);
        tch::Sizes         cost(n);
        for (std::size_t i = 0; i < n; ++i) {
            float tot = 0.0f;
            for (std::size_t k = 0; k < 3; ++k) {
                float cs_ = 0.0f;
                for (std::size_t j = 0; j < counts[i]; ++j) cs_ += static_cast<float>(5 * (i % 3) + 2 * j + k + 1);
                tot += cs_ * cs_;
            }
            ref[i]  = 2.0f * tot + 1.0f;
            cost[i] = 4u * counts[i] * counts[i];
        }
        cs::Model plain = cs::create_model<tch::RaggedTop<tch::RInnerPlain>>(default_config(), counts, std::size_t{0});
        plain.setup();
        const auto pout = tch::forward_out(plain);
        CHECK(pout == ref);
        for (const std::size_t budget : {std::size_t{64}, std::size_t{100}, std::size_t{128}, std::size_t{1000}}) {
            INFO("budget " << budget);
            cs::Model m = cs::create_model<tch::RaggedTop<tch::RInnerChunk>>(default_config(), counts, budget);
            m.setup();
            CHECK(tch::forward_out(m) == pout);
            CHECK(tch::chunks_of(m) == tch::expected_chunks(cost, budget));
        }
    }
    // two forwards with different counts in ONE model: the chunk count changes
    const auto&   c1 = counts_sets[0];
    const auto&   c2 = counts_sets[1];
    cs::Model     m  = cs::create_model<tch::RaggedTop<tch::RInnerChunk>>(default_config(), c1, std::size_t{100});
    m.setup();
    cs::Model     p  = cs::create_model<tch::RaggedTop<tch::RInnerPlain>>(default_config(), c1, std::size_t{0});
    p.setup();
    const auto    o1 = tch::forward_out(m);
    CHECK(o1 == tch::forward_out(p));
    const std::size_t k1 = tch::chunks_of(m);
    m.init_input("prod.raw", cs::init::from_host(c2));
    p.init_input("prod.raw", cs::init::from_host(c2));
    const auto o2 = tch::forward_out(m);
    CHECK(o2 == tch::forward_out(p));
    CHECK(o1 != o2);
    tch::Sizes cost2(c2.size());
    for (std::size_t i = 0; i < c2.size(); ++i) cost2[i] = 4u * c2[i] * c2[i];
    CHECK(tch::chunks_of(m) == tch::expected_chunks(cost2, 100));
    CHECK(k1 != tch::chunks_of(m));
    // N = 0 is covered by the leading-range tests; an all-zero count set is all-empty blocks
    m.init_input("prod.raw", cs::init::from_host(std::vector<std::uint32_t>(c1.size(), 0)));
    p.init_input("prod.raw", cs::init::from_host(std::vector<std::uint32_t>(c1.size(), 0)));
    CHECK(tch::forward_out(m) == tch::forward_out(p));
    CHECK(tch::chunks_of(m) == 1);
}

TEST_CASE("Chunked: memory after the forward is the eternal roots only", "[chunked][memory]") {
    constexpr std::size_t n = 256;  // t is 16 B per row: 4 KiB internal
    cs::Model plain = cs::create_model<tch::DenseTop<tch::InnerPlain>>(cst::pooled_config(), n, std::size_t{0});
    plain.setup();
    const std::size_t base = cst::pool_in_use(plain);
    tch::forward_out(plain);
    const std::size_t plain_after = cst::pool_in_use(plain);

    cs::Model m = cs::create_model<tch::DenseTop<tch::InnerChunk>>(cst::pooled_config(), n, std::size_t{16 * 32});
    m.setup();
    tch::forward_out(m);
    const std::size_t after1 = cst::pool_in_use(m);
    tch::forward_out(m);
    const std::size_t after2 = cst::pool_in_use(m);
    CHECK(tch::chunks_of(m) == 8);
    CHECK(after1 == after2);          // nothing accumulates across forwards
    CHECK(after1 < plain_after);      // the internal root is only a chunk big (eternal, kept across forwards)
    cs::detail::IOEntry& t = tch::inner_entry(m, "a.t");
    CHECK(t.has_storage());
    CHECK(t.capacity_elems() * sizeof(float) <= 16 * 32);  // its capacity is the largest chunk, never the whole run
    CHECK(after1 >= base);
}

// ---- errors ----------------------------------------------------------------------------------------------------------
namespace tch {

struct GenFixedBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const { o.out[i] = 1.0f; }
};
class GenFixed : public cs::Component<GenFixedBody> {
    std::size_t               n_;
    cs::Output<DType::f32, 1> out_;

public:
    explicit GenFixed(std::size_t n) : n_{n} {}
    void setup() override { declare_output(out_, "out", n_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {}, {out_}, {});
    }
};

/// An internal root declared with a fixed leading length (13, unrelated to N): inside a chunked system the
/// framework sets dim 0, so it works.
class OddInner : public cs::ChunkedSystem {
public:
    explicit OddInner(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override {
        add_subsystem<A>("a");
        add_subsystem<B>("b");
        add_subsystem<GenFixed>("g", std::size_t{13});
        connect("a.t", "b.in");
        promote<DType::f32, 2>("a.in", "in");
        promote<DType::f32, 1>("b.y", "y");
    }
};
class OddTop : public cs::System {
public:
    void setup() override {
        add_subsystem<Prod>("prod", std::size_t{8});
        add_subsystem<OddInner>("inner", std::size_t{1000});
        add_subsystem<Cons>("cons");
        connect("prod.x", "inner.in");
        connect("inner.y", "cons.in");
    }
};

/// Two boundary roots of different N.
class TwoNInner : public cs::ChunkedSystem {
public:
    explicit TwoNInner(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override {
        add_subsystem<A>("a");
        add_subsystem<A>("a2");
        promote<DType::f32, 2>("a.in", "in1");
        promote<DType::f32, 2>("a2.in", "in2");
    }
};
class TwoNTop : public cs::System {
public:
    void setup() override {
        add_subsystem<Prod>("p1", std::size_t{8});
        add_subsystem<Prod>("p2", std::size_t{6});
        add_subsystem<TwoNInner>("inner", std::size_t{1000});
        connect("p1.x", "inner.in1");
        connect("p2.x", "inner.in2");
    }
};

/// A promoted FREE input declared with 6 rows next to a promoted connected input fed by an 8-row source.
class WrongLenInner : public cs::ChunkedSystem {
public:
    explicit WrongLenInner(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override {
        add_subsystem<A>("a", std::size_t{6});
        add_subsystem<A>("a2");
        promote<DType::f32, 2>("a.in", "w");
        promote<DType::f32, 2>("a2.in", "in2");
    }
};
class WrongLenTop : public cs::System {
public:
    void setup() override {
        add_subsystem<Prod>("p", std::size_t{8});
        add_subsystem<WrongLenInner>("inner", std::size_t{1000});
        connect("p.x", "inner.in2");
    }
};

/// An outside producer with a derived dimension 1 (the value of its free input k), registered AFTER the chunked system
/// it feeds: its root is not resolved when the chunked system runs.
struct ProdKBody {
    struct Inputs {
        cs::Input<DType::u32, 1> k;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> x;
    };
    void operator()(sycl::id<2> i, Inputs, Outputs o) const { o.x[i] = 1.0f; }
};
class ProdK : public cs::Component<ProdKBody> {
    cs::Input<DType::u32, 1>  k_;
    cs::Output<DType::f32, 2> x_;

public:
    void setup() override {
        declare_input(k_, "k", 1);
        init_input("k", cs::init::from_host(std::vector<std::uint32_t>{3}));
        declare_output(x_, "x", std::size_t{10}, cs::FromInput{"k", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(x_.shape(), {k_}, {x_}, {}); }
};
class UnresolvedTop : public cs::System {
public:
    void setup() override {
        add_subsystem<InnerChunk>("inner", std::size_t{1000}, false);
        add_subsystem<ProdK>("late");
        connect("late.x", "inner.in");
    }
};

/// A shape driven by a value computed inside: counts are copied inside, then size the ragged output.
struct CopyU32Body {
    struct Inputs {
        cs::Input<DType::u32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> cnt;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.cnt[i] = a.in[i]; }
};
class CopyU32 : public cs::Component<CopyU32Body> {
    cs::Input<DType::u32, 1>  in_;
    cs::Output<DType::u32, 1> cnt_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_output(cnt_, "cnt", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cnt_.shape(), {in_}, {cnt_}, {});
    }
};
class ValueInner : public cs::ChunkedSystem {
public:
    explicit ValueInner(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override {
        add_subsystem<CopyU32>("c");
        add_subsystem<CoordGen>("gen");
        connect("c.cnt", "gen.counts");
        promote<DType::u32, 1>("c.in", "in");
    }
};
class ValueTop : public cs::System {
public:
    void setup() override {
        add_subsystem<CProd>("prod", std::vector<std::uint32_t>{1, 2, 3});
        add_subsystem<ValueInner>("inner", std::size_t{1000});
        connect("prod.counts", "inner.in");
    }
};

/// A ChunkedSystem inside a ChunkedSystem.
class NestOuter : public cs::ChunkedSystem {
public:
    explicit NestOuter(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override { add_subsystem<InnerChunk>("sub", std::size_t{1000}, false); }
};
class NestTop : public cs::System {
public:
    void setup() override {
        add_subsystem<NestOuter>("outer", std::size_t{1000});  // nothing can be wired into sub: it is not promoted
    }
};

}  // namespace tch

TEST_CASE("Chunked: errors", "[chunked][errors]") {
    SECTION("a shape driven by a value computed inside") {
        cs::Model m = cs::create_model<tch::ValueTop>(default_config());
        m.setup();
        tch::expect_throws([&] { m.forward(); }, {"sized by a value computed inside chunked system 'inner'"});
        m.wait();
    }
    SECTION("boundary roots of different N") {
        cs::Model m = cs::create_model<tch::TwoNTop>(default_config());
        m.setup();
        tch::expect_throws([&] { m.forward(); }, {"disagree", "leading length 8", "leading length 6"});
        m.wait();
    }
    SECTION("a promoted input with the wrong leading length") {
        cs::Model m = cs::create_model<tch::WrongLenTop>(default_config());
        m.setup();
        tch::expect_throws([&] { m.forward(); }, {"disagree", "leading length 6", "leading length 8"});
        m.wait();
    }
    SECTION("a promoted input whose dimension 1 is unresolved") {
        cs::Model m = cs::create_model<tch::UnresolvedTop>(default_config());
        std::string what = "<no exception>";
        try {
            m.setup();
            m.forward();
            m.wait();
        } catch (const std::exception& e) {
            what = e.what();
        }
        INFO("thrown: " << what);
        CHECK(what != "<no exception>");
        CHECK(what.find("unresolved dimension 1") != std::string::npos);
        CHECK(what.find("late.x") != std::string::npos);
    }
    SECTION("one index over budget") {
        cs::Model m = cs::create_model<tch::DenseTop<tch::InnerChunk>>(default_config(), std::size_t{10}, std::size_t{8});
        m.setup();
        tch::expect_throws([&] { m.forward(); }, {"index 0 needs 16 bytes", "budget of 8 bytes"});
        m.wait();
    }
    SECTION("nested chunked systems") {
        cs::Model m = cs::create_model<tch::NestTop>(default_config());
        tch::expect_throws([&] { m.setup(); }, {"nested chunked systems are not supported"});
    }
    SECTION("a budget of 0") {
        tch::expect_throws([] { cs::ChunkedSystem c{0}; }, {"budget"});
    }
}


// ---- internal roots are kept across the chunks -------------------------------------------------
namespace tch {

/// Src1: x[i] = i / 2, fixed [n].
struct Src1Body {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> x;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const { o.x[i] = 0.5f * static_cast<float>(i[0]); }
};
class Src1 : public cs::Component<Src1Body> {
    std::size_t               n_;
    cs::Output<DType::f32, 1> x_;

public:
    explicit Src1(std::size_t n) : n_{n} {}
    void setup() override { declare_output(x_, "x", n_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {}, {x_}, {});
    }
};

/// AddBias: out = in + bias. `bias` is a FREE input, internal to the chunked system (its leading length is
/// derived from `in`); its init is a host function that counts its calls and fills the call number, so the
/// value each chunk sees shows when (and how often) the init was applied.
struct AddBiasBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
        cs::Input<DType::f32, 1> bias;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.out[i] = a.in[i] + a.bias[i]; }
};
class AddBias : public cs::Component<AddBiasBody> {
    std::shared_ptr<int>      calls_;
    cs::Input<DType::f32, 1>  in_, bias_;
    cs::Output<DType::f32, 1> out_;

public:
    explicit AddBias(std::shared_ptr<int> calls) : calls_{std::move(calls)} {}
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_input(bias_, "bias", cs::FromInputDim{"in", 0});
        init_input("bias", cs::init::custom_host<DType::f32, 1>([c = calls_](cs::HostViewMut<DType::f32, 1>& v) {
                       ++*c;
                       for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(*c);
                   }));
        declare_output(out_, "out", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {in_, bias_}, {out_}, {});
    }
};
class BiasInner : public cs::ChunkedSystem {
    std::shared_ptr<int> calls_;

public:
    BiasInner(std::size_t budget, std::shared_ptr<int> calls) : cs::ChunkedSystem(budget), calls_{std::move(calls)} {}
    void setup() override {
        add_subsystem<AddBias>("ab", calls_);
        promote<DType::f32, 1>("ab.in", "in");
        promote<DType::f32, 1>("ab.out", "out");
    }
};
class BiasTop : public cs::System {
    std::size_t          n_, budget_;
    std::shared_ptr<int> calls_;

public:
    BiasTop(std::size_t n, std::size_t budget, std::shared_ptr<int> calls) : n_{n}, budget_{budget}, calls_{std::move(calls)} {}
    void setup() override {
        add_subsystem<Src1>("src", n_);
        add_subsystem<BiasInner>("inner", budget_, calls_);
        add_subsystem<Cons>("cons");
        connect("src.x", "inner.in");
        connect("inner.out", "cons.in");
    }
};

}  // namespace tch

TEST_CASE("Chunked: internal roots are allocated once and kept across the chunks and the forwards", "[chunked][held]") {
    SECTION("dense, ten chunks") {
        constexpr std::size_t n = 10;
        cs::Model plain = cs::create_model<tch::DenseTop<tch::InnerPlain>>(default_config(), n, std::size_t{0});
        plain.setup();
        const auto pout = tch::forward_out(plain);
        cs::Model  m    = cs::create_model<tch::DenseTop<tch::InnerChunk>>(default_config(), n, std::size_t{16});
        m.setup();
        cs::detail::IOEntry& t = tch::inner_entry(m, "a.t");  // the internal [N,4] root
        for (int rep = 0; rep < 3; ++rep) {
            const std::size_t before = t.allocation_count();
            const auto        out    = tch::forward_out(m);
            CHECK(t.allocation_count() - before == (rep == 0 ? 1u : 0u));  // eternal: allocated by the first forward only
            CHECK(tch::chunks_of(m) == 10);
            CHECK(out == pout);
            CHECK_FALSE(t.held());
            CHECK(t.has_storage());  // kept after the last chunk
        }
    }
    SECTION("ragged, ten chunks of varying size (resized up and down within the capacity)") {
        const std::vector<std::uint32_t> counts{1, 4, 2, 4, 3, 4, 1, 4, 2, 4};
        tch::Sizes                       cost(counts.size());
        for (std::size_t i = 0; i < counts.size(); ++i) cost[i] = 4u * counts[i] * counts[i];
        cs::Model plain = cs::create_model<tch::RaggedTop<tch::RInnerPlain>>(default_config(), counts, std::size_t{0});
        plain.setup();
        const auto pout = tch::forward_out(plain);
        cs::Model  m    = cs::create_model<tch::RaggedTop<tch::RInnerChunk>>(default_config(), counts, std::size_t{64});
        m.setup();
        cs::detail::IOEntry& g = tch::inner_entry(m, "s.gram");  // the internal ragged root
        for (int rep = 0; rep < 2; ++rep) {
            const std::size_t before = g.allocation_count();
            const auto        out    = tch::forward_out(m);
            CHECK(g.allocation_count() - before == (rep == 0 ? 1u : 0u));
            CHECK(tch::chunks_of(m) == 10);
            CHECK(tch::chunks_of(m) == tch::expected_chunks(cost, 64));
            CHECK(out == pout);
            CHECK_FALSE(g.held());
        }
    }
}

TEST_CASE("Chunked: an internal free input takes its init again in every chunk", "[chunked][held]") {
    constexpr std::size_t n     = 10;
    auto                  calls = std::make_shared<int>(0);
    // 4 bytes of internal port per index, budget 12: chunks [0,3) [3,6) [6,9) [9,10)
    cs::Model m = cs::create_model<tch::BiasTop>(default_config(), n, std::size_t{12}, calls);
    m.setup();
    for (int rep = 0; rep < 2; ++rep) {
        *calls         = 0;
        const auto out = tch::forward_out(m);
        CHECK(tch::chunks_of(m) == 4);
        CHECK(*calls == 4);  // once per chunk
        REQUIRE(out.size() == n);
        for (std::size_t i = 0; i < n; ++i) {
            const float bias = static_cast<float>(i / 3 + 1);  // the call number of its chunk
            CHECK(out[i] == 2.0f * (0.5f * static_cast<float>(i) + bias) + 1.0f);
        }
        CHECK(tch::inner_entry(m, "ab.bias").allocation_count() == 1);  // once, with the first forward (eternal)
    }
}

TEST_CASE("Chunked: the init of an internal free input runs once per chunk per forward, at every budget", "[chunked][held]") {
    constexpr std::size_t n = 10;
    constexpr int         forwards = 3;
    // 4 B of internal port per index, so a chunk holds budget / 4 indices and a forward runs ceil(10 / (budget / 4)) chunks:
    //   budget 1000 -> 250 per chunk, 1 chunk;  40 -> 10, 1;  16 -> 4, 3 (4 + 4 + 2);  12 -> 3, 4;  4 -> 1, 10
    struct Case {
        std::size_t budget, rows, chunks;
    };
    for (const Case c : {Case{1000, 10, 1}, Case{40, 10, 1}, Case{16, 4, 3}, Case{12, 3, 4}, Case{4, 1, 10}}) {
        INFO("budget " << c.budget);
        auto      calls = std::make_shared<int>(0);
        cs::Model m     = cs::create_model<tch::BiasTop>(default_config(), n, c.budget, calls);
        m.setup();
        for (int f = 0; f < forwards; ++f) {
            const auto out = tch::forward_out(m);
            CHECK(tch::chunks_of(m) == c.chunks);
            CHECK(*calls == static_cast<int>(c.chunks) * (f + 1));  // never reset: chunks per forward x forwards so far
            REQUIRE(out.size() == n);
            for (std::size_t i = 0; i < n; ++i) {
                // the init fills its call number; index i is in chunk i / rows of this forward, whose call is f * chunks + chunk + 1
                const float bias = static_cast<float>(static_cast<std::size_t>(f) * c.chunks + i / c.rows + 1);
                CHECK(out[i] == 2.0f * (0.5f * static_cast<float>(i) + bias) + 1.0f);
            }
        }
        CHECK(*calls == static_cast<int>(c.chunks) * forwards);
    }
}


// ---- the framework sets dim 0 of every inside port ---------------------------------------------------
namespace tch {

/// MixFixed: t[i,k] = in[i] * (k + 1). `t` is declared with a FIXED dim 0 (`rows`).
struct MixBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> t;
    };
    void operator()(sycl::id<2> i, Inputs a, Outputs o) const { o.t[i] = a.in[i[0]] * static_cast<float>(i[1] + 1); }
};
class MixFixed : public cs::Component<MixBody> {
    std::size_t               rows_;
    cs::Input<DType::f32, 1>  in_;
    cs::Output<DType::f32, 2> t_;

public:
    explicit MixFixed(std::size_t rows) : rows_{rows} {}
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_output(t_, "t", rows_, 3);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(t_.shape(), {in_}, {t_}, {});
    }
};

/// MixFromInput: the same, with dim 0 of `t` declared FromInput{"n", 0} (the value of n[0]).
struct MixFromInputBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> t;
    };
    void operator()(sycl::id<2> i, Inputs a, Outputs o) const { o.t[i] = a.in[i[0]] * static_cast<float>(i[1] + 1); }
};
class MixFromInput : public cs::Component<MixFromInputBody> {
    cs::Input<DType::u32, 1>  n_;
    cs::Input<DType::f32, 1>  in_;
    cs::Output<DType::f32, 2> t_;

public:
    void setup() override {
        declare_input(n_, "n", cs::unspecified);
        declare_input(in_, "in", cs::unspecified);
        declare_output(t_, "t", cs::FromInput{"n", 0}, 3);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(t_.shape(), {in_}, {t_}, {});
    }
};

/// Sum3: y[i] = sum_k in[i,k].
struct Sum3Body {
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
class Sum3 : public cs::Component<Sum3Body> {
    cs::Input<DType::f32, 2>  in_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified, cs::unspecified);
        declare_output(y_, "y", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {in_}, {y_}, {});
    }
};

/// mode 0: MixFixed(rows); mode 1: MixFromInput (n comes from outside).
void fill_mix(cs::System& s, int mode, std::size_t rows) {
    if (mode == 0)
        s.add_subsystem<MixFixed>("m", rows);
    else
        s.add_subsystem<MixFromInput>("m");
    s.add_subsystem<Sum3>("s");
    s.connect("m.t", "s.in");
    s.promote<DType::f32, 1>("m.in", "in");  // the interface of an opaque chunked system
    s.promote<DType::f32, 1>("s.y", "y");
    if (mode == 1) s.promote<DType::u32, 1>("m.n", "n");
}
class MixPlain : public cs::System {
    int         mode_;
    std::size_t rows_;

public:
    MixPlain(std::size_t, int mode, std::size_t rows) : mode_{mode}, rows_{rows} {}
    void setup() override { fill_mix(*this, mode_, rows_); }
};
class MixChunk : public cs::ChunkedSystem {
    int         mode_;
    std::size_t rows_;

public:
    MixChunk(std::size_t budget, int mode, std::size_t rows) : cs::ChunkedSystem(budget), mode_{mode}, rows_{rows} {}
    void setup() override { fill_mix(*this, mode_, rows_); }
};
template <class Inner>
class MixTop : public cs::System {
    std::size_t n_, budget_, rows_;
    int         mode_;

public:
    MixTop(std::size_t n, std::size_t budget, int mode, std::size_t rows) : n_{n}, budget_{budget}, rows_{rows}, mode_{mode} {}
    void setup() override {
        add_subsystem<Src1>("src", n_);
        if (mode_ == 1) add_subsystem<CProd>("ns", std::vector<std::uint32_t>(n_, static_cast<std::uint32_t>(n_)));  // n[0] = N
        add_subsystem<Inner>("inner", budget_, mode_, rows_);
        add_subsystem<Cons>("cons");
        connect("src.x", "inner.in");
        if (mode_ == 1) connect("ns.counts", "inner.n");
        connect("inner.y", "cons.in");
    }
};

}  // namespace tch

TEST_CASE("Chunked: the framework sets dim 0 of every inside port (fixed, FromInput)", "[chunked][dim0][shape]") {
    constexpr std::size_t n = 10;
    // t is 12 bytes per index (internal [N,3] f32); budget 36: chunks of 3
    std::vector<float> want(n);
    for (std::size_t i = 0; i < n; ++i) want[i] = 2.0f * (0.5f * static_cast<float>(i) * 6.0f) + 1.0f;
    for (const int mode : {0, 1}) {
        INFO("mode " << mode << (mode == 0 ? " (dim 0 declared FIXED, = N)" : " (dim 0 declared FromInput{n,0})"));
        // a specified dim 0 is a constraint checked at the full length, so both declare N (a wrong one is an error: [shape] tests)
        cs::Model plain = cs::create_model<tch::MixTop<tch::MixPlain>>(default_config(), n, std::size_t{0}, mode, n);
        plain.setup();
        const auto pout = tch::forward_out(plain);
        CHECK(pout == want);
        for (const std::size_t budget : {std::size_t{36}, std::size_t{12}, std::size_t{1000}}) {
            cs::Model m = cs::create_model<tch::MixTop<tch::MixChunk>>(default_config(), n, budget, mode, n);
            m.setup();
            CHECK(tch::forward_out(m) == pout);
            CHECK(tch::chunks_of(m) == (budget == 36 ? 4u : budget == 12 ? 10u : 1u));
            CHECK(tch::forward_out(m) == pout);  // and again
        }
    }
    // a chunked system whose internal port has a fixed dim 0 that is not N, with no consumer
    // that is a constraint violation, reported at the full length
    cs::Model odd = cs::create_model<tch::OddTop>(default_config());
    odd.setup();
    tch::expect_throws([&] { (void)tch::forward_out(odd); }, {"shape mismatch", "dimension 0 of 'inner.g.out'", "fixed = 13", "full length 8"});
    odd.wait();
}

TEST_CASE("Chunked: a component reused outside a chunked system keeps its declared dim 0", "[chunked][dim0]") {
    // MixFixed declares 4 rows for t: standalone, in a plain System, t has 4 rows whatever the input length
    cs::Model m = cs::create_model<tch::MixTop<tch::MixPlain>>(default_config(), std::size_t{10}, std::size_t{0}, 0, std::size_t{4});
    m.setup();
    const auto out = tch::forward_out(m);
    REQUIRE(out.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) CHECK(out[i] == 2.0f * (0.5f * static_cast<float>(i) * 6.0f) + 1.0f);
}


// =============================================================================================================================
// Chunked runs against the unchunked twin (the same Top with a plain System), forward for forward, bitwise.
// =============================================================================================================================
namespace tch {
std::vector<float> out_cons(cs::Model& m) {
    auto v = m.view_output<DType::f32, 1>("cons.out");
    return std::vector<float>(v.span().begin(), v.span().end());
}
}  // namespace tch

TEST_CASE("Chunked vs plain: dense chain (with and without the inside output promoted out)", "[chunked][replay]") {
    constexpr std::size_t n = 10;
    for (const bool prom : {false, true})
        for (const std::size_t budget : {std::size_t{160}, std::size_t{48}, std::size_t{16}}) {
            INFO("promote " << prom << ", budget " << budget);
            const auto caps = tcc::against_plain(
                "inner", [&] { return cs::create_model<tch::DenseTop<tch::InnerChunk>>(default_config(), n, budget, prom); },
                [&] { return cs::create_model<tch::DenseTop<tch::InnerPlain>>(default_config(), n, std::size_t{0}, prom); },
                [prom](cs::Model& m) {
                    std::vector<float> out = tch::out_cons(m);
                    if (prom) {  // the promoted [N,4] root, whole
                        auto t = m.view_output<DType::f32, 2>("inner.t");
                        out.insert(out.end(), t.span().begin(), t.span().end());
                    }
                    return out;
                });
            // by hand: t is 16 B per index, so 10 / 3 / 1 indices fit the budgets 160 / 48 / 16
            if (!prom)
                for (const auto& c : caps) CHECK(c.chunks.size() == (budget == 160 ? 1u : budget == 48 ? 4u : 10u));
        }
}

TEST_CASE("Chunked vs plain: a promoted free input and an init between forwards", "[chunked][replay]") {
    constexpr std::size_t n = 10;
    tcc::against_plain("inner", [&] { return cs::create_model<tch::PromTop<tch::PromInnerChunk>>(default_config(), n, std::size_t{48}); },
                       [&] { return cs::create_model<tch::PromTop<tch::PromInnerPlain>>(default_config(), n, std::size_t{0}); },
                       tch::out_cons, 3, [&](cs::Model& m, int f) {
                           std::vector<float> w(n * 3);
                           for (std::size_t q = 0; q < w.size(); ++q) w[q] = static_cast<float>(f) * static_cast<float>(q % 7) + 0.25f;
                           m.init_input("inner.w", cs::init::from_host(w));
                       });
}

TEST_CASE("Chunked vs plain: ragged scratch (FromInputDim, FromInputValues, unspecified), counts changing, zeros", "[chunked][replay]") {
    const std::vector<std::uint32_t> c1{3, 0, 2, 4, 0, 1, 2, 3}, c2{0, 4, 4, 1, 0, 0, 2, 1}, c0(8, 0);
    for (const std::size_t budget : {std::size_t{64}, std::size_t{100}, std::size_t{128}, std::size_t{1000}}) {
        INFO("budget " << budget);
        tcc::against_plain("inner", [&] { return cs::create_model<tch::RaggedTop<tch::RInnerChunk>>(default_config(), c1, budget); },
                           [&] { return cs::create_model<tch::RaggedTop<tch::RInnerPlain>>(default_config(), c1, std::size_t{0}); },
                           tch::out_cons, 4, [&](cs::Model& m, int f) {
                               m.init_input("prod.raw", cs::init::from_host(f == 1 ? c2 : f == 2 ? c0 : c1));
                           });
    }
}

TEST_CASE("Chunked vs plain: dim 0 rules (fixed, FromInput)", "[chunked][replay]") {
    constexpr std::size_t n = 10;
    for (const int mode : {0, 1})
        for (const std::size_t budget : {std::size_t{36}, std::size_t{12}, std::size_t{1000}}) {
            INFO("mode " << mode << ", budget " << budget);
            tcc::against_plain("inner", [&] { return cs::create_model<tch::MixTop<tch::MixChunk>>(default_config(), n, budget, mode, n); },
                               [&] { return cs::create_model<tch::MixTop<tch::MixPlain>>(default_config(), n, std::size_t{0}, mode, n); },
                               tch::out_cons);
        }
}

// ---- the topology cache: validity and the zero-event window --------------------------------------------------------------
namespace tch {
struct DiagFwd {
    cs::diag::EventCounts whole, window;
};
DiagFwd fwd_diag(cs::Model& m) {
    const auto before = cs::diag::snapshot();
    m.forward();
    m.wait();
    return {cs::diag::snapshot() - before, cs::diag::last_window()};
}
void check_quiet_window(const cs::diag::EventCounts& w) {
    CHECK(w.queue_waits == 0);
    CHECK(w.frees == 0);
    CHECK(w.allocations == 0);
    CHECK(w.host_reads == 0);
    CHECK(w.host_read_hits == 0);
    CHECK(w.ragged_uploads == 0);
    CHECK(w.full_passes == 0);
    CHECK(w.resolve_shapes == 0);
}
}  // namespace tch

TEST_CASE("Topology cache: a value dependency rebuilds only when its VALUE changes", "[chunked][replay][cache]") {
    const std::vector<std::uint32_t> c1{3, 0, 2, 4, 0, 1, 2, 3}, c2{0, 4, 4, 1, 0, 0, 2, 1};
    cs::Model                        m = cs::create_model<tch::RaggedTop<tch::RInnerChunk>>(default_config(), c1, std::size_t{100});
    m.setup();
    // `prod` and `gen` run every forward (their outputs' epochs move) but the values and shapes they produce are the same
    const auto f1 = tch::fwd_diag(m);
    CHECK(f1.whole.full_passes == 1);
    tch::check_quiet_window(f1.window);
    for (int rep = 0; rep < 2; ++rep) {
        const auto f = tch::fwd_diag(m);
        CHECK(f.whole.full_passes == 0);
        tch::check_quiet_window(f.window);
    }
    // the same values written again: still no rebuild
    m.init_input("prod.raw", cs::init::from_host(c1));
    CHECK(tch::fwd_diag(m).whole.full_passes == 0);
    // other values: exactly one rebuild, then quiet again
    m.init_input("prod.raw", cs::init::from_host(c2));
    const auto f4 = tch::fwd_diag(m);
    CHECK(f4.whole.full_passes == 1);
    tch::check_quiet_window(f4.window);
    CHECK(tch::fwd_diag(m).whole.full_passes == 0);
    const auto o = tch::out_cons(m);
    cs::Model  p = cs::create_model<tch::RaggedTop<tch::RInnerPlain>>(default_config(), c2, std::size_t{0});
    p.setup();
    CHECK(o == tch::forward_out(p));
}

TEST_CASE("Topology cache: writing an unrelated root (a free input's data) gives no rebuild", "[chunked][replay][cache]") {
    constexpr std::size_t n = 10;
    cs::Model             m = cs::create_model<tch::PromTop<tch::PromInnerChunk>>(default_config(), n, std::size_t{48});
    m.setup();
    CHECK(tch::fwd_diag(m).whole.full_passes == 1);
    for (const float scale : {2.0f, 5.0f}) {
        std::vector<float> w(n * 3);
        for (std::size_t q = 0; q < w.size(); ++q) w[q] = scale * static_cast<float>(q % 7) + 0.25f;
        m.init_input("inner.w", cs::init::from_host(w));
        const auto f = tch::fwd_diag(m);
        CHECK(f.whole.full_passes == 0);  // `w` drives no shape
        tch::check_quiet_window(f.window);
        const auto out = tch::out_cons(m);
        for (std::size_t i = 0; i < n; ++i)
            CHECK(out[i] == 2.0f * ((w[i * 3] + 2.0f * w[i * 3 + 1] + 3.0f * w[i * 3 + 2]) * 10.0f) + 1.0f);  // new data, not a stale copy
    }
}

TEST_CASE("Topology cache: the window of an internal free input with an init", "[chunked][replay][cache]") {
    constexpr std::size_t n     = 10;
    auto                  calls = std::make_shared<int>(0);
    cs::Model             m     = cs::create_model<tch::BiasTop>(default_config(), n, std::size_t{12}, calls);  // 4 chunks
    m.setup();
    for (int rep = 0; rep < 3; ++rep) {
        const auto f = tch::fwd_diag(m);
        std::printf("[replay] internal free input with a custom_host init, forward %d: window host_tasks %llu, host_reads %llu, host_read_hits %llu\n",
                    rep + 1, static_cast<unsigned long long>(f.window.host_tasks), static_cast<unsigned long long>(f.window.host_reads),
                    static_cast<unsigned long long>(f.window.host_read_hits));
        CHECK(f.window.queue_waits == 0);
        CHECK(f.window.frees == 0);
        CHECK(f.window.allocations == 0);
        CHECK(f.window.ragged_uploads == 0);
        CHECK(f.window.full_passes == 0);
        CHECK(f.window.resolve_shapes == 0);
        CHECK(f.window.host_tasks == 4);  // the commits (queued copies), once per chunk
        CHECK(f.window.host_reads == 0);  // custom_host starts from a zeroed host buffer: no device read
        CHECK(f.window.host_read_hits == 0);  // the init's write, once per chunk
        CHECK(f.whole.full_passes == (rep == 0 ? 1u : 0u));
    }
}

// ---- lifetime by counting inside a ChunkedSystem ----------------------------------------------------------------
namespace tch {

/// DenseTop with the chunked system added as a TRANSIENT subsystem.
template <class Inner>
class DenseTopTransient : public cs::System {
    std::size_t n_, budget_;

public:
    DenseTopTransient(std::size_t n, std::size_t budget) : n_{n}, budget_{budget} {}
    void setup() override {
        add_subsystem<Prod>("prod", n_);
        add_subsystem<Inner>("inner", cs::property_list{cs::property::transient{}}, budget_, false);
        add_subsystem<Cons>("cons");
        connect("prod.x", "inner.in");
        connect("inner.y", "cons.in");
    }
};

/// The dense chunked system with b added with run_n{2}: b's counter moves once per FORWARD, not once per chunk.
class RunNInner : public cs::ChunkedSystem {
public:
    RunNInner(std::size_t budget, bool /*promote_t*/) : cs::ChunkedSystem(budget) {}
    void setup() override {
        add_subsystem<A>("a");
        add_subsystem<B>("b", cs::property_list{cs::property::run_n{2}});
        connect("a.t", "b.in");
        promote<DType::f32, 2>("a.in", "in");
        promote<DType::f32, 1>("b.y", "y");
    }
};

/// One forward and its wait, with the whole-forward event counts.
cs::diag::EventCounts counted_forward(cs::Model& m) {
    const auto before = cs::diag::snapshot();
    m.forward();
    m.wait();
    return cs::diag::snapshot() - before;
}

}  // namespace tch

TEST_CASE("Chunked eternal: the internal roots are allocated once across three forwards", "[chunked][lifetime]") {
    cs::Model m = cs::create_model<tch::DenseTop<tch::InnerChunk>>(default_config(), std::size_t{10}, std::size_t{16});
    m.setup();
    const auto f1 = tch::counted_forward(m);
    const auto f2 = tch::counted_forward(m);
    const auto f3 = tch::counted_forward(m);
    CHECK(f1.allocations >= 1);
    CHECK(f2.allocations == 0);
    CHECK(f3.allocations == 0);
    CHECK(f2.frees == 0);
    CHECK(f3.frees == 0);
    CHECK(tch::inner_entry(m, "a.t").allocation_count() == 1);
    CHECK(tch::inner_entry(m, "a.t").present());
}

TEST_CASE("Chunked transient: the chunked system is deleted after its forward and its internal roots go with it",
          "[chunked][lifetime]") {
    constexpr std::size_t n = 10;
    cs::Model plain = cs::create_model<tch::DenseTop<tch::InnerPlain>>(default_config(), n, std::size_t{0});
    plain.setup();
    const auto pout = tch::forward_out(plain);
    cs::Model m = cs::create_model<tch::DenseTopTransient<tch::InnerChunk>>(default_config(), n, std::size_t{16});
    m.setup();
    for (int rep = 0; rep < 3; ++rep) {
        INFO("forward " << rep + 1);
        const auto d = tch::counted_forward(m);
        CHECK(d.queue_waits <= 1);   // only the Model::wait of this helper: frees order themselves on registered events
        if (rep > 0) CHECK(d.allocations == 1);
        CHECK(m.get_subsystem("inner") == nullptr);  // transient: deleted after its turn (chunks_of / a.t are gone with it)
        const auto v = m.view_output<DType::f32, 1>("cons.out");
        CHECK(std::vector<float>(v.span().begin(), v.span().end()) == pout);
    }
}

TEST_CASE("Chunked: a run_n child counts forwards, not chunks", "[chunked][lifetime]") {
    constexpr std::size_t n = 10;
    cs::Model m = cs::create_model<tch::DenseTop<tch::RunNInner>>(default_config(), n, std::size_t{16});
    m.setup();
    auto&             inner = dynamic_cast<cs::ChunkedSystem&>((*m.get_subsystem("inner")));
    REQUIRE(inner.children()[1] != nullptr);
    REQUIRE(inner.children()[1]->name() == "b");
    const bool expected[3] = {true, false, false};  // run_n{2}: deleted after the last chunk of forward 2
    for (int rep = 0; rep < 3; ++rep) {
        INFO("forward " << rep + 1);
        (void)tch::forward_out(m);
        CHECK(tch::chunks_of(m) == 10);  // ten chunks every forward, b skipped in all of them once it has run twice
        CHECK((inner.children()[1] != nullptr) == expected[rep]);
    }
}

TEST_CASE("shape: a specified dim 0 inside a ChunkedSystem is a constraint checked at the full length", "[chunked][dim0][shape]") {
    constexpr std::size_t n = 10;
    SECTION("a fixed dim 0 that is not N: reported once, at the full length, naming both sides") {
        cs::Model m = cs::create_model<tch::MixTop<tch::MixChunk>>(default_config(), n, std::size_t{36}, 0, std::size_t{7});
        m.setup();
        tch::expect_throws([&] { (void)tch::forward_out(m); },
                           {"shape mismatch", "dimension 0 of \'inner.m.t\' is fixed = 7", "full length 10"});
        m.wait();
    }
    SECTION("a fixed dim 0 equal to N is accepted, whatever the chunk length") {
        cs::Model m = cs::create_model<tch::MixTop<tch::MixChunk>>(default_config(), n, std::size_t{36}, 0, n);
        m.setup();
        CHECK_NOTHROW((void)tch::forward_out(m));
        CHECK(tch::chunks_of(m) == 4);  // chunks of 3: the declared 10 was checked at the full length, not per chunk
    }
}