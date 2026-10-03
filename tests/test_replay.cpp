// ChunkedSystem topology cache: the one full-pass build, the hooks and the resize counter. The comparison with the
// unchunked twin, the cache validity and the zero-event window are in test_chunked.cpp and test_ragged_ties.cpp,
// next to the models they use.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace trp {

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
        declare_input(in_, "in", cs::unspecified, cs::unspecified);
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
        declare_input(in_, "in", cs::unspecified, cs::unspecified);
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
        declare_input(in_, "in", cs::unspecified);
        declare_output(out_, "out", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(out_.shape(), {in_}, {out_}, {}); }
};
void fill_inner(cs::System& s) {
    s.add_subsystem<A>("a");
    s.add_subsystem<B>("b");
    s.connect("a.t", "b.in");
    s.promote<DType::f32, 2>("a.in", "in");  // the interface of an opaque chunked system
    s.promote<DType::f32, 1>("b.y", "y");
}
class Inner : public cs::ChunkedSystem {
public:
    explicit Inner(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override { fill_inner(*this); }
};
/// The unchunked twin: the same subtree in a plain System.
class InnerPlain : public cs::System {
public:
    explicit InnerPlain(std::size_t /*budget*/) {}
    void setup() override { fill_inner(*this); }
};
template <class InnerT>
class TopT : public cs::System {
    std::size_t n_, budget_;

public:
    TopT(std::size_t n, std::size_t budget) : n_{n}, budget_{budget} {}
    void setup() override {
        add_subsystem<Prod>("prod", n_);
        add_subsystem<InnerT>("inner", budget_);
        add_subsystem<Cons>("cons");
        connect("prod.x", "inner.in");
        connect("inner.y", "cons.in");  // y crosses the boundary: only t (16 B per index) is internal
    }
};
using Top      = TopT<Inner>;
using PlainTop = TopT<InnerPlain>;

std::vector<float> y_of(cs::Model& m) {
    auto v = m.view_output<DType::f32, 1>("cons.out");
    return std::vector<float>(v.span().begin(), v.span().end());
}
cs::diag::EventCounts fwd(cs::Model& m) {
    const auto b = cs::diag::snapshot();
    m.forward();
    m.wait();
    return cs::diag::snapshot() - b;
}

}  // namespace trp

TEST_CASE("Replay: the topology is built by ONE full pass, cached, and the output equals the unchunked twin's", "[chunked][replay]") {
    constexpr std::size_t n = 10, budget = 56;  // 16 B per index: chunks of 3, 3, 3, 1
    cs::Model             plain = cs::create_model<trp::PlainTop>(default_config(), n, std::size_t{0});
    plain.setup();
    trp::fwd(plain);
    const auto want = trp::y_of(plain);

    cs::Model m = cs::create_model<trp::Top>(default_config(), n, budget);
    m.setup();
    auto& sys = dynamic_cast<cs::ChunkedSystem&>((*m.get_subsystem("inner")));
    CHECK_FALSE(sys.topology_cached());
    const auto a = trp::fwd(m);
    CHECK(a.full_passes == 1);  // the one build
    CHECK(sys.topology_cached());
    CHECK(trp::y_of(m) == want);
    for (int rep = 0; rep < 2; ++rep) {  // unchanged dependencies: no pass, same values
        CHECK(trp::fwd(m).full_passes == 0);
        CHECK(trp::y_of(m) == want);
    }
}

TEST_CASE("Replay: the before-launch hook sees every chunk's cached shape and leading range", "[chunked][replay]") {
    constexpr std::size_t n = 10, budget = 56;
    cs::Model             m = cs::create_model<trp::Top>(default_config(), n, budget);
    m.setup();
    auto&                                           sys = dynamic_cast<cs::ChunkedSystem&>((*m.get_subsystem("inner")));
    std::vector<std::size_t>                        t_rows, in_a, in_b;
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    sys.set_chunk_hooks(
        [&](std::size_t) {
            t_rows.push_back(sys.entry_at("a.t").extents()[0]);  // the internal root: this chunk's rows
            const auto lr = sys.entry_at("a.in").leading_range();
            REQUIRE(lr.has_value());
            ranges.push_back(*lr);
            in_a.push_back(sys.entry_at("a.in").extents()[0]);
        },
        nullptr);
    m.forward();
    m.wait();
    CHECK(t_rows == std::vector<std::size_t>{3, 3, 3, 1});
    CHECK(in_a == std::vector<std::size_t>{3, 3, 3, 1});
    CHECK(ranges == std::vector<std::pair<std::size_t, std::size_t>>{{0, 3}, {3, 6}, {6, 9}, {9, 10}});
    // after the forward the whole run is back: no range, the full shape
    CHECK_FALSE(sys.entry_at("a.in").leading_range().has_value());
    CHECK(sys.entry_at("a.in").extents()[0] == n);
    CHECK(sys.entry_at("b.y").extents()[0] == n);
    CHECK(sys.entry_at("a.t").extents()[0] == n);
    CHECK(sys.last_chunk_count() == 4);
}


TEST_CASE("Replay: the resize counter is 0 in the chunk loop's window and a chunked run keeps the chunk capacity", "[chunked][replay][diag]") {
    constexpr std::size_t n = 10, budget = 56;  // chunks of 3, 3, 3, 1; t is [N,4] f32: 16 B per index
    cs::Model             m = cs::create_model<trp::Top>(default_config(), n, budget);
    m.setup();
    auto& sys = dynamic_cast<cs::ChunkedSystem&>((*m.get_subsystem("inner")));
    for (int rep = 0; rep < 3; ++rep) {
        trp::fwd(m);
        CHECK(cs::diag::last_window().resizes == 0);
        // the internal root's storage holds exactly the largest chunk (3 rows of 4), never the whole run
        CHECK(sys.entry_at("a.t").capacity_elems() == 3 * 4);
    }
}
