// Initializers check every dimension of the root they are applied to, and carry ragged shapes.
//
//   cs::init::Shape / ragged(n, d1[, d2]) / from_host(data, ragged shape)
//   detail::apply_root_init compares the init's shape with the resolved root on the rank and on EVERY dimension
//   (fixed, inherited, value-driven, per-index element by element), at the full range, and names both sides.
//
// Values are checked against sums worked out by hand.

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

namespace tis {

using Sizes = std::vector<std::size_t>;

// ---- a System built by a function --------------------------------------------------------------------------------
class LambdaTop : public cs::System {
    std::function<void(cs::System&)> f_;

public:
    explicit LambdaTop(std::function<void(cs::System&)> f) : f_{std::move(f)} {}
    void setup() override { f_(*this); }
};
cs::Model make(std::function<void(cs::System&)> f) { return cs::create_model<LambdaTop>(default_config(), std::move(f)); }

/// The forward throws; its message holds every needle.
void forward_throws(cs::Model& m, const std::vector<std::string>& needles) {
    std::string what = "<no exception>";
    try {
        m.forward();
        m.wait();
    } catch (const std::exception& e) {
        what = e.what();
    }
    try {
        m.wait();
    } catch (...) {
    }
    INFO("thrown: " << what);
    for (const auto& n : needles) CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}

// ---- Probe: y = x + w; x, w dense rank 1, n a one-element u32 -------------------------------------------------------
struct ProbeBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> w;
        cs::Input<DType::u32, 1> n;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.y[i] = a.x[i] + a.w[i]; }
};
/// mode 0: x, w fixed 5.  mode 1: x unspecified, w = FromInputDim{x, 0}.  mode 2: x fixed 3, w = FromInput{n, 0}.
class Probe : public cs::Component<ProbeBody> {
    int                       mode_;
    cs::Input<DType::f32, 1>  x_, w_;
    cs::Input<DType::u32, 1>  n_;
    cs::Output<DType::f32, 1> y_;

public:
    explicit Probe(int mode) : mode_{mode} {}
    void setup() override {
        declare_input(n_, "n", std::size_t{1});
        if (mode_ == 0) {
            declare_input(x_, "x", std::size_t{5});
            declare_input(w_, "w", std::size_t{5});
        } else if (mode_ == 1) {
            declare_input(x_, "x", cs::unspecified);
            declare_input(w_, "w", cs::FromInputDim{"x", 0});
        } else {
            declare_input(x_, "x", std::size_t{3});
            declare_input(w_, "w", cs::FromInput{"n", 0});
        }
        declare_output(y_, "y", cs::FromInputDim{"x", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {x_, w_, n_}, {y_}, {});
    }
};

// ---- RagSum: sums[i] = the sum of block i of a ragged [N, c_i] input ----------------------------------------------
struct RagSumBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    void operator()(sycl::id<1> id, Inputs i, Outputs o) const {
        auto  b = i.x.block(id[0]);
        float s = 0.0f;
        for (std::size_t q = 0; q < b.size(); ++q) s += b[q];
        o.sums[id] = s;
    }
};
/// mode 0: x unspecified in both dims (a connection or an init shapes it).  mode 1: x has all dims unspecified, for a tie.
/// mode 2: x declared with n = sizes.size() and per_index(sizes).
class RagSum : public cs::Component<RagSumBody> {
    int                            mode_;
    Sizes                          sizes_;
    cs::RaggedInput<DType::f32, 2> x_;
    cs::Output<DType::f32, 1>      sums_;

public:
    RagSum(int mode, Sizes sizes) : mode_{mode}, sizes_{std::move(sizes)} {}
    void setup() override {
        if (mode_ == 0)
            declare_input(x_, "x", cs::unspecified, cs::unspecified);
        else if (mode_ == 1)
            declare_input(x_, "x", cs::unspecified_shape<2>());
        else
            declare_input(x_, "x", sizes_.size(), cs::per_index(sizes_));
        declare_output(sums_, "sums", cs::FromInputDim{"x", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sums_.shape(), {x_}, {sums_}, {});
    }
};

// ---- Id: y = x, both ragged [N, c_i]; x is a free input shaped by its init ---------------------------------------------
struct IdBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 2> x;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> y;
    };
    void operator()(sycl::id<1> id, Inputs i, Outputs o) const {
        auto a = i.x.block(id[0]);
        auto b = o.y.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = a[q];
    }
};
class Id : public cs::Component<IdBody> {
    cs::RaggedInput<DType::f32, 2>  x_;
    cs::RaggedOutput<DType::f32, 2> y_;

public:
    void setup() override {
        declare_input(x_, "x", cs::unspecified, cs::unspecified);
        declare_output(y_, "y", cs::FromInputDim{"x", 0}, cs::FromInputDim{"x", 1});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{y_.size()}, {x_}, {y_}, {});
    }
};

// ---- Dbl: out = 2 in (an INTERNAL root of the chunked subtree: it is what the budget is spent on) ----------------------
struct DblBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.out[i] = 2.0f * a.in[i]; }
};
class Dbl : public cs::Component<DblBody> {
    cs::Input<DType::f32, 1>  in_;
    cs::Output<DType::f32, 1> out_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_output(out_, "out", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(out_.shape(), {in_}, {out_}, {}); }
};

// ---- the subtree under test, as a ChunkedSystem and as its plain twin ------------------------------------------------
struct PlainBase : cs::System {
    explicit PlainBase(std::size_t) {}
};
/// p (mode 0) -> d, with p's input promoted as "x" and d's output as "sums" (p.sums, 4 B per index, is internal).
template <class Base>
class Inner : public Base {
public:
    explicit Inner(std::size_t budget) : Base(budget) {}
    void setup() override {
        this->template add_subsystem<RagSum>("p", 0, Sizes{});
        this->template add_subsystem<Dbl>("d");
        this->connect("p.sums", "d.in");
        this->template promote<DType::f32, 2>("p.x", "x");
        this->template promote<DType::f32, 1>("d.out", "sums");
    }
};
using InnerChunk = Inner<cs::ChunkedSystem>;
using InnerPlain = Inner<PlainBase>;

/// The free input is the subtree's own promoted input, shaped by this init.
template <class InnerT>
class TopFree : public cs::System {
    std::size_t budget_;
    cs::InitSpec init_;

public:
    TopFree(std::size_t budget, cs::InitSpec init) : budget_{budget}, init_{std::move(init)} {}
    void setup() override {
        add_subsystem<InnerT>("inner", budget_);
        init_input("inner.x", init_);
    }
};
/// The free input is an outside component's; the subtree reads its output (under a leading range when chunked).
template <class InnerT>
class TopConn : public cs::System {
    std::size_t budget_;
    cs::InitSpec init_;

public:
    TopConn(std::size_t budget, cs::InitSpec init) : budget_{budget}, init_{std::move(init)} {}
    void setup() override {
        add_subsystem<Id>("src");
        add_subsystem<InnerT>("inner", budget_);
        connect("src.y", "inner.x");
        init_input("src.x", init_);
    }
};
/// Two members tied into "t", shaped by an init on the tie.
class TieTop : public cs::System {
    cs::InitSpec init_;

public:
    explicit TieTop(cs::InitSpec init) : init_{std::move(init)} {}
    void setup() override {
        add_subsystem<RagSum>("a", 1, Sizes{});
        add_subsystem<RagSum>("b", 1, Sizes{});
        tie_inputs<DType::f32, 2>({"a.x", "b.x"}, "t");
        init_input("t", init_);
    }
};

std::vector<float> sums_of(cs::Model& m) {
    auto v = m.view_output<DType::f32, 1>("inner.sums");
    return std::vector<float>(v.span().begin(), v.span().end());
}

std::vector<float> iota(std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i);
    return v;
}

/// The block offsets of blocks [a, b) of a ragged [N, c_i] root, rebased to start at 0 (worked out from the sizes).
std::vector<std::uint64_t> rebased(const Sizes& s, std::size_t a, std::size_t b) {
    std::vector<std::uint64_t> offs{0};
    for (std::size_t i = a; i < b; ++i) offs.push_back(offs.back() + s[i]);
    return offs;
}

// sizes {3, 0, 4, 1, 2} over iota(10): blocks [0 1 2] [] [3 4 5 6] [7] [8 9]
const Sizes              kSizes{3, 0, 4, 1, 2};
const std::vector<float> kSums{3.0f, 0.0f, 18.0f, 7.0f, 17.0f};
const std::vector<float> kDoubled{6.0f, 0.0f, 36.0f, 14.0f, 34.0f};  // what the subtree's "sums" holds

cs::InitSpec ragged_init() { return cs::init::from_host(iota(10), cs::init::ragged(5, kSizes)); }

/// Chunk the subtree against its plain twin, and record the shape of `path` in every chunk. By hand: the chunks tile the
/// five blocks in order, and in each chunk [a, b) the input has dim 0 = b - a, per-index extents kSizes[a..b) and the
/// block offsets rebased from 0.
template <template <class> class Top>
void chunked_matches_plain(std::size_t budget) {
    INFO("budget " << budget);
    tcc::against_plain(
        "inner", [&] { return cs::create_model<Top<InnerChunk>>(default_config(), budget, ragged_init()); },
        [&] { return cs::create_model<Top<InnerPlain>>(default_config(), std::size_t{0}, ragged_init()); }, sums_of);

    cs::Model m = cs::create_model<Top<InnerChunk>>(default_config(), budget, ragged_init());
    m.setup();
    auto&         sys = dynamic_cast<cs::ChunkedSystem&>((*m.get_subsystem("inner")));
    tcc::Recorder rec;
    rec.attach(sys, {"p.x"});
    m.forward();
    m.wait();
    CHECK(sums_of(m) == kDoubled);
    std::size_t next = 0;
    for (const auto& row : rec.chunks) {
        const std::size_t a = next, b = next + row[0].dim0;
        INFO("chunk [" << a << ", " << b << ")");
        REQUIRE(b <= kSizes.size());
        CHECK(row[0].per1 == Sizes(kSizes.begin() + static_cast<std::ptrdiff_t>(a), kSizes.begin() + static_cast<std::ptrdiff_t>(b)));
        CHECK(row[0].offsets == rebased(kSizes, a, b));
        next = b;
    }
    CHECK(next == kSizes.size());
    if (budget >= 1000) CHECK(rec.chunks.size() == 1);  // by hand: it all fits
    else CHECK(rec.chunks.size() > 1);
    // the whole run is back afterwards: five blocks, the full per-index array
    CHECK(sys.entry_at("p.x").extents()[0] == 5);
    CHECK(sys.entry_at("p.x").per_index_raw(1) == kSizes);
}

}  // namespace tis

// =============================================================================================================================
// every dimension is checked, and the message names both sides
// =============================================================================================================================
TEST_CASE("shape: an init that disagrees with a FIXED dim names both sides", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem<tis::Probe>("p", 0);
        s.init_input("p.x", cs::init::fill(1.0f, {4}));
    });
    m.setup();
    tis::forward_throws(m, {"init for 'p.x' has dimension 0 = 4", "'p.x' resolved dimension 0 (fixed) = 5"});
}

TEST_CASE("shape: an init that disagrees with an INHERITED dim names both sides", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem<tis::Probe>("p", 1);
        s.init_input("p.x", cs::init::fill(1.0f, {4}));  // x is shaped by this: 4
        s.init_input("p.w", cs::init::fill(1.0f, {3}));  // w inherits x's dim 0 = 4
    });
    m.setup();
    tis::forward_throws(m, {"init for 'p.w' has dimension 0 = 3", "'p.w' resolved dimension 0 (FromInputDim{'x', 0}) = 4"});
}

TEST_CASE("shape: an init that disagrees with a VALUE-DRIVEN dim names both sides", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem<tis::Probe>("p", 2);
        s.init_input("p.n", cs::init::from_host(std::vector<std::uint32_t>{3}));  // w has 3 elements
        s.init_input("p.w", cs::init::fill(1.0f, {5}));
    });
    m.setup();
    tis::forward_throws(m, {"init for 'p.w' has dimension 0 = 5", "'p.w' resolved dimension 0 (FromInput{'n', 0}) = 3"});
}

TEST_CASE("shape: an init that disagrees with a PER-INDEX dim names both sides, element by element", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem<tis::RagSum>("p", 2, tis::Sizes{3, 1, 4});
        s.init_input("p.x", cs::init::from_host(std::vector<float>(7, 1.0f), cs::init::ragged(3, tis::Sizes{3, 0, 4})));
    });
    m.setup();
    tis::forward_throws(m, {"init for 'p.x' has dimension 1 = [3,0,4]", "'p.x' resolved dimension 1 (PerIndex) = [3,1,4]"});
}

TEST_CASE("shape: an init that agrees with a per-index dim runs", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem<tis::RagSum>("p", 2, tis::Sizes{3, 1, 4});
        s.init_input("p.x", cs::init::from_host(tis::iota(8), cs::init::ragged(3, tis::Sizes{3, 1, 4})));
    });
    m.setup();
    m.forward();
    m.wait();
    // by hand: [0 1 2] [3] [4 5 6 7]
    CHECK(cst::to_vector(m.view_output<DType::f32, 1>("p.sums")) == std::vector<float>{3.0f, 3.0f, 22.0f});
}

TEST_CASE("shape: a rank mismatch names both sides", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem<tis::Probe>("p", 0);
        s.init_input("p.x", cs::init::fill(1.0f, {2, 3}));
    });
    m.setup();
    tis::forward_throws(m, {"init for 'p.x' has rank 2", "'p.x' resolved rank 1"});
}

TEST_CASE("shape: an init with no shape still needs every dim specified", "[shape][init]") {
    std::string what = "<no exception>";
    try {
        auto m = tis::make([](cs::System& s) {
            s.add_subsystem<tis::Probe>("p", 1);
            s.init_input("p.x", cs::init::fill(1.0f));
        });
        m.setup();
    } catch (const std::exception& e) {
        what = e.what();
    }
    CHECK_THAT(what, Catch::Matchers::ContainsSubstring("free input 'p.x' has an unspecified dimension and no initializer that gives it a shape"));
}

TEST_CASE("shape: bare data (from_host(vec)) carries no shape for a rank-2 root", "[shape][init]") {
    // from_host(vec) describes a rank-1 root; a fully specified rank-2 root takes it by size alone, as it always did
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem<tis::RagSum>("p", 2, tis::Sizes{2, 2});
        s.init_input("p.x", cs::init::from_host(tis::iota(4)));
    });
    m.setup();
    m.forward();
    m.wait();
    CHECK(cst::to_vector(m.view_output<DType::f32, 1>("p.sums")) == std::vector<float>{1.0f, 5.0f});  // [0 1] [2 3]
}

// =============================================================================================================================
// ragged shapes
// =============================================================================================================================
TEST_CASE("shape: ragged() builds [n, d1] and [n, d1, d2], and a wrong vector length throws", "[shape][init]") {
    const auto r2 = cs::init::ragged(3, tis::Sizes{3, 0, 4});
    CHECK(r2.get().rank == 2);
    CHECK(r2.get().extents[0] == 3);
    CHECK(r2.get().per[1]);
    CHECK(r2.get().per_index[1] == tis::Sizes{3, 0, 4});
    CHECK(r2.get().elements() == 7);

    const auto r3 = cs::init::ragged(2, tis::Sizes{2, 3}, 4);  // blocks 2 x 4 and 3 x 4
    CHECK(r3.get().rank == 3);
    CHECK_FALSE(r3.get().per[2]);
    CHECK(r3.get().extents[2] == 4);
    CHECK(r3.get().elements() == 20);

    const auto r33 = cs::init::ragged(2, tis::Sizes{2, 3}, tis::Sizes{5, 1});  // blocks 10 and 3
    CHECK(r33.get().elements() == 13);

    const auto dense = cs::init::ragged(4, 3);  // no per-index vector: a dense [4, 3]
    CHECK_FALSE(dense.get().ragged());
    CHECK(dense.get().elements() == 12);

    CHECK_THROWS_WITH(cs::init::ragged(3, tis::Sizes{1, 2}),
                      Catch::Matchers::ContainsSubstring("dimension 1 has 2 extents but dimension 0 has 3 indices"));
    CHECK_THROWS_AS(cs::init::ragged(3, tis::Sizes{1, 2}), std::invalid_argument);
    CHECK_THROWS_WITH(cs::init::ragged(2, 4, tis::Sizes{1, 2, 3}),
                      Catch::Matchers::ContainsSubstring("dimension 2 has 3 extents but dimension 0 has 2 indices"));
    CHECK_NOTHROW(cs::init::ragged(0, tis::Sizes{}));  // n = 0 is legal
}

TEST_CASE("shape: a Shape is the vector or the list, spelled as the API spells it", "[shape][init]") {
    const cs::init::Shape a{3, 4};
    const cs::init::Shape b = std::vector<std::size_t>{3, 4};
    CHECK(a.get().rank == 2);
    CHECK(a.get().extents == b.get().extents);
    CHECK_THROWS_AS(cs::init::Shape(std::vector<std::size_t>{}), std::invalid_argument);
    CHECK_THROWS_AS(cs::init::Shape(std::vector<std::size_t>{1, 2, 3, 4}), std::invalid_argument);
}

TEST_CASE("shape: ragged from_host needs the sum of the block sizes", "[shape][init]") {
    const auto shape = cs::init::ragged(3, tis::Sizes{3, 0, 4});  // 7 elements
    SECTION("a good size") { CHECK_NOTHROW(cs::init::from_host(std::vector<float>(7, 1.0f), shape)); }
    SECTION("a wrong size") {
        CHECK_THROWS_WITH(cs::init::from_host(std::vector<float>(6, 1.0f), shape),
                          Catch::Matchers::ContainsSubstring("init::from_host has 6 values but its shape has 7 elements"));
        CHECK_THROWS_AS(cs::init::from_host(std::vector<float>(8, 1.0f), shape), std::invalid_argument);
    }
    SECTION("a size-0 leading dimension") {
        const auto empty = cs::init::ragged(0, tis::Sizes{});
        CHECK_NOTHROW(cs::init::from_host(std::vector<float>{}, empty));
        CHECK_THROWS_WITH(cs::init::from_host(std::vector<float>(1, 1.0f), empty),
                          Catch::Matchers::ContainsSubstring("has 1 values but its shape has 0 elements"));
    }
    SECTION("the product for a dense shape") {
        CHECK_NOTHROW(cs::init::from_host(std::vector<float>(6, 1.0f), {2, 3}));
        CHECK_THROWS_AS(cs::init::from_host(std::vector<float>(5, 1.0f), {2, 3}), std::invalid_argument);
    }
}

// =============================================================================================================================
// a free ragged input shaped only by its init
// =============================================================================================================================
TEST_CASE("shape: a free ragged input is shaped only by its init (unchunked)", "[shape][init][ragged]") {
    auto m = cs::create_model<tis::TopFree<tis::InnerPlain>>(default_config(), std::size_t{0}, tis::ragged_init());
    m.setup();
    m.forward();
    m.wait();
    CHECK(tis::sums_of(m) == tis::kDoubled);
    auto v = m.view_ragged_input<DType::f32, 2>("inner.x");
    CHECK(v.size() == 5);
    CHECK(v.extents(1) == tis::kSizes);
    CHECK(v.offsets == std::vector<std::uint64_t>{0, 3, 3, 7, 8, 10});  // the blocks end to end
    CHECK(v.data == tis::iota(10));
    m.forward();  // and again: no second init, same values
    m.wait();
    CHECK(tis::sums_of(m) == tis::kDoubled);
}

TEST_CASE("shape: a free ragged input shaped only by its init, chunked", "[shape][init][ragged][chunked]") {
    for (const std::size_t budget : {std::size_t{1000}, std::size_t{16}, std::size_t{4}}) tis::chunked_matches_plain<tis::TopFree>(budget);
}

TEST_CASE("shape: a ragged init under a leading range", "[shape][init][ragged][chunked][leading_range]") {
    // an outside free ragged input, shaped by its init, read through a connection by a chunked subtree
    for (const std::size_t budget : {std::size_t{1000}, std::size_t{16}, std::size_t{4}}) tis::chunked_matches_plain<tis::TopConn>(budget);
}

TEST_CASE("shape: a ragged init on a tie", "[shape][init][ragged][tie]") {
    auto m = cs::create_model<tis::TieTop>(default_config(), tis::ragged_init());
    m.setup();
    m.forward();
    m.wait();
    for (const char* p : {"a.sums", "b.sums"}) CHECK(cst::to_vector(m.view_output<DType::f32, 1>(p)) == tis::kSums);
    auto v = m.view_ragged_input<DType::f32, 2>("a.x");
    CHECK(v.extents(1) == tis::kSizes);
    CHECK(v.offsets == std::vector<std::uint64_t>{0, 3, 3, 7, 8, 10});
}

TEST_CASE("shape: a ragged init that disagrees with the declared sizes throws before the first run", "[shape][init][ragged]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem<tis::RagSum>("p", 2, tis::Sizes{3, 1, 4});
        s.init_input("p.x", cs::init::from_host(tis::iota(8), cs::init::ragged(2, tis::Sizes{4, 4})));  // two blocks, not three
    });
    m.setup();
    tis::forward_throws(m, {"init for 'p.x' has dimension 0 = 2", "'p.x' resolved dimension 0 (fixed) = 3"});
}
