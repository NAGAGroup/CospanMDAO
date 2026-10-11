// Initializers check every dimension of the root they are applied to, and carry ragged shapes.
//
//   cs::init::Shape / ragged(n, d1) / from_host(data, ragged shape)
//   the init's shape is compared with the resolved root on the rank and on EVERY dimension
//   (fixed, hooked, per-index counts element by element), at the full range, and names both sides.
//   Design review, section 10: there are no derived dimensions (FromInputDim, FromInput, ...); ragged is rank 2 only.
//
// Design review, section 2: setup checks wiring rules only and shapes are reactive, so every init shape check fires at the
// first forward (the variable's materialization), not at setup. Initial values are declared with init_input in setup()
// and replaced by Model::update_input; views are taken after the first forward and sync()ed before they are read
// (section 5). The chunked cases (ChunkedSystem, leading ranges) are deleted: ChunkedSystem is dead code.
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
        m.run();
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

/// mode 0: x, w fixed 5.  mode 1: x, w unspecified (shaped by their initializers).  mode 2: x fixed 3, w unspecified
/// (a System shape hook sets it from the value of n). The output takes x's shape in pre_forward.
class Probe : public cs::Component<Probe> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> w;
        cs::Input<DType::u32, 1> n;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& a, const Outputs& o) { o.y[i] = a.x[i] + a.w[i]; }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       mode_;

public:
    explicit Probe(int mode) : mode_{mode} {}
    void setup() override {
        declare_input(inputs.n, "n", std::size_t{1});
        if (mode_ == 0) {
            declare_input(inputs.x, "x", std::size_t{5});
            declare_input(inputs.w, "w", std::size_t{5});
        } else if (mode_ == 1) {
            declare_input(inputs.x, "x", cs::unspecified);
            declare_input(inputs.w, "w", cs::unspecified);
        } else {
            declare_input(inputs.x, "x", std::size_t{3});
            declare_input(inputs.w, "w", cs::unspecified);
        }
        declare_output(outputs.y, "y", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};

// ---- RagSum: sums[i] = the sum of block i of a ragged [N, c_i] input ----------------------------------------------

/// mode 0: x unspecified in both dims (a connection or an init shapes it).  mode 1: the same, for a tie.
/// mode 2: x declared with cs::counts(sizes) (N = sizes.size()). The output has one sum per index.
class RagSum : public cs::Component<RagSum> {
public:
    struct Inputs {
        cs::RaggedInput<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    static void compute_impl(sycl::id<1> id, const Inputs& i, const Outputs& o) {
        auto  b = i.x.block(id[0]);
        float s = 0.0f;
        for (std::size_t q = 0; q < b.size(); ++q) s += b[q];
        o.sums[id] = s;
    }

    Inputs  inputs;
    Outputs outputs;

private:
    int                            mode_;
    Sizes                          sizes_;

public:
    RagSum(int mode, Sizes sizes) : mode_{mode}, sizes_{std::move(sizes)} {}
    void setup() override {
        if (mode_ == 0 || mode_ == 1)
            declare_input(inputs.x, "x");
        else
            declare_input(inputs.x, "x", cs::counts(sizes_));
        declare_output(outputs.sums, "sums", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.sums, shape_of(inputs.x).n); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.sums.shape()));
    }
};

// ---- Dbl: out = 2 in ---------------------------------------------------------------------------------------------------

class Dbl : public cs::Component<Dbl> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& a, const Outputs& o) { o.out[i] = 2.0f * a.in[i]; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.in, "in", cs::unspecified);
        declare_output(outputs.out, "out", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.out, shape_of(inputs.in)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(outputs.out.shape())); }
};

// ---- the subtree under test, a plain System ----------------------------------------------------------------------------
/// p (mode 0) -> d, with p's input promoted as "x" and d's output as "sums".
class Inner : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<RagSum>("p", 0, Sizes{}));
        add_subsystem(cs::subsystem_desc<Dbl>("d"));
        connect("p.sums", "d.in");
        promote<DType::f32, 2>("p.x", "x");
        promote<DType::f32, 1>("d.out", "sums");
    }
};

/// The free input is the subtree's own promoted input, shaped by this init.
class TopFree : public cs::System {
    cs::InitSpec init_;

public:
    explicit TopFree(cs::InitSpec init) : init_{std::move(init)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Inner>("inner"));
        init_input("inner.x", init_);
    }
};
/// Two members tied into "t", shaped by an init on the tie.
class TieTop : public cs::System {
    cs::InitSpec init_;

public:
    explicit TieTop(cs::InitSpec init) : init_{std::move(init)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<RagSum>("a", 1, Sizes{}));
        add_subsystem(cs::subsystem_desc<RagSum>("b", 1, Sizes{}));
        tie_inputs<DType::f32, 2>({"a.x", "b.x"}, "t");
        init_input("t", init_);
    }
};

/// A synced host copy of a rank-1 output (the model has run).
std::vector<float> out1(cs::Model& m, const char* path) { return cst::to_vector(cst::synced_output<DType::f32, 1>(m, path)); }

std::vector<float> iota(std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i);
    return v;
}

// sizes {3, 0, 4, 1, 2} over iota(10): blocks [0 1 2] [] [3 4 5 6] [7] [8 9]
const Sizes              kSizes{3, 0, 4, 1, 2};
const std::vector<float> kSums{3.0f, 0.0f, 18.0f, 7.0f, 17.0f};
const std::vector<float> kDoubled{6.0f, 0.0f, 36.0f, 14.0f, 34.0f};  // what the subtree's "sums" holds

cs::InitSpec ragged_init() { return cs::init::from_host(iota(10), cs::init::ragged(5, kSizes)); }

/// The ragged view `v` (synced) holds the five blocks of kSizes over iota(10), end to end.
template <class View>
void check_blocks(const View& v) {
    CHECK(v.n() == 5);
    CHECK(v.block_offsets() == std::vector<std::uint64_t>{0, 3, 3, 7, 8, 10});  // the blocks end to end
    for (std::size_t i = 0; i < kSizes.size(); ++i) CHECK(v.block(i).size() == kSizes[i]);
    CHECK(std::vector<float>(v.span().begin(), v.span().end()) == iota(10));
}

}  // namespace tis

// =============================================================================================================================
// every dimension is checked, and the message names both sides (at the first forward, section 2)
// =============================================================================================================================
TEST_CASE("shape: an init that disagrees with a FIXED dim names both sides", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tis::Probe>("p", 0));
        s.init_input("p.x", cs::init::fill(1.0f, {4}));
    });
    CHECK_NOTHROW(m.setup());  // the check fires when the variable is materialized (section 2)
    tis::forward_throws(m, {"init for 'p.x' has dimension 0 = 4", "'p.x' resolved dimension 0 (fixed) = 5"});
}

// The old INHERITED case (w = FromInputDim{x, 0}, an init on w disagreeing with the dim it inherits) is DELETED: derived
// dimensions are removed (design review, section 10); w would simply be shaped by its own initializer.
TEST_CASE("shape: an init that disagrees with a HOOKED dim names both sides", "[shape][init]") {
    // was: a VALUE-DRIVEN dim (FromInput{n, 0}); a size from an upstream value is a shape hook now
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tis::Probe>("p", 2));
        s.init_input("p.n", cs::init::from_host(std::vector<std::uint32_t>{3}));  // w has 3 elements
        s.shape_hook("p.w", cs::dim::from_value("p.n", 0));
        s.init_input("p.w", cs::init::fill(1.0f, {5}));
    });
    m.setup();
    tis::forward_throws(m, {"init for 'p.w' has dimension 0 = 5", "'p.w' resolved dimension 0 (set by a shape hook) = 3"});
}

TEST_CASE("shape: an init that disagrees with a PER-INDEX dim names both sides, element by element", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tis::RagSum>("p", 2, tis::Sizes{3, 1, 4}));
        s.init_input("p.x", cs::init::from_host(std::vector<float>(7, 1.0f), cs::init::ragged(3, tis::Sizes{3, 0, 4})));
    });
    m.setup();
    tis::forward_throws(m, {"init for 'p.x' has dimension 1 = [3,0,4]", "'p.x' resolved dimension 1 (fixed counts [3,1,4]) = [3,1,4]"});
}

TEST_CASE("shape: an init that agrees with a per-index dim runs", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tis::RagSum>("p", 2, tis::Sizes{3, 1, 4}));
        s.init_input("p.x", cs::init::from_host(tis::iota(8), cs::init::ragged(3, tis::Sizes{3, 1, 4})));
    });
    m.setup();
    m.run();
    m.wait();
    // by hand: [0 1 2] [3] [4 5 6 7]
    CHECK(tis::out1(m, "p.sums") == std::vector<float>{3.0f, 3.0f, 22.0f});
}

TEST_CASE("shape: a rank mismatch names both sides", "[shape][init]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tis::Probe>("p", 0));
        s.init_input("p.x", cs::init::fill(1.0f, {2, 3}));
    });
    m.setup();
    tis::forward_throws(m, {"init for 'p.x' has rank 2", "'p.x' resolved rank 1"});
}

TEST_CASE("shape: an init with no shape still needs every dim specified; thrown at the first forward", "[shape][init]") {
    // Design review, section 2: this was a setup error (check_ties); setup checks wiring rules only now and the shape is
    // reactive. variable.hpp's shape_unknown does not carry the old "and no initializer that gives it a shape" tail.
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tis::Probe>("p", 1));
        s.init_input("p.x", cs::init::fill(1.0f));
    });
    CHECK_NOTHROW(m.setup());
    tis::forward_throws(m, {"its shape is not known yet"});
}

TEST_CASE("shape: bare data (from_host(vec)) carries no shape for a rank-2 root", "[shape][init]") {
    // from_host(vec) describes a rank-1 root; a fully specified rank-2 root takes it by size alone, as it always did
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tis::RagSum>("p", 2, tis::Sizes{2, 2}));
        s.init_input("p.x", cs::init::from_host(tis::iota(4)));
    });
    m.setup();
    m.run();
    m.wait();
    CHECK(tis::out1(m, "p.sums") == std::vector<float>{1.0f, 5.0f});  // [0 1] [2 3]
}

// =============================================================================================================================
// ragged shapes
// =============================================================================================================================
TEST_CASE("shape: ragged() builds [n, d1] (rank 2 only), and a wrong vector length throws", "[shape][init]") {
    const auto r2 = cs::init::ragged(3, tis::Sizes{3, 0, 4});
    CHECK(r2.get().rank == 2);
    CHECK(r2.get().extents[0] == 3);
    CHECK(r2.get().per);
    CHECK(r2.get().per_index == tis::Sizes{3, 0, 4});
    CHECK(r2.get().elements() == 7);
    // the rank-3 forms (ragged(n, sizes, d2), ragged(n, sizes, sizes)) are deleted: ragged is rank 2 only (section 10)

    const auto dense = cs::init::ragged(4, 3);  // no per-index vector: a dense [4, 3]
    CHECK_FALSE(dense.get().ragged());
    CHECK(dense.get().elements() == 12);

    CHECK_THROWS_WITH(cs::init::ragged(3, tis::Sizes{1, 2}),
                      Catch::Matchers::ContainsSubstring("the counts have 2 entries but dimension 0 has 3 indices"));
    CHECK_THROWS_AS(cs::init::ragged(3, tis::Sizes{1, 2}), std::invalid_argument);
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
TEST_CASE("shape: a free ragged input is shaped only by its init", "[shape][init][ragged]") {
    auto m = cs::create_model<tis::TopFree>(default_config(), tis::ragged_init());
    m.setup();
    m.run();
    m.wait();
    CHECK(tis::out1(m, "inner.sums") == tis::kDoubled);
    auto v = m.view_ragged_input<DType::f32, 2>("inner.x");
    v.sync();
    tis::check_blocks(v);
    m.run();  // and again: no second init, same values
    m.wait();
    CHECK(tis::out1(m, "inner.sums") == tis::kDoubled);
}

TEST_CASE("shape: a ragged init on a tie", "[shape][init][ragged][tie]") {
    auto m = cs::create_model<tis::TieTop>(default_config(), tis::ragged_init());
    m.setup();
    m.run();
    m.wait();
    for (const char* p : {"a.sums", "b.sums"}) CHECK(tis::out1(m, p) == tis::kSums);
    auto v = m.view_ragged_input<DType::f32, 2>("a.x");
    v.sync();
    tis::check_blocks(v);
}

TEST_CASE("shape: a ragged init that disagrees with the declared sizes throws at the first forward", "[shape][init][ragged]") {
    auto m = tis::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tis::RagSum>("p", 2, tis::Sizes{3, 1, 4}));
        s.init_input("p.x", cs::init::from_host(tis::iota(8), cs::init::ragged(2, tis::Sizes{4, 4})));  // two blocks, not three
    });
    CHECK_NOTHROW(m.setup());
    tis::forward_throws(m, {"init for 'p.x' has dimension 0 = 2", "'p.x' resolved dimension 0 (fixed) = 3"});
}
