// The shape model (internal/design-review-2026-10-03.md, section 10), end to end.
//
//   1. definition merge at setup   connect / tie merge the shape DEFINITIONS (fixed, unspecified) and throw at setup when two
//                                  fixed values differ; promote carries no shape
//   2. set_shape checks            a producer sets an output in prepare_forward, checked against the definition at the forward;
//                                  a tied output is set by its tie's System; set_shape outside prepare_forward throws
//   3. shape hooks                 from_shape, from_value, custom, fixed_at_run, a hook on an input tie alias, a hook on an
//                                  output tie alias (atomic members), ragged counts_from / offsets_from; setup errors; replay
//   4. the run point               a hook declared after its target's child still runs before that child
//
// Every fixture is prefixed Sh: test_system.cpp has its own namespace tshape with other classes, and both are in one binary.
// Message needles come from variable.hpp (check_setter, shape_unknown), wiring.hpp (merged_definition, connect, tie_inputs),
// system.hpp (install_hook, compute_hook_slots) and component.hpp (settable_output).

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

namespace tshape {

using Needles = std::vector<std::string>;
using U32s    = std::vector<std::uint32_t>;

// ---- helpers --------------------------------------------------------------------------------------------------------
class ShTop : public cs::System {
    std::function<void(cs::System&)> f_;

public:
    explicit ShTop(std::function<void(cs::System&)> f) : f_{std::move(f)} {}
    void setup() override { f_(*this); }
};
inline cs::Model sh_make(std::function<void(cs::System&)> f) { return cs::create_model<ShTop>(default_config(), std::move(f)); }

inline void sh_check(const std::string& what, const Needles& needles) {
    INFO("thrown: " << what);
    CHECK(what != "<no exception>");
    for (const auto& n : needles) CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}
/// Building the model and running setup() throws; the message holds every needle.
inline void sh_setup_throws(const std::function<void(cs::System&)>& build, const Needles& needles) {
    std::string what = "<no exception>";
    try {
        auto m = sh_make(build);
        m.setup();
    } catch (const std::exception& e) {
        what = e.what();
    }
    sh_check(what, needles);
}
/// setup() passes, the first forward throws; the message holds every needle.
inline void sh_forward_throws(cs::Model& m, const Needles& needles) {
    std::string what = "<no exception>";
    try {
        m.run();
        m.wait();
    } catch (const std::exception& e) {
        what = e.what();
    }
    try {
        m.wait();  // earlier components of the failed forward were submitted
    } catch (...) {
    }
    sh_check(what, needles);
}

inline std::vector<float> sh_iota(std::size_t n, float scale = 1.0f) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = scale * static_cast<float>(i + 1);
    return v;
}

// ---- ShProd: y[i] = i + 1; y is declared `declared` (>= 0: fixed) or unspecified; the producer sets n elements ------------------
/// ShProd's n: the producer passes cs::as_is (its dimension comes from the definition or a hook).
inline constexpr std::size_t kAsIs = static_cast<std::size_t>(-1);

/// n: the number it sets (kAsIs: cs::as_is). A producer sets a number only on an UNSPECIFIED dimension, so with a declared
/// (fixed) y it sets nothing unless `force` asks for the refused call.
class ShProd : public cs::Component<ShProd> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& o) { o.y[i] = static_cast<float>(i[0] + 1); }

    Inputs  inputs;
    Outputs outputs;

private:
    std::size_t n_;
    int         declared_;
    bool        force_;

public:
    explicit ShProd(std::size_t n, int declared = -1, bool force = false) : n_{n}, declared_{declared}, force_{force} {}
    void declare() override {
        if (declared_ >= 0)
            declare_output(outputs.y, "y", declared_);
        else
            declare_output(outputs.y, "y", cs::unspecified);
    }
    void prepare_forward() override {
        if (n_ == kAsIs)
            set_shape(outputs.y, cs::as_is);
        else if (declared_ < 0 || force_)
            set_shape(outputs.y, n_);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

// ---- ShBad: calls set_shape in compute_forward, outside prepare_forward ---------------------------------------------------------
class ShBad : public cs::Component<ShBad> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& o) { o.y[i] = static_cast<float>(i[0] + 1); }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.y, "y", cs::unspecified); }
    void prepare_forward() override { set_shape(outputs.y, 3); }  // legal: the output has its shape when compute_forward runs
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        set_shape(outputs.y, 3);  // illegal here
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};

// ---- ShScale: y = 2 x; x fixed (>= 0) or unspecified, y follows x -------------------------------------------------------

class ShScale : public cs::Component<ShScale> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i]; }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       fixed_;

public:
    explicit ShScale(int fixed = -1) : fixed_{fixed} {}
    void declare() override {
        if (fixed_ >= 0)
            declare_input(inputs.x, "x", fixed_);
        else
            declare_input(inputs.x, "x", cs::unspecified);
        declare_output(outputs.y, "y", cs::unspecified);
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

// ---- ShScale2: y = 2 x, rank 2; each x dimension fixed (>= 0) or unspecified ------------------------------------------------

class ShScale2 : public cs::Component<ShScale2> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    static void compute_forward(sycl::id<2> ij, const Inputs& in, const Outputs& out) { out.y[ij] = 2.0f * in.x[ij]; }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       d0_, d1_;

public:
    explicit ShScale2(int d0 = -1, int d1 = -1) : d0_{d0}, d1_{d1} {}
    void declare() override {
        if (d0_ >= 0 && d1_ >= 0)
            declare_input(inputs.x, "x", d0_, d1_);
        else if (d0_ >= 0)
            declare_input(inputs.x, "x", d0_, cs::unspecified);
        else if (d1_ >= 0)
            declare_input(inputs.x, "x", cs::unspecified, d1_);
        else
            declare_input(inputs.x, "x", cs::unspecified_shape<2>());
        declare_output(outputs.y, "y", cs::unspecified_shape<2>());
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0], shape_of(inputs.x)[1]); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

// ---- ShN: n[0] = v, a one-element u32 output (a value source for from_value) -------------------------------------------------

class ShN : public cs::Component<ShN> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> v;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::u32, 1> n;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.n[i] = static_cast<std::uint32_t>(in.v[std::size_t{0}]); }

    Inputs  inputs;
    Outputs outputs;

private:
    float                     v_;

public:
    explicit ShN(float v) : v_{v} {}
    void declare() override {
        declare_input(inputs.v, "v", 1);
        init_input("v", cs::init::fill(v_)); declare_output(outputs.n, "n", 1); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.n.shape())); }
};

// ---- ShU32: out = raw, a u32 vector given by the host (counts or offsets for a ragged hook) ----------------------------------------

class ShU32 : public cs::Component<ShU32> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> raw;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> out;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& a, const Outputs& o) { o.out[i] = a.raw[i]; }

    Inputs  inputs;
    Outputs outputs;

private:
    U32s                      init_;

public:
    explicit ShU32(U32s init) : init_{std::move(init)} {}
    void declare() override {
        declare_input(inputs.raw, "raw", init_.size());
        init_input("raw", cs::init::from_host(init_));
        declare_output(outputs.out, "out", init_.size());
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.out.shape())); }
};

// ---- ShRagSink: a free ragged input r (rank 2); sums[i] = the size of block i -----------------------------------------------------

class ShRagSink : public cs::Component<ShRagSink> {
public:
    struct Inputs {
        cs::RaggedInput<DType::f32, 2> r;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    static void compute_forward(sycl::id<1> id, const Inputs& in, const Outputs& out) { out.sums[id] = static_cast<float>(in.r.block(id[0]).size()); }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.r, "r");
        declare_output(outputs.sums, "sums", cs::unspecified);
    }
    void prepare_forward() override { set_shape(outputs.sums, shape_of(inputs.r).n); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.sums.shape())); }
};

// ---- ShZero / ShAdd: atomic tie_outputs members (unspecified), zero then accumulate --------------------------------------------

class ShZero : public cs::Component<ShZero> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> acc;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.acc[i].store(0.0f); }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.acc, "acc", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.acc.shape())); }
};

class ShAdd : public cs::Component<ShAdd> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> v;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> acc;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.acc[i].fetch_add(in.v[std::size_t{0}]); }

    Inputs  inputs;
    Outputs outputs;

private:
    float                            v_;

public:
    explicit ShAdd(float v) : v_{v} {}
    void declare() override {
        declare_input(inputs.v, "v", 1);
        init_input("v", cs::init::fill(v_)); declare_output(outputs.acc, "acc", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.acc.shape())); }
};

// ---- ShCube: a rank-3 producer, y [?, ?, 3]; `mode` picks which set_shape call it makes ---------------------------------------
// 0: set_shape(y, as_is, 7, as_is)     the legal one: dimension 1 is the producer's
// 1: set_shape(y, 5, 7, as_is)         a number on dimension 0 (hook-owned in the tests that use it)
// 2: set_shape(y, as_is, 7, 3)         a number on the declared dimension 2, even though it matches
// 3: set_shape(y, as_is, 7)            two entries for a rank-3 output

class ShCube : public cs::Component<ShCube> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 3> y;
    };
    static void compute_forward(sycl::id<3> i, const Inputs&, const Outputs& o) { o.y[i] = static_cast<float>(i[0] * 100 + i[1] * 10 + i[2]); }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       mode_;

public:
    explicit ShCube(int mode = 0) : mode_{mode} {}
    void declare() override { declare_output(outputs.y, "y", cs::unspecified, cs::unspecified, 3); }
    void prepare_forward() override {
        switch (mode_) {
            case 0: set_shape(outputs.y, cs::as_is, 7, cs::as_is); break;
            case 1: set_shape(outputs.y, 5, 7, cs::as_is); break;
            case 2: set_shape(outputs.y, cs::as_is, 7, 3); break;
            default: set_shape(outputs.y, cs::as_is, 7); break;
        }
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

// ---- ShRagGen: a ragged producer [N, c_i]: set_shape(pts, as_is, counts), the n left to a hook ---------------------------------

class ShRagGen : public cs::Component<ShRagGen> {
public:
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> pts;
    };
    static void compute_forward(sycl::id<1> id, const Inputs&, const Outputs& o) {
        auto b = o.pts.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = static_cast<float>(id[0] * 10 + q);
    }

    Inputs  inputs;
    Outputs outputs;

private:
    std::vector<std::size_t>        counts_;

public:
    explicit ShRagGen(std::vector<std::size_t> counts) : counts_{std::move(counts)} {}
    void declare() override { declare_output(outputs.pts, "pts"); }
    void prepare_forward() override { set_shape(outputs.pts, cs::as_is, counts_); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{counts_.size()}));
    }
};

}  // namespace tshape

// =============================================================================================================================
// 1. definition merge at setup
// =============================================================================================================================
TEST_CASE("shape definitions: connecting fixed 4 into fixed 5 throws at connect", "[shapes][definition]") {
    tshape::sh_setup_throws(
        [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{4}, 4));
            s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c", 5));
            s.connect("p.y", "c.x");
        },
        {"dimension 0", "declares 4", "declares 5", "the extents must match"});
}

TEST_CASE("shape definitions: an unspecified dimension takes the fixed one it meets", "[shapes][definition]") {
    SECTION("a fixed output feeds an unspecified input") {
        auto m = tshape::sh_make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{4}, 4));
            s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));
            s.connect("p.y", "c.x");
        });
        m.setup();
        m.run();
        m.wait();
        CHECK(cst::synced_input<DType::f32, 1>(m, "c.x").shape()[0] == 4);
        CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "c.y")) == tshape::sh_iota(4, 2.0f));
    }
    SECTION("an unspecified output feeds a fixed input: the dimension is 4 and comes from the definition (as_is)") {
        auto m = tshape::sh_make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", tshape::kAsIs));
            s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c", 4));
            s.connect("p.y", "c.x");
        });
        m.setup();
        m.run();
        m.wait();
        CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "c.y")) == tshape::sh_iota(4, 2.0f));
    }
    SECTION("an unspecified output feeds a fixed input: a producer that sets a number throws at the forward, even 4") {
        for (const std::size_t n : {std::size_t{3}, std::size_t{4}}) {
            auto m = tshape::sh_make([n](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", n));
                s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c", 4));
                s.connect("p.y", "c.x");
            });
            m.setup();
            tshape::sh_forward_throws(m, {"dimension 0 is declared as 4", "takes a number only on an unspecified dimension"});
        }
    }
}

TEST_CASE("shape definitions: tie members fixed on different dimensions merge", "[shapes][definition][tie]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale2>("a", 4, -1));  // fixes dimension 0
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale2>("b", -1, 3));  // fixes dimension 1
        s.tie_inputs<DType::f32, 2>({"a.x", "b.x"}, "t");
        s.init_input("t", cs::init::fill(1.0f, {4, 3}));
    });
    m.setup();
    m.run();
    m.wait();
    const auto ax = cst::synced_input<DType::f32, 2>(m, "a.x");
    CHECK(ax.shape()[0] == 4);
    CHECK(ax.shape()[1] == 3);
    CHECK(cst::to_vector(cst::synced_output<DType::f32, 2>(m, "b.y")) == std::vector<float>(12, 2.0f));
}

TEST_CASE("shape definitions: a conflicting tie spec throws at tie_inputs", "[shapes][definition][tie]") {
    tshape::sh_setup_throws(
        [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("a", 4));
            s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("b"));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::shape(6));
        },
        {"tie_inputs", "dimension 0", "the extents must match"});
}

TEST_CASE("shape definitions: promote carries no shape", "[shapes][definition][promote]") {
    // promote<DT, R>(path, name) takes no ShapeSpec: the promoted name is the same variable with the member's own definition.
    // An unspecified output promoted and then connected to a fixed-5 input stays unspecified until it meets it; the connect
    // gives it 5.
    class Inner : public cs::System {
    public:
        void setup() override {
            add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", tshape::kAsIs));  // the 5 comes from the connected input
            promote<DType::f32, 1>("p.y", "y");
        }
    };
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<Inner>("in"));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c", 5));
        s.connect("in.y", "c.x");
    });
    m.setup();
    m.run();
    m.wait();
    CHECK(cst::synced_output<DType::f32, 1>(m, "in.y").shape()[0] == 5);
    CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "c.y")) == tshape::sh_iota(5, 2.0f));
}

// =============================================================================================================================
// 2. set_shape checks
// =============================================================================================================================
TEST_CASE("set_shape: a number on a declared-fixed output dimension throws at the call, even when it matches", "[shapes][set_shape]") {
    for (const std::size_t n : {std::size_t{3}, std::size_t{4}}) {
        auto m = tshape::sh_make([n](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", n, 4, true)); });
        CHECK_NOTHROW(m.setup());
        tshape::sh_forward_throws(m, {"dimension 0 is declared as 4", "takes a number only on an unspecified dimension", "'p.y'"});
    }
}

TEST_CASE("set_shape: on a tied output throws the tied message", "[shapes][set_shape][tie]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("a", std::size_t{4}));
        s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("b", std::size_t{4}));
        s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(4), cs::TieOutputsPolicy::allow_plain);
    });
    CHECK_NOTHROW(m.setup());
    tshape::sh_forward_throws(m, {"is tied; its shape is set by its tie's System (a shape hook), not by a member"});
}

TEST_CASE("set_shape: outside prepare_forward throws", "[shapes][set_shape]") {
    auto m = tshape::sh_make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tshape::ShBad>("p")); });
    CHECK_NOTHROW(m.setup());
    tshape::sh_forward_throws(m, {"set_shape", "legal inside prepare_forward() only"});
}

// =============================================================================================================================
// 3. shape hooks
// =============================================================================================================================
TEST_CASE("hook from_shape: an input takes a dimension of an upstream output's shape", "[shapes][hook]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{6}));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));
        s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
    });
    m.setup();
    m.run();
    m.wait();
    CHECK(cst::synced_input<DType::f32, 1>(m, "c.x").shape()[0] == 6);
    CHECK(cst::synced_output<DType::f32, 1>(m, "c.y").shape()[0] == 6);
}

TEST_CASE("hook from_value: two dimensions from two different sources", "[shapes][hook]") {
    // (shape_hook.hpp batches the reads into one host wait; that is not observable here, so only the result is checked.)
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("na", 3.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nb", 5.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale2>("c"));
        s.shape_hook("c.x", cs::dim::from_value("na.n", 0), cs::dim::from_value("nb.n", 0));
    });
    m.setup();
    m.run();
    m.wait();
    const auto x = cst::synced_input<DType::f32, 2>(m, "c.x");
    CHECK(x.shape()[0] == 3);
    CHECK(x.shape()[1] == 5);
    CHECK(cst::synced_output<DType::f32, 2>(m, "c.y").shape()[0] == 3);
    CHECK(cst::synced_output<DType::f32, 2>(m, "c.y").shape()[1] == 5);
}

TEST_CASE("hook from_value: a source that is gone leaves its dimension alone; the other still follows its source", "[shapes][hook][r04]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nb", 5.0f));
        // na is added after nb, so the hook runs after na's turn (and before its do_once dearms it on run 1)
        s.add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<tshape::ShN>("na", 3.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale2>("c"));
        s.shape_hook("c.x", cs::dim::from_value("na.n", 0), cs::dim::from_value("nb.n", 0));
    });
    m.setup();
    const auto shape_is = [&](std::size_t d0, std::size_t d1) {
        const auto x = cst::synced_input<DType::f32, 2>(m, "c.x");
        CHECK(x.shape()[0] == d0);
        CHECK(x.shape()[1] == d1);
    };
    m.run();
    m.wait();
    shape_is(3, 5);
    m.run();  // na ran its one turn: na.n is gone, dimension 0 keeps 3
    m.wait();
    shape_is(3, 5);
    m.update_input("nb.v", cs::init::fill(7.0f));
    m.run();  // dimension 1 still follows nb.n
    m.wait();
    shape_is(3, 7);
}

TEST_CASE("hook custom: a function of the hook context", "[shapes][hook]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{4}));
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("k", 3.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));
        // 4 * 3: a dimension of a shape times a value
        s.shape_hook("c.x", cs::dim::custom([](cs::ShapeHookCtx& ctx) { return ctx.shape_of("p.y").extents[0] * ctx.value_of("k.n", 0); }));
    });
    m.setup();
    m.run();
    m.wait();
    CHECK(cst::synced_input<DType::f32, 1>(m, "c.x").shape()[0] == 12);
}

TEST_CASE("hook fixed_at_run: the dimension becomes specified, so a conflicting fixed connect throws at setup", "[shapes][hook]") {
    const auto build = [](int consumer) {
        return [consumer](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("a", std::size_t{5}));
            s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("b", std::size_t{5}));
            s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c", consumer));
            s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(), cs::TieOutputsPolicy::allow_plain);
            s.shape_hook("t", cs::dim::fixed_at_run(5));
            s.connect("t", "c.x");
        };
    };
    SECTION("conflicting: the input declares 7") {
        tshape::sh_setup_throws(build(7), {"connect", "dimension 0", "the extents must match"});
    }
    SECTION("agreeing: the input declares 5") {
        auto m = tshape::sh_make(build(5));
        CHECK_NOTHROW(m.setup());
    }
}

TEST_CASE("hook on an input tie alias", "[shapes][hook][tie]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nn", 7.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("a"));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("b"));
        s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t");
        s.shape_hook("t", cs::dim::from_value("nn.n", 0));
        s.init_input("t", cs::init::fill(1.5f));
    });
    m.setup();
    m.run();
    m.wait();
    CHECK(cst::synced_input<DType::f32, 1>(m, "a.x").shape()[0] == 7);
    CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "a.y")) == std::vector<float>(7, 3.0f));
    CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "b.y")) == std::vector<float>(7, 3.0f));
}

TEST_CASE("hook on an output tie alias whose members are atomic Adders", "[shapes][hook][tie][atomic]") {
    // The output alias is sized only by the hook (no shape on the tie). The Adders carry no set_shape (a tied output cannot)
    // and are ordered after the zeroing writer by depends_on (design review, section 3).
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nn", 6.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShZero>("z"));
        s.add_subsystem(cs::subsystem_desc<tshape::ShAdd>("a", 1.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShAdd>("b", 2.0f));
        s.tie_outputs<DType::atomic_f32, 1>({"z.acc", "a.acc", "b.acc"}, "sum");  // require_atomic: all are atomic
        s.shape_hook("sum", cs::dim::from_value("nn.n", 0));
        s.depends_on("a", "z");
        s.depends_on("b", "z");
    });
    m.setup();
    for (int r = 0; r < 2; ++r) {
        INFO("forward " << r + 1);
        m.run();
        m.wait();
        CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "sum")) == std::vector<float>(6, 3.0f));  // 0 + 1 + 2
    }
}

TEST_CASE("hook ragged counts_from and offsets_from: n(), block_offsets() and total", "[shapes][hook][ragged]") {
    const tshape::U32s counts{2, 0, 3, 1};
    const tshape::U32s offsets{0, 2, 2, 5, 6};
    const std::vector<std::uint64_t> want_offs{0, 2, 2, 5, 6};
    const auto                       run = [&](bool by_offsets) {
        auto m = tshape::sh_make([&](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::ShU32>("src", by_offsets ? offsets : counts));
            s.add_subsystem(cs::subsystem_desc<tshape::ShRagSink>("c"));
            if (by_offsets)
                s.shape_hook("c.r", cs::ragged::offsets_from("src.out"));
            else
                s.shape_hook("c.r", cs::ragged::counts_from("src.out"));
        });
        m.setup();
        m.run();
        m.wait();
        auto v = m.view_ragged_input<DType::f32, 2>("c.r");
        v.sync();
        CHECK(v.n() == 4);
        CHECK(v.block_offsets() == want_offs);
        CHECK(v.block_offsets().back() == 6);  // the total
        for (std::size_t i = 0; i < counts.size(); ++i) CHECK(v.block(i).size() == counts[i]);
        CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "c.sums")) == std::vector<float>{2, 0, 3, 1});
    };
    SECTION("counts_from (rank-1 counts)") { run(false); }
    SECTION("offsets_from") { run(true); }
}

TEST_CASE("hook errors at setup", "[shapes][hook][errors]") {
    SECTION("a target that is an output tie member: hook the alias instead") {
        tshape::sh_setup_throws(
            [](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nn", 3.0f));
                s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("a", tshape::kAsIs));
                s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("b", tshape::kAsIs));
                s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(), cs::TieOutputsPolicy::allow_plain);
                s.shape_hook("a.y", cs::dim::from_value("nn.n", 0));
            },
            {"shape_hook", "'a.y'", "member of a tie_outputs", "hook the tie alias instead", "'t'"});
    }
    SECTION("a target that is already connected") {
        tshape::sh_setup_throws(
            [](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{3}));
                s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));
                s.connect("p.y", "c.x");
                s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
            },
            {"shape_hook", "'c.x'", "it is connected or tied", "exclude each other"});
    }
    SECTION("connect into a hooked input") {
        tshape::sh_setup_throws(
            [](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{3}));
                s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));
                s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
                s.connect("p.y", "c.x");
            },
            {"connect", "'c.x'", "is the target of a shape hook", "exclude each other"});
    }
    SECTION("a hooked dimension that is fixed in the definition") {
        tshape::sh_setup_throws(
            [](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{3}));
                s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c", 4));
                s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
            },
            {"shape_hook", "'c.x'", "dimension 0 is fixed", "a hook can only own an unspecified dimension"});
    }
    SECTION("two hooks on one dimension") {
        tshape::sh_setup_throws(
            [](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{3}));
                s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));
                s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
                s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
            },
            {"shape_hook", "'c.x'", "dimension 0 is set by a shape hook", "only one hook owns it"});
    }
    SECTION("a source produced by a later child is no setup error any more: the hook works or fails when it runs") {
        auto m = tshape::sh_make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));
            s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{3}));  // registered AFTER the target's child
            s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
        });
        CHECK_NOTHROW(m.setup());
    }
}

TEST_CASE("hook replay: a transient target child, rebuilt each forward, still gets its hooked shape", "[shapes][hook][transient]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{6}));
        s.add_subsystem(cs::property_list{cs::property::transient{}}, cs::subsystem_desc<tshape::ShScale>("tr"));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("sink"));
        s.shape_hook("tr.x", cs::dim::from_shape("p.y", 0));
        s.connect("tr.y", "sink.x");
    });
    m.setup();
    for (int r = 0; r < 3; ++r) {
        INFO("forward " << r + 1);
        CHECK_NOTHROW(m.run());
        m.wait();
        REQUIRE(cs::detail::ModelAccess::node(m, "tr") != nullptr);  // r04: the node stays; out of scope after its turn, rearmed next forward
        CHECK_FALSE(cs::detail::ModelAccess::node(m, "tr")->in_scope);
        CHECK(cst::synced_output<DType::f32, 1>(m, "sink.y").shape()[0] == 6);
    }
}

// =============================================================================================================================
// 4. the run point
// =============================================================================================================================
TEST_CASE("hook run point: a hook declared after the target's child still runs before that child", "[shapes][hook][order]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", std::size_t{5}));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));       // the target's child
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("later"));   // a child after it
        s.connect("c.y", "later.x");
        s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));              // declared last, runs before "c"
    });
    m.setup();
    m.run();
    m.wait();
    CHECK(cst::synced_input<DType::f32, 1>(m, "c.x").shape()[0] == 5);
    CHECK(cst::synced_output<DType::f32, 1>(m, "later.y").shape()[0] == 5);
}

// =============================================================================================================================
// 5. P5: per-dimension set_shape, hooks on outputs, wiring anywhere
// =============================================================================================================================
TEST_CASE("set_shape rank 3: a hooked dimension, a producer's dimension and a declared one", "[shapes][set_shape][p5]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nn", 5.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShCube>("p", 0));
        s.shape_hook("p.y", cs::dim::from_value("nn.n", 0), cs::dim::as_is(), cs::dim::as_is());  // a hook on an untied output
    });
    m.setup();
    m.run();
    m.wait();
    const auto y = cst::synced_output<DType::f32, 3>(m, "p.y");
    CHECK(y.shape()[0] == 5);
    CHECK(y.shape()[1] == 7);
    CHECK(y.shape()[2] == 3);
}

TEST_CASE("set_shape: a number on a specified dimension throws at the call", "[shapes][set_shape][p5]") {
    auto m = tshape::sh_make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tshape::ShCube>("p", 2)); });
    CHECK_NOTHROW(m.setup());
    tshape::sh_forward_throws(m, {"dimension 2 is declared as 3", "takes a number only on an unspecified dimension", "'p.y'"});
}

TEST_CASE("set_shape: a number on a hook-owned dimension throws at the call", "[shapes][set_shape][p5]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nn", 5.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShCube>("p", 1));
        s.shape_hook("p.y", cs::dim::from_value("nn.n", 0), cs::dim::as_is(), cs::dim::as_is());
    });
    CHECK_NOTHROW(m.setup());
    tshape::sh_forward_throws(m, {"dimension 0 is owned by a shape hook", "cannot give it a number", "'p.y'"});
}

TEST_CASE("set_shape: an entry count that differs from the rank throws", "[shapes][set_shape][p5]") {
    auto m = tshape::sh_make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tshape::ShCube>("p", 3)); });
    CHECK_NOTHROW(m.setup());
    tshape::sh_forward_throws(m, {"set_shape('p.y')", "one entry per dimension"});
}

TEST_CASE("hook on an untied output: the hook runs before the producer, which passes as_is", "[shapes][hook][p5]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nn", 6.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("p", tshape::kAsIs));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));
        s.connect("p.y", "c.x");
        s.shape_hook("p.y", cs::dim::from_value("nn.n", 0));
    });
    m.setup();
    m.run();
    m.wait();
    CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "c.y")) == tshape::sh_iota(6, 2.0f));
}

TEST_CASE("hook on a tied member output throws and points at the alias", "[shapes][hook][tie][p5]") {
    tshape::sh_setup_throws(
        [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nn", 3.0f));
            s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("a", tshape::kAsIs));
            s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("b", tshape::kAsIs));
            s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(), cs::TieOutputsPolicy::allow_plain);
            s.shape_hook("b.y", cs::dim::from_value("nn.n", 0));
        },
        {"shape_hook", "'b.y'", "hook the tie alias instead", "'t'"});
}

TEST_CASE("shape_hook and depends_on called after setup work", "[shapes][hook][p5]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nn", 4.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("a", std::size_t{2}));
        s.add_subsystem(cs::subsystem_desc<tshape::ShProd>("b", std::size_t{2}));
        s.add_subsystem(cs::subsystem_desc<tshape::ShScale>("c"));
        s.init_input("c.x", cs::init::fill(1.5f));
    });
    m.setup();
    CHECK_NOTHROW(m.get_subsystem("").shape_hook("c.x", cs::dim::from_value("nn.n", 0)));
    CHECK_NOTHROW(m.get_subsystem("").depends_on("b", "a"));
    m.run();
    m.wait();
    CHECK(cst::synced_input<DType::f32, 1>(m, "c.x").shape()[0] == 4);
    CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "c.y")) == std::vector<float>(4, 3.0f));
}

TEST_CASE("ragged set_shape(out, as_is, counts): the n is owned by a hook", "[shapes][set_shape][hook][ragged][p5]") {
    auto m = tshape::sh_make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tshape::ShN>("nn", 4.0f));
        s.add_subsystem(cs::subsystem_desc<tshape::ShRagGen>("g", std::vector<std::size_t>{2, 0, 3, 1}));
        s.shape_hook("g.pts", cs::dim::from_value("nn.n", 0), cs::dim::as_is());  // n alone; the counts are the producer's
    });
    m.setup();
    m.run();
    m.wait();
    auto v = m.view_ragged_output<DType::f32, 2>("g.pts");
    v.sync();
    CHECK(v.n() == 4);
    const std::vector<std::size_t> counts{2, 0, 3, 1};
    for (std::size_t i = 0; i < counts.size(); ++i) CHECK(v.block(i).size() == counts[i]);
}
