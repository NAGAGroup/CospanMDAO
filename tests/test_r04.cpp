// =============================================================================
// tests/test_r04.cpp — r04: flat model metadata (docs/spec/revisions/r04-2026-10-11-flat-model-metadata.md)
// =============================================================================
//
//   lifecycle   structural calls after finalize; Model::add_subsystem after setup; a second setup() resets and old handles
//               throw; a plain cs::System root built from main
//   names       dotted local names; duplicates refused; the root adds no dot
//   ties        a tie of a tie (absorbed)
//   cycles      the cycle warning is decided by leaf ordinals, across systems
//   hooks       placement after the last producer; the ordering error at finalize
//   pushes      a tied-output push with members out of scope
//   depends_on  ordering edge and its declaration error
//   policies    an out-of-scope subtree skips its policies; run_n / do_once / transient; rearm
//   pending     shape known up front, shape unknown lazy, update_input collapsing
//   devices     the initializer device of tied and free inputs
//   fused       a fused group's stage names; get_subsystem on a stage throws
//   views       a view before finalize throws
//   listing     list_inputs / list_outputs

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
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
using cst::to_vector;

namespace tr04 {

constexpr std::size_t kN = 8;

/// `f` must throw with every needle in its message (and nothing else is checked).
template <class F>
void expect_throws_all(F&& f, const std::vector<std::string>& needles) {
    std::string what = "<no exception>";
    try {
        f();
    } catch (const std::exception& e) {
        what = e.what();
    }
    INFO("thrown: " << what);
    CHECK(what != "<no exception>");
    for (const auto& n : needles) CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}

/// std::cerr captured for the lifetime of the object (the cycle warning goes there).
struct CerrCapture {
    std::ostringstream buf;
    std::streambuf*    old;
    CerrCapture() : old{std::cerr.rdbuf(buf.rdbuf())} {}
    ~CerrCapture() { std::cerr.rdbuf(old); }
};

std::vector<float> filled(float v) { return std::vector<float>(kN, v); }
std::vector<float> iota_from_one() {
    std::vector<float> v(kN);
    for (std::size_t i = 0; i < kN; ++i) v[i] = static_cast<float>(i + 1);
    return v;
}
void run(cs::Model& m) {
    m.run();
    m.wait();
}
std::vector<float> out(cs::Model& m, std::string_view path) { return to_vector(cst::synced_output<DType::f32, 1>(m, path)); }

// ---- components -----------------------------------------------------------------------------------

/// y[i] = v (kN elements); v is a free input of extent 1.
class Gen : public cs::Component<Gen> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> v;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& o) { o.y[i] = in.v[std::size_t{0}]; }

    Inputs  inputs;
    Outputs outputs;

private:
    float v_;

public:
    explicit Gen(float v) : v_{v} {}
    void setup() override {
        declare_input(inputs.v, "v", 1);
        init_input("v", cs::init::fill(v_));
        declare_output(outputs.y, "y", static_cast<int>(kN));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

/// y = x + 1; x has kN elements (optionally with an initializer).
class Inc : public cs::Component<Inc> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& o) { o.y[i] = in.x[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

private:
    std::optional<float> x0_;

public:
    explicit Inc(std::optional<float> x0 = std::nullopt) : x0_{x0} {}
    void setup() override {
        declare_input(inputs.x, "x", static_cast<int>(kN));
        declare_output(outputs.y, "y", static_cast<int>(kN));
        if (x0_) init_input("x", cs::init::fill(*x0_));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

/// y = x; both dimensions unspecified: x takes its shape from a connection or a hook, y follows x.
class Copy : public cs::Component<Copy> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& o) { o.y[i] = in.x[i]; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_output(outputs.y, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

/// Writes y[i] = i + 1 for i in [lo, hi) and nothing else; two of them tied together make the whole 1..kN.
class Half : public cs::Component<Half> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> lo, hi;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& o) {
        if (i[0] >= in.lo[std::size_t{0}] && i[0] < in.hi[std::size_t{0}]) o.y[i] = static_cast<float>(i[0] + 1);
    }

    Inputs  inputs;
    Outputs outputs;

private:
    unsigned lo_, hi_;

public:
    Half(unsigned lo, unsigned hi) : lo_{lo}, hi_{hi} {}
    void setup() override {
        declare_input(inputs.lo, "lo", 1);
        init_input("lo", cs::init::fill(lo_));
        declare_input(inputs.hi, "hi", 1);
        init_input("hi", cs::init::fill(hi_));
        declare_output(outputs.y, "y", static_cast<int>(kN));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

// ---- systems -------------------------------------------------------------------------------------
/// A root System whose setup() runs a callable (kept by the model: a second setup() builds it again).
class TopFn : public cs::System {
    std::function<void(cs::System&)> build_;

public:
    explicit TopFn(std::function<void(cs::System&)> b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

/// A System of a Gen "g" and an Inc "c" connected g.y -> c.x.
class Pair : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Gen>("g", 2.0f));
        add_subsystem(cs::subsystem_desc<Inc>("c"));
        connect("g.y", "c.x");
    }
};

cs::Model make(std::function<void(cs::System&)> build) { return cs::create_model<TopFn>(default_config(), std::move(build)); }

}  // namespace tr04

// =============================================================================================
// lifecycle
// =============================================================================================
TEST_CASE("r04 lifecycle: a plain System root is built from main with model.add_subsystem", "[r04][lifecycle]") {
    cs::Model model = cs::create_model<cs::System>(default_config());
    model.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 3.0f));
    model.add_subsystem(cs::subsystem_desc<tr04::Inc>("c"));
    model.setup();
    model.get_subsystem("").connect("g.y", "c.x");  // editable between setup and finalize
    tr04::run(model);
    CHECK(tr04::out(model, "c.y") == tr04::filled(4.0f));
}

TEST_CASE("r04 lifecycle: model.add_subsystem after setup is a logic error", "[r04][lifecycle][errors]") {
    cs::Model model = cs::create_model<cs::System>(default_config());
    model.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 3.0f));
    model.setup();
    CHECK_THROWS_AS(model.add_subsystem(cs::subsystem_desc<tr04::Gen>("h", 1.0f)), std::logic_error);
}

TEST_CASE("r04 lifecycle: structural calls after finalize are logic errors; run, views and update_input are legal", "[r04][lifecycle][errors]") {
    cs::Model model = tr04::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tr04::Pair>("p"));
        s.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 1.0f));
    });
    model.setup();
    model.finalize();
    CHECK_THROWS_AS(model.finalize(), std::logic_error);  // once
    auto root = model.get_subsystem("");
    CHECK_THROWS_AS(root.connect("g.y", "p.c.x"), std::logic_error);
    CHECK_THROWS_AS(root.init_input("p.g.v", cs::init::fill(1.0f)), std::logic_error);
    CHECK_THROWS_AS(root.set_device(cs::detail::default_device()), std::logic_error);
    tr04::run(model);
    CHECK(tr04::out(model, "p.c.y") == tr04::filled(3.0f));
    CHECK_NOTHROW(model.update_input("g.v", cs::init::fill(5.0f)));
    tr04::run(model);
    CHECK(tr04::out(model, "g.y") == tr04::filled(5.0f));
}

TEST_CASE("r04 lifecycle: a second setup() resets the model and old handles throw", "[r04][lifecycle]") {
    cs::Model model = tr04::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tr04::Pair>("p")); });
    model.setup();
    auto old_root = model.get_subsystem("");
    auto old_p    = model.get_subsystem("p");
    CHECK(old_root.list_outputs().size() == 2);
    tr04::run(model);
    CHECK(tr04::out(model, "p.c.y") == tr04::filled(3.0f));

    model.setup();  // reset: a fresh metadata object, the root built again
    tr04::expect_throws_all([&] { (void)old_root.list_outputs(); }, {"stale"});
    tr04::expect_throws_all([&] { (void)old_p.list_inputs(); }, {"stale"});
    auto fresh = model.get_subsystem("");
    CHECK(fresh.list_outputs().size() == 2);
    tr04::run(model);
    CHECK(tr04::out(model, "p.c.y") == tr04::filled(3.0f));
}

// =============================================================================================
// names
// =============================================================================================
TEST_CASE("r04 names: dotted local names resolve by concatenation; the root adds no dot", "[r04][names]") {
    cs::Model model = tr04::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tr04::Gen>("a.b", 2.0f));  // a dot inside a local name is allowed
        s.add_subsystem(cs::subsystem_desc<tr04::Pair>("s"));
        s.add_subsystem(cs::subsystem_desc<tr04::Inc>("d"));
        s.connect("a.b.y", "d.x");  // wiring names are full-name strings, never parsed
    });
    model.setup();
    tr04::run(model);
    CHECK(tr04::out(model, "d.y") == tr04::filled(3.0f));
    CHECK(cs::detail::ModelAccess::subsystem(model, "")->path().empty());
    CHECK(cs::detail::ModelAccess::subsystem(model, "a.b")->path() == "a.b");
    CHECK(cs::detail::ModelAccess::subsystem(model, "s.g")->path() == "s.g");
    CHECK(cs::detail::ModelAccess::instance(model, "s.g.y")->full_path() == "s.g.y");
    CHECK(cs::detail::ModelAccess::instance(model, "a.b.y") != nullptr);
}

TEST_CASE("r04 names: a full name is unique across the model", "[r04][names][errors]") {
    SECTION("a duplicate subsystem") {
        cs::Model model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 1.0f));
            s.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 1.0f));
        });
        tr04::expect_throws_all([&] { model.setup(); }, {"'g' is already a subsystem or alias name"});
    }
    SECTION("an alias that collides with a subsystem") {
        cs::Model model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 1.0f));
            s.promote<DType::f32, 1>("g.y", "g");
        });
        tr04::expect_throws_all([&] { model.setup(); }, {"'g' is already a subsystem or alias name"});
    }
    SECTION("an empty name") {
        cs::Model model = tr04::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tr04::Gen>("", 1.0f)); });
        tr04::expect_throws_all([&] { model.setup(); }, {"must be non-empty"});
    }
}

// =============================================================================================
// ties of ties
// =============================================================================================
TEST_CASE("r04 ties: a tie of a tie is allowed; the inner alias is absorbed and the outer one owns the variable", "[r04][tie]") {
    cs::Model model = tr04::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tr04::Inc>("a"));
        s.add_subsystem(cs::subsystem_desc<tr04::Inc>("b"));
        s.add_subsystem(cs::subsystem_desc<tr04::Inc>("c"));
        s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "ab", cs::shape(tr04::kN));
        s.tie_inputs<DType::f32, 1>({"ab", "c.x"}, "abc", cs::shape(tr04::kN));
    });
    model.setup();
    CHECK(cs::detail::ModelAccess::instance(model, "ab")->absorbed);
    CHECK_FALSE(cs::detail::ModelAccess::instance(model, "abc")->absorbed);
    CHECK(cs::detail::ModelAccess::instance(model, "ab")->meta == cs::detail::ModelAccess::instance(model, "abc")->meta);
    CHECK(cs::detail::ModelAccess::instance(model, "c.x")->meta == cs::detail::ModelAccess::instance(model, "a.x")->meta);
    model.update_input("abc", cs::init::fill(2.0f));
    tr04::run(model);
    for (const char* p : {"a.y", "b.y", "c.y"}) CHECK(tr04::out(model, p) == tr04::filled(3.0f));
}

// =============================================================================================
// cycles
// =============================================================================================
TEST_CASE("r04 cycles: the warning is decided by the leaf ordinals, across systems", "[r04][cycle]") {
    SECTION("a consumer in an earlier system than its producer warns") {
        tr04::CerrCapture cap;
        cs::Model         model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tr04::TopFn>("s1", std::function<void(cs::System&)>{[](cs::System& in) {
                                in.add_subsystem(cs::subsystem_desc<tr04::Inc>("p"));  // reads s2.c.y, which runs after it
                            }}));
            s.add_subsystem(cs::subsystem_desc<tr04::Pair>("s2"));
            s.connect("s2.c.y", "s1.p.x");
            s.init_input("s1.p.x", cs::init::fill(10.0f));
        });
        model.setup();
        CHECK(cap.buf.str().find("this is a cycle") != std::string::npos);
        tr04::run(model);
        CHECK(tr04::out(model, "s1.p.y") == tr04::filled(11.0f));
    }
    SECTION("a free input read before its producer warns") {
        tr04::CerrCapture cap;
        cs::Model         model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tr04::Inc>("p"));  // reads q.y, which runs after it
            s.add_subsystem(cs::subsystem_desc<tr04::Pair>("q"));
            s.connect("q.c.y", "p.x");
            s.init_input("p.x", cs::init::fill(10.0f));
        });
        model.setup();
        CHECK(cap.buf.str().find("this is a cycle") != std::string::npos);
        tr04::run(model);
        CHECK(tr04::out(model, "p.y") == tr04::filled(11.0f));  // the initializer on the first run
        tr04::run(model);
        CHECK(tr04::out(model, "p.y") == tr04::filled(4.0f));   // q.c.y (= 2 + 1) + 1
    }
    SECTION("a correctly ordered connection warns about nothing") {
        tr04::CerrCapture cap;
        cs::Model         model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tr04::Pair>("q"));
            s.add_subsystem(cs::subsystem_desc<tr04::Inc>("p"));
            s.connect("q.c.y", "p.x");
        });
        model.setup();
        CHECK(cap.buf.str().find("cycle") == std::string::npos);
        tr04::run(model);
        CHECK(tr04::out(model, "p.y") == tr04::filled(4.0f));
    }
}

// =============================================================================================
// shape hooks
// =============================================================================================
TEST_CASE("r04 hooks: placed after the last producer of the source; a source produced too late is an error at finalize", "[r04][hook]") {
    SECTION("producer first: the hook sets the shape in time") {
        cs::Model model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tr04::Gen>("p", 1.0f));
            s.add_subsystem(cs::subsystem_desc<tr04::Copy>("c"));
            s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
        });
        model.setup();
        CHECK_NOTHROW(model.finalize());
        tr04::run(model);
        CHECK(cst::synced_input<DType::f32, 1>(model, "c.x").shape()[0] == tr04::kN);
        CHECK(tr04::out(model, "c.y").size() == tr04::kN);
    }
    SECTION("producer after the first holder of the target: finalize throws, and can be retried") {
        cs::Model model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tr04::Copy>("c"));
            s.add_subsystem(cs::subsystem_desc<tr04::Gen>("p", 1.0f));
            s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
        });
        model.setup();
        tr04::expect_throws_all([&] { model.finalize(); }, {"shape hook on 'c.x'", "p.y", "cannot set the shape in time"});
        tr04::expect_throws_all([&] { model.run(); }, {"shape hook on 'c.x'"});  // still not finalized, same error
    }
}

// =============================================================================================
// tied pushes
// =============================================================================================
TEST_CASE("r04 pushes: a tied output is pushed once after its last member; members out of scope are skipped", "[r04][tie][push]") {
    const auto build = [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tr04::Half>("a", 0u, 4u));
        s.add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<tr04::Half>("b", 4u, 8u));
        s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(tr04::kN), cs::TieOutputsPolicy::allow_plain);
        s.add_subsystem(cs::subsystem_desc<tr04::Copy>("c"));
        s.connect("t", "c.x");
    };
    SECTION("one member leaves scope after the first run: the push still happens for the other") {
        cs::Model model = tr04::make(build);
        model.setup();
        tr04::run(model);
        CHECK(tr04::out(model, "c.y") == tr04::iota_from_one());
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "b"));
        tr04::run(model);
        CHECK(tr04::out(model, "c.y") == tr04::iota_from_one());  // a rewrote its half, b's half was kept
    }
    SECTION("both members out of scope: nothing is pushed and nothing throws") {
        cs::Model model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<tr04::Half>("a", 0u, 4u));
            s.add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<tr04::Half>("b", 4u, 8u));
            s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(tr04::kN), cs::TieOutputsPolicy::allow_plain);
        });
        model.setup();
        tr04::run(model);
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "a"));
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "b"));
        CHECK_NOTHROW(tr04::run(model));
        CHECK(tr04::out(model, "t") == tr04::iota_from_one());
    }
}

// =============================================================================================
// depends_on
// =============================================================================================
TEST_CASE("r04 depends_on: declared before finalize, placed at finalize; a later subsystem cannot be depended on", "[r04][depends_on]") {
    cs::Model model = tr04::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tr04::Gen>("a", 1.0f));
        s.add_subsystem(cs::subsystem_desc<tr04::Gen>("b", 2.0f));
        s.depends_on("b", "a");
        s.depends_on("b", "a");  // a duplicate edge is ignored
        s.depends_on("b", "a");
    });
    model.setup();
    tr04::expect_throws_all([&] { model.get_subsystem("").depends_on("a", "b"); }, {"depends_on", "added before"});
    tr04::expect_throws_all([&] { model.get_subsystem("").depends_on("a", "zz"); }, {"depends_on", "'zz' is not a subsystem"});
    tr04::run(model);
    tr04::run(model);
    CHECK(tr04::out(model, "a.y") == tr04::filled(1.0f));
    CHECK(tr04::out(model, "b.y") == tr04::filled(2.0f));
}

// =============================================================================================
// run policies and scope
// =============================================================================================
TEST_CASE("r04 policies: an out-of-scope subtree is skipped whole, its policies included", "[r04][policy]") {
    cs::Model model = tr04::make([](cs::System& s) {
        s.add_subsystem(cs::property_list{cs::property::run_n{1}}, cs::subsystem_desc<tr04::Pair>("s"));
        s.set_run_policy("s.c", cs::property_list{cs::property::run_n{3}});
    });
    model.setup();
    tr04::run(model);
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "s"));
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "s.c"));
    CHECK_NOTHROW(tr04::run(model));
    CHECK_NOTHROW(tr04::run(model));
    // s.c is the only holder of its output and it is out of scope: there is nothing to view
    CHECK_THROWS_AS((model.view_output<DType::f32, 1>("s.c.y")), std::logic_error);
}

TEST_CASE("r04 policies: run_n, do_once and transient on leaves, and rearm", "[r04][policy][rearm]") {
    SECTION("run_n{2}: in scope for two turns") {
        cs::Model model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::property_list{cs::property::run_n{2}}, cs::subsystem_desc<tr04::Gen>("g", 1.0f));
            s.add_subsystem(cs::subsystem_desc<tr04::Inc>("c"));
            s.connect("g.y", "c.x");
        });
        model.setup();
        tr04::run(model);
        CHECK(cs::detail::ModelAccess::in_scope(model, "g"));
        tr04::run(model);
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "g"));
        tr04::run(model);
        CHECK(tr04::out(model, "c.y") == tr04::filled(2.0f));  // c keeps reading g's last value
    }
    SECTION("do_once, then rearm_subsystem runs it once more with the new value") {
        cs::Model model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<tr04::Gen>("g", 1.0f));
            s.add_subsystem(cs::subsystem_desc<tr04::Inc>("c"));
            s.connect("g.y", "c.x");
        });
        model.setup();
        tr04::run(model);
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "g"));
        model.update_input("g.v", cs::init::fill(7.0f));
        tr04::run(model);
        CHECK(tr04::out(model, "c.y") == tr04::filled(2.0f));  // g did not run: the update has no effect on the output yet
        model.rearm_subsystem("g");
        CHECK(cs::detail::ModelAccess::in_scope(model, "g"));
        tr04::run(model);
        CHECK(tr04::out(model, "c.y") == tr04::filled(8.0f));
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "g"));
    }
    SECTION("transient: out of scope after every run, back in for the next") {
        cs::Model model = tr04::make([](cs::System& s) {
            s.add_subsystem(cs::property_list{cs::property::transient{}}, cs::subsystem_desc<tr04::Gen>("g", 1.0f));
            s.add_subsystem(cs::subsystem_desc<tr04::Inc>("c"));
            s.connect("g.y", "c.x");
        });
        model.setup();
        for (int k = 0; k < 3; ++k) {
            tr04::run(model);
            CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "g"));
            CHECK(tr04::out(model, "c.y") == tr04::filled(2.0f));
        }
    }
    SECTION("rearm_subsystem of an unknown name, and before finalize, are errors") {
        cs::Model model = tr04::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 1.0f)); });
        model.setup();
        CHECK_THROWS_AS(model.rearm_subsystem("g"), std::logic_error);  // not finalized yet
        tr04::run(model);
        CHECK_THROWS_AS(model.rearm_subsystem("zz"), std::invalid_argument);
    }
}

// =============================================================================================
// the pending pass
// =============================================================================================
TEST_CASE("r04 pending: a settled shape is materialized at the start of the first run; an unsettled one lazily", "[r04][pending]") {
    cs::Model model = tr04::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tr04::Gen>("p", 1.0f));
        s.add_subsystem(cs::subsystem_desc<tr04::Copy>("c"));
        s.shape_hook("c.x", cs::dim::from_shape("p.y", 0));
    });
    model.setup();
    model.finalize();
    const auto px = cs::detail::ModelAccess::instance(model, "p.y");
    const auto cx = cs::detail::ModelAccess::instance(model, "c.x");
    CHECK_FALSE(px->meta->is_materialized());  // nothing is allocated by finalize
    CHECK(px->meta->shape_resolvable());       // fixed dimensions: settled up front
    CHECK_FALSE(cx->meta->shape_resolvable()); // hooked: left to the first holder
    tr04::run(model);
    CHECK(px->meta->is_materialized());
    CHECK(cx->meta->is_materialized());
}

TEST_CASE("r04 pending: several update_input calls before a run collapse to the last", "[r04][pending]") {
    cs::Model model = tr04::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 1.0f)); });
    model.setup();
    model.update_input("g.v", cs::init::fill(4.0f));
    model.update_input("g.v", cs::init::fill(6.0f));
    tr04::run(model);
    CHECK(tr04::out(model, "g.y") == tr04::filled(6.0f));
    model.update_input("g.v", cs::init::fill(8.0f));
    model.update_input("g.v", cs::init::fill(9.0f));
    tr04::run(model);
    CHECK(tr04::out(model, "g.y") == tr04::filled(9.0f));
}

// =============================================================================================
// the initializer device
// =============================================================================================
TEST_CASE("r04 devices: a tie's initializer runs on the tie's device, a free input's on the device it was declared on", "[r04][devices]") {
    const sycl::device dev = cs::detail::default_device();
    cs::Model          model = tr04::make([&](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tr04::Inc>("a"));
        s.add_subsystem(cs::subsystem_desc<tr04::Inc>("b"));
        s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::shape(tr04::kN), dev);
        s.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 1.0f));
        s.promote<DType::f32, 1>("g.v", "gv");
    });
    model.setup();
    const auto t = cs::detail::ModelAccess::instance(model, "t");
    REQUIRE(t->meta->init_device.has_value());
    CHECK(*t->meta->init_device == dev);
    CHECK(t->meta->init_home() == dev);
    const auto v = cs::detail::ModelAccess::instance(model, "g.v");
    CHECK_FALSE(v->meta->init_device.has_value());
    CHECK(v->meta->init_home() == v->device);  // the free input's own device, unchanged by the promotion "gv"
    CHECK(cs::detail::ModelAccess::instance(model, "gv")->meta == v->meta);
    tr04::run(model);
    CHECK(tr04::out(model, "g.y") == tr04::filled(1.0f));
}

// =============================================================================================
// fused groups
// =============================================================================================
TEST_CASE("r04 fused: a group is one leaf; its stages are named a.b.x but are not subsystems", "[r04][fused]") {
    cs::Model model = tr04::make([](cs::System& s) {
        s.add_subsystem(cst::fuse<cst::FuseMode::par>("grp", cs::subsystem_desc<tr04::Gen>("a", 1.0f), cs::subsystem_desc<tr04::Gen>("b", 2.0f)));
        s.add_subsystem(cs::subsystem_desc<tr04::Inc>("c"));
        s.connect("grp.b.y", "c.x");
    });
    model.setup();
    CHECK_NOTHROW((void)model.get_subsystem("grp"));  // the group is a leaf of the tree
    CHECK(model.get_subsystem("grp").is_leaf());
    tr04::expect_throws_all([&] { (void)model.get_subsystem("grp.a"); }, {"stage of a fused group"});
    CHECK(cs::detail::ModelAccess::instance(model, "grp.a.y") != nullptr);  // the stage's ports are reachable by full name
    CHECK(cs::detail::ModelAccess::instance(model, "grp.b.v") != nullptr);
    tr04::run(model);
    CHECK(tr04::out(model, "grp.a.y") == tr04::filled(1.0f));
    CHECK(tr04::out(model, "c.y") == tr04::filled(3.0f));
    const auto outs = model.get_subsystem("grp").list_outputs();
    CHECK(std::find(outs.begin(), outs.end(), "grp.a.y") != outs.end());
    CHECK(std::find(outs.begin(), outs.end(), "grp.b.y") != outs.end());
}

// =============================================================================================
// views
// =============================================================================================
TEST_CASE("r04 views: a view before finalize is a logic error", "[r04][views][errors]") {
    cs::Model model = tr04::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tr04::Pair>("p")); });
    model.setup();
    tr04::expect_throws_all([&] { (void)model.view_output<DType::f32, 1>("p.c.y"); }, {"view_output", "only after finalize"});
    tr04::expect_throws_all([&] { (void)model.get_output<DType::f32, 1>("p.c.y").view(); }, {"only after finalize"});
    CHECK_THROWS_AS((model.view_output<DType::f32, 1>("p.c.y")), std::logic_error);
    model.finalize();
    CHECK_NOTHROW((void)model.view_output<DType::f32, 1>("p.c.y"));  // the shape is fixed: it materializes the variable
}

// =============================================================================================
// listing
// =============================================================================================
TEST_CASE("r04 listing: list_inputs and list_outputs walk the subtree in declaration order", "[r04][listing]") {
    cs::Model model = tr04::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tr04::Pair>("p"));
        s.add_subsystem(cs::subsystem_desc<tr04::Gen>("g", 1.0f));
        s.promote<DType::f32, 1>("p.c.y", "py");
    });
    model.setup();
    const auto has = [](const std::vector<std::string>& v, const std::string& n) { return std::find(v.begin(), v.end(), n) != v.end(); };
    const auto outs = model.list_outputs();
    CHECK(has(outs, "p.g.y"));
    CHECK(has(outs, "p.c.y"));
    CHECK(has(outs, "g.y"));
    CHECK(has(outs, "py"));
    const auto ins = model.list_inputs();
    CHECK(has(ins, "p.g.v"));
    CHECK(has(ins, "p.c.x"));
    CHECK(has(ins, "g.v"));
    // a subsystem's own handle lists only its subtree
    const auto pouts = model.get_subsystem("p").list_outputs();
    CHECK(has(pouts, "p.g.y"));
    CHECK_FALSE(has(pouts, "g.y"));
    CHECK_FALSE(has(pouts, "py"));  // the alias belongs to the root
    CHECK(model.get_subsystem("g").list_inputs() == std::vector<std::string>{"g.v"});
    CHECK(model.get_subsystem("g").list_outputs() == std::vector<std::string>{"g.y"});
}
