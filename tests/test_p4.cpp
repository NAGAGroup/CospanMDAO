// P4 tests: setup, wiring and the System loop (docs/plan-section11.md P4; design review 11.2 and 11.3).
//
//   wiring anywhere     connect, promote, tie_*, init_input after setup; two materialized variables cannot merge
//   run-policy override a grandchild made eternal by its grandparent; applied from pre_forward; not replayed on rearm
//   rearm               legal outside pre_forward
//   cycles              a connect that closes a cycle keeps the target's initializer; no initializer: the first forward throws
//   tie devices         tie_outputs members must be on the tie's device (at the call and at every forward); a subtree moved
//                       by add_subsystem takes its aliases with it; tie_inputs's device is where the alias and its init live
//   no ties of ties     both kinds refused, also through a promotion; promoting a tie works
//   tied-output pushes  the owning System pushes right after the last member, so a consumer after it sees the whole value
//   system loop         precompute never runs a child's precompute; replay skips a connection whose other side is gone
//   merges              a connect after setup moves a holder's container into the winning variable
//
// Every case goes through a cospan::Model. The components live in namespace tp4 (other test files have their own: two
// same-named classes in one executable would be an ODR clash). Cases that need two devices skip without them.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
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

namespace tp4 {

constexpr std::size_t kN = 8;

/// `f` must throw with every needle in its message.
template <class F>
void expect_throws_all(F&& f, const std::vector<std::string>& needles) {
    std::string what = "<no exception>";
    try {
        f();
    } catch (const std::exception& e) {
        what = e.what();
    }
    INFO("thrown: " << what);
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
std::vector<float> in(cs::Model& m, std::string_view path) { return to_vector(cst::synced_input<DType::f32, 1>(m, path)); }

// ---- components ----------------------------------------------------------------------------------

/// y[i] = v, kN elements.
class Gen : public cs::Component<Gen> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> v;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& o) { o.y[i] = in.v[std::size_t{0}]; }

    Inputs  inputs;
    Outputs outputs;

private:
    float                     v_;

public:
    explicit Gen(float v) : v_{v} {}
    void setup() override {
        declare_input(inputs.v, "v", 1);
        init_input("v", cs::init::fill(v_)); declare_output(outputs.y, "y", static_cast<int>(kN)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

/// Writes y[i] = i + 1 for i in [lo, hi) and nothing else: two of them tied together make the whole 1..kN.
class Half : public cs::Component<Half> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> lo, hi;  // a free input of extent 1, initialized from the constructor's value
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
    unsigned                  lo_, hi_;

public:
    Half(unsigned lo, unsigned hi) : lo_{lo}, hi_{hi} {}
    void setup() override {
        declare_input(inputs.lo, "lo", 1);
        init_input("lo", cs::init::fill(lo_));
        declare_input(inputs.hi, "hi", 1);
        init_input("hi", cs::init::fill(hi_)); declare_output(outputs.y, "y", static_cast<int>(kN)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};

/// y = a * x + b, kN elements; `x0` is the initial value of x (a component's own init_input); `pre` counts pre_forward calls.
class Axpb : public cs::Component<Axpb> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> a, b;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& o) { o.y[i] = in.a[std::size_t{0}] * in.x[i] + in.b[std::size_t{0}]; }

    Inputs  inputs;
    Outputs outputs;

private:
    float                     a_, b_;
    std::optional<float>      x0_;
    int*                      pre_;

public:
    Axpb(float a, float b, std::optional<float> x0 = std::nullopt, int* pre = nullptr) : a_{a}, b_{b}, x0_{x0}, pre_{pre} {}
    void setup() override {
        declare_input(inputs.a, "a", 1);
        init_input("a", cs::init::fill(a_));
        declare_input(inputs.b, "b", 1);
        init_input("b", cs::init::fill(b_));
        declare_input(inputs.x, "x", static_cast<int>(kN));
        declare_output(outputs.y, "y", static_cast<int>(kN));
        if (x0_) init_input("x", cs::init::fill(*x0_));
    }
    void pre_forward() override {
        if (pre_ != nullptr) ++*pre_;
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

/// A host body: y = x + 1 on the host.

class HostInc : public cs::Component<HostInc> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(cs::Host, const Inputs& in, const Outputs& o) {
        for (std::size_t i = 0; i < in.x.size(); ++i) o.y[i] = in.x[i] + 1.0f;
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.x, "x", static_cast<int>(kN));
        declare_output(outputs.y, "y", static_cast<int>(kN));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(); }
};

// ---- systems --------------------------------------------------------------------------------------
/// A root System whose setup() runs a callable.
class TopFn : public cs::System {
    std::function<void(cs::System&)> build_;

public:
    explicit TopFn(std::function<void(cs::System&)> b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

/// Built once per construction: how often a child was built or rebuilt.
inline int g_mid_built = 0;

/// A System with one TRANSIENT child c (y = 2 * 3 + 0 = 6).
class Mid : public cs::System {
public:
    Mid() { ++g_mid_built; }
    void setup() override {
        add_subsystem(cs::property_list{cs::property::transient{}}, cs::subsystem_desc<Axpb>("c", 2.0f, 0.0f, std::optional<float>{3.0f}));
    }
};

/// The grandparent of c. mode 0: no override; 1: override "b.c" to eternal in setup; 2: override it from pre_forward on the
/// second forward. `rearm_at`: the forward on whose pre_forward "b" is rearmed (0: never). `post_rearm`: rearm "b" from
/// post_forward on the first forward instead (rearm outside pre_forward).
class OverTop : public cs::System {
    int  mode_, rearm_at_;
    bool post_rearm_;
    int  n_ = 0;

public:
    explicit OverTop(int mode, int rearm_at = 0, bool post_rearm = false) : mode_{mode}, rearm_at_{rearm_at}, post_rearm_{post_rearm} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Mid>("b"));
        if (mode_ == 1) override_run_policy("b.c");
    }
    void pre_forward() override {
        ++n_;
        if (mode_ == 2 && n_ == 2) override_run_policy("b.c", cs::property_list{});
        if (rearm_at_ == n_) rearm("b");
    }
    void post_forward() override {
        if (post_rearm_ && n_ == 1) rearm("b");
    }
};

/// A cycle of two components: p.x reads q.y, which runs after p. mode 0: the initializer is given at the top with init_input
/// after the connections; 1: the target p declares its own initializer (10); 2: none.
class CycleTop : public cs::System {
    int mode_;

public:
    explicit CycleTop(int mode) : mode_{mode} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("p", 1.0f, 1.0f, mode_ == 1 ? std::optional<float>{10.0f} : std::nullopt));
        add_subsystem(cs::subsystem_desc<Axpb>("q", 1.0f, 0.0f));
        connect("p.y", "q.x");
        connect("q.y", "p.x");  // the cycle: p is registered before q
        if (mode_ == 0) init_input("p.x", cs::init::fill(10.0f));
    }
};

/// Counts how many child pre_forwards ran before this System's own, at each forward.
inline int              g_child_pre = 0;
inline std::vector<int> g_seen;
class PreTop : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("a", 1.0f, 0.0f, std::optional<float>{}, &g_child_pre)); }
    void pre_forward() override { g_seen.push_back(g_child_pre); }
};

/// p (do_once, y = 5) feeds c (y = 2 x + 1); rearms c on the second forward.
class ReplayTop : public cs::System {
    int n_ = 0;

public:
    void setup() override {
        add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<Gen>("p", 5.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("c", 2.0f, 1.0f));
        connect("p.y", "c.x");
    }
    void pre_forward() override {
        if (++n_ == 2) rearm("c");
    }
};

/// A System with a tie (input or output) inside, optionally promoted.
class TieInner : public cs::System {
    bool outputs_, promote_;

public:
    TieInner(bool outputs, bool promote) : outputs_{outputs}, promote_{promote} {}
    void setup() override {
        if (outputs_) {
            add_subsystem(cs::subsystem_desc<Gen>("a", 1.0f));
            add_subsystem(cs::subsystem_desc<Gen>("b", 1.0f));
            tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(kN), cs::TieOutputsPolicy::allow_plain);
        } else {
            add_subsystem(cs::subsystem_desc<Axpb>("a", 2.0f, 0.0f));
            add_subsystem(cs::subsystem_desc<Axpb>("b", 3.0f, 0.0f));
            tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::shape(kN));
        }
        if (promote_) promote<DType::f32, 1>("t", "pt");
    }
};

/// Ties the inner tie (or its promotion, `via_promotion`) with one more member in the parent: refused.
class TieOfTie : public cs::System {
    bool outputs_, via_promotion_;

public:
    TieOfTie(bool outputs, bool via_promotion) : outputs_{outputs}, via_promotion_{via_promotion} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<TieInner>("in", outputs_, via_promotion_));
        const char* inner = via_promotion_ ? "in.pt" : "in.t";
        if (outputs_) {
            add_subsystem(cs::subsystem_desc<Gen>("c", 1.0f));
            tie_outputs<DType::f32, 1>({inner, "c.y"}, "tt", cs::shape(kN), cs::TieOutputsPolicy::allow_plain);
        } else {
            add_subsystem(cs::subsystem_desc<Axpb>("c", 4.0f, 0.0f));
            tie_inputs<DType::f32, 1>({inner, "c.x"}, "tt", cs::shape(kN));
        }
    }
};

// ---- two devices -----------------------------------------------------------------------------------
/// The default device and one other, when the environment has two.
bool two_devices(sycl::device& def, sycl::device& other) {
    sycl::device first, host;
    bool         have_first = false, have_host = false;
    for (const sycl::device& d : sycl::device::get_devices()) {
        if (d.get_backend() == sycl::backend::omp) {
            if (!have_host) host = d;
            have_host = true;
        } else {
            if (!have_first) first = d;
            have_first = true;
        }
    }
    if (!have_first || !have_host) return false;
    def = cs::detail::default_device();
    if (def == first) {
        other = host;
        return true;
    }
    if (def == host) {
        other = first;
        return true;
    }
    return false;
}

}  // namespace tp4

// =============================================================================================
// wiring is legal anywhere
// =============================================================================================
TEST_CASE("wiring after setup: connect, promote, init_input, tie_inputs and tie_outputs are legal", "[p4][wiring]") {
    SECTION("connect") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::Gen>("p", 5.0f));
            s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("c", 2.0f, 1.0f));
        });
        model.setup();
        model.root().connect("p.y", "c.x");  // after setup, before anything has run
        tp4::run(model);
        CHECK(tp4::out(model, "c.y") == tp4::filled(11.0f));
    }
    SECTION("promote") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::Gen>("p", 5.0f));
        });
        model.setup();
        model.root().promote<DType::f32, 1>("p.y", "y");
        tp4::run(model);
        CHECK(tp4::out(model, "y") == tp4::filled(5.0f));
    }
    SECTION("init_input") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("c", 2.0f, 1.0f));
        });
        model.setup();
        model.root().init_input("c.x", cs::init::fill(3.0f));
        tp4::run(model);
        CHECK(tp4::out(model, "c.y") == tp4::filled(7.0f));
    }
    SECTION("tie_inputs") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("a", 2.0f, 0.0f));
            s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("b", 3.0f, 0.0f));
        });
        model.setup();
        model.root().tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::shape(tp4::kN));
        model.update_input("t", cs::init::fill(1.0f));
        tp4::run(model);
        CHECK(tp4::out(model, "a.y") == tp4::filled(2.0f));
        CHECK(tp4::out(model, "b.y") == tp4::filled(3.0f));
    }
    SECTION("tie_outputs") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::Half>("a", 0u, 4u));
            s.add_subsystem(cs::subsystem_desc<tp4::Half>("b", 4u, 8u));
        });
        model.setup();
        model.root().tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(tp4::kN), cs::TieOutputsPolicy::allow_plain);
        tp4::run(model);
        CHECK(tp4::out(model, "t") == tp4::iota_from_one());
    }
}

TEST_CASE("wiring: merging two materialized variables is an error at the call", "[p4][wiring][errors]") {
    cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tp4::Gen>("p", 5.0f));
        s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("c", 2.0f, 1.0f));
    });
    model.setup();
    tp4::run(model);  // p.y and c.x are both materialized now
    tp4::expect_throws_all([&] { model.root().connect("p.y", "c.x"); },
                           {"connect", "merging two materialized variables is not defined (yet)"});
    tp4::run(model);  // the failed call changed nothing: the model still runs
    CHECK(tp4::out(model, "c.y") == tp4::filled(1.0f));
}

// =============================================================================================
// run-policy override
// =============================================================================================
TEST_CASE("override_run_policy: a transient grandchild made eternal by its grandparent keeps its data", "[p4][policy]") {
    SECTION("without the override it is deleted after its turn") {
        cs::Model model = cs::create_model<tp4::OverTop>(default_config(), 0);
        model.setup();
        tp4::run(model);
        CHECK(model.get_subsystem("b.c") == nullptr);
    }
    SECTION("applied in the grandparent's setup") {
        cs::Model model = cs::create_model<tp4::OverTop>(default_config(), 1);
        model.setup();
        tp4::run(model);
        REQUIRE(model.get_subsystem("b.c") != nullptr);
        const auto var = model.root().instance_at("b.c.y")->meta;
        CHECK(tp4::out(model, "b.c.y") == tp4::filled(6.0f));
        for (int k = 0; k < 2; ++k) {
            tp4::run(model);
            REQUIRE(model.get_subsystem("b.c") != nullptr);
            CHECK(model.root().instance_at("b.c.y")->meta == var);  // the same variable: its data was kept
        }
        CHECK(tp4::out(model, "b.c.y") == tp4::filled(6.0f));
    }
    SECTION("applied after setup, through the root") {
        cs::Model model = cs::create_model<tp4::OverTop>(default_config(), 0);
        model.setup();
        model.root().override_run_policy("b.c");
        tp4::run(model);
        tp4::run(model);
        CHECK(model.get_subsystem("b.c") != nullptr);
    }
    SECTION("a policy for a child of a Component or an unknown name is an error") {
        cs::Model model = cs::create_model<tp4::OverTop>(default_config(), 0);
        model.setup();
        tp4::expect_throws_all([&] { model.root().override_run_policy("b.zz"); }, {"override_run_policy('b.zz')", "no such subsystem"});
        tp4::expect_throws_all([&] { model.root().override_run_policy("zz.c"); }, {"has no subsystem 'zz'"});
        tp4::expect_throws_all([&] { model.root().override_run_policy("b.c.y"); }, {"is a Component"});
    }
    SECTION("the other policies") {
        cs::Model model = cs::create_model<tp4::OverTop>(default_config(), 0);
        model.setup();
        model.root().override_run_policy("b", cs::property::run_n{2});  // an eternal System takes two turns
        tp4::run(model);
        CHECK(model.get_subsystem("b") != nullptr);
        tp4::run(model);
        CHECK(model.get_subsystem("b") == nullptr);
        model.root().override_run_policy("b", cs::property::do_once{});  // gone and not transient: it comes back, once
        REQUIRE(model.get_subsystem("b") != nullptr);
        tp4::run(model);
        CHECK(model.get_subsystem("b") == nullptr);
        tp4::expect_throws_all([&] { model.root().override_run_policy("b", cs::property_list{cs::property::transient{}, cs::property::run_n{2}}); },
                               {"transient cannot be combined with run_n or do_once"});
    }
}

TEST_CASE("override_run_policy: applied from pre_forward", "[p4][policy]") {
    cs::Model model = cs::create_model<tp4::OverTop>(default_config(), 2);
    model.setup();
    tp4::run(model);  // forward 1: c is transient and gone after its turn
    CHECK(model.get_subsystem("b.c") == nullptr);
    tp4::run(model);  // forward 2: pre_forward makes it eternal (it is rebuilt at once: it was gone)
    REQUIRE(model.get_subsystem("b.c") != nullptr);
    const auto var = model.root().instance_at("b.c.y")->meta;
    CHECK(tp4::out(model, "b.c.y") == tp4::filled(6.0f));
    tp4::run(model);  // forward 3: it kept its variable
    REQUIRE(model.get_subsystem("b.c") != nullptr);
    CHECK(model.root().instance_at("b.c.y")->meta == var);
}

TEST_CASE("override_run_policy is not logged: rearming the parent applies the declared policy again", "[p4][policy][rearm]") {
    cs::Model model = cs::create_model<tp4::OverTop>(default_config(), 1, 3);  // override in setup; rearm "b" on forward 3
    model.setup();
    tp4::run(model);
    tp4::run(model);
    CHECK(model.get_subsystem("b.c") != nullptr);  // eternal by the override
    tp4::run(model);                                // forward 3: "b" is rebuilt, c with its declared (transient) policy
    CHECK(model.get_subsystem("b.c") == nullptr);
}

// =============================================================================================
// rearm outside pre_forward
// =============================================================================================
TEST_CASE("rearm is legal outside pre_forward", "[p4][rearm]") {
    tp4::g_mid_built = 0;
    cs::Model model  = cs::create_model<tp4::OverTop>(default_config(), 0, 0, true);  // rearms "b" from post_forward
    model.setup();
    CHECK(tp4::g_mid_built == 1);
    CHECK_NOTHROW(tp4::run(model));  // used to throw: "rearm is legal inside pre_forward() only"
    CHECK(tp4::g_mid_built == 2);    // b was built again by the rearm in post_forward
    CHECK_NOTHROW(tp4::run(model));
    CHECK(tp4::g_mid_built == 2);
}

// =============================================================================================
// cycles
// =============================================================================================
TEST_CASE("cycle: a two-component cycle reads its initializer first, then the produced value", "[p4][cycle]") {
    for (const int mode : {0, 1}) {  // 0: init_input at the top after the connections; 1: the target's own initializer
        INFO("mode " << mode);
        tp4::CerrCapture cap;
        cs::Model        model = cs::create_model<tp4::CycleTop>(default_config(), mode);
        model.setup();
        CHECK(cap.buf.str().find("this is a cycle") != std::string::npos);  // the warning stays
        CHECK(model.root().instance_at("p.x")->meta->cycle);
        tp4::run(model);
        // p read the initializer (10) on the first forward, though p.x is connected: p.y = 10 + 1. (p.x is q.y's variable, so
        // after the forward it holds what q produced.)
        CHECK(tp4::out(model, "p.y") == tp4::filled(11.0f));
        CHECK(tp4::out(model, "q.y") == tp4::filled(11.0f));
        CHECK(tp4::in(model, "p.x") == tp4::filled(11.0f));
        tp4::run(model);
        // the second forward: p read q's output of the first (11): p.y = 11 + 1
        CHECK(tp4::out(model, "p.y") == tp4::filled(12.0f));
        CHECK(tp4::out(model, "q.y") == tp4::filled(12.0f));
    }
}

TEST_CASE("cycle: without an initializer the first forward throws, naming the variable and the cycle", "[p4][cycle][errors]") {
    tp4::CerrCapture cap;
    cs::Model        model = cs::create_model<tp4::CycleTop>(default_config(), 2);
    model.setup();  // not an error at setup: an initializer may still come
    tp4::expect_throws_all([&] { model.run(); }, {"closes a cycle", "'p.x' reads 'q.y', which runs after it", "has no initializer"});
}

TEST_CASE("cycle: an initializer set by update_input after setup counts", "[p4][cycle]") {
    tp4::CerrCapture cap;
    cs::Model        model = cs::create_model<tp4::CycleTop>(default_config(), 2);
    model.setup();
    model.update_input("p.x", cs::init::fill(20.0f));
    tp4::run(model);
    CHECK(tp4::out(model, "p.y") == tp4::filled(21.0f));
    tp4::run(model);
    CHECK(tp4::out(model, "p.y") == tp4::filled(22.0f));  // now the produced value
}

TEST_CASE("cycle: a correctly ordered connection is not a cycle", "[p4][cycle]") {
    cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tp4::Gen>("p", 5.0f));
        s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("c", 2.0f, 1.0f, std::optional<float>{99.0f}));
        s.connect("p.y", "c.x");  // c's own initializer is dropped: the source supplies the values
    });
    model.setup();
    CHECK_FALSE(model.root().instance_at("c.x")->meta->cycle);
    tp4::run(model);
    CHECK(tp4::out(model, "c.y") == tp4::filled(11.0f));
}

// =============================================================================================
// tie devices
// =============================================================================================
TEST_CASE("tie devices: the default device is where a tie_inputs alias lives and its initializer runs", "[p4][tie][devices]") {
    cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("a", 2.0f, 0.0f));
        s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("b", 3.0f, 0.0f));
        s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(tp4::kN));
        s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(tp4::kN), cs::TieOutputsPolicy::allow_plain);
    });
    model.setup();
    const sycl::device def = cs::detail::default_device();
    CHECK(model.root().instance_at("x")->device == def);
    REQUIRE(model.root().instance_at("x")->meta->init_device.has_value());
    CHECK(*model.root().instance_at("x")->meta->init_device == def);
    CHECK(model.root().instance_at("t")->device == def);
}

TEST_CASE("tie devices: tie_outputs refuses a member on another device at the call", "[p4][tie][devices][errors]") {
    sycl::device def, other;
    if (!tp4::two_devices(def, other)) SKIP("a non-OpenMP device and the OpenMP host device are not both visible");
    SECTION("the tie is on the default device; the member is elsewhere") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [&](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::Gen>("a", 1.0f));
            s.add_subsystem(other, cs::subsystem_desc<tp4::Gen>("b", 1.0f));
            s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(tp4::kN), cs::TieOutputsPolicy::allow_plain);
        });
        tp4::expect_throws_all([&] { model.setup(); }, {"tie_outputs", "every member on the tie's device", "'b.y' is on"});
    }
    SECTION("the tie is on the given device; a member is on the default one") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [&](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::Gen>("a", 1.0f));
            s.add_subsystem(other, cs::subsystem_desc<tp4::Gen>("b", 1.0f));
            s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(tp4::kN), cs::TieOutputsPolicy::allow_plain, other);
        });
        tp4::expect_throws_all([&] { model.setup(); }, {"tie_outputs", "every member on the tie's device", "'a.y' is on"});
    }
    SECTION("every member on the given device: fine") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [&](cs::System& s) {
            s.add_subsystem(other, cs::subsystem_desc<tp4::Half>("a", 0u, 4u));
            s.add_subsystem(other, cs::subsystem_desc<tp4::Half>("b", 4u, 8u));
            s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(tp4::kN), cs::TieOutputsPolicy::allow_plain, other);
        });
        CHECK_NOTHROW(model.setup());
        CHECK(model.root().instance_at("t")->device == other);
        tp4::run(model);
        CHECK(tp4::out(model, "t") == tp4::iota_from_one());
    }
}

TEST_CASE("tie devices: a member moved by set_device after the tie errors at the next forward", "[p4][tie][devices][errors]") {
    sycl::device def, other;
    if (!tp4::two_devices(def, other)) SKIP("a non-OpenMP device and the OpenMP host device are not both visible");
    cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tp4::Gen>("a", 1.0f));
        s.add_subsystem(cs::subsystem_desc<tp4::Gen>("b", 1.0f));
        s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(tp4::kN), cs::TieOutputsPolicy::allow_plain);
    });
    model.setup();
    tp4::run(model);  // fine: both on the default device
    model.get_subsystem("b")->set_device(other);
    tp4::expect_throws_all([&] { model.run(); },
                           {"tie_outputs 't'", "member 'b.y' is on", "but the tie is on", "every member must be on the tie's device"});
}

TEST_CASE("tie devices: a device given to add_subsystem moves the whole subtree, aliases included", "[p4][tie][devices]") {
    sycl::device def, other;
    if (!tp4::two_devices(def, other)) SKIP("a non-OpenMP device and the OpenMP host device are not both visible");
    // TieInner ties on the default device in its own setup; the outer call moves members and alias together.
    cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [&](cs::System& s) {
        s.add_subsystem(other, cs::subsystem_desc<tp4::TieInner>("in", true, false));
    });
    model.setup();
    CHECK(model.root().instance_at("in.a.y")->device == other);
    CHECK(model.root().instance_at("in.t")->device == other);  // the alias moved with them
    CHECK_NOTHROW(tp4::run(model));                              // so the run-time check passes
    CHECK(tp4::out(model, "in.t") == tp4::filled(1.0f));
}

TEST_CASE("tie devices: tie_inputs on a given device, and a moved subtree moves the alias and its initializer", "[p4][tie][devices]") {
    sycl::device def, other;
    if (!tp4::two_devices(def, other)) SKIP("a non-OpenMP device and the OpenMP host device are not both visible");
    SECTION("an explicit device") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [&](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("a", 2.0f, 0.0f));
            s.add_subsystem(cs::subsystem_desc<tp4::Axpb>("b", 3.0f, 0.0f));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(tp4::kN), other);
            s.init_input("x", cs::init::fill(2.0f));
        });
        model.setup();
        CHECK(model.root().instance_at("x")->device == other);
        REQUIRE(model.root().instance_at("x")->meta->init_device.has_value());
        CHECK(*model.root().instance_at("x")->meta->init_device == other);
        tp4::run(model);
        CHECK(tp4::out(model, "a.y") == tp4::filled(4.0f));  // members on the default device got the copy
        CHECK(tp4::out(model, "b.y") == tp4::filled(6.0f));
    }
    SECTION("the subtree moved") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [&](cs::System& s) {
            s.add_subsystem(other, cs::subsystem_desc<tp4::TieInner>("in", false, false));
        });
        model.setup();
        CHECK(model.root().instance_at("in.t")->device == other);
        REQUIRE(model.root().instance_at("in.t")->meta->init_device.has_value());
        CHECK(*model.root().instance_at("in.t")->meta->init_device == other);
    }
}

// =============================================================================================
// no ties of ties
// =============================================================================================
TEST_CASE("ties: a tie of a tie is refused, for both kinds, directly and through a promotion", "[p4][tie][errors]") {
    for (const bool outputs : {false, true})
        for (const bool via_promotion : {false, true}) {
            INFO((outputs ? "tie_outputs" : "tie_inputs") << (via_promotion ? ", the tie promoted first" : ", the tie itself"));
            cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [=](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tp4::TieOfTie>("top", outputs, via_promotion));
            });
            tp4::expect_throws_all([&] { model.setup(); },
                                   {outputs ? "tie_outputs" : "tie_inputs", "tying a tie is not allowed (yet)"});
        }
}

TEST_CASE("ties: the input and output marks are kept separately", "[p4][tie]") {
    // A tie's variable carries only the mark of its own kind; tie_outputs also push-suppresses its variable.
    cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tp4::TieInner>("i", false, false));
        s.add_subsystem(cs::subsystem_desc<tp4::TieInner>("o", true, false));
    });
    model.setup();
    const auto ti = model.root().instance_at("i.t")->meta;
    const auto to = model.root().instance_at("o.t")->meta;
    CHECK(ti->tied_input);
    CHECK_FALSE(ti->tied_output);
    CHECK(to->tied_output);
    CHECK_FALSE(to->tied_input);
    CHECK(to->push_suppressed);
}

TEST_CASE("ties: promoting a tie works, for both kinds", "[p4][tie]") {
    SECTION("tie_inputs") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::TieInner>("in", false, true));
        });
        model.setup();
        CHECK(model.root().instance_at("in.pt")->meta == model.root().instance_at("in.t")->meta);
        model.update_input("in.pt", cs::init::fill(2.0f));
        tp4::run(model);
        CHECK(tp4::out(model, "in.a.y") == tp4::filled(4.0f));
        CHECK(tp4::out(model, "in.b.y") == tp4::filled(6.0f));
    }
    SECTION("tie_outputs") {
        cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tp4::TieInner>("in", true, true));
        });
        model.setup();
        CHECK(model.root().instance_at("in.pt")->meta == model.root().instance_at("in.t")->meta);
        tp4::run(model);
        CHECK(tp4::out(model, "in.pt") == tp4::filled(1.0f));
    }
}

// =============================================================================================
// tied-output pushes
// =============================================================================================
TEST_CASE("tied outputs: a consumer on another device right after the members reads the whole tie in the same forward",
          "[p4][tie][devices][push]") {
    sycl::device def, other;
    if (!tp4::two_devices(def, other)) SKIP("a non-OpenMP device and the OpenMP host device are not both visible");
    cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [&](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tp4::Half>("a", 0u, 4u));  // each member writes half of the tie
        s.add_subsystem(cs::subsystem_desc<tp4::Half>("b", 4u, 8u));
        s.add_subsystem(other, cs::subsystem_desc<tp4::Axpb>("c", 1.0f, 0.0f));  // reads the tie on the other device
        s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(tp4::kN), cs::TieOutputsPolicy::allow_plain);
        s.connect("t", "c.x");
    });
    model.setup();
    CHECK(model.root().instance_at("a.y")->meta->push_suppressed);
    tp4::run(model);  // ONE forward: the push comes right after b, before c
    CHECK(tp4::out(model, "c.y") == tp4::iota_from_one());
    tp4::run(model);
    CHECK(tp4::out(model, "c.y") == tp4::iota_from_one());
}

// =============================================================================================
// the System loop
// =============================================================================================
TEST_CASE("System precompute runs the user's pre_forward and never a child's precompute", "[p4][loop]") {
    tp4::g_child_pre = 0;
    tp4::g_seen.clear();
    cs::Model model = cs::create_model<tp4::PreTop>(default_config());
    model.setup();
    for (int k = 0; k < 3; ++k) tp4::run(model);
    // The child's precompute (its pre_forward) comes after the parent's in each forward: the parent saw 0, 1, 2.
    CHECK(tp4::g_seen == std::vector<int>{0, 1, 2});
    CHECK(tp4::g_child_pre == 3);
}

TEST_CASE("replay skips a logged connection whose other side was deleted: the rebuilt child runs with a fresh variable",
          "[p4][replay][rearm]") {
    cs::Model model = cs::create_model<tp4::ReplayTop>(default_config());
    model.setup();
    tp4::run(model);  // p (do_once) wrote 5; c = 2 * 5 + 1; p is deleted after its turn
    CHECK(tp4::out(model, "c.y") == tp4::filled(11.0f));
    CHECK(model.get_subsystem("p") == nullptr);
    CHECK_NOTHROW(tp4::run(model));  // pre_forward rearms c: the connect to the deleted p is skipped
    CHECK(tp4::out(model, "c.y") == tp4::filled(1.0f));  // c.x is a fresh variable (zeros): 2 * 0 + 1
}

// =============================================================================================
// merges absorb containers
// =============================================================================================
TEST_CASE("a connect after setup moves the host holder's container into the device variable", "[p4][merge]") {
    cs::Model model = cs::create_model<tp4::TopFn>(default_config(), [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tp4::Gen>("p", 1.0f));
        s.add_subsystem(cs::subsystem_desc<tp4::HostInc>("h"));
    });
    model.setup();
    const auto p_y    = model.root().instance_at("p.y");
    const auto h_x    = model.root().instance_at("h.x");
    const auto before = h_x->data;  // the host placement's own (empty) container
    REQUIRE(before != nullptr);
    model.root().connect("p.y", "h.x");
    CHECK(h_x->meta == p_y->meta);
    CHECK(h_x->data == before);      // the winner had none for the host key: it adopted this one, identity kept
    CHECK(p_y->data != h_x->data);   // two placements of one variable: device and host
    tp4::run(model);
    CHECK(tp4::out(model, "h.y") == tp4::filled(2.0f));
}

// =============================================================================================
// implicit zero initializer
// =============================================================================================
TEST_CASE("init: the implicit zero is not a declaration, so a cycle with no explicit initializer still throws", "[p4][init][cycle][errors]") {
    tp4::CerrCapture cap;
    cs::Model        model = cs::create_model<tp4::CycleTop>(default_config(), 2);
    model.setup();
    tp4::expect_throws_all([&] { model.run(); }, {"closes a cycle", "'p.x' reads 'q.y', which runs after it", "has no initializer"});
}
