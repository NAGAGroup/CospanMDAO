// =============================================================================
// tests/test_r05.cpp — r05: components and drivers, placements by context, push ranking
// (docs/spec/revisions/r05-2026-10-13-components-drivers-placements.md)
// =============================================================================
//
//   drivers     a lone component is a leaf `c` with ports `c.x`; an error inside a component names its full path; a fused
//               group's stage is `f.s` (its ports `f.s.x`) and get_subsystem on it throws
//   host        a HostComponent between two kernel components: the values are right
//   placements  two host holders of one context share ONE host placement; a host view takes its holder's device
//   push        VarMeta::push_from over a variable with placements in two contexts (skipped with one backend)

#include <cstddef>
#include <functional>
#include <memory>
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

namespace tr05 {

constexpr std::size_t kN = 8;

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

std::vector<float> filled(float v) { return std::vector<float>(kN, v); }
void               run(cs::Model& m) {
    m.run();
    m.wait();
}
std::vector<float> out(cs::Model& m, std::string_view path) { return to_vector(cst::synced_output<DType::f32, 1>(m, path)); }

// ---- components -----------------------------------------------------------------------------------

/// y = x + 1; x has kN elements.
class Inc : public cs::Component<Inc> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& o) { o.y[i] = in.x[i] + 1.0f; }
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", static_cast<int>(kN));
        declare_output(outputs.y, "y", static_cast<int>(kN));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

/// y = x with an initial value for x: the source of every chain here (x = v, so y = v).
class Src : public cs::Component<Src> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& o) { o.y[i] = in.x[i]; }
    Inputs  inputs;
    Outputs outputs;
    float   v_;
    explicit Src(float v) : v_{v} {}
    void declare() override {
        declare_input(inputs.x, "x", static_cast<int>(kN));
        init_input("x", cs::init::fill(v_));
        declare_output(outputs.y, "y", static_cast<int>(kN));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

/// A HOST component: y = 2 x, on the host, over host USM.
class HostDouble : public cs::HostComponent<HostDouble> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(const Inputs& in, const Outputs& o) {
        for (std::size_t i = 0; i < in.x.size(); ++i) o.y[i] = 2.0f * in.x[i];
    }
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", static_cast<int>(kN));
        declare_output(outputs.y, "y", static_cast<int>(kN));
    }
};

/// Declares "x" twice: its error names the component's full path.
class Dup : public cs::Component<Dup> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& o) { o.y[i] = in.x[i]; }
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", 4);
        declare_input(inputs.x, "x", 4);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

// ---- systems -------------------------------------------------------------------------------------
class TopFn : public cs::System {
    std::function<void(cs::System&)> build_;

public:
    explicit TopFn(std::function<void(cs::System&)> b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

/// A System "s" holding one Dup "c".
class DupHolder : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Dup>("c")); }
};

cs::Model make(std::function<void(cs::System&)> build) { return cs::create_model<TopFn>(default_config(), std::move(build)); }

/// Two devices of ONE platform (so of one context), if the machine has them.
bool two_devices_of_one_context(sycl::device& a, sycl::device& b) {
    for (const auto& p : sycl::platform::get_platforms()) {
        const auto ds = p.get_devices();
        if (ds.size() >= 2) {
            a = ds[0];
            b = ds[1];
            return true;
        }
    }
    return false;
}

/// Two devices of different platforms (so of two contexts), if the machine has them.
bool two_contexts(sycl::device& a, sycl::device& b) {
    const sycl::device first = cs::detail::default_device();
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices())
            if (!(d.get_platform() == first.get_platform())) {
                a = first;
                b = d;
                return true;
            }
    return false;
}

}  // namespace tr05

// =============================================================================================
// drivers
// =============================================================================================

TEST_CASE("r05 drivers: a lone component's input is c.x and get_subsystem(c) is a leaf", "[r05][drivers]") {
    cs::Model model = tr05::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tr05::Inc>("c")); });
    model.setup();

    CHECK(cs::detail::ModelAccess::instance(model, "c.x") != nullptr);
    CHECK(cs::detail::ModelAccess::instance(model, "c.y") != nullptr);
    CHECK(cs::detail::ModelAccess::component(model, "c") != nullptr);
    REQUIRE(cs::detail::ModelAccess::subsystem(model, "c") != nullptr);
    CHECK(cs::detail::ModelAccess::subsystem(model, "c")->is_leaf());
    CHECK(model.get_subsystem("c").is_leaf());
    CHECK(model.get_subsystem("c").list_inputs() == std::vector<std::string>{"c.x"});
}

TEST_CASE("r05 drivers: an error inside a component names its full path", "[r05][drivers]") {
    cs::Model model = tr05::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tr05::DupHolder>("s")); });
    tr05::expect_throws_all([&] { model.setup(); }, {"'s.c'", "'x' is declared twice"});
}

TEST_CASE("r05 drivers: a fused stage is f.s with ports f.s.x, and get_subsystem(f.s) throws", "[r05][drivers][fused]") {
    cs::Model model = tr05::make([](cs::System& s) { s.add_subsystem(cs::fuse_components("f", cs::subsystem_desc<tr05::Inc>("s"))); });
    model.setup();

    CHECK(cs::detail::ModelAccess::instance(model, "f.s.x") != nullptr);
    CHECK(cs::detail::ModelAccess::instance(model, "f.s.y") != nullptr);
    CHECK(cs::detail::ModelAccess::component(model, "f.s") != nullptr);
    CHECK(model.get_subsystem("f").is_leaf());
    tr05::expect_throws_all([&] { (void)model.get_subsystem("f.s"); }, {"f.s"});
}

// =============================================================================================
// host components
// =============================================================================================

TEST_CASE("r05 host: a kernel producer feeds a HostComponent, which feeds a kernel consumer", "[r05][host]") {
    cs::Model model = tr05::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tr05::Src>("g", 3.0f));
        s.add_subsystem(cs::subsystem_desc<tr05::HostDouble>("h"));
        s.add_subsystem(cs::subsystem_desc<tr05::Inc>("c"));
        s.connect("g.y", "h.x");
        s.connect("h.y", "c.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 2; ++fwd) {
        tr05::run(model);
        CHECK(tr05::out(model, "h.y") == tr05::filled(6.0f));
        CHECK(tr05::out(model, "c.y") == tr05::filled(7.0f));
    }
}

// =============================================================================================
// placements
// =============================================================================================

TEST_CASE("r05 placements: host holders of one context share ONE host placement", "[r05][placements]") {
    SECTION("on one device") {
        cs::Model model = tr05::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tr05::HostDouble>("h0"));
            s.add_subsystem(cs::subsystem_desc<tr05::HostDouble>("h1"));
            s.connect("h0.y", "h1.x");
        });
        model.setup();
        const auto y = cs::detail::ModelAccess::instance(model, "h0.y");
        const auto x = cs::detail::ModelAccess::instance(model, "h1.x");
        REQUIRE(y);
        REQUIRE(x);
        CHECK(y->key() == x->key());
        CHECK(y->data == x->data);
    }
    SECTION("on two devices of one context") {
        sycl::device a, b;
        if (!tr05::two_devices_of_one_context(a, b)) SKIP("needs two devices of one context");
        cs::Model model = tr05::make([&](cs::System& s) {
            s.add_subsystem(a, cs::subsystem_desc<tr05::HostDouble>("h0"));
            s.add_subsystem(b, cs::subsystem_desc<tr05::HostDouble>("h1"));
            s.connect("h0.y", "h1.x");
        });
        model.setup();
        const auto y = cs::detail::ModelAccess::instance(model, "h0.y");
        const auto x = cs::detail::ModelAccess::instance(model, "h1.x");
        REQUIRE(y);
        REQUIRE(x);
        CHECK(y->device == a);
        CHECK(x->device == b);
        CHECK(y->key() == x->key());  // host keys are per context
        CHECK(y->data == x->data);
    }
}

TEST_CASE("r05 placements: a host view takes its holder's device", "[r05][placements][views]") {
    // a device that is not the default one if the machine has it, else the default (the check still holds)
    sycl::device target = cs::detail::default_device();
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices())
            if (!(d == cs::detail::default_device())) target = d;

    cs::Model model = tr05::make([&](cs::System& s) {
        s.add_subsystem(target, cs::subsystem_desc<tr05::Src>("g", 5.0f));
    });
    model.setup();
    tr05::run(model);
    const auto view   = cst::synced_input<DType::f32, 1>(model, "g.x");
    const auto holder = cs::detail::ModelAccess::instance(model, "g.x");
    REQUIRE(holder);
    CHECK(holder->device == target);

    std::shared_ptr<cs::detail::Instance> v;
    for (const auto& i : holder->meta->live_instances())
        if (i->role == cs::detail::Role::host_view) v = i;
    REQUIRE(v);
    CHECK(v->device == holder->device);
    CHECK(to_vector(view) == tr05::filled(5.0f));
}

// =============================================================================================
// push ranking
// =============================================================================================

TEST_CASE("r05 push: push_from reaches every placement once, across two contexts", "[r05][push]") {
    sycl::device a, b;
    if (!tr05::two_contexts(a, b)) SKIP("needs two backends (two contexts)");

    // g.y is produced on a; read on a (host), on b (host) and on b (device): placements in two contexts, a host placement in each
    cs::Model model = tr05::make([&](cs::System& s) {
        s.add_subsystem(a, cs::subsystem_desc<tr05::Src>("g", 3.0f));
        s.add_subsystem(a, cs::subsystem_desc<tr05::HostDouble>("ha"));
        s.add_subsystem(b, cs::subsystem_desc<tr05::HostDouble>("hb"));
        s.add_subsystem(b, cs::subsystem_desc<tr05::Inc>("cb"));
        s.connect("g.y", "ha.x");
        s.connect("g.y", "hb.x");
        s.connect("g.y", "cb.x");
    });
    model.setup();
    tr05::run(model);
    CHECK(tr05::out(model, "ha.y") == tr05::filled(6.0f));
    CHECK(tr05::out(model, "hb.y") == tr05::filled(6.0f));
    CHECK(tr05::out(model, "cb.y") == tr05::filled(4.0f));

    const auto src = cs::detail::ModelAccess::instance(model, "g.y");
    REQUIRE(src);
    REQUIRE(src->data);
    cs::detail::VarMeta& meta = *src->meta;

    std::size_t bound = 0;
    std::vector<sycl::context> contexts;
    for (const auto& p : meta.data) {
        const auto d = p.second.lock();
        if (!d || !d->bound()) continue;
        ++bound;
        bool seen = false;
        for (const auto& c : contexts) seen = seen || c == d->key.ctx;
        if (!seen) contexts.push_back(d->key.ctx);
    }
    CHECK(contexts.size() >= 2);  // placements in two contexts
    REQUIRE(bound >= 3);

    // one copy per target, every target reached once (src itself is not a target)
    const std::vector<sycl::event> evs = meta.push_from(src->data);
    CHECK(evs.size() + 1 == bound);
    sycl::event::wait(evs);

    // and the contents are right everywhere: another run reads the pushed data
    tr05::run(model);
    CHECK(tr05::out(model, "ha.y") == tr05::filled(6.0f));
    CHECK(tr05::out(model, "hb.y") == tr05::filled(6.0f));
    CHECK(tr05::out(model, "cb.y") == tr05::filled(4.0f));
}
