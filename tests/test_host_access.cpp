// Host access (internal/design-review-2026-10-03.md, section 11.4).
//
//   1. handles       Model::get_input<DT, Rank>(path) (view, update) and get_output (view); the kind-mismatch errors
//   2. views         a view can be made at any time: before the first forward an unconnected input shows its initialized
//                    values after sync; a view made before the first forward shows the computed values after a forward
//   3. view_input    a component reads one of its own inputs through a host view in prepare_forward and sizes its output
//
// Every fixture is prefixed Ha (all test files are one binary).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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

namespace tha {

using Needles = std::vector<std::string>;
using Floats  = std::vector<float>;

class HaTop : public cs::System {
    std::function<void(cs::System&)> f_;

public:
    explicit HaTop(std::function<void(cs::System&)> f) : f_{std::move(f)} {}
    void setup() override { f_(*this); }
};
inline cs::Model ha_make(std::function<void(cs::System&)> f) { return cs::create_model<HaTop>(default_config(), std::move(f)); }

/// The call throws and the message holds every needle.
template <class F>
void ha_throws(F&& call, const Needles& needles) {
    std::string what = "<no exception>";
    try {
        call();
    } catch (const std::exception& e) {
        what = e.what();
    }
    INFO("thrown: " << what);
    CHECK(what != "<no exception>");
    for (const auto& n : needles) CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}

template <class View>
Floats ha_values(const View& v) {
    v.sync();
    return Floats(v.span().begin(), v.span().end());
}

// ---- HaScale: y = 2 x; x is a free input of 4 with an initial value, y has 4 elements ----------------------------------------

class HaScale : public cs::Component<HaScale> {
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

    void declare() override {
        declare_input(inputs.x, "x", 4);
        declare_output(outputs.y, "y", 4);
        init_input("x", cs::init::from_host(Floats{1, 2, 3, 4}));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

// ---- HaMaxSize: y[i] = i + 1, with max(v) elements: v is read through view_input in prepare_forward ------------------------------

class HaMaxSize : public cs::Component<HaMaxSize> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> v;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.y[i] = static_cast<float>(i[0] + 1); }

    Inputs  inputs;
    Outputs outputs;

private:
    std::optional<cs::HostView<DType::u32, 1>>    view_;  // made on the first forward, kept: it is a holder on the host

public:
    void declare() override {
        declare_input(inputs.v, "v", 5);
        declare_output(outputs.y, "y", cs::unspecified);
        init_input("v", cs::init::from_host(std::vector<std::uint32_t>{3, 1, 2, 3, 2}));
    }
    void prepare_forward() override {
        if (!view_) view_ = view_input(inputs.v);
        view_->sync();
        const auto s = view_->span();
        set_shape(outputs.y, static_cast<std::size_t>(*std::max_element(s.begin(), s.end())));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

}  // namespace tha

TEST_CASE("handles: get_input gives a view and update, get_output a view", "[host_access][handle]") {
    auto m = tha::ha_make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tha::HaScale>("c")); });
    m.setup();
    auto in  = m.get_input<DType::f32, 1>("c.x");
    auto out = m.get_output<DType::f32, 1>("c.y");

    m.run();
    m.wait();
    auto xv = in.view();
    auto yv = out.view();
    CHECK(tha::ha_values(xv) == tha::Floats{1, 2, 3, 4});
    CHECK(tha::ha_values(yv) == tha::Floats{2, 4, 6, 8});
    CHECK(xv.shape()[0] == 4);

    in.update(cs::init::from_host(tha::Floats{10, 20, 30, 40}));  // the same as update_input
    CHECK(tha::ha_values(xv) == tha::Floats{1, 2, 3, 4});         // nothing ran yet: the old values
    m.run();
    m.wait();
    CHECK(tha::ha_values(xv) == tha::Floats{10, 20, 30, 40});
    CHECK(tha::ha_values(yv) == tha::Floats{20, 40, 60, 80});

    m.update_input("c.x", cs::init::from_host(tha::Floats{5, 5, 5, 5}));  // the shorthand still works
    m.run();
    m.wait();
    CHECK(tha::ha_values(yv) == tha::Floats{10, 10, 10, 10});
    CHECK(tha::ha_values(m.view_output<DType::f32, 1>("c.y")) == tha::Floats{10, 10, 10, 10});  // and so does make_view's
}

TEST_CASE("handles: get_input on an output and get_output on an input throw, naming the path and its kind", "[host_access][handle][errors]") {
    auto m = tha::ha_make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tha::HaScale>("c")); });
    m.setup();
    tha::ha_throws([&] { (void)m.get_input<DType::f32, 1>("c.y"); }, {"get_input", "'c.y'", "an output"});
    tha::ha_throws([&] { (void)m.get_output<DType::f32, 1>("c.x"); }, {"get_output", "'c.x'", "an input"});
    tha::ha_throws([&] { (void)m.get_input<DType::f32, 2>("c.x"); }, {"get_input", "'c.x'", "wrong dtype or rank"});
    tha::ha_throws([&] { (void)m.get_input<DType::f32, 1>("c.nope"); }, {"c.nope"});
}

TEST_CASE("views any time: a free initialized input shows its initial values after sync, before any forward", "[host_access][view]") {
    auto m = tha::ha_make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tha::HaScale>("c")); });
    m.setup();
    m.finalize();  // r04: views are available after finalize
    auto xv = m.get_input<DType::f32, 1>("c.x").view();  // no forward has run
    CHECK(xv.shape()[0] == 4);
    CHECK(tha::ha_values(xv) == tha::Floats{1, 2, 3, 4});
    auto yv = m.get_output<DType::f32, 1>("c.y").view();  // an output before any run: whatever is there, sized
    yv.sync();
    CHECK(yv.size() == 4);

    m.run();  // the variable was materialized by the view: the run goes on from it
    m.wait();
    CHECK(tha::ha_values(xv) == tha::Floats{1, 2, 3, 4});
    CHECK(tha::ha_values(yv) == tha::Floats{2, 4, 6, 8});
}

TEST_CASE("views any time: a view made after setup and before the first forward shows the computed values after the forward",
          "[host_access][view]") {
    auto m = tha::ha_make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tha::HaScale>("c")); });
    m.setup();
    m.finalize();  // r04: views are available after finalize
    auto yv = m.get_output<DType::f32, 1>("c.y").view();
    m.run();
    yv.sync();  // no wait(): sync waits on the host container's write list
    CHECK(tha::ha_values(yv) == tha::Floats{2, 4, 6, 8});
}

TEST_CASE("views any time: a view whose shape cannot be resolved yet throws and the model is unharmed", "[host_access][view][errors]") {
    auto m = tha::ha_make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tha::HaMaxSize>("p")); });
    m.setup();
    m.finalize();  // r04: views are available after finalize
    // p.y's size is set by its producer in prepare_forward: before a run it is unknown
    tha::ha_throws([&] { (void)m.get_output<DType::f32, 1>("p.y").view(); }, {"p.y"});
    m.run();
    m.wait();
    CHECK(tha::ha_values(m.get_output<DType::f32, 1>("p.y").view()) == tha::Floats{1, 2, 3});
}

TEST_CASE("view_input: a component sizes its output from the max of an input read in prepare_forward, over two forwards", "[host_access][component]") {
    auto m = tha::ha_make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tha::HaMaxSize>("p")); });
    m.setup();
    auto y = m.get_output<DType::f32, 1>("p.y");

    m.run();  // v = {3,1,2,3,2}: max 3
    m.wait();
    {
        const auto yv = y.view();
        CHECK(yv.shape()[0] == 3);
        CHECK(tha::ha_values(yv) == tha::Floats{1, 2, 3});
    }

    m.update_input("p.v", cs::init::from_host(std::vector<std::uint32_t>{1, 6, 2, 5, 2}));  // max 6
    m.run();
    m.wait();
    {
        const auto yv = y.view();
        CHECK(yv.shape()[0] == 6);
        CHECK(tha::ha_values(yv) == tha::Floats{1, 2, 3, 4, 5, 6});
    }

    m.update_input("p.v", cs::init::from_host(std::vector<std::uint32_t>{2, 2, 2, 2, 2}));  // and back down: max 2
    m.run();
    m.wait();
    const auto yv = y.view();
    CHECK(yv.shape()[0] == 2);
    CHECK(tha::ha_values(yv) == tha::Floats{1, 2});
}
