// P3 event and holder tests, every case through a cospan::Model.
//
//   a second holder         a holder on a new placement (a host view) gets its own container, a copy of the first
//   first materialization   asked by an input it is global (every live container is sized before any other holder runs)
//                           and it initializes
//
// (The destruction-gate and two-readers cases need a kernel held running; they were removed and wait for a test mocking
// system, see docs/ROADMAP.md.)

#include <algorithm>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;
using cst::default_config;
using cst::to_vector;

namespace tev {

constexpr std::size_t kN = 16;

// ---- bodies and components ---------------------------------------------------------------------

/// y[i] = 1, kN elements.
class Fill : public cs::Component<Fill> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs&, const Outputs& out) { out.y[i] = 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override { declare_output(outputs.y, "y", static_cast<int>(kN)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

/// y = x + 1; y has kN elements, x takes the shape of what it is connected to or tied into. `on_pre`, when given, runs
/// at the start of pre_forward, after this component's inputs have been materialized.
class Plus : public cs::Component<Plus> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = in.x[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

private:
    std::function<void()>     on_pre_;

public:
    explicit Plus(std::function<void()> on_pre = {}) : on_pre_{std::move(on_pre)} {}
    void setup() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_output(outputs.y, "y", static_cast<int>(kN));
    }
    void pre_forward() override {
        if (on_pre_) on_pre_();
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

/// A root System whose setup() runs a callable.
class TopFn : public cs::System {
    std::function<void(cs::System&)> build_;

public:
    explicit TopFn(std::function<void(cs::System&)> b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

/// How many of a variable's placements currently hold a bound container.
std::size_t bound_placements(const cs::detail::VarMeta& m) {
    std::size_t n = 0;
    for (const auto& [key, weak] : m.data)
        if (const auto d = weak.lock(); d && d->bound()) ++n;
    return n;
}

}  // namespace tev

// =============================================================================================
// A second holder
// =============================================================================================
TEST_CASE("a second holder on a new placement gets its own container, a copy of the first", "[events][holders]") {
    cs::Model model = cs::create_model<tev::TopFn>(default_config(), [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tev::Fill>("p"));
    });
    model.setup();
    model.run();
    model.wait();

    const auto               p_y  = cs::detail::ModelAccess::instance(model, "p.y");
    const cs::detail::VarMeta& meta = *p_y->meta;
    REQUIRE(p_y->data != nullptr);
    CHECK(tev::bound_placements(meta) == 1);  // the device container, and nothing else

    {
        auto v = model.view_output<DType::f32, 1>("p.y");  // a holder on the host placement, new
        CHECK(tev::bound_placements(meta) == 2);

        std::shared_ptr<cs::detail::DeviceDataBase> host;
        for (const auto& [key, weak] : meta.data)
            if (key.kind == sk::kind::host) host = weak.lock();
        REQUIRE(host != nullptr);
        CHECK(host != p_y->data);
        CHECK(host->bound());
        CHECK(host->size() == tev::kN);
        CHECK(host->raw() != p_y->data->raw());           // a copy, not the same storage
        CHECK_FALSE(host->read_deps().empty());           // the copy is its write, queued from the device container

        v.sync();
        CHECK(to_vector(v) == std::vector<float>(tev::kN, 1.0f));
    }

    // The view is gone and its container with it; the variable keeps the placement's entry, with nothing behind it.
    CHECK(tev::bound_placements(meta) == 1);
    CHECK(meta.data.size() == 2);
}

// =============================================================================================
// The first materialization
// =============================================================================================
TEST_CASE("the first materialization, asked by an input, is global and initializes", "[events][materialize]") {
    cs::Model* mp       = nullptr;
    int        observed = 0;
    // Runs inside a's pre_forward: a's inputs are materialized, b has not run. b's holder of the shared variable already
    // has its container, bound and sized, because the first materialization is global.
    auto observe = [&] {
        ++observed;
        const auto bx = cs::detail::ModelAccess::instance(*mp, "b.x");
        CHECK(bx->meta->is_materialized());
        REQUIRE(bx->data != nullptr);
        CHECK(bx->data->bound());
        CHECK(bx->data->size() == 4);
    };

    cs::Model model = cs::create_model<tev::TopFn>(default_config(), [observe](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tev::Plus>("a", std::function<void()>{observe}));
        s.add_subsystem(cs::subsystem_desc<tev::Plus>("b"));
        s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(4));
        s.init_input("x", cs::init::fill(3.0f));
    });
    mp = &model;
    model.setup();

    // Before the first forward every holder has its container, empty and unbound.
    for (const char* p : {"a.x", "b.x"}) {
        INFO(p);
        const auto inst = cs::detail::ModelAccess::instance(model, p);
        REQUIRE(inst->data != nullptr);
        CHECK_FALSE(inst->data->bound());
        CHECK(inst->data->size() == 0);
        CHECK_FALSE(inst->meta->is_materialized());
    }

    model.run();
    model.wait();
    CHECK(observed == 1);

    // Asked by an input, so it initializes: the free input holds the tie's initial value, in both consumers.
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "a.x")) == std::vector<float>(4, 3.0f));
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "b.x")) == std::vector<float>(4, 3.0f));
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")).front() == 4.0f);
}
