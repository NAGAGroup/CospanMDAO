// Behavior of the 2026-10-03 rework that no other test file checks (docs/design-review-2026-10-03.md, section 5).
//
//   update_input on a free input is applied at the NEXT forward; a view synced before it shows the old value
//   several update_input calls before one forward: the last one wins
//   a host view outlives the Model and still holds its values
//   Model::wait after a forward surfaces an exception thrown inside a host-body component (wait_and_throw)
//
// ALREADY COVERED ELSEWHERE, so not repeated here:
//   update_input on a connected input prints the warning and the value is unchanged   test_model.cpp (init: update_input on a connected input ...)
//                                                                                     and test_system.cpp (a model init on the alias warns)
//   forward() twice (three times) with no wait in between: correct values            test_system.cpp (three forwards enqueued back to back, then one wait)
//   the outer device wins                                                              test_system.cpp (placement cases)

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace trework {

constexpr std::size_t kN = 8;

// ---- TrCopy: y = x, x a free input of kN ----------------------------------------------------------------------------------

class TrCopy : public cs::Component<TrCopy> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = in.x[i]; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.x, "x", static_cast<int>(kN));
        declare_output(outputs.y, "y", static_cast<int>(kN));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

// ---- TrThrow: a HOST body that throws -------------------------------------------------------------------------------------

class TrThrow : public cs::Component<TrThrow> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(cs::Host, const Inputs&, const Outputs&) { throw std::runtime_error("trework: thrown inside a host body"); }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.x, "x", static_cast<int>(kN));
        declare_output(outputs.y, "y", static_cast<int>(kN));
        init_input("x", cs::init::fill(1.0f));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(); }
};

template <class C>
class TrTop : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<C>("a")); }
};

inline std::vector<float> filled(float v) { return std::vector<float>(kN, v); }

}  // namespace trework

TEST_CASE("update_input on a free input is applied at the next forward; a view synced before it shows the old value",
          "[rework][update_input]") {
    cs::Model m = cs::create_model<trework::TrTop<trework::TrCopy>>(default_config());
    m.setup();
    m.update_input("a.x", cs::init::fill(1.0f));
    cst::forward_and_wait(m);
    auto vx = cst::synced_input<DType::f32, 1>(m, "a.x");
    auto vy = cst::synced_output<DType::f32, 1>(m, "a.y");
    CHECK(cst::to_vector(vx) == trework::filled(1.0f));
    CHECK(cst::to_vector(vy) == trework::filled(1.0f));

    m.update_input("a.x", cs::init::fill(2.0f));  // marks the variable stale and queues no work
    vx.sync();
    vy.sync();
    CHECK(cst::to_vector(vx) == trework::filled(1.0f));  // no forward yet: still the old value
    CHECK(cst::to_vector(vy) == trework::filled(1.0f));

    cst::forward_and_wait(m);
    CHECK(cst::to_vector(cst::synced_input<DType::f32, 1>(m, "a.x")) == trework::filled(2.0f));
    CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "a.y")) == trework::filled(2.0f));
}

TEST_CASE("several update_input calls before one forward: the last one wins", "[rework][update_input]") {
    cs::Model m = cs::create_model<trework::TrTop<trework::TrCopy>>(default_config());
    m.setup();
    m.update_input("a.x", cs::init::fill(3.0f));
    m.update_input("a.x", cs::init::fill(4.0f));
    m.update_input("a.x", cs::init::fill(5.0f));
    cst::forward_and_wait(m);
    CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "a.y")) == trework::filled(5.0f));

    // and again between two later forwards
    m.update_input("a.x", cs::init::fill(6.0f));
    m.update_input("a.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}));
    cst::forward_and_wait(m);
    CHECK(cst::to_vector(cst::synced_output<DType::f32, 1>(m, "a.y")) == std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
}

TEST_CASE("a host view outlives the Model and still holds its values", "[rework][view]") {
    std::optional<cs::HostView<DType::f32, 1>> view;
    {
        cs::Model m = cs::create_model<trework::TrTop<trework::TrCopy>>(default_config());
        m.setup();
        m.update_input("a.x", cs::init::fill(7.0f));
        cst::forward_and_wait(m);
        view.emplace(m.view_output<DType::f32, 1>("a.y"));
    }  // the Model is destroyed here
    view->sync();
    CHECK(view->size() == trework::kN);
    CHECK(cst::to_vector(*view) == trework::filled(7.0f));
}

TEST_CASE("Model::wait after forward surfaces an exception thrown inside a host-body component", "[rework][wait]") {
    // forward() only enqueues and wait() is wait_and_throw on both queues (model.hpp), so a host task's exception surfaces
    // from wait(), not from forward(). Only that wait() throws is asserted, not the text.
    cs::Model m = cs::create_model<trework::TrTop<trework::TrThrow>>(default_config());
    m.setup();
    CHECK_NOTHROW(m.run());
    CHECK_THROWS(m.wait());
    try {
        m.wait();  // drain whatever is left before the Model is destroyed
    } catch (...) {
    }
}
