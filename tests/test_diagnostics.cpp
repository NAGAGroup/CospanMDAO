// Event counters (include/naga/cospan/diagnostics.hpp).
//
// Only the counters that still have a caller exist (queue_waits, host_tasks); the chunk window and the
// ChunkedSystem baselines that used to live here are gone with the chunked systems (dead code until their redesign).
//
//   api    reset / snapshot / count_* / operator-
//   wired  Model::wait counts one queue wait; a host-body launch counts one host task per forward

#include <cstddef>
#include <cstdint>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace tdiag {

// A host-body producer: y[i] = i, run as a host task.

class HostIota : public cs::HostComponent<HostIota> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(const Inputs&, const Outputs& out) {
        for (std::size_t i = 0; i < out.y.size(); ++i) out.y[i] = static_cast<float>(i);
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.y, "y", 8); }
};

class HostTop : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<HostIota>("iota")); }
};

}  // namespace tdiag

TEST_CASE("Diagnostics: the counters API", "[diag]") {
    cs::diag::reset();
    const auto z = cs::diag::snapshot();
    CHECK(z.queue_waits == 0);
    CHECK(z.host_tasks == 0);
    cs::diag::count_queue_waits(3);
    cs::diag::count_host_tasks();
    const auto a = cs::diag::snapshot();
    CHECK(a.queue_waits == 3);
    CHECK(a.host_tasks == 1);
    cs::diag::count_queue_waits(2);
    cs::diag::count_host_tasks(4);
    const auto d = cs::diag::snapshot() - a;
    CHECK(d.queue_waits == 2);
    CHECK(d.host_tasks == 4);
    cs::diag::reset();
    CHECK(cs::diag::snapshot().queue_waits == 0);
    CHECK(cs::diag::snapshot().host_tasks == 0);
}

TEST_CASE("Diagnostics: Model::wait and a host launch are counted", "[diag]") {
    cs::Model m = cs::create_model<tdiag::HostTop>(default_config());
    m.setup();

    const auto before = cs::diag::snapshot();
    m.run();
    const auto after_fwd = cs::diag::snapshot() - before;
    // forward() enqueues and does not wait (design review, section 5), so no queue wait is counted by it.
    CHECK(after_fwd.queue_waits == 0);
    CHECK(after_fwd.host_tasks == 1);  // the host body is submitted as one host task

    m.wait();
    const auto after_wait = cs::diag::snapshot() - before;
    CHECK(after_wait.queue_waits == 1);
    CHECK(after_wait.host_tasks == 1);
}
