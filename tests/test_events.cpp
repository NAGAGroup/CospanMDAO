// P3 event and holder tests, every case through a cospan::Model.
//
//   destruction gate        a kernel spins on a host flag; destroying the model does not complete until the flag opens
//   two readers             two readers of one variable do not wait on each other (checked through their events)
//   a second holder         a holder on a new placement (a host view) gets its own container, a copy of the first
//   first materialization   asked by an input it is global (every live container is sized before any other holder runs)
//                           and it initializes
//
// The spinning kernels need host USM the kernel can poll; it comes from the model's own context, so the same tests run
// on the OpenMP backend and on a device backend.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
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
struct FillBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = 1.0f; }
};

/// y[i] = 1, kN elements.
class Fill : public cs::Component<FillBody> {
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override { declare_output(y_, "y", static_cast<int>(kN)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {}, {y_}, {}); }
};

struct PlusBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = in.x[i] + 1.0f; }
};

/// y = x + 1; y has kN elements, x takes the shape of what it is connected to or tied into. `on_pre`, when given, runs
/// at the start of pre_forward, after this component's inputs have been materialized.
class Plus : public cs::Component<PlusBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
    std::function<void()>     on_pre_;

public:
    explicit Plus(std::function<void()> on_pre = {}) : on_pre_{std::move(on_pre)} {}
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(y_, "y", static_cast<int>(kN));
    }
    void pre_forward() override {
        if (on_pre_) on_pre_();
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(x_.shape(), {x_}, {y_}, {}); }
};

/// y = x + 1, after the host flag is non-zero: the kernel spins on it (host USM, polled with system scope).
struct SpinBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    int* flag;
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system> a{*flag};
        while (a.load() == 0) {
        }
        out.y[i] = in.x[i] + 1.0f;
    }
};

class Spin : public cs::Component<SpinBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
    int*                      flag_;

public:
    explicit Spin(int* flag) : flag_{flag} {}
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(y_, "y", static_cast<int>(kN));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {x_}, {y_}, SpinBody{flag_});
    }
};

/// A root System whose setup() runs a callable.
class TopFn : public cs::System {
    std::function<void(cs::System&)> build_;

public:
    explicit TopFn(std::function<void(cs::System&)> b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

// ---- host flag -----------------------------------------------------------------------------------
/// One int of host USM from the model's own context, zero (closed). Declare it BEFORE the model: it is freed after it.
class HostFlag {
public:
    explicit HostFlag(const cs::ModelConfig& cfg)
        : ctx_{cfg.ctx.context_of(cs::detail::default_device())}, p_{sycl::malloc_host<int>(1, ctx_)} {
        REQUIRE(p_ != nullptr);
        *p_ = 0;
    }
    HostFlag(const HostFlag&)            = delete;
    HostFlag& operator=(const HostFlag&) = delete;
    ~HostFlag() { sycl::free(p_, ctx_); }

    int* get() const { return p_; }
    void open() { std::atomic_ref<int>{*p_}.store(1); }
    void close() { std::atomic_ref<int>{*p_}.store(0); }

private:
    sycl::context ctx_;
    int*          p_;
};

/// Opens the flag when it goes out of scope. Declare it AFTER the model: it is destroyed first, so a failing REQUIRE cannot
/// leave the model's destructor waiting on a kernel nobody releases.
struct Opener {
    HostFlag& flag;
    ~Opener() { flag.open(); }
};

bool is_done(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}

/// Polls instead of blocking, so a property that fails is reported, not hung.
bool wait_done(const sycl::event& e, std::chrono::milliseconds limit = std::chrono::seconds{10}) {
    const auto until = std::chrono::steady_clock::now() + limit;
    while (!is_done(e)) {
        if (std::chrono::steady_clock::now() > until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return true;
}

/// Long enough that work which does not wait for a closed gate would have finished by now.
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds{150}); }

bool all_done(const std::vector<sycl::event>& evs) {
    return std::all_of(evs.begin(), evs.end(), [](const sycl::event& e) { return is_done(e); });
}

/// How many of a variable's placements currently hold a bound container.
std::size_t bound_placements(const cs::detail::VarMeta& m) {
    std::size_t n = 0;
    for (const auto& [key, weak] : m.data)
        if (const auto d = weak.lock(); d && d->bound()) ++n;
    return n;
}

}  // namespace tev

// =============================================================================================
// The destruction gate
// =============================================================================================
TEST_CASE("destruction gate: destroying a model waits for a kernel still using its data, then frees cleanly",
          "[events][lifetime]") {
    cs::ModelConfig cfg = default_config();
    tev::HostFlag   flag{cfg};
    int*            f = flag.get();

    auto model = std::make_unique<cs::Model>(cs::create_model<tev::TopFn>(std::move(cfg), [f](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tev::Fill>("p"));
        s.add_subsystem(cs::subsystem_desc<tev::Spin>("spin", f));
        s.connect("p.y", "spin.x");
    }));
    tev::Opener opener{flag};  // after the model: destroyed first
    model->setup();
    model->forward();  // enqueues: the spin kernel waits for the flag

    std::atomic<bool> destroyed{false};
    std::thread       t{[&] {
        model.reset();  // frees every container, each waiting on its own events
        destroyed = true;
    }};
    tev::settle();
    const bool destroyed_early = destroyed.load();

    flag.open();
    t.join();

    REQUIRE_FALSE(destroyed_early);
    CHECK(destroyed.load());
}

// =============================================================================================
// Readers do not wait on each other
// =============================================================================================
TEST_CASE("two readers of one variable do not wait on each other, and both are on its read list", "[events][readers]") {
    if (cs::detail::default_device().get_backend() == sycl::backend::omp)
        SKIP("the OpenMP backend never runs two kernels at once, so unordered-ness can't be observed here");

    cs::ModelConfig cfg = default_config();
    tev::HostFlag   flag{cfg};
    int*            f = flag.get();

    cs::Model model = cs::create_model<tev::TopFn>(std::move(cfg), [f](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tev::Fill>("p"));
        s.add_subsystem(cs::subsystem_desc<tev::Spin>("a", f));
        s.add_subsystem(cs::subsystem_desc<tev::Plus>("b"));
        s.connect("p.y", "a.x");
        s.connect("p.y", "b.x");
    });
    tev::Opener opener{flag};  // after the model: destroyed first
    model.setup();

    // One forward with the gate open compiles every kernel (a first launch can block the submitting call behind running
    // work), then the gate closes for the forward under test.
    flag.open();
    model.forward();
    model.wait();
    flag.close();
    model.forward();

    const std::vector<sycl::event> a_ev = model.get_subsystem("a")->events();
    const std::vector<sycl::event> b_ev = model.get_subsystem("b")->events();
    REQUIRE(a_ev.size() == 1);
    REQUIRE(b_ev.size() == 1);

    // b reads p.y beside a: it is ordered after p's write only, so it finishes while a still spins.
    REQUIRE(tev::wait_done(b_ev.front()));
    CHECK_FALSE(tev::is_done(a_ev.front()));

    // Both reads are on the container's read list: the next writer of p.y waits for both.
    const auto                     p_y   = model.root().instance_at("p.y");
    const std::vector<sycl::event> reads = p_y->data->write_deps();
    CHECK(std::find(reads.begin(), reads.end(), a_ev.front()) != reads.end());
    CHECK(std::find(reads.begin(), reads.end(), b_ev.front()) != reads.end());
    CHECK_FALSE(tev::all_done(reads));  // a's read is still running

    flag.open();
    model.wait();
    CHECK(tev::all_done(reads));
    auto b_y = cst::synced_output<DType::f32, 1>(model, "b.y");
    CHECK(to_vector(b_y) == std::vector<float>(tev::kN, 2.0f));
}

// =============================================================================================
// A second holder
// =============================================================================================
TEST_CASE("a second holder on a new placement gets its own container, a copy of the first", "[events][holders]") {
    cs::Model model = cs::create_model<tev::TopFn>(default_config(), [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tev::Fill>("p"));
    });
    model.setup();
    model.forward();
    model.wait();

    const auto               p_y  = model.root().instance_at("p.y");
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
        const auto bx = mp->root().instance_at("b.x");
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
        const auto inst = model.root().instance_at(p);
        REQUIRE(inst->data != nullptr);
        CHECK_FALSE(inst->data->bound());
        CHECK(inst->data->size() == 0);
        CHECK_FALSE(inst->meta->is_materialized());
    }

    model.forward();
    model.wait();
    CHECK(observed == 1);

    // Asked by an input, so it initializes: the free input holds the tie's initial value, in both consumers.
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "a.x")) == std::vector<float>(4, 3.0f));
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "b.x")) == std::vector<float>(4, 3.0f));
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")).front() == 4.0f);
}
