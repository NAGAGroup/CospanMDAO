// =============================================================================
// examples/sellar/sellar.cpp — the Sellar MDA problem on CospanMDAO
// =============================================================================
//
// Sellar is the canonical multidisciplinary-analysis-and-optimization (MDAO)
// test problem — the "hello world" of OpenMDAO, which this framework descends
// from. Two disciplines are mutually coupled (y1 depends on y2 and vice
// versa), so evaluating the system means settling a fixed point; optimizing
// it means differentiating THROUGH that fixed point.
//
//   Discipline 1:  y1 = z1^2 + z2 + x - 0.2*y2
//   Discipline 2:  y2 = sqrt(y1) + z1 + z2
//   Objective:     f  = x^2 + z2 + y1 + exp(-y2)
//   Constraints:   y1 >= 3.16,  y2 <= 24   (quadratic penalty here)
//   Bounds:        x >= 0, z2 >= 0         (quadratic penalty here)
//
// Known optimum: z1 ~= 1.9776, z2 = 0, x = 0, with y1 = 3.16, f ~= 3.1834.
//
// What this demonstrates, end to end:
//   * Components declaring scalar I/O and hand-written forward/backward
//     SYCL kernels (no tape — adjoints flow through grad-pointer aliasing
//     and atomic accumulation).
//   * A System composing the disciplines, closing the y1<->y2 cycle with
//     connect(), sharing design variables via promotions, and marking
//     Discipline 1 as the entry point (= the register cut: the y2 value it
//     reads is the previous sweep's — exactly how a Gauss-Seidel MDA sweeps).
//   * EquilibriumExecutor settling the coupled system to a fixed point and
//     solving the adjoint fixed point for gradients (IFT, not BPTT).
//   * AdamOptimizer driving the promoted design variables.
//
// Run (CPU):  ACPP_VISIBILITY_MASK=omp ./build/dev/examples/sellar
// =============================================================================

#include <naga/cospan/core/component.hpp>
#include <naga/cospan/core/system.hpp>
#include <naga/cospan/core/equilibrium_executor.hpp>
#include <naga/cospan/core/transfer.hpp>
#include <naga/cospan/opt/adam.hpp>

#include <slabkit/slabkit.hpp>
#include <sycl/sycl.hpp>

#include <cstdio>
#include <memory>

namespace nc = naga::cospan::core;

// Penalty weight for constraints/bounds (quadratic exterior penalty).
constexpr float kPenalty = 100.0f;

// ── Discipline 1: y1 = z1^2 + z2 + x - 0.2*y2 ────────────────────────────────

struct Dis1Forward {
    nc::Input<1>  z1, z2, x, y2;
    nc::Output<1> y1;
    sycl::range<1> range() const { return sycl::range<1>{1}; }
    void operator()(sycl::id<1>) const {
        y1.write(0, z1.get(0) * z1.get(0) + z2.get(0) + x.get(0) -
                        0.2f * y2.get(0));
    }
};

struct Dis1Backward {
    nc::Input<1>  z1, z2, x, y2;
    nc::Output<1> y1;
    sycl::range<1> range() const { return sycl::range<1>{1}; }
    void operator()(sycl::id<1>) const {
        const float g = y1.get_grad(0);
        z1.accum_grad(0, g * 2.0f * z1.get(0));
        z2.accum_grad(0, g);
        x.accum_grad(0, g);
        y2.accum_grad(0, g * -0.2f);
    }
};

class Discipline1 final : public nc::Component {
public:
    void declare(nc::Context& ctx) override {
        declare_input<1>(ctx, "z1", sycl::range<1>{1});
        declare_input<1>(ctx, "z2", sycl::range<1>{1});
        declare_input<1>(ctx, "x",  sycl::range<1>{1});
        declare_input<1>(ctx, "y2", sycl::range<1>{1});
        declare_output<1>(ctx, "y1", sycl::range<1>{1});
    }
    void forward(nc::Context& ctx, sycl::queue& q) override {
        execute_compute(q, Dis1Forward{get_input<1>(ctx, "z1"),
                                       get_input<1>(ctx, "z2"),
                                       get_input<1>(ctx, "x"),
                                       get_input<1>(ctx, "y2"),
                                       get_output<1>(ctx, "y1")});
    }
    void backward(nc::Context& ctx, sycl::queue& q) override {
        execute_compute(q, Dis1Backward{get_input<1>(ctx, "z1"),
                                        get_input<1>(ctx, "z2"),
                                        get_input<1>(ctx, "x"),
                                        get_input<1>(ctx, "y2"),
                                        get_output<1>(ctx, "y1")});
    }
};

// ── Discipline 2: y2 = sqrt(y1) + z1 + z2 ────────────────────────────────────

struct Dis2Forward {
    nc::Input<1>  z1, z2, y1;
    nc::Output<1> y2;
    sycl::range<1> range() const { return sycl::range<1>{1}; }
    void operator()(sycl::id<1>) const {
        y2.write(0, sycl::sqrt(sycl::fmax(y1.get(0), 0.0f)) + z1.get(0) +
                        z2.get(0));
    }
};

struct Dis2Backward {
    nc::Input<1>  z1, z2, y1;
    nc::Output<1> y2;
    sycl::range<1> range() const { return sycl::range<1>{1}; }
    void operator()(sycl::id<1>) const {
        const float g = y2.get_grad(0);
        y1.accum_grad(0, g * 0.5f / sycl::sqrt(sycl::fmax(y1.get(0), 1e-8f)));
        z1.accum_grad(0, g);
        z2.accum_grad(0, g);
    }
};

class Discipline2 final : public nc::Component {
public:
    void declare(nc::Context& ctx) override {
        declare_input<1>(ctx, "z1", sycl::range<1>{1});
        declare_input<1>(ctx, "z2", sycl::range<1>{1});
        declare_input<1>(ctx, "y1", sycl::range<1>{1});
        declare_output<1>(ctx, "y2", sycl::range<1>{1});
    }
    void forward(nc::Context& ctx, sycl::queue& q) override {
        execute_compute(q, Dis2Forward{get_input<1>(ctx, "z1"),
                                       get_input<1>(ctx, "z2"),
                                       get_input<1>(ctx, "y1"),
                                       get_output<1>(ctx, "y2")});
    }
    void backward(nc::Context& ctx, sycl::queue& q) override {
        execute_compute(q, Dis2Backward{get_input<1>(ctx, "z1"),
                                        get_input<1>(ctx, "z2"),
                                        get_input<1>(ctx, "y1"),
                                        get_output<1>(ctx, "y2")});
    }
};

// ── Objective: f = x^2 + z2 + y1 + exp(-y2) + penalties ─────────────────────

struct ObjForward {
    nc::Input<1>  x, z2, y1, y2;
    nc::Output<1> f;
    sycl::range<1> range() const { return sycl::range<1>{1}; }
    void operator()(sycl::id<1>) const {
        const float xv = x.get(0), z2v = z2.get(0);
        const float y1v = y1.get(0), y2v = y2.get(0);
        const float c1 = sycl::fmax(3.16f - y1v, 0.0f);  // y1 >= 3.16
        const float c2 = sycl::fmax(y2v - 24.0f, 0.0f);  // y2 <= 24
        const float b1 = sycl::fmax(-xv, 0.0f);          // x  >= 0
        const float b2 = sycl::fmax(-z2v, 0.0f);         // z2 >= 0
        f.write(0, xv * xv + z2v + y1v + sycl::exp(-y2v) +
                       kPenalty * (c1 * c1 + c2 * c2 + b1 * b1 + b2 * b2));
    }
};

struct ObjBackward {
    nc::Input<1>  x, z2, y1, y2;
    nc::Output<1> f;
    sycl::range<1> range() const { return sycl::range<1>{1}; }
    void operator()(sycl::id<1>) const {
        const float g = f.get_grad(0);  // objective grad seeded 1.0 upstream
        const float xv = x.get(0), z2v = z2.get(0);
        const float y1v = y1.get(0), y2v = y2.get(0);
        const float c1 = sycl::fmax(3.16f - y1v, 0.0f);
        const float c2 = sycl::fmax(y2v - 24.0f, 0.0f);
        const float b1 = sycl::fmax(-xv, 0.0f);
        const float b2 = sycl::fmax(-z2v, 0.0f);
        x.accum_grad(0, g * (2.0f * xv - 2.0f * kPenalty * b1));
        z2.accum_grad(0, g * (1.0f - 2.0f * kPenalty * b2));
        y1.accum_grad(0, g * (1.0f - 2.0f * kPenalty * c1));
        y2.accum_grad(0, g * (-sycl::exp(-y2v) + 2.0f * kPenalty * c2));
    }
};

class SellarObjective final : public nc::Component {
public:
    void declare(nc::Context& ctx) override {
        declare_input<1>(ctx, "x",  sycl::range<1>{1});
        declare_input<1>(ctx, "z2", sycl::range<1>{1});
        declare_input<1>(ctx, "y1", sycl::range<1>{1});
        declare_input<1>(ctx, "y2", sycl::range<1>{1});
        declare_output<1>(ctx, "f", sycl::range<1>{1});
        assign_objective(ctx, "f");
    }
    void forward(nc::Context& ctx, sycl::queue& q) override {
        execute_compute(q, ObjForward{get_input<1>(ctx, "x"),
                                      get_input<1>(ctx, "z2"),
                                      get_input<1>(ctx, "y1"),
                                      get_input<1>(ctx, "y2"),
                                      get_output<1>(ctx, "f")});
    }
    void backward(nc::Context& ctx, sycl::queue& q) override {
        execute_compute(q, ObjBackward{get_input<1>(ctx, "x"),
                                       get_input<1>(ctx, "z2"),
                                       get_input<1>(ctx, "y1"),
                                       get_input<1>(ctx, "y2"),
                                       get_output<1>(ctx, "f")});
    }
};

// ── The MDA system: composition, coupling cycle, shared design variables ─────

class SellarMDA final : public nc::System {
public:
    SellarMDA()
        : dis1_(std::make_shared<Discipline1>()),
          dis2_(std::make_shared<Discipline2>()),
          obj_(std::make_shared<SellarObjective>()) {}

    void declare(nc::Context& ctx) override {
        // dis1 is the register cut: added via add_entry_point (which adds the
        // subsystem AND marks it), so the connection INTO it (dis2/y2 ->
        // dis1/y2) becomes a previous-sweep read. That turns the cycle into a
        // layered DAG the executor can sweep — Gauss-Seidel MDA, expressed
        // structurally.
        add_entry_point(ctx, "dis1", dis1_);
        add_subsystem(ctx, "dis2", dis2_);
        add_subsystem(ctx, "obj",  obj_);

        // Coupling: y1 feeds dis2; y2 feeds BACK into dis1 (the cycle).
        connect(ctx, "dis1/y1", "dis2/y1");
        connect(ctx, "dis2/y2", "dis1/y2");
        connect(ctx, "dis1/y1", "obj/y1");
        connect(ctx, "dis2/y2", "obj/y2");

        // Shared design variables: one storage each, aliased into every
        // consumer (connection = pointer aliasing; no copies anywhere).
        promote_input<1>(ctx, "z1", sycl::range<1>{1}, {"dis1/z1", "dis2/z1"});
        promote_input<1>(ctx, "z2", sycl::range<1>{1},
                         {"dis1/z2", "dis2/z2", "obj/z2"});
        promote_input<1>(ctx, "x", sycl::range<1>{1}, {"dis1/x", "obj/x"});
        assign_design_var(ctx, "z1");
        assign_design_var(ctx, "z2");
        assign_design_var(ctx, "x");
    }

private:
    std::shared_ptr<Discipline1>     dis1_;
    std::shared_ptr<Discipline2>     dis2_;
    std::shared_ptr<SellarObjective> obj_;
};

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    auto dev = sycl::device{sycl::default_selector_v};
    std::printf("[sellar] device: %s\n",
                dev.get_info<sycl::info::device::name>().c_str());

    auto sk_ctx = slabkit::ContextBuilder()
                      .devices({dev})
                      .node_size(256)
                      .slab_size_per_device(8ull << 20)  // 8 MiB is plenty
                      .build();
    auto ctx = std::make_shared<nc::Context>(sk_ctx, 0);
    auto sys = std::make_shared<SellarMDA>();

    nc::EquilibriumConfig ecfg;
    ecfg.fwd_tol        = 1e-6f;
    ecfg.adj_tol        = 1e-6f;
    ecfg.max_fwd_iters  = 200;
    ecfg.max_adj_iters  = 100;
    ecfg.check_interval = 1;
    ecfg.anderson_depth = 0;  // plain Richardson — this fixed point is tame

    auto exec = nc::EquilibriumExecutor::create(sys, ctx, ecfg);
    exec->setup();
    exec->init();

    // NOTE: bind the queue AFTER setup() — the queue pool is built at
    // finalize, so earlier references would dangle.
    auto& q = ctx->queue();

    // Classic Sellar starting point: x=1, z1=5, z2=2.
    nc::fill(q, sys->get_initializer<1>(*ctx, "x"),  1.0f);
    nc::fill(q, sys->get_initializer<1>(*ctx, "z1"), 5.0f);
    nc::fill(q, sys->get_initializer<1>(*ctx, "z2"), 2.0f);

    naga::cospan::opt::AdamConfig acfg;
    acfg.learning_rate            = 2e-2f;
    acfg.max_iterations           = 1000;
    acfg.reset_between_iterations = false;
    acfg.callback_interval        = 100;
    naga::cospan::opt::AdamOptimizer adam(acfg);

    auto res = adam.optimize(*exec);

    float xv, z1v, z2v, y1v, y2v, fv;
    nc::download(q, &xv,  sys->get_input<1>(*ctx, "dis1/x"),  1);
    nc::download(q, &z1v, sys->get_input<1>(*ctx, "dis1/z1"), 1);
    nc::download(q, &z2v, sys->get_input<1>(*ctx, "dis1/z2"), 1);
    nc::download(q, &y1v, sys->get_output<1>(*ctx, "dis1/y1"), 1);
    nc::download(q, &y2v, sys->get_output<1>(*ctx, "dis2/y2"), 1);
    nc::download(q, &fv,  sys->get_output<1>(*ctx, "obj/f"),  1);

    std::printf("[sellar] finished: %zu iterations, objective %.6f\n",
                res.iterations, res.objective_value);
    std::printf("[sellar]   x  = %+.4f   (expect ~0)\n", xv);
    std::printf("[sellar]   z1 = %+.4f   (expect ~1.978)\n", z1v);
    std::printf("[sellar]   z2 = %+.4f   (expect ~0)\n", z2v);
    std::printf("[sellar]   y1 = %+.4f   (expect ~3.16, constraint-active)\n", y1v);
    std::printf("[sellar]   y2 = %+.4f   (expect ~3.755)\n", y2v);
    std::printf("[sellar]   f  = %+.4f   (known optimum ~3.183)\n", fv);

    // Loose sanity gate so this doubles as a smoke test.
    const bool ok = fv > 2.5f && fv < 4.0f && y1v > 2.9f && z1v > 1.5f &&
                    z1v < 2.5f;
    std::printf("[sellar] %s\n", ok ? "OK" : "FAILED (outside expected basin)");
    return ok ? 0 : 1;
}
