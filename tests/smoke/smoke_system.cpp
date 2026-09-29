// Smoke check for System (not the test suite). The System tree is driven by a
// test-only root driver that performs, per verb, what the Model will.
//
//   a. Sellar-shaped forward. SellarMDA{d1, d2} ties z, promotes y1/y2/yy/x;
//      Top{mda, obj, g1, g2} ties x and z again (a tie of ties), connects by
//      promoted names, feeds obj.y2 through a Slice of an output vector, marks
//      design variables / objective / constraints, and checks each result
//      against a host reference. d1's free input y2 is set in d1's own init
//      (99) and then in the MDA System's init (1.5): the System pass has the
//      last word.
//
// Slice source note: an input cannot be a connect source (r01 §8.1), so the
// Slice reads element 1 of an OUTPUT vector (d2's yy = {y1, y2}), not of z.

#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;
using cs::detail::HostViewAccess;

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// ---- the test-only root driver --------------------------------------------------
namespace naga::cospan::detail {

struct RootDriver {
    sycl::device dev;
    sk::queue&   q;
    cs::System&  root;
    cs::Substrates subs;

    RootDriver(const sycl::device& d, sk::queue& queue, cs::System& r) : dev{d}, q{queue}, root{r} {}

    /// run_setup -> get_exec_info -> substrates (arena from bytes if > 0, pool)
    /// -> prepare_exec -> compile -> run_init -> run_system_init.
    void start() {
        SubsystemAccess::place(root, "top", "top", nullptr);
        root.run_setup();
        std::size_t total = 0;
        for (const auto& [d, bytes] : root.get_exec_info(dev)) total += bytes;
        cs::MemorySubstrate m;
        // Slack for the arena's per-allocation alignment.
        if (total > 0) m.arena.emplace(q, dev, sk::kind::device, total + 4096 * 64);
        m.pool.emplace(q, dev, sk::kind::device, 1 << 20);
        subs.emplace(dev, std::move(m));
        root.prepare_exec(q, subs);
        root.compile();
        cs::ExecCtx ctx{dev, &q, nullptr, &subs.at(dev)};
        root.run_init(ctx);
        root.run_system_init();
    }

    void forward() {
        (void)root.run_compute_forward();
        q.wait();
    }
};

}  // namespace naga::cospan::detail

using cs::detail::RootDriver;

// ---- a. the Sellar-shaped forward ---------------------------------------------------
constexpr float kInitY2   = 1.5f;   // the MDA System's init: the last word
constexpr float kD1InitY2 = 99.0f;  // d1's own init: overridden

struct D1Body {
    struct Inputs {
        cs::Input<DType::f32, 1> z;
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> y2;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y1;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        out.y1[i] = in.z[std::size_t{0}] * in.z[std::size_t{0}] + in.z[std::size_t{1}] + in.x[std::size_t{0}] -
                    0.2f * in.y2[std::size_t{0}];
    }
};

class D1 : public cs::Component<D1Body> {
    cs::Input<DType::f32, 1>  z_, x_, y2_;
    cs::Output<DType::f32, 1> y1_;

public:
    void setup() override {
        declare_input(z_, "z", 2);
        declare_input(x_, "x", 1);
        declare_input(y2_, "y2", 1);
        declare_output(y1_, "y1", 1);
    }
    void init(sk::queue&, const sycl::device&) override { fill_input("y2", kD1InitY2); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y1_.shape(), {z_, x_, y2_}, {y1_}, {});
    }
};

struct D2Body {
    struct Inputs {
        cs::Input<DType::f32, 1> z;
        cs::Input<DType::f32, 1> y1_in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y2;
        cs::Output<DType::f32, 1> yy;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        const float y1 = in.y1_in[std::size_t{0}];
        const float y2 = sycl::sqrt(y1) + in.z[std::size_t{0}] + in.z[std::size_t{1}];
        out.y2[i]                = y2;
        out.yy[std::size_t{0}]   = y1;
        out.yy[std::size_t{1}]   = y2;
    }
};

class D2 : public cs::Component<D2Body> {
    cs::Input<DType::f32, 1>  z_, y1_in_;
    cs::Output<DType::f32, 1> y2_, yy_;

public:
    void setup() override {
        declare_input(z_, "z", 2);
        declare_input(y1_in_, "y1_in", cs::from_source_shape<1>());
        declare_output(y2_, "y2", 1);
        declare_output(yy_, "yy", 2);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y2_.shape(), {z_, y1_in_}, {y2_, yy_}, {});
    }
};

struct ObjBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> z;
        cs::Input<DType::f32, 1> y1;
        cs::Input<DType::f32, 1> y2;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> f;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        const float x = in.x[std::size_t{0}];
        out.f[i] = x * x + in.z[std::size_t{1}] + in.y1[std::size_t{0}] + sycl::exp(-in.y2[std::size_t{0}]);
    }
};

class Obj : public cs::Component<ObjBody> {
    cs::Input<DType::f32, 1>  x_, z_, y1_, y2_;
    cs::Output<DType::f32, 1> f_;

public:
    void setup() override {
        declare_input(x_, "x", 1);
        declare_input(z_, "z", 2);
        declare_input(y1_, "y1", cs::from_source_shape<1>());
        declare_input(y2_, "y2", cs::from_source_shape<1>());
        declare_output(f_, "f", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(f_.shape(), {x_, z_, y1_, y2_}, {f_}, {});
    }
};

// g1 = 3.16 - y1, g2 = y2 - 24: one body each.
struct G1Body {
    struct Inputs {
        cs::Input<DType::f32, 1> y1;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> g;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.g[i] = 3.16f - in.y1[std::size_t{0}]; }
};

class G1 : public cs::Component<G1Body> {
    cs::Input<DType::f32, 1>  y1_;
    cs::Output<DType::f32, 1> g_;

public:
    void setup() override {
        declare_input(y1_, "y1", cs::from_source_shape<1>());
        declare_output(g_, "g", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(g_.shape(), {y1_}, {g_}, {});
    }
};

struct G2Body {
    struct Inputs {
        cs::Input<DType::f32, 1> y2;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> g;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.g[i] = in.y2[std::size_t{0}] - 24.0f; }
};

class G2 : public cs::Component<G2Body> {
    cs::Input<DType::f32, 1>  y2_;
    cs::Output<DType::f32, 1> g_;

public:
    void setup() override {
        declare_input(y2_, "y2", cs::from_source_shape<1>());
        declare_output(g_, "g", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(g_.shape(), {y2_}, {g_}, {});
    }
};

class SellarMDA : public cs::System {
public:
    void setup() override {
        add_subsystem<D1>("d1");
        add_subsystem<D2>("d2");
        connect("d1.y1", "d2.y1_in");
        tie_inputs({"d1.z", "d2.z"}, "z");
        promote("d1.x", "x");
        promote("d1.y1", "y1");
        promote("d2.y2", "y2");
        promote("d2.yy", "yy");
    }
    // The System pass runs after d1's own init: this is the last word.
    void init(sk::queue&, const sycl::device&) override { fill_input("d1.y2", kInitY2); }
};

class Top : public cs::System {
public:
    void setup() override {
        add_subsystem<SellarMDA>("mda");
        add_subsystem<Obj>("obj");
        add_subsystem<G1>("g1");
        add_subsystem<G2>("g2");

        tie_inputs({"mda.x", "obj.x"}, "x");
        tie_inputs({"mda.z", "obj.z"}, "z");  // a tie of a tie alias
        connect("mda.y1", "obj.y1");
        connect("mda.yy", "obj.y2", cs::Slice{cs::subarray<1>({1}, {1})});  // element 1 of yy = y2
        connect("mda.y1", "g1.y1");
        connect("mda.y2", "g2.y2");

        add_design_var("z");
        add_design_var("x");
        add_objective("obj.f");
        cs::Bounds first;
        first.upper = std::vector<double>{1.0};
        add_constraint("g1.g", first);  // replaced by the next mark of the kind
        cs::Bounds second;
        second.upper = std::vector<double>{0.0};
        add_constraint("g1.g", second);
        cs::Bounds b2;
        b2.upper = std::vector<double>{0.0};
        add_constraint("g2.g", b2);
    }
    void init(sk::queue&, const sycl::device&) override {
        const float z[2] = {5.0f, 2.0f};
        set_input_from_host<float>("z", std::span<const float>{z, 2});
        fill_input("x", 1.0f);
    }
};

static bool near(float got, float want) {
    return std::fabs(got - want) <= 1e-5f * std::fmax(1.0f, std::fabs(want));
}

static float scalar_of(sk::queue& q, cs::detail::IOEntry& e) {
    auto h = HostViewAccess::view<DType::f32, 1>(e, q);
    return h.size() == 1 ? h[std::size_t{0}] : std::nanf("");
}

static void case_a(const sycl::device& dev, sk::queue& q) {
    std::printf("-- a. Sellar-shaped forward: mda{d1,d2} + obj + g1 + g2, ties, promotions, a Slice, marks\n");
    Top         top;
    RootDriver  d{dev, q, top};
    d.start();
    d.forward();

    // Host reference in the same float arithmetic.
    const float z0 = 5.0f, z1 = 2.0f, x = 1.0f;
    const float y1 = z0 * z0 + z1 + x - 0.2f * kInitY2;
    const float y2 = std::sqrt(y1) + z0 + z1;
    const float f  = x * x + z1 + y1 + std::exp(-y2);
    const float g1 = 3.16f - y1, g2 = y2 - 24.0f;

    check(near(scalar_of(q, top.entry_at("mda.d1.y1")), y1), "y1 == z0^2 + z1 + x - 0.2*y2 (y2 = the System-init value)");
    check(scalar_of(q, top.entry_at("mda.d1.y2")) == kInitY2, "d1.y2: the System pass init has the last word over d1's own init");
    check(near(scalar_of(q, top.entry_at("mda.d2.y2")), y2), "y2 == sqrt(y1) + z0 + z1");
    check(near(scalar_of(q, top.entry_at("mda.y2")), y2), "the promoted alias mda.y2 reads d2.y2");
    check(near(scalar_of(q, top.entry_at("obj.f")), f), "f == x^2 + z1 + y1 + exp(-y2) (obj.y2 via the Slice of yy)");
    check(near(scalar_of(q, top.entry_at("g1.g")), g1), "g1 == 3.16 - y1");
    check(near(scalar_of(q, top.entry_at("g2.g")), g2), "g2 == y2 - 24");

    // The Slice: obj.y2 sees element 1 of yy.
    const auto& v = top.entry_at("obj.y2").view();
    check(v.has_value() && v->offset[0] == 1 && v->extents[0] == 1, "obj.y2 carries the view {offset 1, extent 1}");

    // Marks ride on the roots; the last mark of a kind wins.
    const auto& zdv = top.entry_at("z").design_var();
    const auto& xdv = top.entry_at("x").design_var();
    check(zdv.has_value() && xdv.has_value(), "design variables z and x are recorded");
    check(zdv && zdv->root == top.entry_at("mda.d1.z").root() && zdv->root == top.entry_at("obj.z").root(),
          "z's mark sits on the ONE root shared by d1.z, d2.z and obj.z");
    check(top.entry_at("obj.f").objective().has_value(), "objective f is recorded");
    const auto& c1 = top.entry_at("g1.g").constraint();
    check(c1.has_value() && c1->bounds.upper && (*c1->bounds.upper)[0] == 0.0,
          "a later add_constraint on g1 replaced the first (upper 1.0 -> 0.0)");
    check(top.entry_at("g2.g").constraint().has_value(), "constraint g2 is recorded");
}

int main() {
    try {
        sycl::device dev{sycl::default_selector_v};
        std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        sk::context ctx(dev);
        sk::queue   q(ctx, dev);
        case_a(dev, q);
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    std::printf(failures == 0 ? "SMOKE SYSTEM PASS\n" : "SMOKE SYSTEM FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
