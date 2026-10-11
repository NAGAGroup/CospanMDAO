// Model tests, every case through a cospan::Model (create_model / setup / forward /
// wait / view_* / update_input); the only reach past those is the Model's own retrieval
// (root().instance_at, get_subsystem).
//
// Reading results (docs/design-review-2026-10-03.md, section 5): forward() only enqueues; a host view is obtained
// after the first forward and sync()ed before it is read (cst::synced_output and friends do both).
//
//   1. Sellar forward   the same Systems templated on the dtype, run in f32 and f64
//                       over several design points against a host reference in double
//   2. Marks            design variables, objective, constraints: counts, bounds, and
//                       the tied z appearing once (one record per root)
//   3. Lifecycle        run / view_output before setup, setup twice, a moved-from
//                       Model
//   4. Host access      unknown path, view_input on an output, update_input on an output
//   5. Memory report    a small static model computes; the pool report (there is no arena)
//   6. Pool             a pool on the device and a value-driven (shape hook) output root
//   7. Two devices      producer and consumer on different devices of one context
//   9. Descriptors      create_model forms, model_config_desc, pool_desc / node_size_desc / async_handler_desc, the overlap errors,
//                       uncovered devices, memory_report contents
//
// The components live in namespace tmod.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <functional>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;
using cst::default_config;
using cst::to_vector;

namespace tmod {

/// `f` must throw with every needle in its message.
template <class F>
void expect_throws(F&& f, const std::vector<std::string>& needles) {
    std::string what = "<no exception>";
    try {
        f();
    } catch (const std::exception& e) {
        what = e.what();
    }
    INFO("thrown: " << what);
    for (const auto& n : needles) CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}

// ---- the Sellar Systems, templated on the dtype ---------------------------------------------------
//   d1:  y1 = z0^2 + z1 + x - 0.2 y2        (y2 is a free input, set from the host)
//   d2:  y2 = sqrt(y1) + z0 + z1            (y1 connected from d1)
//   obj: f  = x^2 + z1 + y1 + exp(-y2)      (y1 from d1, y2 from d2)
//   g1 = 3.16 - y1, g2 = y2 - 24
template <DType DT>
using T_ = cs::cpp_type_t<DT>;

template <DType DT>
class D1 : public cs::Component<D1<DT>> {
public:
    struct Inputs {
        cs::Input<DT, 1> z;
        cs::Input<DT, 1> x;
        cs::Input<DT, 1> y2;
    };
    struct Outputs {
        cs::Output<DT, 1> y1;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        out.y1[i] = in.z[std::size_t{0}] * in.z[std::size_t{0}] + in.z[std::size_t{1}] + in.x[std::size_t{0}] -
                    static_cast<T_<DT>>(0.2) * in.y2[std::size_t{0}];
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        this->declare_input(inputs.z, "z", cs::unspecified_shape<1>());  // z and x are tie members: the tie alias owns their shape
        this->declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        this->declare_input(inputs.y2, "y2", 1);
        this->declare_output(outputs.y1, "y1", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(outputs.y1.shape()));
    }
};

template <DType DT>
class D2 : public cs::Component<D2<DT>> {
public:
    struct Inputs {
        cs::Input<DT, 1> z;
        cs::Input<DT, 1> y1;
    };
    struct Outputs {
        cs::Output<DT, 1> y2;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        out.y2[i] = sycl::sqrt(in.y1[std::size_t{0}]) + in.z[std::size_t{0}] + in.z[std::size_t{1}];
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        this->declare_input(inputs.z, "z", cs::unspecified_shape<1>());
        this->declare_input(inputs.y1, "y1", cs::unspecified_shape<1>());
        this->declare_output(outputs.y2, "y2", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(outputs.y2.shape()));
    }
};

template <DType DT>
class Obj : public cs::Component<Obj<DT>> {
public:
    struct Inputs {
        cs::Input<DT, 1> x;
        cs::Input<DT, 1> z;
        cs::Input<DT, 1> y1;
        cs::Input<DT, 1> y2;
    };
    struct Outputs {
        cs::Output<DT, 1> f;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        const T_<DT> x = in.x[std::size_t{0}];
        out.f[i]       = x * x + in.z[std::size_t{1}] + in.y1[std::size_t{0}] + sycl::exp(-in.y2[std::size_t{0}]);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        this->declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        this->declare_input(inputs.z, "z", cs::unspecified_shape<1>());
        this->declare_input(inputs.y1, "y1", cs::unspecified_shape<1>());
        this->declare_input(inputs.y2, "y2", cs::unspecified_shape<1>());
        this->declare_output(outputs.f, "f", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(outputs.f.shape()));
    }
};

template <DType DT>
class G1 : public cs::Component<G1<DT>> {
public:
    struct Inputs {
        cs::Input<DT, 1> y1;
    };
    struct Outputs {
        cs::Output<DT, 1> g;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        out.g[i] = static_cast<T_<DT>>(3.16) - in.y1[std::size_t{0}];
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        this->declare_input(inputs.y1, "y1", cs::unspecified_shape<1>());
        this->declare_output(outputs.g, "g", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(outputs.g.shape()));
    }
};

template <DType DT>
class G2 : public cs::Component<G2<DT>> {
public:
    struct Inputs {
        cs::Input<DT, 1> y2;
    };
    struct Outputs {
        cs::Output<DT, 1> g;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        out.g[i] = in.y2[std::size_t{0}] - static_cast<T_<DT>>(24.0);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        this->declare_input(inputs.y2, "y2", cs::unspecified_shape<1>());
        this->declare_output(outputs.g, "g", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(outputs.g.shape()));
    }
};

/// z is tied across d1, d2 and obj; x across d1 and obj; the marks are the classic Sellar set.
template <DType DT>
class SellarTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<D1<DT>>("d1"));
        add_subsystem(cs::subsystem_desc<D2<DT>>("d2"));
        add_subsystem(cs::subsystem_desc<Obj<DT>>("obj"));
        add_subsystem(cs::subsystem_desc<G1<DT>>("g1"));
        add_subsystem(cs::subsystem_desc<G2<DT>>("g2"));

        tie_inputs<DT, 1>({"d1.z", "d2.z", "obj.z"}, "z", cs::shape(2));
        tie_inputs<DT, 1>({"d1.x", "obj.x"}, "x", cs::shape(1));
        connect("d1.y1", "d2.y1");
        connect("d1.y1", "obj.y1");
        connect("d2.y2", "obj.y2");
        connect("d1.y1", "g1.y1");
        connect("d2.y2", "g2.y2");

        cs::Bounds zb;
        zb.lower = std::vector<double>{-10.0, 0.0};
        zb.upper = std::vector<double>{10.0, 10.0};
        add_design_var("z", zb);
        cs::Bounds xb;
        xb.lower = std::vector<double>{0.0};
        xb.upper = std::vector<double>{10.0};
        add_design_var("x", xb);
        add_objective("obj.f");
        cs::Bounds c1;
        c1.upper = std::vector<double>{0.0};
        add_constraint("g1.g", c1);
        cs::Bounds c2;
        c2.upper = std::vector<double>{0.0};
        add_constraint("g2.g", c2);
    }
};

struct Point {
    double z0, z1, x, y2;
};
struct Ref {
    double y1, y2, f, g1, g2;
};

Ref sellar_ref(double z0, double z1, double x, double y2_in) {
    Ref r;
    r.y1 = z0 * z0 + z1 + x - 0.2 * y2_in;
    r.y2 = std::sqrt(r.y1) + z0 + z1;
    r.f  = x * x + z1 + r.y1 + std::exp(-r.y2);
    r.g1 = 3.16 - r.y1;
    r.g2 = r.y2 - 24.0;
    return r;
}

/// One value of a rank-1 output.
template <DType DT>
T_<DT> scalar_out(cs::Model& m, const char* path) {
    auto v = cst::synced_output<DT, 1>(m, path);
    REQUIRE(v.size() == 1);
    return v[std::size_t{0}];
}

template <DType DT>
void check_near(cs::Model& m, const char* path, double want, double abs_tol, double rel_tol, int point) {
    const double got = static_cast<double>(scalar_out<DT>(m, path));
    INFO("point " << point << ", " << path << ": got " << got << ", want " << want);
    CHECK(cst::close(got, want, abs_tol, rel_tol));
}

template <DType DT>
void sellar_forward(double abs_tol, double rel_tol) {
    using T = T_<DT>;
    cs::Model model = cs::create_model<SellarTop<DT>>(default_config());  // a concrete type: no `template` keyword needed
    model.setup();

    // (z0, z1, x, d1.y2); every y1 stays positive.
    const std::vector<Point> points = {
        {5.0, 2.0, 1.0, 1.5},     // the classic start
        {1.9776, 0.0, 0.0, 3.75}, // near the optimum: g1 close to zero
        {3.0, 1.0, 0.5, 2.0},
        {0.5, 4.0, 2.0, 0.0},
        {-2.0, 1.0, 3.0, 1.0},
    };
    int n = 0;
    for (const Point& p : points) {
        const std::array<T, 2> z{static_cast<T>(p.z0), static_cast<T>(p.z1)};
        const T                x = static_cast<T>(p.x), y2 = static_cast<T>(p.y2);
        model.update_input("z", cs::init::from_host(std::vector<T>{z.begin(), z.end()}));
        model.update_input("x", cs::init::from_host(std::vector<T>{x}));
        model.update_input("d1.y2", cs::init::from_host(std::vector<T>{y2}));
        cst::forward_and_wait(model);
        CHECK(cst::synced_input<DT, 1>(model, "d2.z")[std::size_t{1}] == z[1]);  // the tie reads back everywhere
        CHECK(cst::synced_input<DT, 1>(model, "obj.x")[std::size_t{0}] == x);

        // The reference runs in double on the values the device actually saw.
        const Ref r = sellar_ref(static_cast<double>(z[0]), static_cast<double>(z[1]), static_cast<double>(x),
                                 static_cast<double>(y2));
        REQUIRE(r.y1 > 0.0);
        check_near<DT>(model, "d1.y1", r.y1, abs_tol, rel_tol, n);
        check_near<DT>(model, "d2.y2", r.y2, abs_tol, rel_tol, n);
        check_near<DT>(model, "obj.f", r.f, abs_tol, rel_tol, n);
        check_near<DT>(model, "g1.g", r.g1, abs_tol, rel_tol, n);
        check_near<DT>(model, "g2.g", r.g2, abs_tol, rel_tol, n);
        ++n;
    }
}

// ---- Static memory model (case 5) ------------------------------------------------------------------
//   gen:  y  f32[100]    400 B          (root)
//   cvt:  a  <- gen.y    (a connected input: a peer of gen.y, no storage of its own)
//         d  f64[64]     512 B          (root)
//   comb: b, c f32[8]    tied into the top alias "bc": one root, 32 B
//         e  <- cvt.d    (a connected input: a peer)
//         o  f32[16]     64 B           (root), promoted to "out" (a peer alias)
// Distinct roots: gen.y, cvt.d, bc, comb.o.
constexpr std::size_t kGenBytes = 100 * sizeof(float);
constexpr std::size_t kCvtBytes = 64 * sizeof(double);
constexpr std::size_t kBcBytes  = 8 * sizeof(float);
constexpr std::size_t kOutBytes = 16 * sizeof(float);
constexpr std::size_t kExpectedStatic = kGenBytes + kCvtBytes + kBcBytes + kOutBytes;  // 1008

class Gen : public cs::Component<Gen> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs&, const Outputs& out) { out.y[i] = static_cast<float>(i[0]); }

    Inputs  inputs;
    Outputs outputs;

    void setup() override { declare_output(outputs.y, "y", 100); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};

class Cvt : public cs::Component<Cvt> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> a;
    };
    struct Outputs {
        cs::Output<DType::f64, 1> d;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.d[i] = 2.0 * static_cast<double>(in.a[i]); }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.a, "a", cs::unspecified_shape<1>());
        declare_output(outputs.d, "d", 64);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.d.shape()));
    }
};

class Comb : public cs::Component<Comb> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> b;
        cs::Input<DType::f32, 1> c;
        cs::Input<DType::f64, 1> e;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> o;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        const std::size_t k = i[0] % 8;
        out.o[i]            = in.b[k] + in.c[k] + static_cast<float>(in.e[std::size_t{0}]);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.b, "b", cs::unspecified_shape<1>());  // tied into the top alias "bc" (fixed 8)
        declare_input(inputs.c, "c", cs::unspecified_shape<1>());
        declare_input(inputs.e, "e", cs::unspecified_shape<1>());
        declare_output(outputs.o, "o", 16);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.o.shape()));
    }
};

class MemTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Gen>("gen"));
        add_subsystem(cs::subsystem_desc<Cvt>("cvt"));
        add_subsystem(cs::subsystem_desc<Comb>("comb"));
        tie_inputs<DType::f32, 1>({"comb.b", "comb.c"}, "bc", cs::shape(8));
        connect("gen.y", "cvt.a");
        connect("cvt.d", "comb.e");
        promote<DType::f32, 1>("comb.o", "out");
    }
};

// ---- Value-driven size on a pool (case 6) -----------------------------------------------------------
// Bins: a free u32 input n and an output of n floats (bins[i] = i + 1).
// Bins tests a hook on an INPUT: it takes a free input `ref` that its System hooks with from_value(n) (hook_bins) and
// copies its shape in pre_forward. (Hooks on outputs are tested in test_shapes.cpp.)

class Bins : public cs::Component<Bins> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> n;
        cs::Input<DType::f32, 1> ref;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> bins;
    };
    static void compute_impl(sycl::id<1> i, const Inputs&, const Outputs& out) { out.bins[i] = static_cast<float>(i[0]) + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.n, "n", 1);
        declare_input(inputs.ref, "ref", cs::unspecified);
        declare_output(outputs.bins, "bins", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.bins, shape_of(inputs.ref)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.bins.shape()));
    }
};
class PoolTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Bins>("bins"));
        shape_hook("bins.ref", cs::dim::from_value("bins.n", 0));
    }
};

// ---- Two devices (case 7) ---------------------------------------------------------------------------
constexpr std::size_t kLen = 8;

class Axpb : public cs::Component<Axpb> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> a, b;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = in.a[std::size_t{0}] * in.x[i] + in.b[std::size_t{0}]; }

    Inputs  inputs;
    Outputs outputs;

private:
    float                     a_, b_;

public:
    Axpb(float a, float b) : a_{a}, b_{b} {}
    void setup() override {
        declare_input(inputs.a, "a", 1);
        init_input("a", cs::init::fill(a_));
        declare_input(inputs.b, "b", 1);
        init_input("b", cs::init::fill(b_));
        declare_input(inputs.x, "x", static_cast<int>(kLen));
        declare_output(outputs.y, "y", static_cast<int>(kLen));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};

// ---- Initializer-only inputs and empty roots -------------------------------------------------
// Scale: y = 2 x, x shaped by its source, y by x.

class Scale : public cs::Component<Scale> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i]; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_output(outputs.y, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }  // the output follows the input (section 10: no derived dims)
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};
// bins.n drives bins.bins (n floats); sc doubles them. With n = 0 the root is present and empty.
class EmptyTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Bins>("bins"));
        shape_hook("bins.ref", cs::dim::from_value("bins.n", 0));
        add_subsystem(cs::subsystem_desc<Scale>("sc"));
        connect("bins.bins", "sc.x");
    }
};

// One Axpb (y = x) with a free input a.x.
class OneAxTop : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("a", 1.0f, 0.0f)); }
};

// An Axpb that declares its own initial value 1.
class AxpbInit1 : public Axpb {
public:
    AxpbInit1(float a, float b) : Axpb{a, b} {}
    void setup() override {
        Axpb::setup();
        init_input("x", cs::init::fill(1.0f));
    }
};
// The component declares 1; the System (depth 0) declares 2.
class DeclaredTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<AxpbInit1>("a", 1.0f, 0.0f));
        init_input("a.x", cs::init::fill(2.0f));
    }
};

// p (y = x + 1) feeds q (y = x): q.x is a connected input.
class ChainTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("p", 1.0f, 1.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("q", 1.0f, 0.0f));
        connect("p.y", "q.x");
    }
};

int g_host_inits = 0;  // counts the custom_host initializer's runs

// src (default device): y = x. mid (the other device): y = 2 src.y + 1. out (default device): y = mid.y + 2.
class TwoDevTop : public cs::System {
    sycl::device other_;

public:
    explicit TwoDevTop(const sycl::device& other) : other_{other} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("src", 1.0f, 0.0f));
        add_subsystem(other_, cs::subsystem_desc<Axpb>("mid", 2.0f, 1.0f));  // the device is an add_subsystem argument (section 2)
        add_subsystem(cs::subsystem_desc<Axpb>("out", 1.0f, 2.0f));
        connect("src.y", "mid.x");
        connect("mid.y", "out.x");
    }
};

/// A non-OpenMP device and the OpenMP host device, when both are visible.
bool pick_two(sycl::device& first, sycl::device& host) {
    bool have_first = false, have_host = false;
    for (const sycl::device& d : sycl::device::get_devices()) {
        if (d.get_backend() == sycl::backend::omp) {
            if (!have_host) host = d;
            have_host = true;
        } else {
            if (!have_first) first = d;
            have_first = true;
        }
    }
    return have_first && have_host;
}

cs::Model::MemoryReport::PerDevice report_on(const cs::Model& m, const sycl::device& dev) {
    const auto rep = m.memory_report();
    const auto it  = rep.devices.find(dev);
    REQUIRE(it != rep.devices.end());
    return it->second;
}

}  // namespace tmod

// =============================================================================================
// 1. Sellar forward
// =============================================================================================
TEST_CASE("Sellar forward: f32 and f64 over several design points", "[model][sellar]") {
    SECTION("f32: rel 1e-5, abs 1e-5") { tmod::sellar_forward<DType::f32>(1e-5, 1e-5); }
    SECTION("f64: rel 1e-12") { tmod::sellar_forward<DType::f64>(1e-12, 1e-12); }
}

// =============================================================================================
// 2. Marks
// =============================================================================================
TEST_CASE("Marks: design variables, objective, constraints, one record per root", "[model][marks]") {
    auto model = cs::create_model<tmod::SellarTop<DType::f32>>(default_config());
    model.setup();
    auto& sys = model.root();

    REQUIRE(model.design_vars().size() == 2);
    REQUIRE(model.objectives().size() == 1);
    REQUIRE(model.constraints().size() == 2);

    // A Mark is {path, inst, rec}; the instance is held weakly (design review, section 1) and the variable it names
    // is its `meta`, shared by every instance of the variable. This replaces the old per-root entry comparison.
    const auto zr = sys.instance_at("z")->meta;
    const auto xr = sys.instance_at("x")->meta;
    // The three components' z entries share the one variable, so one record serves all three.
    CHECK(sys.instance_at("d1.z")->meta == zr);
    CHECK(sys.instance_at("d2.z")->meta == zr);
    CHECK(sys.instance_at("obj.z")->meta == zr);
    CHECK(sys.instance_at("d1.x")->meta == xr);
    CHECK(sys.instance_at("obj.x")->meta == xr);

    std::size_t on_z = 0, on_x = 0;
    for (const auto& d : model.design_vars()) {
        const auto inst = d.inst.lock();
        REQUIRE(inst);
        if (inst->meta == zr) ++on_z;
        if (inst->meta == xr) ++on_x;
    }
    CHECK(on_z == 1);
    CHECK(on_x == 1);

    for (const auto& d : model.design_vars()) {
        const auto inst = d.inst.lock();
        REQUIRE(inst);
        REQUIRE(d.rec.bounds.lower.has_value());
        REQUIRE(d.rec.bounds.upper.has_value());
        if (inst->meta == zr) {
            CHECK(*d.rec.bounds.lower == std::vector<double>{-10.0, 0.0});
            CHECK(*d.rec.bounds.upper == std::vector<double>{10.0, 10.0});
        } else {
            CHECK(*d.rec.bounds.lower == std::vector<double>{0.0});
            CHECK(*d.rec.bounds.upper == std::vector<double>{10.0});
        }
        CHECK_FALSE(d.rec.equals.has_value());
    }

    {
        const auto inst = model.objectives()[0].inst.lock();
        REQUIRE(inst);
        CHECK(inst->meta == sys.instance_at("obj.f")->meta);
    }

    const auto g1 = sys.instance_at("g1.g")->meta;
    const auto g2 = sys.instance_at("g2.g")->meta;
    std::size_t on_g1 = 0, on_g2 = 0;
    for (const auto& c : model.constraints()) {
        const auto inst = c.inst.lock();
        REQUIRE(inst);
        if (inst->meta == g1) ++on_g1;
        if (inst->meta == g2) ++on_g2;
        REQUIRE(c.rec.bounds.upper.has_value());
        CHECK(*c.rec.bounds.upper == std::vector<double>{0.0});
        CHECK_FALSE(c.rec.bounds.lower.has_value());
    }
    CHECK(on_g1 == 1);
    CHECK(on_g2 == 1);
}

namespace tmod {
/// Marks under a run policy (design review, section 9). `mode`: 0 eternal; 1 transient; 2 do_once; 3 run_n{2};
/// 4 an eternal System `in` (marking its own Axpb) under a transient `outer`.
class MarkInner : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("a", 1.0f, 0.0f));
        add_design_var("a.x");
    }
};
class MarkOuter : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<MarkInner>("in")); }
};
class MarkScopeTop : public cs::System {
    int  mode_;
    char kind_;  // 'd' design var, 'c' constraint

public:
    MarkScopeTop(int mode, char kind) : mode_{mode}, kind_{kind} {}
    void setup() override {
        if (mode_ == 4) {
            add_subsystem(cs::property_list{cs::property::transient{}}, cs::subsystem_desc<MarkOuter>("outer"));
            return;
        }
        const cs::property_list props = mode_ == 1   ? cs::property_list{cs::property::transient{}}
                                        : mode_ == 2 ? cs::property_list{cs::property::do_once{}}
                                        : mode_ == 3 ? cs::property_list{cs::property::run_n{2}}
                                                     : cs::property_list{};
        add_subsystem(props, cs::subsystem_desc<Axpb>("a", 1.0f, 0.0f));
        if (kind_ == 'd')
            add_design_var("a.x");
        else
            add_constraint("a.y");
    }
};
}  // namespace tmod

TEST_CASE("Marks: no scope rule; whatever is collected at setup stays", "[model][marks]") {
    // Design review, 11.2 (P4): marks have no scope rule. The old finalize walk that refused a run policy on the way
    // (transient, do_once, run_n) is gone; every mark is collected, by absolute path.
    SECTION("an eternal subsystem") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 0, 'd');
        CHECK_NOTHROW(model.setup());
        REQUIRE(model.design_vars().size() == 1);
        CHECK(model.design_vars()[0].path == "a.x");
    }
    SECTION("a transient subsystem") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 1, 'd');
        CHECK_NOTHROW(model.setup());
        REQUIRE(model.design_vars().size() == 1);
        CHECK(model.design_vars()[0].path == "a.x");
    }
    SECTION("a do_once subsystem, a constraint") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 2, 'c');
        CHECK_NOTHROW(model.setup());
        REQUIRE(model.constraints().size() == 1);
        CHECK(model.constraints()[0].path == "a.y");
    }
    SECTION("a run_n subsystem") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 3, 'd');
        CHECK_NOTHROW(model.setup());
        REQUIRE(model.design_vars().size() == 1);
    }
    SECTION("an eternal System under a transient one") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 4, 'd');
        CHECK_NOTHROW(model.setup());
        REQUIRE(model.design_vars().size() == 1);
        CHECK(model.design_vars()[0].path == "outer.in.a.x");
    }
}

// =============================================================================================
// 3. Lifecycle errors
// =============================================================================================
TEST_CASE("Lifecycle: errors before setup, setup twice, moved-from", "[model][lifecycle]") {
    using Top = tmod::SellarTop<DType::f32>;

    SECTION("run before setup throws") {
        auto m = cs::create_model<Top>(default_config());
        tmod::expect_throws([&] { m.run(); }, {"before setup"});
    }
    SECTION("view_output before setup throws") {
        auto m = cs::create_model<Top>(default_config());
        tmod::expect_throws([&] { (void)m.view_output<DType::f32, 1>("obj.f"); }, {"before setup"});
    }
    SECTION("setup twice throws") {
        auto m = cs::create_model<Top>(default_config());
        m.setup();
        tmod::expect_throws([&] { m.setup(); }, {"runs once"});
    }
    SECTION("run on a moved-from Model throws; the moved-to Model runs") {
        auto m = cs::create_model<Top>(default_config());
        m.setup();
        cs::Model moved = std::move(m);
        tmod::expect_throws([&] { m.run(); }, {"moved-from"});  // NOLINT(bugprone-use-after-move)
        CHECK_NOTHROW(moved.run());
        CHECK_NOTHROW(moved.wait());
    }
}

// =============================================================================================
// 4. Host access errors
// =============================================================================================
TEST_CASE("Host access: errors name the path; update_input on an output throws", "[model][host]") {
    SECTION("errors") {
        auto model = cs::create_model<tmod::SellarTop<DType::f32>>(default_config());
        model.setup();
        const std::vector<float> two{1.0f, 2.0f};

        // A view can be obtained only after a run (design review, section 5): these cases were about what a view
        // names, so the model runs once first (the old order, a view straight after setup, now throws).
        model.run();

        // Unknown path: the message carries the path it could not resolve.
        tmod::expect_throws([&] { (void)model.view_output<DType::f32, 1>("nope.y"); }, {"nope.y"});
        tmod::expect_throws([&] { (void)model.view_output<DType::f32, 1>("obj.nope"); }, {"obj.nope"});
        // The wrong kind of entry.
        tmod::expect_throws([&] { (void)model.view_input<DType::f32, 1>("obj.f"); }, {"obj.f", "an output"});
        tmod::expect_throws([&] { (void)model.view_output<DType::f32, 1>("d1.z"); }, {"d1.z", "an input"});
        // An update on an output is an error.
        tmod::expect_throws([&] { model.update_input("obj.f", cs::init::from_host(two)); }, {"obj.f", "an output"});
        // The right call still works after that.
        CHECK_NOTHROW(model.update_input("z", cs::init::from_host(two)));
        model.wait();  // the forward above was enqueued; let it finish before the Model goes out of scope
    }
}

// =============================================================================================
// 5. Memory report
// =============================================================================================
TEST_CASE("Memory report: a static model computes; the pool report after setup and after a forward", "[model][memory]") {
    const sycl::device dev{sycl::default_selector_v};
    auto               model = cs::create_model<tmod::MemTop>(cst::pooled_config());
    model.setup();

    // Setup allocates nothing (design review, section 2).
    const auto after_setup = tmod::report_on(model, dev);
    CHECK(after_setup.pool_bytes_in_use == 0);

    std::array<float, 8> bc{};
    for (std::size_t i = 0; i < bc.size(); ++i) bc[i] = static_cast<float>(i + 1);
    model.update_input("bc", cs::init::from_host(std::vector<float>{bc.begin(), bc.end()}));
    cst::forward_and_wait(model);
    const auto after_fwd = tmod::report_on(model, dev);  // between runs, before any host view exists

    // The model computes what it says (so the roots really hold data).
    const auto gen = to_vector(cst::synced_output<DType::f32, 1>(model, "gen.y"));
    const auto cvt = to_vector(cst::synced_output<DType::f64, 1>(model, "cvt.d"));
    const auto out = to_vector(cst::synced_output<DType::f32, 1>(model, "out"));
    REQUIRE(gen.size() == 100);
    REQUIRE(cvt.size() == 64);
    REQUIRE(out.size() == 16);
    CHECK(gen[99] == 99.0f);
    CHECK(cvt[63] == 126.0);
    for (std::size_t i = 0; i < out.size(); ++i) CHECK(out[i] == 2.0f * bc[i % 8]);  // e[0] = cvt.d[0] = 0

    std::printf("[memory] expected static %zu B (gen.y %zu + cvt.d %zu + bc %zu + comb.o %zu)\n"
                "[memory] pool %zu B, in use after setup %zu B, after a forward %zu B; OS allocations %zu (%zu B live)\n",
                tmod::kExpectedStatic, tmod::kGenBytes, tmod::kCvtBytes, tmod::kBcBytes, tmod::kOutBytes, after_fwd.pool_bytes,
                after_setup.pool_bytes_in_use, after_fwd.pool_bytes_in_use, after_fwd.os_allocations, after_fwd.os_bytes_in_use);

    // The rule: the pool's bytes in use on a device are the sum, over the live containers on it, of each container's
    // capacity rounded up to whole nodes. The live containers after a forward are the four distinct roots (the peers
    // cvt.a, comb.e, comb.b/c and the promoted out share their root's container): 400, 512, 32 and 64 bytes, a node each.
    CHECK(after_fwd.pool_bytes == (std::size_t{1} << 20));  // a pool is sized up front
    // Plus what each of the three kernel components keeps for its own launch (argument, pointer and geometry arrays).
    CHECK(after_fwd.pool_bytes_in_use == cst::pool_bytes_of(tmod::kGenBytes) + cst::pool_bytes_of(tmod::kCvtBytes) +
                                             cst::pool_bytes_of(tmod::kBcBytes) + cst::pool_bytes_of(tmod::kOutBytes) +
                                             cst::solo_launch_bytes(3));
    CHECK(after_fwd.pool_bytes_in_use == 4 * cst::pool_node_bytes + cst::solo_launch_bytes(3));
    CHECK(after_fwd.os_allocations == 0);  // everything came from the pool
}

TEST_CASE("Memory report: a device without a configured pool is a zero-byte pool, and its allocations count as OS ones",
          "[model][memory]") {
    // Every context device gets a pool, possibly of 0 bytes (default_config() sets no size); the report lists a device only
    // once it has bytes or has recorded an allocation.
    const sycl::device dev{sycl::default_selector_v};
    auto               model = cs::create_model<tmod::MemTop>(default_config());
    model.setup();
    CHECK(model.memory_report().devices.empty());  // nothing has allocated yet, and no device has pool bytes
    std::array<float, 8> bc{};
    model.update_input("bc", cs::init::from_host(std::vector<float>{bc.begin(), bc.end()}));
    cst::forward_and_wait(model);
    const auto after = tmod::report_on(model, dev);
    CHECK(after.pool_bytes == 0);
    CHECK(after.pool_bytes_in_use == 0);
    CHECK(after.os_allocations > 0);
    CHECK(after.os_bytes_in_use > 0);
}

// =============================================================================================
// 6. Pool
// =============================================================================================
TEST_CASE("Pool: a value-driven root allocates from the pool and follows update_input", "[model][pool]") {
    const sycl::device dev{sycl::default_selector_v};
    // 1 MiB on the default device, in nodes of the default 1024 B
    auto model = cs::create_model<tmod::PoolTop>(cs::model_config_desc{cs::pool_desc{dev, std::size_t{1} << 20}});
    model.setup();
    // A pool is sized up front (section 2): its whole capacity is reported before the first allocation, with nothing in use.
    CHECK(tmod::report_on(model, dev).pool_bytes == (std::size_t{1} << 20));
    CHECK(tmod::report_on(model, dev).pool_bytes_in_use == 0);

    std::uint32_t n = 8;
    model.update_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{n}));
    cst::forward_and_wait(model);
    {
        const auto rep = tmod::report_on(model, dev);
        std::printf("[pool] n=%u: pool %zu B, in use %zu B\n", n, rep.pool_bytes, rep.pool_bytes_in_use);
        // Three live containers: bins.n (u32[1]), bins.ref (f32[n], sized by the hook) and bins.bins (f32[n]).
        // ... and the launch arrays of the one kernel component (bins).
        CHECK(rep.pool_bytes_in_use == cst::pool_bytes_of(sizeof(std::uint32_t)) + 2 * cst::pool_bytes_of(n * sizeof(float)) + cst::solo_launch_bytes(1));
        CHECK(rep.pool_bytes_in_use == 3 * cst::pool_node_bytes + cst::solo_launch_bytes(1));
        const auto bins = to_vector(cst::synced_output<DType::f32, 1>(model, "bins.bins"));
        REQUIRE(bins.size() == n);
        for (std::size_t i = 0; i < bins.size(); ++i) CHECK(bins[i] == static_cast<float>(i + 1));
    }

    n = 3000;  // 12000 B: several nodes
    model.update_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{n}));
    cst::forward_and_wait(model);
    {
        const auto rep = tmod::report_on(model, dev);
        std::printf("[pool] n=%u: pool %zu B, in use %zu B\n", n, rep.pool_bytes, rep.pool_bytes_in_use);
        // 12000 B is twelve nodes (12288 B); the old, smaller storage was freed when the containers grew.
        CHECK(rep.pool_bytes_in_use == cst::pool_bytes_of(sizeof(std::uint32_t)) + 2 * cst::pool_bytes_of(n * sizeof(float)) + cst::solo_launch_bytes(1));
        CHECK(rep.pool_bytes_in_use == cst::pool_node_bytes + 2 * 12 * cst::pool_node_bytes + cst::solo_launch_bytes(1));
        const auto bins = to_vector(cst::synced_output<DType::f32, 1>(model, "bins.bins"));
        REQUIRE(bins.size() == n);
        bool ok = true;
        for (std::size_t i = 0; i < bins.size(); ++i) ok = ok && bins[i] == static_cast<float>(i + 1);
        CHECK(ok);
    }
}

// =============================================================================================
// 7. Two devices
// =============================================================================================
TEST_CASE("Two devices: a producer on one, its consumer on the other", "[model][devices]") {
    sycl::device first, host;
    if (!tmod::pick_two(first, host)) SKIP("a non-OpenMP device and the OpenMP host device are not both visible");

    // There is no model-wide default device (design review, section 2): every instance takes AdaptiveCpp's default and a device
    // given to add_subsystem overrides a subtree. So the default-selector device must be one of the two, and `mid` goes to
    // the other one; the case skips otherwise.
    const sycl::device dflt = cs::detail::default_device();
    if (dflt != first && dflt != host) SKIP("the default device is neither of the two devices picked");
    const sycl::device other = (dflt == first) ? host : first;

    // a real pool on each of the two devices, so both report pool bytes in use
    auto model = cs::create_model<tmod::TwoDevTop>(
        cs::model_config_desc{cs::pool_desc{std::vector<sycl::device>{first, host}, std::size_t{1} << 20}}, other);
    model.setup();

    CHECK(model.root().instance_at("src.y")->device == dflt);
    CHECK(model.root().instance_at("mid.y")->device == other);

    for (int round = 0; round < 2; ++round) {
        std::vector<float> x(tmod::kLen), want_mid(tmod::kLen), want_out(tmod::kLen);
        for (std::size_t i = 0; i < tmod::kLen; ++i) {
            x[i]        = static_cast<float>(i + 1) * static_cast<float>(round + 1);
            want_mid[i] = 2.0f * x[i] + 1.0f;
            want_out[i] = want_mid[i] + 2.0f;
        }
        INFO("round " << round);
        model.update_input("src.x", cs::init::from_host(x));
        cst::forward_and_wait(model);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "src.y")) == x);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "mid.y")) == want_mid);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "out.y")) == want_out);
    }

    // Both devices have a pool (the pool_desc) and a device-kind container draws from its device's pool, so bytes are in
    // use on both after a forward.
    const auto rep = model.memory_report();
    CHECK(rep.devices.size() == 2);
    CHECK(rep.devices.at(first).pool_bytes_in_use > 0);
    CHECK(rep.devices.at(host).pool_bytes_in_use > 0);
}

// =============================================================================================
// 8. Initializer-only inputs and empty roots
// =============================================================================================
TEST_CASE("empty: a size-0 root is present and empty; its consumer runs as a no-op", "[model][init][empty]") {
    auto model = cs::create_model<tmod::EmptyTop>(default_config());
    model.setup();
    // Design review, 11.4: a view can be made at any time; it fails only when the shape cannot be resolved yet. bins.bins is
    // sized by its producer in pre_forward, so before the first forward its shape is not known.
    tmod::expect_throws([&] { (void)model.view_output<DType::f32, 1>("bins.bins"); }, {"bins.bins", "not known yet"});

    model.update_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{0}));
    cst::forward_and_wait(model);
    {
        // A zero-element variable is present with a host copy of size 0.
        auto bins = cst::synced_output<DType::f32, 1>(model, "bins.bins");  // present, size 0
        CHECK(bins.size() == 0);
        CHECK(bins.shape()[0] == 0);
        auto y = cst::synced_output<DType::f32, 1>(model, "sc.y");
        CHECK(y.size() == 0);
        CHECK(y.shape()[0] == 0);
    }

    // The empty root grows when its driving value does.
    model.update_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{4}));
    cst::forward_and_wait(model);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "bins.bins")) == std::vector<float>{1, 2, 3, 4});
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sc.y")) == std::vector<float>{2, 4, 6, 8});
}

TEST_CASE("init: update_input before the first forward is applied", "[model][init]") {
    auto model = cs::create_model<tmod::OneAxTop>(default_config());
    model.setup();
    std::vector<float> x(tmod::kLen);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i) + 0.5f;
    model.update_input("a.x", cs::init::from_host(x));
    cst::forward_and_wait(model);
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "a.x")) == x);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == x);
}

TEST_CASE("init: update_input between runs re-initializes once", "[model][init]") {
    auto model = cs::create_model<tmod::OneAxTop>(default_config());
    model.setup();
    const auto host_init = [](float v) {
        return cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostInit<DType::f32, 1>&)>{
            [v](cs::HostInit<DType::f32, 1>& h) {
                ++tmod::g_host_inits;
                for (std::size_t k = 0; k < h.size(); ++k) h[k] = v;
            }});
    };
    tmod::g_host_inits = 0;
    model.update_input("a.x", host_init(5.0f));
    cst::forward_and_wait(model);
    CHECK(tmod::g_host_inits == 1);
    cst::forward_and_wait(model);
    CHECK(tmod::g_host_inits == 1);  // not re-applied without a new call
    model.update_input("a.x", host_init(7.0f));
    cst::forward_and_wait(model);
    CHECK(tmod::g_host_inits == 2);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == std::vector<float>(tmod::kLen, 7.0f));
    cst::forward_and_wait(model);
    CHECK(tmod::g_host_inits == 2);
}

TEST_CASE("init: Model init beats the Component's and the System's declared init", "[model][init]") {
    {
        auto model = cs::create_model<tmod::DeclaredTop>(default_config());
        model.setup();
        cst::forward_and_wait(model);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) ==
              std::vector<float>(tmod::kLen, 2.0f));  // the System over the Component
    }
    {
        auto model = cs::create_model<tmod::DeclaredTop>(default_config());
        model.setup();
        model.update_input("a.x", cs::init::fill(3.0f));
        cst::forward_and_wait(model);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) ==
              std::vector<float>(tmod::kLen, 3.0f));  // the Model over both
    }
}

TEST_CASE("init: update_input on a connected input warns and is not applied; on an output it throws",
          "[model][init]") {
    auto model = cs::create_model<tmod::ChainTop>(default_config());
    model.setup();

    std::ostringstream captured;
    std::streambuf*    old = std::cerr.rdbuf(captured.rdbuf());
    try {
        model.update_input("q.x", cs::init::fill(9.0f));
    } catch (...) {
        std::cerr.rdbuf(old);
        throw;
    }
    std::cerr.rdbuf(old);
    // The warning's wording changed with the rework: it names update_input (model.hpp, per design review section 5).
    CHECK(captured.str() ==
          "naga::cospan: warning: input 'q.x' is connected to an output; update_input is never applied\n");

    cst::forward_and_wait(model);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "q.y")) ==
          std::vector<float>(tmod::kLen, 1.0f));  // p.y = 1, not 9

    CHECK_THROWS_AS(model.update_input("p.y", cs::init::fill(1.0f)), std::invalid_argument);
}


// =============================================================================================
// 9. Configuration by descriptors (model_config_desc, pool_desc, node_size_desc, async_handler_desc)
// =============================================================================================
namespace tmod {

/// One Axpb p (y = 1 * x + b, x zero): every element of p.y is b.
class ArgTop : public cs::System {
    float b_;

public:
    explicit ArgTop(float b) : b_{b} {}
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("p", 1.0f, b_)); }
};

/// The same, with p placed on `dev` (add_subsystem is given the device).
class PlacedTop : public cs::System {
    sycl::device dev_;
    float        b_;

public:
    PlacedTop(const sycl::device& dev, float b) : dev_{dev}, b_{b} {}
    void setup() override { add_subsystem(dev_, cs::subsystem_desc<Axpb>("p", 1.0f, b_)); }
};

/// A host component that throws; the exception reaches the queue asynchronously.
class Thrower : public cs::Component<Thrower> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(cs::Host, const Inputs&, const Outputs&) { throw std::runtime_error("tmod: asynchronous boom"); }
    Inputs  inputs;
    Outputs outputs;
    void setup() override {
        declare_input(inputs.x, "x", 4);
        declare_output(outputs.y, "y", 4);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(); }
};
class ThrowTop : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Thrower>("t")); }
};

inline int         g_async_errors = 0;   // what the async_handler_desc handler saw
inline std::string g_async_message;

/// Every root device SYCL reports (what the Model's context covers).
inline std::vector<sycl::device> all_root_devices() {
    const sk::context ctx{};
    return std::vector<sycl::device>(ctx.devices().begin(), ctx.devices().end());
}

/// forward, wait, and p.y read back: true when every element is `want`.
inline bool p_is(cs::Model& m, float want) {
    cst::forward_and_wait(m);
    return to_vector(cst::synced_output<DType::f32, 1>(m, "p.y")) == std::vector<float>(kLen, want);
}

}  // namespace tmod

TEST_CASE("create_model: no arguments, root arguments only, and a model_config_desc with root arguments", "[model][config]") {
    SECTION("no arguments") {
        auto model = cs::create_model<tmod::ChainTop>();
        model.setup();
        cst::forward_and_wait(model);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "q.y")) == std::vector<float>(tmod::kLen, 1.0f));
    }
    SECTION("root arguments only") {
        auto model = cs::create_model<tmod::ArgTop>(3.0f);
        model.setup();
        CHECK(tmod::p_is(model, 3.0f));
    }
    SECTION("a model_config_desc, then root arguments") {
        auto model = cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{cs::all_devices, std::size_t{1} << 20}}, 4.0f);
        model.setup();
        CHECK(tmod::p_is(model, 4.0f));
    }
    SECTION("a model_config_desc lvalue, then no root arguments") {
        const cs::model_config_desc cfg{};
        auto                        model = cs::create_model<tmod::ChainTop>(cfg);
        model.setup();
        cst::forward_and_wait(model);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "q.y")) == std::vector<float>(tmod::kLen, 1.0f));
    }
}

TEST_CASE("model_config_desc: descriptors in either order", "[model][config]") {
    const sycl::device dev{sycl::default_selector_v};
    const auto         handler = [](sycl::exception_list) {};
    SECTION("pool_desc, then async_handler_desc") {
        auto model = cs::create_model<tmod::ArgTop>(
            cs::model_config_desc{cs::pool_desc{dev, std::size_t{1} << 20}, cs::async_handler_desc{handler}}, 2.0f);
        model.setup();
        CHECK(tmod::p_is(model, 2.0f));
        CHECK(tmod::report_on(model, dev).pool_bytes == (std::size_t{1} << 20));
    }
    SECTION("async_handler_desc, then pool_desc") {
        auto model = cs::create_model<tmod::ArgTop>(
            cs::model_config_desc{cs::async_handler_desc{handler}, cs::pool_desc{dev, std::size_t{1} << 20}}, 2.0f);
        model.setup();
        CHECK(tmod::p_is(model, 2.0f));
        CHECK(tmod::report_on(model, dev).pool_bytes == (std::size_t{1} << 20));
    }
    SECTION("two pool_descs on different devices, with a handler between them") {
        const auto devs = tmod::all_root_devices();
        if (devs.size() < 2) SKIP("only one device is visible");
        auto model = cs::create_model<tmod::ArgTop>(
            cs::model_config_desc{cs::pool_desc{devs[0], std::size_t{1} << 20}, cs::async_handler_desc{handler},
                                  cs::pool_desc{devs[1], std::size_t{1} << 21}},
            2.0f);
        model.setup();
        CHECK(tmod::p_is(model, 2.0f));
        CHECK(tmod::report_on(model, devs[0]).pool_bytes == (std::size_t{1} << 20));
        CHECK(tmod::report_on(model, devs[1]).pool_bytes == (std::size_t{1} << 21));
    }
}

TEST_CASE("pool_desc: a device, a list of devices and cs::all_devices", "[model][config][pool]") {
    const auto         devs = tmod::all_root_devices();
    const sycl::device dev{sycl::default_selector_v};
    constexpr std::size_t kBytes = std::size_t{1} << 20;
    SECTION("one device: only that device has a pool") {
        auto model = cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{dev, kBytes}}, 1.0f);
        model.setup();
        const auto rep = model.memory_report();
        CHECK(rep.devices.size() == 1);
        CHECK(tmod::report_on(model, dev).pool_bytes == kBytes);
    }
    SECTION("a list of devices: each has a pool of the given size") {
        auto model = cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{devs, kBytes}}, 1.0f);
        model.setup();
        const auto rep = model.memory_report();
        CHECK(rep.devices.size() == devs.size());
        for (const sycl::device& d : devs) CHECK(tmod::report_on(model, d).pool_bytes == kBytes);
    }
    SECTION("cs::all_devices: every device has a pool") {
        auto model = cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{cs::all_devices, kBytes}}, 1.0f);
        model.setup();
        const auto rep = model.memory_report();
        CHECK(rep.devices.size() == devs.size());
        for (const sycl::device& d : devs) CHECK(tmod::report_on(model, d).pool_bytes == kBytes);
        CHECK(tmod::p_is(model, 1.0f));
    }
}

TEST_CASE("pool_desc: a per-pool node size, and the default node size of 1024", "[model][config][pool]") {
    const sycl::device dev{sycl::default_selector_v};
    std::size_t        in_use_default = 0, in_use_4096 = 0;
    {
        auto model = cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{dev, std::size_t{1} << 20}}, 1.0f);
        model.setup();
        REQUIRE(tmod::p_is(model, 1.0f));
        in_use_default = tmod::report_on(model, dev).pool_bytes_in_use;
        CHECK(tmod::report_on(model, dev).pool_bytes == (std::size_t{1} << 20));
    }
    {
        auto model = cs::create_model<tmod::ArgTop>(
            cs::model_config_desc{cs::pool_desc{dev, std::size_t{1} << 20, cs::node_size_desc{4096}}}, 1.0f);
        model.setup();
        REQUIRE(tmod::p_is(model, 1.0f));
        in_use_4096 = tmod::report_on(model, dev).pool_bytes_in_use;
        CHECK(tmod::report_on(model, dev).pool_bytes == (std::size_t{1} << 20));
    }
    // Every container takes whole nodes: multiples of 1024 by default, of 4096 with the larger node.
    CHECK(in_use_default > 0);
    CHECK(in_use_default % 1024 == 0);
    CHECK(in_use_4096 % 4096 == 0);
    CHECK(in_use_4096 >= in_use_default);
}

TEST_CASE("pool_desc: overlapping coverage is rejected, naming the device", "[model][config][pool]") {
    const sycl::device dev{sycl::default_selector_v};
    const std::string  name = dev.get_info<sycl::info::device::name>();
    constexpr std::size_t kBytes = 4096;
    SECTION("a device covered by two pool_descs") {
        tmod::expect_throws([&] { (void)cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{dev, kBytes}, cs::pool_desc{dev, kBytes}}, 1.0f); },
                            {name});
        CHECK_THROWS_AS(cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{dev, kBytes}, cs::pool_desc{dev, kBytes}}, 1.0f),
                        std::invalid_argument);
    }
    SECTION("a device twice in one list") {
        const std::vector<sycl::device> twice{dev, dev};
        tmod::expect_throws([&] { (void)cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{twice, kBytes}}, 1.0f); }, {name});
        CHECK_THROWS_AS(cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{twice, kBytes}}, 1.0f), std::invalid_argument);
    }
    SECTION("cs::all_devices combined with a device pool_desc, in either order") {
        CHECK_THROWS_AS(cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{cs::all_devices, kBytes}, cs::pool_desc{dev, kBytes}}, 1.0f),
                        std::invalid_argument);
        tmod::expect_throws([&] { (void)cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{dev, kBytes}, cs::pool_desc{cs::all_devices, kBytes}}, 1.0f); },
                            {name});
    }
    SECTION("cs::all_devices combined with another cs::all_devices") {
        CHECK_THROWS_AS(cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{cs::all_devices, kBytes}, cs::pool_desc{cs::all_devices, kBytes}}, 1.0f),
                        std::invalid_argument);
    }
}

TEST_CASE("an uncovered device is a pool of 0 bytes: its allocations take the OS route and are counted", "[model][config][memory]") {
    const sycl::device dev{sycl::default_selector_v};
    const auto         devs = tmod::all_root_devices();
    // Cover a device other than the default one when there is one; otherwise no pool_desc at all.
    cs::model_config_desc cfg{};
    for (const sycl::device& d : devs)
        if (d != dev) {
            cfg = cs::model_config_desc{cs::pool_desc{d, std::size_t{1} << 20}};
            break;
        }
    auto model = cs::create_model<tmod::ArgTop>(std::move(cfg), 5.0f);
    model.setup();
    CHECK(tmod::p_is(model, 5.0f));
    const auto rep = tmod::report_on(model, dev);
    CHECK(rep.pool_bytes == 0);
    CHECK(rep.pool_bytes_in_use == 0);
    CHECK(rep.os_allocations > 0);
    CHECK(rep.os_bytes_in_use > 0);
}

TEST_CASE("a subsystem placed on a device no pool_desc names runs correctly", "[model][config][devices]") {
    const sycl::device dflt{sycl::default_selector_v};
    const auto         devs = tmod::all_root_devices();
    if (devs.size() < 2) SKIP("only one device is visible");
    sycl::device other = devs[0];
    for (const sycl::device& d : devs)
        if (d != dflt) {
            other = d;
            break;
        }
    // The only pool is on the default device; p goes to the other one, which no pool_desc names.
    auto model = cs::create_model<tmod::PlacedTop>(cs::model_config_desc{cs::pool_desc{dflt, std::size_t{1} << 20}}, other, 6.0f);
    model.setup();
    CHECK(model.root().instance_at("p.y")->device == other);
    CHECK(tmod::p_is(model, 6.0f));
    CHECK(tmod::report_on(model, other).os_allocations > 0);
}

TEST_CASE("memory_report lists devices whose pool has bytes or that recorded an allocation, and no others", "[model][config][memory]") {
    const auto         devs = tmod::all_root_devices();
    const sycl::device dflt{sycl::default_selector_v};
    SECTION("no pool and nothing run: empty") {
        auto model = cs::create_model<tmod::ArgTop>(1.0f);
        model.setup();
        CHECK(model.memory_report().devices.empty());
    }
    SECTION("a pool on the default device only: that device, before and after a run") {
        auto model = cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::pool_desc{dflt, std::size_t{1} << 20}}, 1.0f);
        model.setup();
        CHECK(model.memory_report().devices.size() == 1);
        CHECK(tmod::p_is(model, 1.0f));
        const auto rep = model.memory_report();
        CHECK(rep.devices.size() == 1);
        for (const sycl::device& d : devs) CHECK(rep.devices.count(d) == (d == dflt ? 1u : 0u));
    }
    SECTION("an uncovered device appears once it has allocated") {
        if (devs.size() < 2) SKIP("only one device is visible");
        sycl::device other = devs[0];
        for (const sycl::device& d : devs)
            if (d != dflt) {
                other = d;
                break;
            }
        auto model = cs::create_model<tmod::PlacedTop>(other, 1.0f);
        model.setup();
        CHECK(model.memory_report().devices.empty());
        CHECK(tmod::p_is(model, 1.0f));
        const auto rep = model.memory_report();
        CHECK(rep.devices.count(other) == 1);
        CHECK(rep.devices.at(other).pool_bytes == 0);
        CHECK(rep.devices.at(other).os_allocations > 0);
    }
}

TEST_CASE("async_handler_desc: a model built with a handler runs and waits normally; a host task's exception still surfaces from wait()", "[model][config][async]") {
    // SlabKit keeps a host callable's exception itself (queue.cpp, state_throw_asynchronous: "the runtime has no way to be
    // told about it"), so it never reaches the sycl::async_handler; wait() rethrows it. The handler is exercised by the runtime's
    // own asynchronous errors, which no portable cospan model can provoke; here it is checked to be installed without disturbing
    // either path.
    tmod::g_async_errors = 0;
    auto handler = [](sycl::exception_list list) {
        for (const std::exception_ptr& e : list) {
            (void)e;
            ++tmod::g_async_errors;
        }
    };
    SECTION("a normal model") {
        auto model = cs::create_model<tmod::ArgTop>(cs::model_config_desc{cs::async_handler_desc{handler}}, 7.0f);
        model.setup();
        CHECK(tmod::p_is(model, 7.0f));
        CHECK(tmod::g_async_errors == 0);
    }
    SECTION("a host task that throws") {
        auto model = cs::create_model<tmod::ThrowTop>(cs::model_config_desc{cs::async_handler_desc{handler}});
        model.setup();
        model.run();
        CHECK_THROWS_AS(model.wait(), std::runtime_error);
    }
}
