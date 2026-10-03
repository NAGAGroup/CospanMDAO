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
//   3. Lifecycle        forward / view_output before setup, setup twice, a moved-from
//                       Model
//   4. Host access      unknown path, view_input on an output, update_input on an output
//   5. Memory report    a small static model computes; the pool report (there is no arena)
//   6. Pool             a pool on the device and a value-driven (shape hook) output root
//   7. Two devices      producer and consumer on different devices of one context
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
struct D1Body {
    struct Inputs {
        cs::Input<DT, 1> z;
        cs::Input<DT, 1> x;
        cs::Input<DT, 1> y2;
    };
    struct Outputs {
        cs::Output<DT, 1> y1;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        out.y1[i] = in.z[std::size_t{0}] * in.z[std::size_t{0}] + in.z[std::size_t{1}] + in.x[std::size_t{0}] -
                    static_cast<T_<DT>>(0.2) * in.y2[std::size_t{0}];
    }
};
template <DType DT>
class D1 : public cs::Component<D1Body<DT>> {
    cs::Input<DT, 1>  z_, x_, y2_;
    cs::Output<DT, 1> y1_;

public:
    void setup() override {
        this->declare_input(z_, "z", cs::unspecified_shape<1>());  // z and x are tie members: the tie alias owns their shape
        this->declare_input(x_, "x", cs::unspecified_shape<1>());
        this->declare_input(y2_, "y2", 1);
        this->declare_output(y1_, "y1", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(y1_.shape(), {z_, x_, y2_}, {y1_}, {});
    }
};

template <DType DT>
struct D2Body {
    struct Inputs {
        cs::Input<DT, 1> z;
        cs::Input<DT, 1> y1;
    };
    struct Outputs {
        cs::Output<DT, 1> y2;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        out.y2[i] = sycl::sqrt(in.y1[std::size_t{0}]) + in.z[std::size_t{0}] + in.z[std::size_t{1}];
    }
};
template <DType DT>
class D2 : public cs::Component<D2Body<DT>> {
    cs::Input<DT, 1>  z_, y1_;
    cs::Output<DT, 1> y2_;

public:
    void setup() override {
        this->declare_input(z_, "z", cs::unspecified_shape<1>());
        this->declare_input(y1_, "y1", cs::unspecified_shape<1>());
        this->declare_output(y2_, "y2", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(y2_.shape(), {z_, y1_}, {y2_}, {});
    }
};

template <DType DT>
struct ObjBody {
    struct Inputs {
        cs::Input<DT, 1> x;
        cs::Input<DT, 1> z;
        cs::Input<DT, 1> y1;
        cs::Input<DT, 1> y2;
    };
    struct Outputs {
        cs::Output<DT, 1> f;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        const T_<DT> x = in.x[std::size_t{0}];
        out.f[i]       = x * x + in.z[std::size_t{1}] + in.y1[std::size_t{0}] + sycl::exp(-in.y2[std::size_t{0}]);
    }
};
template <DType DT>
class Obj : public cs::Component<ObjBody<DT>> {
    cs::Input<DT, 1>  x_, z_, y1_, y2_;
    cs::Output<DT, 1> f_;

public:
    void setup() override {
        this->declare_input(x_, "x", cs::unspecified_shape<1>());
        this->declare_input(z_, "z", cs::unspecified_shape<1>());
        this->declare_input(y1_, "y1", cs::unspecified_shape<1>());
        this->declare_input(y2_, "y2", cs::unspecified_shape<1>());
        this->declare_output(f_, "f", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(f_.shape(), {x_, z_, y1_, y2_}, {f_}, {});
    }
};

template <DType DT>
struct G1Body {
    struct Inputs {
        cs::Input<DT, 1> y1;
    };
    struct Outputs {
        cs::Output<DT, 1> g;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        out.g[i] = static_cast<T_<DT>>(3.16) - in.y1[std::size_t{0}];
    }
};
template <DType DT>
class G1 : public cs::Component<G1Body<DT>> {
    cs::Input<DT, 1>  y1_;
    cs::Output<DT, 1> g_;

public:
    void setup() override {
        this->declare_input(y1_, "y1", cs::unspecified_shape<1>());
        this->declare_output(g_, "g", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(g_.shape(), {y1_}, {g_}, {});
    }
};

template <DType DT>
struct G2Body {
    struct Inputs {
        cs::Input<DT, 1> y2;
    };
    struct Outputs {
        cs::Output<DT, 1> g;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        out.g[i] = in.y2[std::size_t{0}] - static_cast<T_<DT>>(24.0);
    }
};
template <DType DT>
class G2 : public cs::Component<G2Body<DT>> {
    cs::Input<DT, 1>  y2_;
    cs::Output<DT, 1> g_;

public:
    void setup() override {
        this->declare_input(y2_, "y2", cs::unspecified_shape<1>());
        this->declare_output(g_, "g", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(g_.shape(), {y2_}, {g_}, {});
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

struct GenBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = static_cast<float>(i[0]); }
};
class Gen : public cs::Component<GenBody> {
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override { declare_output(y_, "y", 100); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {}, {y_}, {});
    }
};

struct CvtBody {
    struct Inputs {
        cs::Input<DType::f32, 1> a;
    };
    struct Outputs {
        cs::Output<DType::f64, 1> d;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.d[i] = 2.0 * static_cast<double>(in.a[i]); }
};
class Cvt : public cs::Component<CvtBody> {
    cs::Input<DType::f32, 1>  a_;
    cs::Output<DType::f64, 1> d_;

public:
    void setup() override {
        declare_input(a_, "a", cs::unspecified_shape<1>());
        declare_output(d_, "d", 64);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(d_.shape(), {a_}, {d_}, {});
    }
};

struct CombBody {
    struct Inputs {
        cs::Input<DType::f32, 1> b;
        cs::Input<DType::f32, 1> c;
        cs::Input<DType::f64, 1> e;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> o;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        const std::size_t k = i[0] % 8;
        out.o[i]            = in.b[k] + in.c[k] + static_cast<float>(in.e[std::size_t{0}]);
    }
};
class Comb : public cs::Component<CombBody> {
    cs::Input<DType::f32, 1>  b_, c_;
    cs::Input<DType::f64, 1>  e_;
    cs::Output<DType::f32, 1> o_;

public:
    void setup() override {
        declare_input(b_, "b", cs::unspecified_shape<1>());  // tied into the top alias "bc" (fixed 8)
        declare_input(c_, "c", cs::unspecified_shape<1>());
        declare_input(e_, "e", cs::unspecified_shape<1>());
        declare_output(o_, "o", 16);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(o_.shape(), {b_, c_, e_}, {o_}, {});
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
// REVIEW: a component cannot read a value and a hook cannot target an output (section 10), so Bins takes a free input `ref`
// that its System hooks with from_value(n) (hook_bins) and copies its shape in pre_forward.
struct BinsBody {
    struct Inputs {
        cs::Input<DType::u32, 1> n;
        cs::Input<DType::f32, 1> ref;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> bins;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.bins[i] = static_cast<float>(i[0]) + 1.0f; }
};
class Bins : public cs::Component<BinsBody> {
    cs::Input<DType::u32, 1>  n_;
    cs::Input<DType::f32, 1>  ref_;
    cs::Output<DType::f32, 1> bins_;

public:
    void setup() override {
        declare_input(n_, "n", 1);
        declare_input(ref_, "ref", cs::unspecified);
        declare_output(bins_, "bins", cs::unspecified);
    }
    void pre_forward() override { set_shape(bins_, shape_of(ref_)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(bins_.shape(), {n_, ref_}, {bins_}, {});
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

struct AxpbBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    float a, b;
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = a * in.x[i] + b; }
};
class Axpb : public cs::Component<AxpbBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
    float                     a_, b_;

public:
    Axpb(float a, float b) : a_{a}, b_{b} {}
    void setup() override {
        declare_input(x_, "x", static_cast<int>(kLen));
        declare_output(y_, "y", static_cast<int>(kLen));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {x_}, {y_}, {a_, b_});
    }
};

// ---- Initializer-only inputs and empty roots -------------------------------------------------
// Scale: y = 2 x, x shaped by its source, y by x.
struct ScaleBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = 2.0f * in.x[i]; }
};
class Scale : public cs::Component<ScaleBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(y_, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(y_, shape_of(x_)); }  // the output follows the input (section 10: no derived dims)
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {x_}, {y_}, {});
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

TEST_CASE("Marks: only on subsystems that never go out of scope", "[model][marks][errors]") {
    // Design review, section 9 (Jack, 2026-10-03): Model::setup's finalize walks the slots from the root to the mark's
    // declaring subsystem and refuses a run policy (transient, do_once, run_n) on the way.
    SECTION("an eternal subsystem: fine") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 0, 'd');
        CHECK_NOTHROW(model.setup());
        CHECK(model.design_vars().size() == 1);
    }
    SECTION("a transient subsystem") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 1, 'd');
        tmod::expect_throws([&] { model.setup(); },
                            {"'a.x' is a design variable on a subsystem that can go out of scope", "('a' has a run policy)",
                             "marks must live on subsystems that never go out of scope"});
    }
    SECTION("a do_once subsystem, a constraint") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 2, 'c');
        tmod::expect_throws([&] { model.setup(); },
                            {"'a.y' is a constraint on a subsystem that can go out of scope", "('a' has a run policy)"});
    }
    SECTION("a run_n subsystem") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 3, 'd');
        tmod::expect_throws([&] { model.setup(); }, {"design variable", "('a' has a run policy)"});
    }
    SECTION("an eternal System under a transient one: the walk starts at the root") {
        auto model = cs::create_model<tmod::MarkScopeTop>(default_config(), 4, 'd');
        tmod::expect_throws([&] { model.setup(); },
                            {"'outer.in.a.x' is a design variable", "('outer' has a run policy)"});
    }
}

// =============================================================================================
// 3. Lifecycle errors
// =============================================================================================
TEST_CASE("Lifecycle: errors before setup, setup twice, moved-from", "[model][lifecycle]") {
    using Top = tmod::SellarTop<DType::f32>;

    SECTION("forward before setup throws") {
        auto m = cs::create_model<Top>(default_config());
        tmod::expect_throws([&] { m.forward(); }, {"before setup"});
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
    SECTION("forward on a moved-from Model throws; the moved-to Model runs") {
        auto m = cs::create_model<Top>(default_config());
        m.setup();
        cs::Model moved = std::move(m);
        tmod::expect_throws([&] { m.forward(); }, {"moved-from"});  // NOLINT(bugprone-use-after-move)
        CHECK_NOTHROW(moved.forward());
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
        model.forward();

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

    // REVIEW: the old equality (arena capacity less the setup slack == the sum of the distinct static roots) has no
    // counterpart. The pool counts nodes (1024 B), so whether in-use bytes equal, or only bound, the sum of the roots
    // depends on node rounding and on which allocations the pool serves; the headers do not settle it. Only "in use > 0"
    // and "in use <= capacity" are asserted.
    CHECK(after_fwd.pool_bytes_in_use > 0);
    CHECK(after_fwd.pool_bytes >= after_fwd.pool_bytes_in_use);
}

TEST_CASE("Memory report: a device without a configured pool is a zero-byte pool, and its allocations count as OS ones",
          "[model][memory]") {
    // Design review, section 9: every context device gets a pool, possibly of 0 bytes (default_config() sets no size).
    const sycl::device dev{sycl::default_selector_v};
    auto               model = cs::create_model<tmod::MemTop>(default_config());
    model.setup();
    const auto before = tmod::report_on(model, dev);  // an entry exists before anything ran
    CHECK(before.pool_bytes == 0);
    CHECK(before.os_allocations == 0);
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
    cs::ModelConfig    cfg = default_config();
    cfg.default_pool_bytes = std::size_t{1} << 20;  // 1 MiB, in nodes of the default 1024 B
    auto model             = cs::create_model<tmod::PoolTop>(std::move(cfg));
    model.setup();
    // REVIEW: asserted after setup in the old case (arena era). A pool is sized up front (section 2), so its capacity
    // should already be reported; not settled from the headers whether slot_count() is nonzero before the first allocation.
    CHECK(tmod::report_on(model, dev).pool_bytes > 0);

    std::uint32_t n = 8;
    model.update_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{n}));
    cst::forward_and_wait(model);
    {
        const auto rep = tmod::report_on(model, dev);
        std::printf("[pool] n=%u: pool %zu B, in use %zu B\n", n, rep.pool_bytes, rep.pool_bytes_in_use);
        CHECK(rep.pool_bytes_in_use > 0);
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
        CHECK(rep.pool_bytes_in_use > 0);
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

    // REVIEW: ModelConfig has no default_device (design review, section 2): every instance takes AdaptiveCpp's default
    // and a device given to add_subsystem overrides a subtree. The old case made `first` the default and put `mid` on
    // `host`. Here the default-selector device must be one of the two, and `mid` goes to the other one; the case skips
    // otherwise. Whether that is the intended replacement is not settled.
    const sycl::device dflt = cs::detail::default_device();
    if (dflt != first && dflt != host) SKIP("the default device is neither of the two devices picked");
    const sycl::device other = (dflt == first) ? host : first;

    const std::array<sycl::device, 2> devs{first, host};
    cs::ModelConfig                   cfg{std::span<const sycl::device>{devs}};
    cfg.default_pool_bytes = std::size_t{1} << 20;  // a real pool on each device, so both report pool bytes in use
    auto model             = cs::create_model<tmod::TwoDevTop>(std::move(cfg), other);
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

    // REVIEW: replaces "arena_bytes > 0 on both devices". Assumes a device-kind allocation on each device is served by
    // that device's pool (so in use > 0 on both after a forward); the headers do not settle which allocations the pool serves.
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
    // Design review, section 5: a host view can be obtained only after a run. Before the first forward a view throws
    // (it used to throw "has no data: it is not materialized" before the owner had run).
    tmod::expect_throws([&] { (void)model.view_output<DType::f32, 1>("bins.bins"); }, {"only after a run"});

    model.update_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{0}));
    cst::forward_and_wait(model);
    {
        // REVIEW: "present, size 0" is the old expectation. A host view needs the host copy to have data
        // (host_view.hpp's host_store throws when it has none); whether a zero-element variable has a host copy is
        // not settled from the headers.
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
