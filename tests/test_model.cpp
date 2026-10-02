// Model tests, every case through a cospan::Model (create_model / setup / forward /
// wait / view_* / set_*); the only reach past those is the Model's own retrieval
// (root().entry_at).
//
//   1. Sellar forward   the same Systems templated on the dtype, run in f32 and f64
//                       over several design points against a host reference in double
//   2. Marks            design variables, objective, constraints: counts, bounds, and
//                       the tied z appearing once (one record per root)
//   3. Lifecycle        forward / view_output before setup, setup twice, a moved-from
//                       Model, a default_device outside the context (skipped when the
//                       runtime enumerates one device)
//   4. Host access      unknown path, view_input on an output, init_input on an output
//   5. Memory report    a small static model: arena capacity (less the setup slack)
//                       against the sum of flat_size x dtype_size over distinct roots
//   6. Pool             a pool on the device and a value-driven (FromInput) output root
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
        add_subsystem<D1<DT>>("d1");
        add_subsystem<D2<DT>>("d2");
        add_subsystem<Obj<DT>>("obj");
        add_subsystem<G1<DT>>("g1");
        add_subsystem<G2<DT>>("g2");

        tie_inputs<DT, 1>({"d1.z", "d2.z", "obj.z"}, "z", cs::alias::fixed(2));
        tie_inputs<DT, 1>({"d1.x", "obj.x"}, "x", cs::alias::fixed(1));
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
    auto v = m.view_output<DT, 1>(path);
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
        model.init_input("z", cs::init::from_host(std::vector<T>{z.begin(), z.end()}));
        model.init_input("x", cs::init::from_host(std::vector<T>{x}));
        model.init_input("d1.y2", cs::init::from_host(std::vector<T>{y2}));
        model.forward();
        model.wait();
        CHECK(model.view_input<DT, 1>("d2.z")[std::size_t{1}] == z[1]);  // the tie reads back everywhere
        CHECK(model.view_input<DT, 1>("obj.x")[std::size_t{0}] == x);

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
constexpr std::size_t kArenaSlack     = 4096 * 64;

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
        add_subsystem<Gen>("gen");
        add_subsystem<Cvt>("cvt");
        add_subsystem<Comb>("comb");
        tie_inputs<DType::f32, 1>({"comb.b", "comb.c"}, "bc", cs::alias::fixed(8));
        connect("gen.y", "cvt.a");
        connect("cvt.d", "comb.e");
        promote<DType::f32, 1>("comb.o", "out");
    }
};

// ---- Value-driven size on a pool (case 6) -----------------------------------------------------------
// Bins: a free u32 input n and an output of n floats (bins[i] = i + 1).
struct BinsBody {
    struct Inputs {
        cs::Input<DType::u32, 1> n;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> bins;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.bins[i] = static_cast<float>(i[0]) + 1.0f; }
};
class Bins : public cs::Component<BinsBody> {
    cs::Input<DType::u32, 1>  n_;
    cs::Output<DType::f32, 1> bins_;

public:
    void setup() override {
        declare_input(n_, "n", 1);
        declare_output(bins_, "bins", cs::FromInput{"n", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(bins_.shape(), {n_}, {bins_}, {});
    }
};
class PoolTop : public cs::System {
public:
    void setup() override { add_subsystem<Bins>("bins"); }
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
        declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {x_}, {y_}, {});
    }
};
// bins.n drives bins.bins (n floats); sc doubles them. With n = 0 the root is present and empty.
class EmptyTop : public cs::System {
public:
    void setup() override {
        add_subsystem<Bins>("bins");
        add_subsystem<Scale>("sc");
        connect("bins.bins", "sc.x");
    }
};

// One Axpb (y = x) with a free input a.x.
class OneAxTop : public cs::System {
public:
    void setup() override { add_subsystem<Axpb>("a", 1.0f, 0.0f); }
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
        add_subsystem<AxpbInit1>("a", 1.0f, 0.0f);
        init_input("a.x", cs::init::fill(2.0f));
    }
};

// p (y = x + 1) feeds q (y = x): q.x is a connected input.
class ChainTop : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("p", 1.0f, 1.0f);
        add_subsystem<Axpb>("q", 1.0f, 0.0f);
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
        add_subsystem<Axpb>("src", 1.0f, 0.0f);
        add_subsystem<Axpb>("mid", cs::property_list{cs::property::device{other_}}, 2.0f, 1.0f);
        add_subsystem<Axpb>("out", 1.0f, 2.0f);
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

// ---- A partial writer (case 4): Hold writes only y[0]; the host owns y[1..3] ----------------------------
struct HoldBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = in.x[i]; }
};
class Hold : public cs::Component<HoldBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(x_, "x", 1);
        declare_output(y_, "y", 4);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {x_}, {y_}, {});  // launches over x: one item
    }
};

struct SumBody {
    struct Inputs {
        cs::Input<DType::f32, 1> y;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> s;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        float t = 0.0f;
        for (std::size_t k = 0; k < in.y.shape()[0]; ++k) t += in.y[k];
        out.s[i] = t;
    }
};
class SumY : public cs::Component<SumBody> {
    cs::Input<DType::f32, 1>  y_;
    cs::Output<DType::f32, 1> s_;

public:
    void setup() override {
        declare_input(y_, "y", cs::unspecified_shape<1>());
        declare_output(s_, "s", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(s_.shape(), {y_}, {s_}, {});
    }
};
class HoldTop : public cs::System {
public:
    void setup() override {
        add_subsystem<Hold>("p");
        add_subsystem<SumY>("c");
        connect("p.y", "c.y");
    }
};

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

    const auto* zr = sys.entry_at("z").root();
    const auto* xr = sys.entry_at("x").root();
    // The three components' z entries share the one root, so one record serves all three.
    CHECK(sys.entry_at("d1.z").root() == zr);
    CHECK(sys.entry_at("d2.z").root() == zr);
    CHECK(sys.entry_at("obj.z").root() == zr);
    CHECK(sys.entry_at("d1.x").root() == xr);
    CHECK(sys.entry_at("obj.x").root() == xr);

    std::size_t on_z = 0, on_x = 0;
    for (const auto& d : model.design_vars()) {
        if (d.root == zr) ++on_z;
        if (d.root == xr) ++on_x;
    }
    CHECK(on_z == 1);
    CHECK(on_x == 1);

    for (const auto& d : model.design_vars()) {
        REQUIRE(d.bounds.lower.has_value());
        REQUIRE(d.bounds.upper.has_value());
        if (d.root == zr) {
            CHECK(*d.bounds.lower == std::vector<double>{-10.0, 0.0});
            CHECK(*d.bounds.upper == std::vector<double>{10.0, 10.0});
        } else {
            CHECK(*d.bounds.lower == std::vector<double>{0.0});
            CHECK(*d.bounds.upper == std::vector<double>{10.0});
        }
        CHECK_FALSE(d.equals.has_value());
    }

    CHECK(model.objectives()[0].root == sys.entry_at("obj.f").root());

    const auto* g1 = sys.entry_at("g1.g").root();
    const auto* g2 = sys.entry_at("g2.g").root();
    std::size_t on_g1 = 0, on_g2 = 0;
    for (const auto& c : model.constraints()) {
        if (c.root == g1) ++on_g1;
        if (c.root == g2) ++on_g2;
        REQUIRE(c.bounds.upper.has_value());
        CHECK(*c.bounds.upper == std::vector<double>{0.0});
        CHECK_FALSE(c.bounds.lower.has_value());
    }
    CHECK(on_g1 == 1);
    CHECK(on_g2 == 1);
    model.wait();  // setup() submitted initializer work; wait before the Model goes out of scope
}

// =============================================================================================
// 3. Lifecycle errors
// =============================================================================================
TEST_CASE("Lifecycle: errors before setup, setup twice, moved-from, default_device", "[model][lifecycle]") {
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
        m.wait();  // the first setup() submitted initializer work
    }
    SECTION("forward on a moved-from Model throws; the moved-to Model runs") {
        auto m = cs::create_model<Top>(default_config());
        m.setup();
        cs::Model moved = std::move(m);
        tmod::expect_throws([&] { m.forward(); }, {"moved-from"});  // NOLINT(bugprone-use-after-move)
        CHECK_NOTHROW(moved.forward());
        CHECK_NOTHROW(moved.wait());
    }
    SECTION("a default_device outside the context throws at construction") {
        const auto all = sycl::device::get_devices();
        if (all.size() < 2) SKIP("only one device is visible (" << all.size() << "); nothing to name outside the context");
        cs::ModelConfig cfg{std::span<const sycl::device>{all.data(), 1}};
        cfg.default_device = all[1];
        tmod::expect_throws([&] { auto bad = cs::create_model<Top>(std::move(cfg)); }, {"default_device"});
    }
}

// =============================================================================================
// 4. Host access errors
// =============================================================================================
TEST_CASE("Host access: errors name the path; init_input on an output throws", "[model][host]") {
    SECTION("errors") {
        auto model = cs::create_model<tmod::SellarTop<DType::f32>>(default_config());
        model.setup();
        const std::vector<float> two{1.0f, 2.0f};

        // Unknown path: the message carries the path it could not resolve.
        tmod::expect_throws([&] { (void)model.view_output<DType::f32, 1>("nope.y"); }, {"nope.y"});
        tmod::expect_throws([&] { (void)model.view_output<DType::f32, 1>("obj.nope"); }, {"obj.nope"});
        // The wrong kind of entry.
        tmod::expect_throws([&] { (void)model.view_input<DType::f32, 1>("obj.f"); }, {"obj.f", "an output"});
        tmod::expect_throws([&] { (void)model.view_output<DType::f32, 1>("d1.z"); }, {"d1.z", "an input"});
        // An initial value on an output is an error.
        tmod::expect_throws([&] { model.init_input("obj.f", cs::init::from_host(two)); }, {"obj.f", "an output"});
        // The right call still works after that.
        CHECK_NOTHROW(model.init_input("z", cs::init::from_host(two)));
        model.wait();  // setup() submitted work
    }
}

// =============================================================================================
// 5. Memory report, measured
// =============================================================================================
TEST_CASE("Memory report: arena capacity against the distinct static roots", "[model][memory]") {
    const sycl::device dev{sycl::default_selector_v};
    auto               model = cs::create_model<tmod::MemTop>(default_config());
    model.setup();

    const auto after_setup = tmod::report_on(model, dev);

    std::array<float, 8> bc{};
    for (std::size_t i = 0; i < bc.size(); ++i) bc[i] = static_cast<float>(i + 1);
    model.init_input("bc", cs::init::from_host(std::vector<float>{bc.begin(), bc.end()}));
    model.forward();
    model.wait();
    const auto after_fwd = tmod::report_on(model, dev);

    // The model computes what it says (so the roots really hold data).
    const auto gen = to_vector(model.view_output<DType::f32, 1>("gen.y"));
    const auto cvt = to_vector(model.view_output<DType::f64, 1>("cvt.d"));
    const auto out = to_vector(model.view_output<DType::f32, 1>("out"));
    REQUIRE(gen.size() == 100);
    REQUIRE(cvt.size() == 64);
    REQUIRE(out.size() == 16);
    CHECK(gen[99] == 99.0f);
    CHECK(cvt[63] == 126.0);
    for (std::size_t i = 0; i < out.size(); ++i) CHECK(out[i] == 2.0f * bc[i % 8]);  // e[0] = cvt.d[0] = 0

    const std::size_t cap      = after_fwd.arena_bytes;
    const std::size_t measured = cap >= tmod::kArenaSlack ? cap - tmod::kArenaSlack : 0;
    std::printf("[memory] expected static %zu B (gen.y %zu + cvt.d %zu + bc %zu + comb.o %zu)\n"
                "[memory] arena capacity %zu B; minus slack %zu B = %zu B (difference to expected: %+lld B)\n"
                "[memory] arena in use: %zu B after setup, %zu B after a forward; pool %zu B\n",
                tmod::kExpectedStatic, tmod::kGenBytes, tmod::kCvtBytes, tmod::kBcBytes, tmod::kOutBytes, cap,
                tmod::kArenaSlack, measured, static_cast<long long>(measured) - static_cast<long long>(tmod::kExpectedStatic),
                after_setup.arena_bytes_in_use, after_fwd.arena_bytes_in_use, after_fwd.pool_bytes);

    CHECK(cap >= tmod::kArenaSlack);
    CHECK(measured == tmod::kExpectedStatic);
    CHECK(after_fwd.arena_bytes_in_use <= cap);
    CHECK(after_fwd.arena_bytes_in_use >= tmod::kExpectedStatic);
}

// =============================================================================================
// 6. Pool
// =============================================================================================
TEST_CASE("Pool: a value-driven root allocates from the pool and follows init_input", "[model][pool]") {
    const sycl::device dev{sycl::default_selector_v};
    cs::ModelConfig    cfg = default_config();
    cfg.default_pool_bytes = std::size_t{1} << 20;  // 1 MiB, in nodes of the default 1024 B
    auto model             = cs::create_model<tmod::PoolTop>(std::move(cfg));
    model.setup();
    CHECK(tmod::report_on(model, dev).pool_bytes > 0);

    std::uint32_t n = 8;
    model.init_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{n}));
    model.forward();
    model.wait();
    {
        const auto rep = tmod::report_on(model, dev);
        std::printf("[pool] n=%u: pool %zu B, in use %zu B\n", n, rep.pool_bytes, rep.pool_bytes_in_use);
        CHECK(rep.pool_bytes_in_use > 0);
        const auto bins = to_vector(model.view_output<DType::f32, 1>("bins.bins"));
        REQUIRE(bins.size() == n);
        for (std::size_t i = 0; i < bins.size(); ++i) CHECK(bins[i] == static_cast<float>(i + 1));
    }

    n = 3000;  // 12000 B: several nodes
    model.init_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{n}));
    model.forward();
    model.wait();
    {
        const auto rep = tmod::report_on(model, dev);
        std::printf("[pool] n=%u: pool %zu B, in use %zu B\n", n, rep.pool_bytes, rep.pool_bytes_in_use);
        CHECK(rep.pool_bytes_in_use > 0);
        const auto bins = to_vector(model.view_output<DType::f32, 1>("bins.bins"));
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

    const std::array<sycl::device, 2> devs{first, host};
    cs::ModelConfig                   cfg{std::span<const sycl::device>{devs}};
    cfg.default_device = first;
    auto model         = cs::create_model<tmod::TwoDevTop>(std::move(cfg), host);
    model.setup();

    CHECK(model.root().entry_at("src.y").placement().device == first);
    CHECK(model.root().entry_at("mid.y").placement().device == host);
    const auto rep = model.memory_report();
    CHECK(rep.devices.size() == 2);
    CHECK(rep.devices.at(first).arena_bytes > 0);
    CHECK(rep.devices.at(host).arena_bytes > 0);

    for (int round = 0; round < 2; ++round) {
        std::vector<float> x(tmod::kLen), want_mid(tmod::kLen), want_out(tmod::kLen);
        for (std::size_t i = 0; i < tmod::kLen; ++i) {
            x[i]        = static_cast<float>(i + 1) * static_cast<float>(round + 1);
            want_mid[i] = 2.0f * x[i] + 1.0f;
            want_out[i] = want_mid[i] + 2.0f;
        }
        INFO("round " << round);
        model.init_input("src.x", cs::init::from_host(x));
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("src.y")) == x);
        CHECK(to_vector(model.view_output<DType::f32, 1>("mid.y")) == want_mid);
        CHECK(to_vector(model.view_output<DType::f32, 1>("out.y")) == want_out);
    }
}

// =============================================================================================
// 8. Initializer-only inputs and empty roots
// =============================================================================================
TEST_CASE("D139: a size-0 root is present and empty; its consumer runs as a no-op", "[model][init][empty]") {
    auto model = cs::create_model<tmod::EmptyTop>(default_config());
    model.setup();
    // Before the owner has run, a view throws.
    tmod::expect_throws([&] { (void)model.view_output<DType::f32, 1>("bins.bins"); }, {"no data until its owner"});

    model.init_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{0}));
    model.forward();
    model.wait();
    {
        auto bins = model.view_output<DType::f32, 1>("bins.bins");  // present, size 0
        CHECK(bins.size() == 0);
        CHECK(bins.shape()[0] == 0);
        auto y = model.view_output<DType::f32, 1>("sc.y");
        CHECK(y.size() == 0);
        CHECK(y.shape()[0] == 0);
    }

    // The empty root grows when its driving value does.
    model.init_input("bins.n", cs::init::from_host(std::vector<std::uint32_t>{4}));
    model.forward();
    model.wait();
    CHECK(to_vector(model.view_output<DType::f32, 1>("bins.bins")) == std::vector<float>{1, 2, 3, 4});
    CHECK(to_vector(model.view_output<DType::f32, 1>("sc.y")) == std::vector<float>{2, 4, 6, 8});
}

TEST_CASE("D139: init_input before the first forward is applied", "[model][init]") {
    auto model = cs::create_model<tmod::OneAxTop>(default_config());
    model.setup();
    std::vector<float> x(tmod::kLen);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i) + 0.5f;
    model.init_input("a.x", cs::init::from_host(x));
    model.forward();
    model.wait();
    CHECK(to_vector(model.view_input<DType::f32, 1>("a.x")) == x);
    CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == x);
}

TEST_CASE("D139: init_input between runs re-initializes once", "[model][init]") {
    auto model = cs::create_model<tmod::OneAxTop>(default_config());
    model.setup();
    const auto host_init = [](float v) {
        return cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostViewMut<DType::f32, 1>&)>{
            [v](cs::HostViewMut<DType::f32, 1>& h) {
                ++tmod::g_host_inits;
                for (std::size_t k = 0; k < h.size(); ++k) h[k] = v;
            }});
    };
    tmod::g_host_inits = 0;
    model.init_input("a.x", host_init(5.0f));
    model.forward();
    model.wait();
    CHECK(tmod::g_host_inits == 1);
    model.forward();
    model.wait();
    CHECK(tmod::g_host_inits == 1);  // not re-applied without a new call
    model.init_input("a.x", host_init(7.0f));
    model.forward();
    model.wait();
    CHECK(tmod::g_host_inits == 2);
    CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == std::vector<float>(tmod::kLen, 7.0f));
    model.forward();
    model.wait();
    CHECK(tmod::g_host_inits == 2);
}

TEST_CASE("D139: Model init beats the Component's and the System's declared init", "[model][init]") {
    {
        auto model = cs::create_model<tmod::DeclaredTop>(default_config());
        model.setup();
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) ==
              std::vector<float>(tmod::kLen, 2.0f));  // the System over the Component
    }
    {
        auto model = cs::create_model<tmod::DeclaredTop>(default_config());
        model.setup();
        model.init_input("a.x", cs::init::fill(3.0f));
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) ==
              std::vector<float>(tmod::kLen, 3.0f));  // the Model over both
    }
}

TEST_CASE("D139: init_input on a connected input warns and is not applied; on an output it throws",
          "[model][init]") {
    auto model = cs::create_model<tmod::ChainTop>(default_config());
    model.setup();

    std::ostringstream captured;
    std::streambuf*    old = std::cerr.rdbuf(captured.rdbuf());
    try {
        model.init_input("q.x", cs::init::fill(9.0f));
    } catch (...) {
        std::cerr.rdbuf(old);
        throw;
    }
    std::cerr.rdbuf(old);
    CHECK(captured.str() ==
          "naga::cospan: warning: input 'q.x' is connected to an output; your initializer may not run\n");

    model.forward();
    model.wait();
    CHECK(to_vector(model.view_output<DType::f32, 1>("q.y")) ==
          std::vector<float>(tmod::kLen, 1.0f));  // p.y = 1, not 9

    CHECK_THROWS_AS(model.init_input("p.y", cs::init::fill(1.0f)), std::invalid_argument);
}