// System tests, every case through a cospan::Model.
//
// Each case builds a small root System, makes a Model (create_model / setup),
// and checks through the Model (forward / wait / view_* / set_* and the
// marks). There is no driver and no entry access here; the only thing reached
// past the Model's verbs is the Model's own retrieval (root().entry_at,
// get_subsystem) and, for the structural-call check, root() itself.
//
//   connect       identity; Slice of a rank-1 output vector; a 2-D subarray of
//                 a rank-2 output; a Slice of a promoted alias that carries a
//                 Slice (offsets add)
//   promote       child -> mid -> top chain; the promoted path and the full
//                 path name the same root
//   ties          tie_inputs (one root; a tie of a tie across two levels);
//                 tie_outputs (atomic writers accumulate; plain throws unless
//                 allow_plain)
//   marks         design vars / constraints / objective recorded with their
//                 paths and bounds; the last mark of a kind replaces; the
//                 ill-formed ones throw
//   init order    Component pass, System pass, children before owners; a
//                 System's init has the last word over a Component's
//   transient     release after forward; a kept output; the revive re-runs
//                 only the subtree's inits; back-to-back forwards, one wait
//   structural    fan-in, tie after connect, structural call outside setup,
//                 connect from an input, unknown paths
//   placement     property::device equal to the parent's is a no-op
//   Sellar        the smoke_system Sellar-shaped forward, with values
//
// The components live in namespace tsys (not tc): tests/test_component.cpp has
// its own, and two same-named classes in one executable would be an ODR clash.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
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
using cst::all_close;
using cst::close;
using cst::default_config;
using cst::to_vector;
using cst::wait_and_view;

namespace tsys {

/// The message of the exception `f` throws; the case fails if `f` does not throw
/// or its message lacks `needle`. A function, not a macro, so template commas are fine.
template <class F>
void expect_throws(F&& f, const std::string& needle) {
    std::string what = "<no exception>";
    try {
        f();
    } catch (const std::exception& e) {
        what = e.what();
    }
    INFO("thrown: " << what);
    CHECK_THAT(what, Catch::Matchers::ContainsSubstring(needle));
}

/// A Model of `Sys` whose setup() must throw with `needle` in the message.
template <class Sys>
void expect_setup_throws(const std::string& needle) {
    expect_throws(
        [] {
            cs::Model model = cs::create_model<Sys>(default_config());
            model.setup();
        },
        needle);
}

constexpr std::size_t kLen = 8;

// Every Component and System that logs appends its path in init().
std::vector<std::string> g_inits;

std::size_t count_of(const std::string& path) {
    std::size_t n = 0;
    for (const auto& p : g_inits) n += (p == path);
    return n;
}
std::size_t index_of(const std::string& path) {  // first occurrence, or npos
    for (std::size_t i = 0; i < g_inits.size(); ++i)
        if (g_inits[i] == path) return i;
    return std::string::npos;
}

std::vector<float> iota_plus(float first, std::size_t n, float scale = 1.0f, float offset = 0.0f) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = scale * (first + static_cast<float>(i)) + offset;
    return v;
}

// ---- Src: y[i] = i + 1 (logs its init) ------------------------------------------------------
struct SrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = static_cast<float>(i[0] + 1); }
};

class Src : public cs::Component<SrcBody> {
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override { declare_output(y_, "y", static_cast<int>(kLen)); }
    void init(sk::queue&, const sycl::device&) override { g_inits.push_back(path()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {}, {y_}, {});
    }
};

// ---- Axpb: y = a * x + b, x of kLen (logs its init; x0 is a free-input init value) ------------
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
    std::optional<float>      x0_;

public:
    Axpb(float a, float b, std::optional<float> x0 = std::nullopt) : a_{a}, b_{b}, x0_{x0} {}
    void setup() override {
        declare_input(x_, "x", static_cast<int>(kLen));
        declare_output(y_, "y", static_cast<int>(kLen));
    }
    void init(sk::queue&, const sycl::device&) override {
        g_inits.push_back(path());
        if (x0_) fill_input("x", *x0_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {x_}, {y_}, {a_, b_});
    }
};

// ---- Copy1 / Copy2: y = x, the input takes the shape of what it is connected to --------------
struct Copy1Body {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = in.x[i]; }
};

class Copy1 : public cs::Component<Copy1Body> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(x_, "x", cs::from_source_shape<1>());
        declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {x_}, {y_}, {});
    }
};

struct Copy2Body {
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    void operator()(sycl::id<2> ij, Inputs in, Outputs out) const { out.y[ij] = in.x[ij]; }
};

class Copy2 : public cs::Component<Copy2Body> {
    cs::Input<DType::f32, 2>  x_;
    cs::Output<DType::f32, 2> y_;

public:
    void setup() override {
        declare_input(x_, "x", cs::from_source_shape<2>());
        declare_output(y_, "y", cs::from_input_shape<2>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {x_}, {y_}, {});
    }
};

// ---- Iota2D: out[i][j] = 1000 i + j ----------------------------------------------------------
struct Iota2DBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> out;
    };
    void operator()(sycl::id<2> ij, Inputs, Outputs o) const { o.out[ij] = static_cast<float>(ij[0] * 1000 + ij[1]); }
};

class Iota2D : public cs::Component<Iota2DBody> {
    cs::Output<DType::f32, 2> out_;
    int                       rows_, cols_;

public:
    Iota2D(int rows, int cols) : rows_{rows}, cols_{cols} {}
    void setup() override { declare_output(out_, "out", rows_, cols_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {}, {out_}, {});
    }
};

// ---- Adder: acc[i] += v, an ATOMIC output (zeroed in init) ------------------------------------
struct AdderBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> acc;
    };
    float v;
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.acc[i].fetch_add(v); }
};

class Adder : public cs::Component<AdderBody> {
    cs::Output<DType::atomic_f32, 1> acc_;
    float                            v_;

public:
    explicit Adder(float v) : v_{v} {}
    void setup() override { declare_output(acc_, "acc", static_cast<int>(kLen)); }
    void init(sk::queue&, const sycl::device&) override { zero_output("acc"); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(acc_.shape(), {}, {acc_}, {v_});
    }
};

// ---- Pick: f[0] = 2 * x[0], a scalar output (one value, so it may be an objective) ------------
struct PickBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> f;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.f[i] = 2.0f * in.x[std::size_t{0}]; }
};

class Pick : public cs::Component<PickBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> f_;

public:
    void setup() override {
        declare_input(x_, "x", static_cast<int>(kLen));
        declare_output(f_, "f", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{1}, {x_}, {f_}, {});
    }
};

/// A System whose init logs its path.
class LogSystem : public cs::System {
public:
    void init(sk::queue&, const sycl::device&) override { g_inits.push_back(path()); }
};

// ---- connect --------------------------------------------------------------------------------
class ConnectSys : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("prod");
        add_subsystem<Axpb>("cons", 2.0f, 1.0f);
        connect("prod.y", "cons.x");
    }
};

class Slice1Sys : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("prod");
        add_subsystem<Copy1>("c");
        connect("prod.y", "c.x", cs::Slice{cs::subarray<1>({2}, {3})});  // elements 2, 3, 4
    }
};

class Slice2Sys : public cs::System {
public:
    void setup() override {
        add_subsystem<Iota2D>("src", 4, 5);
        add_subsystem<Copy2>("c");
        connect("src.out", "c.x", cs::Slice{cs::subarray<2>({1, 2}, {2, 3})});  // rows 1..2, cols 2..4
    }
};

/// The inner System's `cx` is a promoted alias of an input that carries {2, 4}.
class ComposeInner : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("p");
        add_subsystem<Copy1>("c");
        connect("p.y", "c.x", cs::Slice{cs::subarray<1>({2}, {4})});  // elements 2..5
        promote("c.x", "cx");
    }
};

class ComposeTop : public cs::System {
public:
    void setup() override {
        add_subsystem<ComposeInner>("in");
        add_subsystem<Copy1>("c2");
        connect("in.cx", "c2.x", cs::Slice{cs::subarray<1>({1}, {2})});  // offsets add: 2 + 1 = 3, elements 3..4
    }
};

// ---- promote --------------------------------------------------------------------------------
class PromoteMid : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("child");
        promote("child.y", "y");
    }
};

class PromoteTop : public cs::System {
public:
    void setup() override {
        add_subsystem<PromoteMid>("mid");
        add_subsystem<Axpb>("c", 3.0f, 0.0f);
        promote("mid.y", "y");
        connect("y", "c.x");  // from the promoted alias
    }
};

// ---- ties -----------------------------------------------------------------------------------
class TieInSys : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", 2.0f, 0.0f);
        add_subsystem<Axpb>("b", 3.0f, 0.0f);
        tie_inputs({"a.x", "b.x"}, "x");
    }
};

class TieInInner : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", 2.0f, 0.0f);
        add_subsystem<Axpb>("b", 3.0f, 0.0f);
        tie_inputs({"a.x", "b.x"}, "t");
    }
};

class TieInTop : public cs::System {
public:
    void setup() override {
        add_subsystem<TieInInner>("in");
        add_subsystem<Axpb>("c", 4.0f, 0.0f);
        tie_inputs({"in.t", "c.x"}, "tt");  // a tie of a tie alias: three members, one root
    }
};

class TieOutAtomic : public cs::System {
public:
    void setup() override {
        add_subsystem<Adder>("a", 1.0f);
        add_subsystem<Adder>("b", 2.0f);
        add_subsystem<Copy1>("c");
        tie_outputs({"a.acc", "b.acc"}, "sum");  // require_atomic (the default): both are atomic
        connect("sum", "c.x");                    // a tie alias is a source
    }
};

class TieOutPlain : public cs::System {
    bool allow_;

public:
    explicit TieOutPlain(bool allow = false) : allow_{allow} {}
    void setup() override {
        add_subsystem<Src>("a");
        add_subsystem<Src>("b");
        if (allow_) {
            tie_outputs({"a.y", "b.y"}, "t", cs::TieOutputsPolicy::allow_plain);
        } else {
            tie_outputs({"a.y", "b.y"}, "t");
        }
    }
};

// ---- marks ----------------------------------------------------------------------------------
class MarksSys : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("src");
        add_subsystem<Axpb>("a", 1.0f, 0.0f);
        add_subsystem<Axpb>("con", 1.0f, 0.0f);
        add_subsystem<Pick>("obj");
        connect("src.y", "con.x");

        cs::Bounds first;
        first.lower = std::vector<double>{0.0};
        first.upper = std::vector<double>{10.0};
        add_design_var("a.x", first);  // replaced by the next design var on a.x
        cs::Bounds second;
        second.lower = std::vector<double>{-1.0};
        second.upper = std::vector<double>{20.0};
        add_design_var("a.x", second);
        add_design_var("obj.x");

        cs::Scaling s1;
        s1.ref = std::vector<double>{3.0};
        add_objective("obj.f", s1);  // replaced too
        cs::Scaling s2;
        s2.ref = std::vector<double>{2.0};
        add_objective("obj.f", s2);

        cs::Bounds c1;
        c1.upper = std::vector<double>{5.0};
        add_constraint("a.y", c1);  // replaced by the next constraint on a.y
        cs::Bounds c2;
        c2.upper = std::vector<double>{7.0};
        add_constraint("a.y", c2);
        cs::Bounds c3;
        c3.lower = std::vector<double>{1.0};
        add_constraint("con.y", c3);
    }
};

class DesignVarConnected : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", 1.0f, 0.0f);
        add_subsystem<Axpb>("c", 1.0f, 0.0f);
        connect("a.y", "c.x");
        add_design_var("c.x");  // computed, not designed
    }
};

class DesignVarDynamic : public cs::System {
public:
    void setup() override {
        add_subsystem<Copy1>("c");  // c.x takes its shape from a source it never gets
        add_design_var("c.x");
    }
};

class ObjectiveBig : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("src");
        add_objective("src.y");  // flat size 8
    }
};

// ---- init order -----------------------------------------------------------------------------
constexpr float kD1Init = 99.0f;   // c1's own init: overridden
constexpr float kSysInit = 1.5f;   // the owner System's init: the last word

class InnerB : public LogSystem {
public:
    void setup() override {
        add_subsystem<Axpb>("c1", 1.0f, 0.0f, kD1Init);
        add_subsystem<Axpb>("c2", 1.0f, 0.0f);
        connect("c1.y", "c2.x");
    }
    void init(sk::queue& q, const sycl::device& d) override {
        LogSystem::init(q, d);
        fill_input("c1.x", kSysInit);  // the System pass runs after c1's own init
    }
};

class TopB : public LogSystem {
public:
    void setup() override {
        add_subsystem<InnerB>("inner");
        add_subsystem<Axpb>("c3", 1.0f, 0.0f);
        connect("inner.c2.y", "c3.x");
    }
};

// ---- transient ------------------------------------------------------------------------------
class TransSys : public LogSystem {
public:
    void setup() override {
        add_subsystem<Src>("prod");
        add_subsystem<Axpb>("mid", 2.0f, 0.0f);
        add_subsystem<Axpb>("cons", 1.0f, 3.0f);
        connect("prod.y", "mid.x");
        connect("mid.y", "cons.x");
    }
};

class TopC : public LogSystem {
public:
    void setup() override {
        add_subsystem<TransSys>("tr", cs::property_list{cs::property::transient{}});
        add_subsystem<Axpb>("sink", 1.0f, 0.5f);
        connect("tr.cons.y", "sink.x");  // the consumer's output feeds a Component OUTSIDE tr
    }
};

// ---- structural errors ----------------------------------------------------------------------
class ThreeAxpb : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", 1.0f, 0.0f);
        add_subsystem<Axpb>("b", 1.0f, 0.0f);
        add_subsystem<Axpb>("c", 1.0f, 0.0f);
    }
};

class FanIn : public ThreeAxpb {
public:
    void setup() override {
        ThreeAxpb::setup();
        connect("a.y", "c.x");
        connect("b.y", "c.x");  // a second source into the same input
    }
};

class TieAfterConnect : public ThreeAxpb {
public:
    void setup() override {
        ThreeAxpb::setup();
        connect("a.y", "c.x");
        tie_inputs({"c.x", "b.x"}, "t");  // c.x is already connected
    }
};

class ConnectFromInput : public ThreeAxpb {
public:
    void setup() override {
        ThreeAxpb::setup();
        connect("b.x", "c.x");  // an input is never a source (D61)
    }
};

class ConnectUnknownEntry : public ThreeAxpb {
public:
    void setup() override {
        ThreeAxpb::setup();
        connect("a.nope", "c.x");
    }
};

class ConnectUnknownSub : public ThreeAxpb {
public:
    void setup() override {
        ThreeAxpb::setup();
        connect("zz.y", "c.x");
    }
};

// ---- placement ------------------------------------------------------------------------------
class TopPlace : public cs::System {
    sycl::device dev_;
    bool         placed_;

public:
    TopPlace(const sycl::device& d, bool placed) : dev_{d}, placed_{placed} {}
    void setup() override {
        // Placed: property::device{the parent's own device}; unplaced: no property at all.
        const cs::property_list props =
            placed_ ? cs::property_list{cs::property::device{dev_}} : cs::property_list{};
        add_subsystem<Src>("src");
        add_subsystem<Axpb>("kid", props, 2.0f, 1.0f);  // a Component placed on the parent's device
        // The helper form: a plain System on that device holding two Components.
        add_subsystem("grp", props, child<Axpb>("g1", 1.0f, 2.0f), child<Axpb>("g2", 3.0f, 0.0f));
        connect("src.y", "kid.x");
        connect("kid.y", "grp.g1.x");
        connect("grp.g1.y", "grp.g2.x");
    }
};

// ---- Sellar ---------------------------------------------------------------------------------
constexpr float kSellarD1Y2 = 99.0f;  // d1's own init: overridden
constexpr float kSellarY2   = 1.5f;   // the MDA System's init: the last word

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
    void init(sk::queue&, const sycl::device&) override { fill_input("y2", kSellarD1Y2); }
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
        out.y2[i]              = y2;
        out.yy[std::size_t{0}] = y1;
        out.yy[std::size_t{1}] = y2;
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
        out.f[i]      = x * x + in.z[std::size_t{1}] + in.y1[std::size_t{0}] + sycl::exp(-in.y2[std::size_t{0}]);
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
    void init(sk::queue&, const sycl::device&) override { fill_input("d1.y2", kSellarY2); }
};

class SellarTop : public cs::System {
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
        cs::Bounds b1;
        b1.upper = std::vector<double>{0.0};
        add_constraint("g1.g", b1);
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

/// The one value of a rank-1 output.
float scalar_of(cs::Model& model, const char* path) { return to_vector(model.view_output<DType::f32, 1>(path)).at(0); }

/// The mark on `root`, or null.
const cs::MarkRecord* mark_on(std::span<const cs::MarkRecord> marks, const cs::detail::IOEntry* root) {
    for (const auto& m : marks)
        if (m.root == root) return &m;
    return nullptr;
}

}  // namespace tsys

// =============================================================================================
// 1. connect
// =============================================================================================
TEST_CASE("connect: identity, Slice of rank 1 and 2, view composition", "[system][connect]") {
    SECTION("identity: producer to consumer, checked on values") {
        cs::Model model = cs::create_model<tsys::ConnectSys>(default_config());
        model.setup();
        auto y = wait_and_view<DType::f32, 1>(model, "cons.y");
        CHECK(all_close(to_vector(y), tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.0f)));  // 2 (i + 1) + 1
        CHECK(model.root().entry_at("prod.y").root() == model.root().entry_at("cons.x").root());
    }

    SECTION("Slice: the consumer reads {offset, extent} of a rank-1 output") {
        cs::Model model = cs::create_model<tsys::Slice1Sys>(default_config());
        model.setup();
        auto y = wait_and_view<DType::f32, 1>(model, "c.y");
        REQUIRE(y.size() == 3);
        CHECK(to_vector(y) == std::vector<float>{3.0f, 4.0f, 5.0f});  // prod.y[2..4] = 3, 4, 5
        // The connected input itself shows the same region.
        CHECK(to_vector(model.view_input<DType::f32, 1>("c.x")) == std::vector<float>{3.0f, 4.0f, 5.0f});
    }

    SECTION("Slice: a 2-D subarray of a rank-2 output") {
        cs::Model model = cs::create_model<tsys::Slice2Sys>(default_config());
        model.setup();
        auto y = wait_and_view<DType::f32, 2>(model, "c.y");
        REQUIRE(y.shape() == sycl::range<2>{2, 3});
        CHECK(to_vector(y) == std::vector<float>{1002.0f, 1003.0f, 1004.0f, 2002.0f, 2003.0f, 2004.0f});
    }

    SECTION("composition: a Slice of a promoted alias that carries a Slice adds the offsets") {
        cs::Model model = cs::create_model<tsys::ComposeTop>(default_config());
        model.setup();
        model.forward();
        model.wait();
        // The inner consumer sees elements 2..5; the alias carries the same view.
        CHECK(to_vector(model.view_output<DType::f32, 1>("in.c.y")) == std::vector<float>{3.0f, 4.0f, 5.0f, 6.0f});
        CHECK(to_vector(model.view_output<DType::f32, 1>("in.cx")) == std::vector<float>{3.0f, 4.0f, 5.0f, 6.0f});
        // The outer consumer sees element 1..2 OF THAT: root elements 3..4 = 4, 5.
        auto y = model.view_output<DType::f32, 1>("c2.y");
        REQUIRE(y.size() == 2);
        CHECK(to_vector(y) == std::vector<float>{4.0f, 5.0f});
        const auto& v = model.root().entry_at("c2.x").view();
        REQUIRE(v.has_value());
        CHECK(v->offset[0] == 3);
        CHECK(v->extents[0] == 2);
    }
}

// =============================================================================================
// 2. promote
// =============================================================================================
TEST_CASE("promote: a two-level chain reads the same root", "[system][promote]") {
    cs::Model model = cs::create_model<tsys::PromoteTop>(default_config());
    model.setup();
    model.forward();
    model.wait();

    const auto want = tsys::iota_plus(1.0f, tsys::kLen);
    CHECK(to_vector(model.view_output<DType::f32, 1>("mid.child.y")) == want);  // the full path
    CHECK(to_vector(model.view_output<DType::f32, 1>("mid.y")) == want);        // promoted once
    CHECK(to_vector(model.view_output<DType::f32, 1>("y")) == want);            // promoted twice
    CHECK(all_close(to_vector(model.view_output<DType::f32, 1>("c.y")), tsys::iota_plus(1.0f, tsys::kLen, 3.0f)));

    auto& sys = model.root();
    CHECK(sys.entry_at("y").root() == sys.entry_at("mid.child.y").root());
    CHECK(sys.entry_at("mid.y").root() == sys.entry_at("mid.child.y").root());
    CHECK(sys.entry_at("c.x").root() == sys.entry_at("mid.child.y").root());  // connected from the alias
}

// =============================================================================================
// 3. ties
// =============================================================================================
TEST_CASE("ties: tie_inputs gives one root; a tie of a tie gives three members and one root",
          "[system][tie]") {
    std::vector<float> x(tsys::kLen);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i) * 0.5f + 1.0f;

    SECTION("one tie: set_input on the alias is seen by both") {
        cs::Model model = cs::create_model<tsys::TieInSys>(default_config());
        model.setup();
        CHECK(model.root().entry_at("a.x").root() == model.root().entry_at("b.x").root());
        CHECK(model.root().entry_at("x").root() == model.root().entry_at("a.x").root());
        model.set_input<DType::f32, 1>("x", std::span<const float>{x});
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_input<DType::f32, 1>("a.x")) == x);
        CHECK(to_vector(model.view_input<DType::f32, 1>("b.x")) == x);
        std::vector<float> want2(x), want3(x);
        for (auto& v : want2) v *= 2.0f;
        for (auto& v : want3) v *= 3.0f;
        CHECK(all_close(to_vector(model.view_output<DType::f32, 1>("a.y")), want2));
        CHECK(all_close(to_vector(model.view_output<DType::f32, 1>("b.y")), want3));
    }

    SECTION("a tie of a tie alias across two levels") {
        cs::Model model = cs::create_model<tsys::TieInTop>(default_config());
        model.setup();
        auto& sys = model.root();
        const auto* root = sys.entry_at("tt").root();
        CHECK(sys.entry_at("in.a.x").root() == root);
        CHECK(sys.entry_at("in.b.x").root() == root);
        CHECK(sys.entry_at("c.x").root() == root);
        CHECK(sys.entry_at("in.t").root() == root);
        model.set_input<DType::f32, 1>("tt", std::span<const float>{x});
        model.forward();
        model.wait();
        for (const char* path : {"in.a.x", "in.b.x", "c.x", "in.t"})
            CHECK(to_vector(model.view_input<DType::f32, 1>(path)) == x);
        std::vector<float> want4(x);
        for (auto& v : want4) v *= 4.0f;
        CHECK(all_close(to_vector(model.view_output<DType::f32, 1>("c.y")), want4));
    }
}

TEST_CASE("ties: tie_outputs", "[system][tie]") {
    SECTION("require_atomic on atomic outputs: both writers accumulate into the one root") {
        cs::Model model = cs::create_model<tsys::TieOutAtomic>(default_config());
        model.setup();
        CHECK(model.root().entry_at("a.acc").root() == model.root().entry_at("b.acc").root());
        auto sum = wait_and_view<DType::f32, 1>(model, "sum");
        CHECK(to_vector(sum) == std::vector<float>(tsys::kLen, 3.0f));  // 1 + 2
        CHECK(to_vector(model.view_output<DType::f32, 1>("c.y")) == std::vector<float>(tsys::kLen, 3.0f));
        // Not reset between forwards: a second forward accumulates again.
        auto again = wait_and_view<DType::f32, 1>(model, "sum");
        CHECK(to_vector(again) == std::vector<float>(tsys::kLen, 6.0f));
    }

    SECTION("plain outputs without allow_plain throw") {
        tsys::expect_setup_throws<tsys::TieOutPlain>("not atomic");
        tsys::expect_throws(
            [] {
                cs::Model model = cs::create_model<tsys::TieOutPlain>(default_config(), false);
                model.setup();
            },
            "require_atomic");
    }

    SECTION("plain outputs with allow_plain are accepted") {
        cs::Model model = cs::create_model<tsys::TieOutPlain>(default_config(), true);
        model.setup();
        CHECK(model.root().entry_at("a.y").root() == model.root().entry_at("b.y").root());
        CHECK(to_vector(wait_and_view<DType::f32, 1>(model, "t")) == tsys::iota_plus(1.0f, tsys::kLen));
    }
}

// =============================================================================================
// 4. marks
// =============================================================================================
TEST_CASE("marks: recorded on the Model with paths and bounds; the last of a kind replaces", "[system][marks]") {
    cs::Model model = cs::create_model<tsys::MarksSys>(default_config());
    model.setup();
    auto& sys = model.root();

    const auto dvs = model.design_vars();
    const auto cons = model.constraints();
    const auto objs = model.objectives();
    CHECK(dvs.size() == 2);   // a.x (once, replaced) and obj.x
    CHECK(cons.size() == 2);  // a.y (once, replaced) and con.y
    CHECK(objs.size() == 1);  // obj.f

    SECTION("design vars") {
        const auto* ax = tsys::mark_on(dvs, sys.entry_at("a.x").root());
        REQUIRE(ax != nullptr);
        REQUIRE(ax->bounds.lower.has_value());
        REQUIRE(ax->bounds.upper.has_value());
        CHECK(*ax->bounds.lower == std::vector<double>{-1.0});  // the second mark won
        CHECK(*ax->bounds.upper == std::vector<double>{20.0});
        const auto* ox = tsys::mark_on(dvs, sys.entry_at("obj.x").root());
        REQUIRE(ox != nullptr);
        CHECK(!ox->bounds.lower.has_value());
        CHECK(!ox->bounds.upper.has_value());
    }

    SECTION("constraints") {
        const auto* ay = tsys::mark_on(cons, sys.entry_at("a.y").root());
        REQUIRE(ay != nullptr);
        REQUIRE(ay->bounds.upper.has_value());
        CHECK(*ay->bounds.upper == std::vector<double>{7.0});  // 5.0 was replaced
        CHECK(!ay->bounds.lower.has_value());
        // con.y's root is src.y's? No: con.y is its own output root.
        const auto* cy = tsys::mark_on(cons, sys.entry_at("con.y").root());
        REQUIRE(cy != nullptr);
        REQUIRE(cy->bounds.lower.has_value());
        CHECK(*cy->bounds.lower == std::vector<double>{1.0});
        CHECK(!cy->bounds.upper.has_value());
    }

    SECTION("the objective") {
        const auto* f = tsys::mark_on(objs, sys.entry_at("obj.f").root());
        REQUIRE(f != nullptr);
        REQUIRE(f->scaling.ref.size() == 1);
        CHECK(f->scaling.ref[0] == 2.0);  // the second mark's scaling
    }
}

TEST_CASE("marks: ill-formed marks throw at setup", "[system][marks]") {
    SECTION("a design var on a connected input") {
        tsys::expect_setup_throws<tsys::DesignVarConnected>("connected inputs are computed, not designed");
    }
    SECTION("an objective of flat size > 1") {
        tsys::expect_setup_throws<tsys::ObjectiveBig>("flat size 8");
    }
    SECTION("a dynamic design var") {
        tsys::expect_setup_throws<tsys::DesignVarDynamic>("dynamic");
    }
}

TEST_CASE("marks: the Model's spans are unavailable before setup", "[system][marks]") {
    cs::Model model = cs::create_model<tsys::MarksSys>(default_config());
    tsys::expect_throws([&] { (void)model.design_vars(); }, "before setup() has completed");
}

// =============================================================================================
// 5. init order
// =============================================================================================
TEST_CASE("init order: Component pass, then System pass, children before owners, the System has the last word",
          "[system][init]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::TopB>(default_config());
    model.setup();

    REQUIRE(tsys::g_inits.size() == 5);  // 3 Components + 2 Systems, each once
    const std::size_t last_comp =
        std::max({tsys::index_of("top.inner.c1"), tsys::index_of("top.inner.c2"), tsys::index_of("top.c3")});
    const std::size_t first_sys = std::min(tsys::index_of("top.inner"), tsys::index_of("top"));
    CHECK(tsys::count_of("top.inner.c1") == 1);
    CHECK(tsys::count_of("top.inner.c2") == 1);
    CHECK(tsys::count_of("top.c3") == 1);
    CHECK(last_comp < first_sys);                                          // Component pass first
    CHECK(tsys::index_of("top.inner") < tsys::index_of("top"));            // a child System before its owner

    // c1's own init wrote 99; the owner System's init wrote 1.5 afterwards.
    CHECK(to_vector(model.view_input<DType::f32, 1>("inner.c1.x")) == std::vector<float>(tsys::kLen, tsys::kSysInit));
    model.forward();
    model.wait();
    CHECK(to_vector(model.view_output<DType::f32, 1>("c3.y")) == std::vector<float>(tsys::kLen, tsys::kSysInit));
}

// =============================================================================================
// 6. transient
// =============================================================================================
namespace {
std::vector<float> want_cons() { return tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 3.0f); }  // 2 (i + 1) + 3
std::vector<float> want_sink() {
    auto v = want_cons();
    for (auto& e : v) e += 0.5f;
    return v;
}
}  // namespace

TEST_CASE("transient: release after forward, kept outputs, revive, back-to-back forwards", "[system][transient][os-free]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::TopC>(default_config());
    model.setup();
    auto& tr = dynamic_cast<cs::System&>(model.get_subsystem("tr"));

    SECTION("release after forward: host access to a released root throws, naming the System") {
        model.forward();
        model.wait();
        CHECK(tr.transient());
        CHECK(tr.released());
        auto& sys = model.root();
        CHECK(sys.entry_at("tr.prod.y").root()->released());
        CHECK(sys.entry_at("tr.mid.y").root()->released());
        CHECK(!sys.entry_at("tr.mid.y").root()->has_storage());
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("tr.mid.y"); }, "top.tr");
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("tr.prod.y"); }, "transient System");
    }

    SECTION("an output connected outside the subtree is kept") {
        model.forward();
        model.wait();
        auto& sys  = model.root();
        auto* cons = sys.entry_at("tr.cons.y").root();
        CHECK(cons->has_storage());
        CHECK(!cons->released());
        CHECK(!cons->transient());
        CHECK(sys.entry_at("tr.mid.y").root()->transient());  // the intermediates are the transient ones
        CHECK(to_vector(model.view_output<DType::f32, 1>("tr.cons.y")) == want_cons());
        CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == want_sink());
    }

    SECTION("forward 2 revives and re-runs only the subtree's inits") {
        model.forward();
        model.wait();
        CHECK(tsys::count_of("top.tr.prod") == 1);  // setup's init: nothing re-ran yet
        CHECK(tsys::count_of("top.tr") == 1);
        CHECK(tr.released());

        model.forward();  // revive: re-allocate, re-run tr's subtree init (both passes), run, release
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("tr.cons.y")) == want_cons());
        CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == want_sink());
        for (const char* p : {"top.tr.prod", "top.tr.mid", "top.tr.cons", "top.tr"}) {
            INFO(p);
            CHECK(tsys::count_of(p) == 2);  // both passes re-ran
        }
        CHECK(tsys::count_of("top.sink") == 1);  // nothing outside tr re-ran
        CHECK(tsys::count_of("top") == 1);
        CHECK(tr.released());
    }

    SECTION("three forwards enqueued back to back, then one wait") {
        model.forward();
        model.forward();
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("tr.cons.y")) == want_cons());
        CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == want_sink());
        CHECK(tsys::count_of("top.tr.prod") == 3);  // setup, then a revive before forwards 2 and 3
        CHECK(tsys::count_of("top.sink") == 1);
        CHECK(tr.released());
    }
}

// =============================================================================================
// 7. structural errors
// =============================================================================================
TEST_CASE("structural errors", "[system][errors]") {
    SECTION("fan-in: a second connect into the same input") {
        tsys::expect_setup_throws<tsys::FanIn>("already has a source");
    }
    SECTION("tie after connect") {
        tsys::expect_setup_throws<tsys::TieAfterConnect>("tie_inputs");
        tsys::expect_setup_throws<tsys::TieAfterConnect>("not a free input");
    }
    SECTION("connect from an input (D61)") {
        tsys::expect_setup_throws<tsys::ConnectFromInput>("cannot be a source");
    }
    SECTION("an unknown path in connect names the path") {
        tsys::expect_setup_throws<tsys::ConnectUnknownEntry>("a.nope");
        tsys::expect_setup_throws<tsys::ConnectUnknownSub>("zz");
    }
    SECTION("a structural call outside setup()") {
        cs::Model model = cs::create_model<tsys::ThreeAxpb>(default_config());
        model.setup();
        auto& sys = model.root();
        tsys::expect_throws([&] { sys.connect("a.y", "b.x"); }, "setup() only");
        tsys::expect_throws([&] { sys.promote("a.y", "y"); }, "setup() only");
        tsys::expect_throws([&] { sys.tie_inputs({"a.x", "b.x"}, "t"); }, "setup() only");
        tsys::expect_throws([&] { sys.add_design_var("a.x"); }, "setup() only");
    }
    SECTION("an unknown path in view_output names the path") {
        cs::Model model = cs::create_model<tsys::ThreeAxpb>(default_config());
        model.setup();
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("a.nope"); }, "a.nope");
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("zz.y"); }, "zz");
        // The kind is checked too.
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("a.x"); }, "is an input");
        tsys::expect_throws([&] { (void)model.view_input<DType::f32, 1>("a.y"); }, "is an output");
    }
}

// =============================================================================================
// 8. placement
// =============================================================================================
TEST_CASE("placement: property::device equal to the parent's is a no-op", "[system][placement]") {
    const sycl::device dev{sycl::default_selector_v};  // the Model's default device

    cs::Model plain = cs::create_model<tsys::TopPlace>(default_config(), dev, false);
    cs::Model placed = cs::create_model<tsys::TopPlace>(default_config(), dev, true);
    plain.setup();
    placed.setup();

    const cs::Placement here{dev, sk::kind::device};
    for (const char* path : {"kid.x", "kid.y", "grp.g1.x", "grp.g2.y"}) {
        INFO(path);
        CHECK(plain.root().entry_at(path).placement() == here);
        CHECK(placed.root().entry_at(path).placement() == here);
        CHECK(plain.root().entry_at(path).placement() == placed.root().entry_at(path).placement());
    }

    std::vector<float> want_kid = tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.0f);  // kid: 2x + 1
    std::vector<float> want_g2(tsys::kLen);
    for (std::size_t i = 0; i < want_g2.size(); ++i) want_g2[i] = 3.0f * (want_kid[i] + 2.0f);  // g1: x + 2, g2: 3x
    for (cs::Model* m : {&plain, &placed}) {
        auto kid = wait_and_view<DType::f32, 1>(*m, "kid.y");
        CHECK(all_close(to_vector(kid), want_kid));
        CHECK(all_close(to_vector(m->view_output<DType::f32, 1>("grp.g2.y")), want_g2));
    }
}

// =============================================================================================
// Sellar
// =============================================================================================
TEST_CASE("Sellar-shaped forward: ties of ties, promotions, a Slice, marks, init last word", "[system][sellar]") {
    cs::Model model = cs::create_model<tsys::SellarTop>(default_config());
    model.setup();
    model.forward();
    model.wait();

    // Host reference in the same float arithmetic.
    const float z0 = 5.0f, z1 = 2.0f, x = 1.0f;
    const float y1 = z0 * z0 + z1 + x - 0.2f * tsys::kSellarY2;
    const float y2 = std::sqrt(y1) + z0 + z1;
    const float f  = x * x + z1 + y1 + std::exp(-y2);
    const float g1 = 3.16f - y1, g2 = y2 - 24.0f;

    CHECK(close(tsys::scalar_of(model, "mda.d1.y1"), y1));
    CHECK(close(tsys::scalar_of(model, "mda.d2.y2"), y2));
    CHECK(close(tsys::scalar_of(model, "mda.y2"), y2));  // the promoted alias reads d2.y2
    CHECK(close(tsys::scalar_of(model, "obj.f"), f));     // obj.y2 is element 1 of yy, through the Slice
    CHECK(close(tsys::scalar_of(model, "g1.g"), g1));
    CHECK(close(tsys::scalar_of(model, "g2.g"), g2));
    // d1's own init wrote 99; the MDA System's init wrote 1.5 afterwards.
    CHECK(to_vector(model.view_input<DType::f32, 1>("mda.d1.y2")) == std::vector<float>{tsys::kSellarY2});

    auto& sys = model.root();
    const auto* zroot = sys.entry_at("z").root();
    CHECK(sys.entry_at("mda.d1.z").root() == zroot);
    CHECK(sys.entry_at("mda.d2.z").root() == zroot);
    CHECK(sys.entry_at("obj.z").root() == zroot);

    // Marks ride on the roots; z's sits on the ONE shared root.
    CHECK(model.design_vars().size() == 2);
    CHECK(tsys::mark_on(model.design_vars(), zroot) != nullptr);
    CHECK(tsys::mark_on(model.design_vars(), sys.entry_at("x").root()) != nullptr);
    CHECK(model.objectives().size() == 1);
    CHECK(tsys::mark_on(model.objectives(), sys.entry_at("obj.f").root()) != nullptr);
    CHECK(model.constraints().size() == 2);
    CHECK(tsys::mark_on(model.constraints(), sys.entry_at("g1.g").root()) != nullptr);
    CHECK(tsys::mark_on(model.constraints(), sys.entry_at("g2.g").root()) != nullptr);
}
