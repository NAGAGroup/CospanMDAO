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

// The test-only root driver (start / enqueue / wait / forward) is shared with
// smoke_parallel.
#include "driver.hpp"

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

// ---- shared pieces of b-e: small fixed-shape components that log their init -----------
constexpr std::size_t kLen = 8;

static std::vector<std::string> g_inits;  // every Component and System appends its path in init

static std::size_t count_of(const std::string& path) {
    std::size_t n = 0;
    for (const auto& p : g_inits) n += (p == path);
    return n;
}
static std::size_t index_of(const std::string& path) {  // first occurrence, or npos
    for (std::size_t i = 0; i < g_inits.size(); ++i)
        if (g_inits[i] == path) return i;
    return std::string::npos;
}

static std::vector<float> values_of(sk::queue& q, cs::detail::IOEntry& e) {
    auto               h = HostViewAccess::view<DType::f32, 1>(e, q);
    std::vector<float> v(h.size());
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = h[i];
    return v;
}

/// Component whose init logs its path.
template <class Body>
class Logged : public cs::Component<Body> {
public:
    void init(sk::queue&, const sycl::device&) override { g_inits.push_back(this->path()); }
};

/// System whose init logs its path.
class LogSystem : public cs::System {
public:
    void init(sk::queue&, const sycl::device&) override { g_inits.push_back(path()); }
};

// Src: y[i] = i + 1.
struct SrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = static_cast<float>(i[0] + 1); }
};

class Src : public Logged<SrcBody> {
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override { declare_output(y_, "y", static_cast<int>(kLen)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {}, {y_}, {});
    }
};

// Axpb: y[i] = a * x[i] + b.
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

class Axpb : public Logged<AxpbBody> {
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

// ---- b. init order -------------------------------------------------------------------
class InnerB : public LogSystem {
public:
    void setup() override {
        add_subsystem<Axpb>("c1", 1.0f, 0.0f);
        add_subsystem<Axpb>("c2", 1.0f, 0.0f);
        connect("c1.y", "c2.x");
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

static void case_b(const sycl::device& dev, sk::queue& q) {
    std::printf("-- b. init order: Component pass, then System pass, children before owners\n");
    g_inits.clear();
    TopB       top;
    RootDriver d{dev, q, top};
    d.start();

    check(g_inits.size() == 5, "b: five inits ran (3 Components, 2 Systems), each once");
    const std::size_t last_comp = std::max({index_of("top.inner.c1"), index_of("top.inner.c2"), index_of("top.c3")});
    const std::size_t first_sys = std::min(index_of("top.inner"), index_of("top"));
    check(count_of("top.inner.c1") == 1 && count_of("top.inner.c2") == 1 && count_of("top.c3") == 1 &&
              last_comp < first_sys,
          "b: every Component init precedes every System init");
    check(index_of("top.inner") < index_of("top"), "b: a child System's init precedes its owner's");
}

// ---- c. transient --------------------------------------------------------------------
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

static void case_c(const sycl::device& dev, sk::queue& q) {
    std::printf("-- c. transient: prod -> mid -> cons inside tr, cons.y feeds sink outside\n");
    g_inits.clear();
    TopC       top;
    RootDriver d{dev, q, top};
    d.start();

    std::vector<float> want_cons(kLen), want_sink(kLen);
    for (std::size_t i = 0; i < kLen; ++i) {
        want_cons[i] = 2.0f * static_cast<float>(i + 1) + 3.0f;  // prod -> mid (2x) -> cons (x+3)
        want_sink[i] = want_cons[i] + 0.5f;                       // sink (x+0.5)
    }

    d.forward();
    auto& tr   = dynamic_cast<cs::System&>(top.get_subsystem("tr"));
    auto& prod = *top.entry_at("tr.prod.y").root();
    auto& mid  = *top.entry_at("tr.mid.y").root();
    auto& cons = *top.entry_at("tr.cons.y").root();
    check(tr.transient() && tr.released(), "c: after forward 1, tr is transient and released");
    check(!mid.has_storage() && mid.released() && mid.transient() && !prod.has_storage() && prod.released(),
          "c: the intermediate roots (prod.y, mid.y) report no storage (released)");
    check(cons.has_storage() && !cons.released() && !cons.transient(),
          "c: the outside-connected output root (cons.y) is not transient and still has storage");
    check(values_of(q, top.entry_at("tr.cons.y")) == want_cons, "c: forward 1: cons.y (kept) has the right values");
    check(values_of(q, top.entry_at("sink.y")) == want_sink, "c: forward 1: sink.y has the right values");

    bool threw = false, names_tr = false;
    try {
        (void)HostViewAccess::view<DType::f32, 1>(top.entry_at("tr.mid.y"), q);
    } catch (const std::exception& e) {
        threw    = true;
        names_tr = std::string{e.what()}.find("top.tr") != std::string::npos;
        std::printf("      released-root HostView threw: %s\n", e.what());
    }
    check(threw && names_tr, "c: a HostView on the released root throws, naming the transient System");

    check(count_of("top.tr.prod") == 1 && count_of("top.tr.mid") == 1 && count_of("top.tr") == 1,
          "c: before the revive, tr's subtree inits ran once");
    d.forward();  // revive: re-allocate, re-run tr's subtree init (both passes), run, release
    check(values_of(q, top.entry_at("tr.cons.y")) == want_cons, "c: forward 2 (revived): cons.y is the same");
    check(values_of(q, top.entry_at("sink.y")) == want_sink, "c: forward 2 (revived): sink.y is the same");
    check(count_of("top.tr.prod") == 2 && count_of("top.tr.mid") == 2 && count_of("top.tr.cons") == 2 &&
              count_of("top.tr") == 2 && count_of("top.sink") == 1 && count_of("top") == 1,
          "c: the revive re-ran tr's subtree init in both passes; nothing outside tr re-ran");
    check(tr.released() && !mid.has_storage(), "c: after forward 2, tr is released again");
}

// ---- d. structural errors --------------------------------------------------------------
/// Runs `T`'s setup as a root; returns whether it threw, with the message.
template <class T>
static bool setup_throws(std::string& msg) {
    T s;
    cs::detail::SubsystemAccess::place(s, "top", "top", nullptr);
    try {
        s.run_setup();
    } catch (const std::exception& e) {
        msg = e.what();
        return true;
    }
    return false;
}

static bool has(const std::string& msg, const char* what) { return msg.find(what) != std::string::npos; }

class FanIn : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", 1.0f, 0.0f);
        add_subsystem<Axpb>("b", 1.0f, 0.0f);
        add_subsystem<Axpb>("c", 1.0f, 0.0f);
        connect("a.y", "c.x");
        connect("b.y", "c.x");  // second source into the same input
    }
};

class TieAfterConnect : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", 1.0f, 0.0f);
        add_subsystem<Axpb>("b", 1.0f, 0.0f);
        add_subsystem<Axpb>("c", 1.0f, 0.0f);
        connect("a.y", "c.x");
        tie_inputs({"c.x", "b.x"}, "t");  // c.x is already connected
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

class OutsideSetup : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", 1.0f, 0.0f);
        add_subsystem<Axpb>("b", 1.0f, 0.0f);
    }
};

static void case_d() {
    std::printf("-- d. structural errors: each in its own small System\n");
    std::string msg;
    check(setup_throws<FanIn>(msg) && has(msg, "connect") && has(msg, "already has a source"),
          "d: fan-in (a second connect into the same input) throws");
    std::printf("      %s\n", msg.c_str());
    msg.clear();
    check(setup_throws<TieAfterConnect>(msg) && has(msg, "tie_inputs") && has(msg, "not a free input"),
          "d: tie_inputs after that input is connected throws");
    std::printf("      %s\n", msg.c_str());
    msg.clear();
    check(setup_throws<DesignVarConnected>(msg) && has(msg, "add_design_var") &&
              has(msg, "connected inputs are computed, not designed"),
          "d: add_design_var on a connected input throws");
    std::printf("      %s\n", msg.c_str());
    msg.clear();

    // connect from outside setup(): the System is fully set up, then called into.
    OutsideSetup s;
    cs::detail::SubsystemAccess::place(s, "top", "top", nullptr);
    s.run_setup();
    bool threw = false;
    try {
        s.connect("a.y", "b.x");
    } catch (const std::exception& e) {
        threw = has(e.what(), "setup() only");
        std::printf("      %s\n", e.what());
    }
    check(threw, "d: a structural call (connect) from outside setup() throws");
}

// ---- e. property::device ---------------------------------------------------------------
class TopE : public LogSystem {
    sycl::device dev_;

public:
    explicit TopE(const sycl::device& d) : dev_{d} {}
    void setup() override {
        const cs::property_list on_dev{cs::property::device{dev_}};
        add_subsystem<Src>("src");
        add_subsystem<Axpb>("kid", on_dev, 2.0f, 1.0f);  // a Component placed on the parent's own device
        // The helper form: a plain System on the same device holding two Components.
        add_subsystem("grp", on_dev, child<Axpb>("g1", 1.0f, 2.0f), child<Axpb>("g2", 3.0f, 0.0f));
        connect("src.y", "kid.x");
        connect("kid.y", "grp.g1.x");
        connect("grp.g1.y", "grp.g2.x");
    }
};

static void case_e(const sycl::device& dev, sk::queue& q) {
    std::printf("-- e. property::device{the parent's device} on a Component and on a helper-form System\n");
    TopE       top{dev};
    RootDriver d{dev, q, top};
    d.start();
    d.forward();

    std::vector<float> want_kid(kLen), want_g2(kLen);
    for (std::size_t i = 0; i < kLen; ++i) {
        want_kid[i] = 2.0f * static_cast<float>(i + 1) + 1.0f;  // kid: 2x + 1
        want_g2[i]  = 3.0f * (want_kid[i] + 2.0f);              // g1: x + 2, g2: 3x
    }
    const cs::Placement here{dev, sk::kind::device};
    check(top.entry_at("kid.x").placement() == here && top.entry_at("grp.g2.y").placement() == here,
          "e: the placed children's entries sit on (dev, device)");
    check(values_of(q, top.entry_at("kid.y")) == want_kid, "e: the device-placed Component produces 2x + 1");
    check(values_of(q, top.entry_at("grp.g2.y")) == want_g2,
          "e: the device-placed helper-form System's chain produces 3 * (2x + 1 + 2)");
}

int main() {
    try {
        sycl::device dev{sycl::default_selector_v};
        std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        sk::context ctx(dev);
        sk::queue   q(ctx, dev);
        case_a(dev, q);
        case_b(dev, q);
        case_c(dev, q);
        case_d();
        case_e(dev, q);
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    std::printf(failures == 0 ? "SMOKE SYSTEM PASS\n" : "SMOKE SYSTEM FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
