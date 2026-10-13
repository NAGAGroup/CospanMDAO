// System tests, every case through a cospan::Model.
//
// Each case builds a small root System, makes a Model (create_model / setup), and checks through the Model
// (forward / wait / view_* / update_input and the marks). Reading results (docs/design-review-2026-10-03.md, section 5):
// forward() only enqueues; a host view is obtained after the first forward and sync()ed before it is read
// (cst::synced_output / synced_input / wait_and_view do both). The only things reached past the Model's verbs are the
// Model's own retrieval (root().instance_at, get_subsystem) and, for the structural-call check, root() itself.
//
//   connect       identity (Slices are not ported)
//   promote       child -> mid -> top chain; the promoted path and the full path name the same variable
//   ties          tie_inputs (one variable; a tie of a tie across two levels);
//                 tie_outputs (atomic writers accumulate behind a declared depends_on; plain throws unless allow_plain)
//   marks         design vars / constraints / objective recorded with their paths and bounds; the last mark of a kind
//                 replaces; the ill-formed ones throw
//   init          init_input metadata: the last declaration wins; Model::update_input on an eternal free input between forwards
//   transient     run policy: deleted after its turn, kept outputs, transient Components, promotion, nesting, a free input
//                 re-initialized each forward, back-to-back forwards
//   run-N         n runs; transient subtree allocated/initialized per run
//   run policy    run_n, do_once, transient (and the clamp table); rearm from prepare_forward
//   depends_on    declaration errors; an ordering with no data between the two children
//   structural    fan-in, tie after connect, structural call outside setup, connect from an input, unknown paths
//   placement     a device equal to the parent's is a no-op; the outer device wins; a rebuilt child keeps it
//   shape         shape errors are reactive: thrown at the first forward, not at setup (design review, section 2)
//   Sellar        a Sellar-shaped forward, with values
//
// The components live in namespace tsys (not tc): tests/test_component.cpp has
// its own, and two same-named classes in one executable would be an ODR clash.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <cstdint>
#include <iostream>
#include <span>
#include <sstream>
#include <thread>
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
    cs::Model model = cs::create_model<Sys>(default_config());
    expect_throws([&] { model.setup(); }, needle);
}

/// `f` must throw with every needle in its message.
template <class F>
void expect_throws_all(F&& f, const std::vector<std::string>& needles) {
    std::string what = "<no exception>";
    try {
        f();
    } catch (const std::exception& e) {
        what = e.what();
    }
    INFO("thrown: " << what);
    for (const auto& n : needles) CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}

constexpr std::size_t kLen = 8;

// A logging init (custom_host) appends its tag each time the framework applies it.
std::vector<std::string> g_inits;

/// An init that logs `tag` at each application and fills the root with v.
cs::InitSpec logging_fill(std::string tag, float v) {
    return cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostInit<DType::f32, 1>&)>{
        [tag = std::move(tag), v](cs::HostInit<DType::f32, 1>& h) {
            g_inits.push_back(tag);
            for (std::size_t k = 0; k < h.size(); ++k) h[k] = v;
        }});
}

/// The value DoOnceInit's `p.x` is given when its init is applied: set by a test before a rearm.
float g_p_x = 4.0f;

/// Like logging_fill, but the value is read from `*src` at each application.
cs::InitSpec logging_fill_from(std::string tag, const float* src) {
    return cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostInit<DType::f32, 1>&)>{
        [tag = std::move(tag), src](cs::HostInit<DType::f32, 1>& h) {
            g_inits.push_back(tag);
            for (std::size_t k = 0; k < h.size(); ++k) h[k] = *src;
        }});
}

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

// ---- Src: y[i] = i + 1 ------------------------------------------------------

class Src : public cs::Component<Src> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.y[i] = static_cast<float>(i[0] + 1); }

    Inputs  inputs;
    Outputs outputs;

private:
    bool                      tied_;

public:
    /// `tied`: y has all dims unspecified (a tie_outputs member; the tie alias owns the shape).
    explicit Src(bool tied = false) : tied_{tied} {}
    void declare() override {
        if (tied_)
            declare_output(outputs.y, "y", cs::unspecified_shape<1>());
        else
            declare_output(outputs.y, "y", static_cast<int>(kLen));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};

// ---- Axpb: y = a * x + b, x of kLen (x0: a fill init on x; log: a logging init instead) ------------

/// Selects the Axpb whose input x has all dims unspecified (a tie_inputs member).
struct TiedX {};
inline constexpr TiedX tied_x{};

class Axpb : public cs::Component<Axpb> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> a, b;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = in.a[std::size_t{0}] * in.x[i] + in.b[std::size_t{0}]; }

    Inputs  inputs;
    Outputs outputs;

private:
    float                     a_, b_;
    std::optional<float>      x0_;
    std::string               log_;
    bool                      tied_ = false;

public:
    Axpb(float a, float b, std::optional<float> x0 = std::nullopt, std::string log = {})
        : a_{a}, b_{b}, x0_{x0}, log_{std::move(log)} {}
    Axpb(TiedX, float a, float b, std::optional<float> x0 = std::nullopt) : a_{a}, b_{b}, x0_{x0}, tied_{true} {}
    void declare() override {
        declare_input(inputs.a, "a", 1);
        init_input("a", cs::init::fill(a_));
        declare_input(inputs.b, "b", 1);
        init_input("b", cs::init::fill(b_));
        if (tied_)
            declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        else
            declare_input(inputs.x, "x", static_cast<int>(kLen));
        declare_output(outputs.y, "y", static_cast<int>(kLen));
        if (!log_.empty())
            init_input("x", logging_fill(log_, x0_.value_or(0.0f)));
        else if (x0_)
            init_input("x", cs::init::fill(*x0_));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};

// ---- Copy1 / Copy2: y = x, the input takes the shape of what it is connected to --------------

class Copy1 : public cs::Component<Copy1> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = in.x[i]; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_output(outputs.y, "y", cs::unspecified_shape<1>());
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }  // the output follows the input (section 10: no derived dims)
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};

class Copy2 : public cs::Component<Copy2> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    static void compute_forward(sycl::id<2> ij, const Inputs& in, const Outputs& out) { out.y[ij] = in.x[ij]; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<2>());
        declare_output(outputs.y, "y", cs::unspecified_shape<2>());
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0], shape_of(inputs.x)[1]); }  // the output follows the input (section 10: no derived dims)
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};

// ---- Iota2D: out[i][j] = 1000 i + j ----------------------------------------------------------

class Iota2D : public cs::Component<Iota2D> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> out;
    };
    static void compute_forward(sycl::id<2> ij, const Inputs&, const Outputs& o) { o.out[ij] = static_cast<float>(ij[0] * 1000 + ij[1]); }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       rows_, cols_;

public:
    Iota2D(int rows, int cols) : rows_{rows}, cols_{cols} {}
    void declare() override { declare_output(outputs.out, "out", rows_, cols_); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.out.shape()));
    }
};

// ---- Adder: acc[i] += v, an ATOMIC output (zeroed by a ZeroAcc producer) ------------------------------------

class Adder : public cs::Component<Adder> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> v;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> acc;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.acc[i].fetch_add(in.v[std::size_t{0}]); }

    Inputs  inputs;
    Outputs outputs;

private:
    float                            v_;

public:
    explicit Adder(float v) : v_{v} {}
    void declare() override {
        declare_input(inputs.v, "v", 1);
        init_input("v", cs::init::fill(v_)); declare_output(outputs.acc, "acc", cs::unspecified_shape<1>()); }  // a tie_outputs member
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.acc.shape()));
    }
};

// ---- ZeroAcc: the zeroing producer (outputs are never initialized, r01 §6.0) ---

class ZeroAcc : public cs::Component<ZeroAcc> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> acc;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.acc[i].store(0.0f); }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.acc, "acc", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.acc.shape()));
    }
};

// ---- Pick: f[0] = 2 * x[0], a scalar output (one value, so it may be an objective) ------------

class Pick : public cs::Component<Pick> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> f;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.f[i] = 2.0f * in.x[std::size_t{0}]; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", static_cast<int>(kLen));
        declare_output(outputs.f, "f", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{1}));
    }
};

// ---- connect --------------------------------------------------------------------------------
class ConnectSys : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("prod"));
        add_subsystem(cs::subsystem_desc<Axpb>("cons", 2.0f, 1.0f));
        connect("prod.y", "cons.x");
    }
};

// ---- promote --------------------------------------------------------------------------------
class PromoteMid : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("child"));
        promote<DType::f32, 1>("child.y", "y");
    }
};

class PromoteTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<PromoteMid>("mid"));
        add_subsystem(cs::subsystem_desc<Axpb>("c", 3.0f, 0.0f));
        promote<DType::f32, 1>("mid.y", "y");
        connect("y", "c.x");  // from the promoted alias
    }
};

// ---- ties -----------------------------------------------------------------------------------
class TieInSys : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("a", tied_x, 2.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("b", tied_x, 3.0f, 0.0f));
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(kLen));
    }
};

class TieInInner : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("a", tied_x, 2.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("b", tied_x, 3.0f, 0.0f));
        // An alias carries a shape, so it cannot itself be a tie member: pass the members up by promotion.
        promote<DType::f32, 1>("a.x", "ax");
        promote<DType::f32, 1>("b.x", "bx");
    }
};

class TieInTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<TieInInner>("in"));
        add_subsystem(cs::subsystem_desc<Axpb>("c", tied_x, 4.0f, 0.0f));
        // Three members, two of them promotions from the level below: one root.
        tie_inputs<DType::f32, 1>({"in.ax", "in.bx", "c.x"}, "tt", cs::shape(kLen));
    }
};

class TieOutAtomic : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<ZeroAcc>("z"));  // zeroes the variable the Adders accumulate on
        add_subsystem(cs::subsystem_desc<Adder>("a", 1.0f));
        add_subsystem(cs::subsystem_desc<Adder>("b", 2.0f));
        add_subsystem(cs::subsystem_desc<Copy1>("c"));
        tie_outputs<DType::atomic_f32, 1>({"z.acc", "a.acc", "b.acc"}, "sum",
                                          cs::shape(kLen));  // require_atomic (the default): all are atomic
        connect("sum", "c.x");                    // a tie alias is a source
        // Execution order is data dependencies only (design review, section 3) and tied writers do not wait on each
        // other: the Adders accumulate onto what z writes, so the edges are declared (it used to be registration order).
        depends_on("a", "z");
        depends_on("b", "z");
    }
};

class TieOutPlain : public cs::System {
    bool allow_;

public:
    explicit TieOutPlain(bool allow = false) : allow_{allow} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("a", true));
        add_subsystem(cs::subsystem_desc<Src>("b", true));
        if (allow_) {
            tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(kLen), cs::TieOutputsPolicy::allow_plain);
        } else {
            tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(kLen));
        }
    }
};

// ---- marks ----------------------------------------------------------------------------------
class MarksSys : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("src"));
        add_subsystem(cs::subsystem_desc<Axpb>("a", 1.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("con", 1.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Pick>("obj"));
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
        add_subsystem(cs::subsystem_desc<Axpb>("a", 1.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("c", 1.0f, 0.0f));
        connect("a.y", "c.x");
        add_design_var("c.x");  // computed, not designed
    }
};

class DesignVarDynamic : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Copy1>("c"));  // c.x takes its shape from a source it never gets
        add_design_var("c.x");
    }
};

class ObjectiveBig : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("src"));
        add_objective("src.y");  // flat size 8
    }
};

// ---- init: the last declaration wins -----------------------------------------------------------
constexpr float kD1Init = 99.0f;   // c1's own init: overridden
constexpr float kSysInit = 1.5f;   // the owner System's init_input: declared after its children's setup, so it wins

class InnerB : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("c1", 1.0f, 0.0f, kD1Init));
        add_subsystem(cs::subsystem_desc<Axpb>("c2", 1.0f, 0.0f));
        connect("c1.y", "c2.x");
        init_input("c1.x", cs::init::fill(kSysInit));
    }
};

class TopB : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<InnerB>("inner"));
        add_subsystem(cs::subsystem_desc<Axpb>("c3", 1.0f, 0.0f));
        connect("inner.c2.y", "c3.x");
    }
};

// ---- transient ------------------------------------------------------------------------------
class TransSys : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("prod"));
        add_subsystem(cs::subsystem_desc<Axpb>("mid", 2.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("cons", 1.0f, 3.0f));
        connect("prod.y", "mid.x");
        connect("mid.y", "cons.x");
    }
};

class TopC : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::property_list{cs::property::transient{}}, cs::subsystem_desc<TransSys>("tr"));
        add_subsystem(cs::subsystem_desc<Axpb>("sink", 1.0f, 0.5f));
        connect("tr.cons.y", "sink.x");  // the consumer's output feeds a Component OUTSIDE tr
    }
};

// ---- transient (registration metadata) -------------------------------------------------------
inline std::size_t pool_in_use_of(cs::Model& m) { return cst::pool_in_use(m); }

const cs::property_list kTransient{cs::property::transient{}};

class OneAxpb : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("a", 2.0f, 1.0f)); }  // free x, no init: zeros
};

/// An eternal free input with a logged init (x = 5): the init is applied by the owner on its first run.
class OneAxpbLog : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("a", 2.0f, 1.0f, 5.0f, std::string{"a"})); }
};

/// a (transient Src) -> b (transient Axpb) -> c (plain Axpb): a.y is transient, b.y is eternal.
class TransComps : public cs::System {
public:
    void setup() override {
        add_subsystem(kTransient, cs::subsystem_desc<Src>("a"));
        add_subsystem(kTransient, cs::subsystem_desc<Axpb>("b", 2.0f, 1.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("c", 1.0f, 0.5f));
        connect("a.y", "b.x");
        connect("b.y", "c.x");
    }
};

/// p (transient Src) read by t (transient) and n (plain): p.y has a non-transient registration.
class SharedRead : public cs::System {
public:
    void setup() override {
        add_subsystem(kTransient, cs::subsystem_desc<Src>("p"));
        add_subsystem(kTransient, cs::subsystem_desc<Axpb>("t", 1.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("n", 1.0f, 0.0f));
        connect("p.y", "t.x");
        connect("p.y", "n.x");
    }
};

class TransPromo : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("p"));
        promote<DType::f32, 1>("p.y", "y");
    }
};

/// "tr" is transient and promotes p.y as "y"; with `repromote` the root System promotes tr.y again.
class PromoInTrans : public cs::System {
    bool repromote_;

public:
    explicit PromoInTrans(bool repromote) : repromote_{repromote} {}
    void setup() override {
        add_subsystem(kTransient, cs::subsystem_desc<TransPromo>("tr"));
        if (repromote_) promote<DType::f32, 1>("tr.y", "out");
    }
};

class NestedOuter : public cs::System {
public:
    void setup() override { add_subsystem(kTransient, cs::subsystem_desc<TransSys>("inner")); }
};

class NestedTrans : public cs::System {
public:
    void setup() override { add_subsystem(kTransient, cs::subsystem_desc<NestedOuter>("outer")); }
};

/// p.x is a free input inside the transient subtree: a transient root, initialized (logged) at each allocation.
class TransAxpbSys : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("p", 2.0f, 1.0f, 7.0f, std::string{"tr.p"})); }
};

class TransFreeInput : public cs::System {
public:
    void setup() override {
        add_subsystem(kTransient, cs::subsystem_desc<TransAxpbSys>("tr"));
        add_subsystem(cs::subsystem_desc<Axpb>("sink", 1.0f, 0.0f));
        connect("tr.p.y", "sink.x");
    }
};

// ---- run-N ----------------------------------------------------------------------------------
/// One Axpb in a plain System: y = 2 x, x a free input ("a.x"). Its parent adds it with run_n.
class OpAxpb : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("a", 2.0f, 0.0f)); }
};

/// A root holding one OpAxpb added with run_n{n}, named "op"; free input "op.a.x", output "op.a.y". A plain
/// sink reads op.a.y: its eternal answer keeps the output alive after the last run; op.a.x is freed then.
class RunNTop : public cs::System {
    std::size_t n_;

public:
    explicit RunNTop(std::size_t n) : n_{n} {}
    void setup() override {
        add_subsystem(cs::property_list{cs::property::run_n{n_}}, cs::subsystem_desc<OpAxpb>("op"));
        add_subsystem(cs::subsystem_desc<Axpb>("sink", 1.0f, 0.0f));
        connect("op.a.y", "sink.x");
    }
};

/// The transient run-N subtree: prod (free input x = 4, logged) -> mid -> cons, y = 2 x + 3 end to end.
class OpTrans : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("prod", 1.0f, 0.0f, 4.0f, std::string{"op.prod"}));
        add_subsystem(cs::subsystem_desc<Axpb>("mid", 2.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("cons", 1.0f, 3.0f));
        connect("prod.y", "mid.x");
        connect("mid.y", "cons.x");
    }
};

/// op (OpTrans, run_n{n}) -> sink (y = x + 0.5): the consumer's output feeds the sink outside (an eternal partner), so it is eternal.
class RunNTransTop : public cs::System {
    std::size_t n_;

public:
    explicit RunNTransTop(std::size_t n) : n_{n} {}
    void setup() override {
        add_subsystem(cs::property_list{cs::property::run_n{n_}}, cs::subsystem_desc<OpTrans>("op"));
        add_subsystem(cs::subsystem_desc<Axpb>("sink", 1.0f, 0.5f));
        connect("op.cons.y", "sink.x");
    }
};

// ---- run policy fixtures: a System that logs its tag each time its owner runs it ------------------
std::vector<std::string> g_runs;

std::size_t runs_of(const std::string& tag) {
    return static_cast<std::size_t>(std::count(g_runs.begin(), g_runs.end(), tag));
}

/// A plain System base for roots whose tests re-arm children (r04: a System has no prepare_forward; the test calls
/// model.rearm_subsystem between runs, see rearm_next).
class RearmSys : public cs::System {};

/// Re-arms `name` between runs (r04: Model::rearm_subsystem; it used to be a System's prepare_forward at the next forward).
inline void rearm_next(cs::Model& model, const std::string& name) { model.rearm_subsystem(name); }

/// A leaf that logs its tag each time it runs: the run counter of a RunCounted System (a System itself has no forward).
class Tick : public cs::Component<Tick> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.y[i] = 0.0f; }

    Inputs  inputs;
    Outputs outputs;

private:
    std::string tag_;

public:
    explicit Tick(std::string tag) : tag_{std::move(tag)} {}
    void declare() override { declare_output(outputs.y, "y", 1); }
    void prepare_forward() override { g_runs.push_back(tag_); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};

class RunCounted : public cs::System {
    std::string                      tag_;
    std::function<void(cs::System&)> kids_;  ///< adds children inside this System's own setup()

public:
    explicit RunCounted(std::string tag, std::function<void(cs::System&)> kids = {})
        : tag_{std::move(tag)}, kids_{std::move(kids)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Tick>("tick", tag_));  // first child: runs once each time this System is entered
        if (kids_) kids_(*this);
    }
};

/// A root whose structure is given by a callable run inside its setup().
class PolicyTop : public RearmSys {
    std::function<void(cs::System&)> build_;

public:
    explicit PolicyTop(std::function<void(cs::System&)> build) : build_{std::move(build)} {}
    void setup() override { build_(*this); }
};

/// Everything written to std::cerr while it lives.
class CerrCapture {
    std::ostringstream ss_;
    std::streambuf*    old_;

public:
    CerrCapture() : old_{std::cerr.rdbuf(ss_.rdbuf())} {}
    ~CerrCapture() { std::cerr.rdbuf(old_); }
    CerrCapture(const CerrCapture&)            = delete;
    CerrCapture& operator=(const CerrCapture&) = delete;
    std::string str() const { return ss_.str(); }
};

// ---- lifetime fixtures ---------------------------------------------------------------------
/// p (transient Src) -> t (transient Axpb, y = 2 x + 1) -> sink (plain Axpb). p.y reaches t through its promotion "py",
/// which the (eternal) root holds; t.y is held by the sink.
class TransChain : public cs::System {
public:
    void setup() override {
        add_subsystem(kTransient, cs::subsystem_desc<Src>("p"));
        add_subsystem(kTransient, cs::subsystem_desc<Axpb>("t", 2.0f, 1.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("sink", 1.0f, 0.0f));
        promote<DType::f32, 1>("p.y", "py");
        connect("py", "t.x");
        connect("t.y", "sink.x");
    }
};

/// A transient System holding one Axpb "c" added with run_n{5} (a clamp to transient, with a warning).
class TransRunNKid : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::property_list{cs::property::run_n{5}}, cs::subsystem_desc<Axpb>("c", 2.0f, 0.0f, 4.0f, std::string{"tr.c"}));
    }
};

/// p (Src, do_once) -> r (plain Axpb, y = 2 x), or -> t (transient Axpb).
class DoOnceProducer : public cs::System {
    bool transient_reader_;

public:
    explicit DoOnceProducer(bool transient_reader) : transient_reader_{transient_reader} {}
    void setup() override {
        add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<Src>("p"));
        if (transient_reader_)
            add_subsystem(kTransient, cs::subsystem_desc<Copy1>("t"));  // x unspecified: see the do_once/transient-reader case
        else
            add_subsystem(cs::subsystem_desc<Axpb>("r", 2.0f, 0.0f));
        connect("p.y", transient_reader_ ? "t.x" : "r.x");
    }
};

/// Two plain writers tie_outputs'd into "t" and read by rd, all inside a transient System; rd's output feeds an
/// eternal sink outside.
class TransTieSys : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("a", true));
        add_subsystem(cs::subsystem_desc<Src>("b", true));
        add_subsystem(cs::subsystem_desc<Axpb>("rd", 1.0f, 0.0f));
        tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(kLen), cs::TieOutputsPolicy::allow_plain);
        connect("t", "rd.x");
    }
};
class TransTieOutTop : public cs::System {
public:
    void setup() override {
        add_subsystem(kTransient, cs::subsystem_desc<TransTieSys>("tr"));
        add_subsystem(cs::subsystem_desc<Axpb>("sink", 1.0f, 0.0f));
        connect("tr.rd.y", "sink.x");
    }
};

/// Two tied members, only `a` with an init of its own (100); the tie declares x = 3 afterwards, which is the last declaration and wins.
class TieInit : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("a", tied_x, 2.0f, 0.0f, 100.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("b", tied_x, 3.0f, 0.0f));
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(kLen));
        init_input("x", cs::init::fill(3.0f));
    }
};

/// Both tied members carry an init of their own (100, 200). With `tie_level`, the System also declares x = 7 after the tie
/// (the last declaration: it wins); without it the later merged member (b, 200) wins. The last declaration wins, section 2.
class TieInitTwo : public cs::System {
    bool tie_level_;

public:
    explicit TieInitTwo(bool tie_level) : tie_level_{tie_level} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("a", tied_x, 2.0f, 0.0f, 100.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("b", tied_x, 3.0f, 0.0f, 200.0f));
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(kLen));
        if (tie_level_) init_input("x", cs::init::fill(7.0f));
    }
};

/// A transient System holding a free-input Axpb whose x is marked as a design variable by the root.
class MarkedHolder : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("a", 2.0f, 1.0f)); }
};
class MarkedInTransient : public cs::System {
public:
    void setup() override {
        add_subsystem(kTransient, cs::subsystem_desc<MarkedHolder>("tr"));
        add_design_var("tr.a.x");
    }
};

/// A root promotion of a transient child's output.
class PromoteFromTransient : public cs::System {
public:
    void setup() override {
        add_subsystem(kTransient, cs::subsystem_desc<Src>("p"));
        promote<DType::f32, 1>("p.y", "y");
    }
};

/// p (Axpb, do_once) -> r (plain Axpb): p.x is a free input with a logged init of g_p_x (the System declares it, and a
/// rebuilt p gets it again by replay); p.x and p.y are rebuilt after a rearm.
class DoOnceInit : public RearmSys {
public:
    void setup() override {
        add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<Axpb>("p", 1.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("r", 2.0f, 0.0f));
        connect("p.y", "r.x");
        init_input("p.x", logging_fill_from("p", &g_p_x));
    }
};

std::vector<float> filled(float v) { return std::vector<float>(kLen, v); }

/// Set the initial value of the free input at `path` to x = v (all elements); the owner applies it at its next run.
void set_x(cs::Model& model, const char* path, float v) {
    model.update_input(path, cs::init::from_host(filled(v)));
}

// ---- structural errors ----------------------------------------------------------------------
class ThreeAxpb : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("a", 1.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("b", 1.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<Axpb>("c", 1.0f, 0.0f));
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
        tie_inputs<DType::f32, 1>({"c.x", "b.x"}, "t", cs::shape(kLen));  // c.x is already connected
    }
};

class ConnectFromInput : public ThreeAxpb {
public:
    void setup() override {
        ThreeAxpb::setup();
        connect("b.x", "c.x");  // an input is never a source
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
        add_subsystem(cs::subsystem_desc<Src>("src"));
        // Placed: the parent's own device is an add_subsystem argument; unplaced: no device at all.
        const auto kid = cs::subsystem_desc<Axpb>("kid", 2.0f, 1.0f);  // a Component
        // A plain System holding two Components (the helper form is gone: system_from).
        const auto grp = cs::system_from("grp", cs::subsystem_desc<Axpb>("g1", 1.0f, 2.0f), cs::subsystem_desc<Axpb>("g2", 3.0f, 0.0f));
        if (placed_) {
            add_subsystem(dev_, kid);
            add_subsystem(dev_, grp);
        } else {
            add_subsystem(kid);
            add_subsystem(grp);
        }
        connect("src.y", "kid.x");
        connect("kid.y", "grp.g1.x");
        connect("grp.g1.y", "grp.g2.x");
    }
};

// ---- rebuilt consumer of a deleted do_once producer ----------------------------------------------
/// p (Src, do_once) -> r (plain Axpb, y = 2 x); the root promotes p.y as "y", so the variable outlives p.
class ConsumerRebuild : public RearmSys {
public:
    void setup() override {
        add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<Src>("p"));
        add_subsystem(cs::subsystem_desc<Axpb>("r", 2.0f, 0.0f));
        connect("p.y", "r.x");
        promote<DType::f32, 1>("p.y", "y");
    }
};

// ---- depends_on: an ordering with no data between the two children ----------------------------------
std::atomic<float> g_side{0.0f};  // the side channel: A writes it, B reads it; no variable joins them

/// A: a host task that sleeps, then writes the side channel. Its own output only says it ran.

class SlowWriter : public cs::HostComponent<SlowWriter> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> done;
    };
    static void compute_forward(const Inputs&, const Outputs& out) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        g_side.store(42.0f);
        out.done[std::size_t{0}] = 1.0f;
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.done, "done", 1); }
};

/// B: a host task that copies the side channel into its output.

class SideReader : public cs::HostComponent<SideReader> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> got;
    };
    static void compute_forward(const Inputs&, const Outputs& out) { out.got[std::size_t{0}] = g_side.load(); }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.got, "got", 1); }
};

/// a (SlowWriter) then b (SideReader); `dep`: b depends_on a.
class DependsTop : public cs::System {
    bool dep_;

public:
    explicit DependsTop(bool dep) : dep_{dep} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<SlowWriter>("a"));
        add_subsystem(cs::subsystem_desc<SideReader>("b"));
        if (dep_) depends_on("b", "a");
    }
};

// ---- placement across two devices ---------------------------------------------------------------------
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

/// An inner System whose author places its child on device `x` (a recommendation: the outer device wins).
class PlacedInner : public cs::System {
    sycl::device x_;

public:
    explicit PlacedInner(const sycl::device& x) : x_{x} {}
    void setup() override { add_subsystem(x_, cs::subsystem_desc<Axpb>("k", 1.0f, 0.0f)); }
};

/// The parent adds the inner System on device `y`.
class DevTop : public RearmSys {
    sycl::device x_, y_;

public:
    DevTop(const sycl::device& x, const sycl::device& y) : x_{x}, y_{y} {}
    void setup() override { add_subsystem(y_, cs::subsystem_desc<PlacedInner>("in", x_)); }
};

// ---- Sellar ---------------------------------------------------------------------------------
constexpr float kSellarD1Y2 = 99.0f;  // d1's own init: overridden
constexpr float kSellarY2   = 1.5f;   // the MDA System's init_input (depth 1): the last word

class D1 : public cs::Component<D1> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> z;
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> y2;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y1;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        out.y1[i] = in.z[std::size_t{0}] * in.z[std::size_t{0}] + in.z[std::size_t{1}] + in.x[std::size_t{0}] -
                    0.2f * in.y2[std::size_t{0}];
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.z, "z", cs::unspecified_shape<1>());  // z and x are tie members: the tie alias owns their shape
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_input(inputs.y2, "y2", 1);
        declare_output(outputs.y1, "y1", 1);
        init_input("y2", cs::init::fill(kSellarD1Y2));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.y1.shape()));
    }
};

class D2 : public cs::Component<D2> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> z;
        cs::Input<DType::f32, 1> y1_in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y2;
        cs::Output<DType::f32, 1> yy;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        const float y1 = in.y1_in[std::size_t{0}];
        const float y2 = sycl::sqrt(y1) + in.z[std::size_t{0}] + in.z[std::size_t{1}];
        out.y2[i]              = y2;
        out.yy[std::size_t{0}] = y1;
        out.yy[std::size_t{1}] = y2;
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.z, "z", cs::unspecified_shape<1>());
        declare_input(inputs.y1_in, "y1_in", cs::unspecified_shape<1>());
        declare_output(outputs.y2, "y2", 1);
        declare_output(outputs.yy, "yy", 2);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.y2.shape()));
    }
};

class Obj : public cs::Component<Obj> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> z;
        cs::Input<DType::f32, 1> y1;
        cs::Input<DType::f32, 1> y2;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> f;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        const float x = in.x[std::size_t{0}];
        out.f[i]      = x * x + in.z[std::size_t{1}] + in.y1[std::size_t{0}] + sycl::exp(-in.y2[std::size_t{0}]);
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_input(inputs.z, "z", cs::unspecified_shape<1>());
        declare_input(inputs.y1, "y1", cs::unspecified_shape<1>());
        declare_input(inputs.y2, "y2", cs::unspecified_shape<1>());
        declare_output(outputs.f, "f", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.f.shape()));
    }
};

class G1 : public cs::Component<G1> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> y1;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> g;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.g[i] = 3.16f - in.y1[std::size_t{0}]; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.y1, "y1", cs::unspecified_shape<1>());
        declare_output(outputs.g, "g", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.g.shape()));
    }
};

class G2 : public cs::Component<G2> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> y2;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> g;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.g[i] = in.y2[std::size_t{0}] - 24.0f; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.y2, "y2", cs::unspecified_shape<1>());
        declare_output(outputs.g, "g", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.g.shape()));
    }
};

class SellarMDA : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<D1>("d1"));
        add_subsystem(cs::subsystem_desc<D2>("d2"));
        connect("d1.y1", "d2.y1_in");
        promote<DType::f32, 1>("d1.z", "z1");  // z and x are tied at the top (tie members are never aliases with a shape)
        promote<DType::f32, 1>("d2.z", "z2");
        promote<DType::f32, 1>("d1.x", "x");
        promote<DType::f32, 1>("d1.y1", "y1");
        promote<DType::f32, 1>("d2.y2", "y2");
        promote<DType::f32, 1>("d2.yy", "yy");
        init_input("d1.y2", cs::init::fill(kSellarY2));  // beats d1's own (depth 2)
    }
};

class SellarTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<SellarMDA>("mda"));
        add_subsystem(cs::subsystem_desc<Obj>("obj"));
        add_subsystem(cs::subsystem_desc<G1>("g1"));
        add_subsystem(cs::subsystem_desc<G2>("g2"));

        tie_inputs<DType::f32, 1>({"mda.x", "obj.x"}, "x", cs::shape(1));
        tie_inputs<DType::f32, 1>({"mda.z1", "mda.z2", "obj.z"}, "z", cs::shape(2));  // members through promotions
        connect("mda.y1", "obj.y1");
        connect("mda.y2", "obj.y2");  // it was element 1 of mda.yy through a Slice; slices are not ported
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
        init_input("z", cs::init::from_host(std::vector<float>{5.0f, 2.0f}));
        init_input("x", cs::init::fill(1.0f));
    }
};

/// The one value of a rank-1 output.
float scalar_of(cs::Model& model, const char* path) { return to_vector(cst::synced_output<DType::f32, 1>(model, path)).at(0); }

/// The record of the mark whose instance's variable is `meta`, or null. (A Mark holds its instance weakly: design review,
/// section 1; the variable it names is the instance's `meta`.)
const cs::MarkRecord* mark_on(std::span<const cs::Mark> marks, const std::shared_ptr<cs::detail::VarMeta>& meta) {
    for (const auto& m : marks) {
        const auto inst = m.inst.lock();
        if (inst && inst->meta == meta) return &m.rec;
    }
    return nullptr;
}

}  // namespace tsys

// =============================================================================================
// 1. connect
// =============================================================================================
TEST_CASE("connect: identity, producer to consumer", "[system][connect]") {
    // (The Slice sections, connect(..., Slice) of a rank-1 and a rank-2 subarray and their composition, are deleted:
    // slices are dead code until their redesign.)
    cs::Model model = cs::create_model<tsys::ConnectSys>(default_config());
    model.setup();
    auto y = wait_and_view<DType::f32, 1>(model, "cons.y");
    CHECK(all_close(to_vector(y), tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.0f)));  // 2 (i + 1) + 1
    CHECK(cs::detail::ModelAccess::instance(model, "prod.y")->meta == cs::detail::ModelAccess::instance(model, "cons.x")->meta);  // one variable
}

// =============================================================================================
// 2. promote
// =============================================================================================
TEST_CASE("promote: a two-level chain reads the same variable", "[system][promote]") {
    cs::Model model = cs::create_model<tsys::PromoteTop>(default_config());
    model.setup();
    model.run();
    model.wait();

    const auto want = tsys::iota_plus(1.0f, tsys::kLen);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "mid.child.y")) == want);  // the full path
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "mid.y")) == want);        // promoted once
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "y")) == want);            // promoted twice
    CHECK(all_close(to_vector(cst::synced_output<DType::f32, 1>(model, "c.y")), tsys::iota_plus(1.0f, tsys::kLen, 3.0f)));

    auto sys = model.get_subsystem("");
    const auto var = cs::detail::ModelAccess::instance(model, "mid.child.y")->meta;
    CHECK(cs::detail::ModelAccess::instance(model, "y")->meta == var);
    CHECK(cs::detail::ModelAccess::instance(model, "mid.y")->meta == var);
    CHECK(cs::detail::ModelAccess::instance(model, "c.x")->meta == var);  // connected from the alias
}

// =============================================================================================
// 3. ties
// =============================================================================================
TEST_CASE("ties: tie_inputs gives one variable; a tie of a tie gives three members and one variable",
          "[system][tie]") {
    std::vector<float> x(tsys::kLen);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i) * 0.5f + 1.0f;

    SECTION("one tie: update_input on the alias is seen by both") {
        cs::Model model = cs::create_model<tsys::TieInSys>(default_config());
        model.setup();
        CHECK(cs::detail::ModelAccess::instance(model, "a.x")->meta == cs::detail::ModelAccess::instance(model, "b.x")->meta);
        CHECK(cs::detail::ModelAccess::instance(model, "x")->meta == cs::detail::ModelAccess::instance(model, "a.x")->meta);
        model.update_input("x", cs::init::from_host(x));
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "a.x")) == x);
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "b.x")) == x);
        std::vector<float> want2(x), want3(x);
        for (auto& v : want2) v *= 2.0f;
        for (auto& v : want3) v *= 3.0f;
        CHECK(all_close(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")), want2));
        CHECK(all_close(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")), want3));
    }

    SECTION("a tie of a tie alias across two levels") {
        cs::Model model = cs::create_model<tsys::TieInTop>(default_config());
        model.setup();
        auto sys = model.get_subsystem("");
        const auto var = cs::detail::ModelAccess::instance(model, "tt")->meta;
        CHECK(cs::detail::ModelAccess::instance(model, "in.a.x")->meta == var);
        CHECK(cs::detail::ModelAccess::instance(model, "in.b.x")->meta == var);
        CHECK(cs::detail::ModelAccess::instance(model, "c.x")->meta == var);
        CHECK(cs::detail::ModelAccess::instance(model, "in.ax")->meta == var);
        CHECK(cs::detail::ModelAccess::instance(model, "in.bx")->meta == var);
        model.update_input("tt", cs::init::from_host(x));
        model.run();
        model.wait();
        for (const char* path : {"in.a.x", "in.b.x", "c.x", "in.ax", "in.bx"})
            CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, path)) == x);
        std::vector<float> want4(x);
        for (auto& v : want4) v *= 4.0f;
        CHECK(all_close(to_vector(cst::synced_output<DType::f32, 1>(model, "c.y")), want4));
    }
}

TEST_CASE("ties: tie_outputs", "[system][tie]") {
    SECTION("require_atomic on atomic outputs: both writers accumulate into the one variable") {
        cs::Model model = cs::create_model<tsys::TieOutAtomic>(default_config());
        model.setup();
        CHECK(cs::detail::ModelAccess::instance(model, "a.acc")->meta == cs::detail::ModelAccess::instance(model, "b.acc")->meta);
        auto sum = wait_and_view<DType::f32, 1>(model, "sum");
        CHECK(to_vector(sum) == std::vector<float>(tsys::kLen, 3.0f));  // 1 + 2
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "c.y")) == std::vector<float>(tsys::kLen, 3.0f));
        // The zeroing producer is declared first and both Adders depend_on it, so a second forward gives the same sum.
        auto again = wait_and_view<DType::f32, 1>(model, "sum");
        CHECK(to_vector(again) == std::vector<float>(tsys::kLen, 3.0f));
    }

    SECTION("plain outputs without allow_plain throw") {
        tsys::expect_setup_throws<tsys::TieOutPlain>("not atomic");
        cs::Model model = cs::create_model<tsys::TieOutPlain>(default_config(), false);
        tsys::expect_throws([&] { model.setup(); }, "require_atomic");
    }

    SECTION("plain outputs with allow_plain are accepted") {
        cs::Model model = cs::create_model<tsys::TieOutPlain>(default_config(), true);
        model.setup();
        CHECK(cs::detail::ModelAccess::instance(model, "a.y")->meta == cs::detail::ModelAccess::instance(model, "b.y")->meta);
        CHECK(to_vector(wait_and_view<DType::f32, 1>(model, "t")) == tsys::iota_plus(1.0f, tsys::kLen));
    }
}

// =============================================================================================
// 4. marks
// =============================================================================================
TEST_CASE("marks: recorded on the Model with paths and bounds; the last of a kind replaces", "[system][marks]") {
    cs::Model model = cs::create_model<tsys::MarksSys>(default_config());
    model.setup();
    auto sys = model.get_subsystem("");
    // A Mark is {path, inst, rec}; the variable it names is the instance's `meta` (design review, section 1).
    const auto var = [&](const char* path) { return cs::detail::ModelAccess::instance(model, path)->meta; };

    const auto dvs = model.design_vars();
    const auto cons = model.constraints();
    const auto objs = model.objectives();
    CHECK(dvs.size() == 2);   // a.x (once, replaced) and obj.x
    CHECK(cons.size() == 2);  // a.y (once, replaced) and con.y
    CHECK(objs.size() == 1);  // obj.f

    SECTION("design vars") {
        const auto* ax = tsys::mark_on(dvs, var("a.x"));
        REQUIRE(ax != nullptr);
        REQUIRE(ax->bounds.lower.has_value());
        REQUIRE(ax->bounds.upper.has_value());
        CHECK(*ax->bounds.lower == std::vector<double>{-1.0});  // the second mark won
        CHECK(*ax->bounds.upper == std::vector<double>{20.0});
        const auto* ox = tsys::mark_on(dvs, var("obj.x"));
        REQUIRE(ox != nullptr);
        CHECK(!ox->bounds.lower.has_value());
        CHECK(!ox->bounds.upper.has_value());
    }

    SECTION("constraints") {
        const auto* ay = tsys::mark_on(cons, var("a.y"));
        REQUIRE(ay != nullptr);
        REQUIRE(ay->bounds.upper.has_value());
        CHECK(*ay->bounds.upper == std::vector<double>{7.0});  // 5.0 was replaced
        CHECK(!ay->bounds.lower.has_value());
        // con.y's variable is its own output (not src.y's).
        const auto* cy = tsys::mark_on(cons, var("con.y"));
        REQUIRE(cy != nullptr);
        REQUIRE(cy->bounds.lower.has_value());
        CHECK(*cy->bounds.lower == std::vector<double>{1.0});
        CHECK(!cy->bounds.upper.has_value());
    }

    SECTION("the objective") {
        const auto* f = tsys::mark_on(objs, var("obj.f"));
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
TEST_CASE("init: the last declaration wins (System init_input beats the Component's, children below owners)",
          "[system][init]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::TopB>(default_config());
    model.setup();

    // c1 declared 99; the owner System `inner` declared 1.5 afterwards (its setup runs after its children's) and wins.
    // The init is applied by the owner (c1) in the first forward.
    // c2.x is connected, so nothing is declared on it and it is not initialized.
    model.run();
    model.wait();
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "inner.c1.x")) == std::vector<float>(tsys::kLen, tsys::kSysInit));
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "c3.y")) == std::vector<float>(tsys::kLen, tsys::kSysInit));
}

TEST_CASE("init: init_input on an eternal free input between forwards", "[system][init]") {
    cs::Model model = cs::create_model<tsys::OneAxpb>(default_config());
    model.setup();
    // No init declared: zeros, so y = 2 * 0 + 1.
    model.run();
    model.wait();
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(1.0f));
    tsys::set_x(model, "a.x", 1.0f);
    model.run();
    model.wait();
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(3.0f));
    tsys::set_x(model, "a.x", 4.0f);
    model.run();
    model.wait();
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(9.0f));
}

TEST_CASE("init: an eternal free input is initialized by its owner on the first forward only", "[system][init]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::OneAxpbLog>(default_config());
    model.setup();
    model.finalize();  // r04: views are available after finalize
    CHECK(tsys::count_of("a") == 0);  // nothing is allocated or initialized at setup
    // Design review 11.4: a view can be made at any time. It materializes a.x like any holder, so the init runs now (once)
    // and the view shows its values after sync.
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "a.x")) == tsys::filled(5.0f));
    CHECK(tsys::count_of("a") == 1);
    model.run();
    model.run();
    model.run();
    model.wait();
    CHECK(tsys::count_of("a") == 1);  // allocated once, initialized once, never released
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "a.x")) == tsys::filled(5.0f));
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(11.0f));  // 2 * 5 + 1
}

TEST_CASE("init: Model::update_input before the first forward beats the declared init", "[system][init]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::OneAxpbLog>(default_config());
    model.setup();
    tsys::set_x(model, "a.x", 3.0f);  // the owner applies it at its allocation, in place of the declared init
    model.run();
    model.wait();
    CHECK(tsys::count_of("a") == 0);  // the declared (logging) init never ran
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(7.0f));  // 2 * 3 + 1
}

// =============================================================================================
// 6. transient (registration metadata, r01 §4.4)
// =============================================================================================
namespace {
std::vector<float> want_cons() { return tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 3.0f); }  // 2 (i + 1) + 3
std::vector<float> want_sink() {
    auto v = want_cons();
    for (auto& e : v) e += 0.5f;
    return v;
}
std::size_t pool_in_use(cs::Model& m) { return cst::pool_in_use(m); }
}  // namespace

TEST_CASE("transient: deleted after its turn, kept outputs, back-to-back forwards", "[system][transient]") {
    cs::Model model = cs::create_model<tsys::TopC>(cst::pooled_config());
    model.setup();
    CHECK(cs::detail::ModelAccess::in_scope(model, "tr"));  // built at setup, deleted after its first turn

    // The pool's bytes in use are the sum of the live containers' capacities rounded up to whole nodes. After the first
    // forward the transient's prod.y and mid.y are gone; the eternal cons.y (= sink.x) and sink.y stay: two nodes of 8 floats.
    // (plus the launch arrays of the one eternal kernel component, the sink: argument, pointer and geometry arrays, and its
    // two scalar free inputs a and b: a node each)
    constexpr std::size_t kEternal = 4 * cst::pool_node_bytes + cst::solo_launch_bytes(4);
    SECTION("host access to a deleted subsystem's variable throws") {
        model.run();
        model.wait();
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "tr"));  // the transient is deleted after its turn
        tsys::expect_throws([&] { (void)cst::synced_output<DType::f32, 1>(model, "tr.mid.y"); }, "no holder of this variable has data");
        tsys::expect_throws([&] { (void)cst::synced_output<DType::f32, 1>(model, "tr.prod.y"); }, "no holder of this variable has data");
    }

    SECTION("an output connected outside the subtree stays eternal") {
        model.run();
        model.wait();
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "tr"));  // the intermediates went with the transient
        // tr.cons.y's variable is held by the sink's port outside: it is still readable
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "sink.x")) == want_cons());
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sink.y")) == want_sink());
    }

    SECTION("a second forward rebuilds it and gives the same values") {
        CHECK(pool_in_use(model) == 0);  // nothing at setup
        model.run();
        model.wait();
        const std::size_t after1 = pool_in_use(model);
        CHECK(after1 == kEternal);
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "sink.x")) == want_cons());
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sink.y")) == want_sink());
        CHECK(pool_in_use(model) == after1);
    }

    SECTION("three forwards enqueued back to back, then one wait") {
        model.run();
        model.wait();
        const std::size_t after1 = pool_in_use(model);
        model.run();
        model.run();
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "sink.x")) == want_cons());
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sink.y")) == want_sink());
        CHECK(pool_in_use(model) == after1);
    }
}

TEST_CASE("transient: sequential Components A then B are deleted after their turns", "[system][transient]") {
    cs::Model model = cs::create_model<tsys::TransComps>(cst::pooled_config());
    model.setup();

    model.run();
    model.wait();
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "a"));  // A and B are transient: deleted after their turns, A's output with them
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "b"));
    tsys::expect_throws([&] { (void)cst::synced_output<DType::f32, 1>(model, "a.y"); }, "no holder of this variable has data");
    // B's output is held by the plain C's port: still readable (b.y = 2 (i + 1) + 1)
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "c.x")) == tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.0f));
    // c = 1 * (2 * (i + 1) + 1) + 0.5
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "c.y")) == tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.5f));
}

TEST_CASE("transient: a variable read by a transient and a non-transient Component stays readable", "[system][transient]") {
    cs::Model model = cs::create_model<tsys::SharedRead>(cst::pooled_config());
    model.setup();
    model.run();
    model.wait();
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));  // the transient producer is deleted; the plain reader's port holds p.y
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "n.x")) == tsys::iota_plus(1.0f, tsys::kLen));
}

TEST_CASE("transient: promotion inside a transient stays transient; re-promoted above it is eternal",
          "[system][transient]") {
    SECTION("promoted inside the transient subsystem only") {
        cs::Model model = cs::create_model<tsys::PromoInTrans>(cst::pooled_config(), false);
        model.setup();
        model.run();
        model.wait();
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "tr"));
        tsys::expect_throws([&] { (void)cst::synced_output<DType::f32, 1>(model, "tr.y"); }, "no holder of this variable has data");
    }
    SECTION("re-promoted at a non-transient level") {
        cs::Model model = cs::create_model<tsys::PromoInTrans>(cst::pooled_config(), true);
        model.setup();
        model.run();
        model.wait();
        // the root System's promotion holds the variable: eternal
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "out")) == tsys::iota_plus(1.0f, tsys::kLen));
    }
}

TEST_CASE("transient: nested transients, the outer one is named", "[system][transient]") {
    cs::Model model = cs::create_model<tsys::NestedTrans>(cst::pooled_config());
    model.setup();
    const std::size_t base = pool_in_use(model);
    model.run();
    model.wait();
    CHECK(pool_in_use(model) == base + cst::solo_launch_bytes(3));
    tsys::expect_throws([&] { (void)cst::synced_output<DType::f32, 1>(model, "outer.inner.cons.y"); }, "no holder of this variable has data");
}

TEST_CASE("transient: a free input inside a transient subtree is re-initialized each forward", "[system][transient]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::TransFreeInput>(cst::pooled_config());
    model.setup();
    CHECK(tsys::count_of("tr.p") == 0);  // a transient root has no storage at setup
    model.run();
    model.run();
    model.run();
    model.wait();
    CHECK(tsys::count_of("tr.p") == 3);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sink.y")) == tsys::filled(15.0f));  // 2 * 7 + 1
    tsys::expect_throws([&] { (void)cst::synced_input<DType::f32, 1>(model, "tr.p.x"); }, "no holder of this variable has data");
}

// =============================================================================================
// 6b. run-N (property::run_n on add_subsystem; do-once is n = 1)
// =============================================================================================
TEST_CASE("run-N: n = 1 runs once; once its child is out of scope, an update_input on it is ignored", "[system][runn]") {
    cs::Model model = cs::create_model<tsys::RunNTop>(default_config(), std::size_t{1});
    model.setup();

    tsys::set_x(model, "op.a.x", 1.0f);
    model.run();
    model.wait();
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "sink.x")) == tsys::filled(2.0f));  // op.a.y's variable
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "op"));  // run_n{1}: deleted after its turn

    // Nothing can apply a later init: the child is gone, and so is its free input. The output is held by the sink.
    CHECK_NOTHROW(tsys::set_x(model, "op.a.x", 5.0f));  // r04: ignored (no holder is in scope), not an error
    model.run();
    model.wait();
    tsys::expect_throws([&] { (void)cst::synced_input<DType::f32, 1>(model, "op.a.x"); }, "no holder of this variable has data");
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "sink.x")) == tsys::filled(2.0f));  // op.a.y's variable, unchanged
}

TEST_CASE("run-N: n = 3 follows the input for three forwards, then holds", "[system][runn]") {
    cs::Model model = cs::create_model<tsys::RunNTop>(default_config(), std::size_t{3});
    model.setup();

    const float xs[5] = {1.0f, 2.0f, 4.0f, 7.0f, 9.0f};
    for (int k = 0; k < 5; ++k) {
        INFO("forward " << k + 1);
        if (k < 3)
            tsys::set_x(model, "op.a.x", xs[k]);
        else  // op is out of scope after its third run: an update is ignored (r04)
            CHECK_NOTHROW(tsys::set_x(model, "op.a.x", xs[k]));
        model.run();
        model.wait();
        const float held = k < 3 ? xs[k] : xs[2];  // the 3rd value from then on
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "sink.x")) == tsys::filled(2.0f * held));  // op.a.y's variable
    }
}

TEST_CASE("run-N subtree: its variables live for the n runs and are freed after the last, unless an eternal partner keeps them",
          "[system][runn]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::RunNTransTop>(cst::pooled_config(), std::size_t{2});
    model.setup();
    const std::size_t base = tsys::pool_in_use_of(model);
    CHECK(tsys::count_of("op.prod") == 0);  // nothing is allocated at setup

    // run 1: the variables come into scope, the free input is initialized once; the consumer's output feeds the sink
    // outside (eternal), the rest stays allocated for run 2
    // The pool's bytes in use are the sum of the live containers' capacities rounded up to whole nodes (8 floats = one
    // node each). Run 1 holds five containers: prod.x (the free input), prod.y (= mid.x), mid.y (= cons.x), cons.y
    // (= sink.x) and sink.y; after the last run the subtree's three are freed and the two eternal ones stay.
    CHECK(base == 0);
    model.run();
    model.wait();
    CHECK(tsys::count_of("op.prod") == 1);
    CHECK(cs::detail::ModelAccess::in_scope(model, "op"));
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "op.mid.y")) == tsys::filled(8.0f));   // 2 * 4
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "op.cons.y")) == tsys::filled(11.0f));  // 2 * 4 + 3
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sink.y")) == tsys::filled(11.5f));
    const std::size_t after1 = tsys::pool_in_use_of(model);
    // plus the launch arrays of prod, mid, cons and sink, and the scalar inputs a and b of each of the four (a node each)
    CHECK(after1 == 13 * cst::pool_node_bytes + cst::solo_launch_bytes(4));

    // run 2 is the last: the subtree's variables are freed, the eternal one stays; the init is not applied again
    model.run();
    model.wait();
    CHECK(tsys::count_of("op.prod") == 1);
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "op"));  // deleted after its last run: mid.y went with it
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "sink.x")) == tsys::filled(11.0f));  // op.cons.y's variable, held by the sink
    const std::size_t after2 = tsys::pool_in_use_of(model);
    CHECK(after2 == 4 * cst::pool_node_bytes + cst::solo_launch_bytes(4));  // the sink keeps its launch arrays and its a, b

    // forward 3: past n, nothing runs, so nothing is allocated, initialized or released
    model.run();
    model.wait();
    CHECK(tsys::count_of("op.prod") == 1);
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "sink.x")) == tsys::filled(11.0f));  // unchanged
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sink.y")) == tsys::filled(11.5f));
    CHECK(tsys::pool_in_use_of(model) == after2);

    // a view of the deleted subsystem's variable throws
    tsys::expect_throws([&] { (void)cst::synced_output<DType::f32, 1>(model, "op.mid.y"); }, "no holder of this variable has data");
}

TEST_CASE("run-N subtree: n = 3, the roots stay for runs 1..3 and the pool drops after the third", "[system][runn]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::RunNTransTop>(cst::pooled_config(), std::size_t{3});
    model.setup();
    std::size_t held = 0, freed = 0;
    for (int k = 1; k <= 5; ++k) {
        INFO("forward " << k);
        model.run();
        model.wait();
        const std::size_t now = tsys::pool_in_use_of(model);
        if (k == 1) {
            held = now;
            // prod.x, prod.y, mid.y, cons.y, sink.y: a node each, plus four launch-array sets and the scalars a, b of the four
            CHECK(held == 13 * cst::pool_node_bytes + cst::solo_launch_bytes(4));
        }
        if (k <= 3 && k >= 1 && k < 3) CHECK(now == held);  // runs 1 and 2: everything stays allocated
        if (k == 3) {
            freed = now;
            CHECK(freed < held);
            CHECK(freed == 4 * cst::pool_node_bytes + cst::solo_launch_bytes(4));  // cons.y (= sink.x), sink.y, the sink's a, b and launch arrays
        }
        if (k > 3) CHECK(now == freed);
        CHECK(tsys::count_of("op.prod") == 1);  // allocated and initialized once, with the first run
    }
}

TEST_CASE("run-N: n = 0 throws at construction", "[system][runn]") {
    CHECK_THROWS_AS(cs::property_list{cs::property::run_n{0}}, std::invalid_argument);
    tsys::expect_throws([] { cs::property_list p{cs::property::run_n{0}}; }, "run_n needs n >= 1");
}

// =============================================================================================
// 6c. run policy as an add_subsystem property (run_n, do_once; clamping; rearm)
// =============================================================================================
namespace {
/// A model of a PolicyTop built by `build`; set up (stderr captured into `err`) and returned.
cs::Model policy_model(std::function<void(cs::System&)> build, std::string& err) {
    tsys::g_runs.clear();
    cs::Model model = cs::create_model<tsys::PolicyTop>(default_config(), std::move(build));
    tsys::CerrCapture cap;
    model.setup();
    err = cap.str();
    return model;
}
/// A System "p" (props `pp`) holding one System "c" (props `cp`).
void add_pair(cs::System& s, const cs::property_list& pp, const cs::property_list& cp) {
    s.add_subsystem(pp, cs::subsystem_desc<tsys::RunCounted>("p", std::string{"p"},
                                                             std::function<void(cs::System&)>{[cp](cs::System& p) {
                                                                 p.add_subsystem(cp, cs::subsystem_desc<tsys::RunCounted>("c", std::string{"c"}));
                                                             }}));
}
void forwards(cs::Model& model, int n) {
    for (int k = 0; k < n; ++k) model.run();
    model.wait();
}
}  // namespace

TEST_CASE("run_policy: run_n(3) runs exactly 3 forwards, then is skipped", "[system][run_policy]") {
    std::string err;
    cs::Model   model = policy_model(
        [](cs::System& s) {
            s.add_subsystem(cs::property_list{cs::property::run_n{3}}, cs::subsystem_desc<tsys::RunCounted>("a", std::string{"a"}));
        },
        err);
    CHECK(err.empty());
    for (int k = 1; k <= 5; ++k) {
        INFO("forward " << k);
        forwards(model, 1);
        CHECK(tsys::runs_of("a") == static_cast<std::size_t>(k <= 3 ? k : 3));
    }
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "a"));  // run_n{3}: deleted by its parent after its third turn
}

TEST_CASE("run_policy: do_once runs once", "[system][run_policy]") {
    std::string err;
    cs::Model   model = policy_model(
        [](cs::System& s) {
            s.add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<tsys::RunCounted>("a", std::string{"a"}));
        },
        err);
    forwards(model, 1);
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "a"));  // do_once: deleted right after its forward
    forwards(model, 3);
    CHECK(tsys::runs_of("a") == 1);
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "a"));
}

TEST_CASE("run_policy: run_n{0} throws invalid_argument", "[system][run_policy]") {
    CHECK_THROWS_AS(cs::property_list{cs::property::run_n{0}}, std::invalid_argument);
    CHECK_THROWS_AS((cs::property_list{cs::property::run_n{0}, cs::property::transient{}}), std::invalid_argument);
}

TEST_CASE("run_policy: run_n together with do_once throws", "[system][run_policy]") {
    CHECK_THROWS_AS((cs::property_list{cs::property::run_n{2}, cs::property::do_once{}}), std::invalid_argument);
    CHECK_THROWS_AS((cs::property_list{cs::property::do_once{}, cs::property::run_n{2}}), std::invalid_argument);
}

TEST_CASE("run_policy: the clamp table", "[system][run_policy]") {
    using cs::property::do_once;
    using cs::property::run_n;
    using cs::property::transient;
    std::string err;

    SECTION("an eternal child under run_n(3) runs 3 times, no warning") {
        cs::Model model = policy_model(
            [](cs::System& s) {
                add_pair(s, cs::property_list{run_n{3}}, cs::property_list{});
            },
            err);
        CHECK(err.empty());
        forwards(model, 5);
        CHECK(tsys::runs_of("p") == 3);
        CHECK(tsys::runs_of("c") == 3);
    }
    SECTION("run_n(5) under run_n(2) runs 2 times (the child goes with its parent)") {
        cs::Model model = policy_model(
            [](cs::System& s) {
                add_pair(s, cs::property_list{run_n{2}}, cs::property_list{run_n{5}});
            },
            err);
        forwards(model, 5);
        CHECK(tsys::runs_of("c") == 2);
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p.c"));
    }
    SECTION("run_n(5) under transient: both run every forward") {
        cs::Model model = policy_model(
            [](cs::System& s) {
                add_pair(s, cs::property_list{transient{}}, cs::property_list{run_n{5}});
            },
            err);
        forwards(model, 7);
        CHECK(tsys::runs_of("p") == 7);
        CHECK(tsys::runs_of("c") == 7);
    }
    SECTION("transient under run_n(3) stays transient and stops when the parent stops") {
        cs::Model model = policy_model(
            [](cs::System& s) {
                add_pair(s, cs::property_list{run_n{3}}, cs::property_list{transient{}});
            },
            err);
        CHECK(err.empty());
        forwards(model, 6);
        CHECK(tsys::runs_of("p") == 3);
        CHECK(tsys::runs_of("c") == 3);
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));
    }
}

TEST_CASE("run_policy: rearm re-enables an exhausted run_n parent and its descendants", "[system][run_policy][rearm]") {
    std::string err;
    cs::Model   model = policy_model(
        [](cs::System& s) {
            add_pair(s, cs::property_list{cs::property::run_n{2}}, cs::property_list{});
        },
        err);
    CHECK(err.empty());
    forwards(model, 4);
    CHECK(tsys::runs_of("p") == 2);
    CHECK(tsys::runs_of("c") == 2);
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));

    // r04: Model::rearm_subsystem between runs puts "p" and its subtree back in scope with the counter back to run_n(2).
    tsys::rearm_next(model, "p");
    forwards(model, 1);
    CHECK(cs::detail::ModelAccess::in_scope(model, "p"));  // rearmed, and it has taken one of its two turns
    CHECK(cs::detail::ModelAccess::in_scope(model, "p.c"));
    forwards(model, 3);
    CHECK(tsys::runs_of("p") == 4);
    CHECK(tsys::runs_of("c") == 4);
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));
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
    SECTION("connect from an input") {
        tsys::expect_setup_throws<tsys::ConnectFromInput>("cannot be a source");
    }
    SECTION("an unknown path in connect names the path") {
        tsys::expect_setup_throws<tsys::ConnectUnknownEntry>("a.nope");
        tsys::expect_setup_throws<tsys::ConnectUnknownSub>("zz");
    }
    SECTION("a call outside setup(): legal from main until finalize (tests/test_p4.cpp), a logic error after") {
        cs::Model model = cs::create_model<tsys::ThreeAxpb>(default_config());
        model.setup();
        auto sys = model.get_subsystem("");
        // r04: every declaration call is legal between setup() and finalize; their own rules still apply. The connect works
        // (nothing has run), and the tie that follows it meets the ordinary rule: b.x is no longer a free input.
        CHECK_NOTHROW(sys.connect("a.y", "b.x"));
        tsys::expect_throws([&] { sys.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::shape(1)); }, "not a free input");
        CHECK_NOTHROW(sys.depends_on("b", "a"));
        model.run();  // finalize
        tsys::expect_throws([&] { sys.add_design_var("a.x"); }, "illegal after finalize");
        tsys::expect_throws([&] { sys.depends_on("b", "a"); }, "illegal after finalize");
        model.wait();
    }
    SECTION("an unknown path in view_output names the path") {
        cs::Model model = cs::create_model<tsys::ThreeAxpb>(default_config());
        model.setup();
        model.run();  // a view can be obtained only after a run (design review, section 5)
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("a.nope"); }, "a.nope");
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("zz.y"); }, "zz");
        // The kind is checked too.
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("a.x"); }, "is an input");
        tsys::expect_throws([&] { (void)model.view_input<DType::f32, 1>("a.y"); }, "is an output");
        model.wait();
    }
}

// =============================================================================================
// 8. placement
// =============================================================================================
TEST_CASE("placement: a device equal to the parent's is a no-op", "[system][placement]") {
    const sycl::device dev{sycl::default_selector_v};  // the Model's default device

    cs::Model plain = cs::create_model<tsys::TopPlace>(default_config(), dev, false);
    cs::Model placed = cs::create_model<tsys::TopPlace>(default_config(), dev, true);
    plain.setup();
    placed.setup();

    for (const char* path : {"kid.x", "kid.y", "grp.g1.x", "grp.g2.y"}) {
        INFO(path);
        for (cs::Model* m : {&plain, &placed}) {
            const auto inst = cs::detail::ModelAccess::instance(*m, path);
            CHECK(inst->device == dev);
            CHECK(inst->storage_kind == sk::kind::device);
        }
    }

    std::vector<float> want_kid = tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.0f);  // kid: 2x + 1
    std::vector<float> want_g2(tsys::kLen);
    for (std::size_t i = 0; i < want_g2.size(); ++i) want_g2[i] = 3.0f * (want_kid[i] + 2.0f);  // g1: x + 2, g2: 3x
    for (cs::Model* m : {&plain, &placed}) {
        auto kid = wait_and_view<DType::f32, 1>(*m, "kid.y");
        CHECK(all_close(to_vector(kid), want_kid));
        CHECK(all_close(to_vector(cst::synced_output<DType::f32, 1>(*m, "grp.g2.y")), want_g2));
    }
}

// =============================================================================================
// Sellar
// =============================================================================================
TEST_CASE("Sellar-shaped forward: ties of ties, promotions, marks, init last word", "[system][sellar]") {
    cs::Model model = cs::create_model<tsys::SellarTop>(default_config());
    model.setup();
    model.run();
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
    CHECK(close(tsys::scalar_of(model, "obj.f"), f));     // obj.y2 is connected from mda.y2 (it was a Slice of mda.yy)
    CHECK(close(tsys::scalar_of(model, "g1.g"), g1));
    CHECK(close(tsys::scalar_of(model, "g2.g"), g2));
    // d1's own init wrote 99; the MDA System's init wrote 1.5 afterwards.
    CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "mda.d1.y2")) == std::vector<float>{tsys::kSellarY2});

    auto sys = model.get_subsystem("");
    const auto zvar = cs::detail::ModelAccess::instance(model, "z")->meta;
    CHECK(cs::detail::ModelAccess::instance(model, "mda.d1.z")->meta == zvar);
    CHECK(cs::detail::ModelAccess::instance(model, "mda.d2.z")->meta == zvar);
    CHECK(cs::detail::ModelAccess::instance(model, "obj.z")->meta == zvar);

    // Marks ride on the variables; z's sits on the ONE shared variable.
    CHECK(model.design_vars().size() == 2);
    CHECK(tsys::mark_on(model.design_vars(), zvar) != nullptr);
    CHECK(tsys::mark_on(model.design_vars(), cs::detail::ModelAccess::instance(model, "x")->meta) != nullptr);
    CHECK(model.objectives().size() == 1);
    CHECK(tsys::mark_on(model.objectives(), cs::detail::ModelAccess::instance(model, "obj.f")->meta) != nullptr);
    CHECK(model.constraints().size() == 2);
    CHECK(tsys::mark_on(model.constraints(), cs::detail::ModelAccess::instance(model, "g1.g")->meta) != nullptr);
    CHECK(tsys::mark_on(model.constraints(), cs::detail::ModelAccess::instance(model, "g2.g")->meta) != nullptr);
}

// =============================================================================================
// 7. ties own the shape; promotions mirror live; rank-templated aliases
// =============================================================================================
namespace tsys2 {

// bins (VBins): a free u32 input n and an output b of n floats (b[i] = i + 1); a VALUE-driven output.

class VBins : public cs::Component<VBins> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> n;
        cs::Input<DType::f32, 1> ref;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> b;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.b[i] = static_cast<float>(i[0]) + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

private:
    bool                      as_is_;  // true: b's size comes from a definition (an alias), so the producer passes cs::as_is

public:
    explicit VBins(bool as_is = false) : as_is_{as_is} {}
    // This fixture tests a hook on an INPUT: the size by value goes through the free input `ref`, which the System hooks
    // (add_bins: from_value of `n`), and b copies its shape in prepare_forward. Hooks on outputs are tested in test_shapes.cpp.
    void declare() override {
        declare_input(inputs.n, "n", 1);
        declare_input(inputs.ref, "ref", cs::unspecified);
        declare_output(outputs.b, "b", cs::unspecified);
    }
    void prepare_forward() override {
        if (as_is_)
            set_shape(outputs.b, cs::as_is);
        else
            set_shape(outputs.b, shape_of(inputs.ref)[0]);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.b.shape()));
    }
};

/// Adds a VBins child and hooks its `ref` with the value of its `n`.
inline void add_bins(cs::System& s, const char* name = "v", bool as_is = false) {
    s.add_subsystem(cs::subsystem_desc<VBins>(name, as_is));
    s.shape_hook(std::string{name} + ".ref", cs::dim::from_value(std::string{name} + ".n", 0));
}

// NCopy: n[0] = v[0], the bin count as an OUTPUT (an alias shape reads an upstream output).

class NCopy : public cs::Component<NCopy> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> v;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> n;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.n[i] = in.v[i]; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.v, "v", 1);
        declare_output(outputs.n, "n", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.n.shape()));
    }
};

// TScale: y = 2 x; x has all dims unspecified (a tie member), y follows x's shape.

class TScale : public cs::Component<TScale> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i]; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_output(outputs.y, "y", cs::unspecified_shape<1>());
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }  // the output follows the input (section 10: no derived dims)
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};

// TIota: o[i] = i + 1; o has all dims unspecified (a tie_outputs member).

class TIota : public cs::Component<TIota> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> o;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.o[i] = static_cast<float>(i[0]) + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.o, "o", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.o.shape()));
    }
};

/// A root System whose setup() runs a callable.
class FnSys : public cs::System {
    std::function<void(cs::System&)> build_;

public:
    explicit FnSys(std::function<void(cs::System&)> b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

inline cs::Model make(std::function<void(cs::System&)> build, cs::model_config_desc cfg = default_config()) {
    return cs::create_model<FnSys>(std::move(cfg), std::move(build));
}

std::vector<float> one_to(std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i) + 1.0f;
    return v;
}
std::vector<float> doubled(std::vector<float> v) {
    for (auto& e : v) e *= 2.0f;
    return v;
}
std::vector<std::uint32_t> one_u32(std::uint32_t n) { return std::vector<std::uint32_t>{n}; }

/// Two TScale members tied into "x", and the transient version of the same.
class TieTwoScales : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<TScale>("a"));
        add_subsystem(cs::subsystem_desc<TScale>("b"));
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(tsys::kLen));
    }
};
class TransTieTop : public cs::System {
public:
    void setup() override { add_subsystem(tsys::kTransient, cs::subsystem_desc<TieTwoScales>("tr")); }
};

}  // namespace tsys2

TEST_CASE("alias: a promoted value-driven output shows its member's current shape", "[system][alias][promote]") {
    auto model = tsys2::make([](cs::System& s) {
        tsys2::add_bins(s);
        s.promote<DType::f32, 1>("v.b", "b");
        s.promote<DType::u32, 1>("v.n", "n");
    });
    model.setup();
    model.finalize();  // r04: views are available after finalize
    // a view can be made at any time, but this output's size is set by its producer: not known before a run
    tsys::expect_throws([&] { (void)cst::synced_output<DType::f32, 1>(model, "b"); }, "its shape is not known yet");
    for (const std::uint32_t n : {3u, 7u, 2u}) {  // the size changes between runs
        INFO("n = " << n);
        model.update_input("n", cs::init::from_host(tsys2::one_u32(n)));
        model.run();
        model.wait();
        auto v = cst::synced_output<DType::f32, 1>(model, "b");
        CHECK(v.shape()[0] == n);
        CHECK(v.size() == n);
        CHECK(to_vector(v) == tsys2::one_to(n));
    }
}

TEST_CASE("alias: views of an alias before its root's owner has run work (the shape is declared)", "[system][alias]") {
    cs::Model model = cs::create_model<tsys2::TieTwoScales>(default_config());
    model.setup();
    model.finalize();  // r04: views are available after finalize
    CHECK(cst::synced_input<DType::f32, 1>(model, "x").shape()[0] == tsys::kLen);  // design review 11.4: views any time
    CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == tsys::kLen);
    model.run();
    model.wait();
    CHECK(cst::synced_input<DType::f32, 1>(model, "x").shape()[0] == tsys::kLen);
    CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == tsys::kLen);
}

TEST_CASE("reactive shapes: a hook whose source is produced by a later child is no setup error",
          "[system][reactive]") {
    // P5: there is no source-ordering check. The hook runs before the first child that holds its target; a source that is not
    // produced yet gives whatever is there (here v.n is read before nb has written it), and the model sets up.
    // v is registered BEFORE nb, but v.n is connected to nb.n: v.ref's size is the value of v.n, which nb has not produced yet.
    auto model = tsys2::make([](cs::System& s) {
        tsys2::add_bins(s);
        s.add_subsystem(cs::subsystem_desc<tsys2::NCopy>("nb"));
        s.connect("nb.n", "v.n");
    });
    CHECK_NOTHROW(model.setup());
}

/// std::cerr captured for the lifetime of the object (rdbuf swap).
struct CerrCapture {
    std::ostringstream buf;
    std::streambuf*    old;
    CerrCapture() : old{std::cerr.rdbuf(buf.rdbuf())} {}
    ~CerrCapture() { std::cerr.rdbuf(old); }
    std::string str() const { return buf.str(); }
};

TEST_CASE("cycle warning: a consumer registered before its producer warns once, naming both ports",
          "[system][cycle]") {
    std::string text;
    {
        CerrCapture cap;
        cs::Model   model = tsys2::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("a", 2.0f, 0.0f));        // reads b.y, registered before b
            s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("b", 1.0f, 0.0f, 3.0f));  // b.y = 3
            s.connect("b.y", "a.x");
        });
        model.setup();
        text = cap.str();  // printed at setup (by the connect call), not at a forward
    }
    INFO("cerr: " << text);
    // The text is wiring.hpp warn_cycle's.
    const std::string line =
        "naga::cospan: warning: 'a.x' reads 'b.y', which runs after it: this is a cycle; without a solver the consumer sees "
        "the producer's previous value (or its initial state on the first forward)\n";
    CHECK(text.find(line) != std::string::npos);
    CHECK(text.find(line) == text.rfind(line));  // one line for the one connection
    // P4 settles what a cycle does at run time (tests/test_p4.cpp): a.x has no initializer here, so the first forward of
    // this model throws, naming the variable and the cycle; with an initializer the consumer reads it on the first forward
    // and the produced value on the second.
}

TEST_CASE("cycle warning: a correctly ordered model prints nothing", "[system][cycle]") {
    CerrCapture cap;
    cs::Model   model = tsys2::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("b", 1.0f, 0.0f, 3.0f));
        s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("a", 2.0f, 0.0f));
        s.connect("b.y", "a.x");
    });
    model.setup();
    model.run();
    model.wait();
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(6.0f));
    CHECK(cap.str().find("naga::cospan: warning") == std::string::npos);  // (the runtime's own warnings share std::cerr)
    CHECK(cap.str().find("this is a cycle") == std::string::npos);
}

TEST_CASE("alias: tie errors", "[system][alias][errors]") {
    const auto setup_throws = [](std::function<void(cs::System&)> build, const std::vector<std::string>& needles) {
        tsys::expect_throws_all(
            [&] {
                auto model = tsys2::make(build);
                model.setup();
            },
            needles);
    };
    // Shapes are reactive (design review, section 2): a shape that nothing supplies is not a setup error. It fails at the
    // first forward, when the variable is materialized (or a rule is read).
    const auto forward_throws = [](std::function<void(cs::System&)> build, const std::vector<std::string>& parts) {
        auto model = tsys2::make(build);
        model.setup();  // no error here
        std::string what = "<no exception>";
        try {
            model.run();
        } catch (const std::exception& e) {
            what = e.what();
        }
        model.wait();
        INFO("thrown: " << what);
        for (const std::string& p : parts) CHECK(what.find(p) != std::string::npos);
    };
    const auto two_scales = [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
        s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
    };
    SECTION("a tie of one member") {
        setup_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.tie_inputs<DType::f32, 1>({"a.x"}, "t", cs::shape(4));
            },
            {"a tie needs two or more members; use promote for one"});
    }
    SECTION("dtype mismatch") {
        setup_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.tie_inputs<DType::f64, 1>({"a.x", "b.x"}, "t", cs::shape(4));
            },
            {"dtype"});
    }
    SECTION("rank mismatch") {
        setup_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.tie_inputs<DType::f32, 2>({"a.x", "b.x"}, "t", cs::shape(4, 4));
            },
            {"rank"});
        setup_throws(
            [&](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("a", 1.0f, 0.0f));
                s.promote<DType::f32, 2>("a.x", "ax");
            },
            {"rank"});
    }
    SECTION("a tied member whose fixed dim contradicts the tie") {  // a member may declare dims; they are constraints
        // The wording is wiring.hpp check_compatible's (the old text was "shape mismatch: 'a.x' dimension 0 is fixed").
        setup_throws(
            [&](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("a", 1.0f, 0.0f));  // x is fixed kLen
                s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("b", tsys::tied_x, 1.0f, 0.0f));
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::shape(tsys::kLen + 1));
            },
            {"dimension 0", "the extents must match", "'a.x'"});
    }
    SECTION("an unspecified free input nothing supplies: thrown at the first forward") {
        forward_throws([&](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a")); },
                       {"its shape is not known yet"});
    }
    // DELETED: "an output alias cannot have an unspecified shape". tie_outputs now takes cs::shape() (unspecified) by default
    // and a shape_hook on the alias supplies the size (section 10); it is no longer a setup error.
    SECTION("an unspecified tie_outputs alias nothing supplies: thrown at the first forward") {
        forward_throws(
            [&](cs::System& s) {
                s.add_subsystem(cs::subsystem_desc<tsys2::TIota>("w1"));
                s.add_subsystem(cs::subsystem_desc<tsys2::TIota>("w2"));
                s.tie_outputs<DType::f32, 1>({"w1.o", "w2.o"}, "t", cs::shape(), cs::TieOutputsPolicy::allow_plain);
            },
            {"its shape is not known yet"});
    }
    SECTION("an unspecified tie_inputs alias nothing supplies: thrown at the first forward") {
        forward_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t");
            },
            {"its shape is not known yet"});
    }
    SECTION("a hook whose source is produced by a later-registered child sets up: no source-ordering check (P5)") {
        for (const bool from_value : {true, false}) {
            auto model = tsys2::make([&](cs::System& s) {
                two_scales(s);
                if (from_value)
                    s.add_subsystem(cs::subsystem_desc<tsys2::NCopy>("nb"));  // registered AFTER the tie's members
                else
                    tsys2::add_bins(s);
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t");
                if (from_value)
                    s.shape_hook("t", cs::dim::from_value("nb.n", 0));
                else
                    s.shape_hook("t", cs::dim::from_shape("v.b", 0));
            });
            CHECK_NOTHROW(model.setup());
        }
    }
    SECTION("a from_value hook naming a float variable") {
        // was: "an alias shape rule naming an input is not an output". A hook's source may be an input now; a size by value
        // must have an integer dtype.
        setup_throws(
            [&](cs::System& s) {
                tsys2::add_bins(s);
                two_scales(s);
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t");
                s.shape_hook("t", cs::dim::from_value("v.ref", 0));
            },
            {"shape_hook", "'t'", "'v.ref'", "must have an integer dtype"});
    }
}

TEST_CASE("alias: tie_inputs shapes: fixed, from_output_shape, from_output_values", "[system][alias][tie]") {
    SECTION("fixed") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(5));
        });
        model.setup();
        model.update_input("x", cs::init::from_host(tsys2::one_to(5)));
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys2::doubled(tsys2::one_to(5)));
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == tsys2::doubled(tsys2::one_to(5)));
    }
    SECTION("from_output_shape: the shape follows the upstream output between runs") {
        auto model = tsys2::make([](cs::System& s) {
            tsys2::add_bins(s);
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x");
            s.shape_hook("x", cs::dim::from_shape("v.b", 0));
        });
        model.setup();
        for (const std::uint32_t n : {3u, 6u, 4u}) {
            INFO("n = " << n);
            model.update_input("v.n", cs::init::from_host(tsys2::one_u32(n)));
            model.run();
            model.wait();
            CHECK(cst::synced_input<DType::f32, 1>(model, "x").shape()[0] == n);
            CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == n);
            CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == std::vector<float>(n, 0.0f));
        }
    }
    SECTION("from_output_values: the extents are the values of an upstream output") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tsys2::NCopy>("nb"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x");
            s.shape_hook("x", cs::dim::from_value("nb.n", 0));
        });
        model.setup();
        for (const std::uint32_t n : {4u, 9u, 1u}) {
            INFO("n = " << n);
            model.update_input("nb.v", cs::init::from_host(tsys2::one_u32(n)));
            model.update_input("x", cs::init::fill(1.5f));
            model.run();
            model.wait();
            CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == n);
            CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == std::vector<float>(n, 3.0f));
            CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == std::vector<float>(n, 3.0f));
        }
    }
}

TEST_CASE("alias: tie_outputs shapes: fixed and from_output_values", "[system][alias][tie]") {
    SECTION("fixed") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tsys2::TIota>("w1"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TIota>("w2"));
            s.add_subsystem(cs::subsystem_desc<tsys::Copy1>("c"));
            s.tie_outputs<DType::f32, 1>({"w1.o", "w2.o"}, "t", cs::shape(6), cs::TieOutputsPolicy::allow_plain);
            s.connect("t", "c.x");
        });
        model.setup();
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "t")) == tsys2::one_to(6));
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "c.y")) == tsys2::one_to(6));
    }
    SECTION("from_output_values: the size changes between runs") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tsys2::NCopy>("nb"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TIota>("w1"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TIota>("w2"));
            s.add_subsystem(cs::subsystem_desc<tsys::Copy1>("c"));
            s.tie_outputs<DType::f32, 1>({"w1.o", "w2.o"}, "t", cs::shape(), cs::TieOutputsPolicy::allow_plain);
            s.shape_hook("t", cs::dim::from_value("nb.n", 0));
            s.connect("t", "c.x");
        });
        model.setup();
        for (const std::uint32_t n : {3u, 8u, 5u}) {
            INFO("n = " << n);
            model.update_input("nb.v", cs::init::from_host(tsys2::one_u32(n)));
            model.run();
            model.wait();
            CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "t")) == tsys2::one_to(n));
            CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "c.y")) == tsys2::one_to(n));
        }
    }
}

TEST_CASE("alias: a connected tie_inputs alias joins the producer's root", "[system][alias][tie]") {
    const auto build = [](std::optional<cs::ShapeSpec> shape, bool as_is = false) {  // nullopt: tie_inputs without a shape (unspecified)
        return [shape, as_is](cs::System& s) {
            tsys2::add_bins(s, "v", as_is);
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
            if (shape)
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", *shape);
            else
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x");
            s.connect("v.b", "x");
        };
    };
    SECTION("members take the connected output's shape; the producer owns the root") {
        auto model = tsys2::make(build(std::nullopt));
        model.setup();
        CHECK(cs::detail::ModelAccess::instance(model, "a.x")->meta == cs::detail::ModelAccess::instance(model, "v.b")->meta);
        for (const std::uint32_t n : {4u, 2u}) {
            INFO("n = " << n);
            model.update_input("v.n", cs::init::from_host(tsys2::one_u32(n)));
            model.run();
            model.wait();
            CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == n);
            CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys2::doubled(tsys2::one_to(n)));
            CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == tsys2::doubled(tsys2::one_to(n)));
        }
    }
    SECTION("a mismatching fixed AliasShape errors on the first run") {
        // Section 10: the connect merges the tie's declared extent into v.b's definition; v's set_shape (from v.ref) is checked
        // against it at the forward (variable.hpp, check_setter).
        auto model = tsys2::make(build(cs::shape(5)));
        model.setup();
        model.update_input("v.n", cs::init::from_host(tsys2::one_u32(4)));
        tsys::expect_throws_all([&] { model.run(); }, {"dimension 0 is declared as 5", "takes a number only on an unspecified dimension"});
        model.wait();
    }
    SECTION("a fixed AliasShape: a producer number throws even when it matches") {
        auto model = tsys2::make(build(cs::shape(4)));
        model.setup();
        model.update_input("v.n", cs::init::from_host(tsys2::one_u32(4)));
        tsys::expect_throws_all([&] { model.run(); }, {"dimension 0 is declared as 4", "takes a number only on an unspecified dimension"});
        model.wait();
    }
    SECTION("a fixed AliasShape with a producer passing as_is keeps its size when the source's size changes") {
        auto model = tsys2::make(build(cs::shape(4), true));
        model.setup();
        model.update_input("v.n", cs::init::from_host(tsys2::one_u32(4)));
        CHECK_NOTHROW(model.run());
        model.wait();
        model.update_input("v.n", cs::init::from_host(tsys2::one_u32(6)));
        CHECK_NOTHROW(model.run());
        model.wait();
        CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == 4);
    }
    SECTION("a matching fixed AliasShape runs") {
        auto model = tsys2::make(build(cs::shape(4), true));
        model.setup();
        model.update_input("v.n", cs::init::from_host(tsys2::one_u32(4)));
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys2::doubled(tsys2::one_to(4)));
    }
    SECTION("a model init on the alias warns and is skipped") {
        auto model = tsys2::make(build(std::nullopt));
        model.setup();
        std::ostringstream captured;
        std::streambuf*    old = std::cerr.rdbuf(captured.rdbuf());
        try {
            model.update_input("x", cs::init::fill(9.0f));
        } catch (...) {
            std::cerr.rdbuf(old);
            throw;
        }
        std::cerr.rdbuf(old);
        CHECK(captured.str() ==
              "naga::cospan: warning: input 'x' is connected to an output; update_input is never applied\n");
        model.update_input("v.n", cs::init::from_host(tsys2::one_u32(3)));
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys2::doubled(tsys2::one_to(3)));  // not 9
    }
}

TEST_CASE("alias: a tie alias inside a transient subsystem is released and the pool returns to its baseline",
          "[system][alias][transient]") {
    cs::Model model = cs::create_model<tsys2::TransTieTop>(cst::pooled_config());
    model.setup();
    const std::size_t base = tsys::pool_in_use_of(model);
    for (int round = 0; round < 2; ++round) {
        INFO("forward " << round + 1);
        model.run();
        model.wait();
        auto sys = model.get_subsystem("");
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "tr"));  // the transient System is deleted after its turn
        CHECK(tsys::pool_in_use_of(model) == base + cst::solo_launch_bytes(2));
        tsys::expect_throws([&] { (void)cst::synced_input<DType::f32, 1>(model, "tr.x"); }, "no holder of this variable has data");
    }
}

// =============================================================================================
// 6d. lifetime by counting
// =============================================================================================
TEST_CASE("lifetime: a bare transient producer feeds a transient reader through a promotion, values right on run 2", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::TransChain>(cst::pooled_config());
    model.setup();
    std::size_t steady = 0;  // the pool in use after the first run: p.y (held by "py"), t.y (= sink.x) and sink.y, a node each
    for (int run = 1; run <= 3; ++run) {
        INFO("run " << run);
        model.run();
        model.wait();
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));  // transient: deleted after its turn
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "t"));
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sink.y")) == tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.0f));
        if (run == 1) {
            steady = pool_in_use(model);
            CHECK(steady == 5 * cst::pool_node_bytes + cst::solo_launch_bytes(3));  // and the sink's a, b and launch arrays
        } else {
            CHECK(pool_in_use(model) == steady);
        }
    }
}

TEST_CASE("lifetime: transient parent forces a run_n(5) child to transient", "[system][lifetime]") {
    tsys::g_inits.clear();
    std::string err;
    cs::Model   model = policy_model([](cs::System& s) { s.add_subsystem(tsys::kTransient, cs::subsystem_desc<tsys::TransRunNKid>("tr")); }, err);
    forwards(model, 7);
    // transient: rebuilt at the start of every forward, so its free input is materialized and initialized again each time
    CHECK(tsys::count_of("tr.c") == 7);
}

TEST_CASE("lifetime: a run_n(3) parent's roots are freed after the third run unless an eternal partner keeps them",
          "[system][lifetime]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::RunNTransTop>(cst::pooled_config(), std::size_t{3});
    model.setup();
    auto sys = model.get_subsystem("");
    for (int k = 1; k <= 4; ++k) {
        INFO("forward " << k);
        model.run();
        model.wait();
        CHECK(cs::detail::ModelAccess::in_scope(model, "op") == (k < 3));  // deleted by its parent after the third run
        if (k < 3) CHECK_NOTHROW(cst::synced_output<DType::f32, 1>(model, "op.mid.y"));  // still alive: its variable is readable
    }
    CHECK(tsys::count_of("op.prod") == 1);
}

TEST_CASE("lifetime: a do_once producer with an eternal reader is eternal; its value is held across runs", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::DoOnceProducer>(cst::pooled_config(), false);
    model.setup();
    for (int run = 1; run <= 3; ++run) {
        INFO("run " << run);
        model.run();
        model.wait();
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));  // do_once: deleted; r's port keeps the variable alive
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "r.y")) == tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 0.0f));
    }
}

TEST_CASE("lifetime: a tie_outputs variable under a transient System is deleted every run, values right", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::TransTieOutTop>(cst::pooled_config());
    model.setup();
    std::size_t steady = 0;  // the pool in use after the first run
    for (int run = 1; run <= 2; ++run) {
        INFO("run " << run);
        model.run();
        model.wait();
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "tr"));
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sink.y")) == tsys::iota_plus(1.0f, tsys::kLen));
        if (run == 1)
            steady = pool_in_use(model);  // rd.y, kept by the sink, outlives the forward
        else
            CHECK(pool_in_use(model) == steady);
    }
}

TEST_CASE("lifetime: a tie_inputs variable is initialized by the tie; a member's own init loses to it", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::TieInit>(default_config());
    model.setup();
    model.run();
    model.wait();
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(6.0f));  // 2 * 3, not 2 * 100
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == tsys::filled(9.0f));  // 3 * 3
}

TEST_CASE("lifetime: two tied members with inits: the tie's init wins, else the later member's; nothing throws",
          "[system][lifetime][init]") {
    // Design review, section 2 (corrected by Jack, 2026-10-03): the last declaration wins. There is no same-level error.
    SECTION("a tie-level init_input declared after the tie") {
        cs::Model model = cs::create_model<tsys::TieInitTwo>(default_config(), true);
        REQUIRE_NOTHROW(model.setup());
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(14.0f));  // 2 * 7
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == tsys::filled(21.0f));  // 3 * 7
    }
    SECTION("no tie-level init: the later merged member (b) wins") {
        cs::Model model = cs::create_model<tsys::TieInitTwo>(default_config(), false);
        REQUIRE_NOTHROW(model.setup());
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(400.0f));  // 2 * 200
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == tsys::filled(600.0f));  // 3 * 200
    }
}

TEST_CASE("lifetime: a model-level promotion is eternal", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::PromoteFromTransient>(cst::pooled_config());
    model.setup();
    for (int run = 1; run <= 2; ++run) {
        model.run();
        model.wait();
        // the promotion's alias holds the variable, though the transient producer is gone
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "y")) == tsys::iota_plus(1.0f, tsys::kLen));
    }
}

TEST_CASE("lifetime: a mark under a transient System is collected at setup and stays", "[system][lifetime]") {
    // Changed (design review, 11.2, P4): marks have no scope rule. They are collected at setup and stay; the transient
    // System's variable lives only while it does, and a forward does not touch the collected mark.
    cs::Model model = cs::create_model<tsys::MarkedInTransient>(cst::pooled_config());
    REQUIRE_NOTHROW(model.setup());
    REQUIRE(model.design_vars().size() == 1);
    CHECK(model.design_vars()[0].path == "tr.a.x");
    model.run();
    model.wait();
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "tr"));  // deleted after its turn
    REQUIRE(model.design_vars().size() == 1);
    CHECK(model.design_vars()[0].path == "tr.a.x");
}

TEST_CASE("lifetime: rearm sends a do_once's variables back out of scope and they are initialized again", "[system][lifetime][rearm]") {
    tsys::g_inits.clear();
    tsys::g_p_x = 4.0f;
    cs::Model model = cs::create_model<tsys::DoOnceInit>(cst::pooled_config());
    model.setup();
    forwards(model, 3);
    CHECK(tsys::count_of("p") == 1);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "r.y")) == tsys::filled(8.0f));  // 2 * (1 * 4)
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));

    tsys::rearm_next(model, "p");  // rebuilt by the parent at the start of the next forward: counter back to do_once
    forwards(model, 3);
    CHECK(tsys::count_of("p") == 2);  // p.x was allocated and initialized again
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "r.y")) == tsys::filled(8.0f));
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));
}

TEST_CASE("lifetime: transient together with run_n or do_once throws, naming the subsystem", "[system][lifetime]") {
    std::string err;
    tsys::expect_throws(
        [&] {
            (void)policy_model(
                [](cs::System& s) {
                    s.add_subsystem(cs::property_list{cs::property::transient{}, cs::property::run_n{2}},
                                    cs::subsystem_desc<tsys::RunCounted>("a", std::string{"a"}));
                },
                err);
        },
        "add_subsystem('a'): transient cannot be combined with run_n or do_once");
    tsys::expect_throws(
        [&] {
            (void)policy_model(
                [](cs::System& s) {
                    s.add_subsystem(cs::property_list{cs::property::do_once{}, cs::property::transient{}},
                                    cs::subsystem_desc<tsys::RunCounted>("b", std::string{"b"}));
                },
                err);
        },
        "add_subsystem('b'): transient cannot be combined with run_n or do_once");
}

namespace tsys2 {
/// A System with a FINITE value source: nb (NCopy, run_n{2}) gives the tie's extent by VALUE (a from_value shape hook on the tie), a and b
/// are the members. nb takes two turns and is deleted after the second. A hook reads its source in the same forward, right
/// before the first child that has the target (design review, section 10), so the source must still be alive in the forward
/// that first sets the shape: a do_once producer would be deleted before the hook ran. With `hold_n`, the System also promotes
/// nb.n as "n": the promotion holds nb's variable after nb is deleted, so its VALUE stays readable.
class ValueSourceSys : public cs::System {
    bool hold_n_;

public:
    explicit ValueSourceSys(bool hold_n = false) : hold_n_{hold_n} {}
    void setup() override {
        add_subsystem(cs::property_list{cs::property::run_n{2}}, cs::subsystem_desc<NCopy>("nb"));
        add_subsystem(cs::subsystem_desc<TScale>("a"));
        add_subsystem(cs::subsystem_desc<TScale>("b"));
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x");
        shape_hook("x", cs::dim::from_value("nb.n", 0));
        if (hold_n_) promote<DType::u32, 1>("nb.n", "n");
    }
};
}  // namespace tsys2

TEST_CASE("lifetime: a transient System with a finite value source for a hook is deleted on schedule",
          "[system][lifetime][alias]") {
    // Was: "a value-driven alias shape keeps a finite producer's output alive until it has materialized". Under section 10
    // nothing keeps a producer's output alive for a tie or a hook: the hook reads the source's value in the forward that sets
    // the shape, so the source producer must be alive then (nb takes two turns). Each forward rebuilds the transient System
    // with a fresh nb, so every forward sees a live source; the System itself is deleted after its turn.
    auto model = tsys2::make([](cs::System& s) { s.add_subsystem(tsys::kTransient, cs::subsystem_desc<tsys2::ValueSourceSys>("s")); },
                             cst::pooled_config());
    model.setup();
    model.update_input("s.nb.v", cs::init::from_host(tsys2::one_u32(4)));
    for (int k = 1; k <= 2; ++k) {
        INFO("forward " << k);
        CHECK_NOTHROW(model.run());
        model.wait();
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "s"));  // deleted after its turn, on schedule
    }
}

TEST_CASE("lifetime: a hook whose value source is gone keeps the last shape",
          "[system][lifetime][alias]") {
    // Design review, sections 9 and 10: a hook source that is gone (its instance expired) can never change again, so while the
    // target's shape is known the hook does nothing and the last shape stands; "... does not exist right now" is for a shape
    // that was never known. nb takes two turns: forwards 1 and 2 read it, forward 3 runs after nb is gone. Nothing needs to
    // hold nb.n for the SHAPE to survive. Promoting nb.n (hold) additionally keeps its VALUE readable.
    for (const bool hold : {false, true}) {
        DYNAMIC_SECTION((hold ? "nb.n is promoted: its value stays readable too" : "nothing holds nb.n")) {
            auto model = tsys2::make(
                [hold](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tsys2::ValueSourceSys>("s", hold)); },
                cst::pooled_config());
            model.setup();
            model.update_input("s.nb.v", cs::init::from_host(tsys2::one_u32(4)));
            for (int k = 1; k <= 3; ++k) {
                INFO("forward " << k);
                CHECK_NOTHROW(model.run());
                model.wait();
                CHECK(cst::synced_input<DType::f32, 1>(model, "s.x").shape()[0] == 4);  // the last shape stands from forward 3 on
                CHECK((!cs::detail::ModelAccess::in_scope(model, "s.nb")) == (k >= 2));  // deleted after its second turn
                CHECK(cs::detail::ModelAccess::in_scope(model, "s"));
                if (hold) CHECK(to_vector(cst::synced_output<DType::u32, 1>(model, "s.n")) == std::vector<std::uint32_t>{4});  // the value
            }
        }
    }
}
// =============================================================================================
// Shared ownership and parent-owned children: the Round 4 tests
// =============================================================================================
namespace tsys3 {
/// Two eternal Axpb members tied to one x; x declares a LOGGING init one level up (it logs at each application).
class TieLogInit : public tsys::RearmSys {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<tsys::Axpb>("a", tsys::tied_x, 2.0f, 0.0f));
        add_subsystem(cs::subsystem_desc<tsys::Axpb>("b", tsys::tied_x, 3.0f, 0.0f));
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(tsys::kLen));
        init_input("x", tsys::logging_fill("tie.x", 3.0f));
    }
};
/// p -> t -> u, all TRANSIENT, then an eternal sink: every stage is a transient reader of a transient producer.
class TransThree : public cs::System {
public:
    void setup() override {
        add_subsystem(tsys::kTransient, cs::subsystem_desc<tsys::Src>("p"));
        add_subsystem(tsys::kTransient, cs::subsystem_desc<tsys::Axpb>("t", 2.0f, 1.0f));
        add_subsystem(tsys::kTransient, cs::subsystem_desc<tsys::Axpb>("u", 1.0f, 0.5f));
        add_subsystem(cs::subsystem_desc<tsys::Axpb>("sink", 1.0f, 0.0f));
        // A transient's data goes when it is dearmed after its turn. What a later transient reads must be held by
        // something that stays in scope: promote it, and connect the promotion.
        promote<DType::f32, 1>("p.y", "py");
        promote<DType::f32, 1>("t.y", "ty");
        connect("py", "t.x");
        connect("ty", "u.x");
        connect("u.y", "sink.x");
    }
};
}  // namespace tsys3

TEST_CASE("lifetime: a variable survives its producer being deleted, through an eternal consumer", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::DoOnceProducer>(default_config(), false);
    model.setup();
    for (int run = 1; run <= 3; ++run) {
        INFO("run " << run);
        model.run();
        model.wait();
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));  // the producer is gone after its one turn
        // r holds the variable through its port
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "r.x")) == tsys::iota_plus(1.0f, tsys::kLen));
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "r.y")) == tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 0.0f));
    }
}

TEST_CASE("lifetime: a chain of transients keeps its data through promotions", "[system][lifetime][transient]") {
    cs::Model model = cs::create_model<tsys3::TransThree>(cst::pooled_config());
    model.setup();
    std::size_t steady = 0;  // the pool in use after the first run
    for (int run = 1; run <= 3; ++run) {
        INFO("run " << run);
        model.run();
        model.wait();
        // each transient is rearmed before its own turn and dearmed right after it
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "t"));
        CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "u"));
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "sink.y")) == tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.5f));
        if (run == 1)
            steady = pool_in_use(model);  // p.y and t.y (held by their promotions) and u.y (held by the sink) outlive the forward
        else
            CHECK(pool_in_use(model) == steady);
    }
}

TEST_CASE("init: a declared init applies only at materialization, not when a child is rearmed while the variable is live",
          "[system][init][rearm]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys3::TieLogInit>(default_config());
    model.setup();
    forwards(model, 1);
    CHECK(tsys::count_of("tie.x") == 1);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(6.0f));
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == tsys::filled(9.0f));

    tsys::rearm_next(model, "b");  // a keeps x live, so b's new port joins a live variable
    forwards(model, 1);
    CHECK(tsys::count_of("tie.x") == 1);  // no second init
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == tsys::filled(9.0f));
}

TEST_CASE("init: Model::update_input on a live variable is applied at the next run, once", "[system][init]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys3::TieLogInit>(default_config());
    model.setup();
    forwards(model, 1);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(6.0f));

    model.update_input("x", cs::init::from_host(tsys::filled(5.0f)));
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(6.0f));  // nothing is written until the run
    forwards(model, 1);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(10.0f));
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == tsys::filled(15.0f));
    forwards(model, 1);
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tsys::filled(10.0f));  // applied once, not again
}

TEST_CASE("get_subsystem: an undeclared name throws, a dearmed one is still a node", "[system][run_policy]") {
    cs::Model model = cs::create_model<tsys::DoOnceProducer>(default_config(), false);
    model.setup();
    CHECK(cs::detail::ModelAccess::in_scope(model, "p"));
    tsys::expect_throws([&] { (void)model.get_subsystem("nope"); }, "no subsystem 'nope'");
    forwards(model, 1);
    CHECK_FALSE(cs::detail::ModelAccess::in_scope(model, "p"));
    CHECK(cs::detail::ModelAccess::in_scope(model, "r"));
    CHECK_NOTHROW((void)model.get_subsystem("p"));  // out of scope, but still a node of the tree
    tsys::expect_throws([&] { (void)model.get_subsystem("nope"); }, "no subsystem 'nope'");
}

// =============================================================================================
// 6e. shape authority: unspecified dims are supplied from outside; specified dims are constraints
// =============================================================================================
namespace tshape {

// ---- fixtures -------------------------------------------------------------------------------
/// Scale2: y = 2 x, rank 2; x is wholly unspecified, y follows it.

class Scale2 : public cs::Component<Scale2> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    static void compute_forward(sycl::id<2> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i]; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<2>());
        declare_output(outputs.y, "y", cs::unspecified_shape<2>());
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0], shape_of(inputs.x)[1]); }  // the output follows the input (section 10: no derived dims)
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

/// An output whose dimension is unspecified and that no tie owns (a setup error).

class OutUnspec : public cs::Component<OutUnspec> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.y[i] = 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.y, "y", cs::unspecified); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

/// FixedOut(n): y[i] = i + 1, y fixed n.

class FixedOut : public cs::Component<FixedOut> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.y[i] = static_cast<float>(i[0] + 1); }

    Inputs  inputs;
    Outputs outputs;

private:
    std::size_t               n_;

public:
    explicit FixedOut(std::size_t n) : n_{n} {}
    void declare() override { declare_output(outputs.y, "y", static_cast<int>(n_)); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

// DELETED fixtures: Pair (an input declared INHERITED from another: FromInputDim) and ByVal (an input declared VALUE-DRIVEN:
// FromInput). Both declaration kinds are removed (section 10); a value-driven input is a shape hook now.

/// Forward must throw with every needle; the stream is drained afterwards.
template <class Needles>
void forward_throws(cs::Model& model, const Needles& needles) {
    std::string what = "<no exception>";
    try {
        model.run();
    } catch (const std::exception& e) {
        what = e.what();
    }
    model.wait();
    INFO("thrown: " << what);
    for (const char* n : needles) CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}

std::vector<float> iota_doubled(std::size_t n, float scale = 2.0f) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = scale * static_cast<float>(i + 1);
    return v;
}

}  // namespace tshape

TEST_CASE("shape: unspecified dims are supplied by a connection", "[system][shape]") {
    auto model = tsys2::make([](cs::System& s) {
        tsys2::add_bins(s);
        s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
        s.connect("v.b", "a.x");
    });
    model.setup();
    for (const std::uint32_t n : {5u, 3u}) {  // the supplier changes between runs, the consumer follows
        model.update_input("v.n", cs::init::from_host(tsys2::one_u32(n)));
        model.run();
        model.wait();
        CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == n);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tshape::iota_doubled(n));
    }
}

TEST_CASE("shape: unspecified dims are supplied by a tie (an initializer on it, or a connection into it)", "[system][shape]") {
    SECTION("an initializer on the tie") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x");  // no shape argument: unspecified
            s.init_input("x", cs::init::fill(3.0f, {5}));
        });
        model.setup();
        model.run();
        model.wait();
        CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == 5);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == std::vector<float>(5, 6.0f));
    }
    SECTION("a connection into the tie") {
        auto model = tsys2::make([](cs::System& s) {
            tsys2::add_bins(s);
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x");
            s.connect("v.b", "x");
        });
        model.setup();
        model.update_input("v.n", cs::init::from_host(tsys2::one_u32(4)));
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == tshape::iota_doubled(4));
    }
}

TEST_CASE("shape: every initializer kind supplies the shape of a free unspecified input", "[system][shape]") {
    const auto run = [](cs::InitSpec init, std::size_t want_len) {
        auto model = tsys2::make([init](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.init_input("a.x", init);
        });
        model.setup();
        model.run();
        model.wait();
        CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == want_len);
        return to_vector(cst::synced_output<DType::f32, 1>(model, "a.y"));
    };
    SECTION("fill(v, shape)") { CHECK(run(cs::init::fill(2.0f, {4}), 4) == std::vector<float>(4, 4.0f)); }
    SECTION("zero(shape)") { CHECK(run(cs::init::zero({3}), 3) == std::vector<float>(3, 0.0f)); }
    SECTION("from_host(data): rank 1 carries its size") {
        CHECK(run(cs::init::from_host(std::vector<float>{1.0f, 2.0f, 3.0f}), 3) == std::vector<float>{2.0f, 4.0f, 6.0f});
    }
    SECTION("from_host(data, shape)") {
        CHECK(run(cs::init::from_host(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}), 4) == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f});
    }
    SECTION("custom_host(fn, shape)") {
        const auto v = run(cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostInit<DType::f32, 1>&)>{
                               [](cs::HostInit<DType::f32, 1>& h) {
                                   for (std::size_t k = 0; k < h.size(); ++k) h[k] = static_cast<float>(k);
                               }},
                                                              std::vector<std::size_t>{3}),
                           3);
        CHECK(v == std::vector<float>{0.0f, 2.0f, 4.0f});
    }
    SECTION("custom(fn, shape): the shape alone is checked") {
        const auto v = run(cs::init::custom<DType::f32, 1>(
                               cs::init::CustomFn<DType::f32, 1>{[](sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                                                                   cs::Initializer<DType::f32, 1>) { return q.single_task(dev, deps, []() {}); }},
                               std::vector<std::size_t>{6}),
                           6);
        CHECK(v.size() == 6);
    }
    SECTION("rank 2: from_host(data, shape) and fill(v, shape)") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::Scale2>("a"));
            s.init_input("a.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}));
        });
        model.setup();
        model.run();
        model.wait();
        CHECK(cst::synced_input<DType::f32, 2>(model, "a.x").shape()[0] == 2);
        CHECK(cst::synced_input<DType::f32, 2>(model, "a.x").shape()[1] == 3);
        CHECK(to_vector(cst::synced_output<DType::f32, 2>(model, "a.y")) == std::vector<float>{2, 4, 6, 8, 10, 12});
    }
}

TEST_CASE("shape: an unspecified dim nothing supplies is thrown at the first forward, not at setup", "[system][shape]") {
    // Design review, section 2: setup checks wiring rules only and shapes are reactive. Both of these used to be setup errors
    // (check_ties); the REQUIRE_THROWS moved from setup() to forward().
    SECTION("an untied output with an unspecified dim that its producer never sets") {
        auto model = tsys2::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tshape::OutUnspec>("o")); });
        CHECK_NOTHROW(model.setup());
        // variable.hpp shape_unknown names "its producer, in prepare_forward (set_shape)" for an unset output dimension.
        tshape::forward_throws(model, std::vector<const char*>{"its shape is not known yet", "its producer, in prepare_forward (set_shape)"});
    }
    SECTION("a free input with an unspecified dim and no shape-giving initializer") {
        auto model = tsys2::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a")); });
        CHECK_NOTHROW(model.setup());
        // (the old message also said "and no initializer that gives it a shape"; variable.hpp's shape_unknown does not)
        tshape::forward_throws(model, std::vector<const char*>{"its shape is not known yet"});
    }
    SECTION("fill(v) carries no shape") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.init_input("a.x", cs::init::fill(1.0f));
        });
        CHECK_NOTHROW(model.setup());
        tshape::forward_throws(model, std::vector<const char*>{"its shape is not known yet"});
    }
}

TEST_CASE("shape: a specified dim is a constraint; a violation names both sides", "[system][shape]") {
    // Fixed against fixed is a wiring rule (wiring.hpp, merged_definition: "dimension d: ... declares n but ... declares m:
    // the extents must match"), thrown at setup. A producer that sets an output dimension differently from the declared one
    // is checked when it sets it, at the forward (variable.hpp, check_setter: "dimension d is declared as n but <who> gave m").
    SECTION("fixed, at setup, against a tie") {
        tsys::expect_throws_all(
            [&] {
                auto model = tsys2::make([](cs::System& s) {
                    s.add_subsystem(cs::subsystem_desc<tshape::FixedOut>("a", std::size_t{5}));
                    s.add_subsystem(cs::subsystem_desc<tshape::FixedOut>("b", std::size_t{5}));
                    s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(8), cs::TieOutputsPolicy::allow_plain);
                });
                model.setup();
            },
            {"dimension 0", "the extents must match", "'a.y'"});
    }
    SECTION("fixed, against a connected output: both are fixed, so the connect throws at setup") {
        tsys::expect_throws_all(
            [&] {
                auto model = tsys2::make([](cs::System& s) {
                    s.add_subsystem(cs::subsystem_desc<tshape::FixedOut>("p", std::size_t{5}));
                    s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("c", 1.0f, 0.0f));  // x fixed kLen
                    s.connect("p.y", "c.x");
                });
                model.setup();
            },
            {"dimension 0", "the extents must match", "'p.y'", "'c.x'"});
    }
    // DELETED: "inherited (FromInputDim)", "value-driven (FromInput)" and "a matching constraint is silent" (the last used the
    // value-driven ByVal fixture). Derived dims and value-driven component declarations are gone (section 10); a hooked input
    // takes no connection, so "a constraint against a connected output" cannot arise for them. tests/test_shapes.cpp covers
    // the hook errors and the set_shape checks.
}

TEST_CASE("shape: the rank must match", "[system][shape]") {
    auto model = tsys2::make([](cs::System& s) {
        tsys2::add_bins(s);  // a rank-1 output
        s.add_subsystem(cs::subsystem_desc<tshape::Scale2>("a"));  // a rank-2 input
        s.connect("v.b", "a.x");
    });
    // The rank is a wiring rule: connect throws at setup (wiring.hpp, check_compatible).
    std::string what = "<no exception>";
    try {
        model.setup();
    } catch (const std::exception& e) {
        what = e.what();
    }
    INFO("thrown: " << what);
    for (const char* n : {"'v.b' has rank 1", "'a.x' has rank 2", "the rank must match"})
        CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}

TEST_CASE("shape: a tied output member with a conflicting fixed dim", "[system][shape][tie]") {
    SECTION("conflicting") {
        tsys::expect_throws_all(
            [&] {
                auto model = tsys2::make([](cs::System& s) {
                    s.add_subsystem(cs::subsystem_desc<tshape::FixedOut>("a", std::size_t{4}));
                    s.add_subsystem(cs::subsystem_desc<tshape::FixedOut>("b", std::size_t{6}));
                    s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(6), cs::TieOutputsPolicy::allow_plain);
                });
                model.setup();
            },
            {"dimension 0", "the extents must match", "'a.y'"});
    }
    SECTION("agreeing: the member declares what the tie says") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tshape::FixedOut>("a", std::size_t{6}));
            s.add_subsystem(cs::subsystem_desc<tshape::FixedOut>("b", std::size_t{6}));
            s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::shape(6), cs::TieOutputsPolicy::allow_plain);
        });
        CHECK_NOTHROW(model.setup());
    }
}

// =============================================================================================
// depends_on (design review, section 3)
// =============================================================================================
TEST_CASE("depends_on: B reads what A wrote through a side channel, with no variable between them", "[system][depends_on]") {
    // A is a host task that sleeps 50 ms and then writes a global; B is a host task that copies the global into its own
    // output. They share no variable, so only the declared edge orders them: B's submission waits on A's event.
    tsys::g_side.store(0.0f);
    cs::Model model = cs::create_model<tsys::DependsTop>(default_config(), true);
    model.setup();
    auto got = wait_and_view<DType::f32, 1>(model, "b.got");
    CHECK(got[std::size_t{0}] == 42.0f);
}

TEST_CASE("depends_on: declaration errors", "[system][depends_on][errors]") {
    using Build = std::function<void(cs::System&)>;
    const auto setup_throws = [](Build build, const std::string& needle) {
        tsys::expect_throws(
            [&] {
                auto model = tsys2::make(build);
                model.setup();
            },
            needle);
    };
    const auto two = [](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("a", 1.0f, 0.0f));
        s.add_subsystem(cs::subsystem_desc<tsys::Axpb>("b", 1.0f, 0.0f));
    };
    SECTION("a child can depend only on one added before it") {
        setup_throws(
            [&](cs::System& s) {
                two(s);
                s.depends_on("a", "b");  // b was added after a
            },
            "can depend only on a subsystem added before it");
        setup_throws(
            [&](cs::System& s) {
                two(s);
                s.depends_on("a", "a");
            },
            "can depend only on a subsystem added before it");
    }
    SECTION("unknown names") {
        setup_throws(
            [&](cs::System& s) {
                two(s);
                s.depends_on("zz", "a");
            },
            "'zz' is not a subsystem of the root System");
        setup_throws(
            [&](cs::System& s) {
                two(s);
                s.depends_on("b", "zz");
            },
            "'zz' is not a subsystem of the root System");
    }
    SECTION("a duplicate edge is ignored") {
        auto model = tsys2::make([&](cs::System& s) {
            two(s);
            s.depends_on("b", "a");
            s.depends_on("b", "a");
        });
        CHECK_NOTHROW(model.setup());
        model.run();
        model.wait();
    }
}

// =============================================================================================
// placement across two devices (design review, section 2: the outer device wins)
// =============================================================================================
TEST_CASE("placement: a child placed by its author on device X, added by its parent on device Y, ends up on Y", "[system][placement]") {
    sycl::device first, host;
    if (!tsys::pick_two(first, host)) SKIP("a non-OpenMP device and the OpenMP host device are not both visible");
    // the author (PlacedInner) says `first`; the parent (DevTop) adds it on `host`
    cs::Model model = cs::create_model<tsys::DevTop>(default_config(), first, host);
    model.setup();
    CHECK(cs::detail::ModelAccess::instance(model, "in.k.x")->device == host);
    CHECK(cs::detail::ModelAccess::instance(model, "in.k.y")->device == host);

    model.update_input("in.k.x", cs::init::fill(3.0f));
    model.run();
    model.wait();
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "in.k.y")) == tsys::filled(3.0f));
}

// =============================================================================================
// implicit zero initializer, and re-initialization on reshape
// =============================================================================================
namespace tsys2 {
/// A TScale "a" whose free input x follows the shape of v.b (a hook on a free input), v.n giving the size.
inline void hooked_free_input(cs::System& s, std::optional<float> fill) {
    add_bins(s);
    s.add_subsystem(cs::subsystem_desc<TScale>("a"));
    s.shape_hook("a.x", cs::dim::from_shape("v.b", 0));
    if (fill) s.init_input("a.x", cs::init::fill(*fill));
}
/// Resizes v.n to each of `sizes`, runs, and checks a.x has that size and every element of a.y is `scale * value`.
inline void run_sizes(cs::Model& model, std::initializer_list<std::uint32_t> sizes, const char* y_path, float want) {
    for (const std::uint32_t n : sizes) {
        INFO("n = " << n);
        model.update_input("v.n", cs::init::from_host(one_u32(n)));
        model.run();
        model.wait();
        CHECK(cst::synced_input<DType::f32, 1>(model, "a.x").shape()[0] == n);
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, y_path)) == std::vector<float>(n, want));
    }
}
}  // namespace tsys2

TEST_CASE("init: reshaping a materialized free input runs the implicit zero again", "[system][init][reshape]") {
    auto model = tsys2::make([](cs::System& s) { tsys2::hooked_free_input(s, std::nullopt); });
    model.setup();
    tsys2::run_sizes(model, {3u, 6u, 4u, 8u}, "a.y", 0.0f);  // x grew and shrank: all zeros each time
}

TEST_CASE("init: reshaping a materialized free input runs its declared fill again", "[system][init][reshape]") {
    auto model = tsys2::make([](cs::System& s) { tsys2::hooked_free_input(s, 7.0f); });
    model.setup();
    tsys2::run_sizes(model, {3u, 6u, 4u, 8u}, "a.y", 14.0f);  // the new elements are the fill too, not leftovers
}

TEST_CASE("init: reshaping a tie_inputs alias runs its initializer again", "[system][init][reshape]") {
    for (const bool explicit_fill : {false, true}) {
        INFO((explicit_fill ? "explicit fill(7)" : "implicit zero"));
        auto model = tsys2::make([explicit_fill](cs::System& s) {
            tsys2::add_bins(s);
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x");
            s.shape_hook("x", cs::dim::from_shape("v.b", 0));
            if (explicit_fill) s.init_input("x", cs::init::fill(7.0f));
        });
        model.setup();
        for (const std::uint32_t n : {3u, 6u, 4u, 8u}) {
            INFO("n = " << n);
            model.update_input("v.n", cs::init::from_host(tsys2::one_u32(n)));
            model.run();
            model.wait();
            const std::vector<float> want(n, explicit_fill ? 14.0f : 0.0f);
            CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == want);
            CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == want);
        }
    }
}

TEST_CASE("init: in a tie an explicit initializer beats an implicit one, whichever member declares it", "[system][init][tie]") {
    for (const bool a_declares : {true, false}) {
        INFO((a_declares ? "a.x declares the fill, b.x nothing" : "b.x declares the fill, a.x nothing"));
        auto model = tsys2::make([a_declares](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
            s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
            s.init_input(a_declares ? "a.x" : "b.x", cs::init::fill(7.0f));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(5));
        });
        model.setup();
        model.run();
        model.wait();
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == std::vector<float>(5, 14.0f));
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == std::vector<float>(5, 14.0f));
    }
}

TEST_CASE("init: in a tie with two explicit initializers the last one wins", "[system][init][tie]") {
    auto model = tsys2::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
        s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
        s.init_input("a.x", cs::init::fill(3.0f));
        s.init_input("b.x", cs::init::fill(5.0f));
        s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(5));
    });
    model.setup();
    model.run();
    model.wait();
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == std::vector<float>(5, 10.0f));
    CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == std::vector<float>(5, 10.0f));
}

TEST_CASE("init: a connected input and an output are never initialized, not even when the chain is reshaped", "[system][init][reshape]") {
    auto model = tsys2::make([](cs::System& s) {
        tsys2::add_bins(s);
        s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("a"));
        s.add_subsystem(cs::subsystem_desc<tsys2::TScale>("b"));
        s.shape_hook("a.x", cs::dim::from_shape("v.b", 0));
        s.init_input("a.x", cs::init::fill(7.0f));
        s.connect("a.y", "b.x");
    });
    model.setup();
    for (const std::uint32_t n : {3u, 6u, 4u}) {
        INFO("n = " << n);
        model.update_input("v.n", cs::init::from_host(tsys2::one_u32(n)));
        model.run();
        model.wait();
        // a.y (an output) holds what a wrote, b.x (connected) is that same data, and b.y is computed from it: nothing zeroed them
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "a.y")) == std::vector<float>(n, 14.0f));
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "b.x")) == std::vector<float>(n, 14.0f));
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == std::vector<float>(n, 28.0f));
    }
}
