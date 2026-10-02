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
//   init          init_input metadata: the highest level wins; Model::init_input on a
//                 durable free input between forwards
//   transient     registration metadata: release after the last user, kept
//                 outputs, transient Components, promotion, nesting, a free
//                 input re-initialized each forward, back-to-back forwards
//   run-N         n runs; transient subtree allocated/initialized per run
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
#include <functional>
#include <memory>
#include <optional>
#include <cstdint>
#include <iostream>
#include <span>
#include <sstream>
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
    model.wait();  // setup() may have submitted initializer work before it threw
}

constexpr std::size_t kLen = 8;

// A logging init (custom_host) appends its tag each time the framework applies it.
std::vector<std::string> g_inits;

/// An init that logs `tag` at each application and fills the root with v.
cs::InitSpec logging_fill(std::string tag, float v) {
    return cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostViewMut<DType::f32, 1>&)>{
        [tag = std::move(tag), v](cs::HostViewMut<DType::f32, 1>& h) {
            g_inits.push_back(tag);
            for (std::size_t k = 0; k < h.size(); ++k) h[k] = v;
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
struct SrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = static_cast<float>(i[0] + 1); }
};

class Src : public cs::Component<SrcBody> {
    cs::Output<DType::f32, 1> y_;
    bool                      tied_;

public:
    /// `tied`: y is rank-only (a tie_outputs member; the tie alias owns the shape).
    explicit Src(bool tied = false) : tied_{tied} {}
    void setup() override {
        if (tied_)
            declare_output(y_, "y", cs::unspecified_shape<1>());
        else
            declare_output(y_, "y", static_cast<int>(kLen));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {}, {y_}, {});
    }
};

// ---- Axpb: y = a * x + b, x of kLen (x0: a fill init on x; log: a logging init instead) ------------
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

/// Selects the Axpb whose input x is rank-only (a tie_inputs member).
struct TiedX {};
inline constexpr TiedX tied_x{};

class Axpb : public cs::Component<AxpbBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
    float                     a_, b_;
    std::optional<float>      x0_;
    std::string               log_;
    bool                      tied_ = false;

public:
    Axpb(float a, float b, std::optional<float> x0 = std::nullopt, std::string log = {})
        : a_{a}, b_{b}, x0_{x0}, log_{std::move(log)} {}
    Axpb(TiedX, float a, float b, std::optional<float> x0 = std::nullopt) : a_{a}, b_{b}, x0_{x0}, tied_{true} {}
    void setup() override {
        if (tied_)
            declare_input(x_, "x", cs::unspecified_shape<1>());
        else
            declare_input(x_, "x", static_cast<int>(kLen));
        declare_output(y_, "y", static_cast<int>(kLen));
        if (!log_.empty())
            init_input("x", logging_fill(log_, x0_.value_or(0.0f)));
        else if (x0_)
            init_input("x", cs::init::fill(*x0_));
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
        declare_input(x_, "x", cs::unspecified_shape<1>());
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
        declare_input(x_, "x", cs::unspecified_shape<2>());
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

// ---- Adder: acc[i] += v, an ATOMIC output (zeroed by a ZeroAcc producer) ------------------------------------
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
    void setup() override { declare_output(acc_, "acc", cs::unspecified_shape<1>()); }  // a tie_outputs member
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(acc_.shape(), {}, {acc_}, {v_});
    }
};

// ---- ZeroAcc: the zeroing producer (outputs are never initialized, r01 §6.0) ---
struct ZeroAccBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> acc;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.acc[i].store(0.0f); }
};

class ZeroAcc : public cs::Component<ZeroAccBody> {
    cs::Output<DType::atomic_f32, 1> acc_;

public:
    void setup() override { declare_output(acc_, "acc", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(acc_.shape(), {}, {acc_}, {});
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
        promote<DType::f32, 1>("c.x", "cx");
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
        promote<DType::f32, 1>("child.y", "y");
    }
};

class PromoteTop : public cs::System {
public:
    void setup() override {
        add_subsystem<PromoteMid>("mid");
        add_subsystem<Axpb>("c", 3.0f, 0.0f);
        promote<DType::f32, 1>("mid.y", "y");
        connect("y", "c.x");  // from the promoted alias
    }
};

// ---- ties -----------------------------------------------------------------------------------
class TieInSys : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", tied_x, 2.0f, 0.0f);
        add_subsystem<Axpb>("b", tied_x, 3.0f, 0.0f);
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::alias::fixed(kLen));
    }
};

class TieInInner : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", tied_x, 2.0f, 0.0f);
        add_subsystem<Axpb>("b", tied_x, 3.0f, 0.0f);
        // An alias carries a shape, so it cannot itself be a tie member: pass the members up by promotion.
        promote<DType::f32, 1>("a.x", "ax");
        promote<DType::f32, 1>("b.x", "bx");
    }
};

class TieInTop : public cs::System {
public:
    void setup() override {
        add_subsystem<TieInInner>("in");
        add_subsystem<Axpb>("c", tied_x, 4.0f, 0.0f);
        // Three members, two of them promotions from the level below: one root.
        tie_inputs<DType::f32, 1>({"in.ax", "in.bx", "c.x"}, "tt", cs::alias::fixed(kLen));
    }
};

class TieOutAtomic : public cs::System {
public:
    void setup() override {
        add_subsystem<ZeroAcc>("z");  // first in registration order: zeroes the root the Adders accumulate on
        add_subsystem<Adder>("a", 1.0f);
        add_subsystem<Adder>("b", 2.0f);
        add_subsystem<Copy1>("c");
        tie_outputs<DType::atomic_f32, 1>({"z.acc", "a.acc", "b.acc"}, "sum",
                                          cs::alias::fixed(kLen));  // require_atomic (the default): all are atomic
        connect("sum", "c.x");                    // a tie alias is a source
    }
};

class TieOutPlain : public cs::System {
    bool allow_;

public:
    explicit TieOutPlain(bool allow = false) : allow_{allow} {}
    void setup() override {
        add_subsystem<Src>("a", true);
        add_subsystem<Src>("b", true);
        if (allow_) {
            tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::alias::fixed(kLen), cs::TieOutputsPolicy::allow_plain);
        } else {
            tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::alias::fixed(kLen));
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

// ---- init: the highest level wins -----------------------------------------------------------
constexpr float kD1Init = 99.0f;   // c1's own init: overridden
constexpr float kSysInit = 1.5f;   // the owner System's init_input: depth 1 beats depth 2

class InnerB : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("c1", 1.0f, 0.0f, kD1Init);
        add_subsystem<Axpb>("c2", 1.0f, 0.0f);
        connect("c1.y", "c2.x");
        init_input("c1.x", cs::init::fill(kSysInit));
    }
};

class TopB : public cs::System {
public:
    void setup() override {
        add_subsystem<InnerB>("inner");
        add_subsystem<Axpb>("c3", 1.0f, 0.0f);
        connect("inner.c2.y", "c3.x");
    }
};

// ---- transient ------------------------------------------------------------------------------
class TransSys : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("prod");
        add_subsystem<Axpb>("mid", 2.0f, 0.0f);
        add_subsystem<Axpb>("cons", 1.0f, 3.0f);
        connect("prod.y", "mid.x");
        connect("mid.y", "cons.x");
    }
};

class TopC : public cs::System {
public:
    void setup() override {
        add_subsystem<TransSys>("tr", cs::property_list{cs::property::transient{}});
        add_subsystem<Axpb>("sink", 1.0f, 0.5f);
        connect("tr.cons.y", "sink.x");  // the consumer's output feeds a Component OUTSIDE tr
    }
};

// ---- transient (registration metadata) -------------------------------------------------------
inline std::size_t pool_in_use_of(cs::Model& m) { return cst::pool_in_use(m); }

const cs::property_list kTransient{cs::property::transient{}};

class OneAxpb : public cs::System {
public:
    void setup() override { add_subsystem<Axpb>("a", 2.0f, 1.0f); }  // free x, no init: zeros
};

/// A durable free input with a logged init (x = 5): the init is applied by the owner on its first run.
class OneAxpbLog : public cs::System {
public:
    void setup() override { add_subsystem<Axpb>("a", 2.0f, 1.0f, 5.0f, std::string{"a"}); }
};

/// a (transient Src) -> b (transient Axpb) -> c (plain Axpb): a.y is transient, b.y is durable.
class TransComps : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("a", kTransient);
        add_subsystem<Axpb>("b", kTransient, 2.0f, 1.0f);
        add_subsystem<Axpb>("c", 1.0f, 0.5f);
        connect("a.y", "b.x");
        connect("b.y", "c.x");
    }
};

/// p (transient Src) read by t (transient) and n (plain): p.y has a non-transient registration.
class SharedRead : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("p", kTransient);
        add_subsystem<Axpb>("t", kTransient, 1.0f, 0.0f);
        add_subsystem<Axpb>("n", 1.0f, 0.0f);
        connect("p.y", "t.x");
        connect("p.y", "n.x");
    }
};

class TransPromo : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("p");
        promote<DType::f32, 1>("p.y", "y");
    }
};

/// "tr" is transient and promotes p.y as "y"; with `repromote` the root System promotes tr.y again.
class PromoInTrans : public cs::System {
    bool repromote_;

public:
    explicit PromoInTrans(bool repromote) : repromote_{repromote} {}
    void setup() override {
        add_subsystem<TransPromo>("tr", kTransient);
        if (repromote_) promote<DType::f32, 1>("tr.y", "out");
    }
};

class NestedOuter : public cs::System {
public:
    void setup() override { add_subsystem<TransSys>("inner", kTransient); }
};

class NestedTrans : public cs::System {
public:
    void setup() override { add_subsystem<NestedOuter>("outer", kTransient); }
};

/// p.x is a free input inside the transient subtree: a transient root, initialized (logged) at each allocation.
class TransAxpbSys : public cs::System {
public:
    void setup() override { add_subsystem<Axpb>("p", 2.0f, 1.0f, 7.0f, std::string{"tr.p"}); }
};

class TransFreeInput : public cs::System {
public:
    void setup() override {
        add_subsystem<TransAxpbSys>("tr", kTransient);
        add_subsystem<Axpb>("sink", 1.0f, 0.0f);
        connect("tr.p.y", "sink.x");
    }
};

// ---- run-N ----------------------------------------------------------------------------------
/// One Axpb in a plain System: y = 2 x, x a free input ("a.x"). Its parent adds it with run_n.
class OpAxpb : public cs::System {
public:
    void setup() override { add_subsystem<Axpb>("a", 2.0f, 0.0f); }
};

/// A root holding one OpAxpb added with run_n{n}, named "op"; free input "op.a.x", output "op.a.y". A plain
/// sink reads op.a.y: its eternal answer keeps the output alive after the last run; op.a.x is freed then.
class RunNTop : public cs::System {
    std::size_t n_;

public:
    explicit RunNTop(std::size_t n) : n_{n} {}
    void setup() override {
        add_subsystem<OpAxpb>("op", cs::property_list{cs::property::run_n{n_}});
        add_subsystem<Axpb>("sink", 1.0f, 0.0f);
        connect("op.a.y", "sink.x");
    }
};

/// The transient run-N subtree: prod (free input x = 4, logged) -> mid -> cons, y = 2 x + 3 end to end.
class OpTrans : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("prod", 1.0f, 0.0f, 4.0f, std::string{"op.prod"});
        add_subsystem<Axpb>("mid", 2.0f, 0.0f);
        add_subsystem<Axpb>("cons", 1.0f, 3.0f);
        connect("prod.y", "mid.x");
        connect("mid.y", "cons.x");
    }
};

/// op (OpTrans, run_n{n}) -> sink (y = x + 0.5): the consumer's output feeds the sink outside (an eternal partner), so it is durable.
class RunNTransTop : public cs::System {
    std::size_t n_;

public:
    explicit RunNTransTop(std::size_t n) : n_{n} {}
    void setup() override {
        add_subsystem<OpTrans>("op", cs::property_list{cs::property::run_n{n_}});
        add_subsystem<Axpb>("sink", 1.0f, 0.5f);
        connect("op.cons.y", "sink.x");
    }
};

// ---- run policy fixtures: a System that logs its tag each time its owner runs it ------------------
std::vector<std::string> g_runs;

std::size_t runs_of(const std::string& tag) {
    return static_cast<std::size_t>(std::count(g_runs.begin(), g_runs.end(), tag));
}

class RunCounted : public cs::System {
    std::string                      tag_;
    std::function<void(cs::System&)> kids_;  ///< adds children inside this System's own setup()

public:
    explicit RunCounted(std::string tag, std::function<void(cs::System&)> kids = {})
        : tag_{std::move(tag)}, kids_{std::move(kids)} {}
    void setup() override {
        if (kids_) kids_(*this);
    }
    std::unique_ptr<cs::LaunchDescription> run_compute_forward() override {
        g_runs.push_back(tag_);
        return System::run_compute_forward();
    }
};

/// A root whose structure is given by a callable run inside its setup().
class PolicyTop : public cs::System {
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
/// p (transient Src) -> t (transient Axpb, y = 2 x + 1) -> sink (plain Axpb): t.y is eternal (the sink answers -1);
/// p.y and t.x come into scope and go out of it with every run.
class TransChain : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("p", kTransient);
        add_subsystem<Axpb>("t", kTransient, 2.0f, 1.0f);
        add_subsystem<Axpb>("sink", 1.0f, 0.0f);
        connect("p.y", "t.x");
        connect("t.y", "sink.x");
    }
};

/// A transient System holding one Axpb "c" added with run_n{5} (a clamp to transient, with a warning).
class TransRunNKid : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("c", cs::property_list{cs::property::run_n{5}}, 2.0f, 0.0f, 4.0f, std::string{"tr.c"});
    }
};

/// p (Src, do_once) -> r (plain Axpb, y = 2 x), or -> t (transient Axpb).
class DoOnceProducer : public cs::System {
    bool transient_reader_;

public:
    explicit DoOnceProducer(bool transient_reader) : transient_reader_{transient_reader} {}
    void setup() override {
        add_subsystem<Src>("p", cs::property_list{cs::property::do_once{}});
        if (transient_reader_)
            add_subsystem<Axpb>("t", kTransient, 1.0f, 0.0f);
        else
            add_subsystem<Axpb>("r", 2.0f, 0.0f);
        connect("p.y", transient_reader_ ? "t.x" : "r.x");
    }
};

/// Two plain writers tie_outputs'd into "t" and read by rd, all inside a transient System; rd's output feeds an
/// eternal sink outside.
class TransTieSys : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("a", true);
        add_subsystem<Src>("b", true);
        add_subsystem<Axpb>("rd", 1.0f, 0.0f);
        tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::alias::fixed(kLen), cs::TieOutputsPolicy::allow_plain);
        connect("t", "rd.x");
    }
};
class TransTieOutTop : public cs::System {
public:
    void setup() override {
        add_subsystem<TransTieSys>("tr", kTransient);
        add_subsystem<Axpb>("sink", 1.0f, 0.0f);
        connect("tr.rd.y", "sink.x");
    }
};

/// Two tied members with inits of their own (100, 200); the tie declares x = 3 one level up.
class TieInit : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("a", tied_x, 2.0f, 0.0f, 100.0f);
        add_subsystem<Axpb>("b", tied_x, 3.0f, 0.0f, 200.0f);
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::alias::fixed(kLen));
        init_input("x", cs::init::fill(3.0f));
    }
};

/// A transient System holding a free-input Axpb whose x is marked as a design variable by the root.
class MarkedHolder : public cs::System {
public:
    void setup() override { add_subsystem<Axpb>("a", 2.0f, 1.0f); }
};
class MarkedInTransient : public cs::System {
public:
    void setup() override {
        add_subsystem<MarkedHolder>("tr", kTransient);
        add_design_var("tr.a.x");
    }
};

/// A root promotion of a transient child's output.
class PromoteFromTransient : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("p", kTransient);
        promote<DType::f32, 1>("p.y", "y");
    }
};

/// p (Axpb, free x = 4 logged, do_once) -> r (plain Axpb): p.x and p.y are rebuilt after a rearm.
class DoOnceInit : public cs::System {
public:
    void setup() override {
        add_subsystem<Axpb>("p", cs::property_list{cs::property::do_once{}}, 1.0f, 0.0f, 4.0f, std::string{"p"});
        add_subsystem<Axpb>("r", 2.0f, 0.0f);
        connect("p.y", "r.x");
    }
};

std::vector<float> filled(float v) { return std::vector<float>(kLen, v); }

/// Set the initial value of the free input at `path` to x = v (all elements); the owner applies it at its next run.
void set_x(cs::Model& model, const char* path, float v) {
    model.init_input(path, cs::init::from_host(filled(v)));
}

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
        tie_inputs<DType::f32, 1>({"c.x", "b.x"}, "t", cs::alias::fixed(kLen));  // c.x is already connected
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
constexpr float kSellarY2   = 1.5f;   // the MDA System's init_input (depth 1): the last word

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
        declare_input(z_, "z", cs::unspecified_shape<1>());  // z and x are tie members: the tie alias owns their shape
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_input(y2_, "y2", 1);
        declare_output(y1_, "y1", 1);
        init_input("y2", cs::init::fill(kSellarD1Y2));
    }
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
        declare_input(z_, "z", cs::unspecified_shape<1>());
        declare_input(y1_in_, "y1_in", cs::unspecified_shape<1>());
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
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_input(z_, "z", cs::unspecified_shape<1>());
        declare_input(y1_, "y1", cs::unspecified_shape<1>());
        declare_input(y2_, "y2", cs::unspecified_shape<1>());
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
        declare_input(y1_, "y1", cs::unspecified_shape<1>());
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
        declare_input(y2_, "y2", cs::unspecified_shape<1>());
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
        add_subsystem<SellarMDA>("mda");
        add_subsystem<Obj>("obj");
        add_subsystem<G1>("g1");
        add_subsystem<G2>("g2");

        tie_inputs<DType::f32, 1>({"mda.x", "obj.x"}, "x", cs::alias::fixed(1));
        tie_inputs<DType::f32, 1>({"mda.z1", "mda.z2", "obj.z"}, "z", cs::alias::fixed(2));  // members through promotions
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
        init_input("z", cs::init::from_host(std::vector<float>{5.0f, 2.0f}));
        init_input("x", cs::init::fill(1.0f));
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

    SECTION("one tie: init_input on the alias is seen by both") {
        cs::Model model = cs::create_model<tsys::TieInSys>(default_config());
        model.setup();
        CHECK(model.root().entry_at("a.x").root() == model.root().entry_at("b.x").root());
        CHECK(model.root().entry_at("x").root() == model.root().entry_at("a.x").root());
        model.init_input("x", cs::init::from_host(x));
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
        CHECK(sys.entry_at("in.ax").root() == root);
        CHECK(sys.entry_at("in.bx").root() == root);
        model.init_input("tt", cs::init::from_host(x));
        model.forward();
        model.wait();
        for (const char* path : {"in.a.x", "in.b.x", "c.x", "in.ax", "in.bx"})
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
        // The zeroing producer runs first in every forward, so a second forward gives the same sum.
        auto again = wait_and_view<DType::f32, 1>(model, "sum");
        CHECK(to_vector(again) == std::vector<float>(tsys::kLen, 3.0f));
    }

    SECTION("plain outputs without allow_plain throw") {
        tsys::expect_setup_throws<tsys::TieOutPlain>("not atomic");
        cs::Model model = cs::create_model<tsys::TieOutPlain>(default_config(), false);
        tsys::expect_throws([&] { model.setup(); }, "require_atomic");
        model.wait();  // setup() may have submitted initializer work before it threw
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
    model.wait();  // setup() submitted initializer work; wait before the Model goes out of scope
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
TEST_CASE("init: the highest level wins (System init_input beats the Component's, children below owners)",
          "[system][init]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::TopB>(default_config());
    model.setup();

    // c1 declared 99 (depth 2); the owner System `inner` declared 1.5 (depth 1) and wins.
    // The init is applied by the owner (c1) in the first forward.
    // c2.x is connected, so nothing is declared on it and it is not initialized.
    model.forward();
    model.wait();
    CHECK(to_vector(model.view_input<DType::f32, 1>("inner.c1.x")) == std::vector<float>(tsys::kLen, tsys::kSysInit));
    CHECK(to_vector(model.view_output<DType::f32, 1>("c3.y")) == std::vector<float>(tsys::kLen, tsys::kSysInit));
}

TEST_CASE("init: init_input on a durable free input between forwards", "[system][init]") {
    cs::Model model = cs::create_model<tsys::OneAxpb>(default_config());
    model.setup();
    // No init declared: zeros, so y = 2 * 0 + 1.
    model.forward();
    model.wait();
    CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys::filled(1.0f));
    tsys::set_x(model, "a.x", 1.0f);
    model.forward();
    model.wait();
    CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys::filled(3.0f));
    tsys::set_x(model, "a.x", 4.0f);
    model.forward();
    model.wait();
    CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys::filled(9.0f));
}

TEST_CASE("init: a durable free input is initialized by its owner on the first forward only", "[system][init]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::OneAxpbLog>(default_config());
    model.setup();
    CHECK(tsys::count_of("a") == 0);  // nothing is allocated or initialized at setup
    tsys::expect_throws([&] { (void)model.view_input<DType::f32, 1>("a.x"); }, "no data until its owner");
    tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("a.y"); }, "no data until its owner");
    model.forward();
    model.forward();
    model.forward();
    model.wait();
    CHECK(tsys::count_of("a") == 1);  // allocated once, initialized once, never released
    CHECK(to_vector(model.view_input<DType::f32, 1>("a.x")) == tsys::filled(5.0f));
    CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys::filled(11.0f));  // 2 * 5 + 1
}

TEST_CASE("init: Model::init_input before the first forward beats the declared init", "[system][init]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::OneAxpbLog>(default_config());
    model.setup();
    tsys::set_x(model, "a.x", 3.0f);  // the owner applies it at its allocation, in place of the declared init
    model.forward();
    model.wait();
    CHECK(tsys::count_of("a") == 0);  // the declared (logging) init never ran
    CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys::filled(7.0f));  // 2 * 3 + 1
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
// An eternal root whose partners include a transient or finite one is not arena-eligible: it lives in the
// pool, one node here, for as long as the model.
const std::size_t kEternalNode = cst::pooled_config().pool_node_size;
}  // namespace

TEST_CASE("transient: release after the last user, kept outputs, back-to-back forwards", "[system][transient]") {
    cs::Model model = cs::create_model<tsys::TopC>(cst::pooled_config());
    model.setup();
    auto&             tr   = dynamic_cast<cs::System&>(model.get_subsystem("tr"));
    auto&             sys  = model.root();
    const std::size_t base = pool_in_use(model);
    CHECK(tr.requested_run() == -2);

    SECTION("host access to a released root throws, naming who released it") {
        model.forward();
        model.wait();
        CHECK(sys.entry_at("tr.prod.y").root()->lifetime_count() == 0);  // out of scope again
        CHECK(sys.entry_at("tr.prod.y").root()->released());
        CHECK(sys.entry_at("tr.mid.y").root()->released());
        CHECK(!sys.entry_at("tr.mid.y").root()->has_storage());
        CHECK(pool_in_use(model) == base + kEternalNode);  // everything the subtree allocated is back, except the eternal output
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("tr.mid.y"); }, "tr");
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("tr.prod.y"); },
                            "was released after its last use (by 'tr.mid')");
    }

    SECTION("an output connected outside the subtree stays durable") {
        model.forward();
        model.wait();
        auto* cons = sys.entry_at("tr.cons.y").root();
        CHECK(cons->has_storage());
        CHECK(!cons->released());
        CHECK(cons->lifetime_count() == -1);                                   // the sink outside answers -1: eternal
        CHECK(sys.entry_at("tr.mid.y").root()->lifetime_count() == 0);       // the intermediates went out of scope
        CHECK(to_vector(model.view_output<DType::f32, 1>("tr.cons.y")) == want_cons());
        CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == want_sink());
    }

    SECTION("a second forward allocates again and gives the same values") {
        model.forward();
        model.wait();
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("tr.cons.y")) == want_cons());
        CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == want_sink());
        CHECK(pool_in_use(model) == base + kEternalNode);
    }

    SECTION("three forwards enqueued back to back, then one wait") {
        model.forward();
        model.forward();
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("tr.cons.y")) == want_cons());
        CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == want_sink());
        CHECK(pool_in_use(model) == base + kEternalNode);
    }
}

TEST_CASE("transient: sequential Components A then B share a root, released after B", "[system][transient]") {
    cs::Model model = cs::create_model<tsys::TransComps>(cst::pooled_config());
    model.setup();
    auto&             sys  = model.root();
    const std::size_t base = pool_in_use(model);

    model.forward();
    model.wait();
    CHECK(sys.entry_at("a.y").root()->lifetime_count() == 0);   // A's output, read by transient B only: out of scope
    CHECK(sys.entry_at("b.y").root()->lifetime_count() == -1);  // B's output is read by the plain C: eternal
    CHECK(sys.entry_at("a.y").root()->released_by() == "b");    // B is its last partner
    CHECK(sys.entry_at("a.y").root()->released());
    CHECK(pool_in_use(model) == base + kEternalNode);  // b.y (read by the plain C) is the eternal one
    tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("a.y"); }, "was released after its last use (by 'b')");
    // c = 1 * (2 * (i + 1) + 1) + 0.5
    CHECK(to_vector(model.view_output<DType::f32, 1>("c.y")) == tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.5f));
}

TEST_CASE("transient: a root read by a transient and a non-transient Component stays durable", "[system][transient]") {
    cs::Model model = cs::create_model<tsys::SharedRead>(cst::pooled_config());
    model.setup();
    model.forward();
    model.wait();
    CHECK(model.root().entry_at("p.y").root()->lifetime_count() == -1);  // the plain reader answers -1
    CHECK(to_vector(model.view_output<DType::f32, 1>("p.y")) == tsys::iota_plus(1.0f, tsys::kLen));
}

TEST_CASE("transient: promotion inside a transient stays transient; re-promoted above it is durable",
          "[system][transient]") {
    SECTION("promoted inside the transient subsystem only") {
        cs::Model model = cs::create_model<tsys::PromoInTrans>(cst::pooled_config(), false);
        model.setup();
        model.forward();
        model.wait();
        CHECK(model.root().entry_at("tr.y").root()->lifetime_count() == 0);
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("tr.y"); }, "was released after its last use (by 'tr.p')");
    }
    SECTION("re-promoted at a non-transient level") {
        cs::Model model = cs::create_model<tsys::PromoInTrans>(cst::pooled_config(), true);
        model.setup();
        model.forward();
        model.wait();
        CHECK(model.root().entry_at("tr.y").root()->lifetime_count() == -1);  // the root System's promotion answers -1
        CHECK(to_vector(model.view_output<DType::f32, 1>("out")) == tsys::iota_plus(1.0f, tsys::kLen));
    }
}

TEST_CASE("transient: nested transients, the outer one is named", "[system][transient]") {
    cs::Model model = cs::create_model<tsys::NestedTrans>(cst::pooled_config());
    model.setup();
    const std::size_t base = pool_in_use(model);
    model.forward();
    model.wait();
    CHECK(pool_in_use(model) == base);
    tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("outer.inner.cons.y"); },
                        "was released after its last use");
}

TEST_CASE("transient: a free input inside a transient subtree is re-initialized each forward", "[system][transient]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::TransFreeInput>(cst::pooled_config());
    model.setup();
    CHECK(tsys::count_of("tr.p") == 0);  // a transient root has no storage at setup
    model.forward();
    model.forward();
    model.forward();
    model.wait();
    CHECK(tsys::count_of("tr.p") == 3);
    CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == tsys::filled(15.0f));  // 2 * 7 + 1
    tsys::expect_throws([&] { (void)model.view_input<DType::f32, 1>("tr.p.x"); },
                        "was released after its last use (by 'tr.p')");
}

// =============================================================================================
// 6b. run-N (property::run_n on add_subsystem; do-once is n = 1)
// =============================================================================================
TEST_CASE("run-N: n = 1 runs once; a later init_input stays pending (the owner does not run again)", "[system][runn]") {
    cs::Model model = cs::create_model<tsys::RunNTop>(default_config(), std::size_t{1});
    model.setup();

    tsys::set_x(model, "op.a.x", 1.0f);
    model.forward();
    model.wait();
    CHECK(to_vector(model.view_output<DType::f32, 1>("op.a.y")) == tsys::filled(2.0f));

    tsys::set_x(model, "op.a.x", 5.0f);
    model.forward();
    model.wait();
    // The owner applies a pending init in its own run; it did not run, so nothing changed. Its input
    // root was freed after the single run (no eternal partner); the output is kept by the sink.
    tsys::expect_throws([&] { (void)model.view_input<DType::f32, 1>("op.a.x"); },
                        "was released after its last use (by 'op.a')");
    CHECK(to_vector(model.view_output<DType::f32, 1>("op.a.y")) == tsys::filled(2.0f));
}

TEST_CASE("run-N: n = 3 follows the input for three forwards, then holds", "[system][runn]") {
    cs::Model model = cs::create_model<tsys::RunNTop>(default_config(), std::size_t{3});
    model.setup();

    const float xs[5] = {1.0f, 2.0f, 4.0f, 7.0f, 9.0f};
    for (int k = 0; k < 5; ++k) {
        INFO("forward " << k + 1);
        tsys::set_x(model, "op.a.x", xs[k]);
        model.forward();
        model.wait();
        const float held = k < 3 ? xs[k] : xs[2];  // the 3rd value from then on
        CHECK(to_vector(model.view_output<DType::f32, 1>("op.a.y")) == tsys::filled(2.0f * held));
    }
}

TEST_CASE("run-N subtree: its roots live for the n runs and are freed after the last, unless an eternal partner keeps them",
          "[system][runn]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::RunNTransTop>(cst::pooled_config(), std::size_t{2});
    model.setup();
    auto&             sys  = model.root();
    const std::size_t base = tsys::pool_in_use_of(model);
    CHECK(tsys::count_of("op.prod") == 0);  // nothing is allocated at setup

    // run 1: the roots come into scope (counts 2 and 4), the free input is initialized once; the consumer's
    // output feeds the sink outside (eternal), the rest stays allocated for run 2
    model.forward();
    model.wait();
    CHECK(tsys::count_of("op.prod") == 1);
    CHECK(sys.entry_at("op.mid.y").root()->has_storage());
    CHECK(sys.entry_at("op.mid.y").root()->lifetime_count() > 0);
    CHECK(sys.entry_at("op.cons.y").root()->lifetime_count() == -1);
    CHECK(to_vector(model.view_output<DType::f32, 1>("op.cons.y")) == tsys::filled(11.0f));  // 2 * 4 + 3
    CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == tsys::filled(11.5f));
    const std::size_t after1 = tsys::pool_in_use_of(model);
    CHECK(after1 > base);

    // run 2 is the last: the subtree's roots are freed (one wait), the eternal one stays; the init is not applied again
    model.forward();
    model.wait();
    CHECK(tsys::count_of("op.prod") == 1);
    CHECK(!sys.entry_at("op.mid.y").root()->has_storage());
    CHECK(sys.entry_at("op.mid.y").root()->lifetime_count() == 0);
    CHECK(to_vector(model.view_output<DType::f32, 1>("op.cons.y")) == tsys::filled(11.0f));
    const std::size_t after2 = tsys::pool_in_use_of(model);
    CHECK(after2 < after1);

    // forward 3: past n, nothing runs, so nothing is allocated, initialized or released
    model.forward();
    model.wait();
    CHECK(tsys::count_of("op.prod") == 1);
    CHECK(!sys.entry_at("op.mid.y").root()->has_storage());
    CHECK(to_vector(model.view_output<DType::f32, 1>("op.cons.y")) == tsys::filled(11.0f));  // unchanged
    CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == tsys::filled(11.5f));
    CHECK(tsys::pool_in_use_of(model) == after2);

    // a view of the released mid root throws, naming who released it
    CHECK(sys.entry_at("op.mid.y").root()->released());
    tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("op.mid.y"); },
                        "was released after its last use (by 'op.cons')");
    model.wait();
}

TEST_CASE("run-N subtree: n = 3, the roots stay for runs 1..3 and the pool drops after the third", "[system][runn]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::RunNTransTop>(cst::pooled_config(), std::size_t{3});
    model.setup();
    std::size_t held = 0, freed = 0;
    for (int k = 1; k <= 5; ++k) {
        INFO("forward " << k);
        model.forward();
        model.wait();
        const std::size_t now = tsys::pool_in_use_of(model);
        if (k == 1) held = now;
        if (k <= 3 && k >= 1 && k < 3) CHECK(now == held);  // runs 1 and 2: everything stays allocated
        if (k == 3) {
            freed = now;
            CHECK(freed < held);
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
    s.add_subsystem<tsys::RunCounted>("p", pp, std::string{"p"}, [cp](cs::System& p) {
        p.add_subsystem<tsys::RunCounted>("c", cp, std::string{"c"});
    });
}
void forwards(cs::Model& model, int n) {
    for (int k = 0; k < n; ++k) model.forward();
    model.wait();
}
}  // namespace

TEST_CASE("run_policy: run_n(3) runs exactly 3 forwards, then is skipped", "[system][run_policy]") {
    std::string err;
    cs::Model   model = policy_model(
        [](cs::System& s) {
            s.add_subsystem<tsys::RunCounted>("a", cs::property_list{cs::property::run_n{3}}, std::string{"a"});
        },
        err);
    CHECK(err.empty());
    for (int k = 1; k <= 5; ++k) {
        INFO("forward " << k);
        forwards(model, 1);
        CHECK(tsys::runs_of("a") == static_cast<std::size_t>(k <= 3 ? k : 3));
    }
    CHECK(model.get_subsystem("a").run_counter() == 0);
    CHECK(model.get_subsystem("a").requested_run() == 3);
}

TEST_CASE("run_policy: do_once runs once", "[system][run_policy]") {
    std::string err;
    cs::Model   model = policy_model(
        [](cs::System& s) {
            s.add_subsystem<tsys::RunCounted>("a", cs::property_list{cs::property::do_once{}}, std::string{"a"});
        },
        err);
    forwards(model, 4);
    CHECK(tsys::runs_of("a") == 1);
    CHECK(model.get_subsystem("a").requested_run() == 1);
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
    SECTION("run_n(5) under run_n(2) runs 2 times and warns") {
        cs::Model model = policy_model(
            [](cs::System& s) {
                add_pair(s, cs::property_list{run_n{2}}, cs::property_list{run_n{5}});
            },
            err);
        INFO("stderr: " << err);
        CHECK_THAT(err, Catch::Matchers::ContainsSubstring(
                            "naga::cospan: warning: 'p.c' asks to run 5 time(s) but 'p' allows 2; clamped to 2"));
        forwards(model, 5);
        CHECK(tsys::runs_of("c") == 2);
        CHECK(model.get_subsystem("p.c").requested_run() == 5);
    }
    SECTION("run_n(5) under transient becomes transient: every forward, and warns") {
        cs::Model model = policy_model(
            [](cs::System& s) {
                add_pair(s, cs::property_list{transient{}}, cs::property_list{run_n{5}});
            },
            err);
        INFO("stderr: " << err);
        CHECK_THAT(err, Catch::Matchers::ContainsSubstring("'p.c' asks to run 5 time(s) but 'p' allows transient "
                                                           "(every run, revived); clamped to"));
        CHECK(model.get_subsystem("p.c").run_counter() == -2);
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
        CHECK(model.get_subsystem("p.c").run_counter() == -2);
        forwards(model, 6);
        CHECK(tsys::runs_of("p") == 3);
        CHECK(tsys::runs_of("c") == 3);
        CHECK(model.get_subsystem("p.c").run_counter() == -2);
    }
}

TEST_CASE("run_policy: rearm re-enables an exhausted run_n parent and its descendants", "[system][run_policy]") {
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
    CHECK(model.get_subsystem("p").run_counter() == 0);

    model.get_subsystem("p").rearm();
    CHECK(model.get_subsystem("p").run_counter() == 2);
    CHECK(model.get_subsystem("p.c").run_counter() == 2);
    forwards(model, 4);
    CHECK(tsys::runs_of("p") == 4);
    CHECK(tsys::runs_of("c") == 4);
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
        tsys::expect_throws([&] { sys.promote<DType::f32, 1>("a.y", "y"); }, "setup() only");
        tsys::expect_throws([&] { sys.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::alias::fixed(1)); }, "setup() only");
        tsys::expect_throws([&] { sys.add_design_var("a.x"); }, "setup() only");
        model.wait();  // setup() submitted initializer work
    }
    SECTION("an unknown path in view_output names the path") {
        cs::Model model = cs::create_model<tsys::ThreeAxpb>(default_config());
        model.setup();
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("a.nope"); }, "a.nope");
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("zz.y"); }, "zz");
        // The kind is checked too.
        tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("a.x"); }, "is an input");
        tsys::expect_throws([&] { (void)model.view_input<DType::f32, 1>("a.y"); }, "is an output");
        model.wait();  // setup() submitted initializer work
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


// =============================================================================================
// 7. ties own the shape; promotions mirror live; rank-templated aliases
// =============================================================================================
namespace tsys2 {

// bins (VBins): a free u32 input n and an output b of n floats (b[i] = i + 1); a VALUE-driven output.
struct VBinsBody {
    struct Inputs {
        cs::Input<DType::u32, 1> n;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> b;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.b[i] = static_cast<float>(i[0]) + 1.0f; }
};
class VBins : public cs::Component<VBinsBody> {
    cs::Input<DType::u32, 1>  n_;
    cs::Output<DType::f32, 1> b_;

public:
    void setup() override {
        declare_input(n_, "n", 1);
        declare_output(b_, "b", cs::FromInput{"n", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(b_.shape(), {n_}, {b_}, {});
    }
};

// NCopy: n[0] = v[0], the bin count as an OUTPUT (an alias shape reads an upstream output).
struct NCopyBody {
    struct Inputs {
        cs::Input<DType::u32, 1> v;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> n;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.n[i] = in.v[i]; }
};
class NCopy : public cs::Component<NCopyBody> {
    cs::Input<DType::u32, 1>  v_;
    cs::Output<DType::u32, 1> n_;

public:
    void setup() override {
        declare_input(v_, "v", 1);
        declare_output(n_, "n", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(n_.shape(), {v_}, {n_}, {});
    }
};

// TScale: y = 2 x; x is rank-only (a tie member), y follows x's shape.
struct TScaleBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = 2.0f * in.x[i]; }
};
class TScale : public cs::Component<TScaleBody> {
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

// TIota: o[i] = i + 1; o is rank-only (a tie_outputs member).
struct TIotaBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> o;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.o[i] = static_cast<float>(i[0]) + 1.0f; }
};
class TIota : public cs::Component<TIotaBody> {
    cs::Output<DType::f32, 1> o_;

public:
    void setup() override { declare_output(o_, "o", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(o_.shape(), {}, {o_}, {});
    }
};

/// A root System whose setup() runs a callable.
class FnSys : public cs::System {
    std::function<void(cs::System&)> build_;

public:
    explicit FnSys(std::function<void(cs::System&)> b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

inline cs::Model make(std::function<void(cs::System&)> build, cs::ModelConfig cfg = default_config()) {
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
        add_subsystem<TScale>("a");
        add_subsystem<TScale>("b");
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::alias::fixed(tsys::kLen));
    }
};
class TransTieTop : public cs::System {
public:
    void setup() override { add_subsystem<TieTwoScales>("tr", tsys::kTransient); }
};

}  // namespace tsys2

TEST_CASE("D139-2: a promoted value-driven output shows its member's current shape", "[system][alias][promote]") {
    auto model = tsys2::make([](cs::System& s) {
        s.add_subsystem<tsys2::VBins>("v");
        s.promote<DType::f32, 1>("v.b", "b");
        s.promote<DType::u32, 1>("v.n", "n");
    });
    model.setup();
    tsys::expect_throws([&] { (void)model.view_output<DType::f32, 1>("b"); }, "no data until its owner");
    for (const std::uint32_t n : {3u, 7u, 2u}) {  // the size changes between runs
        INFO("n = " << n);
        model.init_input("n", cs::init::from_host(tsys2::one_u32(n)));
        model.forward();
        model.wait();
        auto v = model.view_output<DType::f32, 1>("b");
        CHECK(v.shape()[0] == n);
        CHECK(v.size() == n);
        CHECK(to_vector(v) == tsys2::one_to(n));
    }
}

TEST_CASE("D139-2: views of an alias before its root's owner has run throw", "[system][alias]") {
    cs::Model model = cs::create_model<tsys2::TieTwoScales>(default_config());
    model.setup();
    tsys::expect_throws([&] { (void)model.view_input<DType::f32, 1>("x"); }, "no data until its owner");
    tsys::expect_throws([&] { (void)model.view_input<DType::f32, 1>("a.x"); }, "no data until its owner");
    model.forward();
    model.wait();
    CHECK(model.view_input<DType::f32, 1>("x").shape()[0] == tsys::kLen);
    CHECK(model.view_input<DType::f32, 1>("a.x").shape()[0] == tsys::kLen);
}

TEST_CASE("D139-2: tie errors", "[system][alias][errors]") {
    const auto setup_throws = [](std::function<void(cs::System&)> build, const std::string& needle) {
        tsys::expect_throws(
            [&] {
                auto model = tsys2::make(build);
                model.setup();
            },
            needle);
    };
    const auto two_scales = [](cs::System& s) {
        s.add_subsystem<tsys2::TScale>("a");
        s.add_subsystem<tsys2::TScale>("b");
    };
    SECTION("a tie of one member") {
        setup_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.tie_inputs<DType::f32, 1>({"a.x"}, "t", cs::alias::fixed(4));
            },
            "a tie needs two or more members; use promote for one");
    }
    SECTION("dtype mismatch") {
        setup_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.tie_inputs<DType::f64, 1>({"a.x", "b.x"}, "t", cs::alias::fixed(4));
            },
            "dtype");
    }
    SECTION("rank mismatch") {
        setup_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.tie_inputs<DType::f32, 2>({"a.x", "b.x"}, "t", cs::alias::fixed(4, 4));
            },
            "rank");
        setup_throws(
            [&](cs::System& s) {
                s.add_subsystem<tsys::Axpb>("a", 1.0f, 0.0f);
                s.promote<DType::f32, 2>("a.x", "ax");
            },
            "rank");
    }
    SECTION("a tied member whose fixed dim contradicts the tie") {  // a member may declare dims; they are constraints
        setup_throws(
            [&](cs::System& s) {
                s.add_subsystem<tsys::Axpb>("a", 1.0f, 0.0f);  // x is fixed kLen
                s.add_subsystem<tsys::Axpb>("b", tsys::tied_x, 1.0f, 0.0f);
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::alias::fixed(tsys::kLen + 1));
            },
            "shape mismatch: 'a.x' dimension 0 is fixed");
    }
    SECTION("an unspecified free input nothing supplies") {
        setup_throws([&](cs::System& s) { s.add_subsystem<tsys2::TScale>("a"); }, "free input 'a.x' has an unspecified dimension");
    }
    SECTION("an output alias cannot have an unspecified shape") {
        setup_throws(
            [&](cs::System& s) {
                s.add_subsystem<tsys2::TIota>("w1");
                s.add_subsystem<tsys2::TIota>("w2");
                // not expressible with the public factories (tie_outputs requires a shape); the internal default is the case
                s.tie_outputs<DType::f32, 1>({"w1.o", "w2.o"}, "t", cs::detail::unspecified_alias_shape(),
                                             cs::TieOutputsPolicy::allow_plain);
            },
            "an output alias needs a shape");
    }
    SECTION("an unspecified tie_inputs alias nothing supplies") {
        setup_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t");
            },
            "nothing supplies it");
    }
    SECTION("from_output_values names a later-registered producer") {
        setup_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.add_subsystem<tsys2::NCopy>("nb");  // registered AFTER the tie's members
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::alias::from_output_values("nb.n"));
            },
            "not registered before");
    }
    SECTION("from_output_shape names a later-registered producer") {
        setup_throws(
            [&](cs::System& s) {
                two_scales(s);
                s.add_subsystem<tsys2::VBins>("v");
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::alias::from_output_shape("v.b"));
            },
            "not registered before");
    }
    SECTION("an alias shape source that is not an output") {
        setup_throws(
            [&](cs::System& s) {
                s.add_subsystem<tsys2::VBins>("v");
                two_scales(s);
                s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "t", cs::alias::from_output_shape("v.n"));
            },
            "not an output");
    }
}

TEST_CASE("D139-2: tie_inputs shapes: fixed, from_output_shape, from_output_values", "[system][alias][tie]") {
    SECTION("fixed") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tsys2::TScale>("a");
            s.add_subsystem<tsys2::TScale>("b");
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::alias::fixed(5));
        });
        model.setup();
        model.init_input("x", cs::init::from_host(tsys2::one_to(5)));
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys2::doubled(tsys2::one_to(5)));
        CHECK(to_vector(model.view_output<DType::f32, 1>("b.y")) == tsys2::doubled(tsys2::one_to(5)));
    }
    SECTION("from_output_shape: the shape follows the upstream output between runs") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tsys2::VBins>("v");
            s.add_subsystem<tsys2::TScale>("a");
            s.add_subsystem<tsys2::TScale>("b");
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::alias::from_output_shape("v.b"));
        });
        model.setup();
        for (const std::uint32_t n : {3u, 6u, 4u}) {
            INFO("n = " << n);
            model.init_input("v.n", cs::init::from_host(tsys2::one_u32(n)));
            model.forward();
            model.wait();
            CHECK(model.view_input<DType::f32, 1>("x").shape()[0] == n);
            CHECK(model.view_input<DType::f32, 1>("a.x").shape()[0] == n);
            CHECK(to_vector(model.view_output<DType::f32, 1>("b.y")) == std::vector<float>(n, 0.0f));
        }
    }
    SECTION("from_output_values: the extents are the values of an upstream output") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tsys2::NCopy>("nb");
            s.add_subsystem<tsys2::TScale>("a");
            s.add_subsystem<tsys2::TScale>("b");
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::alias::from_output_values("nb.n"));
        });
        model.setup();
        for (const std::uint32_t n : {4u, 9u, 1u}) {
            INFO("n = " << n);
            model.init_input("nb.v", cs::init::from_host(tsys2::one_u32(n)));
            model.init_input("x", cs::init::fill(1.5f));
            model.forward();
            model.wait();
            CHECK(model.view_input<DType::f32, 1>("a.x").shape()[0] == n);
            CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == std::vector<float>(n, 3.0f));
            CHECK(to_vector(model.view_output<DType::f32, 1>("b.y")) == std::vector<float>(n, 3.0f));
        }
    }
}

TEST_CASE("D139-2: tie_outputs shapes: fixed and from_output_values", "[system][alias][tie]") {
    SECTION("fixed") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tsys2::TIota>("w1");
            s.add_subsystem<tsys2::TIota>("w2");
            s.add_subsystem<tsys::Copy1>("c");
            s.tie_outputs<DType::f32, 1>({"w1.o", "w2.o"}, "t", cs::alias::fixed(6), cs::TieOutputsPolicy::allow_plain);
            s.connect("t", "c.x");
        });
        model.setup();
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("t")) == tsys2::one_to(6));
        CHECK(to_vector(model.view_output<DType::f32, 1>("c.y")) == tsys2::one_to(6));
    }
    SECTION("from_output_values: the size changes between runs") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tsys2::NCopy>("nb");
            s.add_subsystem<tsys2::TIota>("w1");
            s.add_subsystem<tsys2::TIota>("w2");
            s.add_subsystem<tsys::Copy1>("c");
            s.tie_outputs<DType::f32, 1>({"w1.o", "w2.o"}, "t", cs::alias::from_output_values("nb.n"),
                                         cs::TieOutputsPolicy::allow_plain);
            s.connect("t", "c.x");
        });
        model.setup();
        for (const std::uint32_t n : {3u, 8u, 5u}) {
            INFO("n = " << n);
            model.init_input("nb.v", cs::init::from_host(tsys2::one_u32(n)));
            model.forward();
            model.wait();
            CHECK(to_vector(model.view_output<DType::f32, 1>("t")) == tsys2::one_to(n));
            CHECK(to_vector(model.view_output<DType::f32, 1>("c.y")) == tsys2::one_to(n));
        }
    }
}

TEST_CASE("D139-2: a connected tie_inputs alias joins the producer's root", "[system][alias][tie]") {
    const auto build = [](std::optional<cs::AliasShape> shape) {  // nullopt: tie_inputs without a shape (unspecified)
        return [shape](cs::System& s) {
            s.add_subsystem<tsys2::VBins>("v");
            s.add_subsystem<tsys2::TScale>("a");
            s.add_subsystem<tsys2::TScale>("b");
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
        CHECK(model.root().entry_at("a.x").root() == model.root().entry_at("v.b").root());
        for (const std::uint32_t n : {4u, 2u}) {
            INFO("n = " << n);
            model.init_input("v.n", cs::init::from_host(tsys2::one_u32(n)));
            model.forward();
            model.wait();
            CHECK(model.view_input<DType::f32, 1>("a.x").shape()[0] == n);
            CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys2::doubled(tsys2::one_to(n)));
            CHECK(to_vector(model.view_output<DType::f32, 1>("b.y")) == tsys2::doubled(tsys2::one_to(n)));
        }
    }
    SECTION("a mismatching fixed AliasShape errors on the first run") {
        auto model = tsys2::make(build(cs::alias::fixed(5)));
        model.setup();
        model.init_input("v.n", cs::init::from_host(tsys2::one_u32(4)));
        tsys::expect_throws(
            [&] {
                model.forward();
                model.wait();
            },
            "declares the shape");
        model.wait();
    }
    SECTION("a matching fixed AliasShape runs") {
        auto model = tsys2::make(build(cs::alias::fixed(4)));
        model.setup();
        model.init_input("v.n", cs::init::from_host(tsys2::one_u32(4)));
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys2::doubled(tsys2::one_to(4)));
    }
    SECTION("a model init on the alias warns and is skipped") {
        auto model = tsys2::make(build(std::nullopt));
        model.setup();
        std::ostringstream captured;
        std::streambuf*    old = std::cerr.rdbuf(captured.rdbuf());
        try {
            model.init_input("x", cs::init::fill(9.0f));
        } catch (...) {
            std::cerr.rdbuf(old);
            throw;
        }
        std::cerr.rdbuf(old);
        CHECK(captured.str() ==
              "naga::cospan: warning: input 'x' is connected to an output; your initializer may not run\n");
        model.init_input("v.n", cs::init::from_host(tsys2::one_u32(3)));
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys2::doubled(tsys2::one_to(3)));  // not 9
    }
}

TEST_CASE("D139-2: a tie alias inside a transient subsystem is released and the pool returns to its baseline",
          "[system][alias][transient]") {
    cs::Model model = cs::create_model<tsys2::TransTieTop>(cst::pooled_config());
    model.setup();
    const std::size_t base = tsys::pool_in_use_of(model);
    for (int round = 0; round < 2; ++round) {
        INFO("forward " << round + 1);
        model.forward();
        model.wait();
        auto& sys = model.root();
        CHECK(sys.entry_at("tr.x").root()->lifetime_count() == 0);
        CHECK(sys.entry_at("tr.x").root()->released());
        CHECK(!sys.entry_at("tr.x").root()->has_storage());
        CHECK(tsys::pool_in_use_of(model) == base);
        tsys::expect_throws([&] { (void)model.view_input<DType::f32, 1>("tr.x"); }, "was released after its last use");
    }
}

// =============================================================================================
// 6d. lifetime by counting
// =============================================================================================
TEST_CASE("lifetime: a bare transient producer and a transient reader are freed every run, values right on run 2", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::TransChain>(cst::pooled_config());
    model.setup();
    auto&             sys  = model.root();
    const std::size_t base = pool_in_use(model);
    for (int run = 1; run <= 3; ++run) {
        INFO("run " << run);
        model.forward();
        model.wait();
        CHECK(sys.entry_at("p.y").root()->released());
        CHECK(sys.entry_at("p.y").root()->lifetime_count() == 0);
        CHECK(sys.entry_at("t.y").root()->lifetime_count() == -1);
        CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 1.0f));
        CHECK(pool_in_use(model) == base + kEternalNode);
    }
}

TEST_CASE("lifetime: transient parent forces a run_n(5) child to transient", "[system][lifetime]") {
    tsys::g_inits.clear();
    std::string err;
    cs::Model   model = policy_model([](cs::System& s) { s.add_subsystem<tsys::TransRunNKid>("tr", tsys::kTransient); }, err);
    CHECK_THAT(err, Catch::Matchers::ContainsSubstring("'tr.c' asks to run 5 time(s) but 'tr' allows transient"));
    CHECK(model.get_subsystem("tr.c").run_counter() == -2);
    forwards(model, 7);
    // transient: its free input root is revived, allocated and initialized again at every forward
    CHECK(tsys::count_of("tr.c") == 7);
}

TEST_CASE("lifetime: a run_n(3) parent's roots are freed after the third run unless an eternal partner keeps them",
          "[system][lifetime]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::RunNTransTop>(cst::pooled_config(), std::size_t{3});
    model.setup();
    auto& sys = model.root();
    for (int k = 1; k <= 4; ++k) {
        INFO("forward " << k);
        model.forward();
        model.wait();
        const bool alive = sys.entry_at("op.mid.y").root()->has_storage();
        CHECK(alive == (k < 3));  // freed by the third run, not before
        CHECK(sys.entry_at("op.cons.y").root()->has_storage());  // the sink outside: eternal
    }
    CHECK(sys.entry_at("op.mid.y").root()->released_by() == "op.cons");
    CHECK(tsys::count_of("op.prod") == 1);
}

TEST_CASE("lifetime: a do_once producer with an eternal reader is eternal; its value is held across runs", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::DoOnceProducer>(cst::pooled_config(), false);
    model.setup();
    for (int run = 1; run <= 3; ++run) {
        INFO("run " << run);
        model.forward();
        model.wait();
        CHECK(model.get_subsystem("p").run_counter() == 0);
        CHECK(model.root().entry_at("p.y").root()->lifetime_count() == -1);
        CHECK(model.root().entry_at("p.y").root()->has_storage());
        CHECK(to_vector(model.view_output<DType::f32, 1>("r.y")) == tsys::iota_plus(1.0f, tsys::kLen, 2.0f, 0.0f));
    }
}

TEST_CASE("lifetime: a do_once producer with a transient reader fails on run 2, naming the root, the producer and the reader",
          "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::DoOnceProducer>(default_config(), true);
    model.setup();
    model.forward();  // run 1: both run, the root is freed after them
    model.wait();
    CHECK(model.root().entry_at("p.y").root()->lifetime_count() == 0);
    const auto again = [&] { model.forward(); };
    tsys::expect_throws(again, "'p.y' is revived by 't'");
    tsys::expect_throws(again, "its producer 'p' has stopped");
    tsys::expect_throws(again, "promote it, or don't mark its holder transient");
    model.wait();
}

TEST_CASE("lifetime: a tie_outputs root under a transient System is freed every run, values right", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::TransTieOutTop>(cst::pooled_config());
    model.setup();
    const std::size_t base = pool_in_use(model);
    for (int run = 1; run <= 2; ++run) {
        INFO("run " << run);
        model.forward();
        model.wait();
        auto* t = model.root().entry_at("tr.t").root();
        CHECK(t->owner() == &model.get_subsystem("tr"));  // a tie root is owned by its declaring System
        CHECK(t->released());
        CHECK(t->lifetime_count() == 0);
        CHECK(to_vector(model.view_output<DType::f32, 1>("sink.y")) == tsys::iota_plus(1.0f, tsys::kLen));
        CHECK(pool_in_use(model) == base + kEternalNode);  // rd.y, kept by the sink
    }
}

TEST_CASE("lifetime: a tie_inputs root is initialized by the tie; the members' own inits are ignored", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::TieInit>(default_config());
    model.setup();
    model.forward();
    model.wait();
    CHECK(model.root().entry_at("x").root()->owner() == &model.root());  // the tie root: owned by the declaring System
    CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tsys::filled(6.0f));  // 2 * 3, not 2 * 100
    CHECK(to_vector(model.view_output<DType::f32, 1>("b.y")) == tsys::filled(9.0f));  // 3 * 3, not 3 * 200
}

TEST_CASE("lifetime: a model-level promotion is eternal", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::PromoteFromTransient>(cst::pooled_config());
    model.setup();
    for (int run = 1; run <= 2; ++run) {
        model.forward();
        model.wait();
        CHECK(model.root().entry_at("p.y").root()->lifetime_count() == -1);
        CHECK(to_vector(model.view_output<DType::f32, 1>("y")) == tsys::iota_plus(1.0f, tsys::kLen));
    }
}

TEST_CASE("lifetime: a marked root under a transient System does not throw and is eternal", "[system][lifetime]") {
    cs::Model model = cs::create_model<tsys::MarkedInTransient>(cst::pooled_config());
    CHECK_NOTHROW(model.setup());
    for (int run = 1; run <= 2; ++run) {
        model.forward();
        model.wait();
        CHECK(model.root().entry_at("tr.a.x").root()->lifetime_count() == -1);
        CHECK(model.root().entry_at("tr.a.x").root()->has_storage());
    }
}

TEST_CASE("lifetime: rearm sends a do_once's roots back out of scope and they are initialized again", "[system][lifetime]") {
    tsys::g_inits.clear();
    cs::Model model = cs::create_model<tsys::DoOnceInit>(cst::pooled_config());
    model.setup();
    forwards(model, 3);
    CHECK(tsys::count_of("p") == 1);
    CHECK(to_vector(model.view_output<DType::f32, 1>("r.y")) == tsys::filled(8.0f));  // 2 * (1 * 4)
    CHECK(model.get_subsystem("p").run_counter() == 0);

    model.get_subsystem("p").rearm();
    CHECK(model.get_subsystem("p").run_counter() == 1);
    CHECK(model.root().entry_at("p.y").root()->lifetime_count() == 0);
    CHECK_FALSE(model.root().entry_at("p.y").root()->has_storage());
    forwards(model, 3);
    CHECK(tsys::count_of("p") == 2);  // p.x was allocated and initialized again
    CHECK(to_vector(model.view_output<DType::f32, 1>("r.y")) == tsys::filled(8.0f));
    CHECK(model.get_subsystem("p").run_counter() == 0);
}

TEST_CASE("lifetime: transient together with run_n or do_once throws, naming the subsystem", "[system][lifetime]") {
    std::string err;
    tsys::expect_throws(
        [&] {
            (void)policy_model(
                [](cs::System& s) {
                    s.add_subsystem<tsys::RunCounted>("a", cs::property_list{cs::property::transient{}, cs::property::run_n{2}},
                                                      std::string{"a"});
                },
                err);
        },
        "add_subsystem('a'): transient cannot be combined with run_n or do_once");
    tsys::expect_throws(
        [&] {
            (void)policy_model(
                [](cs::System& s) {
                    s.add_subsystem<tsys::RunCounted>("b", cs::property_list{cs::property::do_once{}, cs::property::transient{}},
                                                      std::string{"b"});
                },
                err);
        },
        "add_subsystem('b'): transient cannot be combined with run_n or do_once");
}

namespace tsys2 {
/// A transient System: nb (NCopy, do_once) gives the tie's extent by VALUE (alias::from_output_values), a and b are the members.
class ValueSourceSys : public cs::System {
public:
    void setup() override {
        add_subsystem<NCopy>("nb", cs::property_list{cs::property::do_once{}});
        add_subsystem<TScale>("a");
        add_subsystem<TScale>("b");
        tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::alias::from_output_values("nb.n"));
    }
};
}  // namespace tsys2

TEST_CASE("lifetime: a value-driven alias shape keeps a finite producer's output alive until it has materialized",
          "[system][lifetime][alias]") {
    auto model = tsys2::make([](cs::System& s) { s.add_subsystem<tsys2::ValueSourceSys>("s", tsys::kTransient); },
                             cst::pooled_config());
    model.setup();
    model.init_input("s.nb.v", cs::init::from_host(tsys2::one_u32(4)));
    // nb.n is written by the do_once producer (answer 1) and read by the tie's declaring System (a non-writer partner, 1):
    // the producer's own decrement does not free it, the alias's does, after the value was read.
    CHECK_NOTHROW(model.forward());
    model.wait();
    auto* n = model.root().entry_at("s.nb.n").root();
    CHECK(n->released());
    CHECK(n->released_by() == "s");
    CHECK(model.root().entry_at("s.x").root()->owner() == &model.get_subsystem("s"));
    CHECK(model.root().entry_at("s.x").root()->extents()[0] == 4);
}

// =============================================================================================
// 6e. shape authority: unspecified dims are supplied from outside; specified dims are constraints
// =============================================================================================
namespace tshape {

// ---- fixtures -------------------------------------------------------------------------------
/// Scale2: y = 2 x, rank 2; x is wholly unspecified, y follows it.
struct S2Body {
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    void operator()(sycl::id<2> i, Inputs in, Outputs out) const { out.y[i] = 2.0f * in.x[i]; }
};
class Scale2 : public cs::Component<S2Body> {
    cs::Input<DType::f32, 2>  x_;
    cs::Output<DType::f32, 2> y_;

public:
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<2>());
        declare_output(y_, "y", cs::from_input_shape<2>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {x_}, {y_}, {}); }
};

/// An output whose dimension is unspecified and that no tie owns (a setup error).
struct OutBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = 1.0f; }
};
class OutUnspec : public cs::Component<OutBody> {
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override { declare_output(y_, "y", cs::unspecified); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {}, {y_}, {}); }
};

/// FixedOut(n): y[i] = i + 1, y fixed n.
struct FixedOutBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = static_cast<float>(i[0] + 1); }
};
class FixedOut : public cs::Component<FixedOutBody> {
    std::size_t               n_;
    cs::Output<DType::f32, 1> y_;

public:
    explicit FixedOut(std::size_t n) : n_{n} {}
    void setup() override { declare_output(y_, "y", static_cast<int>(n_)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {}, {y_}, {}); }
};

/// Pair: inputs a (unspecified) and b (declared INHERITED from a: FromInputDim{a, 0}); a fixed output of one element.
struct PairBody {
    struct Inputs {
        cs::Input<DType::f32, 1> a;
        cs::Input<DType::f32, 1> b;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = in.a[i] + in.b[i]; }
};
class Pair : public cs::Component<PairBody> {
    cs::Input<DType::f32, 1>  a_, b_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(a_, "a", cs::unspecified);
        declare_input(b_, "b", cs::FromInputDim{"a", 0});
        declare_output(y_, "y", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {a_, b_}, {y_}, {}); }
};

/// ByVal: input n (u32, one element) and v declared VALUE-DRIVEN: FromInput{n, 0}.
struct ByValBody {
    struct Inputs {
        cs::Input<DType::u32, 1> n;
        cs::Input<DType::f32, 1> v;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = 1.0f; }
};
class ByVal : public cs::Component<ByValBody> {
    cs::Input<DType::u32, 1>  n_;
    cs::Input<DType::f32, 1>  v_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(n_, "n", 1);
        declare_input(v_, "v", cs::FromInput{"n", 0});
        declare_output(y_, "y", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {n_, v_}, {y_}, {}); }
};

/// Forward must throw with every needle; the stream is drained afterwards.
template <class Needles>
void forward_throws(cs::Model& model, const Needles& needles) {
    std::string what = "<no exception>";
    try {
        model.forward();
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
        s.add_subsystem<tsys2::VBins>("v");
        s.add_subsystem<tsys2::TScale>("a");
        s.connect("v.b", "a.x");
    });
    model.setup();
    for (const std::uint32_t n : {5u, 3u}) {  // the supplier changes between runs, the consumer follows
        model.init_input("v.n", cs::init::from_host(tsys2::one_u32(n)));
        model.forward();
        model.wait();
        CHECK(model.view_input<DType::f32, 1>("a.x").shape()[0] == n);
        CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tshape::iota_doubled(n));
    }
}

TEST_CASE("shape: unspecified dims are supplied by a sliced connection", "[system][shape]") {
    auto model = tsys2::make([](cs::System& s) {
        s.add_subsystem<tsys::Src>("p");
        s.add_subsystem<tsys2::TScale>("a");
        s.connect("p.y", "a.x", cs::Slice{cs::subarray<1>({2}, {3})});  // elements 2, 3, 4 = 3, 4, 5
    });
    model.setup();
    model.forward();
    model.wait();
    CHECK(model.view_input<DType::f32, 1>("a.x").shape()[0] == 3);
    CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == std::vector<float>{6.0f, 8.0f, 10.0f});
}

TEST_CASE("shape: unspecified dims are supplied by a tie (an initializer on it, or a connection into it)", "[system][shape]") {
    SECTION("an initializer on the tie") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tsys2::TScale>("a");
            s.add_subsystem<tsys2::TScale>("b");
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x");  // no shape argument: unspecified
            s.init_input("x", cs::init::fill(3.0f, {5}));
        });
        model.setup();
        model.forward();
        model.wait();
        CHECK(model.view_input<DType::f32, 1>("a.x").shape()[0] == 5);
        CHECK(to_vector(model.view_output<DType::f32, 1>("b.y")) == std::vector<float>(5, 6.0f));
    }
    SECTION("a connection into the tie") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tsys2::VBins>("v");
            s.add_subsystem<tsys2::TScale>("a");
            s.add_subsystem<tsys2::TScale>("b");
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x");
            s.connect("v.b", "x");
        });
        model.setup();
        model.init_input("v.n", cs::init::from_host(tsys2::one_u32(4)));
        model.forward();
        model.wait();
        CHECK(to_vector(model.view_output<DType::f32, 1>("a.y")) == tshape::iota_doubled(4));
    }
}

TEST_CASE("shape: every initializer kind supplies the shape of a free unspecified input", "[system][shape]") {
    const auto run = [](cs::InitSpec init, std::size_t want_len) {
        auto model = tsys2::make([init](cs::System& s) {
            s.add_subsystem<tsys2::TScale>("a");
            s.init_input("a.x", init);
        });
        model.setup();
        model.forward();
        model.wait();
        CHECK(model.view_input<DType::f32, 1>("a.x").shape()[0] == want_len);
        return to_vector(model.view_output<DType::f32, 1>("a.y"));
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
        const auto v = run(cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostViewMut<DType::f32, 1>&)>{
                               [](cs::HostViewMut<DType::f32, 1>& h) {
                                   for (std::size_t k = 0; k < h.size(); ++k) h[k] = static_cast<float>(k);
                               }},
                                                              std::vector<std::size_t>{3}),
                           3);
        CHECK(v == std::vector<float>{0.0f, 2.0f, 4.0f});
    }
    SECTION("custom(fn, shape): the shape alone is checked") {
        const auto v = run(cs::init::custom<DType::f32, 1>(
                               std::function<void(sk::queue&, cs::Initializer<DType::f32, 1>)>{[](sk::queue&, cs::Initializer<DType::f32, 1>) {}},
                               std::vector<std::size_t>{6}),
                           6);
        CHECK(v.size() == 6);
    }
    SECTION("rank 2: from_host(data, shape) and fill(v, shape)") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tshape::Scale2>("a");
            s.init_input("a.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}));
        });
        model.setup();
        model.forward();
        model.wait();
        CHECK(model.view_input<DType::f32, 2>("a.x").shape()[0] == 2);
        CHECK(model.view_input<DType::f32, 2>("a.x").shape()[1] == 3);
        CHECK(to_vector(model.view_output<DType::f32, 2>("a.y")) == std::vector<float>{2, 4, 6, 8, 10, 12});
    }
}

TEST_CASE("shape: the two setup errors", "[system][shape]") {
    SECTION("an untied output with an unspecified dim") {
        tsys::expect_throws(
            [&] {
                auto model = tsys2::make([](cs::System& s) { s.add_subsystem<tshape::OutUnspec>("o"); });
                model.setup();
            },
            "output 'o.y' has an unspecified dimension but is not a member of a tie_outputs");
    }
    SECTION("a free input with an unspecified dim and no shape-giving initializer") {
        tsys::expect_throws(
            [&] {
                auto model = tsys2::make([](cs::System& s) { s.add_subsystem<tsys2::TScale>("a"); });
                model.setup();
            },
            "free input 'a.x' has an unspecified dimension and no initializer that gives it a shape");
        tsys::expect_throws(  // fill(v) carries no shape
            [&] {
                auto model = tsys2::make([](cs::System& s) {
                    s.add_subsystem<tsys2::TScale>("a");
                    s.init_input("a.x", cs::init::fill(1.0f));
                });
                model.setup();
            },
            "no initializer that gives it a shape");
    }
}

TEST_CASE("shape: a specified dim is a constraint; a violation names both sides", "[system][shape]") {
    SECTION("fixed, at setup, against a tie") {
        tsys::expect_throws(
            [&] {
                auto model = tsys2::make([](cs::System& s) {
                    s.add_subsystem<tshape::FixedOut>("a", std::size_t{5});
                    s.add_subsystem<tshape::FixedOut>("b", std::size_t{5});
                    s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::alias::fixed(8), cs::TieOutputsPolicy::allow_plain);
                });
                model.setup();
            },
            "shape mismatch: 'a.y' dimension 0 is fixed = 5 but its shape is supplied by 't' (a tie)");
    }
    SECTION("fixed, against a connected output") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tshape::FixedOut>("p", std::size_t{5});
            s.add_subsystem<tsys::Axpb>("c", 1.0f, 0.0f);  // x fixed kLen
            s.connect("p.y", "c.x");
        });
        const char* needles[] = {"shape mismatch", "'c.x' dimension 0 is fixed", "'p.y' (the connected output)"};
        try {
            model.setup();
            tshape::forward_throws(model, needles);
        } catch (const std::exception& e) {
            const std::string what = e.what();  // a fixed vs fixed mismatch is reported at setup
            for (const char* n : needles) CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
        }
    }
    SECTION("inherited (FromInputDim)") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tsys::Src>("p");  // kLen
            s.add_subsystem<tshape::FixedOut>("q", std::size_t{5});
            s.add_subsystem<tshape::Pair>("pair");
            s.connect("p.y", "pair.a");
            s.connect("q.y", "pair.b");
        });
        model.setup();
        tshape::forward_throws(model, std::vector<const char*>{"'pair.b' dimension 0 is FromInputDim{'a', 0}", "'q.y' (the connected output)"});
    }
    SECTION("value-driven (FromInput)") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tshape::FixedOut>("q", std::size_t{5});
            s.add_subsystem<tshape::ByVal>("bv");
            s.connect("q.y", "bv.v");
            s.init_input("bv.n", cs::init::from_host(tsys2::one_u32(3)));
        });
        model.setup();
        tshape::forward_throws(model, std::vector<const char*>{"'bv.v' dimension 0 is FromInput{'n', 0} = 3", "'q.y' (the connected output)", "is 5"});
    }
    SECTION("a matching constraint is silent") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tshape::FixedOut>("q", std::size_t{3});
            s.add_subsystem<tshape::ByVal>("bv");
            s.connect("q.y", "bv.v");
            s.init_input("bv.n", cs::init::from_host(tsys2::one_u32(3)));
        });
        model.setup();
        CHECK_NOTHROW(model.forward());
        model.wait();
    }
}

TEST_CASE("shape: the rank must match", "[system][shape]") {
    auto model = tsys2::make([](cs::System& s) {
        s.add_subsystem<tsys2::VBins>("v");  // a rank-1 output
        s.add_subsystem<tshape::Scale2>("a");  // a rank-2 input
        s.connect("v.b", "a.x");
    });
    // A connected input takes its unspecified dims at setup, so the rank is checked there.
    std::string what = "<no exception>";
    try {
        model.setup();
    } catch (const std::exception& e) {
        what = e.what();
    }
    for (const char* n : {"'a.x' has rank 2", "'v.b'", "has rank 1", "the rank must match"})
        CHECK_THAT(what, Catch::Matchers::ContainsSubstring(n));
}

TEST_CASE("shape: a tied output member with a conflicting fixed dim", "[system][shape][tie]") {
    SECTION("conflicting") {
        tsys::expect_throws(
            [&] {
                auto model = tsys2::make([](cs::System& s) {
                    s.add_subsystem<tshape::FixedOut>("a", std::size_t{4});
                    s.add_subsystem<tshape::FixedOut>("b", std::size_t{6});
                    s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::alias::fixed(6), cs::TieOutputsPolicy::allow_plain);
                });
                model.setup();
            },
            "shape mismatch: 'a.y' dimension 0 is fixed = 4");
    }
    SECTION("agreeing: the member declares what the tie says") {
        auto model = tsys2::make([](cs::System& s) {
            s.add_subsystem<tshape::FixedOut>("a", std::size_t{6});
            s.add_subsystem<tshape::FixedOut>("b", std::size_t{6});
            s.tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "t", cs::alias::fixed(6), cs::TieOutputsPolicy::allow_plain);
        });
        CHECK_NOTHROW(model.setup());
    }
}