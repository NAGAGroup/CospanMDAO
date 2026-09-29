// Smoke check for ParallelSystem (not the test suite), r01 §7.5, §7.3, §11.3.1.
// Driven by the shared test-only RootDriver (driver.hpp).
//
//   a. Helper form. Src feeds a parallel System of three independent
//      components (plain body, group body, host body); a consumer after the
//      Parallel reads all three outputs. Values are checked against host
//      references; then three forwards run back to back without waiting.
//   b. Typed form: a class derived from ParallelSystem with two children, one
//      of them placed with property::device{the same device}.
//   c. A child input connected from a SIBLING's output (any connection, here a
//      FromSourceDim one): the forward throws, naming both children.
//   d. Components only: a System child is rejected at compile time (a concept,
//      because a requires-expression outside a template is ill-formed rather
//      than false); the runtime-only rules (device-only properties, flavors)
//      are checked by the throws they raise.

#include <cmath>
#include <concepts>
#include <cstdio>
#include <exception>
#include <memory>
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

#include "driver.hpp"

using cs::detail::RootDriver;

constexpr std::size_t kLen = 8;

static std::vector<float> values_of(sk::queue& q, cs::detail::IOEntry& e) {
    auto               h = HostViewAccess::view<DType::f32, 1>(e, q);
    std::vector<float> v(h.size());
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = h[i];
    return v;
}

// ---- components ----------------------------------------------------------------------------
// Src: y[i] = i + 1.
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
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {}, {y_}, {});
    }
};

// Axpb (plain body): y[i] = a * x[i] + b.
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

// Square (group body): y[i] = x[i]^2. The work-group (16) is larger than the
// extent (8), so the launch is padded and the body guards its index.
struct SquareBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const auto g = it.get_global_id();
        if (g[0] < in.x.shape()[0]) out.y[g] = in.x[g] * in.x[g];
    }
};

class Square : public cs::Component<SquareBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(x_, "x", static_cast<int>(kLen));
        declare_output(y_, "y", static_cast<int>(kLen));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), sycl::range<1>{16}, {x_}, {y_}, {});
    }
};

// Prefix (host body): z[i] = z[i-1] + x[i]. Its input shape comes from the
// source (a FromSourceDim), which is allowed: the source is not a sibling.
struct PrefixBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> z;
    };
    void operator()(cs::Host, Inputs in, Outputs out) const {
        float acc = 0.0f;
        for (std::size_t i = 0; i < in.x.size(); ++i) {
            acc += in.x[i];
            out.z[i] = acc;
        }
    }
};

class Prefix : public cs::Component<PrefixBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> z_;

public:
    void setup() override {
        declare_input(x_, "x", cs::from_source_shape<1>());
        declare_output(z_, "z", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config({x_}, {z_}, {});
    }
};

// Sum3 (plain body): s[i] = p[i] + g[i] + h[i].
struct Sum3Body {
    struct Inputs {
        cs::Input<DType::f32, 1> p, g, h;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> s;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.s[i] = in.p[i] + in.g[i] + in.h[i]; }
};

class Sum3 : public cs::Component<Sum3Body> {
    cs::Input<DType::f32, 1>  p_, g_, h_;
    cs::Output<DType::f32, 1> s_;

public:
    void setup() override {
        declare_input(p_, "p", static_cast<int>(kLen));
        declare_input(g_, "g", static_cast<int>(kLen));
        declare_input(h_, "h", static_cast<int>(kLen));
        declare_output(s_, "s", static_cast<int>(kLen));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(s_.shape(), {p_, g_, h_}, {s_}, {});
    }
};

// Follow (plain body): y[i] = x[i] + 1, with the shape of x taken from its source.
struct FollowBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = in.x[i] + 1.0f; }
};

class Follow : public cs::Component<FollowBody> {
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

// ---- a. the helper form --------------------------------------------------------------------
class TopA : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("src");
        // Three independent components in one ParallelSystem.
        add_subsystem("par", cs::property_list{cs::property::parallel{}}, child<Axpb>("plain", 2.0f, 1.0f),
                      child<Square>("group"), child<Prefix>("host"));
        add_subsystem<Sum3>("sum");
        connect("src.y", "par.plain.x");
        connect("src.y", "par.group.x");
        connect("src.y", "par.host.x");
        connect("par.plain.y", "sum.p");
        connect("par.group.y", "sum.g");
        connect("par.host.z", "sum.h");
    }
};

static void case_a(const sycl::device& dev, sk::queue& q) {
    std::printf("-- a. helper form: src -> parallel{plain, group, host} -> sum\n");
    TopA       top;
    RootDriver d{dev, q, top};
    d.start();

    check(dynamic_cast<cs::ParallelSystem*>(&top.get_subsystem("par")) != nullptr,
          "a: property::parallel built a ParallelSystem");

    std::vector<float> want_p(kLen), want_g(kLen), want_h(kLen), want_s(kLen);
    float              acc = 0.0f;
    for (std::size_t i = 0; i < kLen; ++i) {
        const float x = static_cast<float>(i + 1);
        acc += x;
        want_p[i] = 2.0f * x + 1.0f;
        want_g[i] = x * x;
        want_h[i] = acc;
        want_s[i] = want_p[i] + want_g[i] + want_h[i];
    }

    d.forward();
    check(values_of(q, top.entry_at("par.plain.y")) == want_p, "a: plain body: y == 2x + 1");
    check(values_of(q, top.entry_at("par.group.y")) == want_g, "a: group body: y == x^2");
    check(values_of(q, top.entry_at("par.host.z")) == want_h, "a: host body (under the group): z == prefix sum");
    check(values_of(q, top.entry_at("sum.s")) == want_s, "a: the consumer after the Parallel reads all three outputs");

    // Three forwards in a row, nothing waits between them (§7.3: the ordering
    // is the stream's), then wait once.
    d.enqueue();
    d.enqueue();
    d.enqueue();
    d.wait();
    check(values_of(q, top.entry_at("sum.s")) == want_s, "a: three forwards enqueued back to back, then one wait: same sums");
    check(values_of(q, top.entry_at("par.host.z")) == want_h, "a: ... and the host body's output is intact");
}

// ---- b. the typed form ---------------------------------------------------------------------
class TwoPar : public cs::ParallelSystem {
    sycl::device dev_;

public:
    explicit TwoPar(const sycl::device& d) : dev_{d} {}
    void setup() override {
        add_subsystem<Axpb>("a", 2.0f, 1.0f);
        // The one property a ParallelSystem child takes: its own device.
        add_subsystem<Axpb>("b", cs::property_list{cs::property::device{dev_}}, 3.0f, 0.5f);
    }
};

class TopB : public cs::System {
    sycl::device dev_;

public:
    explicit TopB(const sycl::device& d) : dev_{d} {}
    void setup() override {
        add_subsystem<Src>("src");
        add_subsystem<TwoPar>("par", dev_);
        connect("src.y", "par.a.x");
        connect("src.y", "par.b.x");
    }
};

static void case_b(const sycl::device& dev, sk::queue& q) {
    std::printf("-- b. typed form: class TwoPar : ParallelSystem, child b placed with property::device\n");
    TopB       top{dev};
    RootDriver d{dev, q, top};
    d.start();
    d.forward();

    std::vector<float> want_a(kLen), want_b(kLen);
    for (std::size_t i = 0; i < kLen; ++i) {
        const float x = static_cast<float>(i + 1);
        want_a[i]     = 2.0f * x + 1.0f;
        want_b[i]     = 3.0f * x + 0.5f;
    }
    const cs::Placement here{dev, sk::kind::device};
    check(top.entry_at("par.b.x").placement() == here, "b: the placed child's entries sit on (dev, device)");
    check(values_of(q, top.entry_at("par.a.y")) == want_a, "b: child a: y == 2x + 1");
    check(values_of(q, top.entry_at("par.b.y")) == want_b, "b: child b (device-placed): y == 3x + 0.5");
}

// ---- c. a connection between siblings --------------------------------------------------
class TopC : public cs::System {
public:
    void setup() override {
        add_subsystem("par", cs::property_list{cs::property::parallel{}}, child<Axpb>("a", 1.0f, 0.0f),
                      child<Follow>("b"));
        connect("par.a.y", "par.b.x");  // b.x takes its shape from a sibling's output
    }
};

static void case_c(const sycl::device& dev, sk::queue& q) {
    std::printf("-- c. input connected from a sibling's output: the forward throws\n");
    TopC       top;
    RootDriver d{dev, q, top};
    d.start();  // the wiring itself is legal; only the forward refuses
    bool        threw = false;
    std::string what;
    try {
        d.enqueue();
    } catch (const std::exception& e) {
        threw = true;
        what  = e.what();
        std::printf("      %s\n", what.c_str());
    }
    d.wait();
    check(threw, "c: the forward throws");
    check(what.find("top.par.a") != std::string::npos && what.find("top.par.b") != std::string::npos,
          "c: the message names both siblings");
}

// ---- d. Components only ----------------------------------------------------------------------
struct SomeSystem : cs::System {};

// True when P::add_subsystem<T>(name) is a valid call. A concept, not a bare
// requires-expression: outside a template an invalid expression is a hard error.
template <class P, class T>
concept CanAdd = requires(P& p) { p.template add_subsystem<T>(std::string{"x"}); };

template <class P>
concept CanAddUntyped = requires(P& p) { p.add_subsystem(std::string{"x"}); };

static_assert(CanAdd<cs::System, SomeSystem>, "control: a plain System takes a System child");
static_assert(CanAdd<cs::System, Src>, "control: a plain System takes a Component child");
static_assert(CanAdd<cs::ParallelSystem, Src>, "a ParallelSystem takes a Component child");
static_assert(!CanAdd<cs::ParallelSystem, SomeSystem>, "a ParallelSystem must reject a System child");
static_assert(!CanAdd<cs::ParallelSystem, cs::ParallelSystem>, "nested Parallel is not a concept");
static_assert(CanAddUntyped<cs::System>, "control: a plain System takes an untyped child System");
static_assert(!CanAddUntyped<cs::ParallelSystem>, "a ParallelSystem has no untyped children");

// Runtime-only rules: each is a Top whose setup() breaks one, and start() throws.
class BadProps : public cs::ParallelSystem {
public:
    void setup() override { add_subsystem<Src>("s", cs::property_list{cs::property::transient{}}); }
};
class TopBadProps : public cs::System {
public:
    void setup() override { add_subsystem<BadProps>("p"); }
};
class TopTypedParallelOnSystem : public cs::System {
public:
    void setup() override { add_subsystem<SomeSystem>("p", cs::property_list{cs::property::parallel{}}); }
};
class TopParallelAndFused : public cs::System {
public:
    void setup() override {
        add_subsystem("p", cs::property_list{cs::property::parallel{}, cs::property::fused{}}, child<Src>("s"));
    }
};
class TopFused : public cs::System {
public:
    // property::fused{parallel} builds a parallel-mode FusedSystem (step 3b-3); one valid stage, so start() succeeds.
    void setup() override {
        add_subsystem("p", cs::property_list{cs::property::fused{cs::property::fused::Mode::parallel}},
                      child<Src>("s"));
    }
};

template <class Top>
static bool start_throws(const sycl::device& dev, sk::queue& q) {
    try {
        Top        top;
        RootDriver d{dev, q, top};
        d.start();
    } catch (const std::exception& e) {
        std::printf("      %s\n", e.what());
        return true;
    }
    return false;
}

static void case_d(const sycl::device& dev, sk::queue& q) {
    std::printf("-- d. Components only: compile-time checks (static_assert, above) and the runtime rules\n");
    check(true, "d: a System child, a nested Parallel and an untyped child are rejected at compile time");
    check(start_throws<TopBadProps>(dev, q), "d: a ParallelSystem child with a property other than device throws");
    check(start_throws<TopTypedParallelOnSystem>(dev, q),
          "d: property::parallel on a typed non-Parallel child throws");
    check(start_throws<TopParallelAndFused>(dev, q), "d: parallel and fused together throw");
    check(!start_throws<TopFused>(dev, q), "d: property::fused{parallel} builds a parallel-mode FusedSystem (no throw)");
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
        case_d(dev, q);
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    std::printf(failures == 0 ? "SMOKE PARALLEL PASS\n" : "SMOKE PARALLEL FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
