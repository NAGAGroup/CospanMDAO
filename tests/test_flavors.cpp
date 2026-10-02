// Parallel and Fused tests, every case through a cospan::Model.
//
//   Parallel       helper form (plain + group with scratch + host, fed by a
//                  producer, read by a consumer); typed form with a device
//                  child; an input connected to a sibling's output throws
//                  (unspecified and fixed-shape); tied outputs throw unless
//                  allow_tied_outputs(); rejections; a Parallel nested under
//                  a plain System
//   Fused, seq.    three plain stages; two group+scratch stages (B's own
//                  barrier, own local slices); rank-2 (wg {8,8}) and rank-3
//                  (7x9x11) waves; stage ranges may differ; ties are not checked
//   Fused, par.    three independent stages; a connection throws and
//                  allow_tied_outputs() does not lift it; tied outputs throw and
//                  allow_tied_outputs() lets them run
//   Fused, both    value-driven sizes (refused when driven by a stage of the
//                  same fuse, allowed from outside); the local-memory limit
//                  (skipped where the device reports none, e.g. omp)
//   Fused, errors  custom-op, host and mixed-rank children throw at setup
//   Equivalence    four stages as a plain System and as a sequential
//                  FusedSystem give bitwise-equal outputs
//
// The components live in namespace tfl. No case is tagged [os-free]: since
// the free lane every case runs on CUDA, the value-driven case whose root
// grows included.

#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;
using cst::default_config;
using cst::to_vector;

namespace tfl {

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

using Build = std::function<void(cs::System&)>;

/// A root (or nested) System whose setup() runs a callable.
class Top : public cs::System {
    Build build_;

public:
    explicit Top(Build b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

/// A typed ParallelSystem / FusedSystem whose setup() runs a callable.
class Par : public cs::ParallelSystem {
    std::function<void(cs::ParallelSystem&)> build_;

public:
    explicit Par(std::function<void(cs::ParallelSystem&)> b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};
class Wave : public cs::FusedSystem {
    std::function<void(cs::FusedSystem&)> build_;

public:
    Wave(Mode m, std::optional<sycl::range<3>> wg, std::function<void(cs::FusedSystem&)> b)
        : cs::FusedSystem(m, wg), build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

cs::Model make(Build b) {
    cs::Model m = cs::create_model<Top>(default_config(), std::move(b));
    return m;
}

void run_three(cs::Model& m) {
    m.forward();
    m.forward();
    m.forward();
    m.wait();
}

template <int R>
std::vector<float> out_of(cs::Model& m, const char* path) {
    return to_vector(m.view_output<DType::f32, R>(path));
}

std::vector<float> reverse_in_groups(const std::vector<float>& in, std::size_t wg) {
    std::vector<float> out(in.size());
    for (std::size_t g = 0; g < in.size(); ++g) {
        const std::size_t base = g / wg * wg, l = g - base, m = base + (wg - 1 - l);
        out[g] = m < in.size() ? in[m] : 0.0f;  // the ragged last group reads zeros
    }
    return out;
}

// ---- Src: y = scale * v + offset, v = the index read as base-100 digits ---------------------------
template <int R>
struct SrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, R> y;
    };
    float scale, offset;
    void operator()(sycl::id<R> i, Inputs, Outputs out) const {
        float v = 0.0f;
        for (int d = 0; d < R; ++d) v = v * 100.0f + static_cast<float>(i[d]);
        out.y[i] = scale * v + offset;
    }
};

template <int R>
class SrcND : public cs::Component<SrcBody<R>> {
    cs::Output<DType::f32, R> y_;
    std::array<int, R>        dims_;
    float                     scale_, offset_;

public:
    explicit SrcND(std::array<int, R> dims, float scale = 1.0f, float offset = 1.0f)
        : dims_{dims}, scale_{scale}, offset_{offset} {}
    void setup() override {
        std::apply([&](auto... n) { this->declare_output(y_, "y", n...); }, dims_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(y_.shape(), {}, {y_}, {scale_, offset_});
    }
};
using Src1 = SrcND<1>;
using Src2 = SrcND<2>;
using Src3 = SrcND<3>;

// ---- Elem: y = f(x), the shape follows the source; Op picks f --------------------------------------
template <int R, int Op>
struct ElemBody {
    struct Inputs {
        cs::Input<DType::f32, R> x;
    };
    struct Outputs {
        cs::Output<DType::f32, R> y;
    };
    void operator()(sycl::id<R> i, Inputs in, Outputs out) const {
        const float x = in.x[i];
        float       y = 0.0f;
        if constexpr (Op == 0) y = 2.0f * x + 1.0f;
        if constexpr (Op == 1) y = x * x;
        if constexpr (Op == 2) y = x + 3.0f;
        if constexpr (Op == 3) y = 0.5f * x - 1.0f;
        if constexpr (Op == 4) y = 3.0f * x;
        out.y[i] = y;
    }
};

template <int R, int Op>
class ElemND : public cs::Component<ElemBody<R, Op>> {
    cs::Input<DType::f32, R>  x_;
    cs::Output<DType::f32, R> y_;

public:
    void setup() override {
        this->declare_input(x_, "x", cs::unspecified_shape<R>());
        this->declare_output(y_, "y", cs::from_input_shape<R>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(x_.shape(), {x_}, {y_}, {});
    }
};
template <int Op>
using Elem = ElemND<1, Op>;

float ref(int op, float x) {
    switch (op) {
        case 0: return 2.0f * x + 1.0f;
        case 1: return x * x;
        case 2: return x + 3.0f;
        case 3: return 0.5f * x - 1.0f;
        default: return 3.0f * x;
    }
}

// ---- Half: a plain body that launches over half of its input (unequal range) ------------------------
struct HalfBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = in.x[i] + 1.0f; }
};
class Half : public cs::Component<HalfBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{x_.shape()[0] / 2}, {x_}, {y_}, {});
    }
};

// ---- group bodies with Scratch: Rev (reverse within the group), Mirror (reads the mirrored element) ---
struct RevBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    struct Scratch {
        cs::Local<DType::f32, 1> tile;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out, Scratch s) const {
        const std::size_t n = it.get_local_range(0), l = it.get_local_linear_id(), g = it.get_global_id(0);
        const std::size_t total = in.x.shape()[0];
        s.tile[l]               = g < total ? in.x[g] : 0.0f;
        sycl::group_barrier(it.get_group());
        if (g < total) out.y[g] = s.tile[n - 1 - l];
    }
};
struct MirrorBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    struct Scratch {
        cs::Local<DType::f32, 1> tile;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out, Scratch s) const {
        const std::size_t n = it.get_local_range(0), l = it.get_local_linear_id(), g = it.get_global_id(0);
        const std::size_t total = in.x.shape()[0];
        s.tile[l]               = 0.0f;  // touches the local buffer: a scratch stage
        // B reads OTHER work-items' writes of stage A: nothing orders them for it, so it calls
        // its own group barrier first (every work-item reaches it, ragged lanes included).
        sycl::group_barrier(it.get_group());
        const std::size_t m = g - l + (n - 1 - l);
        if (g < total) out.y[g] = m < total ? in.x[m] : 0.0f;
    }
};

template <class Body>
class GroupStage : public cs::Component<Body> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
    std::size_t               wg_;

public:
    explicit GroupStage(std::size_t wg = 64) : wg_{wg} {}
    void setup() override {
        this->declare_input(x_, "x", cs::unspecified_shape<1>());
        this->declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        typename Body::Scratch s;
        s.tile.reshape(sycl::range<1>{wg_});
        return this->create_launch_config(x_.shape(), sycl::range<1>{wg_}, {x_}, {y_}, s, {});
    }
};

// ---- Prefix (host body): z[i] = z[i-1] + x[i] ---------------------------------------------------------
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
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(z_, "z", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config({x_}, {z_}, {}); }
};

// ---- Sum3 (fixed size n): s = p + g + h --------------------------------------------------------------
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
    int                       n_;

public:
    explicit Sum3(int n) : n_{n} {}
    void setup() override {
        declare_input(p_, "p", n_);
        declare_input(g_, "g", n_);
        declare_input(h_, "h", n_);
        declare_output(s_, "s", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(s_.shape(), {p_, g_, h_}, {s_}, {});
    }
};

// ---- Adder: acc[i] += v, an ATOMIC output (zeroed in init); two of them tie into one root ---------------
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
    void setup() override { declare_output(acc_, "acc", cs::unspecified_shape<1>()); }  // a tie_outputs member: the alias owns the shape
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(acc_.shape(), {}, {acc_}, {v_});
    }
};

// ---- ZeroAcc: the zeroing producer (outputs are never initialized, r01 §6.0); it runs first and the
// Adders accumulate onto the tied root --------------------------------------------------------------
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

// ---- NSrc / Bins: a value-driven size. NSrc writes the value n; Bins' output has n elements (FromInput) -----
struct NSrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::u32, 1> n;
    };
    float v;
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.n[i] = static_cast<std::uint32_t>(v); }
};
class NSrc : public cs::Component<NSrcBody> {
    cs::Output<DType::u32, 1> n_;
    float                     v_;

public:
    explicit NSrc(float v) : v_{v} {}
    void setup() override { declare_output(n_, "n", 1); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(n_.shape(), {}, {n_}, {v_});
    }
};
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

// ---- BigStage: a group stage whose scratch tile has `elems` floats (the work-group is 64) -----------------
struct BigBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    struct Scratch {
        cs::Local<DType::f32, 1> tile;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out, Scratch s) const {
        const std::size_t l = it.get_local_linear_id(), g = it.get_global_id(0);
        const bool        ok = g < in.x.shape()[0];
        s.tile[l]            = ok ? in.x[g] : 0.0f;  // touches the local buffer (own slot): a scratch stage
        if (ok) out.y[g] = s.tile[l];
    }
};
class BigStage : public cs::Component<BigBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
    std::size_t               elems_;

public:
    explicit BigStage(std::size_t elems) : elems_{elems} {}
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        BigBody::Scratch s;
        s.tile.reshape(sycl::range<1>{elems_});
        return create_launch_config(x_.shape(), sycl::range<1>{64}, {x_}, {y_}, s, {});
    }
};

// ---- custom-op and host children that a wave must refuse ------------------------------------------------
struct NopCustomBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::CustomOp&, Inputs, Outputs) const {}
};
class NopCustom : public cs::Component<NopCustomBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config({x_}, {y_}, {}); }
};
struct NopHostBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::Host, Inputs, Outputs) const {}
};
class NopHost : public cs::Component<NopHostBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config({x_}, {y_}, {}); }
};

// ---- compile-time rejection: a concept, not a bare requires-expression -------------------------------
struct SomeSystem : cs::System {};
template <class P, class T>
concept CanAdd = requires(P& p) { p.template add_subsystem<T>(std::string{"x"}); };
template <class P>
concept CanAddUntyped = requires(P& p) { p.add_subsystem(std::string{"x"}); };

static_assert(CanAdd<cs::System, SomeSystem>, "control: a plain System takes a System child");
static_assert(CanAdd<cs::System, Src1>, "control: a plain System takes a Component child");
static_assert(CanAdd<cs::ParallelSystem, Src1>, "a ParallelSystem takes a Component child");
static_assert(!CanAdd<cs::ParallelSystem, SomeSystem>, "a ParallelSystem must reject a System child");
static_assert(!CanAdd<cs::ParallelSystem, cs::ParallelSystem>, "nested Parallel is not a concept");
static_assert(!CanAddUntyped<cs::ParallelSystem>, "a ParallelSystem has no untyped children");
static_assert(CanAdd<cs::FusedSystem, Src1>, "a FusedSystem takes a Component child");
static_assert(!CanAdd<cs::FusedSystem, SomeSystem>, "a FusedSystem must reject a System child");
static_assert(!CanAddUntyped<cs::FusedSystem>, "a FusedSystem has no untyped children");

constexpr int kLen = 12;  // the Parallel cases; the group body's work-group is 8, so the last group is ragged

}  // namespace tfl

// =============================================================================================
// Parallel
// =============================================================================================
TEST_CASE("parallel: helper form, plain + group with scratch + host, three forwards then one wait",
          "[flavors][parallel]") {
    cs::Model model = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{tfl::kLen});
        s.add_subsystem("par", cs::property_list{cs::property::parallel{}}, s.child<tfl::Elem<0>>("plain"),
                        s.child<tfl::GroupStage<tfl::RevBody>>("group", std::size_t{8}), s.child<tfl::Prefix>("host"));
        s.add_subsystem<tfl::Sum3>("sum", tfl::kLen);
        s.connect("src.y", "par.plain.x");
        s.connect("src.y", "par.group.x");
        s.connect("src.y", "par.host.x");
        s.connect("par.plain.y", "sum.p");
        s.connect("par.group.y", "sum.g");
        s.connect("par.host.z", "sum.h");
    });
    model.setup();
    CHECK(dynamic_cast<cs::ParallelSystem*>(&model.get_subsystem("par")) != nullptr);

    std::vector<float> x(tfl::kLen), want_p(tfl::kLen), want_h(tfl::kLen), want_s(tfl::kLen);
    float              acc = 0.0f;
    for (int i = 0; i < tfl::kLen; ++i) {
        x[i] = static_cast<float>(i + 1);
        acc += x[i];
        want_p[i] = 2.0f * x[i] + 1.0f;
        want_h[i] = acc;
    }
    const auto want_g = tfl::reverse_in_groups(x, 8);
    for (int i = 0; i < tfl::kLen; ++i) want_s[i] = want_p[i] + want_g[i] + want_h[i];

    tfl::run_three(model);
    CHECK(tfl::out_of<1>(model, "par.plain.y") == want_p);
    CHECK(tfl::out_of<1>(model, "par.group.y") == want_g);
    CHECK(tfl::out_of<1>(model, "par.host.z") == want_h);
    CHECK(tfl::out_of<1>(model, "sum.s") == want_s);
}

TEST_CASE("parallel: typed ParallelSystem, a child on property::device equal to the parent's", "[flavors][parallel]") {
    const sycl::device dev{sycl::default_selector_v};  // the Model's default device
    cs::Model model = tfl::make([dev](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{tfl::kLen});
        s.add_subsystem<tfl::Par>("par", std::function<void(cs::ParallelSystem&)>{[dev](cs::ParallelSystem& p) {
            p.add_subsystem<tfl::Elem<0>>("a");
            p.add_subsystem<tfl::Elem<2>>("b", cs::property_list{cs::property::device{dev}});
        }});
        s.connect("src.y", "par.a.x");
        s.connect("src.y", "par.b.x");
    });
    model.setup();
    tfl::run_three(model);
    CHECK(dynamic_cast<tfl::Par*>(&model.get_subsystem("par")) != nullptr);
    const cs::Placement here{dev, sk::kind::device};
    CHECK(model.root().entry_at("par.b.x").placement() == here);
    CHECK(model.root().entry_at("par.a.x").placement() == here);
    std::vector<float> want_a(tfl::kLen), want_b(tfl::kLen);
    for (int i = 0; i < tfl::kLen; ++i) {
        want_a[i] = tfl::ref(0, static_cast<float>(i + 1));
        want_b[i] = tfl::ref(2, static_cast<float>(i + 1));
    }
    CHECK(tfl::out_of<1>(model, "par.a.y") == want_a);
    CHECK(tfl::out_of<1>(model, "par.b.y") == want_b);
}

TEST_CASE("parallel: an input connected to a sibling's output throws at forward, naming both",
          "[flavors][parallel][errors]") {
    SECTION("an input with an unspecified dim") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem("par", cs::property_list{cs::property::parallel{}},
                            s.child<tfl::Src1>("a", std::array<int, 1>{4}), s.child<tfl::Elem<0>>("b"));
            s.connect("par.a.y", "par.b.x");  // b.x takes its shape from a sibling's output
        });
        model.setup();  // the wiring is legal; only the forward refuses
        tfl::expect_throws([&] { model.forward(); }, {"par.a", "par.b", "run unordered"});
        model.wait();
    }
    SECTION("a fixed-shape input") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem("par", cs::property_list{cs::property::parallel{}},
                            s.child<tfl::Src1>("a", std::array<int, 1>{4}), s.child<tfl::Sum3>("b", 4));
            s.connect("par.a.y", "par.b.p");  // not a shape dependency, still a connection
        });
        model.setup();
        tfl::expect_throws([&] { model.forward(); }, {"par.a", "par.b", "run unordered"});
        model.wait();
    }
}

TEST_CASE("parallel: tied outputs throw at forward, naming both, unless allow_tied_outputs()",
          "[flavors][parallel][errors]") {
    const auto tied = [](bool allow) {
        return tfl::make([allow](cs::System& s) {
            s.add_subsystem<tfl::ZeroAcc>("zero");
            s.add_subsystem<tfl::Par>("par", std::function<void(cs::ParallelSystem&)>{[allow](cs::ParallelSystem& p) {
                if (allow) p.allow_tied_outputs();
                p.add_subsystem<tfl::Adder>("a", 1.0f);
                p.add_subsystem<tfl::Adder>("b", 2.0f);
            }});
            s.tie_outputs<DType::atomic_f32, 1>({"zero.acc", "par.a.acc", "par.b.acc"}, "sum",
                                                cs::alias::fixed(12));  // atomic writers, one root
        });
    };
    SECTION("refused") {
        cs::Model model = tied(false);
        model.setup();
        tfl::expect_throws([&] { model.forward(); },
                           {"par.a", "par.b", "share one root", "allow_tied_outputs"});
        model.wait();
    }
    SECTION("allowed: the atomic writers accumulate") {
        cs::Model model = tied(true);
        model.setup();
        model.forward();
        model.wait();
        CHECK(tfl::out_of<1>(model, "sum") == std::vector<float>(12, 3.0f));  // 1 + 2
    }
}

TEST_CASE("parallel: rejections at setup", "[flavors][parallel][errors]") {
    SECTION("a non-device property on a child") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem<tfl::Par>("par", std::function<void(cs::ParallelSystem&)>{[](cs::ParallelSystem& p) {
                p.add_subsystem<tfl::Src1>("s", cs::property_list{cs::property::transient{}}, std::array<int, 1>{4});
            }});
        });
        tfl::expect_throws([&] { model.setup(); }, {"takes property::device only"});
        model.wait();  // setup() may have submitted initializer work before it threw
    }
    SECTION("parallel and fused together") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem("p", cs::property_list{cs::property::parallel{}, cs::property::fused{}},
                            s.child<tfl::Src1>("s", std::array<int, 1>{4}));
        });
        tfl::expect_throws([&] { model.setup(); }, {"two flavors"});
        model.wait();  // setup() may have submitted initializer work before it threw
    }
    // Compile-time rejection of System children is the static_asserts at the top of the file.
}

TEST_CASE("parallel: nested under a plain System, values are correct", "[flavors][parallel]") {
    cs::Model model = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Top>("mid", tfl::Build{[](cs::System& m) {
            m.add_subsystem<tfl::Src1>("src", std::array<int, 1>{tfl::kLen});
            m.add_subsystem("par", cs::property_list{cs::property::parallel{}}, m.child<tfl::Elem<0>>("a"),
                            m.child<tfl::Elem<2>>("b"));
            m.add_subsystem<tfl::Elem<1>>("c");
            m.connect("src.y", "par.a.x");
            m.connect("src.y", "par.b.x");
            m.connect("par.a.y", "c.x");  // a consumer after the Parallel
        }});
    });
    model.setup();
    tfl::run_three(model);
    std::vector<float> want_a(tfl::kLen), want_b(tfl::kLen), want_c(tfl::kLen);
    for (int i = 0; i < tfl::kLen; ++i) {
        const float x = static_cast<float>(i + 1);
        want_a[i]     = tfl::ref(0, x);
        want_b[i]     = tfl::ref(2, x);
        want_c[i]     = tfl::ref(1, want_a[i]);
    }
    CHECK(tfl::out_of<1>(model, "mid.par.a.y") == want_a);
    CHECK(tfl::out_of<1>(model, "mid.par.b.y") == want_b);
    CHECK(tfl::out_of<1>(model, "mid.c.y") == want_c);
}

// =============================================================================================
// Fused, sequential
// =============================================================================================
TEST_CASE("fused sequential: three plain stages, A -> B element-wise, C on a ragged size", "[flavors][fused]") {
    cs::Model model = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{1000});
        s.add_subsystem<tfl::Src1>("src2", std::array<int, 1>{333});
        s.add_subsystem<tfl::Wave>(
            "f", cs::FusedSystem::Mode::sequential, std::optional<sycl::range<3>>{},
            std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                w.add_subsystem<tfl::Elem<0>>("a");
                w.add_subsystem<tfl::Elem<1>>("b");
                w.add_subsystem<tfl::Elem<2>>("c");
            }});
        s.connect("src.y", "f.a.x");
        s.connect("src2.y", "f.c.x");
        s.connect("f.a.y", "f.b.x");
    });
    model.setup();
    tfl::run_three(model);
    std::vector<float> want_y(1000), want_z(1000), want_w(333);
    for (int i = 0; i < 1000; ++i) {
        want_y[i] = tfl::ref(0, static_cast<float>(i + 1));
        want_z[i] = tfl::ref(1, want_y[i]);
    }
    for (int i = 0; i < 333; ++i) want_w[i] = tfl::ref(2, static_cast<float>(i + 1));
    CHECK(tfl::out_of<1>(model, "f.a.y") == want_y);
    CHECK(tfl::out_of<1>(model, "f.b.y") == want_z);
    CHECK(tfl::out_of<1>(model, "f.c.y") == want_w);
}

TEST_CASE("fused sequential: two group+scratch stages, B's own barrier, own local slices, ragged last group",
          "[flavors][fused]") {
    cs::Model model = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{200});
        s.add_subsystem<tfl::Wave>(
            "f", cs::FusedSystem::Mode::sequential, std::optional<sycl::range<3>>{},
            std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                w.add_subsystem<tfl::GroupStage<tfl::RevBody>>("a");
                w.add_subsystem<tfl::GroupStage<tfl::MirrorBody>>("b");
            }});
        s.connect("src.y", "f.a.x");
        s.connect("f.a.y", "f.b.x");
    });
    model.setup();
    model.forward();
    model.wait();
    std::vector<float> x(200);
    for (int i = 0; i < 200; ++i) x[i] = static_cast<float>(i + 1);
    const auto want_a = tfl::reverse_in_groups(x, 64);
    const auto want_b = tfl::reverse_in_groups(want_a, 64);
    CHECK(tfl::out_of<1>(model, "f.a.y") == want_a);
    CHECK(tfl::out_of<1>(model, "f.b.y") == want_b);  // exact only with B's own barrier (there is none between stages)
    // The last group holds 8 elements (200 = 3 * 64 + 8): A reads zeros for the padding lanes.
    CHECK(want_a[192] == 0.0f);
}

TEST_CASE("fused sequential: rank-2 wave with wg {8,8} and rank-3 wave (7x9x11) with the default wg",
          "[flavors][fused]") {
    SECTION("rank 2, 40 x 30, wg {8,8}") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem<tfl::Src2>("src", std::array<int, 2>{40, 30}, 1.0f, 0.0f);
            s.add_subsystem<tfl::Wave>(
                "f", cs::FusedSystem::Mode::sequential, std::optional<sycl::range<3>>{sycl::range<3>{8, 8, 1}},
                std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                    w.add_subsystem<tfl::ElemND<2, 0>>("a");
                    w.add_subsystem<tfl::ElemND<2, 4>>("b");
                }});
            s.connect("src.y", "f.a.x");
            s.connect("f.a.y", "f.b.x");
        });
        model.setup();
        model.forward();
        model.wait();
        std::vector<float> want_a, want_b;
        for (int i = 0; i < 40; ++i)
            for (int j = 0; j < 30; ++j) {
                want_a.push_back(tfl::ref(0, static_cast<float>(i * 100 + j)));
                want_b.push_back(tfl::ref(4, want_a.back()));
            }
        CHECK(tfl::out_of<2>(model, "f.a.y") == want_a);
        CHECK(tfl::out_of<2>(model, "f.b.y") == want_b);
    }
    SECTION("rank 3, 7 x 9 x 11, default wg") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem<tfl::Src3>("src", std::array<int, 3>{7, 9, 11}, 1.0f, 0.0f);
            s.add_subsystem<tfl::Wave>(
                "f", cs::FusedSystem::Mode::sequential, std::optional<sycl::range<3>>{},
                std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                    w.add_subsystem<tfl::ElemND<3, 0>>("a");
                    w.add_subsystem<tfl::ElemND<3, 4>>("b");
                }});
            s.connect("src.y", "f.a.x");
            s.connect("f.a.y", "f.b.x");
        });
        model.setup();
        model.forward();
        model.wait();
        std::vector<float> want_a, want_b;
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 9; ++j)
                for (int k = 0; k < 11; ++k) {
                    want_a.push_back(tfl::ref(0, static_cast<float>((i * 100 + j) * 100 + k)));
                    want_b.push_back(tfl::ref(4, want_a.back()));
                }
        CHECK(tfl::out_of<3>(model, "f.a.y") == want_a);
        CHECK(tfl::out_of<3>(model, "f.b.y") == want_b);
    }
}

TEST_CASE("fused sequential: stages may launch over different ranges; no inter-stage check", "[flavors][fused]") {
    cs::Model model = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{1000});
        s.add_subsystem<tfl::Wave>(
            "f", cs::FusedSystem::Mode::sequential, std::optional<sycl::range<3>>{},
            std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                w.add_subsystem<tfl::Elem<0>>("a");   // launches over 1000
                w.add_subsystem<tfl::Half>("b");      // launches over 500, reads a's output at the same index
            }});
        s.connect("src.y", "f.a.x");
        s.connect("f.a.y", "f.b.x");
    });
    model.setup();
    tfl::run_three(model);
    std::vector<float> want_a(1000), want_b(500);
    for (int i = 0; i < 1000; ++i) want_a[i] = tfl::ref(0, static_cast<float>(i + 1));
    for (int i = 0; i < 500; ++i) want_b[i] = want_a[i] + 1.0f;
    const auto b = tfl::out_of<1>(model, "f.b.y");
    REQUIRE(b.size() == 1000);  // the output follows the input's shape; the body fills the first half
    CHECK(tfl::out_of<1>(model, "f.a.y") == want_a);
    CHECK(std::vector<float>(b.begin(), b.begin() + 500) == want_b);
}

// =============================================================================================
// Fused, parallel
// =============================================================================================
TEST_CASE("fused parallel: three independent stages of different sizes, three forwards", "[flavors][fused]") {
    cs::Model model = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src1", std::array<int, 1>{1000});
        s.add_subsystem<tfl::Src1>("src2", std::array<int, 1>{257});
        s.add_subsystem<tfl::Src1>("src3", std::array<int, 1>{640});
        s.add_subsystem<tfl::Wave>(
            "f", cs::FusedSystem::Mode::parallel, std::optional<sycl::range<3>>{},
            std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                w.add_subsystem<tfl::Elem<0>>("a");                       // plain, 1000
                w.add_subsystem<tfl::Elem<2>>("c");                       // plain, 257
                w.add_subsystem<tfl::GroupStage<tfl::RevBody>>("r");      // group + scratch, 640
            }});
        s.connect("src1.y", "f.a.x");
        s.connect("src2.y", "f.c.x");
        s.connect("src3.y", "f.r.x");
    });
    model.setup();
    tfl::run_three(model);
    std::vector<float> want_a(1000), want_c(257), x3(640);
    for (int i = 0; i < 1000; ++i) want_a[i] = tfl::ref(0, static_cast<float>(i + 1));
    for (int i = 0; i < 257; ++i) want_c[i] = tfl::ref(2, static_cast<float>(i + 1));
    for (int i = 0; i < 640; ++i) x3[i] = static_cast<float>(i + 1);
    CHECK(tfl::out_of<1>(model, "f.a.y") == want_a);
    CHECK(tfl::out_of<1>(model, "f.c.y") == want_c);
    CHECK(tfl::out_of<1>(model, "f.r.y") == tfl::reverse_in_groups(x3, 64));
}

TEST_CASE("fused parallel: a connection between stages throws naming both, and allow_tied_outputs() does not lift it",
          "[flavors][fused][errors]") {
    const auto chain = [](bool allow_ties) {
        return tfl::make([allow_ties](cs::System& s) {
            s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{1000});
            s.add_subsystem<tfl::Wave>(
                "f", cs::FusedSystem::Mode::parallel, std::optional<sycl::range<3>>{},
                std::function<void(cs::FusedSystem&)>{[allow_ties](cs::FusedSystem& w) {
                    if (allow_ties) w.allow_tied_outputs();
                    w.add_subsystem<tfl::Elem<0>>("a");
                    w.add_subsystem<tfl::Elem<1>>("b");
                }});
            s.connect("src.y", "f.a.x");
            s.connect("f.a.y", "f.b.x");  // B reads A's output
        });
    };
    SECTION("refused") {
        cs::Model model = chain(false);
        model.setup();  // only the forward sees the pair
        tfl::expect_throws([&] { model.forward(); }, {"f.a", "f.b", "run unordered"});
        model.wait();
    }
    SECTION("no override: allow_tied_outputs() is about ties only") {
        cs::Model model = chain(true);
        model.setup();
        tfl::expect_throws([&] { model.forward(); }, {"f.a", "f.b", "run unordered"});
        model.wait();
    }
}

TEST_CASE("fused: tied outputs throw in parallel mode unless allow_tied_outputs(); sequential mode does not check",
          "[flavors][fused][errors]") {
    const auto tied = [](cs::FusedSystem::Mode mode, bool allow) {
        return tfl::make([mode, allow](cs::System& s) {
            s.add_subsystem<tfl::ZeroAcc>("zero");
            s.add_subsystem<tfl::Wave>(
                "f", mode, std::optional<sycl::range<3>>{},
                std::function<void(cs::FusedSystem&)>{[allow](cs::FusedSystem& w) {
                    if (allow) w.allow_tied_outputs();
                    w.add_subsystem<tfl::Adder>("a", 1.0f);
                    w.add_subsystem<tfl::Adder>("b", 2.0f);
                }});
            s.tie_outputs<DType::atomic_f32, 1>({"zero.acc", "f.a.acc", "f.b.acc"}, "sum",
                                                cs::alias::fixed(12));  // atomic writers, one root
        });
    };
    SECTION("parallel, refused") {
        cs::Model model = tied(cs::FusedSystem::Mode::parallel, false);
        model.setup();
        tfl::expect_throws([&] { model.forward(); }, {"f.a", "f.b", "share one root", "allow_tied_outputs"});
        model.wait();
    }
    SECTION("parallel, allowed: the atomic writers accumulate") {
        cs::Model model = tied(cs::FusedSystem::Mode::parallel, true);
        model.setup();
        model.forward();
        model.wait();
        CHECK(tfl::out_of<1>(model, "sum") == std::vector<float>(12, 3.0f));
    }
    SECTION("sequential: no check, the writers accumulate") {
        cs::Model model = tied(cs::FusedSystem::Mode::sequential, false);
        model.setup();
        model.forward();
        model.wait();
        CHECK(tfl::out_of<1>(model, "sum") == std::vector<float>(12, 3.0f));
    }
}

// =============================================================================================
// Fused, both modes: value-driven sizes, the local-memory limit
// =============================================================================================
TEST_CASE("fused: a value-driven size driven by a stage of the same fuse throws at setup, in both modes",
          "[flavors][fused][errors]") {
    // (FromInputDim and unspecified dims through connections are allowed: the chains above use both.)
    const auto refused = [](cs::FusedSystem::Mode mode) {
        cs::Model model = tfl::make([mode](cs::System& s) {
            s.add_subsystem<tfl::Wave>("f", mode, std::optional<sycl::range<3>>{},
                                       std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                                           w.add_subsystem<tfl::NSrc>("a", 8.0f);
                                           w.add_subsystem<tfl::Bins>("b");
                                       }});
            s.connect("f.a.n", "f.b.n");  // b's output size is the VALUE of a's output
        });
        tfl::expect_throws([&] { model.setup(); }, {"f.b", "f.a", "VALUE"});
        model.wait();  // setup() may have submitted initializer work before it threw
    };
    SECTION("sequential") { refused(cs::FusedSystem::Mode::sequential); }
    SECTION("parallel") { refused(cs::FusedSystem::Mode::parallel); }
}

TEST_CASE("fused: a value-driven size driven from outside the fuse runs and follows the value, in both modes",
          "[flavors][fused]") {
    const auto run = [](cs::FusedSystem::Mode mode) {
        cs::Model model = tfl::make([mode](cs::System& s) {
            s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{100});
            s.add_subsystem<tfl::Wave>("f", mode, std::optional<sycl::range<3>>{},
                                       std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                                           w.add_subsystem<tfl::Elem<0>>("e");
                                           w.add_subsystem<tfl::Bins>("b");  // n is a free input
                                       }});
            s.connect("src.y", "f.e.x");
        });
        model.setup();
        const std::uint32_t nb1 = 8, nb2 = 13;
        for (const std::uint32_t nb : {nb1, nb2}) {  // the second value grows the output root
            model.init_input("f.b.n", cs::init::from_host(std::vector<std::uint32_t>{nb}));
            model.forward();
            model.wait();
            std::vector<float> want(nb);
            for (std::uint32_t i = 0; i < nb; ++i) want[i] = static_cast<float>(i + 1);
            CHECK(tfl::out_of<1>(model, "f.b.bins") == want);
        }
    };
    SECTION("sequential") { run(cs::FusedSystem::Mode::sequential); }
    SECTION("parallel") { run(cs::FusedSystem::Mode::parallel); }
}

TEST_CASE("fused: a wave's local memory over the device limit throws at forward, naming the scratch stages",
          "[flavors][fused][errors]") {
    const sycl::device dev{sycl::default_selector_v};  // the Model's default device
    const std::size_t  limit = dev.get_info<sycl::info::device::local_mem_size>();
    if (limit > (std::size_t{1} << 24)) SKIP("the device reports no usable local-memory limit (" << limit << " bytes)");
    // Each stage takes 60% of the limit: alone it fits, the two slices of a sequential wave do not,
    // and a parallel wave (disjoint groups, one shared slice) needs only the larger one.
    const std::size_t elems = limit / sizeof(float) * 6 / 10;
    const auto        make_wave = [elems](cs::FusedSystem::Mode mode) {
        return tfl::make([elems, mode](cs::System& s) {
            s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{256});
            s.add_subsystem<tfl::Wave>("f", mode, std::optional<sycl::range<3>>{},
                                       std::function<void(cs::FusedSystem&)>{[elems](cs::FusedSystem& w) {
                                           w.add_subsystem<tfl::BigStage>("a", elems);
                                           w.add_subsystem<tfl::BigStage>("b", elems);
                                       }});
            s.connect("src.y", "f.a.x");
            s.connect("src.y", "f.b.x");
        });
    };
    SECTION("sequential: the slices add up, over the limit") {
        cs::Model model = make_wave(cs::FusedSystem::Mode::sequential);
        model.setup();
        tfl::expect_throws([&] { model.forward(); }, {"f.a", "f.b", "bytes of local memory"});
        model.wait();
    }
    SECTION("parallel: one shared slice, within the limit") {
        cs::Model model = make_wave(cs::FusedSystem::Mode::parallel);
        model.setup();
        model.forward();
        model.wait();
        std::vector<float> x(256);
        for (int i = 0; i < 256; ++i) x[i] = static_cast<float>(i + 1);  // Src1's default offset is 1
        CHECK(tfl::out_of<1>(model, "f.a.y") == x);
        CHECK(tfl::out_of<1>(model, "f.b.y") == x);
    }
}

// =============================================================================================
// Fused, errors at setup
// =============================================================================================
TEST_CASE("fused: a custom-op child, a host child and mixed ranks throw at setup, naming the child",
          "[flavors][fused][errors]") {
    const auto bad_wave = [](auto bad_tag, bool bad_has_input) {
        using Bad = typename decltype(bad_tag)::type;
        return tfl::make([bad_has_input](cs::System& s) {
            s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{1000});
            s.add_subsystem<tfl::Wave>(
                "f", cs::FusedSystem::Mode::parallel, std::optional<sycl::range<3>>{},
                std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                    w.add_subsystem<tfl::Elem<0>>("a");  // a valid stage first, then the offender
                    if constexpr (std::is_same_v<Bad, tfl::Src2>)
                        w.add_subsystem<Bad>("bad", std::array<int, 2>{4, 4});
                    else
                        w.add_subsystem<Bad>("bad");
                }});
            s.connect("src.y", "f.a.x");
            if (bad_has_input) {
                s.add_subsystem<tfl::Src1>("src2", std::array<int, 1>{1000});
                s.connect("src2.y", "f.bad.x");
            }
        });
    };
    SECTION("custom-op child") {
        cs::Model model = bad_wave(std::type_identity<tfl::NopCustom>{}, true);
        tfl::expect_throws([&] { model.setup(); }, {"f.bad", "custom-op", "cannot be fused"});
        model.wait();  // setup() may have submitted initializer work before it threw
    }
    SECTION("host child") {
        cs::Model model = bad_wave(std::type_identity<tfl::NopHost>{}, true);
        tfl::expect_throws([&] { model.setup(); }, {"f.bad", "host", "cannot be fused"});
        model.wait();  // setup() may have submitted initializer work before it threw
    }
    SECTION("children of different ranks") {
        cs::Model model = bad_wave(std::type_identity<tfl::Src2>{}, false);
        tfl::expect_throws([&] { model.setup(); }, {"f.bad", "share one rank"});
        model.wait();  // setup() may have submitted initializer work before it threw
    }
}

// =============================================================================================
// Fused vs unfused
// =============================================================================================
TEST_CASE("fused vs unfused: four stages as a plain System and as a sequential FusedSystem, bitwise-equal",
          "[flavors][fused]") {
    constexpr int kN = 777;
    // Identical, non-round inputs: 0.37 * i - 11.1 (the Src is deterministic).
    const auto feed = [](cs::System& s) { s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{kN}, 0.37f, -11.1f); };

    cs::Model unfused = tfl::make([&](cs::System& s) {
        feed(s);
        s.add_subsystem<tfl::Elem<0>>("a");
        s.add_subsystem<tfl::Elem<1>>("b");
        s.add_subsystem<tfl::Elem<3>>("c");
        s.add_subsystem<tfl::Elem<2>>("d");
        s.connect("src.y", "a.x");
        s.connect("a.y", "b.x");
        s.connect("b.y", "c.x");
        s.connect("c.y", "d.x");
    });
    cs::Model fused = tfl::make([&](cs::System& s) {
        feed(s);
        s.add_subsystem<tfl::Wave>(
            "f", cs::FusedSystem::Mode::sequential, std::optional<sycl::range<3>>{},
            std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                w.add_subsystem<tfl::Elem<0>>("a");
                w.add_subsystem<tfl::Elem<1>>("b");
                w.add_subsystem<tfl::Elem<3>>("c");
                w.add_subsystem<tfl::Elem<2>>("d");
            }});
        s.connect("src.y", "f.a.x");
        s.connect("f.a.y", "f.b.x");
        s.connect("f.b.y", "f.c.x");
        s.connect("f.c.y", "f.d.x");
    });
    unfused.setup();
    fused.setup();
    tfl::run_three(unfused);
    tfl::run_three(fused);

    const auto u = tfl::out_of<1>(unfused, "d.y");
    const auto f = tfl::out_of<1>(fused, "f.d.y");
    REQUIRE(u.size() == static_cast<std::size_t>(kN));
    REQUIRE(f.size() == u.size());
    CHECK(std::memcmp(u.data(), f.data(), u.size() * sizeof(float)) == 0);

    // Not vacuous: the values match a host recomputation.
    std::vector<float> want(kN);
    for (int i = 0; i < kN; ++i) {
        const float x = 0.37f * static_cast<float>(i) + -11.1f;
        want[i]       = tfl::ref(2, tfl::ref(3, tfl::ref(1, tfl::ref(0, x))));
    }
    CHECK(cst::all_close(u, want));
}


// =============================================================================================
// Transient flavors: the children release after the group or wave
// =============================================================================================
TEST_CASE("transient: ParallelSystem and FusedSystem children release after the group or wave",
          "[flavors][transient]") {
    const bool fused = GENERATE(false, true);
    CAPTURE(fused);
    cs::Model model = cs::create_model<tfl::Top>(cst::pooled_config(), tfl::Build{[fused](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{tfl::kLen});
        const cs::property_list props = fused ? cs::property_list{cs::property::fused{}, cs::property::transient{}}
                                              : cs::property_list{cs::property::parallel{}, cs::property::transient{}};
        s.add_subsystem("grp", props, s.child<tfl::Elem<0>>("a"), s.child<tfl::Elem<0>>("b"));
        s.connect("src.y", "grp.a.x");
        s.connect("src.y", "grp.b.x");
    }});
    model.setup();
    // A FusedSystem keeps its wave staging in member usm_vectors drawn from the pool, so the
    // baseline is taken after the FIRST forward: what the group's roots held is then gone, what the
    // fuse stages stays. The roots themselves are checked directly after that first forward.
    model.forward();
    model.wait();
    for (const char* p : {"grp.a.y", "grp.b.y"}) {
        INFO(p);
        CHECK(model.root().entry_at(p).root()->lifetime_count() == 0);
        CHECK(!model.root().entry_at(p).root()->has_storage());
    }
    const auto base = cst::pool_in_use(model);
    model.forward();
    model.forward();
    model.wait();
    CHECK(cst::pool_in_use(model) == base);  // every later forward returns what it allocated
    tfl::expect_throws([&] { (void)model.view_output<DType::f32, 1>("grp.a.y"); }, {"grp"});
}

TEST_CASE("write epochs: Fused and Parallel systems give the same deltas as the plain System (+1 per written root per forward)",
          "[flavors][epoch]") {
    const auto delta_of = [](cs::Model& m, const char* path) {
        const auto before = m.root().entry_at(path).write_epoch();
        m.forward();
        m.wait();
        return m.root().entry_at(path).write_epoch() - before;
    };
    // plain: src -> a -> b at the top level
    cs::Model plain = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{1000});
        s.add_subsystem<tfl::Elem<0>>("a");
        s.add_subsystem<tfl::Elem<1>>("b");
        s.connect("src.y", "a.x");
        s.connect("a.y", "b.x");
    });
    plain.setup();
    // fused sequential: the same two stages in a wave
    cs::Model fused = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{1000});
        s.add_subsystem<tfl::Wave>("f", cs::FusedSystem::Mode::sequential, std::optional<sycl::range<3>>{},
                                   std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                                       w.add_subsystem<tfl::Elem<0>>("a");
                                       w.add_subsystem<tfl::Elem<1>>("b");
                                   }});
        s.connect("src.y", "f.a.x");
        s.connect("f.a.y", "f.b.x");
    });
    fused.setup();
    // fused parallel: independent stages
    cs::Model fpar = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{1000});
        s.add_subsystem<tfl::Wave>("f", cs::FusedSystem::Mode::parallel, std::optional<sycl::range<3>>{},
                                   std::function<void(cs::FusedSystem&)>{[](cs::FusedSystem& w) {
                                       w.add_subsystem<tfl::Elem<0>>("a");
                                       w.add_subsystem<tfl::Elem<2>>("b");
                                   }});
        s.connect("src.y", "f.a.x");
        s.connect("src.y", "f.b.x");
    });
    fpar.setup();
    // typed ParallelSystem: independent children
    cs::Model par = tfl::make([](cs::System& s) {
        s.add_subsystem<tfl::Src1>("src", std::array<int, 1>{1000});
        s.add_subsystem<tfl::Par>("f", std::function<void(cs::ParallelSystem&)>{[](cs::ParallelSystem& p) {
            p.add_subsystem<tfl::Elem<0>>("a");
            p.add_subsystem<tfl::Elem<2>>("b");
        }});
        s.connect("src.y", "f.a.x");
        s.connect("src.y", "f.b.x");
    });
    par.setup();
    for (cs::Model* m : {&plain, &fused, &fpar, &par}) {
        m->forward();  // the first forward allocates
        m->wait();
    }
    for (int r = 0; r < 3; ++r) {
        CHECK(delta_of(plain, "src.y") == 1);
        CHECK(delta_of(plain, "a.y") == 1);
        CHECK(delta_of(plain, "b.y") == 1);
        CHECK(delta_of(fused, "src.y") == 1);
        CHECK(delta_of(fused, "f.a.y") == 1);
        CHECK(delta_of(fused, "f.b.y") == 1);
        CHECK(delta_of(fpar, "f.a.y") == 1);
        CHECK(delta_of(fpar, "f.b.y") == 1);
        CHECK(delta_of(par, "f.a.y") == 1);
        CHECK(delta_of(par, "f.b.y") == 1);
    }
}