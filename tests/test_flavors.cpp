// Fused-component and independent-children tests, every case through a cospan::Model.
//
// ParallelSystem and FusedSystem are gone (docs/rework-migration-2026-10-03.md). A fused group is a component built by
// cs::fuse_components<seq|par>(name, connects_desc({...}), subsystem_desc<...>(...)...) and its ports are reached by path
// ("f.a.x"); a plain System of the same children is cs::system_from(name, ...). Concurrency of independent children
// comes from data dependencies alone (design review, section 3).
//
//   Independent    plain + group with scratch + host children in a system_from group, fed by a producer and read by a
//                  consumer; a group nested under a plain System
//   Fused, seq.    three plain stages; two group+scratch stages (B's own barrier, own local slices); rank-2 and rank-3
//                  waves; stage ranges may differ; the stages' work-group shapes must agree (section 4)
//   Fused, par.    independent plain stages, independent group stages; a connection throws at setup; atomic tied outputs
//                  accumulate, plain ones need allow_plain (the ordinary tie policy, section 9)
//   Fused, both    value-driven sizes (refused when driven by a stage of the same fuse, allowed from outside); the
//                  local-memory limit (skipped where the device reports none, e.g. omp)
//   Equivalence    four stages as a plain System and as a sequential fused component give bitwise-equal outputs
//   Lifetime       a transient group is deleted after its turn; a transient reader of a deleted do_once producer
//
// The compile-time rules of a fused component (no custom-op or host stages, all-group or all-plain, one rank) are
// static_asserts in FusedComponent now; they have no runtime test. The components live in namespace tfl.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
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

/// A synced host copy of an output (the model has run: views exist only after the first forward, section 5).
template <int R>
std::vector<float> out_of(cs::Model& m, const char* path) {
    return to_vector(cst::synced_output<DType::f32, R>(m, path));
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
        this->declare_output(y_, "y", cs::unspecified_shape<R>());
    }
    void pre_forward() override {  // the output follows the input (section 10: no derived dims), one entry per dimension
        const sycl::range<R> s = this->shape_of(x_);
        [&]<std::size_t... I>(std::index_sequence<I...>) { this->set_shape(y_, s[I]...); }(std::make_index_sequence<R>{});
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
        declare_output(y_, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(y_, shape_of(x_)[0]); }  // the output follows the input (section 10: no derived dims)
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
        this->declare_output(y_, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { this->set_shape(y_, this->shape_of(x_)[0]); }  // the output follows the input (section 10: no derived dims)
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
        declare_output(z_, "z", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(z_, shape_of(x_)[0]); }  // the output follows the input (section 10: no derived dims)
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
        // unspecified: the producers set the sizes (a fixed input would make a connected producer's number an error, P5)
        declare_input(p_, "p", cs::unspecified_shape<1>());
        declare_input(g_, "g", cs::unspecified_shape<1>());
        declare_input(h_, "h", cs::unspecified_shape<1>());
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

// ---- NSrc / Bins: a value-driven size. NSrc writes the value n; Bins' output has n elements -----
// REVIEW: a component cannot read a value and a hook cannot target an output (section 10), so Bins takes a free input `ref`
// that its System hooks with from_value(n) (hook_bins), and copies its shape in pre_forward.
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
    void pre_forward() override { set_shape(bins_, shape_of(ref_)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(bins_.shape(), {n_, ref_}, {bins_}, {});
    }
};
/// Hooks the `ref` of the Bins at `path` (e.g. "f.b") with the value of its `n`.
inline void hook_bins(cs::System& s, const std::string& path) {
    s.shape_hook(path + ".ref", cs::dim::from_value(path + ".n", 0));
}

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
        declare_output(y_, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(y_, shape_of(x_)[0]); }  // the output follows the input (section 10: no derived dims)
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
        declare_output(y_, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(y_, shape_of(x_)[0]); }  // the output follows the input (section 10: no derived dims)
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
        declare_output(y_, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(y_, shape_of(x_)[0]); }  // the output follows the input (section 10: no derived dims)
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config({x_}, {y_}, {}); }
};

constexpr int  kLen = 12;  // the group cases; the group body's work-group is 8, so the last group is ragged
constexpr auto kSeq = cs::fuse_strategy::seq;
constexpr auto kPar = cs::fuse_strategy::par;

/// src -> two BigStage children (a, b) in a fused component "f" of strategy S; each stage's scratch tile has `elems` floats.
template <cs::fuse_strategy S>
cs::Model big_wave(std::size_t elems) {
    return make([elems](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<Src1>("src", std::array<int, 1>{256}));
        s.add_subsystem(cs::fuse_components<S>("f", cs::subsystem_desc<BigStage>("a", elems), cs::subsystem_desc<BigStage>("b", elems)));
        s.connect("src.y", "f.a.x");
        s.connect("src.y", "f.b.x");
    });
}

/// zero (ZeroAcc) and a fused "f" of two atomic Adders (1 and 2) tied into one root "sum"; f depends on zero.
template <cs::fuse_strategy S>
cs::Model tied_adders() {
    return make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<ZeroAcc>("zero"));
        s.add_subsystem(cs::fuse_components<S>("f", cs::subsystem_desc<Adder>("a", 1.0f), cs::subsystem_desc<Adder>("b", 2.0f)));
        s.tie_outputs<DType::atomic_f32, 1>({"zero.acc", "f.a.acc", "f.b.acc"}, "sum", cs::shape(12));  // atomic writers
        // Execution order is data dependencies only and tied writers never wait on each other (design review, section 3):
        // the zeroing producer needs an explicit edge.
        s.depends_on("f", "zero");
    });
}

/// A value-driven size driven from OUTSIDE the fuse: it runs and follows the value.
template <cs::fuse_strategy S>
void value_from_outside() {
    cs::Model model = make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<Src1>("src", std::array<int, 1>{100}));
        s.add_subsystem(cs::fuse_components<S>("f", cs::subsystem_desc<Elem<0>>("e"), cs::subsystem_desc<Bins>("b")));  // n is a free input
        s.connect("src.y", "f.e.x");
        hook_bins(s, "f.b");
    });
    model.setup();
    const std::uint32_t nb1 = 8, nb2 = 13;
    for (const std::uint32_t nb : {nb1, nb2}) {  // the second value grows the output
        model.update_input("f.b.n", cs::init::from_host(std::vector<std::uint32_t>{nb}));
        model.forward();
        model.wait();
        std::vector<float> want(nb);
        for (std::uint32_t i = 0; i < nb; ++i) want[i] = static_cast<float>(i + 1);
        CHECK(out_of<1>(model, "f.b.bins") == want);
    }
}

}  // namespace tfl

// =============================================================================================
// Independent children (was Parallel)
// =============================================================================================
TEST_CASE("independent children: plain + group with scratch + host in a system_from group, three forwards then one wait",
          "[flavors][group]") {
    // Was ParallelSystem's helper form. Concurrency of independent children comes from data dependencies (section 3), so the
    // behavior worth keeping is the values: a plain System of the same children.
    cs::Model model = tfl::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{tfl::kLen}));
        s.add_subsystem(cs::system_from("par", cs::subsystem_desc<tfl::Elem<0>>("plain"),
                                        cs::subsystem_desc<tfl::GroupStage<tfl::RevBody>>("group", std::size_t{8}),
                                        cs::subsystem_desc<tfl::Prefix>("host")));
        s.add_subsystem(cs::subsystem_desc<tfl::Sum3>("sum", tfl::kLen));
        s.connect("src.y", "par.plain.x");
        s.connect("src.y", "par.group.x");
        s.connect("src.y", "par.host.x");
        s.connect("par.plain.y", "sum.p");
        s.connect("par.group.y", "sum.g");
        s.connect("par.host.z", "sum.h");
    });
    model.setup();

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

TEST_CASE("independent children: a system_from group nested under a plain System, values are correct", "[flavors][group]") {
    cs::Model model = tfl::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfl::Top>("mid", tfl::Build{[](cs::System& m) {
            m.add_subsystem(cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{tfl::kLen}));
            m.add_subsystem(cs::system_from("par", cs::subsystem_desc<tfl::Elem<0>>("a"), cs::subsystem_desc<tfl::Elem<2>>("b")));
            m.add_subsystem(cs::subsystem_desc<tfl::Elem<1>>("c"));
            m.connect("src.y", "par.a.x");
            m.connect("src.y", "par.b.x");
            m.connect("par.a.y", "c.x");  // a consumer after the group
        }}));
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
        s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{1000}));
        s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src2", std::array<int, 1>{333}));
        s.add_subsystem(cs::fuse_components<tfl::kSeq>("f", cs::connects_desc({{"a.y", "b.x"}}),
                                                       cs::subsystem_desc<tfl::Elem<0>>("a"),
                                                       cs::subsystem_desc<tfl::Elem<1>>("b"),
                                                       cs::subsystem_desc<tfl::Elem<2>>("c")));
        s.connect("src.y", "f.a.x");
        s.connect("src2.y", "f.c.x");
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
        s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{200}));
        s.add_subsystem(cs::fuse_components<tfl::kSeq>("f", cs::connects_desc({{"a.y", "b.x"}}),
                                                       cs::subsystem_desc<tfl::GroupStage<tfl::RevBody>>("a"),
                                                       cs::subsystem_desc<tfl::GroupStage<tfl::MirrorBody>>("b")));
        s.connect("src.y", "f.a.x");
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

TEST_CASE("fused sequential: a rank-2 wave and a rank-3 wave (7x9x11), the default work-group shape", "[flavors][fused]") {
    // The fuse-level work-group override ({8,8}) is gone: group shape belongs to the component (design review, section 4).
    // The rank-2 section therefore runs with the default shape for its rank.
    SECTION("rank 2, 40 x 30") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tfl::Src2>("src", std::array<int, 2>{40, 30}, 1.0f, 0.0f));
            s.add_subsystem(cs::fuse_components<tfl::kSeq>("f", cs::connects_desc({{"a.y", "b.x"}}),
                                                           cs::subsystem_desc<tfl::ElemND<2, 0>>("a"),
                                                           cs::subsystem_desc<tfl::ElemND<2, 4>>("b")));
            s.connect("src.y", "f.a.x");
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
    SECTION("rank 3, 7 x 9 x 11") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tfl::Src3>("src", std::array<int, 3>{7, 9, 11}, 1.0f, 0.0f));
            s.add_subsystem(cs::fuse_components<tfl::kSeq>("f", cs::connects_desc({{"a.y", "b.x"}}),
                                                           cs::subsystem_desc<tfl::ElemND<3, 0>>("a"),
                                                           cs::subsystem_desc<tfl::ElemND<3, 4>>("b")));
            s.connect("src.y", "f.a.x");
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
        s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{1000}));
        s.add_subsystem(cs::fuse_components<tfl::kSeq>("f", cs::connects_desc({{"a.y", "b.x"}}),
                                                       cs::subsystem_desc<tfl::Elem<0>>("a"),   // launches over 1000
                                                       cs::subsystem_desc<tfl::Half>("b")));    // launches over 500, reads a's output at the same index
        s.connect("src.y", "f.a.x");
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

TEST_CASE("fused: stages asking for different work-group shapes throw at forward, naming both", "[flavors][fused][errors]") {
    // New (design review, section 4: group shape belongs to the component): the wave has ONE shape, so two stages that ask
    // for different ones are refused when the launches are collected.
    const auto build = [](auto strategy_tag) {
        constexpr cs::fuse_strategy S = decltype(strategy_tag)::value;
        return tfl::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{256}));
            s.add_subsystem(cs::fuse_components<S>("f", cs::subsystem_desc<tfl::GroupStage<tfl::RevBody>>("a", std::size_t{32}),
                                                   cs::subsystem_desc<tfl::GroupStage<tfl::RevBody>>("b", std::size_t{64})));
            s.connect("src.y", "f.a.x");
            s.connect("src.y", "f.b.x");
        });
    };
    SECTION("sequential") {
        cs::Model model = build(std::integral_constant<cs::fuse_strategy, tfl::kSeq>{});
        model.setup();
        tfl::expect_throws([&] { model.forward(); }, {"f.a", "f.b", "different work-group shapes"});
        model.wait();
    }
    SECTION("parallel") {
        cs::Model model = build(std::integral_constant<cs::fuse_strategy, tfl::kPar>{});
        model.setup();
        tfl::expect_throws([&] { model.forward(); }, {"f.a", "f.b", "different work-group shapes"});
        model.wait();
    }
}

// =============================================================================================
// Fused, parallel
// =============================================================================================
TEST_CASE("fused parallel: independent stages of different sizes, three forwards", "[flavors][fused]") {
    // A fused component's stages are all plain or all work-group bodies (a static_assert), so the old plain + group mix is
    // two sections: three plain stages, and two group+scratch stages.
    SECTION("plain stages") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src1", std::array<int, 1>{1000}));
            s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src2", std::array<int, 1>{257}));
            s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src3", std::array<int, 1>{640}));
            s.add_subsystem(cs::fuse_components<tfl::kPar>("f", cs::subsystem_desc<tfl::Elem<0>>("a"),   // 1000
                                                           cs::subsystem_desc<tfl::Elem<2>>("c"),         // 257
                                                           cs::subsystem_desc<tfl::Elem<4>>("d")));       // 640
            s.connect("src1.y", "f.a.x");
            s.connect("src2.y", "f.c.x");
            s.connect("src3.y", "f.d.x");
        });
        model.setup();
        tfl::run_three(model);
        std::vector<float> want_a(1000), want_c(257), want_d(640);
        for (int i = 0; i < 1000; ++i) want_a[i] = tfl::ref(0, static_cast<float>(i + 1));
        for (int i = 0; i < 257; ++i) want_c[i] = tfl::ref(2, static_cast<float>(i + 1));
        for (int i = 0; i < 640; ++i) want_d[i] = tfl::ref(4, static_cast<float>(i + 1));
        CHECK(tfl::out_of<1>(model, "f.a.y") == want_a);
        CHECK(tfl::out_of<1>(model, "f.c.y") == want_c);
        CHECK(tfl::out_of<1>(model, "f.d.y") == want_d);
    }
    SECTION("group stages with scratch") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src1", std::array<int, 1>{640}));
            s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src2", std::array<int, 1>{200}));
            s.add_subsystem(cs::fuse_components<tfl::kPar>("f", cs::subsystem_desc<tfl::GroupStage<tfl::RevBody>>("r"),
                                                           cs::subsystem_desc<tfl::GroupStage<tfl::MirrorBody>>("m")));
            s.connect("src1.y", "f.r.x");
            s.connect("src2.y", "f.m.x");
        });
        model.setup();
        tfl::run_three(model);
        std::vector<float> x1(640), x2(200);
        for (int i = 0; i < 640; ++i) x1[i] = static_cast<float>(i + 1);
        for (int i = 0; i < 200; ++i) x2[i] = static_cast<float>(i + 1);
        CHECK(tfl::out_of<1>(model, "f.r.y") == tfl::reverse_in_groups(x1, 64));
        CHECK(tfl::out_of<1>(model, "f.m.y") == tfl::reverse_in_groups(x2, 64));  // Mirror reads the mirrored element too
    }
}

TEST_CASE("fused parallel: a connect between stages throws at setup, naming both", "[flavors][fused][errors]") {
    // Was a forward-time error ("run unordered"). The connection is made inside the fused component's setup, and a par
    // component rejects it there (fused_component.hpp: FusedComponent::setup).
    SECTION("an input with an unspecified dim takes its shape from a sibling's output") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components<tfl::kPar>("par", cs::connects_desc({{"a.y", "b.x"}}),
                                                           cs::subsystem_desc<tfl::Src1>("a", std::array<int, 1>{4}),
                                                           cs::subsystem_desc<tfl::Elem<0>>("b")));
        });
        tfl::expect_throws([&] { model.setup(); }, {"par", "connect", "stages run unordered", "'a.y' -> 'b.x'"});
    }
    SECTION("a fixed-shape input") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components<tfl::kPar>("par", cs::connects_desc({{"a.y", "b.p"}}),  // not a shape dependency, still a connection
                                                           cs::subsystem_desc<tfl::Src1>("a", std::array<int, 1>{4}),
                                                           cs::subsystem_desc<tfl::Sum3>("b", 4)));
        });
        tfl::expect_throws([&] { model.setup(); }, {"par", "connect", "stages run unordered", "'a.y' -> 'b.p'"});
    }
    SECTION("a chain A -> B") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{1000}));
            s.add_subsystem(cs::fuse_components<tfl::kPar>("f", cs::connects_desc({{"a.y", "b.x"}}),  // B reads A's output
                                                           cs::subsystem_desc<tfl::Elem<0>>("a"),
                                                           cs::subsystem_desc<tfl::Elem<1>>("b")));
            s.connect("src.y", "f.a.x");
        });
        tfl::expect_throws([&] { model.setup(); }, {"f", "connect", "stages run unordered", "'a.y' -> 'b.x'"});
    }
    // (allow_tied_outputs() is gone, so its "does not lift it" section is deleted.)
}

TEST_CASE("fused: tied outputs follow the ordinary tie policy, in parallel and sequential mode",
          "[flavors][fused][errors]") {
    SECTION("parallel, atomic: the writers accumulate") {
        cs::Model model = tfl::tied_adders<tfl::kPar>();
        model.setup();
        model.forward();
        model.wait();
        CHECK(tfl::out_of<1>(model, "sum") == std::vector<float>(12, 3.0f));  // 1 + 2
    }
    // Par fused components follow the ordinary tie policy and add no check of their own (design review, section 9, Jack
    // 2026-10-03): allow_plain is the "these writers are disjoint" flag, require_atomic refuses plain members at the call.
    SECTION("parallel, plain members with allow_plain: accepted, and it runs") {
        cs::Model model = tfl::make([](cs::System& s) {
            // identical writers (both y[i] = i + 1), so the result does not depend on how they interleave
            s.add_subsystem(cs::fuse_components<tfl::kPar>("f", cs::subsystem_desc<tfl::Src1>("a", std::array<int, 1>{12}),
                                                           cs::subsystem_desc<tfl::Src1>("b", std::array<int, 1>{12})));
            s.tie_outputs<DType::f32, 1>({"f.a.y", "f.b.y"}, "t", cs::shape(12), cs::TieOutputsPolicy::allow_plain);
        });
        CHECK_NOTHROW(model.setup());
        CHECK_NOTHROW(model.forward());
        model.wait();
        std::vector<float> want(12);
        for (int i = 0; i < 12; ++i) want[i] = static_cast<float>(i + 1);
        CHECK(tfl::out_of<1>(model, "t") == want);
    }
    SECTION("parallel, plain members with require_atomic (the default): tie_outputs throws at setup") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components<tfl::kPar>("f", cs::subsystem_desc<tfl::Src1>("a", std::array<int, 1>{12}),
                                                           cs::subsystem_desc<tfl::Src1>("b", std::array<int, 1>{12})));
            s.tie_outputs<DType::f32, 1>({"f.a.y", "f.b.y"}, "t", cs::shape(12));
        });
        tfl::expect_throws([&] { model.setup(); }, {"is not atomic", "require_atomic", "allow_plain"});
    }
    SECTION("sequential: no check, the writers accumulate") {
        cs::Model model = tfl::tied_adders<tfl::kSeq>();
        model.setup();
        model.forward();
        model.wait();
        CHECK(tfl::out_of<1>(model, "sum") == std::vector<float>(12, 3.0f));
    }
}

// =============================================================================================
// Fused, both modes: value-driven sizes, the local-memory limit
// =============================================================================================
TEST_CASE("fused: a value-driven size driven by a stage of the same fuse is refused at setup", "[flavors][fused][errors]") {
    // (Unspecified dims through connections are allowed: the chains above use them; derived dims are gone, section 10.)
    SECTION("sequential: the setup-end check names both stages") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components<tfl::kSeq>("f", cs::connects_desc({{"a.n", "b.n"}}),  // b's output size is the VALUE of a's output
                                                           cs::subsystem_desc<tfl::NSrc>("a", 8.0f),
                                                           cs::subsystem_desc<tfl::Bins>("b")));
            tfl::hook_bins(s, "f.b");
        });
        // P5: the source-ordering check is gone; the hook runs before the child f that holds its target and the model sets up.
        CHECK_NOTHROW(model.setup());
    }
    SECTION("parallel: the connect itself is rejected, before that check") {
        cs::Model model = tfl::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components<tfl::kPar>("f", cs::connects_desc({{"a.n", "b.n"}}),
                                                           cs::subsystem_desc<tfl::NSrc>("a", 8.0f),
                                                           cs::subsystem_desc<tfl::Bins>("b")));
        });
        tfl::expect_throws([&] { model.setup(); }, {"f", "connect", "stages run unordered"});
    }
}

TEST_CASE("fused: a value-driven size driven from outside the fuse runs and follows the value, in both modes",
          "[flavors][fused]") {
    SECTION("sequential") { tfl::value_from_outside<tfl::kSeq>(); }
    SECTION("parallel") { tfl::value_from_outside<tfl::kPar>(); }
}

TEST_CASE("fused: a wave's local memory over the device limit throws at forward, naming the scratch stages",
          "[flavors][fused][errors]") {
    const sycl::device dev{sycl::default_selector_v};  // the runtime's default device
    const std::size_t  limit = dev.get_info<sycl::info::device::local_mem_size>();
    if (limit > (std::size_t{1} << 24)) SKIP("the device reports no usable local-memory limit (" << limit << " bytes)");
    // Each stage takes 60% of the limit: alone it fits, the two slices of a sequential wave do not,
    // and a parallel wave (disjoint groups, one shared slice) needs only the larger one.
    const std::size_t elems = limit / sizeof(float) * 6 / 10;
    SECTION("sequential: the slices add up, over the limit") {
        cs::Model model = tfl::big_wave<tfl::kSeq>(elems);
        model.setup();
        tfl::expect_throws([&] { model.forward(); }, {"f.a", "f.b", "bytes of local memory"});
        model.wait();
    }
    SECTION("parallel: one shared slice, within the limit") {
        cs::Model model = tfl::big_wave<tfl::kPar>(elems);
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
// Fused vs unfused
// =============================================================================================
TEST_CASE("fused vs unfused: four stages as a plain System and as a sequential fused component, bitwise-equal",
          "[flavors][fused]") {
    constexpr int kN = 777;
    // Identical, non-round inputs: 0.37 * i - 11.1 (the Src is deterministic).
    const auto feed = [](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{kN}, 0.37f, -11.1f)); };

    cs::Model unfused = tfl::make([&](cs::System& s) {
        feed(s);
        s.add_subsystem(cs::subsystem_desc<tfl::Elem<0>>("a"));
        s.add_subsystem(cs::subsystem_desc<tfl::Elem<1>>("b"));
        s.add_subsystem(cs::subsystem_desc<tfl::Elem<3>>("c"));
        s.add_subsystem(cs::subsystem_desc<tfl::Elem<2>>("d"));
        s.connect("src.y", "a.x");
        s.connect("a.y", "b.x");
        s.connect("b.y", "c.x");
        s.connect("c.y", "d.x");
    });
    cs::Model fused = tfl::make([&](cs::System& s) {
        feed(s);
        s.add_subsystem(cs::fuse_components<tfl::kSeq>("f", cs::connects_desc({{"a.y", "b.x"}, {"b.y", "c.x"}, {"c.y", "d.x"}}),
                                                       cs::subsystem_desc<tfl::Elem<0>>("a"),
                                                       cs::subsystem_desc<tfl::Elem<1>>("b"),
                                                       cs::subsystem_desc<tfl::Elem<3>>("c"),
                                                       cs::subsystem_desc<tfl::Elem<2>>("d")));
        s.connect("src.y", "f.a.x");
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
// Lifetime
// =============================================================================================
TEST_CASE("transient: a system_from group and a fused component release after their turn", "[flavors][transient]") {
    const bool fused = GENERATE(false, true);
    CAPTURE(fused);
    cs::Model model = cs::create_model<tfl::Top>(cst::pooled_config(), tfl::Build{[fused](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{tfl::kLen}));
        const cs::property_list props{cs::property::transient{}};
        if (fused)
            s.add_subsystem(props, cs::fuse_components<tfl::kPar>("grp", cs::subsystem_desc<tfl::Elem<0>>("a"), cs::subsystem_desc<tfl::Elem<0>>("b")));
        else
            s.add_subsystem(props, cs::system_from("grp", cs::subsystem_desc<tfl::Elem<0>>("a"), cs::subsystem_desc<tfl::Elem<0>>("b")));
        s.connect("src.y", "grp.a.x");
        s.connect("src.y", "grp.b.x");
    }});
    model.setup();
    // A fused component keeps its wave staging in member usm_vectors drawn from the pool, so the baseline is taken after the
    // FIRST forward: what the group's variables held is then gone (the group is deleted with its staging).
    model.forward();
    model.wait();
    CHECK(model.get_subsystem("grp") == nullptr);  // the group is deleted after its turn, its variables with it
    for (const char* p : {"grp.a.y", "grp.b.y"}) {
        INFO(p);
        tfl::expect_throws([&] { (void)cst::synced_output<DType::f32, 1>(model, p); }, {"was deleted"});
    }
    // The pool's bytes in use are the sum of the live containers' capacities rounded up to whole nodes. With the group gone
    // only the eternal src.y is left: 12 floats (48 B) are one node.
    const auto base = cst::pool_in_use(model);
    CHECK(base == cst::pool_bytes_of(tfl::kLen * sizeof(float)));
    CHECK(base == cst::pool_node_bytes);
    model.forward();
    model.forward();
    model.wait();
    CHECK(cst::pool_in_use(model) == base);  // every later forward returns what it allocated
}

TEST_CASE("replay: a transient host reader of a deleted do_once producer reads its last value on every forward",
          "[flavors][transient]") {
    // src.y is written once (do_once, then deleted). The eternal e1 keeps its variable alive. The transient host-kind Prefix is
    // rebuilt every forward and reads the producer's last value (nothing runs the producer again).
    cs::Model model = tfl::make([](cs::System& s) {
        s.add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<tfl::Src1>("src", std::array<int, 1>{tfl::kLen}));
        s.add_subsystem(cs::subsystem_desc<tfl::Elem<0>>("e1"));
        s.add_subsystem(cs::property_list{cs::property::transient{}}, cs::subsystem_desc<tfl::Prefix>("host"));
        s.add_subsystem(cs::subsystem_desc<tfl::Elem<0>>("e2"));
        s.connect("src.y", "e1.x");
        s.connect("src.y", "host.x");
        s.connect("host.z", "e2.x");
    });
    model.setup();
    std::vector<float> want(tfl::kLen);
    float              acc = 0.0f;
    for (int i = 0; i < tfl::kLen; ++i) {
        acc += static_cast<float>(i + 1);
        want[i] = 2.0f * acc + 1.0f;  // e2.y = 2 z + 1, z = the prefix sum of x = i + 1
    }
    for (int fwd = 1; fwd <= 3; ++fwd) {
        INFO("forward " << fwd);
        model.forward();
        model.wait();
        CHECK(tfl::out_of<1>(model, "e2.y") == want);
    }
    CHECK(model.get_subsystem("src") == nullptr);   // the producer ran once and is gone
    CHECK(model.get_subsystem("host") == nullptr);  // the transient host stage is deleted at the end of each forward
}
