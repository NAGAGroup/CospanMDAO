// Smoke check for FusedSystem, sequential mode (not the test suite), r01 §5.4-§5.7, §7.5.
// Driven by the shared test-only RootDriver (driver.hpp). All checks are against host references.
//
//   a. A wave of three PLAIN stages (typed form): A y = 2x + 1 (N = 1000), B z = y*y
//      (element-wise on A's output), C w = x2 + 3 on a DIFFERENT, ragged size (333).
//      One forward, then three back to back without waiting.
//   b. A wave of two GROUP stages with Scratch (N = 200, work-group 64, ragged last group):
//      A reverses within each work-group through Local; B reads the MIRRORED element of A's
//      output at once, which crosses work-items, so it is only right if the barrier the
//      FusedSystem inserts between the two scratch stages is really there.
//   c. A rank-2 wave (40 x 30) of two plain stages, the FusedSystem's work-group {8,8}.
//   d. The helper form, property::fused{sequential}, builds a FusedSystem and runs A + B.

#include <cmath>
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

constexpr std::size_t kN  = 1000;
constexpr std::size_t kN2 = 333;
constexpr std::size_t kGN = 200;  // case b
constexpr std::size_t kWg = 64;   // case b work-group

template <int Rank>
static std::vector<float> values_of(sk::queue& q, cs::detail::IOEntry& e) {
    auto               h = HostViewAccess::view<DType::f32, Rank>(e, q);
    std::vector<float> v(h.size());
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = h[i];
    return v;
}

// ---- components ----------------------------------------------------------------------------
// Src: y[i] = i + 1, of a length given at construction.
struct SrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = static_cast<float>(i[0] + 1); }
};

class Src : public cs::Component<SrcBody> {
    cs::Output<DType::f32, 1> y_;
    std::size_t               n_;

public:
    explicit Src(std::size_t n) : n_{n} {}
    void setup() override { declare_output(y_, "y", static_cast<int>(n_)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {}, {y_}, {});
    }
};

// StepA (plain): y = 2x + 1. The shape follows the source.
struct StepABody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = 2.0f * in.x[i] + 1.0f; }
};

class StepA : public cs::Component<StepABody> {
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

// StepB (plain): z = y * y.
struct StepBBody {
    struct Inputs {
        cs::Input<DType::f32, 1> y;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> z;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.z[i] = in.y[i] * in.y[i]; }
};

class StepB : public cs::Component<StepBBody> {
    cs::Input<DType::f32, 1>  y_;
    cs::Output<DType::f32, 1> z_;

public:
    void setup() override {
        declare_input(y_, "y", cs::from_source_shape<1>());
        declare_output(z_, "z", cs::from_input_shape<1>("y"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {y_}, {z_}, {});
    }
};

// StepC (plain): w = x + 3.
struct StepCBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> w;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.w[i] = in.x[i] + 3.0f; }
};

class StepC : public cs::Component<StepCBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> w_;

public:
    void setup() override {
        declare_input(x_, "x", cs::from_source_shape<1>());
        declare_output(w_, "w", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {x_}, {w_}, {});
    }
};

// ---- a. three plain stages, typed form ----------------------------------------------------
class WaveA : public cs::FusedSystem {
public:
    WaveA() : cs::FusedSystem(Mode::sequential) {}
    void setup() override {
        add_subsystem<StepA>("a");
        add_subsystem<StepB>("b");
        add_subsystem<StepC>("c");
    }
};

class TopA : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("src", kN);
        add_subsystem<Src>("src2", kN2);
        add_subsystem<WaveA>("f");
        connect("src.y", "f.a.x");
        connect("src2.y", "f.c.x");
        connect("f.a.y", "f.b.y");  // B reads A's output element-wise
    }
};

static void case_a(const sycl::device& dev, sk::queue& q) {
    std::printf("-- a. sequential wave of three plain stages (A -> B element-wise, C on a ragged 333)\n");
    TopA       top;
    RootDriver d{dev, q, top};
    d.start();

    std::vector<float> want_y(kN), want_z(kN), want_w(kN2);
    for (std::size_t i = 0; i < kN; ++i) {
        want_y[i] = 2.0f * static_cast<float>(i + 1) + 1.0f;
        want_z[i] = want_y[i] * want_y[i];
    }
    for (std::size_t i = 0; i < kN2; ++i) want_w[i] = static_cast<float>(i + 1) + 3.0f;

    d.forward();
    check(values_of<1>(q, top.entry_at("f.a.y")) == want_y, "a: stage A: y == 2x + 1");
    check(values_of<1>(q, top.entry_at("f.b.z")) == want_z, "a: stage B: z == y*y (element-wise on A's output)");
    check(values_of<1>(q, top.entry_at("f.c.w")) == want_w, "a: stage C: w == x2 + 3 (ragged 333)");

    // Three forwards in a row, nothing waits between them (§7.3), then wait once.
    d.enqueue();
    d.enqueue();
    d.enqueue();
    d.wait();
    check(values_of<1>(q, top.entry_at("f.b.z")) == want_z, "a: three forwards enqueued back to back: z intact");
    check(values_of<1>(q, top.entry_at("f.c.w")) == want_w, "a: ... and w intact");
}

// ---- b. two group stages with Scratch -----------------------------------------------------
// Rev: within each work-group, y[base + l] = x[base + (n-1-l)], zero past the end.
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
        s.tile[l] = g < total ? in.x[g] : 0.0f;
        sycl::group_barrier(it.get_group());
        if (g < total) out.y[g] = s.tile[n - 1 - l];
    }
};

// Mirror: the same mapping, but it reads the MIRRORED element of its input straight
// away, which was written by ANOTHER work-item of the group in the previous stage.
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
        s.tile[l]               = 0.0f;  // touches the shared local buffer: this is a scratch stage
        const std::size_t m = g - l + (n - 1 - l);
        if (g < total) out.y[g] = m < total ? in.x[m] : 0.0f;
    }
};

template <class Body>
class GroupStage : public cs::Component<Body> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        this->declare_input(x_, "x", cs::from_source_shape<1>());
        this->declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        typename Body::Scratch s;
        s.tile.reshape(sycl::range<1>{kWg});
        return this->create_launch_config(x_.shape(), sycl::range<1>{kWg}, {x_}, {y_}, s, {});
    }
};

class WaveB : public cs::FusedSystem {
public:
    WaveB() : cs::FusedSystem(Mode::sequential) {}
    void setup() override {
        add_subsystem<GroupStage<RevBody>>("a");
        add_subsystem<GroupStage<MirrorBody>>("b");
    }
};

class TopB : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("src", kGN);
        add_subsystem<WaveB>("f");
        connect("src.y", "f.a.x");
        connect("f.a.y", "f.b.x");
    }
};

static std::vector<float> reverse_in_groups(const std::vector<float>& in, std::size_t wg) {
    std::vector<float> out(in.size());
    for (std::size_t g = 0; g < in.size(); ++g) {
        const std::size_t base = g / wg * wg, l = g - base, m = base + (wg - 1 - l);
        out[g]                 = m < in.size() ? in[m] : 0.0f;  // the ragged last group reads zeros
    }
    return out;
}

static void case_b(const sycl::device& dev, sk::queue& q) {
    std::printf("-- b. sequential wave of two group stages with Scratch (barrier inserted between them)\n");
    TopB       top;
    RootDriver d{dev, q, top};
    d.start();

    std::vector<float> x(kGN);
    for (std::size_t i = 0; i < kGN; ++i) x[i] = static_cast<float>(i + 1);
    const std::vector<float> want_y = reverse_in_groups(x, kWg);
    const std::vector<float> want_z = reverse_in_groups(want_y, kWg);

    d.forward();
    check(values_of<1>(q, top.entry_at("f.a.y")) == want_y, "b: stage A: reversed within each work-group (ragged last group)");
    check(values_of<1>(q, top.entry_at("f.b.y")) == want_z,
          "b: stage B: mirrored read of A's output is exact (the barrier between the scratch stages)");
}

// ---- c. a rank-2 wave ----------------------------------------------------------------------
constexpr int kRows = 40, kCols = 30;

struct InitBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    void operator()(sycl::id<2> i, Inputs, Outputs out) const { out.y[i] = static_cast<float>(i[0] * 100 + i[1]); }
};

class Init2 : public cs::Component<InitBody> {
    cs::Output<DType::f32, 2> y_;

public:
    void setup() override { declare_output(y_, "y", kRows, kCols); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {}, {y_}, {});
    }
};

struct Twice2Body {  // y = 2x + 1
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    void operator()(sycl::id<2> i, Inputs in, Outputs out) const { out.y[i] = 2.0f * in.x[i] + 1.0f; }
};

class Twice2 : public cs::Component<Twice2Body> {
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

struct Triple2Body {  // z = 3y
    struct Inputs {
        cs::Input<DType::f32, 2> y;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> z;
    };
    void operator()(sycl::id<2> i, Inputs in, Outputs out) const { out.z[i] = 3.0f * in.y[i]; }
};

class Triple2 : public cs::Component<Triple2Body> {
    cs::Input<DType::f32, 2>  y_;
    cs::Output<DType::f32, 2> z_;

public:
    void setup() override {
        declare_input(y_, "y", cs::from_source_shape<2>());
        declare_output(z_, "z", cs::from_input_shape<2>("y"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {y_}, {z_}, {});
    }
};

class WaveC : public cs::FusedSystem {
public:
    WaveC() : cs::FusedSystem(Mode::sequential, sycl::range<3>{8, 8, 1}) {}
    void setup() override {
        add_subsystem<Twice2>("a");
        add_subsystem<Triple2>("b");
    }
};

class TopC : public cs::System {
public:
    void setup() override {
        add_subsystem<Init2>("src");
        add_subsystem<WaveC>("f");
        connect("src.y", "f.a.x");
        connect("f.a.y", "f.b.y");
    }
};

static void case_c(const sycl::device& dev, sk::queue& q) {
    std::printf("-- c. rank-2 sequential wave (40 x 30), the FusedSystem's work-group {8,8}\n");
    TopC       top;
    RootDriver d{dev, q, top};
    d.start();

    std::vector<float> want_y(kRows * kCols), want_z(kRows * kCols);
    for (int i = 0; i < kRows; ++i)
        for (int j = 0; j < kCols; ++j) {
            const std::size_t f = static_cast<std::size_t>(i * kCols + j);
            want_y[f]           = 2.0f * static_cast<float>(i * 100 + j) + 1.0f;
            want_z[f]           = 3.0f * want_y[f];
        }
    d.forward();
    check(values_of<2>(q, top.entry_at("f.a.y")) == want_y, "c: rank-2 stage A: y == 2x + 1");
    check(values_of<2>(q, top.entry_at("f.b.z")) == want_z, "c: rank-2 stage B: z == 3y");
}

// ---- d. the helper form ----------------------------------------------------------------------
class TopD : public cs::System {
public:
    void setup() override {
        add_subsystem<Src>("src", kN);
        add_subsystem("f", cs::property_list{cs::property::fused{}}, child<StepA>("a"), child<StepB>("b"));
        connect("src.y", "f.a.x");
        connect("f.a.y", "f.b.y");
    }
};

static void case_d(const sycl::device& dev, sk::queue& q) {
    std::printf("-- d. helper form: property::fused{sequential} builds a FusedSystem (A + B)\n");
    TopD       top;
    RootDriver d{dev, q, top};
    d.start();
    check(dynamic_cast<cs::FusedSystem*>(&top.get_subsystem("f")) != nullptr,
          "d: property::fused built a FusedSystem");

    std::vector<float> want_y(kN), want_z(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        want_y[i] = 2.0f * static_cast<float>(i + 1) + 1.0f;
        want_z[i] = want_y[i] * want_y[i];
    }
    d.forward();
    check(values_of<1>(q, top.entry_at("f.a.y")) == want_y, "d: A: y == 2x + 1");
    check(values_of<1>(q, top.entry_at("f.b.z")) == want_z, "d: B: z == y*y");
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
    std::printf(failures == 0 ? "SMOKE FUSED PASS\n" : "SMOKE FUSED FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
