// Smoke check for the Model (not the test suite), r01 §9 and §11. Everything
// goes through cospan::Model; there is no RootDriver here.
//
//   a. Sellar-shaped model (the smoke_system case a Systems, copied here):
//      create_model<Top>(ModelConfig{}); setup; set_input on the free input
//      d1.y2 and on the tie aliases z and x; forward; wait; f / g1 / g2 against
//      a host reference; change x with set_input and forward again. The mark
//      spans hold 2 design variables, 1 objective, 2 constraints;
//      memory_report shows arena bytes on the device.
//   b. Two devices in one context: ModelConfig over {a non-OpenMP device, the
//      OpenMP host device}; src on the first (the default device), mid placed on
//      the second with property::device, out on the first again. Skipped when
//      the two are not both visible.
//   c. Errors: forward and host access before setup throw; a second setup
//      throws; a default_device outside the context throws at construction (when
//      two devices are visible); a moved Model still runs (its state is on the
//      heap) and the moved-from one throws.
//   d. view_output_mut: a producer writes only element 0 of a 4-element output;
//      the host writes the other three through a writable view, the forward
//      leaves them alone, and a consumer's sum reflects them.

#include <array>
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

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

static bool near(float got, float want) {
    return std::fabs(got - want) <= 1e-5f * std::fmax(1.0f, std::fabs(want));
}

static bool has(const char* haystack, const char* needle) {
    return std::string{haystack}.find(needle) != std::string::npos;
}

/// True when `fn` throws an std::exception whose message contains `needle`.
template <class F>
static bool throws_with(F&& fn, const char* needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        std::printf("      %s\n", e.what());
        return has(e.what(), needle);
    }
    return false;
}

// ---- the Sellar-shaped Systems (copied from smoke_system.cpp case a) --------------------------
constexpr float kInitY2   = 1.5f;   // the MDA System's init: the last word over d1's own
constexpr float kD1InitY2 = 99.0f;

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

/// One float, read through the Model.
static float scalar_out(cs::Model& m, const char* path) {
    auto v = m.view_output<DType::f32, 1>(path);
    return v.size() == 1 ? v[std::size_t{0}] : std::nanf("");
}

static std::vector<float> values_out(cs::Model& m, const char* path) {
    auto               v = m.view_output<DType::f32, 1>(path);
    const auto         s = v.span();
    return std::vector<float>(s.begin(), s.end());
}

// ---- a. Sellar-shaped forward through the Model -------------------------------------------------
struct SellarRef {
    float y1, y2, f, g1, g2;
};

static SellarRef sellar_reference(float z0, float z1, float x, float y2_in) {
    SellarRef r;
    r.y1 = z0 * z0 + z1 + x - 0.2f * y2_in;
    r.y2 = std::sqrt(r.y1) + z0 + z1;
    r.f  = x * x + z1 + r.y1 + std::exp(-r.y2);
    r.g1 = 3.16f - r.y1;
    r.g2 = r.y2 - 24.0f;
    return r;
}

static void check_sellar(cs::Model& m, const SellarRef& r, const char* tag) {
    const std::string t = tag;
    check(near(scalar_out(m, "mda.d1.y1"), r.y1), (t + ": y1 = z0^2 + z1 + x - 0.2*y2").c_str());
    check(near(scalar_out(m, "mda.d2.y2"), r.y2), (t + ": y2 = sqrt(y1) + z0 + z1").c_str());
    check(near(scalar_out(m, "obj.f"), r.f), (t + ": f (view_output)").c_str());
    check(near(scalar_out(m, "g1.g"), r.g1), (t + ": g1 = 3.16 - y1").c_str());
    check(near(scalar_out(m, "g2.g"), r.g2), (t + ": g2 = y2 - 24").c_str());
}

static void case_a() {
    std::printf("-- a. Sellar-shaped model through create_model<Top>(ModelConfig{})\n");
    const sycl::device dev{sycl::default_selector_v};
    auto               model = cs::create_model<Top>(cs::ModelConfig{});
    model.setup();

    // Drive the free input d1.y2 and the tie aliases z and x.
    const float z[2] = {4.0f, 3.0f};
    float       x    = 2.0f;
    const float y2in = 0.5f;
    model.set_input<DType::f32, 1>("mda.d1.y2", std::span<const float>{&y2in, 1});
    model.set_input<DType::f32, 1>("z", std::span<const float>{z, 2});
    model.set_input<DType::f32, 1>("x", std::span<const float>{&x, 1});
    check(model.view_input<DType::f32, 1>("mda.d1.y2")[std::size_t{0}] == y2in, "a: view_input reads back what set_input wrote (d1.y2)");
    check(model.view_input<DType::f32, 1>("mda.d2.z")[std::size_t{1}] == 3.0f, "a: the tied z reads back through d2.z");

    model.forward();
    model.wait();
    check_sellar(model, sellar_reference(z[0], z[1], x, y2in), "a1");

    x = 3.0f;  // change x and go again
    model.set_input<DType::f32, 1>("x", std::span<const float>{&x, 1});
    model.forward();
    model.wait();
    check_sellar(model, sellar_reference(z[0], z[1], x, y2in), "a2 (x changed)");

    // Marks (collected once, at the end of setup).
    check(model.design_vars().size() == 2, "a: 2 design variables");
    check(model.objectives().size() == 1, "a: 1 objective");
    check(model.constraints().size() == 2, "a: 2 constraints");
    const cs::detail::IOEntry* zr = model.root().entry_at("z").root();
    const cs::detail::IOEntry* xr = model.root().entry_at("x").root();
    bool has_z = false, has_x = false;
    for (const auto& d : model.design_vars()) {
        has_z = has_z || d.root == zr;
        has_x = has_x || d.root == xr;
    }
    check(has_z && has_x, "a: the design variables sit on the roots of z and x");
    check(model.objectives()[0].root == model.root().entry_at("obj.f").root(), "a: the objective sits on obj.f's root");
    bool upper0 = false;
    for (const auto& c : model.constraints())
        if (c.root == model.root().entry_at("g1.g").root())
            upper0 = c.bounds.upper && (*c.bounds.upper)[0] == 0.0;
    check(upper0, "a: g1's constraint is the later mark (upper 0.0)");

    // Retrieval.
    check(model.get_input<DType::f32, 1>("mda.d1.z").shape()[0] == 2, "a: get_input returns the typed wrapper (d1.z has 2 elements)");
    check(model.get_subsystem("mda").name() == "mda", "a: get_subsystem(\"mda\")");

    // Memory report: the root arrays are fixed-shape device roots, so the arena holds them.
    const auto rep = model.memory_report();
    const auto it  = rep.devices.find(dev);
    check(it != rep.devices.end() && it->second.arena_bytes > 0 && it->second.arena_bytes_in_use > 0 &&
              it->second.arena_bytes_in_use <= it->second.arena_bytes,
          "a: memory_report: arena bytes > 0 and in use on the device");
    if (it != rep.devices.end())
        std::printf("      arena %zu bytes, %zu in use; pool %zu bytes; OS allocations %zu\n", it->second.arena_bytes,
                    it->second.arena_bytes_in_use, it->second.pool_bytes, it->second.os_allocations);
}

// ---- b. two devices in one context -----------------------------------------------------------------
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

// src (default device): y = x. mid (other device): y = 2 src.y + 1. out (default device): y = mid.y + 2.
class TopB : public cs::System {
    sycl::device other_;

public:
    explicit TopB(const sycl::device& other) : other_{other} {}
    void setup() override {
        add_subsystem<Axpb>("src", 1.0f, 0.0f);
        add_subsystem<Axpb>("mid", cs::property_list{cs::property::device{other_}}, 2.0f, 1.0f);
        add_subsystem<Axpb>("out", 1.0f, 2.0f);
        connect("src.y", "mid.x");
        connect("mid.y", "out.x");
    }
};

static std::string name_of(const sycl::device& d) { return d.get_info<sycl::info::device::name>(); }

/// A non-OpenMP device and the OpenMP host device, when both are visible.
static bool pick_two(sycl::device& first, sycl::device& host) {
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

static void case_b() {
    sycl::device first, host;
    if (!pick_two(first, host)) {
        std::printf("-- b skipped: one device\n");
        return;
    }
    std::printf("-- b. two devices in one context: src/out on '%s', mid on '%s'\n", name_of(first).c_str(),
                name_of(host).c_str());
    const std::array<sycl::device, 2> devs{first, host};
    cs::ModelConfig                   cfg{std::span<const sycl::device>{devs}};
    cfg.default_device = first;
    auto model         = cs::create_model<TopB>(std::move(cfg), host);
    model.setup();

    check(model.root().entry_at("mid.y").placement().device == host && model.root().entry_at("src.y").placement().device == first,
          "b: mid.y sits on the OpenMP device, src.y on the default device");
    const auto rep = model.memory_report();
    check(rep.devices.size() == 2 && rep.devices.at(first).arena_bytes > 0 && rep.devices.at(host).arena_bytes > 0,
          "b: memory_report: an arena on each of the two devices");

    for (int round = 0; round < 2; ++round) {
        std::vector<float> x(kLen), want_mid(kLen), want_out(kLen);
        for (std::size_t i = 0; i < kLen; ++i) {
            x[i]        = static_cast<float>(i + 1) * static_cast<float>(round + 1);
            want_mid[i] = 2.0f * x[i] + 1.0f;
            want_out[i] = want_mid[i] + 2.0f;
        }
        model.set_input<DType::f32, 1>("src.x", std::span<const float>{x});
        model.forward();
        model.wait();
        const std::string t = round == 0 ? "b1" : "b2 (new input)";
        check(values_out(model, "src.y") == x, (t + ": src.y = x on the first device").c_str());
        check(values_out(model, "mid.y") == want_mid, (t + ": mid.y = 2 src.y + 1 on the OpenMP device").c_str());
        check(values_out(model, "out.y") == want_out, (t + ": out.y = mid.y + 2 back on the first device").c_str());
    }
}

// ---- c. errors and moves -----------------------------------------------------------------------------
static void case_c() {
    std::printf("-- c. errors: before setup, setup twice, default_device outside the context, moves\n");
    auto m = cs::create_model<Top>(cs::ModelConfig{});
    check(throws_with([&] { m.forward(); }, "before setup"), "c: forward before setup throws");
    check(throws_with([&] { (void)m.view_output<DType::f32, 1>("obj.f"); }, "before setup"),
          "c: host access (view_output) before setup throws");
    check(throws_with([&] { m.set_input<DType::f32, 1>("x", std::span<const float>{}); }, "before setup"),
          "c: host access (set_input) before setup throws");
    m.setup();
    check(throws_with([&] { m.setup(); }, "runs once"), "c: a second setup throws");

    // A Model after setup moves; its queue and substrates are on the heap, so the Systems' pointers stay valid.
    cs::Model moved = std::move(m);
    moved.forward();
    moved.wait();
    const SellarRef r = sellar_reference(5.0f, 2.0f, 1.0f, kInitY2);  // Top's own init values
    check(near(scalar_out(moved, "obj.f"), r.f), "c: a moved Model still runs (f = the init-value reference)");
    check(throws_with([&] { m.forward(); }, "moved-from"), "c: the moved-from Model throws");

    // default_device outside the context needs a second device to name.
    const auto all = sycl::device::get_devices();
    if (all.size() < 2) {
        std::printf("-- c: default_device-outside-context check skipped: one device\n");
        return;
    }
    cs::ModelConfig cfg{std::span<const sycl::device>{all.data(), 1}};
    cfg.default_device = all[1];
    check(throws_with([&] { auto bad = cs::create_model<Top>(std::move(cfg)); }, "default_device"),
          "c: a default_device outside the context throws at construction");
}

// ---- d. view_output_mut ------------------------------------------------------------------------------------
// Hold writes only y[0] = x[0] of its 4-element output; the host owns y[1..3].
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

// SumY: s = y[0] + y[1] + y[2] + y[3].
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
        declare_input(y_, "y", cs::from_source_shape<1>());
        declare_output(s_, "s", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(s_.shape(), {y_}, {s_}, {});
    }
};

class TopD : public cs::System {
public:
    void setup() override {
        add_subsystem<Hold>("p");
        add_subsystem<SumY>("c");
        connect("p.y", "c.y");
    }
};

static void case_d() {
    std::printf("-- d. view_output_mut: host-owned elements of an output survive the forward\n");
    auto model = cs::create_model<TopD>(cs::ModelConfig{});
    model.setup();

    {
        auto v = model.view_output_mut<DType::f32, 1>("p.y");
        for (std::size_t k = 0; k < 4; ++k) v[k] = 10.0f * static_cast<float>(k);  // {0, 10, 20, 30}
    }  // written back on destruction
    const float x = 7.0f;
    model.set_input<DType::f32, 1>("p.x", std::span<const float>{&x, 1});
    model.forward();
    model.wait();
    check(scalar_out(model, "c.s") == 7.0f + 10.0f + 20.0f + 30.0f, "d: s = 67: x from the kernel, y[1..3] from the writable view");
    check(values_out(model, "p.y") == std::vector<float>({7.0f, 10.0f, 20.0f, 30.0f}), "d: p.y = {x, 10, 20, 30} after the forward");

    {
        auto v = model.view_output_mut<DType::f32, 1>("p.y");
        v[std::size_t{2}] = 100.0f;
        v.commit();  // the mid-scope flush
    }
    model.forward();
    model.wait();
    check(scalar_out(model, "c.s") == 7.0f + 10.0f + 100.0f + 30.0f, "d: after commit() and a forward, s = 147");

    check(throws_with([&] { (void)model.view_input<DType::f32, 1>("p.y"); }, "an output"), "d: view_input on an output throws");
    check(throws_with([&] { (void)model.view_output_mut<DType::f32, 1>("p.x"); }, "an input"), "d: view_output_mut on an input throws");
}

int main() {
    try {
        const sycl::device dev{sycl::default_selector_v};
        std::printf("device: %s\n", name_of(dev).c_str());
        case_a();
        case_b();
        case_c();
        case_d();
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    std::printf(failures == 0 ? "SMOKE MODEL PASS\n" : "SMOKE MODEL FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
