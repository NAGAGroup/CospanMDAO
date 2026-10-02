// Component tests, every case through a cospan::Model.
//
// Each case builds a tiny top System that holds the components under test,
// makes a Model (create_model / setup), and checks through the Model
// (forward / wait / view_* / set_*). There is no driver and no entry access
// here; the only thing reached past the Model's verbs is the Model's own
// retrieval (root().entry_at, get_subsystem).
//
//   plain body, rank 1/2/3        y = 2x + 1, ragged sizes (1000, 40x30, 7x9x11)
//   WeightedHistogram             atomic output, FromSourceDim, value-driven
//                                 n_bins; ZeroBins + tie_outputs; two forwards, n_bins changed by
//                                 init_input between them
//   TiledTranspose                group body, Local scratch, wg override; 37x53
//   host component chain          producer -> host prefix sum -> consumer
//   custom-op plumbing            device_ptr recorded at submission
//   initial values                fill/zero/from_host/custom/custom_host, zeros by default,
//                                 metadata rules; phase guards (init_input / pre_forward / setup)

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
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
using cst::default_config;
using cst::to_vector;
using cst::wait_and_view;

namespace tc {

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

// ---- plain body, rank 1..3 ---------------------------------------------------------------
template <int R>
struct AffineBody {
    struct Inputs {
        cs::Input<DType::f32, R> x;
    };
    struct Outputs {
        cs::Output<DType::f32, R> y;
    };
    void operator()(sycl::id<R> idx, Inputs in, Outputs out) const { out.y[idx] = 2.0f * in.x[idx] + 1.0f; }
};

template <int R>
class Affine : public cs::Component<AffineBody<R>> {
    cs::Input<DType::f32, R>  x_;
    cs::Output<DType::f32, R> y_;
    sycl::range<R>            dims_;

public:
    explicit Affine(sycl::range<R> dims) : dims_{dims} {}
    void setup() override {
        // A free input with declared dims; the output takes the input's shape.
        if constexpr (R == 1) {
            this->declare_input(x_, "x", static_cast<int>(dims_[0]));
        } else if constexpr (R == 2) {
            this->declare_input(x_, "x", static_cast<int>(dims_[0]), static_cast<int>(dims_[1]));
        } else {
            this->declare_input(x_, "x", static_cast<int>(dims_[0]), static_cast<int>(dims_[1]),
                                static_cast<int>(dims_[2]));
        }
        this->declare_output(y_, "y", cs::from_input_shape<R>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(x_.shape(), {x_}, {y_}, {});
    }
};

template <int R>
class TopAffine : public cs::System {
    sycl::range<R> dims_;

public:
    explicit TopAffine(sycl::range<R> dims) : dims_{dims} {}
    void setup() override { add_subsystem<Affine<R>>("aff", dims_); }
};

template <int R>
void check_affine(sycl::range<R> dims) {
    cs::Model model = cs::create_model<TopAffine<R>>(default_config(), dims);
    model.setup();

    const std::size_t n = dims.size();
    for (int round = 0; round < 2; ++round) {  // a second forward on new input
        std::vector<float> x(n), want(n);
        for (std::size_t k = 0; k < n; ++k) {
            x[k]    = static_cast<float>(k % 97) * 0.5f * static_cast<float>(round + 1);
            want[k] = 2.0f * x[k] + 1.0f;
        }
        model.init_input("aff.x", cs::init::from_host(x));
        auto y = wait_and_view<DType::f32, R>(model, "aff.y");
        REQUIRE(y.shape() == dims);
        CHECK(all_close(to_vector(y), want));
    }
}

// ---- WeightedHistogram --------------------------------------------------------------------
constexpr std::size_t kN = 1000;

float value_of(std::size_t i) { return (static_cast<float>((i * 7) % 1000) + 0.5f) / 1000.0f; }
float weight_of(std::size_t i) { return static_cast<float>(1 + i % 3); }

struct HistProducerBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> values;
        cs::Output<DType::f32, 1> weights;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const {
        out.values[i]  = (static_cast<float>((i[0] * 7) % 1000) + 0.5f) / 1000.0f;
        out.weights[i] = static_cast<float>(1 + i[0] % 3);
    }
};

class HistProducer : public cs::Component<HistProducerBody> {
    cs::Output<DType::f32, 1> values_, weights_;

public:
    void setup() override {
        declare_output(values_, "values", static_cast<int>(kN));
        declare_output(weights_, "weights", static_cast<int>(kN));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{kN}, {}, {values_, weights_}, {});
    }
};

struct HistBody {
    struct Inputs {
        cs::Input<DType::f32, 1> values;
        cs::Input<DType::f32, 1> weights;
        cs::Input<DType::u32, 1> n_bins;
    };
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> bins;
    };
    float lo, hi;
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        const auto b = static_cast<std::size_t>((in.values[i] - lo) / (hi - lo) * static_cast<float>(in.n_bins[0]));
        out.bins[b].fetch_add(in.weights[i]);
    }
};

class WeightedHistogram : public cs::Component<HistBody> {
    cs::Input<DType::f32, 1>         values_, weights_;
    cs::Input<DType::u32, 1>         n_bins_;
    cs::Output<DType::atomic_f32, 1> bins_;
    float                            lo_, hi_;

public:
    WeightedHistogram(float lo, float hi) : lo_{lo}, hi_{hi} {}
    void setup() override {
        declare_input(values_, "values", cs::unspecified);
        declare_input(weights_, "weights", cs::from_input_shape<1>("values"));
        declare_input(n_bins_, "n_bins", cs::unspecified_shape<1>());  // connected from the NBins producer
        declare_output(bins_, "bins", cs::unspecified_shape<1>());  // a tie_outputs member: the alias takes its shape from NBins' value
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(values_.shape(), {values_, weights_, n_bins_}, {bins_}, {lo_, hi_});
    }
};

// The zeroing producer (r01 §6.7): outputs are never initialized, so what the histogram accumulates
// onto is the output of a component that computes zero.
struct ZeroBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> bins;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.bins[i].store(0.0f); }
};

class ZeroBins : public cs::Component<ZeroBody> {
    cs::Output<DType::atomic_f32, 1> bins_;

public:
    void setup() override { declare_output(bins_, "bins", cs::unspecified_shape<1>()); }  // shape: the tie alias's
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(bins_.shape(), {}, {bins_}, {});
    }
};

// The bin count as an OUTPUT (an alias shape reads an upstream output): n[0] = v[0], where v is
// the free input the test drives (promoted as "n_bins").
struct NBinsBody {
    struct Inputs {
        cs::Input<DType::u32, 1> v;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> n;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.n[i] = in.v[i]; }
};

class NBins : public cs::Component<NBinsBody> {
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

class TopHist : public cs::System {
public:
    void setup() override {
        add_subsystem<HistProducer>("prod");
        add_subsystem<NBins>("nb");        // produces the bin count; registered before the tie's members
        add_subsystem<ZeroBins>("zero");  // BEFORE hist: registration order is execution order
        add_subsystem<WeightedHistogram>("hist", 0.0f, 1.0f);
        connect("nb.n", "hist.n_bins");
        // The bins root's shape is the VALUE of nb.n[0], read by the root's owner (zero) each run.
        tie_outputs<DType::atomic_f32, 1>({"zero.bins", "hist.bins"}, "bins", cs::alias::from_output_values("nb.n"));
        promote<DType::u32, 1>("nb.v", "n_bins");  // the test drives the bin count by this name
        connect("prod.values", "hist.values");
        connect("prod.weights", "hist.weights");
    }
};

std::vector<float> histogram_ref(std::uint32_t nb) {
    std::vector<float> ref(nb, 0.0f);
    for (std::size_t i = 0; i < kN; ++i)
        ref[static_cast<std::size_t>((value_of(i) - 0.0f) / (1.0f - 0.0f) * static_cast<float>(nb))] += weight_of(i);
    return ref;
}

// ---- TiledTranspose --------------------------------------------------------------------------
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

struct TransposeBody {
    struct Inputs {
        cs::Input<DType::f32, 2> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> out;
    };
    struct Scratch {
        cs::Local<DType::f32, 2> tile;
    };
    void operator()(cs::GroupItem<2> it, Inputs in, Outputs out, Scratch s) const {
        const auto l = it.get_local_id(), g = it.get_global_id();
        const bool live = g[0] < in.in.shape()[0] && g[1] < in.in.shape()[1];
        if (live) s.tile[{l[0], l[1]}] = in.in[{g[0], g[1]}];
        sycl::group_barrier(it.get_group());
        const auto gt = sycl::id<2>{it.get_group(1) * it.get_local_range(1) + l[0],
                                    it.get_group(0) * it.get_local_range(0) + l[1]};
        if (gt[0] < out.out.shape()[0] && gt[1] < out.out.shape()[1]) out.out[gt] = s.tile[{l[1], l[0]}];
    }
};

class TiledTranspose : public cs::Component<TransposeBody> {
    cs::Input<DType::f32, 2>  in_;
    cs::Output<DType::f32, 2> out_;
    static constexpr std::size_t WG = 16;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified_shape<2>());
        declare_output(out_, "out", cs::FromInputDim{"in", 1}, cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        TransposeBody::Scratch s;
        s.tile.reshape({WG, WG + 1});
        return create_launch_config(in_.shape(), {WG, WG}, {in_}, {out_}, s, {});
    }
};

class TopTranspose : public cs::System {
public:
    void setup() override {
        add_subsystem<Iota2D>("iota", 37, 53);
        add_subsystem<TiledTranspose>("tr");
        connect("iota.out", "tr.in");
    }
};

// ---- 1-D producer and consumer shared by the host chain and the custom-op case --------------------
struct Iota1DBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const { o.y[i] = static_cast<float>(i[0]); }
};

class Iota1D : public cs::Component<Iota1DBody> {
    cs::Output<DType::f32, 1> y_;
    int                       n_;

public:
    explicit Iota1D(int n) : n_{n} {}
    void setup() override { declare_output(y_, "y", n_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {}, {y_}, {});
    }
};

struct Plus1Body {
    struct Inputs {
        cs::Input<DType::f32, 1> z;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> w;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.w[i] = in.z[i] + 1.0f; }
};

class Plus1 : public cs::Component<Plus1Body> {
    cs::Input<DType::f32, 1>  z_;
    cs::Output<DType::f32, 1> w_;

public:
    void setup() override {
        declare_input(z_, "z", cs::unspecified_shape<1>());
        declare_output(w_, "w", cs::from_input_shape<1>("z"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(z_.shape(), {z_}, {w_}, {});
    }
};

// ---- host component chain -------------------------------------------------------------------------
// The host body is a sequential prefix sum: z[i] = z[i-1] + y[i].
struct PrefixBody {
    struct Inputs {
        cs::Input<DType::f32, 1> y;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> z;
    };
    void operator()(cs::Host, Inputs in, Outputs out) const {
        float acc = 0.0f;
        for (std::size_t i = 0; i < in.y.size(); ++i) {
            acc += in.y[i];
            out.z[i] = acc;
        }
    }
};

class PrefixSum : public cs::Component<PrefixBody> {
    cs::Input<DType::f32, 1>  y_;
    cs::Output<DType::f32, 1> z_;

public:
    void setup() override {
        declare_input(y_, "y", cs::unspecified_shape<1>());
        declare_output(z_, "z", cs::from_input_shape<1>("y"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config({y_}, {z_}, {});
    }
};

class TopHostChain : public cs::System {
public:
    void setup() override {
        add_subsystem<Iota1D>("iota", static_cast<int>(kN));
        add_subsystem<PrefixSum>("prefix");
        add_subsystem<Plus1>("plus1");
        connect("iota.y", "prefix.y");
        connect("prefix.z", "plus1.z");
    }
};

// ---- custom-op plumbing ---------------------------------------------------------------------------
// The body records the pointers it is handed and enqueues nothing.
struct RecordBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    std::atomic<const void*>* in_ptr;
    std::atomic<const void*>* out_ptr;
    void operator()(cs::CustomOp& op, Inputs in, Outputs out) const {
        in_ptr->store(op.device_ptr(in.x));
        out_ptr->store(op.device_ptr(out.y));
    }
};

class Recorder : public cs::Component<RecordBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    std::atomic<const void*> in_rec{nullptr}, out_rec{nullptr};
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config({x_}, {y_}, {&in_rec, &out_rec});
    }
};

class TopCustomOp : public cs::System {
public:
    void setup() override {
        add_subsystem<Iota1D>("iota", static_cast<int>(kN));
        add_subsystem<Recorder>("rec");
        add_subsystem<Plus1>("plus1");
        connect("iota.y", "rec.x");
        connect("rec.y", "plus1.z");
    }
};

// ---- initial values ---------------------------------------------------------------------------
std::atomic<int>         g_host_calls{0};
std::atomic<int>         g_custom_calls{0};
std::atomic<std::size_t> g_custom_size{0};

// y = 10 x + w + v + d. x fill, w from_host, u zero, v custom_host, m custom (a device kernel
// writes 10 + i; the body ignores it), d declares nothing (zeros by default).
struct InitDemoBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x, w, u, v, d, m;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        out.y[i] = 10.0f * in.x[std::size_t{0}] + in.w[i] + in.v[i] + in.d[i];
    }
};

class InitDemo : public cs::Component<InitDemoBody> {
    cs::Input<DType::f32, 1>  x_, w_, u_, v_, d_, m_;
    cs::Output<DType::f32, 1> y_;
    cs::Output<DType::f32, 1> extra_;  // only ever declared by the phase-guard hook

public:
    void setup() override {
        declare_input(x_, "x", 1);
        declare_input(w_, "w", 4);
        declare_input(u_, "u", 4);
        declare_input(v_, "v", 4);
        declare_input(d_, "d", 4);
        declare_input(m_, "m", 4);
        declare_output(y_, "y", 4);
        init_input("x", cs::init::fill(2.0f));
        init_input("w", cs::init::from_host(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
        init_input("u", cs::init::zero());
        init_input("v", cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostViewMut<DType::f32, 1>&)>{
                            [](cs::HostViewMut<DType::f32, 1>& h) {
                                ++g_host_calls;
                                for (std::size_t k = 0; k < 4; ++k) h[k] = 100.0f * static_cast<float>(k + 1);
                            }}));
        init_input("m", cs::init::custom<DType::f32, 1>(
                            std::function<void(sk::queue&, cs::Initializer<DType::f32, 1>)>{
                                [](sk::queue& q, cs::Initializer<DType::f32, 1> out) {
                                    ++g_custom_calls;
                                    g_custom_size = out.size();
                                    // A device kernel on the given queue writes m[i] = 10 + i through
                                    // the Initializer's public operator[] (the submit form CoSpan's own
                                    // launches use: queue.submit(device, command-group)).
                                    q.submit(q.get_device(), [=](sycl::handler& h) {
                                        h.parallel_for(sycl::range<1>{out.size()}, [=](sycl::id<1> i) {
                                            out[i] = 10.0f + static_cast<float>(i[0]);
                                        });
                                    });
                                }}));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y_.shape(), {x_, w_, u_, v_, d_, m_}, {y_}, {});
    }

    // Hooks for the phase-guard checks, called by the test after setup.
    void try_declare() { declare_output(extra_, "extra", 1); }
    void try_reshape() { reshape(y_, sycl::range<1>{2}); }
    void try_init() { init_input("x", cs::init::zero()); }
};

class TopInit : public cs::System {
public:
    void setup() override { add_subsystem<InitDemo>("ini"); }
};

/// y = 2 x + 1 on a free 4-vector, with optional init declarations for the metadata rules:
/// mode 0 one init on x, 1 an init on the OUTPUT y, 2 two inits on x (one depth).
class InitAffine : public cs::Component<AffineBody<1>> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
    std::optional<cs::InitSpec> init_;
    int                         mode_;
    bool                        tied_x_;

public:
    explicit InitAffine(std::optional<cs::InitSpec> init = std::nullopt, int mode = 0, bool tied_x = false)
        : init_{std::move(init)}, mode_{mode}, tied_x_{tied_x} {}
    void setup() override {
        if (tied_x_)
            declare_input(x_, "x", cs::unspecified_shape<1>());  // a tie_inputs member: the tie alias owns the shape
        else
            declare_input(x_, "x", 4);
        declare_output(y_, "y", cs::from_input_shape<1>("x"));
        if (init_) init_input("x", *init_);
        if (mode_ == 1) init_input("y", cs::init::zero());
        if (mode_ == 2) init_input("x", cs::init::fill(3.0f));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {x_}, {y_}, {});
    }
};

/// A root System whose setup() runs a callable.
class TopFn : public cs::System {
    std::function<void(cs::System&)> build_;

public:
    explicit TopFn(std::function<void(cs::System&)> b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

}  // namespace tc

// =============================================================================================
// Plain bodies
// =============================================================================================
TEST_CASE("plain body: y = 2x + 1 at ranks 1, 2 and 3", "[component][plain]") {
    SECTION("rank 1, 1000") { tc::check_affine<1>(sycl::range<1>{1000}); }
    SECTION("rank 2, 40 x 30") { tc::check_affine<2>(sycl::range<2>{40, 30}); }
    SECTION("rank 3, 7 x 9 x 11") { tc::check_affine<3>(sycl::range<3>{7, 9, 11}); }
}

// =============================================================================================
// WeightedHistogram
// =============================================================================================
TEST_CASE("WeightedHistogram: atomic output, FromSourceDim, value-driven n_bins", "[component][histogram]") {
    auto model = cs::create_model<tc::TopHist>(default_config());
    model.setup();

    const std::uint32_t nb1 = 8, nb2 = 13;

    SECTION("first forward, n_bins = 8") {
        model.init_input("n_bins", cs::init::from_host(std::vector<std::uint32_t>{nb1}));
        auto bins = wait_and_view<DType::f32, 1>(model, "bins");
        CHECK(bins.size() == nb1);
        CHECK(to_vector(bins) == tc::histogram_ref(nb1));
    }

    SECTION("second forward with n_bins changed by init_input: the root grows and accumulates") {
        model.init_input("n_bins", cs::init::from_host(std::vector<std::uint32_t>{nb1}));
        {
            auto bins = wait_and_view<DType::f32, 1>(model, "bins");
            REQUIRE(bins.size() == nb1);
            CHECK(to_vector(bins) == tc::histogram_ref(nb1));
        }
        model.init_input("n_bins", cs::init::from_host(std::vector<std::uint32_t>{nb2}));
        auto bins = wait_and_view<DType::f32, 1>(model, "bins");
        REQUIRE(bins.size() == nb2);
        // ZeroBins zeroes the grown root each forward, then the histogram accumulates onto it.
        CHECK(to_vector(bins) == tc::histogram_ref(nb2));
    }
}

// =============================================================================================
// TiledTranspose
// =============================================================================================
TEST_CASE("TiledTranspose: group body, Local scratch, work-group override, 37 x 53", "[component][transpose]") {
    auto model = cs::create_model<tc::TopTranspose>(default_config());
    model.setup();
    auto out = wait_and_view<DType::f32, 2>(model, "tr.out");
    REQUIRE(out.shape()[0] == 53);
    REQUIRE(out.shape()[1] == 37);
    bool good = true;
    for (std::size_t j = 0; j < 53 && good; ++j)
        for (std::size_t i = 0; i < 37 && good; ++i) good = out[sycl::id<2>{j, i}] == static_cast<float>(i * 1000 + j);
    CHECK(good);  // out[j][i] == in[i][j]
}

// =============================================================================================
// Host component chain
// =============================================================================================
TEST_CASE("host component chain: producer -> host prefix sum -> consumer", "[component][host]") {
    auto model = cs::create_model<tc::TopHostChain>(default_config());
    model.setup();

    // The host component's I/O are host-kind USM (moved by the ordinary copy rule).
    CHECK(model.root().entry_at("prefix.y").storage_kind() == sk::kind::host);
    CHECK(model.root().entry_at("prefix.z").storage_kind() == sk::kind::host);

    auto w = wait_and_view<DType::f32, 1>(model, "plus1.w");
    REQUIRE(w.size() == tc::kN);
    std::vector<float> want(tc::kN);
    float              acc = 0.0f;
    for (std::size_t i = 0; i < tc::kN; ++i) {
        acc += static_cast<float>(i);
        want[i] = acc + 1.0f;
    }
    CHECK(all_close(to_vector(w), want));

    // And it goes again.
    auto w2 = wait_and_view<DType::f32, 1>(model, "plus1.w");
    CHECK(all_close(to_vector(w2), want));
}

// =============================================================================================
// Custom-op plumbing
// =============================================================================================
TEST_CASE("custom op: device_ptr at submission is the storage the consumer reads", "[component][customop]") {
    auto model = cs::create_model<tc::TopCustomOp>(default_config());
    model.setup();
    auto w = wait_and_view<DType::f32, 1>(model, "plus1.w");

    const sycl::device    dev{sycl::default_selector_v};
    const cs::Placement   dev_pl{dev, sk::kind::device};
    auto&                 rec      = dynamic_cast<tc::Recorder&>(model.get_subsystem("rec"));
    const void*           want_in  = model.root().entry_at("iota.y").root()->storage_on(dev_pl);
    const void*           want_out = model.root().entry_at("rec.y").root()->storage_on(dev_pl);
    CHECK(rec.in_rec.load() != nullptr);
    CHECK(rec.in_rec.load() == want_in);
    CHECK(rec.out_rec.load() != nullptr);
    CHECK(rec.out_rec.load() == want_out);

    // The recorder wrote nothing, so its zero-filled output feeds the plain
    // consumer: w == 0 + 1 everywhere, after the custom op in stream order.
    REQUIRE(w.size() == tc::kN);
    CHECK(to_vector(w) == std::vector<float>(tc::kN, 1.0f));
}

// =============================================================================================
// Initializers and phase guards
// =============================================================================================
TEST_CASE("initial values: fill / zero / from_host / custom / custom_host, and zeros by default", "[component][init]") {
    tc::g_host_calls   = 0;
    tc::g_custom_calls = 0;
    tc::g_custom_size  = 0;
    auto model = cs::create_model<tc::TopInit>(default_config());
    model.setup();
    auto y = wait_and_view<DType::f32, 1>(model, "ini.y");

    const std::vector<float> ramp{1.0f, 2.0f, 3.0f, 4.0f};
    SECTION("inputs") {
        CHECK(model.view_input<DType::f32, 1>("ini.x")[std::size_t{0}] == 2.0f);  // fill
        CHECK(to_vector(model.view_input<DType::f32, 1>("ini.w")) == ramp);       // from_host
        CHECK(to_vector(model.view_input<DType::f32, 1>("ini.u")) == std::vector<float>(4, 0.0f));  // zero
        CHECK(to_vector(model.view_input<DType::f32, 1>("ini.v")) ==
              std::vector<float>({100.0f, 200.0f, 300.0f, 400.0f}));  // custom_host
        CHECK(to_vector(model.view_input<DType::f32, 1>("ini.d")) == std::vector<float>(4, 0.0f));  // default
    }
    SECTION("custom ran once, with the root's shape, and its device kernel wrote the values") {
        CHECK(tc::g_custom_calls.load() == 1);
        CHECK(tc::g_custom_size.load() == 4);
        CHECK(to_vector(model.view_input<DType::f32, 1>("ini.m")) == std::vector<float>({10.0f, 11.0f, 12.0f, 13.0f}));
        CHECK(tc::g_host_calls.load() == 1);  // a durable root is initialized at setup, not per forward
    }
    SECTION("the kernel sees the initialized inputs") {
        // y = 10 x + w + v + d = 20 + {1,2,3,4} + {100,200,300,400} + 0
        CHECK(to_vector(y) == std::vector<float>({121.0f, 222.0f, 323.0f, 424.0f}));
    }
}

TEST_CASE("initial values: metadata rules (connected, output, duplicates, highest level)", "[component][init][rules]") {
    const auto top = [](std::function<void(cs::System&)> build) {
        return cs::create_model<tc::TopFn>(default_config(), std::move(build));
    };

    SECTION("an init on a connected input is ignored") {
        auto model = top([](cs::System& s) {
            s.add_subsystem<tc::InitAffine>("p", std::optional<cs::InitSpec>{cs::init::fill(1.0f)});
            s.add_subsystem<tc::InitAffine>("c", std::optional<cs::InitSpec>{cs::init::fill(100.0f)});
            s.connect("p.y", "c.x");
        });
        model.setup();
        auto y = wait_and_view<DType::f32, 1>(model, "c.y");
        CHECK(to_vector(y) == std::vector<float>(4, 7.0f));  // 2 * (2 * 1 + 1) + 1, not 201
    }
    SECTION("an init on an output throws") {
        tc::expect_throws(
            [&] {
                auto model = top([](cs::System& s) {
                    s.add_subsystem<tc::InitAffine>("a", std::optional<cs::InitSpec>{}, 1);
                });
                model.setup();
            },
            "is an output");
    }
    SECTION("two inits at one depth on one root throw") {
        tc::expect_throws(
            [&] {
                auto model = top([](cs::System& s) {
                    s.add_subsystem<tc::InitAffine>("a", std::optional<cs::InitSpec>{cs::init::fill(1.0f)}, 0, true);
                    s.add_subsystem<tc::InitAffine>("b", std::optional<cs::InitSpec>{cs::init::fill(2.0f)}, 0, true);
                    s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::alias::fixed(4));
                });
                model.setup();
            },
            "two initial values");
    }
    SECTION("a System-level init beats the components' on a tied alias") {
        auto model = top([](cs::System& s) {
            s.add_subsystem<tc::InitAffine>("a", std::optional<cs::InitSpec>{cs::init::fill(1.0f)}, 0, true);
            s.add_subsystem<tc::InitAffine>("b", std::optional<cs::InitSpec>{cs::init::fill(2.0f)}, 0, true);
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::alias::fixed(4));
            s.init_input("x", cs::init::fill(9.0f));
        });
        model.setup();
        auto a = wait_and_view<DType::f32, 1>(model, "a.y");
        CHECK(to_vector(a) == std::vector<float>(4, 19.0f));
        CHECK(to_vector(model.view_output<DType::f32, 1>("b.y")) == std::vector<float>(4, 19.0f));
    }
}

TEST_CASE("phase guards: init_input, reshape and declare outside their phase throw", "[component][init][guards]") {
    auto model = cs::create_model<tc::TopInit>(default_config());
    model.setup();
    auto& c = dynamic_cast<tc::InitDemo&>(model.get_subsystem("ini"));

    SECTION("init_input outside setup() throws") {
        tc::expect_throws([&] { c.try_init(); }, "setup() only");
    }
    SECTION("reshape outside pre_forward() throws") {
        tc::expect_throws([&] { c.try_reshape(); }, "pre_forward() only");
    }
    SECTION("declare outside setup() throws") {
        tc::expect_throws([&] { c.try_declare(); }, "setup() only");
    }
    SECTION("the model is unharmed by the refused calls") {
        tc::expect_throws([&] { c.try_init(); }, "setup() only");
        auto y = wait_and_view<DType::f32, 1>(model, "ini.y");
        CHECK(to_vector(y) == std::vector<float>({121.0f, 222.0f, 323.0f, 424.0f}));
    }
    model.wait();  // setup() submitted initial-value work; wait before the Model goes out of scope
}