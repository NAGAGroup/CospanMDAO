// Component tests, every case through a cospan::Model.
//
// Each case builds a tiny top System that holds the components under test,
// makes a Model (create_model / setup), and checks through the Model
// (forward / wait / view_* / update_input). There is no driver here; the only things reached past the
// Model's verbs are the Model's own retrieval (get_subsystem, root().instance_at).
//
// Reading results (internal/design-review-2026-10-03.md, section 5): forward() only enqueues; a host view is obtained
// after the first forward and sync()ed before it is read (cst::wait_and_view / synced_input do all three).
//
//   index-range body, rank 1/2/3       y = 2x + 1, ragged sizes (1000, 40x30, 7x9x11)
//   WeightedHistogram             atomic output, unspecified dims; ZeroBins + tie_outputs whose shape is a
//                                 System shape hook (cs::dim::from_value of NBins' output); two forwards,
//                                 n_bins changed by update_input between them
//   TiledTranspose                group body, Local scratch, wg override; 37x53
//   host component chain          producer -> host prefix sum -> consumer
//   custom-op plumbing            device_ptr recorded at submission
//   initial values                fill/zero/from_host/custom/custom_host, zeros by default,
//                                 metadata rules; phase guards (init_input / prepare_forward / setup)

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

// ---- index-range body, rank 1..3 ---------------------------------------------------------------
template <int R>
class Affine : public cs::Component<Affine<R>> {
public:
    struct Inputs {
        cs::Input<DType::f32, R> x;
    };
    struct Outputs {
        cs::Output<DType::f32, R> y;
    };
    static void compute_forward(sycl::id<R> idx, const Inputs& in, const Outputs& out) { out.y[idx] = 2.0f * in.x[idx] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

private:
    sycl::range<R> dims_;

public:
    explicit Affine(sycl::range<R> dims) : dims_{dims} {}
    void declare() override {
        // A free input with declared dims; the output's dims are unspecified and prepare_forward gives it the input's shape
        // (design review, section 10: an output is reshaped by its producing component).
        if constexpr (R == 1) {
            this->declare_input(inputs.x, "x", static_cast<int>(dims_[0]));
        } else if constexpr (R == 2) {
            this->declare_input(inputs.x, "x", static_cast<int>(dims_[0]), static_cast<int>(dims_[1]));
        } else {
            this->declare_input(inputs.x, "x", static_cast<int>(dims_[0]), static_cast<int>(dims_[1]),
                                static_cast<int>(dims_[2]));
        }
        this->declare_output(outputs.y, "y", cs::unspecified_shape<R>());
    }
    void prepare_forward() override {
        const sycl::range<R> s = this->shape_of(inputs.x);  // one entry per dimension
        [&]<std::size_t... I>(std::index_sequence<I...>) { this->set_shape(outputs.y, s[I]...); }(std::make_index_sequence<R>{});
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};

template <int R>
class TopAffine : public cs::System {
    sycl::range<R> dims_;

public:
    explicit TopAffine(sycl::range<R> dims) : dims_{dims} {}
    void setup() override { add_subsystem(cs::subsystem_desc<Affine<R>>("aff", dims_)); }
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
        model.update_input("aff.x", cs::init::from_host(x));
        auto y = wait_and_view<DType::f32, R>(model, "aff.y");
        REQUIRE(y.shape() == dims);
        CHECK(all_close(to_vector(y), want));
    }
}

// ---- WeightedHistogram --------------------------------------------------------------------
constexpr std::size_t kN = 1000;

float value_of(std::size_t i) { return (static_cast<float>((i * 7) % 1000) + 0.5f) / 1000.0f; }
float weight_of(std::size_t i) { return static_cast<float>(1 + i % 3); }

class HistProducer : public cs::Component<HistProducer> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> values;
        cs::Output<DType::f32, 1> weights;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) {
        out.values[i]  = (static_cast<float>((i[0] * 7) % 1000) + 0.5f) / 1000.0f;
        out.weights[i] = static_cast<float>(1 + i[0] % 3);
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_output(outputs.values, "values", static_cast<int>(kN));
        declare_output(outputs.weights, "weights", static_cast<int>(kN));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{kN}));
    }
};

class WeightedHistogram : public cs::Component<WeightedHistogram> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> values;
        cs::Input<DType::f32, 1> weights;
        cs::Input<DType::u32, 1> n_bins;
        cs::Input<DType::f32, 1> lo, hi;  // the histogram range: free inputs of extent 1, initialized from the constructor's values
    };
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> bins;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        const float lo = in.lo[std::size_t{0}], hi = in.hi[std::size_t{0}];
        const auto  b  = static_cast<std::size_t>((in.values[i] - lo) / (hi - lo) * static_cast<float>(in.n_bins[0]));
        out.bins[b].fetch_add(in.weights[i]);
    }

    Inputs  inputs;
    Outputs outputs;

private:
    float lo_, hi_;

public:
    WeightedHistogram(float lo, float hi) : lo_{lo}, hi_{hi} {}
    void declare() override {
        declare_input(inputs.lo, "lo", 1);
        declare_input(inputs.hi, "hi", 1);
        init_input("lo", cs::init::fill(lo_));
        init_input("hi", cs::init::fill(hi_));
        declare_input(inputs.values, "values", cs::unspecified);
        declare_input(inputs.weights, "weights", cs::unspecified);  // connected from the producer: it takes the producer's shape
        declare_input(inputs.n_bins, "n_bins", cs::unspecified_shape<1>());  // connected from the NBins producer
        declare_output(outputs.bins, "bins", cs::unspecified_shape<1>());  // a tie_outputs member: the tie's shape hook sets it from NBins' value
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.values.shape()));
    }
};

// The zeroing producer (r01 §6.7): outputs are never initialized, so what the histogram accumulates
// onto is the output of a component that computes zero.

class ZeroBins : public cs::Component<ZeroBins> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> bins;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& out) { out.bins[i].store(0.0f); }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.bins, "bins", cs::unspecified_shape<1>()); }  // shape: the tie alias's
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.bins.shape()));
    }
};

// The bin count as an OUTPUT (a shape hook reads an upstream output's value): n[0] = v[0], where v is
// the free input the test drives (promoted as "n_bins").

class NBins : public cs::Component<NBins> {
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

class TopHist : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<HistProducer>("prod"));
        // nb must be declared BEFORE zero: the tie's shape hook reads the VALUE of nb.n and runs before the first child that
        // has the tied variable (zero), and children run in declaration order; a hook's source needs its producer in an
        // earlier slot (design review, section 10).
        add_subsystem(cs::subsystem_desc<NBins>("nb"));
        add_subsystem(cs::subsystem_desc<ZeroBins>("zero"));
        add_subsystem(cs::subsystem_desc<WeightedHistogram>("hist", 0.0f, 1.0f));
        connect("nb.n", "hist.n_bins");
        // The bins' shape is the VALUE of nb.n[0], set by a hook on the tie alias before zero runs, each forward.
        tie_outputs<DType::atomic_f32, 1>({"zero.bins", "hist.bins"}, "bins");
        shape_hook("bins", cs::dim::from_value("nb.n", 0));
        promote<DType::u32, 1>("nb.v", "n_bins");  // the test drives the bin count by this name
        connect("prod.values", "hist.values");
        connect("prod.weights", "hist.weights");
        // Execution order is data dependencies only (section 3) and tied writers do not wait on each other: hist
        // accumulates onto what zero writes, so the edge is declared (it used to be registration order).
        depends_on("hist", "zero");
    }
};

std::vector<float> histogram_ref(std::uint32_t nb) {
    std::vector<float> ref(nb, 0.0f);
    for (std::size_t i = 0; i < kN; ++i)
        ref[static_cast<std::size_t>((value_of(i) - 0.0f) / (1.0f - 0.0f) * static_cast<float>(nb))] += weight_of(i);
    return ref;
}

// ---- TiledTranspose --------------------------------------------------------------------------

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

class TiledTranspose : public cs::Component<TiledTranspose> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> out;
    };
    using TileMem = cs::LocalMem<cs::LocalArray<DType::f32, 2>>;
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out, TileMem lm) {
        auto [tile] = lm.get(it);
        const auto l = it.get_local_id(), g = it.get_global_id();
        const bool live = g[0] < in.in.shape()[0] && g[1] < in.in.shape()[1];
        if (live) tile[sycl::id<2>{l[0], l[1]}] = in.in[{g[0], g[1]}];
        cs::group_barrier(it.get_group());
        const auto gt = sycl::id<2>{it.get_group(1) * it.get_local_range(1) + l[0],
                                    it.get_group(0) * it.get_local_range(0) + l[1]};
        if (gt[0] < out.out.shape()[0] && gt[1] < out.out.shape()[1]) out.out[gt] = tile[sycl::id<2>{l[1], l[0]}];
    }

    Inputs  inputs;
    Outputs outputs;

private:
    static constexpr std::size_t WG = 16;

public:
    void declare() override {
        declare_input(inputs.in, "in", cs::unspecified_shape<2>());
        declare_output(outputs.out, "out", cs::unspecified_shape<2>());
    }
    void prepare_forward() override {
        const sycl::range<2> s = shape_of(inputs.in);
        set_shape(outputs.out, s[1], s[0]);  // the transpose
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const sycl::range<2> s = inputs.in.shape();  // rounded up to whole work-groups; the body guards the tail
        const sycl::range<2> r{(s[0] + WG - 1) / WG * WG, (s[1] + WG - 1) / WG * WG};
        return create_launch_config(cs::range_desc(r), cs::wg_desc(sycl::range<2>{WG, WG}),
                                    cs::local_desc(sycl::range<2>{WG, WG + 1}));
    }
};

// A group-range component whose range (5) is not a multiple of its work-group shape (2): launching it throws.
class GrpBad : public cs::Component<GrpBad> {
public:
    struct Inputs {};
    struct Outputs { cs::Output<DType::f32, 1> y; };
    static void compute_forward(cs::GroupItem<1> it, const Inputs&, const Outputs& out) {
        if (it.get_global_id(0) < out.y.shape()[0]) out.y[it.get_global_id(0)] = 1.0f;
    }
    Inputs  inputs;
    Outputs outputs;
    void declare() override { declare_output(outputs.y, "y", 5); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{5}), cs::wg_desc(sycl::range<1>{2}));
    }
};

class TopBad : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<GrpBad>("bad")); }
};

// Three arrays of different element types and odd shapes: u8 {3}, f32 {5}, i32 {3, 2}; work-group 4, two groups.
class LocalMixed : public cs::Component<LocalMixed> {
public:
    struct Inputs {};
    struct Outputs { cs::Output<DType::f32, 1> y; };
    using Mem = cs::LocalMem<cs::LocalArray<DType::u8, 1>, cs::LocalArray<DType::f32, 1>, cs::LocalArray<DType::i32, 2>>;
    static void compute_forward(cs::GroupItem<1> it, const Inputs&, const Outputs& out, Mem lm) {
        auto [a, b, c] = lm.get(it);
        const std::size_t l = it.get_local_linear_id();
        if (l < 3) a[l] = static_cast<unsigned char>(l + 1);
        if (l < 5) b[l] = 0.5f * static_cast<float>(l);
        if (l < 6) c[sycl::id<2>{l / 2, l % 2}] = static_cast<int>(10 * l);
        cs::group_barrier(it.get_group());
        out.y[it.get_global_id(0)] = static_cast<float>(a[l % 3] + b[l % 5] + c[sycl::id<2>{(l % 6) / 2, (l % 6) % 2}]);
    }
    Inputs  inputs;
    Outputs outputs;
    void declare() override { declare_output(outputs.y, "y", 8); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{8}), cs::wg_desc(sycl::range<1>{4}),
                                    cs::local_desc(sycl::range<1>{3}, sycl::range<1>{5}, sycl::range<2>{3, 2}));
    }
};

class TopLocalMixed : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<LocalMixed>("lm")); }
};

class TopTranspose : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Iota2D>("iota", 37, 53));
        add_subsystem(cs::subsystem_desc<TiledTranspose>("tr"));
        connect("iota.out", "tr.in");
    }
};

// ---- 1-D producer and consumer shared by the host chain and the custom-op case --------------------

class Iota1D : public cs::Component<Iota1D> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& o) { o.y[i] = static_cast<float>(i[0]); }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_;

public:
    explicit Iota1D(int n) : n_{n} {}
    void declare() override { declare_output(outputs.y, "y", n_); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};

class Plus1 : public cs::Component<Plus1> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> z;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> w;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.w[i] = in.z[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.z, "z", cs::unspecified_shape<1>());
        declare_output(outputs.w, "w", cs::unspecified_shape<1>());
    }
    void prepare_forward() override { set_shape(outputs.w, shape_of(inputs.z)[0]); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.z.shape()));
    }
};

// ---- host component chain -------------------------------------------------------------------------
// The host body is a sequential prefix sum: z[i] = z[i-1] + y[i].

class PrefixSum : public cs::HostComponent<PrefixSum> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> y;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> z;
    };
    static void compute_forward(const Inputs& in, const Outputs& out) {
        float acc = 0.0f;
        for (std::size_t i = 0; i < in.y.size(); ++i) {
            acc += in.y[i];
            out.z[i] = acc;
        }
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.y, "y", cs::unspecified_shape<1>());
        declare_output(outputs.z, "z", cs::unspecified_shape<1>());
    }
    void prepare_forward() override { set_shape(outputs.z, shape_of(inputs.y)[0]); }
};

class TopHostChain : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Iota1D>("iota", static_cast<int>(kN)));
        add_subsystem(cs::subsystem_desc<PrefixSum>("prefix"));
        add_subsystem(cs::subsystem_desc<Plus1>("plus1"));
        connect("iota.y", "prefix.y");
        connect("prefix.z", "plus1.z");
    }
};

// ---- custom-op plumbing ---------------------------------------------------------------------------
// The body records the pointers it is handed and enqueues nothing.

// The pointers the recorder is handed, read by the test.
std::atomic<const void*> g_in_rec{nullptr}, g_out_rec{nullptr};

class Recorder : public cs::CustomComponent<Recorder> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(cs::CustomOp& op, const Inputs& in, const Outputs& out) {
        g_in_rec.store(op.device_ptr(in.x));
        g_out_rec.store(op.device_ptr(out.y));
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_output(outputs.y, "y", cs::unspecified_shape<1>());
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }
};

class TopCustomOp : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Iota1D>("iota", static_cast<int>(kN)));
        add_subsystem(cs::subsystem_desc<Recorder>("rec"));
        add_subsystem(cs::subsystem_desc<Plus1>("plus1"));
        connect("iota.y", "rec.x");
        connect("rec.y", "plus1.z");
    }
};

// ---- initial values ---------------------------------------------------------------------------
std::atomic<int>         g_host_calls{0};
std::atomic<int>         g_custom_calls{0};
std::atomic<std::size_t> g_custom_size{0};

// y = 10 x + w + v + d. x fill, w from_host, u zero, v custom_host, m custom (a device kernel
// writes 10 + i; the body ignores it), d declares nothing (zeros by default: the implicit zero initializer).

class InitDemo : public cs::Component<InitDemo> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x, w, u, v, d, m;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        out.y[i] = 10.0f * in.x[std::size_t{0}] + in.w[i] + in.v[i] + in.d[i];
    }

    Inputs  inputs;
    Outputs outputs;

private:
    cs::Output<DType::f32, 1> extra_;  // only ever declared by the phase-guard hook

public:
    void declare() override {
        declare_input(inputs.x, "x", 1);
        declare_input(inputs.w, "w", 4);
        declare_input(inputs.u, "u", 4);
        declare_input(inputs.v, "v", 4);
        declare_input(inputs.d, "d", 4);
        declare_input(inputs.m, "m", 4);
        declare_output(outputs.y, "y", 4);
        init_input("x", cs::init::fill(2.0f));
        init_input("w", cs::init::from_host(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
        init_input("u", cs::init::zero());
        init_input("v", cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostInit<DType::f32, 1>&)>{
                            [](cs::HostInit<DType::f32, 1>& h) {
                                ++g_host_calls;
                                for (std::size_t k = 0; k < 4; ++k) h[k] = 100.0f * static_cast<float>(k + 1);
                            }}));
        // The custom function gets the unordered queue, the device, the events its work must follow, and the
        // Initializer; it enqueues its own kernel with those events and RETURNS that kernel's event (init.hpp).
        init_input("m", cs::init::custom<DType::f32, 1>(cs::init::CustomFn<DType::f32, 1>{
                            [](sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
                               cs::Initializer<DType::f32, 1> out) {
                                ++g_custom_calls;
                                g_custom_size = out.size();
                                // A device kernel writes m[i] = 10 + i through the Initializer's operator[].
                                return q.parallel_for(dev, deps, sycl::range<1>{out.size()}, [=](sycl::id<1> i) {
                                    out[i] = 10.0f + static_cast<float>(i[0]);
                                });
                            }}));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }

    // Hooks for the phase-guard checks, called by the test after setup.
    void try_declare() { declare_output(extra_, "extra", 1); }
    void try_reshape() { set_shape(outputs.y, 2); }
    void try_init() { init_input("x", cs::init::zero()); }
};

class TopInit : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<InitDemo>("ini")); }
};

/// y = 2 x + 1 on a free 4-vector, with optional init declarations for the metadata rules:
/// mode 0 one init on x, 1 an init on the OUTPUT y, 2 two inits on x (one depth).
class InitAffine : public cs::Component<InitAffine> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> idx, const Inputs& in, const Outputs& out) { out.y[idx] = 2.0f * in.x[idx] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

private:
    std::optional<cs::InitSpec> init_;
    int                         mode_;
    bool                        tied_x_;

public:
    explicit InitAffine(std::optional<cs::InitSpec> init = std::nullopt, int mode = 0, bool tied_x = false)
        : init_{std::move(init)}, mode_{mode}, tied_x_{tied_x} {}
    void declare() override {
        if (tied_x_)
            declare_input(inputs.x, "x", cs::unspecified_shape<1>());  // a tie_inputs member: the tie alias owns the shape
        else
            declare_input(inputs.x, "x", 4);
        // y has the size of x: declared (4) when x is, else set by this producer from x (a number only on an unspecified dimension)
        if (tied_x_)
            declare_output(outputs.y, "y", cs::unspecified_shape<1>());
        else
            declare_output(outputs.y, "y", 4);
        if (init_) init_input("x", *init_);
        if (mode_ == 1) init_input("y", cs::init::zero());
        if (mode_ == 2) init_input("x", cs::init::fill(3.0f));
    }
    void prepare_forward() override {
        if (tied_x_) set_shape(outputs.y, shape_of(inputs.x)[0]);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
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
// Index-range bodies
// =============================================================================================
TEST_CASE("index-range body: y = 2x + 1 at ranks 1, 2 and 3", "[component][index-range]") {
    SECTION("rank 1, 1000") { tc::check_affine<1>(sycl::range<1>{1000}); }
    SECTION("rank 2, 40 x 30") { tc::check_affine<2>(sycl::range<2>{40, 30}); }
    SECTION("rank 3, 7 x 9 x 11") { tc::check_affine<3>(sycl::range<3>{7, 9, 11}); }
}

TEST_CASE("a launch's event is registered on the container of every port it touched", "[component][index-range][events]") {
    auto model = cs::create_model<tc::TopAffine<1>>(default_config(), sycl::range<1>{1000});
    model.setup();

    // Before any forward every holder already has its container (declaration creates it), but it is empty and unbound:
    // setup allocates nothing (design review section 2), so there are no events either.
    const auto xi = cs::detail::ModelAccess::instance(model, "aff.x");
    const auto yi = cs::detail::ModelAccess::instance(model, "aff.y");
    REQUIRE(xi->data != nullptr);
    REQUIRE(yi->data != nullptr);
    CHECK_FALSE(xi->data->bound());
    CHECK_FALSE(yi->data->bound());
    CHECK(xi->data->size() == 0);
    CHECK(xi->data->read_deps().empty());
    CHECK(xi->data->write_deps().empty());
    CHECK(yi->data->read_deps().empty());
    CHECK(yi->data->write_deps().empty());

    model.update_input("aff.x", cs::init::fill(1.0f));
    model.run();

    // The kernel READ x and WROTE y. Its event is the component's last event, and it is behind both containers' lists: a
    // writer of x (which waits on x's reads) and a reader of y (which waits on y's writes) both come after it. Checked
    // through the events themselves: after waiting on those lists the kernel's event is complete.
    const sycl::event kernel = [&] {
        const std::vector<sycl::event> ev = cs::detail::ModelAccess::subsystem(model, "aff")->events();
        REQUIRE(ev.size() == 1);
        return ev.front();
    }();
    REQUIRE(xi->data->bound());
    REQUIRE(yi->data->bound());
    const std::vector<sycl::event> x_writers_wait_on = xi->data->write_deps();
    const std::vector<sycl::event> y_readers_wait_on = yi->data->read_deps();
    REQUIRE_FALSE(x_writers_wait_on.empty());
    REQUIRE_FALSE(y_readers_wait_on.empty());
    sycl::event::wait(x_writers_wait_on);
    CHECK(kernel.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete);
    sycl::event::wait(y_readers_wait_on);
    model.wait();

    // A second forward registers again, and the result is still right.
    model.run();
    CHECK_FALSE(yi->data->read_deps().empty());
    auto y = wait_and_view<DType::f32, 1>(model, "aff.y");
    CHECK(to_vector(y) == std::vector<float>(1000, 3.0f));
}

// =============================================================================================
// WeightedHistogram
// =============================================================================================
TEST_CASE("WeightedHistogram: atomic output, unspecified dims, tie shape set by a from_value hook", "[component][histogram]") {
    auto model = cs::create_model<tc::TopHist>(default_config());
    model.setup();

    const std::uint32_t nb1 = 8, nb2 = 13;

    SECTION("first forward, n_bins = 8") {
        model.update_input("n_bins", cs::init::from_host(std::vector<std::uint32_t>{nb1}));
        auto bins = wait_and_view<DType::f32, 1>(model, "bins");
        CHECK(bins.size() == nb1);
        CHECK(to_vector(bins) == tc::histogram_ref(nb1));
    }

    SECTION("second forward with n_bins changed by update_input: the root grows and accumulates") {
        model.update_input("n_bins", cs::init::from_host(std::vector<std::uint32_t>{nb1}));
        {
            auto bins = wait_and_view<DType::f32, 1>(model, "bins");
            REQUIRE(bins.size() == nb1);
            CHECK(to_vector(bins) == tc::histogram_ref(nb1));
        }
        model.update_input("n_bins", cs::init::from_host(std::vector<std::uint32_t>{nb2}));
        auto bins = wait_and_view<DType::f32, 1>(model, "bins");
        REQUIRE(bins.size() == nb2);
        // ZeroBins zeroes the grown root each forward, then the histogram accumulates onto it.
        CHECK(to_vector(bins) == tc::histogram_ref(nb2));
    }
}

// =============================================================================================
// TiledTranspose
// =============================================================================================
TEST_CASE("TiledTranspose: group-range body, local memory, work-group override, 37 x 53", "[component][transpose]") {
    auto model = cs::create_model<tc::TopTranspose>(default_config());
    model.setup();
    auto out = wait_and_view<DType::f32, 2>(model, "tr.out");
    REQUIRE(out.shape()[0] == 53);
    REQUIRE(out.shape()[1] == 37);
    bool good = true;
    for (std::size_t j = 0; j < 53 && good; ++j)
        for (std::size_t i = 0; i < 37 && good; ++i) good = out(j, i) == static_cast<float>(i * 1000 + j);
    CHECK(good);  // out[j][i] == in[i][j]
}

TEST_CASE("group-range range that is not a multiple of the work-group shape throws, naming the component, range and shape",
          "[component][errors]") {
    tc::expect_throws(
        [] {
            auto model = cs::create_model<tc::TopBad>(default_config());
            model.setup();
            model.run();
            model.wait();
        },
        "is not a multiple of the work-group shape (2)");
    tc::expect_throws(
        [] {
            auto model = cs::create_model<tc::TopBad>(default_config());
            model.setup();
            model.run();
            model.wait();
        },
        "the range (5)");
    tc::expect_throws(
        [] {
            auto model = cs::create_model<tc::TopBad>(default_config());
            model.setup();
            model.run();
            model.wait();
        },
        "bad");
}

TEST_CASE("local memory: three arrays of different element types and odd shapes", "[component][local]") {
    auto model = cs::create_model<tc::TopLocalMixed>(default_config());
    model.setup();
    auto y = wait_and_view<DType::f32, 1>(model, "lm.y");
    REQUIRE(y.shape()[0] == 8);
    bool good = true;
    for (std::size_t g = 0; g < 2; ++g)
        for (std::size_t l = 0; l < 4; ++l) {
            const float want = static_cast<float>((l % 3 + 1) + 0.5 * static_cast<double>(l) + 10 * l);
            good = good && y(g * 4 + l) == want;
        }
    CHECK(good);
}

// =============================================================================================
// Host component chain
// =============================================================================================
TEST_CASE("host component chain: producer -> host prefix sum -> consumer", "[component][host]") {
    auto model = cs::create_model<tc::TopHostChain>(default_config());
    model.setup();

    // The host component's I/O are host-kind USM (moved by the ordinary copy rule).
    CHECK(cs::detail::ModelAccess::instance(model, "prefix.y")->storage_kind == sk::kind::host);
    CHECK(cs::detail::ModelAccess::instance(model, "prefix.z")->storage_kind == sk::kind::host);

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
    (void)wait_and_view<DType::f32, 1>(model, "plus1.w");  // runs the model; only the recorded pointers are checked

    // The device data of the variables, read off the instances (the old per-root storage_on(placement) is gone).
    const void* want_in  = cs::detail::ModelAccess::instance(model, "iota.y")->data->raw();
    const void* want_out = cs::detail::ModelAccess::instance(model, "rec.y")->data->raw();
    CHECK(tc::g_in_rec.load() != nullptr);
    CHECK(tc::g_in_rec.load() == want_in);
    CHECK(tc::g_out_rec.load() != nullptr);
    CHECK(tc::g_out_rec.load() == want_out);
    // (The recorder writes nothing and an output is never initialized, so its values are not checked.)
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
        CHECK(cst::synced_input<DType::f32, 1>(model, "ini.x")[std::size_t{0}] == 2.0f);  // fill
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "ini.w")) == ramp);       // from_host
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "ini.u")) == std::vector<float>(4, 0.0f));  // zero
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "ini.v")) ==
              std::vector<float>({100.0f, 200.0f, 300.0f, 400.0f}));  // custom_host
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "ini.d")) == std::vector<float>(4, 0.0f));  // default
    }
    SECTION("custom ran once, with the root's shape, and its device kernel wrote the values") {
        CHECK(tc::g_custom_calls.load() == 1);
        CHECK(tc::g_custom_size.load() == 4);
        CHECK(to_vector(cst::synced_input<DType::f32, 1>(model, "ini.m")) == std::vector<float>({10.0f, 11.0f, 12.0f, 13.0f}));
        CHECK(tc::g_host_calls.load() == 1);  // an eternal root is initialized once, at its owner's first run, not per forward
    }
    SECTION("the kernel sees the initialized inputs") {
        // y = 10 x + w + v + d = 20 + {1,2,3,4} + {100,200,300,400} + 0
        CHECK(to_vector(y) == std::vector<float>({121.0f, 222.0f, 323.0f, 424.0f}));
    }
}

TEST_CASE("initial values: metadata rules (connected, output, duplicates, last declaration wins)", "[component][init][rules]") {
    const auto top = [](std::function<void(cs::System&)> build) {
        return cs::create_model<tc::TopFn>(default_config(), std::move(build));
    };

    SECTION("an init on a connected input is ignored") {
        auto model = top([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tc::InitAffine>("p", std::optional<cs::InitSpec>{cs::init::fill(1.0f)}));
            s.add_subsystem(cs::subsystem_desc<tc::InitAffine>("c", std::optional<cs::InitSpec>{cs::init::fill(100.0f)}));
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
                    s.add_subsystem(cs::subsystem_desc<tc::InitAffine>("a", std::optional<cs::InitSpec>{}, 1));
                });
                model.setup();
            },
            "is an output");
    }
    SECTION("two members' inits and no tie-level init: the later merged member wins (the last declaration wins, section 2)") {
        auto model = top([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tc::InitAffine>("a", std::optional<cs::InitSpec>{cs::init::fill(1.0f)}, 0, true));
            s.add_subsystem(cs::subsystem_desc<tc::InitAffine>("b", std::optional<cs::InitSpec>{cs::init::fill(2.0f)}, 0, true));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(4));
        });
        CHECK_NOTHROW(model.setup());
        auto a = wait_and_view<DType::f32, 1>(model, "a.y");
        CHECK(to_vector(a) == std::vector<float>(4, 5.0f));  // b's fill(2) is the later member: 2 * 2 + 1
    }
    SECTION("a System-level init beats the components' on a tied alias") {
        auto model = top([](cs::System& s) {
            // Both members carry a component-level init; the System-level init_input runs after their setups, so it is the
            // last declaration and wins, with no throw (section 2).
            s.add_subsystem(cs::subsystem_desc<tc::InitAffine>("a", std::optional<cs::InitSpec>{cs::init::fill(1.0f)}, 0, true));
            s.add_subsystem(cs::subsystem_desc<tc::InitAffine>("b", std::optional<cs::InitSpec>{cs::init::fill(2.0f)}, 0, true));
            s.tie_inputs<DType::f32, 1>({"a.x", "b.x"}, "x", cs::shape(4));
            s.init_input("x", cs::init::fill(9.0f));
        });
        model.setup();
        auto a = wait_and_view<DType::f32, 1>(model, "a.y");
        CHECK(to_vector(a) == std::vector<float>(4, 19.0f));
        CHECK(to_vector(cst::synced_output<DType::f32, 1>(model, "b.y")) == std::vector<float>(4, 19.0f));
    }
}

TEST_CASE("phase guards: init_input, set_shape and declare outside their phase throw", "[component][init][guards]") {
    auto model = cs::create_model<tc::TopInit>(default_config());
    model.setup();
    auto& c = dynamic_cast<tc::InitDemo&>((*cs::detail::ModelAccess::component(model, "ini")));

    SECTION("init_input outside declare() throws") {
        tc::expect_throws([&] { c.try_init(); }, "declare() only");
    }
    SECTION("set_shape outside prepare_forward() throws") {
        tc::expect_throws([&] { c.try_reshape(); }, "prepare_forward() only");
    }
    SECTION("declare outside declare() throws") {
        tc::expect_throws([&] { c.try_declare(); }, "declare() only");
    }
    SECTION("the model is unharmed by the refused calls") {
        tc::expect_throws([&] { c.try_init(); }, "declare() only");
        auto y = wait_and_view<DType::f32, 1>(model, "ini.y");
        CHECK(to_vector(y) == std::vector<float>({121.0f, 222.0f, 323.0f, 424.0f}));
    }
    model.wait();  // a forward may have been enqueued; let it finish before the Model goes out of scope
}

// =============================================================================================
// The synthesized kernel body of a component, as traits (no Model: the kind, the rank, the descriptors)
// =============================================================================================
namespace tct {

// Two components with the SAME ports and the SAME compute_forward signature.
class IndexA : public cs::Component<IndexA> {
public:
    struct Inputs { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = in.x[i]; }
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", 4);
        declare_output(outputs.y, "y", 4);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};
class IndexB : public cs::Component<IndexB> {
public:
    struct Inputs { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = in.x[i]; }
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", 4);
        declare_output(outputs.y, "y", 4);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

class Index3 : public cs::Component<Index3> {
public:
    struct Inputs { cs::Input<DType::f32, 3> x; };
    struct Outputs { cs::Output<DType::f32, 3> y; };
    static void compute_forward(sycl::id<3> i, const Inputs& in, const Outputs& out) { out.y[i] = in.x[i]; }
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", 2, 2, 2);
        declare_output(outputs.y, "y", 2, 2, 2);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

class Grp2 : public cs::Component<Grp2> {
public:
    struct Inputs { cs::Input<DType::f32, 2> x; };
    struct Outputs { cs::Output<DType::f32, 2> y; };
    static void compute_forward(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) { out.y[it.get_global_id()] = in.x[it.get_global_id()]; }
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", 4, 4);
        declare_output(outputs.y, "y", 4, 4);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()), cs::wg_desc(sycl::range<2>{2, 2}));
    }
};

class GrpLocal : public cs::Component<GrpLocal> {
public:
    struct Inputs { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };
    using Mem = cs::LocalMem<cs::LocalArray<DType::f32, 1>>;
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out, Mem) {
        out.y[it.get_global_id()] = in.x[it.get_global_id()];
    }
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", 4);
        declare_output(outputs.y, "y", 4);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()), cs::wg_desc(sycl::range<1>{4}), cs::local_desc(sycl::range<1>{4}));
    }
};

class Custom : public cs::CustomComponent<Custom> {
public:
    struct Inputs { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };
    static void compute_forward(cs::CustomOp&, const Inputs&, const Outputs&) {}
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", 4);
        declare_output(outputs.y, "y", 4);
    }
};

class HostC : public cs::HostComponent<HostC> {
public:
    struct Inputs { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };
    static void compute_forward(const Inputs&, const Outputs&) {}
    Inputs  inputs;
    Outputs outputs;
    void declare() override {
        declare_input(inputs.x, "x", 4);
        declare_output(outputs.y, "y", 4);
    }
};

/// Whether create_launch_config(Ds...) is accepted for the component X (the constraint of the member).
template <class X, class... Ds>
inline constexpr bool accepts_v = cs::detail::launch_descs_ok_v<cs::detail::SynthBody<X>, Ds...>;

template <class X>
using body_of = cs::detail::SynthBody<X>;

using R1 = cs::RangeDesc<1>;
using R2 = cs::RangeDesc<2>;
using W1 = cs::WgDesc<1>;
using W2 = cs::WgDesc<2>;
using L1  = cs::LocalDesc<1>;
using L2  = cs::LocalDesc<2>;
using L11 = cs::LocalDesc<1, 1>;

}  // namespace tct

TEST_CASE("component form: the synthesized body is trivially copyable, and distinct per component", "[component][traits]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<tct::body_of<tct::IndexA>>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<tct::body_of<tct::Grp2>>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<tct::body_of<tct::GrpLocal>>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<tct::body_of<tct::Custom>>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<tct::body_of<tct::HostC>>);
    // identical ports and an identical compute_forward signature, yet two body types (one kernel per component, by type)
    STATIC_REQUIRE(std::is_same_v<tct::IndexA::Inputs, tct::IndexA::Inputs>);
    STATIC_REQUIRE(!std::is_same_v<tct::body_of<tct::IndexA>, tct::body_of<tct::IndexB>>);
    STATIC_REQUIRE(!std::is_same_v<tct::IndexA::body_type, tct::IndexB::body_type>);
}

TEST_CASE("component form: the first parameter of compute_forward fixes the kind and the rank", "[component][traits]") {
    STATIC_REQUIRE(cs::IndexRangeBody<tct::body_of<tct::IndexA>>);
    STATIC_REQUIRE(cs::kernel_rank_v<tct::body_of<tct::IndexA>> == 1);
    STATIC_REQUIRE(cs::IndexRangeBody<tct::body_of<tct::Index3>>);
    STATIC_REQUIRE(cs::kernel_rank_v<tct::body_of<tct::Index3>> == 3);

    STATIC_REQUIRE(cs::GroupRangeBody<tct::body_of<tct::Grp2>>);
    STATIC_REQUIRE(!cs::IndexRangeBody<tct::body_of<tct::Grp2>>);
    STATIC_REQUIRE(cs::kernel_rank_v<tct::body_of<tct::Grp2>> == 2);
    STATIC_REQUIRE(cs::GroupRangeBody<tct::body_of<tct::GrpLocal>>);
    STATIC_REQUIRE(cs::HasLocalMem<tct::body_of<tct::GrpLocal>>);
    STATIC_REQUIRE(!cs::HasLocalMem<tct::body_of<tct::Grp2>>);

    STATIC_REQUIRE(cs::CustomOpBody<tct::body_of<tct::Custom>>);
    STATIC_REQUIRE(!cs::IndexRangeBody<tct::body_of<tct::Custom>>);
    STATIC_REQUIRE(cs::kernel_rank_v<tct::body_of<tct::Custom>> == 0);
    STATIC_REQUIRE(cs::launch_rank_v<tct::body_of<tct::Custom>> == 1);

    STATIC_REQUIRE(cs::HostBody<tct::body_of<tct::HostC>>);
    STATIC_REQUIRE(!cs::CustomOpBody<tct::body_of<tct::HostC>>);
    STATIC_REQUIRE(cs::kernel_rank_v<tct::body_of<tct::HostC>> == 0);

    STATIC_REQUIRE(cs::KernelBodyC<tct::body_of<tct::IndexA>>);
    STATIC_REQUIRE(cs::KernelBodyC<tct::body_of<tct::Grp2>>);
    STATIC_REQUIRE(cs::KernelBodyC<tct::body_of<tct::GrpLocal>>);
    STATIC_REQUIRE(cs::KernelBodyC<tct::body_of<tct::Custom>>);
    STATIC_REQUIRE(cs::KernelBodyC<tct::body_of<tct::HostC>>);
}

TEST_CASE("component form: each launch-descriptor misuse is rejected by a detection trait", "[component][traits]") {
    using namespace tct;
    // accepted forms, in any order
    STATIC_REQUIRE(accepts_v<IndexA, R1>);
    STATIC_REQUIRE(accepts_v<Grp2, W2, R2>);
    STATIC_REQUIRE(accepts_v<GrpLocal, R1, W1, L1>);
    STATIC_REQUIRE(accepts_v<GrpLocal, L1, W1, R1>);

    // a group-range body with a range_desc but no wg_desc (the work-group shape is required), with and without local_desc
    STATIC_REQUIRE(!accepts_v<Grp2, R2>);
    STATIC_REQUIRE(!accepts_v<GrpLocal, R1>);
    STATIC_REQUIRE(!accepts_v<GrpLocal, R1, L1>);
    // wg_desc on an index-range body
    STATIC_REQUIRE(!accepts_v<IndexA, R1, W1>);
    STATIC_REQUIRE(!accepts_v<IndexA, W1, R1>);
    // local_desc on an index-range body
    STATIC_REQUIRE(!accepts_v<IndexA, R1, L1>);
    // local_desc on a group-range body without local memory
    STATIC_REQUIRE(!accepts_v<Grp2, R2, L1>);
    // local_desc whose count or ranks differ from the body's LocalMem
    STATIC_REQUIRE(!accepts_v<GrpLocal, R1, W1, L11>);
    STATIC_REQUIRE(!accepts_v<GrpLocal, R1, W1, L2>);

    // a missing range, for index-range and for group-range bodies
    STATIC_REQUIRE(!accepts_v<IndexA>);
    STATIC_REQUIRE(!accepts_v<IndexA, W1>);
    STATIC_REQUIRE(!accepts_v<Grp2>);
    STATIC_REQUIRE(!accepts_v<Grp2, W2>);
    STATIC_REQUIRE(!accepts_v<GrpLocal, W1, L1>);
    // a range of the wrong rank
    STATIC_REQUIRE(!accepts_v<IndexA, R2>);
    STATIC_REQUIRE(!accepts_v<Grp2, R1>);
    // a repeated descriptor
    STATIC_REQUIRE(!accepts_v<IndexA, R1, R1>);
    STATIC_REQUIRE(!accepts_v<Grp2, R2, W2, W2>);
    // wg_desc or range_desc (or anything) on a custom-op or host component
    STATIC_REQUIRE(!accepts_v<Custom, W1>);
    STATIC_REQUIRE(!accepts_v<Custom, R1>);
    STATIC_REQUIRE(!accepts_v<Custom, R1, W1>);
    STATIC_REQUIRE(!accepts_v<HostC, W1>);
    STATIC_REQUIRE(!accepts_v<HostC, R1>);
    STATIC_REQUIRE(!accepts_v<HostC, R1, W1>);
    // not a descriptor at all
    STATIC_REQUIRE(!accepts_v<IndexA, R1, int>);
}
