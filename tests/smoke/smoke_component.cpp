// Smoke check for Component<KB> (not the test suite): components are driven
// WITHOUT a System through a test-only TestDriver that performs, per verb, what
// the owner would. Components are connected by merging entries (EntryAccess).
//
//   a. WeightedHistogram: atomic output, FromSourceDim, value-driven n_bins
//      (FromInput on the output), fed by a producer component; two forwards, the
//      second with a larger n_bins (accumulate-onto a grown, zero-padded root).
//   b. TiledTranspose: GroupItem<2> body, Local scratch, work-group override.
//   c. A rank-2 plain body, y = 2x + 1.
//   d. A host component (sequential prefix sum) between two GPU components on
//      one device: its I/O are host-kind USM, moved by the ordinary copy rule.
//   e. Custom-op plumbing: the body records the pointers it is handed and
//      enqueues nothing; a plain component after it still runs in order.

#include <atomic>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;
using cs::detail::EntryAccess;
using cs::detail::HostViewAccess;

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// ---- the test-only owner -----------------------------------------------------
namespace naga::cospan::detail {

struct TestDriver {
    sycl::device                 dev;
    sk::queue&                   q;
    std::vector<cs::Subsystem*>  comps;
    cs::Substrates               subs;

    TestDriver(const sycl::device& d, sk::queue& queue) : dev{d}, q{queue} {}

    void add(cs::Subsystem& c, std::string name) {
        SubsystemAccess::place(c, name, name, nullptr);
        comps.push_back(&c);
    }

    void run_setup() {
        for (auto* c : comps) c->run_setup();
    }

    /// The entry called `name` of `c` (inputs and outputs), for wiring and readback.
    static IOEntry& entry(cs::ComponentBase& c, std::string_view name) {
        for (const auto& e : c.get_inputs())
            if (e.get().name() == name) return const_cast<IOEntry&>(e.get());
        for (const auto& e : c.get_outputs())
            if (e.get().name() == name) return const_cast<IOEntry&>(e.get());
        throw std::out_of_range("TestDriver: no entry " + std::string{name});
    }

    /// `consumer.in` becomes a peer of `producer.out`.
    static void connect(cs::ComponentBase& producer, std::string_view out, cs::ComponentBase& consumer,
                        std::string_view in) {
        EntryAccess::merge(entry(producer, out), entry(consumer, in));
    }

    /// get_exec_info -> substrates -> prepare_exec -> compile -> run_init.
    void start() {
        std::size_t total = 0;
        for (auto* c : comps)
            for (const auto& [d, bytes] : c->get_exec_info(dev)) total += bytes;
        // Slack for the arena's per-allocation alignment.
        const std::size_t slack = 4096 * (comps.size() + 1);
        subs.emplace(dev, cs::MemorySubstrate{sk::DeviceArena{q, dev, sk::kind::device, total + slack},
                                              sk::DevicePool{q, dev, sk::kind::device, 1 << 20}});
        for (auto* c : comps) c->prepare_exec(q, subs);
        for (auto* c : comps) c->compile();
        cs::ExecCtx ctx{dev, &q, nullptr, &subs.at(dev)};
        for (auto* c : comps) c->run_init(ctx);
    }

    /// One forward: per component pre_compute -> compute_forward -> submit -> post_compute.
    void forward() {
        cs::ExecCtx ctx{dev, &q, nullptr, &subs.at(dev)};
        for (auto* c : comps) {
            c->run_pre_compute(ctx);
            if (auto d = c->run_compute_forward()) {
                auto t = ctx.target();
                d->submit(t);
            }
            c->run_post_compute(ctx);
        }
        q.wait();
    }
};

}  // namespace naga::cospan::detail

using cs::detail::TestDriver;

// ---- a. producer + WeightedHistogram -----------------------------------------
constexpr std::size_t kN = 1000;

static float value_of(std::size_t i) { return (static_cast<float>((i * 7) % 1000) + 0.5f) / 1000.0f; }
static float weight_of(std::size_t i) { return static_cast<float>(1 + i % 3); }

struct ProducerBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> values;
        cs::Output<DType::f32, 1> weights;
        cs::Output<DType::u32, 1> n_bins;
    };
    uint32_t nb;
    void operator()(sycl::id<1> i, Inputs, Outputs out) const {
        out.values[i]  = (static_cast<float>((i[0] * 7) % 1000) + 0.5f) / 1000.0f;
        out.weights[i] = static_cast<float>(1 + i[0] % 3);
        if (i[0] == 0) out.n_bins[0] = nb;
    }
};

class Producer : public cs::Component<ProducerBody> {
    cs::Output<DType::f32, 1> values_, weights_;
    cs::Output<DType::u32, 1> n_bins_;

public:
    uint32_t nb = 8;
    void setup() override {
        declare_output(values_, "values", static_cast<int>(kN));
        declare_output(weights_, "weights", static_cast<int>(kN));
        declare_output(n_bins_, "n_bins", 1);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{kN}, {}, {values_, weights_, n_bins_}, {nb});
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
        const auto b = static_cast<std::size_t>((in.values[i] - lo) / (hi - lo) *
                                                static_cast<float>(in.n_bins[0]));
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
        declare_input(values_, "values", cs::FromSourceDim{0});
        declare_input(weights_, "weights", cs::from_input_shape<1>("values"));
        declare_input(n_bins_, "n_bins", 1);
        declare_output(bins_, "bins", cs::FromInput{"n_bins", 0});
    }
    void init(sk::queue&, const sycl::device&) override { zero_output("bins"); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(values_.shape(), {values_, weights_, n_bins_}, {bins_}, {lo_, hi_});
    }
};

static std::vector<float> histogram_ref(uint32_t nb) {
    std::vector<float> ref(nb, 0.0f);
    for (std::size_t i = 0; i < kN; ++i)
        ref[static_cast<std::size_t>((value_of(i) - 0.0f) / (1.0f - 0.0f) * static_cast<float>(nb))] +=
            weight_of(i);
    return ref;
}

static bool bins_equal(sk::queue& q, cs::detail::IOEntry& bins, const std::vector<float>& want) {
    auto h = HostViewAccess::view<DType::f32, 1>(bins, q);
    if (h.size() != want.size()) return false;
    for (std::size_t i = 0; i < want.size(); ++i)
        if (h[i] != want[i]) return false;
    return true;
}

// ---- b. tiled transpose ----------------------------------------------------------
struct Iota2DBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> out;
    };
    void operator()(sycl::id<2> ij, Inputs, Outputs o) const {
        o.out[ij] = static_cast<float>(ij[0] * 1000 + ij[1]);
    }
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
        if (gt[0] < out.out.shape()[0] && gt[1] < out.out.shape()[1])
            out.out[gt] = s.tile[{l[1], l[0]}];
    }
};

class TiledTranspose : public cs::Component<TransposeBody> {
    cs::Input<DType::f32, 2>  in_;
    cs::Output<DType::f32, 2> out_;
    static constexpr std::size_t WG = 16;

public:
    void setup() override {
        declare_input(in_, "in", cs::from_source_shape<2>());
        declare_output(out_, "out", cs::FromInputDim{"in", 1}, cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        TransposeBody::Scratch s;
        s.tile.reshape({WG, WG + 1});
        return create_launch_config(in_.shape(), {WG, WG}, {in_}, {out_}, s, {});
    }
};

// ---- c. a rank-2 plain body ---------------------------------------------------------
struct AffineBody {
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    void operator()(sycl::id<2> ij, Inputs in, Outputs out) const { out.y[ij] = 2.0f * in.x[ij] + 1.0f; }
};

class Affine : public cs::Component<AffineBody> {
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

// ---- the run --------------------------------------------------------------------------
static void check_transpose(sk::queue& q, cs::detail::IOEntry& out) {
    auto h = HostViewAccess::view<DType::f32, 2>(out, q);
    bool good = h.shape()[0] == 53 && h.shape()[1] == 37;
    for (std::size_t j = 0; j < 53 && good; ++j)
        for (std::size_t i = 0; i < 37 && good; ++i)
            good = h[sycl::id<2>{j, i}] == static_cast<float>(i * 1000 + j);
    check(good, "TiledTranspose: 37x53 -> 53x37, out[j][i] == in[i][j]");
}

static void check_affine(sk::queue& q, cs::detail::IOEntry& y) {
    auto h = HostViewAccess::view<DType::f32, 2>(y, q);
    bool good = h.shape()[0] == 40 && h.shape()[1] == 30;
    for (std::size_t i = 0; i < 40 && good; ++i)
        for (std::size_t j = 0; j < 30 && good; ++j)
            good = h[sycl::id<2>{i, j}] == 2.0f * static_cast<float>(i * 1000 + j) + 1.0f;
    check(good, "rank-2 plain body: y == 2x + 1 on 40x30");
}

// ---- shared pieces of d and e: a 1-D iota producer and a "+1" consumer ------------------
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
        declare_input(z_, "z", cs::from_source_shape<1>());
        declare_output(w_, "w", cs::from_input_shape<1>("z"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(z_.shape(), {z_}, {w_}, {});
    }
};

// ---- d. a host component between two GPU components ---------------------------------------
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
        declare_input(y_, "y", cs::from_source_shape<1>());
        declare_output(z_, "z", cs::from_input_shape<1>("y"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config({y_}, {z_}, {});
    }
};

static void case_d(const sycl::device& dev, sk::queue& q) {
    std::printf("-- d. host component chain: GPU iota -> host prefix sum -> GPU +1\n");
    Iota1D    iota{static_cast<int>(kN)};
    PrefixSum prefix;
    Plus1     plus1;
    TestDriver d{dev, q};
    d.add(iota, "iota");
    d.add(prefix, "prefix");
    d.add(plus1, "plus1");
    d.run_setup();
    TestDriver::connect(iota, "y", prefix, "y");
    TestDriver::connect(prefix, "z", plus1, "z");
    d.start();

    // The host component's input lives in host-kind USM on the producer's device.
    auto& y_root = *TestDriver::entry(prefix, "y").root();
    auto& z_root = *TestDriver::entry(prefix, "z").root();
    const cs::Placement dev_pl{dev, sk::kind::device}, host_pl{dev, sk::kind::host};
    check(TestDriver::entry(prefix, "y").storage_kind() == sk::kind::host &&
              TestDriver::entry(prefix, "y").placement() == host_pl,
          "d: the host component's input entry is placed (device, host)");
    check(y_root.storage_placements().size() == 2 && y_root.storage_placements()[0] == dev_pl &&
              y_root.storage_placements()[1] == host_pl,
          "d: the producer's root holds [(dev, device) primary, (dev, host)]");
    check(y_root.storage_on(host_pl) != nullptr && y_root.storage_on(host_pl) != y_root.storage_on(dev_pl),
          "d: the host-kind slot is its own allocation");
    check(z_root.storage_placements().size() == 2 && z_root.storage_placements()[0] == host_pl &&
              z_root.storage_placements()[1] == dev_pl,
          "d: the host component's output root holds [(dev, host) primary, (dev, device)]");

    d.forward();
    auto& w = TestDriver::entry(plus1, "w");
    auto  h = HostViewAccess::view<DType::f32, 1>(w, q);
    bool  good = h.size() == kN;
    float acc  = 0.0f;
    for (std::size_t i = 0; i < kN && good; ++i) {
        acc += static_cast<float>(i);
        good = h[i] == acc + 1.0f;
    }
    check(good, "d: w[i] == prefix_sum(i)[i] + 1 against the host reference (N=1000)");
}

// ---- e. custom-op plumbing (no native work) -------------------------------------------------
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
        declare_input(x_, "x", cs::from_source_shape<1>());
        declare_output(y_, "y", cs::from_input_shape<1>("x"));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config({x_}, {y_}, {&in_rec, &out_rec});
    }
};

static void case_e(const sycl::device& dev, sk::queue& q) {
    std::printf("-- e. custom-op plumbing: pointers recorded at submission, nothing enqueued\n");
    Iota1D   iota{static_cast<int>(kN)};
    Recorder rec;
    Plus1    plus1;
    TestDriver d{dev, q};
    d.add(iota, "iota");
    d.add(rec, "rec");
    d.add(plus1, "plus1");
    d.run_setup();
    TestDriver::connect(iota, "y", rec, "x");
    TestDriver::connect(rec, "y", plus1, "z");
    d.start();
    d.forward();

    const cs::Placement dev_pl{dev, sk::kind::device};
    const void* want_in  = TestDriver::entry(iota, "y").storage_on(dev_pl);
    const void* want_out = TestDriver::entry(rec, "y").storage_on(dev_pl);
    check(rec.in_rec.load() != nullptr && rec.in_rec.load() == want_in,
          "e: device_ptr(in) is the producer root's (device, device) storage");
    check(rec.out_rec.load() != nullptr && rec.out_rec.load() == want_out,
          "e: device_ptr(out) is the recorder's own root storage");

    // The recorder wrote nothing, so its zero-filled output feeds the plain
    // consumer: w == 0 + 1 everywhere, after the custom op in stream order.
    auto h = HostViewAccess::view<DType::f32, 1>(TestDriver::entry(plus1, "w"), q);
    bool good = h.size() == kN;
    for (std::size_t i = 0; i < kN && good; ++i) good = h[i] == 1.0f;
    check(good, "e: the plain component after the custom op produces w == 1 (ordering intact)");
}

int main() {
    try {
        sycl::device dev{sycl::default_selector_v};
        std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        sk::context ctx(dev);
        sk::queue   q(ctx, dev);

        Producer          producer;
        WeightedHistogram hist{0.0f, 1.0f};
        Iota2D            iota_t{37, 53}, iota_a{40, 30};
        TiledTranspose    transpose;
        Affine            affine;

        TestDriver d{dev, q};
        d.add(producer, "producer");
        d.add(hist, "hist");
        d.add(iota_t, "iota_t");
        d.add(transpose, "transpose");
        d.add(iota_a, "iota_a");
        d.add(affine, "affine");
        d.run_setup();
        TestDriver::connect(producer, "values", hist, "values");
        TestDriver::connect(producer, "weights", hist, "weights");
        TestDriver::connect(producer, "n_bins", hist, "n_bins");
        TestDriver::connect(iota_t, "out", transpose, "in");
        TestDriver::connect(iota_a, "out", affine, "x");
        d.start();
        std::printf("-- started: setup, exec info, substrates, prepare, compile, init\n");

        auto& bins = TestDriver::entry(hist, "bins");
        auto& tout = TestDriver::entry(transpose, "out");
        auto& ayout = TestDriver::entry(affine, "y");

        producer.nb = 8;
        d.forward();
        check(bins_equal(q, bins, histogram_ref(8)), "forward 1: WeightedHistogram n_bins=8 matches the host reference");
        check_transpose(q, tout);
        check_affine(q, ayout);

        producer.nb = 13;  // value-driven resize: the bins root grows 8 -> 13
        d.forward();
        std::vector<float> want = histogram_ref(13);
        const std::vector<float> first = histogram_ref(8);
        for (std::size_t i = 0; i < first.size(); ++i) want[i] += first[i];  // accumulate onto the zero-padded old bins
        check(bins_equal(q, bins, want), "forward 2: n_bins=13, grown root == zero-padded first bins + new histogram");
        check_transpose(q, tout);
        check_affine(q, ayout);

        case_d(dev, q);
        case_e(dev, q);
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    std::printf(failures == 0 ? "SMOKE COMPONENT PASS\n" : "SMOKE COMPONENT FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
