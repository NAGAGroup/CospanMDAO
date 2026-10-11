// Fused groups (P7a): segments, range modes and local memory, every case through a cospan::Model.
//
//   segments        two segments: values right, each stage sees its own segment's group count
//   range modes     own: a stage sees its own group range; whole: every stage runs over the segment's block
//   local memory    partitioned: stages have disjoint slices; shared: stage 2 reads what stage 1 left (after a group barrier)
//   errors          stages with different work-group shapes are refused at forward, naming both
//   paths           a bare component (a group of one stage); one component type in two groups; a transient group rebuilt
//                   every forward (the process-wide config cache outlives every kernel)
//
// The components live in namespace tfg.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;
using cst::to_vector;

namespace tfg {

constexpr std::size_t kWg = 64;

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

class Top : public cs::System {
    Build build_;

public:
    explicit Top(Build b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

cs::Model make(Build b) { return cs::create_model<Top>(default_config(), std::move(b)); }

std::vector<float> out_of(cs::Model& m, const char* path) { return to_vector(cst::synced_output<DType::f32, 1>(m, path)); }

// ---- Src: y[i] = i + 1 ---------------------------------------------------------------------------

class Src : public cs::Component<Src> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs&, const Outputs& out) { out.y[i] = static_cast<float>(i[0] + 1); }

    Inputs  inputs;
    Outputs outputs;

private:
    int n_;

public:
    explicit Src(int n) : n_{n} {}
    void setup() override { declare_output(outputs.y, "y", n_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }

private:
};

// ---- Plus1 (plain): y = x + 1, the output follows the input ----------------------------------------

class Plus1 : public cs::Component<Plus1> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = in.x[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_output(outputs.y, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(inputs.x.shape())); }
};

// ---- group stages ---------------------------------------------------------------------------------
// One component template, GStage<Kind>: the kind is a compile-time parameter that picks the body (no run-time branch).
//   kTag      y[g] = segment * 1000 + groups(segment 0) * 100 + groups(segment 1) * 10 + segment count (the second term is 0
//             with one segment): what the stage sees of its segment structure.
//   kRange    y[g] = the group range the stage sees (own: its own groups; whole: the segment's block).
//   kLeave    shared local memory, stage 1: leaves t[l] = l + 1 in the stage's slice and passes x through.
//   kTake     shared local memory, stage 2: reads the reversed tile after a group barrier, writing nothing itself.
//   kOwn<B>   partitioned local memory: write t[l] = B + l into the stage's OWN slice, barrier, read the reversed tile.
//   kWrite    segment 1 of the cross-group cases: y[i] = x[i] + 1 (every group writes its own slice).
//   kReadNext<Mode>  segment 2: group g copies the slice a DIFFERENT group (g + 1) wrote. Mode 0: nothing here (a
//             cs::global_barrier() before the segment does the waiting); 1: wait_group(0, (g + 1) % G); 2: wait_earlier_segments().
constexpr int kTag  = 0;
constexpr int kRange = 1;
constexpr int kLeave = 2;
constexpr int kTake  = 3;
constexpr int kWrite = 4;
template <int Mode>
constexpr int kReadNext = 10 + Mode;
template <int Base>
constexpr int kOwn = 1000 + Base;

/// A group stage over its input's shape, with work-group `wg` and `local_bytes` of requested local memory.
template <int Kind>
class GStage : public cs::Component<GStage<Kind>> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::size_t g = it.get_global_id(0);
        if constexpr (Kind == kTag) {
            if (g >= in.x.shape()[0]) return;
            const std::size_t segs = it.get_segment_count();
            const std::size_t gc0  = it.get_segment_group_count(0)[0];
            const std::size_t gc1  = segs > 1 ? it.get_segment_group_count(1)[0] : 0;
            out.y[g]               = static_cast<float>(it.get_segment() * 1000 + gc0 * 100 + gc1 * 10 + segs);
        } else if constexpr (Kind == kRange) {
            if (g < in.x.shape()[0]) out.y[g] = static_cast<float>(it.get_group_range(0));
        } else if constexpr (Kind == kLeave) {
            float* t = it.template local<float>(kWg);
            if (t != nullptr) t[it.get_local_linear_id()] = static_cast<float>(it.get_local_linear_id() + 1);
            if (g < in.x.shape()[0]) out.y[g] = in.x[g];
        } else if constexpr (Kind == kTake) {
            float* t = it.template local<float>(kWg);
            cs::group_barrier(it.get_group());
            if (g < in.x.shape()[0]) out.y[g] = t != nullptr ? t[kWg - 1 - it.get_local_linear_id()] : -1.0f;
        } else if constexpr (Kind >= 1000) {
            constexpr int Base = Kind - 1000;
            float*        t    = it.template local<float>(kWg);
            if (t != nullptr) t[it.get_local_linear_id()] = static_cast<float>(Base + it.get_local_linear_id());
            cs::group_barrier(it.get_group());
            if (g < in.x.shape()[0]) out.y[g] = t != nullptr ? t[kWg - 1 - it.get_local_linear_id()] : -1.0f;
        } else if constexpr (Kind == kWrite) {
            if (g < in.x.shape()[0]) out.y[g] = in.x[g] + 1.0f;
        } else {
            constexpr int Mode = Kind - 10;
            const std::size_t G = it.get_group_range(0), gi = it.get_group(0), wg = it.get_local_range(0), l = it.get_local_linear_id();
            if constexpr (Mode == 1) it.wait_group(0, sycl::id<1>{(gi + 1) % G});  // collective: every work-item calls it
            if constexpr (Mode == 2) it.wait_earlier_segments();                  // collective
            out.y[gi * wg + l] = in.x[((gi + 1) % G) * wg + l];
        }
    }

    Inputs  inputs;
    Outputs outputs;

private:
    std::size_t wg_, local_bytes_;

public:
    explicit GStage(std::size_t wg = kWg, std::size_t local_bytes = 0) : wg_{wg}, local_bytes_{local_bytes} {}
    void setup() override {
        this->declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        this->declare_output(outputs.y, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { this->set_shape(outputs.y, this->shape_of(inputs.x)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = this->create_launch_config(cs::range_desc(inputs.x.shape()), cs::wg_desc(sycl::range<1>{wg_}));
        if (local_bytes_ != 0) d->request_local_bytes(local_bytes_);
        return d;
    }
};

constexpr std::size_t kTile = kWg * sizeof(float);

/// y[g] = the neighbor group's value + 1, over `forwards` forwards with new inputs every time (so a stale wait state or a
/// missed wait shows). Kind: 1 = the group_waits_enabled property, no barrier; 2 = a global_barrier() before segment 2 (the
/// barrier alone gives the group its waits). Mode: see kReadNext.
template <int Mode, int Kind>
void neighbor_case(std::size_t wg, std::size_t G, int forwards) {
    cs::Model model = make([wg](cs::System& s) {
        auto a = cs::subsystem_desc<GStage<kWrite>>("a", wg);
        auto b = cs::subsystem_desc<GStage<kReadNext<Mode>>>("b", wg);
        if constexpr (Kind == 2)
            s.add_subsystem(cs::fuse_components("f", cs::connects_desc({{"a.y", "b.x"}}), cs::segment(a), cs::global_barrier(), cs::segment(b)));
        else
            s.add_subsystem(cs::fuse_components<cs::group_waits_enabled>("f", cs::connects_desc({{"a.y", "b.x"}}), cs::segment(a), cs::segment(b)));
    });
    model.setup();
    const std::size_t N = G * wg;
    for (int k = 0; k < forwards; ++k) {
        INFO("forward " << k << ", " << G << " groups of " << wg);
        std::vector<float> x(N), want(N);
        for (std::size_t i = 0; i < N; ++i) x[i] = static_cast<float>(k * 1000000 + i);
        for (std::size_t g = 0; g < G; ++g)
            for (std::size_t l = 0; l < wg; ++l) want[g * wg + l] = x[((g + 1) % G) * wg + l] + 1.0f;
        model.update_input("f.a.x", cs::init::from_host(x));
        model.run();
        model.wait();
        CHECK(out_of(model, "f.b.y") == want);
    }
}

/// {work-group size, groups} of the two sizes of every cross-group case: a small one, and one with at least 4x the compute
/// units in groups of the largest work-group the device allows, which cannot all be resident at once on a GPU.
struct Shape {
    std::size_t wg, groups;
};
inline Shape big_shape() {
    const sycl::device dev{};
    const std::size_t  wg = std::min<std::size_t>(1024, dev.get_info<sycl::info::device::max_work_group_size>());
    const std::size_t  cu = std::max<std::size_t>(1, dev.get_info<sycl::info::device::max_compute_units>());
    return Shape{wg, 4 * cu};
}

std::vector<float> filled(std::size_t n, float v) { return std::vector<float>(n, v); }

}  // namespace tfg

using TagStage   = tfg::GStage<tfg::kTag>;
using RangeStage = tfg::GStage<tfg::kRange>;

TEST_CASE("fused groups: two segments, each stage sees its own segment's group count", "[fused_groups]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sa", 200));  // 4 groups of 64
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sb", 100));  // 2 groups
        s.add_subsystem(cs::fuse_components("f", cs::segment(cs::subsystem_desc<TagStage>("a")),
                                            cs::segment(cs::subsystem_desc<TagStage>("b"))));
        s.connect("sa.y", "f.a.x");
        s.connect("sb.y", "f.b.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 3; ++fwd) {
        model.run();
        model.wait();
        CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(200, 0 * 1000 + 4 * 100 + 2 * 10 + 2));
        CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(100, 1 * 1000 + 4 * 100 + 2 * 10 + 2));
    }
}

TEST_CASE("fused groups: a bare descriptor is a one-stage segment", "[fused_groups]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sa", 200));
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sb", 100));
        s.add_subsystem(cs::fuse_components("f", cs::subsystem_desc<TagStage>("a"), cs::subsystem_desc<TagStage>("b")));
        s.connect("sa.y", "f.a.x");
        s.connect("sb.y", "f.b.x");
    });
    model.setup();
    model.run();
    model.wait();
    CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(200, 422));
    CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(100, 1422));
}

TEST_CASE("fused groups: range mode own vs whole", "[fused_groups]") {
    const bool whole = GENERATE(false, true);
    CAPTURE(whole);
    cs::Model model = tfg::make([whole](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sa", 200));  // 4 groups
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sb", 100));  // 2 groups
        if (whole)
            s.add_subsystem(cs::fuse_components("f", cs::segment<cs::range_mode::whole>(cs::subsystem_desc<RangeStage>("a"),
                                                                                         cs::subsystem_desc<RangeStage>("b"))));
        else
            s.add_subsystem(cs::fuse_components(
                "f", cs::segment<cs::range_mode::own>(cs::subsystem_desc<RangeStage>("a"), cs::subsystem_desc<RangeStage>("b"))));
        s.connect("sa.y", "f.a.x");
        s.connect("sb.y", "f.b.x");
    });
    model.setup();
    model.run();
    model.wait();
    CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(200, 4.0f));              // the block is the larger count in both modes
    CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(100, whole ? 4.0f : 2.0f));  // own: its own two groups; whole: the block
}

TEST_CASE("fused groups: shared local memory carries a stage's values to the next", "[fused_groups]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("src", 128));
        s.add_subsystem(cs::fuse_components(
            "f", cs::connects_desc({{"a.y", "b.x"}}),
            cs::segment<cs::local_mem::shared>(cs::subsystem_desc<tfg::GStage<tfg::kLeave>>("a", tfg::kWg, tfg::kTile),
                                               cs::subsystem_desc<tfg::GStage<tfg::kTake>>("b", tfg::kWg, tfg::kTile))));
        s.connect("src.y", "f.a.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 2; ++fwd) {
        model.run();
        model.wait();
        std::vector<float> want(128);
        for (std::size_t g = 0; g < want.size(); ++g) want[g] = static_cast<float>(tfg::kWg - (g % tfg::kWg));  // t[63 - l] = 64 - l
        CHECK(tfg::out_of(model, "f.b.y") == want);
    }
}

TEST_CASE("fused groups: partitioned local memory gives each stage its own slice", "[fused_groups]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("src", 128));
        s.add_subsystem(cs::fuse_components(
            "f", cs::segment<cs::local_mem::partitioned>(cs::subsystem_desc<tfg::GStage<tfg::kOwn<1>>>("a", tfg::kWg, tfg::kTile),
                                                         cs::subsystem_desc<tfg::GStage<tfg::kOwn<1000>>>("b", tfg::kWg, tfg::kTile))));
        s.connect("src.y", "f.a.x");
        s.connect("src.y", "f.b.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 2; ++fwd) {
        model.run();
        model.wait();
        std::vector<float> want_a(128), want_b(128);
        for (std::size_t g = 0; g < 128; ++g) {
            const std::size_t l = g % tfg::kWg;
            want_a[g]           = static_cast<float>(1 + (tfg::kWg - 1 - l));
            want_b[g]           = static_cast<float>(1000 + (tfg::kWg - 1 - l));
        }
        CHECK(tfg::out_of(model, "f.a.y") == want_a);
        CHECK(tfg::out_of(model, "f.b.y") == want_b);
    }
}

TEST_CASE("fused groups: stages with different work-group shapes throw at forward, naming both", "[fused_groups][errors]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("src", 256));
        s.add_subsystem(cs::fuse_components("f", cs::segment(cs::subsystem_desc<RangeStage>("a", std::size_t{32}),
                                                             cs::subsystem_desc<RangeStage>("b", std::size_t{64}))));
        s.connect("src.y", "f.a.x");
        s.connect("src.y", "f.b.x");
    });
    model.setup();
    tfg::expect_throws([&] { model.run(); }, {"f.a", "f.b", "different work-group shapes"});
    model.wait();
}

TEST_CASE("fused groups: a bare component launches as a group of one stage", "[fused_groups]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("src", 256));
        s.add_subsystem(cs::subsystem_desc<TagStage>("t"));
        s.connect("src.y", "t.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 3; ++fwd) {
        model.run();
        model.wait();
        CHECK(tfg::out_of(model, "t.y") == tfg::filled(256, 401));  // segment 0, 4 groups, one segment
    }
}

TEST_CASE("fused groups: one component type in two groups", "[fused_groups]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("s1", 200));
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("s2", 100));
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("s3", 64));
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("s4", 192));
        s.add_subsystem(cs::fuse_components("f1", cs::subsystem_desc<TagStage>("a"), cs::subsystem_desc<TagStage>("b")));
        s.add_subsystem(cs::fuse_components("f2", cs::subsystem_desc<TagStage>("c"), cs::subsystem_desc<TagStage>("d")));
        s.connect("s1.y", "f1.a.x");
        s.connect("s2.y", "f1.b.x");
        s.connect("s3.y", "f2.c.x");
        s.connect("s4.y", "f2.d.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 2; ++fwd) {
        model.run();
        model.wait();
        CHECK(tfg::out_of(model, "f1.a.y") == tfg::filled(200, 422));
        CHECK(tfg::out_of(model, "f1.b.y") == tfg::filled(100, 1422));
        CHECK(tfg::out_of(model, "f2.c.y") == tfg::filled(64, 132));    // 1 group, 3 groups
        CHECK(tfg::out_of(model, "f2.d.y") == tfg::filled(192, 1132));
    }
}

TEST_CASE("fused groups: a transient group rebuilt every forward (the config cache outlives every kernel)",
          "[fused_groups][transient]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sa", 200));
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sb", 100));
        s.add_subsystem(cs::property_list{cs::property::transient{}},
                        cs::fuse_components("grp", cs::segment(cs::subsystem_desc<TagStage>("a")),
                                            cs::segment(cs::subsystem_desc<TagStage>("b"))));
        s.add_subsystem(cs::subsystem_desc<tfg::Plus1>("sink"));
        s.connect("sa.y", "grp.a.x");
        s.connect("sb.y", "grp.b.x");
        s.connect("grp.b.y", "sink.x");
    });
    model.setup();
    for (int fwd = 1; fwd <= 5; ++fwd) {
        INFO("forward " << fwd);
        model.run();
        model.wait();
        CHECK(model.get_subsystem("grp") == nullptr);  // deleted after its turn; a new group is built next forward
        CHECK(tfg::out_of(model, "sink.y") == tfg::filled(100, 1423));  // grp.b.y = 1422, plus 1
    }
}

// =============================================================================================
// P7b: boundaries, group waits, connect order
// =============================================================================================
TEST_CASE("fused groups: group_waits_enabled gives the launch its wait state and the segments keep their values", "[fused_groups][group_waits]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sa", 200));  // 4 groups
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sb", 100));  // 2 groups
        s.add_subsystem(cs::fuse_components<cs::group_waits_enabled>("f", cs::segment(cs::subsystem_desc<TagStage>("a")),
                                                                     cs::segment(cs::subsystem_desc<TagStage>("b"))));
        s.connect("sa.y", "f.a.x");
        s.connect("sb.y", "f.b.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 3; ++fwd) {
        model.run();
        model.wait();
        CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(200, 422));  // the (segment, group) indices are unchanged
        CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(100, 1422));
    }
    auto* f = dynamic_cast<cs::detail::FusedBase*>(model.get_subsystem("f"));
    REQUIRE(f != nullptr);
    CHECK(f->has_wait_state());
}

TEST_CASE("fused groups: a group with neither group_waits_enabled nor a global_barrier allocates no wait state", "[fused_groups][group_waits]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sa", 200));
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sb", 100));
        s.add_subsystem(cs::fuse_components("f", cs::segment(cs::subsystem_desc<TagStage>("a")), cs::segment(cs::subsystem_desc<TagStage>("b"))));
        s.connect("sa.y", "f.a.x");
        s.connect("sb.y", "f.b.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 2; ++fwd) {
        model.run();
        model.wait();
        CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(200, 422));
        CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(100, 1422));
    }
    auto* f = dynamic_cast<cs::detail::FusedBase*>(model.get_subsystem("f"));
    REQUIRE(f != nullptr);
    CHECK_FALSE(f->has_wait_state());
}

TEST_CASE("fused groups: global_barrier, a group reads what a different group of the earlier segment wrote", "[fused_groups][group_waits]") {
    const bool big = GENERATE(false, true);
    CAPTURE(big);
    const tfg::Shape sh = big ? tfg::big_shape() : tfg::Shape{64, 5};
    tfg::neighbor_case<0, 2>(sh.wg, sh.groups, 5);  // five forwards: the wait state is reset every time
}

TEST_CASE("fused groups: group_waits_enabled with wait_group on a neighbor group of the earlier segment", "[fused_groups][group_waits]") {
    const bool big = GENERATE(false, true);
    CAPTURE(big);
    const tfg::Shape sh = big ? tfg::big_shape() : tfg::Shape{64, 5};
    tfg::neighbor_case<1, 1>(sh.wg, sh.groups, 5);
}

TEST_CASE("fused groups: wait_group works in a group that has only a global_barrier", "[fused_groups][group_waits]") {
    const bool big = GENERATE(false, true);
    CAPTURE(big);
    const tfg::Shape sh = big ? tfg::big_shape() : tfg::Shape{64, 5};
    tfg::neighbor_case<1, 2>(sh.wg, sh.groups, 5);  // the barrier implies the property; the stage's own wait_group is on top
}

TEST_CASE("fused groups: wait_earlier_segments gives the same result as a global_barrier", "[fused_groups][group_waits]") {
    const bool big = GENERATE(false, true);
    CAPTURE(big);
    const tfg::Shape sh = big ? tfg::big_shape() : tfg::Shape{64, 5};
    tfg::neighbor_case<2, 1>(sh.wg, sh.groups, 5);
}

namespace tfg {
struct X {};  // a stand-in for a stage item in the placement traits
using GB = decltype(cs::global_barrier());
using PlainDesc = decltype(cs::subsystem_desc<Plus1>("p"));
using GroupDesc = decltype(cs::subsystem_desc<TagStage>("g"));
template <class... Ds>
using Checks = cs::detail::stage_checks<cs::detail::all_stage_descs_t<Ds...>>;
}  // namespace tfg

// Compile-time rules: the placement of boundaries and boundaries in plain groups are static_asserts of FusedGroup, so a
// wrong group does not compile; the conditions they test are the traits checked here.
static_assert(cs::detail::boundaries_placed_ok<tfg::X, tfg::GB, tfg::X>());
static_assert(cs::detail::boundaries_placed_ok<tfg::X, tfg::GB, tfg::X, tfg::GB, tfg::X>());
static_assert(!cs::detail::boundaries_placed_ok<tfg::GB, tfg::X>());
static_assert(!cs::detail::boundaries_placed_ok<tfg::X, tfg::GB>());
static_assert(!cs::detail::boundaries_placed_ok<tfg::X, tfg::GB, tfg::GB, tfg::X>());
static_assert(!cs::detail::boundaries_placed_ok<tfg::GB>());
static_assert(cs::detail::is_fuse_option_v<cs::group_waits_enabled>);
static_assert(!cs::detail::is_fuse_option_v<cs::local_mem::shared>);

TEST_CASE("fused groups: boundary placement and plain groups are rejected at compile time", "[fused_groups][errors]") {
    using namespace tfg;
    CHECK(cs::detail::boundaries_placed_ok<X, GB, X>());
    CHECK(cs::detail::boundaries_placed_ok<X, GB, X, GB, X>());
    CHECK_FALSE(cs::detail::boundaries_placed_ok<GB, X>());              // at the start
    CHECK_FALSE(cs::detail::boundaries_placed_ok<X, GB>());              // at the end
    CHECK_FALSE(cs::detail::boundaries_placed_ok<X, GB, GB, X>());       // two in a row
    CHECK_FALSE(cs::detail::boundaries_placed_ok<GB>());                 // nothing but a boundary
    // a global_barrier and the group_waits_enabled property need stages that are all work-group stages: a plain group has
    // no work-group to wait with
    CHECK(Checks<GroupDesc, GroupDesc>::all_group);
    CHECK_FALSE(Checks<PlainDesc, PlainDesc>::all_group);
    CHECK(cs::detail::has_boundary_v<GroupDesc, GB, GroupDesc>);
    CHECK_FALSE(cs::detail::has_boundary_v<GroupDesc, GroupDesc>);
    // fuse_components<group_waits_enabled>("f", segment(plain), segment(plain)) fails the static_assert of FusedGroup
    // (!Waits || Checks::all_group), as does a global_barrier() between plain segments (!has_boundary_v<Items...> ||
    // Checks::all_group); an option other than group_waits_enabled fails the static_assert of fuse_components
    // (is_fuse_option_v, checked above).
}

TEST_CASE("fused groups: a connect to the same or an earlier stage throws at setup", "[fused_groups][errors]") {
    SECTION("backward") {
        cs::Model model = tfg::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tfg::Src>("src", 128));
            s.add_subsystem(cs::fuse_components("f", cs::connects_desc({{"b.y", "a.x"}}), cs::subsystem_desc<TagStage>("a"),
                                                cs::subsystem_desc<TagStage>("b")));
            s.connect("src.y", "f.b.x");
        });
        tfg::expect_throws([&] { model.setup(); }, {"\'a.x\' reads \'b.y\', which runs after it in fused group \'f\'"});
    }
    SECTION("the same stage") {
        cs::Model model = tfg::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components("f", cs::connects_desc({{"a.y", "a.x"}}), cs::subsystem_desc<TagStage>("a")));
        });
        tfg::expect_throws([&] { model.setup(); }, {"\'a.x\' reads \'a.y\', which runs after it in fused group \'f\'"});
    }
    SECTION("backward across a segment") {
        cs::Model model = tfg::make([](cs::System& s) {
            s.add_subsystem(cs::subsystem_desc<tfg::Src>("src", 128));
            s.add_subsystem(cs::fuse_components("f", cs::connects_desc({{"b.y", "a.x"}}), cs::segment(cs::subsystem_desc<TagStage>("a")),
                                                cs::global_barrier(), cs::segment(cs::subsystem_desc<TagStage>("b"))));
            s.connect("src.y", "f.b.x");
        });
        tfg::expect_throws([&] { model.setup(); }, {"which runs after it in fused group \'f\'"});
    }
}

// =============================================================================================
// ties declared in fuse_components (cs::ties_desc)
// =============================================================================================
namespace tfg {

// TAdder: y[i] += v on an ATOMIC output (a tie_outputs member); TZero stores 0 first (outputs are never initialized).

class TAdder : public cs::Component<TAdder> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> v;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i].fetch_add(in.v[std::size_t{0}]); }

    Inputs  inputs;
    Outputs outputs;

private:
    float                            v_;

public:
    explicit TAdder(float v) : v_{v} {}
    void setup() override {
        declare_input(inputs.v, "v", 1);
        init_input("v", cs::init::fill(v_)); declare_output(outputs.y, "y", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

class TZero : public cs::Component<TZero> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs&, const Outputs& out) { out.y[i].store(0.0f); }

    Inputs  inputs;
    Outputs outputs;

    void setup() override { declare_output(outputs.y, "y", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

// TGAdder / TGZero: the same as TAdder / TZero as WORK-GROUP stages (a global_barrier needs group-body stages), work-group kWg.

class TGAdder : public cs::Component<TGAdder> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> v;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> y;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::size_t g = it.get_global_id(0);
        if (g < out.y.shape()[0]) out.y[g].fetch_add(in.v[std::size_t{0}]);
    }

    Inputs  inputs;
    Outputs outputs;

private:
    float                            v_;

public:
    explicit TGAdder(float v) : v_{v} {}
    void setup() override {
        declare_input(inputs.v, "v", 1);
        init_input("v", cs::init::fill(v_)); declare_output(outputs.y, "y", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

class TGZero : public cs::Component<TGZero> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_f32, 1> y;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs&, const Outputs& out) {
        const std::size_t g = it.get_global_id(0);
        if (g < out.y.shape()[0]) out.y[g].store(0.0f);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override { declare_output(outputs.y, "y", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// TSlice: y[i] = v for lo <= i < hi and nothing else, a PLAIN output (disjoint slices of one tie). `reshape`: its
// pre_forward calls set_shape on the output, which a tied member may not do.

class TSlice : public cs::Component<TSlice> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> lo, hi, v;  // a free input of extent 1, initialized from the constructor's value
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        const float k = static_cast<float>(i[0]);
        if (k >= in.lo[std::size_t{0}] && k < in.hi[std::size_t{0}]) out.y[i] = in.v[std::size_t{0}];
    }

    Inputs  inputs;
    Outputs outputs;

private:
    float                     lo_, hi_, v_;
    bool                      reshape_;

public:
    TSlice(float lo, float hi, float v, bool reshape = false) : lo_{lo}, hi_{hi}, v_{v}, reshape_{reshape} {}
    void setup() override {
        declare_input(inputs.lo, "lo", 1);
        init_input("lo", cs::init::fill(lo_));
        declare_input(inputs.hi, "hi", 1);
        init_input("hi", cs::init::fill(hi_));
        declare_input(inputs.v, "v", 1);
        init_input("v", cs::init::fill(v_)); declare_output(outputs.y, "y", cs::unspecified_shape<1>()); }
    void pre_forward() override {
        if (reshape_) set_shape(outputs.y, 12);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.y.shape()));
    }
};

std::vector<float> in_of(cs::Model& m, const char* path) { return to_vector(cst::synced_input<DType::f32, 1>(m, path)); }

/// The group of the atomic cases: z zeroes, a adds 1, b adds 2, all in one segment (each index runs z, a, b in that order).
inline auto atomic_stages() {
    return cs::segment<cs::range_mode::whole>(cs::subsystem_desc<TZero>("z"), cs::subsystem_desc<TAdder>("a", 1.0f),
                                              cs::subsystem_desc<TAdder>("b", 2.0f));
}
/// The group of the plain cases: a writes [0, 6) = 1, b writes [6, 12) = 2.
inline auto slice_stages() {
    return cs::segment<cs::range_mode::whole>(cs::subsystem_desc<TSlice>("a", 0.0f, 6.0f, 1.0f),
                                              cs::subsystem_desc<TSlice>("b", 6.0f, 12.0f, 2.0f));
}
inline std::vector<float> halves() {
    std::vector<float> v(12, 1.0f);
    std::fill(v.begin() + 6, v.end(), 2.0f);
    return v;
}

}  // namespace tfg

TEST_CASE("ties_desc: atomic outputs of three stages tied in the group, the views of the members and the alias agree",
          "[fused_groups][ties]") {
    // The standard initialization pattern: the zeroing stage in its own first segment, a global_barrier, then the segment of
    // the accumulating stages. All stages are work-group stages (the barrier needs them).
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components(
            "f", cs::ties_desc({cs::tie_outputs<DType::atomic_f32, 1>({"z.y", "a.y", "b.y"}, "y", cs::shape(12))}),
            cs::segment(cs::subsystem_desc<tfg::TGZero>("z")), cs::global_barrier(),
            cs::segment(cs::subsystem_desc<tfg::TGAdder>("a", 1.0f), cs::subsystem_desc<tfg::TGAdder>("b", 2.0f))));
    });
    model.setup();
    for (int fwd = 0; fwd < 3; ++fwd) {  // the zeroing stage runs every forward: the sums do not pile up
        model.run();
        model.wait();
        CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(12, 3.0f));
        CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(12, 3.0f));
        CHECK(tfg::out_of(model, "f.y") == tfg::filled(12, 3.0f));
    }
}

TEST_CASE("ties_desc: plain outputs under allow_plain, each stage writes its own slice", "[fused_groups][ties]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components(
            "f", cs::ties_desc({cs::tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "y", cs::shape(12), cs::TieOutputsPolicy::allow_plain)}),
            tfg::slice_stages()));
    });
    model.setup();
    model.run();
    model.wait();
    CHECK(tfg::out_of(model, "f.a.y") == tfg::halves());
    CHECK(tfg::out_of(model, "f.b.y") == tfg::halves());
    CHECK(tfg::out_of(model, "f.y") == tfg::halves());
}

TEST_CASE("ties_desc: connects_desc and ties_desc together, connects first", "[fused_groups][ties]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components("f", cs::connects_desc({{"y", "c.x"}}),
                                            cs::ties_desc({cs::tie_outputs<DType::atomic_f32, 1>({"z.y", "a.y", "b.y"}, "y", cs::shape(12))}),
                                            cs::segment<cs::range_mode::whole>(cs::subsystem_desc<tfg::TZero>("z"), cs::subsystem_desc<tfg::TAdder>("a", 1.0f),
                                                                              cs::subsystem_desc<tfg::TAdder>("b", 2.0f), cs::subsystem_desc<tfg::Plus1>("c"))));
    });
    model.setup();
    model.run();
    model.wait();
    CHECK(tfg::out_of(model, "f.y") == tfg::filled(12, 3.0f));
    CHECK(tfg::out_of(model, "f.c.y") == tfg::filled(12, 4.0f));  // c reads the alias inside the group (one segment: per index z, a, b, c)
}

TEST_CASE("ties_desc: a shape hook on the alias, by path", "[fused_groups][ties][shapes]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("src", 16));
        s.add_subsystem(cs::fuse_components("f", cs::ties_desc({cs::tie_outputs<DType::atomic_f32, 1>({"z.y", "a.y", "b.y"}, "y")}),
                                            tfg::atomic_stages()));
        s.shape_hook("f.y", cs::dim::from_shape("src.y", 0));
    });
    model.setup();
    model.run();
    model.wait();
    CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(16, 3.0f));
    CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(16, 3.0f));
    CHECK(tfg::out_of(model, "f.y") == tfg::filled(16, 3.0f));
}

TEST_CASE("ties_desc: an input tie in the group, both stages read the alias", "[fused_groups][ties]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components("f", cs::ties_desc({cs::tie_inputs<DType::f32, 1>({"p.x", "q.x"}, "x", cs::shape(12))}),
                                            cs::segment<cs::range_mode::whole>(cs::subsystem_desc<tfg::Plus1>("p"),
                                                                              cs::subsystem_desc<tfg::Plus1>("q"))));
    });
    model.setup();
    std::vector<float> ramp(12), want(12);
    for (std::size_t i = 0; i < 12; ++i) {
        ramp[i] = static_cast<float>(i) * 3.0f;
        want[i] = ramp[i] + 1.0f;
    }
    model.update_input("f.x", cs::init::from_host(ramp));
    model.run();
    model.wait();
    CHECK(tfg::in_of(model, "f.x") == ramp);
    CHECK(tfg::in_of(model, "f.p.x") == ramp);
    CHECK(tfg::in_of(model, "f.q.x") == ramp);
    CHECK(tfg::out_of(model, "f.p.y") == want);
    CHECK(tfg::out_of(model, "f.q.y") == want);
}

TEST_CASE("ties_desc: a consumer outside the group reads the alias in the same forward and gets the final values",
          "[fused_groups][ties]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components("f", cs::ties_desc({cs::tie_outputs<DType::atomic_f32, 1>({"z.y", "a.y", "b.y"}, "y", cs::shape(12))}),
                                            tfg::atomic_stages()));
        s.add_subsystem(cs::subsystem_desc<tfg::Plus1>("c"));
        s.connect("f.y", "c.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 3; ++fwd) {
        model.run();
        model.wait();
        CHECK(tfg::out_of(model, "c.y") == tfg::filled(12, 4.0f));
        CHECK(tfg::out_of(model, "f.y") == tfg::filled(12, 3.0f));
    }
}

TEST_CASE("ties_desc: a tie declared by the owner with paths into the group still works", "[fused_groups][ties]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components("f", tfg::atomic_stages()));
        s.tie_outputs<DType::atomic_f32, 1>({"f.z.y", "f.a.y", "f.b.y"}, "sum", cs::shape(12));
    });
    model.setup();
    model.run();
    model.wait();
    CHECK(tfg::out_of(model, "sum") == tfg::filled(12, 3.0f));
    CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(12, 3.0f));
    CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(12, 3.0f));
}

TEST_CASE("ties_desc: errors", "[fused_groups][ties][errors]") {
    SECTION("a tie of a tie: the owner ties a member the group tied") {
        cs::Model model = tfg::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components(
                "f", cs::ties_desc({cs::tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "y", cs::shape(12), cs::TieOutputsPolicy::allow_plain)}),
                cs::segment<cs::range_mode::whole>(cs::subsystem_desc<tfg::TSlice>("a", 0.0f, 6.0f, 1.0f),
                                                   cs::subsystem_desc<tfg::TSlice>("b", 6.0f, 12.0f, 2.0f),
                                                   cs::subsystem_desc<tfg::TSlice>("c", 0.0f, 12.0f, 3.0f))));
            s.tie_outputs<DType::f32, 1>({"f.a.y", "f.c.y"}, "z", cs::shape(12), cs::TieOutputsPolicy::allow_plain);
        });
        tfg::expect_throws([&] { model.setup(); }, {"tie_outputs", "already tied"});
    }
    SECTION("a tie of a tie: the owner ties the alias of the group") {
        cs::Model model = tfg::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components(
                "f", cs::ties_desc({cs::tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "y", cs::shape(12), cs::TieOutputsPolicy::allow_plain)}),
                tfg::slice_stages()));
            s.add_subsystem(cs::subsystem_desc<tfg::TSlice>("g", 0.0f, 12.0f, 3.0f));
            s.tie_outputs<DType::f32, 1>({"f.y", "g.y"}, "z", cs::shape(12), cs::TieOutputsPolicy::allow_plain);
        });
        tfg::expect_throws([&] { model.setup(); }, {"tie_outputs", "tying a tie is not allowed"});
    }
    SECTION("set_shape on a tied member") {
        cs::Model model = tfg::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components(
                "f", cs::ties_desc({cs::tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "y", cs::shape(12), cs::TieOutputsPolicy::allow_plain)}),
                cs::segment<cs::range_mode::whole>(cs::subsystem_desc<tfg::TSlice>("a", 0.0f, 6.0f, 1.0f, true),
                                                   cs::subsystem_desc<tfg::TSlice>("b", 6.0f, 12.0f, 2.0f))));
        });
        model.setup();
        tfg::expect_throws([&] { model.run(); model.wait(); }, {"is tied", "its tie"});
    }
    SECTION("a plain dtype under require_atomic") {
        cs::Model model = tfg::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components("f", cs::ties_desc({cs::tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "y", cs::shape(12))}),
                                                tfg::slice_stages()));
        });
        tfg::expect_throws([&] { model.setup(); }, {"is not atomic", "require_atomic", "allow_plain"});
    }
    SECTION("a member path that does not exist") {
        cs::Model model = tfg::make([](cs::System& s) {
            s.add_subsystem(cs::fuse_components(
                "f", cs::ties_desc({cs::tie_outputs<DType::f32, 1>({"a.y", "nope.y"}, "y", cs::shape(12), cs::TieOutputsPolicy::allow_plain)}),
                tfg::slice_stages()));
        });
        tfg::expect_throws([&] { model.setup(); }, {"nope"});
    }
}
