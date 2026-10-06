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
struct SrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.y[i] = static_cast<float>(i[0] + 1); }
};
class Src : public cs::Component<SrcBody> {
    int n_;

public:
    explicit Src(int n) : n_{n} {}
    void setup() override { declare_output(y_, "y", n_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {}, {y_}, {}); }

private:
    cs::Output<DType::f32, 1> y_;
};

// ---- Plus1 (plain): y = x + 1, the output follows the input ----------------------------------------
struct Plus1Body {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.y[i] = in.x[i] + 1.0f; }
};
class Plus1 : public cs::Component<Plus1Body> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override {
        declare_input(x_, "x", cs::unspecified_shape<1>());
        declare_output(y_, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { set_shape(y_, shape_of(x_)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(x_.shape(), {x_}, {y_}, {}); }
};

// ---- group bodies ---------------------------------------------------------------------------------
/// y[g] = segment * 1000 + groups(segment 0) * 100 + groups(segment 1) * 10 + segment count (the second term is 0 with one
/// segment): what the stage sees of its segment structure.
struct TagBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const std::size_t g = it.get_global_id(0);
        if (g >= in.x.shape()[0]) return;
        const std::size_t segs = it.get_segment_count();
        const std::size_t gc0  = it.get_segment_group_count(0);
        const std::size_t gc1  = segs > 1 ? it.get_segment_group_count(1) : 0;
        out.y[g]               = static_cast<float>(it.get_segment() * 1000 + gc0 * 100 + gc1 * 10 + segs);
    }
};
/// y[g] = the group range the stage sees (own: its own groups; whole: the segment's block).
struct RangeBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const std::size_t g = it.get_global_id(0);
        if (g < in.x.shape()[0]) out.y[g] = static_cast<float>(it.get_group_range(0));
    }
};
/// Shared local memory, stage 1: leaves t[l] = l + 1 in the stage's slice and passes x through.
struct LeaveBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        float*            t = it.local<float>(kWg);
        const std::size_t g = it.get_global_id(0);
        if (t != nullptr) t[it.get_local_linear_id()] = static_cast<float>(it.get_local_linear_id() + 1);
        if (g < in.x.shape()[0]) out.y[g] = in.x[g];
    }
};
/// Shared local memory, stage 2: reads the reversed tile after a group barrier, writing nothing itself.
struct TakeBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        float*            t = it.local<float>(kWg);
        const std::size_t g = it.get_global_id(0);
        sycl::group_barrier(it.get_group());
        if (g < in.x.shape()[0]) out.y[g] = t != nullptr ? t[kWg - 1 - it.get_local_linear_id()] : -1.0f;
    }
};
/// Partitioned local memory: write t[l] = Base + l into the stage's OWN slice, barrier, read the reversed tile.
template <int Base>
struct OwnTileBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        float*            t = it.local<float>(kWg);
        const std::size_t g = it.get_global_id(0);
        if (t != nullptr) t[it.get_local_linear_id()] = static_cast<float>(Base + it.get_local_linear_id());
        sycl::group_barrier(it.get_group());
        if (g < in.x.shape()[0]) out.y[g] = t != nullptr ? t[kWg - 1 - it.get_local_linear_id()] : -1.0f;
    }
};

/// A group stage over its input's shape, with work-group `wg` and `local_bytes` of requested local memory.
template <class Body>
class GStage : public cs::Component<Body> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
    std::size_t               wg_, local_bytes_;

public:
    explicit GStage(std::size_t wg = kWg, std::size_t local_bytes = 0) : wg_{wg}, local_bytes_{local_bytes} {}
    void setup() override {
        this->declare_input(x_, "x", cs::unspecified_shape<1>());
        this->declare_output(y_, "y", cs::unspecified_shape<1>());
    }
    void pre_forward() override { this->set_shape(y_, this->shape_of(x_)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = this->create_launch_config(x_.shape(), sycl::range<1>{wg_}, {x_}, {y_});
        if (local_bytes_ != 0) d->request_local_bytes(local_bytes_);
        return d;
    }
};

constexpr std::size_t kTile = kWg * sizeof(float);

/// Segment 1 of the cross-group cases: y[i] = x[i] + 1 (every group writes its own slice).
struct WriteBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const std::size_t i = it.get_global_id(0);
        if (i < in.x.shape()[0]) out.y[i] = in.x[i] + 1.0f;
    }
};
/// Segment 2: group g copies the slice a DIFFERENT group (g + 1) wrote. Mode 0: nothing here (a cs::global_barrier() before
/// the segment does the waiting); 1: wait_group(0, (g + 1) % G); 2: wait_earlier_segments().
template <int Mode>
struct ReadNextBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const std::size_t G = it.get_group_range(0), g = it.get_group(0), wg = it.get_local_range(0), l = it.get_local_linear_id();
        if constexpr (Mode == 1) it.wait_group(0, (g + 1) % G);  // collective: every work-item calls it
        if constexpr (Mode == 2) it.wait_earlier_segments();     // collective
        out.y[g * wg + l] = in.x[((g + 1) % G) * wg + l];
    }
};

/// y[g] = the neighbor group's value + 1, over `forwards` forwards with new inputs every time (so a stale ticket state or a
/// missed wait shows). Kind: 1 = ticket_order() before segment 2, 2 = global_barrier(). Mode: see ReadNextBody.
template <int Mode, int Kind>
void neighbor_case(std::size_t wg, std::size_t G, int forwards) {
    cs::Model model = make([wg](cs::System& s) {
        auto a = cs::subsystem_desc<GStage<WriteBody>>("a", wg);
        auto b = cs::subsystem_desc<GStage<ReadNextBody<Mode>>>("b", wg);
        if constexpr (Kind == 2)
            s.add_subsystem(cs::fuse_components("f", cs::connects_desc({{"a.y", "b.x"}}), cs::segment(a), cs::global_barrier(), cs::segment(b)));
        else
            s.add_subsystem(cs::fuse_components("f", cs::connects_desc({{"a.y", "b.x"}}), cs::segment(a), cs::ticket_order(), cs::segment(b)));
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
        model.forward();
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

using TagStage   = tfg::GStage<tfg::TagBody>;
using RangeStage = tfg::GStage<tfg::RangeBody>;

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
        model.forward();
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
    model.forward();
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
    model.forward();
    model.wait();
    CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(200, 4.0f));              // the block is the larger count in both modes
    CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(100, whole ? 4.0f : 2.0f));  // own: its own two groups; whole: the block
}

TEST_CASE("fused groups: shared local memory carries a stage's values to the next", "[fused_groups]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("src", 128));
        s.add_subsystem(cs::fuse_components(
            "f", cs::connects_desc({{"a.y", "b.x"}}),
            cs::segment<cs::local_mem::shared>(cs::subsystem_desc<tfg::GStage<tfg::LeaveBody>>("a", tfg::kWg, tfg::kTile),
                                               cs::subsystem_desc<tfg::GStage<tfg::TakeBody>>("b", tfg::kWg, tfg::kTile))));
        s.connect("src.y", "f.a.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 2; ++fwd) {
        model.forward();
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
            "f", cs::segment<cs::local_mem::partitioned>(cs::subsystem_desc<tfg::GStage<tfg::OwnTileBody<1>>>("a", tfg::kWg, tfg::kTile),
                                                         cs::subsystem_desc<tfg::GStage<tfg::OwnTileBody<1000>>>("b", tfg::kWg, tfg::kTile))));
        s.connect("src.y", "f.a.x");
        s.connect("src.y", "f.b.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 2; ++fwd) {
        model.forward();
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
    tfg::expect_throws([&] { model.forward(); }, {"f.a", "f.b", "different work-group shapes"});
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
        model.forward();
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
        model.forward();
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
        model.forward();
        model.wait();
        CHECK(model.get_subsystem("grp") == nullptr);  // deleted after its turn; a new group is built next forward
        CHECK(tfg::out_of(model, "sink.y") == tfg::filled(100, 1423));  // grp.b.y = 1422, plus 1
    }
}


// =============================================================================================
// P7b: boundaries, ticket mode, waits, connect order
// =============================================================================================
TEST_CASE("fused groups: ticket_order puts the launch in ticket mode and the segments keep their values", "[fused_groups][tickets]") {
    cs::Model model = tfg::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sa", 200));  // 4 groups
        s.add_subsystem(cs::subsystem_desc<tfg::Src>("sb", 100));  // 2 groups
        s.add_subsystem(cs::fuse_components("f", cs::segment(cs::subsystem_desc<TagStage>("a")), cs::ticket_order(),
                                            cs::segment(cs::subsystem_desc<TagStage>("b"))));
        s.connect("sa.y", "f.a.x");
        s.connect("sb.y", "f.b.x");
    });
    model.setup();
    for (int fwd = 0; fwd < 3; ++fwd) {
        model.forward();
        model.wait();
        CHECK(tfg::out_of(model, "f.a.y") == tfg::filled(200, 422));  // ticket -> (segment, group) gives the same indices
        CHECK(tfg::out_of(model, "f.b.y") == tfg::filled(100, 1422));
    }
}

TEST_CASE("fused groups: global_barrier, a group reads what a different group of the earlier segment wrote", "[fused_groups][tickets]") {
    const bool big = GENERATE(false, true);
    CAPTURE(big);
    const tfg::Shape sh = big ? tfg::big_shape() : tfg::Shape{64, 5};
    tfg::neighbor_case<0, 2>(sh.wg, sh.groups, 5);  // five forwards: the ticket state is reset every time
}

TEST_CASE("fused groups: ticket_order with wait_group on a neighbor group of the earlier segment", "[fused_groups][tickets]") {
    const bool big = GENERATE(false, true);
    CAPTURE(big);
    const tfg::Shape sh = big ? tfg::big_shape() : tfg::Shape{64, 5};
    tfg::neighbor_case<1, 1>(sh.wg, sh.groups, 5);
}

TEST_CASE("fused groups: wait_earlier_segments gives the same result as a global_barrier", "[fused_groups][tickets]") {
    const bool big = GENERATE(false, true);
    CAPTURE(big);
    const tfg::Shape sh = big ? tfg::big_shape() : tfg::Shape{64, 5};
    tfg::neighbor_case<2, 1>(sh.wg, sh.groups, 5);
}

namespace tfg {
struct X {};  // a stand-in for a stage item in the placement traits
using TO = decltype(cs::ticket_order());
using GB = decltype(cs::global_barrier());
using PlainDesc = decltype(cs::subsystem_desc<Plus1>("p"));
using GroupDesc = decltype(cs::subsystem_desc<TagStage>("g"));
template <class... Ds>
using Checks = cs::detail::stage_checks<cs::detail::all_stage_descs_t<Ds...>>;
}  // namespace tfg

// Compile-time rules: the placement of boundaries and boundaries in plain groups are static_asserts of FusedGroup, so a
// wrong group does not compile; the conditions they test are the traits checked here.
static_assert(cs::detail::boundaries_placed_ok<tfg::X, tfg::TO, tfg::X>());
static_assert(cs::detail::boundaries_placed_ok<tfg::X, tfg::GB, tfg::X, tfg::TO, tfg::X>());
static_assert(!cs::detail::boundaries_placed_ok<tfg::TO, tfg::X>());
static_assert(!cs::detail::boundaries_placed_ok<tfg::X, tfg::GB>());
static_assert(!cs::detail::boundaries_placed_ok<tfg::X, tfg::TO, tfg::GB, tfg::X>());
static_assert(!cs::detail::boundaries_placed_ok<tfg::GB>());

TEST_CASE("fused groups: boundary placement and plain groups are rejected at compile time", "[fused_groups][errors]") {
    using namespace tfg;
    CHECK(cs::detail::boundaries_placed_ok<X, TO, X>());
    CHECK(cs::detail::boundaries_placed_ok<X, GB, X, TO, X>());
    CHECK_FALSE(cs::detail::boundaries_placed_ok<TO, X>());              // at the start
    CHECK_FALSE(cs::detail::boundaries_placed_ok<X, GB>());              // at the end
    CHECK_FALSE(cs::detail::boundaries_placed_ok<X, TO, GB, X>());       // two in a row
    CHECK_FALSE(cs::detail::boundaries_placed_ok<GB>());                 // nothing but a boundary
    // a boundary needs stages that are all work-group stages: a plain group has no work-group to cooperate with
    CHECK(Checks<GroupDesc, GroupDesc>::all_group);
    CHECK_FALSE(Checks<PlainDesc, PlainDesc>::all_group);
    CHECK(cs::detail::has_boundary_v<GroupDesc, TO, GroupDesc>);
    CHECK_FALSE(cs::detail::has_boundary_v<GroupDesc, GroupDesc>);
    // (fuse_components("f", segment(plain), global_barrier(), segment(plain)) fails the static_assert of FusedGroup:
    // !has_boundary_v<Items...> || Checks::all_group)
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
