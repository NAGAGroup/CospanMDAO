// Rank-3 physical launches and the cs::range_split strategies (every case through a cospan::Model, final values only).
//
//   shapes        three segments of different group ranges, per strategy: every item holds its group's linear id
//   own / whole   a stage skips groups outside its OWN range in any dimension (own) or sees the whole segment range (whole)
//   tickets       wait_group on a rank-2 neighbour (group_waits_enabled) and a global_barrier, per strategy, several forwards
//   many groups   more than 65535 groups in rank 2: a fused group, a bare group component, a bare plain component
//   rank 3        stages of rank 3 under flatten_and_split and pad_and_split
//   native        a bare component's physical launch is its declared range; the strategies' physical group coordinates
//   group fns     one case per family of the cs:: group functions, and to_sycl() / the implicit conversions into raw SYCL
//   traits        padded refuses rank 3; at most one range_split option
//
// The components live in namespace trs.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;
using cst::to_vector;

namespace trs {

using Build = std::function<void(cs::System&)>;

class Top : public cs::System {
    Build build_;

public:
    explicit Top(Build b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }
};

cs::Model make(Build b) { return cs::create_model<Top>(default_config(), std::move(b)); }

template <int R>
std::vector<float> out_of(cs::Model& m, const char* path) {
    return to_vector(cst::synced_output<DType::f32, R>(m, path));
}

/// The work-group shape of the rank-R test stages.
template <int R>
sycl::range<R> wg_of() {
    if constexpr (R == 1)
        return sycl::range<1>{8};
    else if constexpr (R == 2)
        return sycl::range<2>{4, 4};
    else
        return sycl::range<3>{2, 2, 2};
}

/// The row-major item values of IdStage<R, Seg> over `groups`: seg * 1e6 + the group's linear id (row-major over `groups`).
template <int R>
std::vector<float> expected_ids(sycl::range<R> groups, int seg) {
    const sycl::range<R> wg = wg_of<R>();
    std::size_t          n  = 1;
    for (int d = 0; d < R; ++d) n *= groups[d] * wg[d];
    std::vector<float> v(n);
    for (std::size_t flat = 0; flat < n; ++flat) {
        std::size_t rest = flat, lin = 0, mul = 1;
        std::array<std::size_t, 3> g{};
        for (int d = R - 1; d >= 0; --d) {
            g[d] = (rest % (groups[d] * wg[d])) / wg[d];
            rest /= groups[d] * wg[d];
        }
        for (int d = 0; d < R; ++d) lin = lin * groups[d] + g[d];
        (void)mul;
        v[flat] = static_cast<float>(seg * 1000000 + static_cast<int>(lin));
    }
    return v;
}

// ---- IdStage<R, Seg>: every item writes seg * 1e6 + its group's linear id (the rank-R group id, segment-relative) ----------

template <int R, int Seg>
class IdStage : public cs::Component<IdStage<R, Seg>> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, R> y;
    };
    static void compute_impl(cs::GroupItem<R> it, const Inputs&, const Outputs& out) {
        out.y[it.get_global_id()] = static_cast<float>(Seg * 1000000 + static_cast<int>(it.get_group_linear_id()));
    }

    Inputs  inputs;
    Outputs outputs;

private:
    sycl::range<R>            groups_;

public:
    explicit IdStage(sycl::range<R> groups) : groups_{groups} {}
    void setup() override {
        const sycl::range<R> wg = wg_of<R>();
        if constexpr (R == 1)
            this->declare_output(outputs.y, "y", static_cast<int>(groups_[0] * wg[0]));
        else if constexpr (R == 2)
            this->declare_output(outputs.y, "y", static_cast<int>(groups_[0] * wg[0]), static_cast<int>(groups_[1] * wg[1]));
        else
            this->declare_output(outputs.y, "y", static_cast<int>(groups_[0] * wg[0]), static_cast<int>(groups_[1] * wg[1]),
                                 static_cast<int>(groups_[2] * wg[2]));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        sycl::range<R> r = groups_;
        for (int d = 0; d < R; ++d) r[d] *= wg_of<R>()[d];
        return this->create_launch_config(cs::range_desc(r), cs::wg_desc(wg_of<R>()));
    }
};

// ---- flags: how many stages ran for each group of a (3, 7) block ----------------------------------------------------------

template <bool Fill>
class FlagStage : public cs::Component<FlagStage<Fill>> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> f;
    };
    static void compute_impl(cs::GroupItem<2> it, const Inputs&, const Outputs& out) {
        if (it.get_local_linear_id() != 0) return;
        if constexpr (Fill)
            out.f[it.get_group_id()] = 0.0f;
        else
            out.f[it.get_group_id()] += 1.0f;
    }

    Inputs  inputs;
    Outputs outputs;

private:
    sycl::range<2>            groups_;

public:
    explicit FlagStage(sycl::range<2> groups) : groups_{groups} {}
    void setup() override { this->declare_output(outputs.f, "f", cs::unspecified_shape<2>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(sycl::range<2>{groups_[0] * 4, groups_[1] * 4}), cs::wg_desc(wg_of<2>()));
    }
};

// ---- WriteStage2 -> ReadNb<W>: group g of the second segment copies the values of group (g + 1) wrapped, of the first ------

class WriteStage2 : public cs::Component<WriteStage2> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    static void compute_impl(cs::GroupItem<2> it, const Inputs&, const Outputs& out) {
        out.y[it.get_global_id()] = static_cast<float>(it.get_group_linear_id() * 16 + it.get_local_linear_id());
    }

    Inputs  inputs;
    Outputs outputs;

private:
    sycl::range<2>            groups_;

public:
    explicit WriteStage2(sycl::range<2> groups) : groups_{groups} {}
    void setup() override { declare_output(outputs.y, "y", static_cast<int>(groups_[0] * 4), static_cast<int>(groups_[1] * 4)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<2>{groups_[0] * 4, groups_[1] * 4}), cs::wg_desc(wg_of<2>()));
    }
};
/// W: wait_group on the neighbour inside the body (needs group_waits_enabled); not W: a global_barrier before the segment.

template <bool W>
class ReadNb : public cs::Component<ReadNb<W>> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> z;
    };
    static void compute_impl(cs::GroupItem<2> it, const Inputs& in, const Outputs& out) {
        const sycl::range<2> gr = it.get_group_range();
        const sycl::id<2>    g  = it.get_group_id();
        const sycl::id<2>    ng{(g[0] + 1) % gr[0], (g[1] + 1) % gr[1]};
        if constexpr (W) it.wait_group(0, ng);  // collective
        const sycl::id<2>    l  = it.get_local_id();
        const sycl::range<2> wg = it.get_local_range();
        out.z[it.get_global_id()] = in.x[sycl::id<2>{ng[0] * wg[0] + l[0], ng[1] * wg[1] + l[1]}];
    }

    Inputs  inputs;
    Outputs outputs;

private:
    sycl::range<2>            groups_;

public:
    explicit ReadNb(sycl::range<2> groups) : groups_{groups} {}
    void setup() override {
        this->declare_input(inputs.x, "x", static_cast<int>(groups_[0] * 4), static_cast<int>(groups_[1] * 4));
        this->declare_output(outputs.z, "z", static_cast<int>(groups_[0] * 4), static_cast<int>(groups_[1] * 4));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(sycl::range<2>{groups_[0] * 4, groups_[1] * 4}), cs::wg_desc(wg_of<2>()));
    }
};

// ---- a plain rank-2 stage: y[i][j] = i * width + j ------------------------------------------------------------------------

class PlainId : public cs::Component<PlainId> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    static void compute_impl(sycl::id<2> i, const Inputs&, const Outputs& out) { out.y[i] = static_cast<float>(i[0] * out.y.shape()[1] + i[1]); }

    Inputs  inputs;
    Outputs outputs;

private:
    sycl::range<2>            r_;

public:
    explicit PlainId(sycl::range<2> r) : r_{r} {}
    void setup() override { declare_output(outputs.y, "y", static_cast<int>(r_[0]), static_cast<int>(r_[1])); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(cs::range_desc(r_)); }
};

// ---- RawStage<K>: 1 when the PHYSICAL group coordinates of the item are the ones kind K promises --------------------------------
//   K = 0 flatten_and_split: (0, 0, linear id within the segment, or the launch's: checked against the linear id of segment 0 only)
//   K = 1 pad_and_split    : (0, gy, *)
//   K = 2 padded           : (segment, gy, gx)
//   K = 3 a bare component : (0, gy, gx), the native launch

template <int K>
class RawStage : public cs::Component<RawStage<K>> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    static void compute_impl(cs::GroupItem<2> it, const Inputs&, const Outputs& out) {
        const sycl::nd_item<3> raw = it;  // the implicit conversion
        const sycl::id<2>      g   = it.get_group_id();
        bool                   ok  = false;
        if constexpr (K == 0)
            ok = raw.get_group(0) == 0 && raw.get_group(1) == 0 && (it.get_segment() != 0 || raw.get_group(2) == it.get_group_linear_id());
        else if constexpr (K == 1)
            ok = raw.get_group(0) == 0 && raw.get_group(1) == g[0];
        else if constexpr (K == 2)
            ok = raw.get_group(0) == it.get_segment() && raw.get_group(1) == g[0] && raw.get_group(2) == g[1];
        else
            ok = raw.get_group(0) == 0 && raw.get_group(1) == g[0] && raw.get_group(2) == g[1];
        out.y[it.get_global_id()] = ok ? 1.0f : 0.0f;
    }

    Inputs  inputs;
    Outputs outputs;

private:
    sycl::range<2>            groups_;

public:
    explicit RawStage(sycl::range<2> groups) : groups_{groups} {}
    void setup() override { this->declare_output(outputs.y, "y", static_cast<int>(groups_[0] * 4), static_cast<int>(groups_[1] * 4)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(sycl::range<2>{groups_[0] * 4, groups_[1] * 4}), cs::wg_desc(wg_of<2>()));
    }
};

// ---- the group functions, one group of 4 x 4 = 16 work-items ------------------------------------------------------------------
// r[0..31]: the group-wide results (written by the leader); r[32 + 6 * t + k]: item t's scan results.

class FamStage : public cs::Component<FamStage> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> r;
    };
    static void compute_impl(cs::GroupItem<2> it, const Inputs&, const Outputs& out) {
        auto              g   = it.get_group();
        float*            L   = it.local<float>(32);
        const std::size_t t   = it.get_local_linear_id();
        const float       x   = static_cast<float>(t + 1);
        const auto        pls = sycl::plus<float>();
        L[t]                  = x;
        cs::group_barrier(g);

        const float red      = cs::reduce_over_group(g, x, pls);
        const float red_init = cs::reduce_over_group(g, x, 10.0f, pls);
        const float red_max  = cs::reduce_over_group(g, x, sycl::maximum<float>());
        const float bc0      = cs::group_broadcast(g, x);
        const float bc_lin   = cs::group_broadcast(g, x, std::size_t{5});
        const float bc_id    = cs::group_broadcast(g, x, sycl::id<2>{1, 2});  // local id (1, 2): linear 6
        const bool  any_t    = cs::any_of_group(g, x == 16.0f);
        const bool  any_f    = cs::any_of_group(g, x > 100.0f);
        const bool  all_t    = cs::all_of_group(g, x > 0.0f);
        const bool  none_t   = cs::none_of_group(g, x > 16.0f);
        const float ex       = cs::exclusive_scan_over_group(g, x, pls);
        const float in       = cs::inclusive_scan_over_group(g, x, pls);
        const float ex_i     = cs::exclusive_scan_over_group(g, x, 100.0f, pls);
        const float in_i     = cs::inclusive_scan_over_group(g, x, pls, 100.0f);
        const float jred     = cs::joint_reduce(g, L, L + 16, pls);
        const float jred_i   = cs::joint_reduce(g, L, L + 16, 1.0f, pls);
        const bool  jany     = cs::joint_any_of(g, L, L + 16, [](float v) { return v == 16.0f; });
        const bool  jall     = cs::joint_all_of(g, L, L + 16, [](float v) { return v > 0.0f; });
        const bool  jnone    = cs::joint_none_of(g, L, L + 16, [](float v) { return v > 16.0f; });
        cs::joint_exclusive_scan(g, L, L + 16, L + 16, pls);
        cs::group_barrier(g);
        const float jex = L[16 + t];
        cs::group_barrier(g);
        cs::joint_inclusive_scan(g, L, L + 16, L + 16, pls);
        cs::group_barrier(g);
        const float jin = L[16 + t];

        cs::sub_group sg  = it.get_sub_group();  // an alias of sycl::sub_group
        const float   px  = cs::permute_group_by_xor(sg, x, 0u);
        const float   sel = cs::select_from_group(sg, x, sg.get_local_id());
        const float   sl  = cs::shift_group_left(sg, x, 0);
        const float   sr  = cs::shift_group_right(sg, x, 0);

        out.r[32 + t * 6 + 0] = ex;
        out.r[32 + t * 6 + 1] = in;
        out.r[32 + t * 6 + 2] = ex_i;
        out.r[32 + t * 6 + 3] = in_i;
        out.r[32 + t * 6 + 4] = jex;
        out.r[32 + t * 6 + 5] = jin;
        if (g.leader()) {
            const float v[19] = {red, red_init, bc0, bc_lin, bc_id, any_t ? 1.f : 0.f, all_t ? 1.f : 0.f, none_t ? 1.f : 0.f, any_f ? 1.f : 0.f,
                                 jred, jred_i, jany ? 1.f : 0.f, jall ? 1.f : 0.f, jnone ? 1.f : 0.f, red_max, px, sel, sl, sr};
            for (std::size_t k = 0; k < 19; ++k) out.r[k] = v[k];
        }
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override { declare_output(outputs.r, "r", 128); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = create_launch_config(cs::range_desc(sycl::range<2>{4, 4}), cs::wg_desc(wg_of<2>()));
        d->request_local_bytes(32 * sizeof(float));
        return d;
    }
};

// ---- to_sycl(): the physical objects, into raw SYCL algorithms -------------------------------------------------------------

class ToSyclStage : public cs::Component<ToSyclStage> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> r;
    };
    static void compute_impl(cs::GroupItem<2> it, const Inputs&, const Outputs& out) {
        const float x = static_cast<float>(it.get_local_linear_id() + 1);
        sycl::group<3> g3 = it.get_group();          // the implicit conversion
        sycl::group_barrier(g3);                       // raw SYCL on the physical group
        const float s1 = sycl::reduce_over_group(it.get_group().to_sycl(), x, sycl::plus<float>());
        const float s2 = sycl::reduce_over_group(it.to_sycl().get_group(), x, sycl::maximum<float>());
        if (it.get_local_linear_id() == 0) {
            out.r[std::size_t{0}] = s1;
            out.r[std::size_t{1}] = s2;
        }
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override { declare_output(outputs.r, "r", 2); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<2>{4, 4}), cs::wg_desc(wg_of<2>()));
    }
};

}  // namespace trs

using Flatten = cs::range_split::flatten_and_split;
using Pad     = cs::range_split::pad_and_split;
using Padded  = cs::range_split::padded;

// ---- shapes ---------------------------------------------------------------------------------------------------------------
/// Three segments of different group ranges; every item holds seg * 1e6 + its group's linear id.
template <class... Opts>
void shapes_case() {
    cs::Model m = trs::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components<Opts...>("f", cs::subsystem_desc<trs::IdStage<2, 0>>("a", sycl::range<2>{3, 5}),
                                                     cs::subsystem_desc<trs::IdStage<2, 1>>("b", sycl::range<2>{4, 2}),
                                                     cs::subsystem_desc<trs::IdStage<2, 2>>("c", sycl::range<2>{1, 7})));
    });
    m.setup();
    for (int fwd = 0; fwd < 2; ++fwd) {
        m.forward();
        m.wait();
        CHECK(trs::out_of<2>(m, "f.a.y") == trs::expected_ids<2>(sycl::range<2>{3, 5}, 0));
        CHECK(trs::out_of<2>(m, "f.b.y") == trs::expected_ids<2>(sycl::range<2>{4, 2}, 1));
        CHECK(trs::out_of<2>(m, "f.c.y") == trs::expected_ids<2>(sycl::range<2>{1, 7}, 2));
    }
}

TEST_CASE("range_split: no option is flatten_and_split, segments of different shapes", "[range_split]") { shapes_case<>(); }

TEMPLATE_TEST_CASE("range_split: segments of different shapes", "[range_split]", Flatten, Pad, Padded) { shapes_case<TestType>(); }

// ---- own and whole ----------------------------------------------------------------------------------------------------------
template <class Mode, class... Opts>
void own_whole_case(bool whole) {
    cs::Model m = trs::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components<Opts...>(
            "g", cs::ties_desc({cs::tie_outputs<DType::f32, 2>({"fill.f", "a.f", "b.f"}, "flags", cs::shape(12, 28), cs::TieOutputsPolicy::allow_plain)}),
            cs::segment(cs::subsystem_desc<trs::FlagStage<true>>("fill", sycl::range<2>{3, 7})), cs::global_barrier(),
            cs::segment<Mode>(cs::subsystem_desc<trs::FlagStage<false>>("a", sycl::range<2>{3, 5}),
                              cs::subsystem_desc<trs::FlagStage<false>>("b", sycl::range<2>{2, 7}))));
    });
    m.setup();
    for (int fwd = 0; fwd < 2; ++fwd) {
        m.forward();
        m.wait();
        const std::vector<float> got = trs::out_of<2>(m, "g.flags");
        // flags is shaped (12, 28) items; the group (y, x) is written by its leader at item (4y, 4x)... the stage writes at the
        // group id (an item index), so the flags of group (y, x) are the items [y][x] of the first 3 x 7
        std::vector<float> want(12 * 28, 0.0f);
        (void)want;
        for (std::size_t y = 0; y < 3; ++y)
            for (std::size_t x = 0; x < 7; ++x) {
                const float expected = whole ? 2.0f : static_cast<float>((y < 3 && x < 5 ? 1 : 0) + (y < 2 && x < 7 ? 1 : 0));
                INFO("group (" << y << ", " << x << ") forward " << fwd);
                CHECK(got[y * 28 + x] == expected);
            }
    }
}

TEMPLATE_TEST_CASE("range_split: own skips groups outside the stage's range in any dimension", "[range_split]", Flatten, Pad, Padded) {
    own_whole_case<cs::range_mode::own, TestType>(false);
}
TEMPLATE_TEST_CASE("range_split: whole runs every stage over the segment's range", "[range_split]", Flatten, Pad, Padded) {
    own_whole_case<cs::range_mode::whole, TestType>(true);
}

// ---- tickets ----------------------------------------------------------------------------------------------------------------
/// Kind 0: wait_group on the neighbour, group_waits_enabled, no barrier. Kind 1: a global_barrier, no wait_group.
template <int Kind, class Split>
void tickets_case() {
    const sycl::range<2> gr{20, 13};
    cs::Model            m = trs::make([gr](cs::System& s) {
        if constexpr (Kind == 0)
            s.add_subsystem(cs::fuse_components<Split, cs::group_waits_enabled>(
                "f", cs::connects_desc({{"w.y", "r.x"}}), cs::segment(cs::subsystem_desc<trs::WriteStage2>("w", gr)),
                cs::segment(cs::subsystem_desc<trs::ReadNb<true>>("r", gr))));
        else
            s.add_subsystem(cs::fuse_components<Split>("f", cs::connects_desc({{"w.y", "r.x"}}), cs::segment(cs::subsystem_desc<trs::WriteStage2>("w", gr)),
                                                       cs::global_barrier(), cs::segment(cs::subsystem_desc<trs::ReadNb<false>>("r", gr))));
    });
    m.setup();
    std::vector<float> want(gr[0] * 4 * gr[1] * 4);
    for (std::size_t y = 0; y < gr[0] * 4; ++y)
        for (std::size_t x = 0; x < gr[1] * 4; ++x) {
            const std::size_t ngy = (y / 4 + 1) % gr[0], ngx = (x / 4 + 1) % gr[1];
            want[y * gr[1] * 4 + x] = static_cast<float>((ngy * gr[1] + ngx) * 16 + (y % 4) * 4 + (x % 4));
        }
    for (int fwd = 0; fwd < 4; ++fwd) {
        m.forward();
        m.wait();
        INFO("forward " << fwd);
        CHECK(trs::out_of<2>(m, "f.r.z") == want);
    }
}

TEMPLATE_TEST_CASE("range_split: tickets with wait_group on a rank-2 neighbour", "[range_split][tickets]", Flatten, Pad, Padded) {
    tickets_case<0, TestType>();
}
TEMPLATE_TEST_CASE("range_split: tickets with a global_barrier", "[range_split][tickets]", Flatten, Pad, Padded) {
    tickets_case<1, TestType>();
}

// ---- more than 65535 groups ----------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE("range_split: 260 x 260 = 67600 groups per stage in a fused group", "[range_split][many]", Flatten, Pad, Padded) {
    cs::Model m = trs::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components<TestType>("f", cs::subsystem_desc<trs::IdStage<2, 0>>("a", sycl::range<2>{260, 260}),
                                                      cs::subsystem_desc<trs::IdStage<2, 1>>("b", sycl::range<2>{260, 260})));
    });
    m.setup();
    m.forward();
    m.wait();
    CHECK(trs::out_of<2>(m, "f.a.y") == trs::expected_ids<2>(sycl::range<2>{260, 260}, 0));
    CHECK(trs::out_of<2>(m, "f.b.y") == trs::expected_ids<2>(sycl::range<2>{260, 260}, 1));
}

TEST_CASE("range_split: 67600 groups in a bare rank-2 group component and a bare plain component", "[range_split][many]") {
    cs::Model m = trs::make([](cs::System& s) {
        s.add_subsystem(cs::subsystem_desc<trs::IdStage<2, 0>>("a", sycl::range<2>{260, 260}));
        s.add_subsystem(cs::subsystem_desc<trs::PlainId>("p", sycl::range<2>{2080, 2080}));  // default 8 x 8: 260 x 260 groups
    });
    m.setup();
    m.forward();
    m.wait();
    CHECK(trs::out_of<2>(m, "a.y") == trs::expected_ids<2>(sycl::range<2>{260, 260}, 0));
    const std::vector<float> p = trs::out_of<2>(m, "p.y");
    bool                     ok = p.size() == 2080u * 2080u;
    for (std::size_t k = 0; ok && k < p.size(); ++k) ok = p[k] == static_cast<float>(k);
    CHECK(ok);
}

// ---- rank 3 -------------------------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE("range_split: rank-3 stages", "[range_split][rank3]", Flatten, Pad) {
    cs::Model m = trs::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components<TestType>("f", cs::subsystem_desc<trs::IdStage<3, 0>>("a", sycl::range<3>{2, 3, 4}),
                                                      cs::subsystem_desc<trs::IdStage<3, 1>>("b", sycl::range<3>{3, 1, 5})));
        s.add_subsystem(cs::subsystem_desc<trs::IdStage<3, 2>>("c", sycl::range<3>{2, 2, 3}));  // a bare rank-3 component
    });
    m.setup();
    m.forward();
    m.wait();
    CHECK(trs::out_of<3>(m, "f.a.y") == trs::expected_ids<3>(sycl::range<3>{2, 3, 4}, 0));
    CHECK(trs::out_of<3>(m, "f.b.y") == trs::expected_ids<3>(sycl::range<3>{3, 1, 5}, 1));
    CHECK(trs::out_of<3>(m, "c.y") == trs::expected_ids<3>(sycl::range<3>{2, 2, 3}, 2));
}

TEST_CASE("range_split: rank-1 stages", "[range_split][rank1]") {
    cs::Model m = trs::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components<Pad>("f", cs::subsystem_desc<trs::IdStage<1, 0>>("a", sycl::range<1>{9}),
                                                 cs::subsystem_desc<trs::IdStage<1, 1>>("b", sycl::range<1>{4})));
    });
    m.setup();
    m.forward();
    m.wait();
    CHECK(trs::out_of<1>(m, "f.a.y") == trs::expected_ids<1>(sycl::range<1>{9}, 0));
    CHECK(trs::out_of<1>(m, "f.b.y") == trs::expected_ids<1>(sycl::range<1>{4}, 1));
}

// ---- the physical coordinates ---------------------------------------------------------------------------------------------------
TEST_CASE("range_split: a bare component's physical launch is its declared range", "[range_split][native]") {
    cs::Model m = trs::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<trs::RawStage<3>>("a", sycl::range<2>{6, 9})); });
    m.setup();
    m.forward();
    m.wait();
    CHECK(trs::out_of<2>(m, "a.y") == std::vector<float>(24u * 36u, 1.0f));
}

TEST_CASE("range_split: flatten_and_split puts the groups along the last dimension", "[range_split][native]") {
    cs::Model m = trs::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components<Flatten>("f", cs::subsystem_desc<trs::RawStage<0>>("a", sycl::range<2>{6, 9}),
                                                     cs::subsystem_desc<trs::RawStage<0>>("b", sycl::range<2>{2, 3})));
    });
    m.setup();
    m.forward();
    m.wait();
    CHECK(trs::out_of<2>(m, "f.a.y") == std::vector<float>(24u * 36u, 1.0f));
    CHECK(trs::out_of<2>(m, "f.b.y") == std::vector<float>(8u * 12u, 1.0f));
}

TEST_CASE("range_split: pad_and_split keeps the leading dimensions hardware-direct", "[range_split][native]") {
    cs::Model m = trs::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components<Pad>("f", cs::subsystem_desc<trs::RawStage<1>>("a", sycl::range<2>{6, 9}),
                                                 cs::subsystem_desc<trs::RawStage<1>>("b", sycl::range<2>{2, 3})));
    });
    m.setup();
    m.forward();
    m.wait();
    CHECK(trs::out_of<2>(m, "f.a.y") == std::vector<float>(24u * 36u, 1.0f));
    CHECK(trs::out_of<2>(m, "f.b.y") == std::vector<float>(8u * 12u, 1.0f));
}

TEST_CASE("range_split: padded makes physical dimension 0 the segment index", "[range_split][native]") {
    cs::Model m = trs::make([](cs::System& s) {
        s.add_subsystem(cs::fuse_components<Padded>("f", cs::subsystem_desc<trs::RawStage<2>>("a", sycl::range<2>{6, 9}),
                                                    cs::subsystem_desc<trs::RawStage<2>>("b", sycl::range<2>{2, 3})));
    });
    m.setup();
    m.forward();
    m.wait();
    CHECK(trs::out_of<2>(m, "f.a.y") == std::vector<float>(24u * 36u, 1.0f));
    CHECK(trs::out_of<2>(m, "f.b.y") == std::vector<float>(8u * 12u, 1.0f));
}

// ---- the group functions ------------------------------------------------------------------------------------------------------------
TEST_CASE("cs:: group functions: every family over the CoSpan group wrapper", "[range_split][group_functions]") {
    static_assert(std::is_same_v<cs::sub_group, sycl::sub_group>);
    cs::Model m = trs::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<trs::FamStage>("a")); });
    m.setup();
    m.forward();
    m.wait();
    const std::vector<float> r = trs::out_of<1>(m, "a.r");
    REQUIRE(r.size() == 128u);
    // the group-wide results
    CHECK(r[0] == 136.0f);   // reduce_over_group
    CHECK(r[1] == 146.0f);   // with init 10
    CHECK(r[14] == 16.0f);   // maximum
    CHECK(r[2] == 1.0f);     // group_broadcast from the leader
    CHECK(r[3] == 6.0f);     // by local linear id 5
    CHECK(r[4] == 7.0f);     // by the rank-2 local id (1, 2)
    CHECK(r[5] == 1.0f);     // any_of_group
    CHECK(r[6] == 1.0f);     // all_of_group
    CHECK(r[7] == 1.0f);     // none_of_group
    CHECK(r[8] == 0.0f);     // any_of_group, false
    CHECK(r[9] == 136.0f);   // joint_reduce
    CHECK(r[10] == 137.0f);  // with init 1
    CHECK(r[11] == 1.0f);    // joint_any_of
    CHECK(r[12] == 1.0f);    // joint_all_of
    CHECK(r[13] == 1.0f);    // joint_none_of
    CHECK(r[15] == 1.0f);    // permute_group_by_xor, mask 0
    CHECK(r[16] == 1.0f);    // select_from_group, own id
    CHECK(r[17] == 1.0f);    // shift_group_left, delta 0
    CHECK(r[18] == 1.0f);    // shift_group_right, delta 0
    // the scans, per work-item t (x = t + 1)
    for (std::size_t t = 0; t < 16; ++t) {
        const float ex = static_cast<float>(t * (t + 1) / 2), in = static_cast<float>((t + 1) * (t + 2) / 2);
        INFO("work-item " << t);
        CHECK(r[32 + t * 6 + 0] == ex);
        CHECK(r[32 + t * 6 + 1] == in);
        CHECK(r[32 + t * 6 + 2] == ex + 100.0f);
        CHECK(r[32 + t * 6 + 3] == in + 100.0f);
        CHECK(r[32 + t * 6 + 4] == ex);
        CHECK(r[32 + t * 6 + 5] == in);
    }
}

TEST_CASE("to_sycl(): the physical item and group go into raw SYCL algorithms", "[range_split][group_functions]") {
    cs::Model m = trs::make([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<trs::ToSyclStage>("a")); });
    m.setup();
    m.forward();
    m.wait();
    const std::vector<float> r = trs::out_of<1>(m, "a.r");
    CHECK(r[0] == 136.0f);
    CHECK(r[1] == 16.0f);
}

// ---- traits --------------------------------------------------------------------------------------------------------------------------
TEST_CASE("range_split: padded refuses rank 3, at most one option", "[range_split][traits]") {
    using cs::detail::Split;
    STATIC_CHECK(cs::detail::split_supports_rank_v<Split::padded, 1>);
    STATIC_CHECK(cs::detail::split_supports_rank_v<Split::padded, 2>);
    STATIC_CHECK(!cs::detail::split_supports_rank_v<Split::padded, 3>);
    STATIC_CHECK(cs::detail::split_supports_rank_v<Split::flatten, 3>);
    STATIC_CHECK(cs::detail::split_supports_rank_v<Split::pad, 3>);
    STATIC_CHECK(cs::detail::split_options_ok_v<>);
    STATIC_CHECK(cs::detail::split_options_ok_v<Flatten, cs::group_waits_enabled>);
    STATIC_CHECK(!cs::detail::split_options_ok_v<Flatten, Pad>);
    STATIC_CHECK(cs::detail::split_of<>() == Split::flatten);
    STATIC_CHECK(cs::detail::split_of<Pad>() == Split::pad);
    STATIC_CHECK(cs::detail::split_of<cs::group_waits_enabled, Padded>() == Split::padded);
}
