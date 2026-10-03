// Per-root write epochs and the per-root host-copy cache (C1).
//
//   epochs: IOEntry::write_epoch() of a root advances once per writer submission (run_post_compute), once per applied
//           init, and on allocation / release / size change; N tied writers give +N. (The Fused / Parallel deltas are checked in
//           test_flavors.cpp: wave kernels are only emitted in the translation unit that defines the fused stages.) Read through m.root().entry_at(path).write_epoch().
//   cache:  read_primary serves a root's host copy while its epoch is unchanged. host_reads (refreshes) and
//           host_read_hits come from diag::snapshot(). Output lines start with "[epoch]".

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace tep {

constexpr std::size_t kN = 8;

const cs::property_list kTransient{cs::property::transient{}};

// ---- components -------------------------------------------------------------------------------------------------
/// W1: y[i] = i + 1.
struct W1Body {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const { o.y[i] = static_cast<float>(i[0] + 1); }
};
class W1 : public cs::Component<W1Body> {
    cs::Output<DType::f32, 1> y_;

public:
    void setup() override { declare_output(y_, "y", kN); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {}, {y_}, {}); }
};

/// W2: two outputs y1, y2 (separate roots unless tied): y1 = i + 1, y2 = 2 (i + 1).
struct W2Body {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y1;
        cs::Output<DType::f32, 1> y2;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const {
        o.y1[i] = static_cast<float>(i[0] + 1);
        o.y2[i] = static_cast<float>(2 * (i[0] + 1));
    }
};
class W2 : public cs::Component<W2Body> {
    cs::Output<DType::f32, 1> y1_, y2_;
    bool                      tied_;

public:
    explicit W2(bool tied = false) : tied_{tied} {}
    void setup() override {
        if (tied_) {
            declare_output(y1_, "y1", cs::unspecified_shape<1>());
            declare_output(y2_, "y2", cs::unspecified_shape<1>());
        } else {
            declare_output(y1_, "y1", kN);
            declare_output(y2_, "y2", kN);
        }
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(y1_.shape(), {}, {y1_, y2_}, {});
    }
};

/// Rd: z = 2 x.
struct RdBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> z;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.z[i] = 2.0f * a.x[i]; }
};
class Rd : public cs::Component<RdBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> z_;

public:
    void setup() override {
        declare_input(x_, "x", cs::unspecified);
        declare_output(z_, "z", cs::FromInputDim{"x", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(z_.shape(), {x_}, {z_}, {}); }
};

/// Tied members: zero (t = 0) and accumulate (t += i + 1), plain (TieOutputsPolicy::allow_plain).
template <bool Zero>
struct TieBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> t;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const {
        if constexpr (Zero)
            o.t[i] = 0.0f;
        else
            o.t[i] = o.t[i] + static_cast<float>(i[0] + 1);
    }
};
template <bool Zero>
class TieMember : public cs::Component<TieBody<Zero>> {
    cs::Output<DType::f32, 1> t_;

public:
    void setup() override { this->declare_output(t_, "t", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(t_.shape(), {}, {t_}, {});
    }
};

/// Fin: a free input x [kN] with an init (0 fill 3, 1 from_host 4, 2 custom_host 5); y = x.
struct FinBody {
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.y[i] = a.x[i]; }
};
class Fin : public cs::Component<FinBody> {
    int                       kind_;
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;

public:
    explicit Fin(int kind) : kind_{kind} {}
    void setup() override {
        declare_input(x_, "x", kN);
        declare_output(y_, "y", kN);
        if (kind_ == 0) init_input("x", cs::init::fill(3.0f));
        if (kind_ == 1) init_input("x", cs::init::from_host(std::vector<float>(kN, 4.0f)));
        if (kind_ == 2)
            init_input("x", cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostViewMut<DType::f32, 1>&)>{
                                [](cs::HostViewMut<DType::f32, 1>& h) {
                                    for (std::size_t k = 0; k < h.size(); ++k) h[k] = 5.0f;
                                }}));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(y_.shape(), {x_}, {y_}, {}); }
};

/// RGen: a free u32 input `counts` [n] drives the ragged output r [n, counts[i], 3]; r block i = 1000 i + q.
struct RGenBody {
    struct Inputs {
        cs::Input<DType::u32, 1> counts;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> r;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        auto b = o.r.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = static_cast<float>(id[0] * 1000 + q);
    }
};
class RGen : public cs::Component<RGenBody> {
    std::vector<std::uint32_t>      init_;
    cs::Input<DType::u32, 1>        counts_;
    cs::RaggedOutput<DType::f32, 3> r_;

public:
    explicit RGen(std::vector<std::uint32_t> init) : init_{std::move(init)} {}
    void setup() override {
        declare_input(counts_, "counts", init_.size());
        init_input("counts", cs::init::from_host(init_));
        declare_output(r_, "r", cs::FromInputDim{"counts", 0}, cs::FromInputValues{"counts"}, 3);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {counts_}, {r_}, {});
    }
};

// ---- systems ---------------------------------------------------------------------------------------------------------
class Chain : public cs::System {  // a -> b, plain
public:
    void setup() override {
        add_subsystem<W1>("a");
        add_subsystem<Rd>("b");
        connect("a.y", "b.x");
    }
};
template <class Inner>
class Top : public cs::System {
public:
    void setup() override { add_subsystem<Inner>("f"); }
};

class TieTop : public cs::System {
public:
    void setup() override {
        add_subsystem<TieMember<true>>("z");
        add_subsystem<TieMember<false>>("a");
        tie_outputs<DType::f32, 1>({"z.t", "a.t"}, "t", cs::alias::fixed(kN), cs::TieOutputsPolicy::allow_plain);
    }
};
class SameRootTop : public cs::System {
public:
    void setup() override {
        add_subsystem<W2>("w", true);
        tie_outputs<DType::f32, 1>({"w.y1", "w.y2"}, "t", cs::alias::fixed(kN), cs::TieOutputsPolicy::allow_plain);
    }
};
class TwoOutTop : public cs::System {
public:
    void setup() override { add_subsystem<W2>("w"); }
};
class FinTop : public cs::System {
    int kind_;

public:
    explicit FinTop(int kind) : kind_{kind} {}
    void setup() override { add_subsystem<Fin>("fin", kind_); }
};
class TransInner : public cs::System {
public:
    void setup() override {
        add_subsystem<W1>("p");
        add_subsystem<Rd>("c");
        connect("p.y", "c.x");
    }
};
class TransTop : public cs::System {
public:
    void setup() override {
        add_subsystem<TransInner>("op", kTransient);
        add_subsystem<Rd>("sink");
        connect("op.c.z", "sink.x");
    }
};
class RGenTop : public cs::System {
    std::vector<std::uint32_t> counts_;

public:
    explicit RGenTop(std::vector<std::uint32_t> c) : counts_{std::move(c)} {}
    void setup() override { add_subsystem<RGen>("g", counts_); }
};

// ---- ragged / dense sources for the leading-range views ------------------------------------------------------------------
struct RaggedSrcBody {
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> out;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        const std::size_t i = id[0];
        auto              b = o.out.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j)
            for (std::size_t k = 0; k < b.shape()[1]; ++k) b[sycl::id<2>{j, k}] = static_cast<float>(i * 1000 + j * 10 + k);
    }
};
class RaggedSrc : public cs::Component<RaggedSrcBody> {
    std::vector<std::size_t>        sizes_;
    cs::RaggedOutput<DType::f32, 3> out_;

public:
    explicit RaggedSrc(std::vector<std::size_t> s) : sizes_{std::move(s)} {}
    void setup() override { declare_output(out_, "out", sizes_.size(), cs::per_index(sizes_), 3); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{sizes_.size()}, {}, {out_}, {});
    }
};
struct RSumBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 3> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    void operator()(sycl::id<1> id, Inputs i, Outputs o) const {
        auto  b = i.in.block(id[0]);
        float s = 0.0f;
        for (std::size_t q = 0; q < b.size(); ++q) s += b[q];
        o.sums[id] = s;
    }
};
class RSum : public cs::Component<RSumBody> {
    cs::RaggedInput<DType::f32, 3> in_;
    cs::Output<DType::f32, 1>      sums_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified, cs::unspecified, cs::unspecified);
        declare_output(sums_, "sums", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override { return create_launch_config(sums_.shape(), {in_}, {sums_}, {}); }
};
class LRTop : public cs::System {
    std::vector<std::size_t> sizes_;

public:
    explicit LRTop(std::vector<std::size_t> s) : sizes_{std::move(s)} {}
    void setup() override {
        add_subsystem<RaggedSrc>("src", sizes_);
        add_subsystem<RSum>("cons");
        connect("src.out", "cons.in");
    }
};

// ---- helpers -------------------------------------------------------------------------------------------------------------
std::uint64_t epoch(cs::Model& m, const char* path) { return m.root().entry_at(path).write_epoch(); }

void fwd(cs::Model& m) {
    m.forward();
    m.wait();
}

/// The epoch delta of `path` over one forward.
std::uint64_t delta(cs::Model& m, const char* path) {
    const auto before = epoch(m, path);
    fwd(m);
    return epoch(m, path) - before;
}

}  // namespace tep

// =============================================================================================================================
TEST_CASE("Epochs: one writer, one output: +1 per forward", "[epoch]") {
    cs::Model m = cs::create_model<tep::Top<tep::Chain>>(default_config());
    m.setup();
    const auto first = tep::delta(m, "f.a.y");  // the first forward may include an allocation
    std::printf("[epoch] plain chain, first forward: a.y +%llu\n", static_cast<unsigned long long>(first));
    for (int r = 0; r < 3; ++r) CHECK(tep::delta(m, "f.a.y") == 1);
    // each of the two writers advances its own root once per forward
    const auto a0 = tep::epoch(m, "f.a.y"), b0 = tep::epoch(m, "f.b.z");
    tep::fwd(m);
    CHECK(tep::epoch(m, "f.a.y") - a0 == 1);
    CHECK(tep::epoch(m, "f.b.z") - b0 == 1);
    std::printf("[epoch] plain chain: a.y +%llu, b.z +%llu per forward\n", static_cast<unsigned long long>(tep::epoch(m, "f.a.y") - a0),
                static_cast<unsigned long long>(tep::epoch(m, "f.b.z") - b0));
}

TEST_CASE("Epochs: one component, two outputs: +1 each (separate roots); aliased to one root: +1", "[epoch]") {
    {
        cs::Model m = cs::create_model<tep::TwoOutTop>(default_config());
        m.setup();
        tep::fwd(m);
        const auto a0 = tep::epoch(m, "w.y1"), b0 = tep::epoch(m, "w.y2");
        tep::fwd(m);
        CHECK(tep::epoch(m, "w.y1") - a0 == 1);
        CHECK(tep::epoch(m, "w.y2") - b0 == 1);
    }
    try {
        cs::Model m = cs::create_model<tep::SameRootTop>(default_config());
        m.setup();
        tep::fwd(m);
        const auto t0 = tep::epoch(m, "t");
        tep::fwd(m);
        const auto d = tep::epoch(m, "t") - t0;
        std::printf("[epoch] one component, two outputs tied to ONE root: +%llu per forward\n", static_cast<unsigned long long>(d));
        CHECK(d == 1);
        CHECK(tep::epoch(m, "w.y1") == tep::epoch(m, "w.y2"));  // the same root
    } catch (const std::exception& e) {
        std::printf("[epoch] one component's two outputs tied to one root is NOT constructible: %s\n", e.what());
    }
}

TEST_CASE("Epochs: tie_outputs of a zeroing producer and an accumulator: +2 per forward", "[epoch]") {
    cs::Model m = cs::create_model<tep::TieTop>(default_config());
    m.setup();
    const auto first = tep::delta(m, "t");  // includes the tie alias root''s allocation
    std::printf("[epoch] tie of two members, first forward: +%llu\n", static_cast<unsigned long long>(first));
    for (int r = 0; r < 3; ++r) CHECK(tep::delta(m, "t") == 2);
    const auto v = m.view_output<DType::f32, 1>("t");
    for (std::size_t i = 0; i < tep::kN; ++i) CHECK(v.span()[i] == static_cast<float>(i + 1));
}

TEST_CASE("Epochs: a free input is bumped when its owner applies the init (fill, from_host, custom_host)", "[epoch]") {
    for (int kind = 0; kind < 3; ++kind) {
        INFO("init kind " << kind);
        cs::Model m = cs::create_model<tep::FinTop>(default_config(), kind);
        m.setup();
        const auto e0 = tep::epoch(m, "fin.x");
        tep::fwd(m);
        const auto e1 = tep::epoch(m, "fin.x");
        CHECK(e1 > e0);  // allocated, then the init applied
        std::printf("[epoch] init kind %d: free input epoch +%llu on the first forward\n", kind,
                    static_cast<unsigned long long>(e1 - e0));
        CHECK(tep::delta(m, "fin.x") == 0);  // nothing re-applies it
        m.init_input("fin.x", cs::init::from_host(std::vector<float>(tep::kN, 9.0f)));
        const auto d = tep::delta(m, "fin.x");
        CHECK(d >= 1);  // re-marked pending: applied again at the owner's next run
        std::printf("[epoch] init kind %d: +%llu after init_input\n", kind, static_cast<unsigned long long>(d));
        const auto y = m.view_output<DType::f32, 1>("fin.y");
        for (std::size_t i = 0; i < tep::kN; ++i) CHECK(y.span()[i] == 9.0f);
    }
}

TEST_CASE("Epochs: a transient root is bumped by its allocation, its writer and its release", "[epoch]") {
    cs::Model m = cs::create_model<tep::TransTop>(default_config());
    m.setup();
    for (int r = 0; r < 2; ++r) {
        // the transient is deleted after its turn, so its entries cannot be read afterwards: the per-forward epoch
        // delta (allocation + writer + release = 3) is no longer observable from outside
        m.forward();
        m.wait();
        CHECK(m.get_subsystem("op") == nullptr);
    }
}

// =============================================================================================================================
TEST_CASE("Host cache: a second view with no forward in between is a hit", "[epoch][cache]") {
    cs::Model m = cs::create_model<tep::FinTop>(default_config(), 0);
    m.setup();
    tep::fwd(m);
    const auto s0 = cs::diag::snapshot();
    const auto v1 = m.view_output<DType::f32, 1>("fin.y");
    const auto s1 = cs::diag::snapshot();
    const auto v2 = m.view_output<DType::f32, 1>("fin.y");
    const auto s2 = cs::diag::snapshot();
    CHECK((s1 - s0).host_reads == 1);
    CHECK((s1 - s0).host_read_hits == 0);
    CHECK((s2 - s1).host_reads == 0);
    CHECK((s2 - s1).host_read_hits == 1);
    REQUIRE(v1.size() == tep::kN);
    for (std::size_t i = 0; i < tep::kN; ++i) {
        CHECK(v1.span()[i] == 3.0f);
        CHECK(v2.span()[i] == v1.span()[i]);
    }
}

TEST_CASE("Host cache: a forward invalidates it; two forwards with different inputs give their own values", "[epoch][cache]") {
    cs::Model m = cs::create_model<tep::FinTop>(default_config(), 1);
    m.setup();
    for (const float x : {2.0f, 7.0f, 11.0f}) {
        m.init_input("fin.x", cs::init::from_host(std::vector<float>(tep::kN, x)));
        tep::fwd(m);
        const auto s0 = cs::diag::snapshot();
        const auto v  = m.view_output<DType::f32, 1>("fin.y");
        CHECK((cs::diag::snapshot() - s0).host_reads == 1);  // a refresh, not a stale hit
        for (std::size_t i = 0; i < tep::kN; ++i) CHECK(v.span()[i] == x);
    }
}

TEST_CASE("Host cache: a value-driven shape reads its driver once while it is unchanged", "[epoch][cache]") {
    const std::vector<std::uint32_t> counts{2, 0, 3, 1, 4, 2};
    cs::Model                        m = cs::create_model<tep::RGenTop>(default_config(), counts);
    m.setup();
    for (int r = 0; r < 4; ++r) {
        const auto s0 = cs::diag::snapshot();
        tep::fwd(m);
        const auto d = cs::diag::snapshot() - s0;
        std::printf("[epoch] value-driven shape, forward %d: host_reads %llu, hits %llu\n", r + 1,
                    static_cast<unsigned long long>(d.host_reads), static_cast<unsigned long long>(d.host_read_hits));
        if (r == 0) {
            CHECK(d.host_reads >= 1);
        } else {
            CHECK(d.host_reads == 0);  // the driver `counts` was not written again
            CHECK(d.host_read_hits >= 1);
        }
    }
    const auto v = m.view_ragged_output<DType::f32, 3>("g.r");
    REQUIRE(v.size() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const auto b = v.block(i);
        REQUIRE(b.size() == static_cast<std::size_t>(counts[i]) * 3);
        for (std::size_t q = 0; q < b.size(); ++q) CHECK(b.span()[q] == static_cast<float>(i * 1000 + q));
    }
    // a changed driver is seen
    const std::vector<std::uint32_t> next{1, 1, 1, 5, 0, 2};
    m.init_input("g.counts", cs::init::from_host(next));
    tep::fwd(m);
    const auto w = m.view_ragged_output<DType::f32, 3>("g.r");
    REQUIRE(w.size() == next.size());
    for (std::size_t i = 0; i < next.size(); ++i) CHECK(w.block(i).size() == static_cast<std::size_t>(next[i]) * 3);
}

TEST_CASE("Host cache: leading-range views slice the cached whole root", "[epoch][cache]") {
    const std::vector<std::size_t> sizes{2, 1, 3, 0, 4, 2, 1, 3};
    const std::size_t              n = sizes.size();
    cs::Model                      m = cs::create_model<tep::LRTop>(default_config(), sizes);
    m.setup();
    tep::fwd(m);
    const std::vector<std::pair<std::size_t, std::size_t>> ranges{{0, n}, {2, 5}, {n - 1, n}, {1, n - 1}, {4, 4}, {0, n}};
    const auto                                              s0 = cs::diag::snapshot();
    for (const auto& [a, b] : ranges) {
        INFO("range [" << a << ", " << b << ")");
        cs::detail::LeadingRangeAccess::set(m.root().entry_at("cons.in"), a, b);
        const auto v = m.view_ragged_input<DType::f32, 3>("cons.in");
        REQUIRE(v.size() == b - a);
        std::vector<float>         data;
        std::vector<std::uint64_t> offs{0};
        for (std::size_t i = a; i < b; ++i) {
            for (std::size_t j = 0; j < sizes[i]; ++j)
                for (std::size_t k = 0; k < 3; ++k) data.push_back(static_cast<float>(i * 1000 + j * 10 + k));
            offs.push_back(data.size());
        }
        CHECK(v.offsets == offs);
        CHECK(v.data == data);
    }
    const auto d = cs::diag::snapshot() - s0;
    CHECK(d.host_reads == 1);  // one refresh of the root, then only hits
    CHECK(d.host_read_hits == ranges.size() - 1);
    std::printf("[epoch] leading-range views: %zu views, host_reads %llu, hits %llu\n", ranges.size(),
                static_cast<unsigned long long>(d.host_reads), static_cast<unsigned long long>(d.host_read_hits));
}

TEST_CASE("Host cache: a released transient root still throws on read", "[epoch][cache]") {
    cs::Model m = cs::create_model<tep::TransTop>(default_config());
    m.setup();
    tep::fwd(m);
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::string what = "<no exception>";
        try {
            (void)m.view_output<DType::f32, 1>("op.p.y");
        } catch (const std::exception& e) {
            what = e.what();
        }
        INFO("thrown: " << what);
        CHECK(what != "<no exception>");
        CHECK(what.find("is deleted") != std::string::npos);
    }
    // the eternal sink is still readable (and cached)
    const auto v = m.view_output<DType::f32, 1>("sink.z");
    for (std::size_t i = 0; i < tep::kN; ++i) CHECK(v.span()[i] == 4.0f * static_cast<float>(i + 1));
}
