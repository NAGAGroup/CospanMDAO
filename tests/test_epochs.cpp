// What survived of the write-epoch and host-cache tests.
//
// The per-root write epochs, the host read cache and the counters that measured them (host_reads, host_read_hits)
// no longer exist (internal/design-review-2026-10-03.md, sections 3 and 5): a read of a result is a host VIEW, obtained after
// a forward and sync()ed before it is read. What is left here is what those cases also proved about VALUES:
//
//   ties     a zeroing producer and an accumulator on one tie_outputs root give the same answer every forward
//   inputs   a free input's initializer (fill, from_host, custom_host) and update_input between runs
//   views    two views of one output; a view per run with a new input each run; a value-driven ragged output
//   deleted  a transient subsystem is gone after its turn, and its variables cannot be viewed by path
//
// The Fused / Parallel deltas that used to be checked here went with ParallelSystem.

#include <cstddef>
#include <cstdint>
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
using cst::forward_and_wait;
using cst::synced_output;

namespace tep {

constexpr std::size_t kN = 8;

const cs::property_list kTransient{cs::property::transient{}};

// ---- components -------------------------------------------------------------------------------------------------
/// W1: y[i] = i + 1.

class W1 : public cs::Component<W1> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& o) { o.y[i] = static_cast<float>(i[0] + 1); }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { declare_output(outputs.y, "y", kN); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

/// Rd: z = 2 x.

class Rd : public cs::Component<Rd> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> z;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& a, const Outputs& o) { o.z[i] = 2.0f * a.x[i]; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", cs::unspecified);
        declare_output(outputs.z, "z", cs::unspecified_shape<1>());
    }
    void prepare_forward() override { set_shape(outputs.z, shape_of(inputs.x)[0]); }  // the output follows the input (section 10: no derived dims)
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.z.shape())); }
};

/// Tied members: zero (t = 0) and accumulate (t += i + 1), plain (TieOutputsPolicy::allow_plain).

template <bool Zero>
class TieMember : public cs::Component<TieMember<Zero>> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> t;
    };
    static void compute_forward(sycl::id<1> i, const Inputs&, const Outputs& o) {
        if constexpr (Zero)
            o.t[i] = 0.0f;
        else
            o.t[i] = o.t[i] + static_cast<float>(i[0] + 1);
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override { this->declare_output(outputs.t, "t", cs::unspecified_shape<1>()); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return this->create_launch_config(cs::range_desc(outputs.t.shape()));
    }
};

/// Fin: a free input x [kN] with an init (0 fill 3, 1 from_host 4, 2 custom_host 5); y = x.

class Fin : public cs::Component<Fin> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& a, const Outputs& o) { o.y[i] = a.x[i]; }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       kind_;

public:
    explicit Fin(int kind) : kind_{kind} {}
    void declare() override {
        declare_input(inputs.x, "x", kN);
        declare_output(outputs.y, "y", kN);
        if (kind_ == 0) init_input("x", cs::init::fill(3.0f));
        if (kind_ == 1) init_input("x", cs::init::from_host(std::vector<float>(kN, 4.0f)));
        if (kind_ == 2)
            init_input("x", cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostInit<DType::f32, 1>&)>{
                                [](cs::HostInit<DType::f32, 1>& h) {
                                    for (std::size_t k = 0; k < h.size(); ++k) h[k] = 5.0f;
                                }}));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(outputs.y.shape())); }
};

/// RGen: a free u32 input `counts` [n] drives the ragged output r [n, counts[i]]; r block i = 1000 i + q.
/// Section 10: ragged is rank 2 (was [n, counts[i], 3]); a component cannot read values, so RGenTop hooks the free ragged input
/// `layout` with counts_from(g.counts) and r copies its shape in prepare_forward.
/// The same device as Fill in test_ragged.cpp: it tests a hook (counts_from) on a ragged INPUT.

class RGen : public cs::Component<RGen> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1>       counts;
        cs::RaggedInput<DType::f32, 2> layout;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> r;
    };
    static void compute_forward(sycl::id<1> id, const Inputs&, const Outputs& o) {
        auto b = o.r.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = static_cast<float>(id[0] * 1000 + q);
    }

    Inputs  inputs;
    Outputs outputs;

private:
    std::vector<std::uint32_t>      init_;

public:
    explicit RGen(std::vector<std::uint32_t> init) : init_{std::move(init)} {}
    void declare() override {
        declare_input(inputs.counts, "counts", init_.size());
        init_input("counts", cs::init::from_host(init_));
        declare_input(inputs.layout, "layout");
        declare_output(outputs.r, "r");
    }
    void prepare_forward() override { set_shape(outputs.r, shape_of(inputs.layout)); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.counts.shape()));
    }
};

// ---- systems ---------------------------------------------------------------------------------------------------------
class TieTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<TieMember<true>>("z"));
        add_subsystem(cs::subsystem_desc<TieMember<false>>("a"));
        tie_outputs<DType::f32, 1>({"z.t", "a.t"}, "t", cs::shape(kN), cs::TieOutputsPolicy::allow_plain);
        // Execution order is data dependencies only (design review, section 3), and tied writers do not wait on each
        // other: the accumulator runs after the zeroing producer only because the edge is declared.
        depends_on("a", "z");
    }
};
class FinTop : public cs::System {
    int kind_;

public:
    explicit FinTop(int kind) : kind_{kind} {}
    void setup() override { add_subsystem(cs::subsystem_desc<Fin>("fin", kind_)); }
};
class TransInner : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<W1>("p"));
        add_subsystem(cs::subsystem_desc<Rd>("c"));
        connect("p.y", "c.x");
    }
};
class TransTop : public cs::System {
public:
    void setup() override {
        add_subsystem(kTransient, cs::subsystem_desc<TransInner>("op"));
        add_subsystem(cs::subsystem_desc<Rd>("sink"));
        connect("op.c.z", "sink.x");
    }
};
class RGenTop : public cs::System {
    std::vector<std::uint32_t> counts_;

public:
    explicit RGenTop(std::vector<std::uint32_t> c) : counts_{std::move(c)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<RGen>("g", counts_));
        shape_hook("g.layout", cs::ragged::counts_from("g.counts"));
    }
};

}  // namespace tep

// =============================================================================================================================
TEST_CASE("Ties: a zeroing producer and an accumulator give i + 1 on every forward", "[epoch][tie]") {
    cs::Model m = cs::create_model<tep::TieTop>(default_config());
    m.setup();
    for (int r = 0; r < 4; ++r) {  // the first forward and three more
        forward_and_wait(m);
        const auto v = synced_output<DType::f32, 1>(m, "t");
        for (std::size_t i = 0; i < tep::kN; ++i) CHECK(v.span()[i] == static_cast<float>(i + 1));
    }
}

TEST_CASE("Inputs: a free input takes its initializer, and update_input replaces it (fill, from_host, custom_host)", "[epoch][init]") {
    for (int kind = 0; kind < 3; ++kind) {
        INFO("init kind " << kind);
        cs::Model m = cs::create_model<tep::FinTop>(default_config(), kind);
        m.setup();
        forward_and_wait(m);
        // update_input marks the variable stale and queues no work; the next run applies it (section 5).
        m.update_input("fin.x", cs::init::from_host(std::vector<float>(tep::kN, 9.0f)));
        forward_and_wait(m);
        const auto y = synced_output<DType::f32, 1>(m, "fin.y");
        for (std::size_t i = 0; i < tep::kN; ++i) CHECK(y.span()[i] == 9.0f);
    }
}

TEST_CASE("Transient: a transient subsystem is deleted after its turn, every forward", "[epoch][transient]") {
    cs::Model m = cs::create_model<tep::TransTop>(default_config());
    m.setup();
    for (int r = 0; r < 2; ++r) {
        // the transient is deleted after its turn, so its variables cannot be read afterwards
        forward_and_wait(m);
        // r04: the node stays in the tree; its turn over, it is out of scope (dearmed)
        REQUIRE(cs::detail::ModelAccess::node(m, "op") != nullptr);
        CHECK_FALSE(cs::detail::ModelAccess::node(m, "op")->in_scope);
    }
}

// =============================================================================================================================
TEST_CASE("Views: two views of one output with no forward in between show the same values", "[epoch][view]") {
    cs::Model m = cs::create_model<tep::FinTop>(default_config(), 0);
    m.setup();
    forward_and_wait(m);
    const auto v1 = synced_output<DType::f32, 1>(m, "fin.y");
    const auto v2 = synced_output<DType::f32, 1>(m, "fin.y");
    REQUIRE(v1.size() == tep::kN);
    for (std::size_t i = 0; i < tep::kN; ++i) {
        CHECK(v1.span()[i] == 3.0f);
        CHECK(v2.span()[i] == v1.span()[i]);
    }
}

TEST_CASE("Views: a forward with a new input gives its own values", "[epoch][view]") {
    cs::Model m = cs::create_model<tep::FinTop>(default_config(), 1);
    m.setup();
    for (const float x : {2.0f, 7.0f, 11.0f}) {
        // update_input queues nothing; the value is applied at the next run (before the first run too)
        m.update_input("fin.x", cs::init::from_host(std::vector<float>(tep::kN, x)));
        forward_and_wait(m);
        const auto v = synced_output<DType::f32, 1>(m, "fin.y");
        for (std::size_t i = 0; i < tep::kN; ++i) CHECK(v.span()[i] == x);
    }
}

TEST_CASE("Views: a value-driven ragged output follows its driver", "[epoch][view][ragged]") {
    const std::vector<std::uint32_t> counts{2, 0, 3, 1, 4, 2};
    cs::Model                        m = cs::create_model<tep::RGenTop>(default_config(), counts);
    m.setup();
    for (int r = 0; r < 4; ++r) forward_and_wait(m);
    // Design review 11.1: a shape change resizes every container of the variable, the host one included. So this FIRST view
    // stays alive across the change and follows it; spans taken before the change are invalid and are re-taken below.
    const auto v = m.view_ragged_output<DType::f32, 2>("g.r");
    v.sync();
    REQUIRE(v.n() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const auto b = v.block(i);
        REQUIRE(b.size() == static_cast<std::size_t>(counts[i]));
        for (std::size_t q = 0; q < b.size(); ++q) CHECK(b[q] == static_cast<float>(i * 1000 + q));
    }
    // a changed driver is seen by the same view after the next forward and a sync
    const std::vector<std::uint32_t> next{1, 1, 1, 5, 0, 2};
    m.update_input("g.counts", cs::init::from_host(next));
    forward_and_wait(m);
    v.sync();
    REQUIRE(v.n() == next.size());
    for (std::size_t i = 0; i < next.size(); ++i) {
        const auto b = v.block(i);
        REQUIRE(b.size() == static_cast<std::size_t>(next[i]));
        for (std::size_t q = 0; q < b.size(); ++q) CHECK(b[q] == static_cast<float>(i * 1000 + q));
    }
    // a view made after the change agrees
    const auto w = m.view_ragged_output<DType::f32, 2>("g.r");
    w.sync();
    REQUIRE(w.n() == next.size());
    for (std::size_t i = 0; i < next.size(); ++i) CHECK(w.block(i).size() == static_cast<std::size_t>(next[i]));
}

TEST_CASE("Views: a dearmed transient's variables cannot be viewed by path", "[epoch][view][transient]") {
    cs::Model m = cs::create_model<tep::TransTop>(default_config());
    m.setup();
    forward_and_wait(m);
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::string what = "<no exception>";
        try {
            (void)m.view_output<DType::f32, 1>("op.p.y");
        } catch (const std::exception& e) {
            what = e.what();
        }
        INFO("thrown: " << what);
        CHECK(what != "<no exception>");
        // r04: the holder is found by name; its variable has no data (the node is out of scope).
        CHECK(what.find("no holder of this variable has data") != std::string::npos);
    }
    // the eternal sink is still readable
    const auto v = synced_output<DType::f32, 1>(m, "sink.z");
    for (std::size_t i = 0; i < tep::kN; ++i) CHECK(v.span()[i] == 4.0f * static_cast<float>(i + 1));
}
