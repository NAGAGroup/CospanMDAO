// Ragged tie aliases, on the section 10 shape model.
//
//   A tie alias takes a SHAPE DEFINITION like a port (cs::shape(...): unspecified, a leading integer, cs::counts(v) or
//   cs::offsets(v)); a ragged size that comes from VALUES is a shape_hook on the alias (cs::ragged::counts_from).
//
// Members are RaggedOutput / RaggedInput declared with all dims unspecified. Plain members use TieOutputsPolicy::allow_plain
// (the producers write one after the other); atomic ragged members take the require_atomic default.
//
// Design review notes for this port:
//  - section 3: execution order is data dependencies only, and tied writers never wait on each other, so every tie whose
//    members must write in an order (a zeroing producer, then an accumulator; a builder, then an in-place transform)
//    declares depends_on("q", "p");
//  - section 5: views are taken after the forward and sync()ed; a ragged view is n() / block_offsets() / block(i) / span();
//  - section 10: ragged is RANK 2 only (every rank-3 block "[N, c, 3]" / "[N, m, m]" is now "[N, c]": block i holds counts[i]
//    elements), there are no derived dimensions and no tie shape rules (alias::dims, from_output_shape, ...); a hook cannot
//    target an output, so a producer sized by values (Gen) copies the shape of a free ragged `layout` input that its System
//    hooks with counts_from.
//
// DELETED CASES AND VARIANTS (reasons):
//  - the alias-shape variants "from_input_shape of a ragged input", "input_dim of a ragged input", "output_dim of an internal
//    ragged output" (shape inheritance): no hook copies a ragged shape (from_shape refuses a per-index dimension) and a tie
//    carries no shape rule, so there is nothing equivalent;
//  - the Sz2 / GenCols column selection ("sizes[i, 1]"): counts_from reads column 0 only;
//  - the error variants "a per-index spec on dimension 0" and "input_values naming an output": unwritable now.
//
// Everything is checked against host references.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace trt {

using Sizes  = std::vector<std::size_t>;
using Counts = std::vector<std::uint32_t>;

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

/// The elements of a synced ragged view, end to end.
template <class View>
std::vector<float> flat_of(const View& v) {
    return std::vector<float>(v.span().begin(), v.span().end());
}

// ---- CProd: counts = a free u32 input (init from the host), copied ----------------------------------------

class CProd : public cs::Component<CProd> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> raw;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> counts;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& a, const Outputs& o) { o.counts[i] = a.raw[i]; }

    Inputs  inputs;
    Outputs outputs;

private:
    Counts                    init_;

public:
    explicit CProd(Counts init) : init_{std::move(init)} {}
    void setup() override {
        declare_input(inputs.raw, "raw", init_.size());
        init_input("raw", cs::init::from_host(init_));
        declare_output(outputs.counts, "counts", init_.size());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.counts.shape()));
    }
};

// ---- Gen: pts [N, c_i], p[i][q] = i * 1000 + q ----------------------------------------------------------------
// `layout` is hooked by the System (counts_from the counts); Gen copies its shape. `counts` is an (unused) connected input, a
// hook source of its own.

class Gen : public cs::Component<Gen> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1>       counts;
        cs::RaggedInput<DType::f32, 2> layout;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> pts;
    };
    static void compute_impl(sycl::id<1> id, const Inputs&, const Outputs& o) {
        auto b = o.pts.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = static_cast<float>(id[0] * 1000 + q);
    }

    Inputs  inputs;
    Outputs outputs;

private:
    int                             mode_;  // 0 copy layout's shape; 1 pts's counts come from a definition (set nothing); 2 copy anyway

public:
    explicit Gen(int mode = 0) : mode_{mode} {}
    void setup() override {
        declare_input(inputs.counts, "counts", cs::unspecified);
        declare_input(inputs.layout, "layout");
        declare_output(outputs.pts, "pts");
    }
    void pre_forward() override {
        if (mode_ != 1) set_shape(outputs.pts, shape_of(inputs.layout));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{inputs.layout.size()}));
    }
};

// ---- tie members: a rank-2 ragged output with all dims unspecified ----------------------------------------------------------
/// Op: 0 zero, 1 accumulate (t += (i+1)(q+1)), 2 build (t = (i+1)(q+1)), 3 factor (t = 2 t + 1, reads its own output).
template <int Op>
class TieMember : public cs::Component<TieMember<Op>> {
public:
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> t;
    };
    static void compute_impl(sycl::id<1> id, const Inputs&, const Outputs& o) {
        auto        b    = o.t.block(id[0]);
        const float base = static_cast<float>(id[0] + 1);
        for (std::size_t q = 0; q < b.size(); ++q) {
            if constexpr (Op == 0) b[q] = 0.0f;
            if constexpr (Op == 1) b[q] += base * static_cast<float>(q + 1);
            if constexpr (Op == 2) b[q] = base * static_cast<float>(q + 1);
            if constexpr (Op == 3) b[q] = 2.0f * b[q] + 1.0f;
        }
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override { this->declare_output(outputs.t, "t"); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(sycl::range<1>{outputs.t.size()}));
    }
};
/// A member that declares its leading dimension (N = 4): it meets the alias's definition at the tie.
class TieMemberLead : public TieMember<0> {
public:
    void setup() override { declare_output(outputs.t, "t", 4); }
};
// ---- atomic ragged members: zero (store), accumulate (fetch_add) ------------------------------------------------

template <int Op>
class TieMemberA : public cs::Component<TieMemberA<Op>> {
public:
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::atomic_f32, 2> t;
    };
    static void compute_impl(sycl::id<1> id, const Inputs&, const Outputs& o) {
        auto        b    = o.t.block(id[0]);
        const float base = static_cast<float>(id[0] + 1);
        for (std::size_t q = 0; q < b.size(); ++q) {
            if constexpr (Op == 0) b[q].store(0.0f);
            if constexpr (Op == 1) b[q].fetch_add(base * static_cast<float>(q + 1));
        }
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override { this->declare_output(outputs.t, "t"); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(sycl::range<1>{outputs.t.size()}));
    }
};
using ZeroA = TieMemberA<0>;
using AccA  = TieMemberA<1>;

using Zero   = TieMember<0>;
using Acc    = TieMember<1>;
using Build0 = TieMember<2>;
using Factor = TieMember<3>;

// ---- SumT: sums[i] = the sum of block i of a ragged input ---------------------------------------------------------------

class SumT : public cs::Component<SumT> {
public:
    struct Inputs {
        cs::RaggedInput<DType::f32, 2> pts;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    static void compute_impl(sycl::id<1> id, const Inputs& in, const Outputs& out) {
        auto  b = in.pts.block(id[0]);
        float s = 0.0f;
        for (std::size_t q = 0; q < b.size(); ++q) s += b[q];
        out.sums[id] = s;
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.pts, "pts");
        declare_output(outputs.sums, "sums", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.sums, shape_of(inputs.pts).n); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.sums.shape()));
    }
};

// ---- tie_outputs top: sz -> gen, then the tie (zero/acc or build/factor), then a sum ---------------------------------
/// The tie alias is hooked with counts_from(source), the source being:
/// shape_mode 0: sz.counts (an upstream OUTPUT)
/// shape_mode 1: sz.raw (a FREE input of an earlier child)
/// shape_mode 2: gen.counts (a CONNECTED input, fed by sz.counts)
template <class P, class Q, bool Atomic = false>
class TieOutTop : public cs::System {
    Counts init_;
    int    shape_mode_;

public:
    TieOutTop(Counts init, int shape_mode) : init_{std::move(init)}, shape_mode_{shape_mode} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<CProd>("sz", init_));
        add_subsystem(cs::subsystem_desc<Gen>("gen"));
        add_subsystem(cs::subsystem_desc<P>("p"));
        add_subsystem(cs::subsystem_desc<Q>("q"));
        add_subsystem(cs::subsystem_desc<SumT>("sum"));
        depends_on("q", "p");  // tied writers never wait on each other (section 3): q needs what p wrote
        connect("sz.counts", "gen.counts");
        shape_hook("gen.layout", cs::ragged::counts_from("sz.counts"));
        if constexpr (Atomic)
            tie_outputs<DType::atomic_f32, 2>({"p.t", "q.t"}, "t");  // require_atomic (the default)
        else
            tie_outputs<DType::f32, 2>({"p.t", "q.t"}, "t", cs::shape(), cs::TieOutputsPolicy::allow_plain);
        const char* source = shape_mode_ == 0 ? "sz.counts" : shape_mode_ == 1 ? "sz.raw" : "gen.counts";
        shape_hook("t", cs::ragged::counts_from(source));
        connect("t", "sum.pts");
    }
};

/// The reference of the tie: block i has counts[i] elements.
struct TieRef {
    std::vector<float>         data, sums;
    std::vector<std::uint64_t> offsets{0};
};
TieRef tie_ref(const Counts& counts, bool factor) {
    TieRef r;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        float s = 0.0f;
        for (std::size_t q = 0; q < counts[i]; ++q) {
            const float base = static_cast<float>((i + 1) * (q + 1));
            const float v    = factor ? 2.0f * base + 1.0f : base;
            r.data.push_back(v);
            s += v;
        }
        r.sums.push_back(s);
        r.offsets.push_back(r.data.size());
    }
    return r;
}

void run_tie_out(cs::Model& m, const Counts& counts, bool factor) {
    m.update_input("sz.raw", cs::init::from_host(counts));
    m.run();
    m.wait();
    const TieRef ref  = tie_ref(counts, factor);
    auto         sums = cst::synced_output<DType::f32, 1>(m, "sum.sums");
    REQUIRE(sums.size() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) CHECK(sums.span()[i] == ref.sums[i]);
    auto pts = m.view_ragged_input<DType::f32, 2>("sum.pts");  // the alias's data, through a peer
    pts.sync();
    CHECK(pts.block_offsets() == ref.offsets);
    CHECK(flat_of(pts) == ref.data);
    CHECK(pts.n() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) CHECK(pts.block(i).size() == counts[i]);
}

// ---- tie_inputs top: gen -> (two ragged inputs tied, connected from a ragged output) -----------------------------------------------

template <int K>
class SumK : public cs::Component<SumK<K>> {
public:
    struct Inputs {
        cs::RaggedInput<DType::f32, 2> pts;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    static void compute_impl(sycl::id<1> id, const Inputs& in, const Outputs& out) {
        auto  b = in.pts.block(id[0]);
        float s = 0.0f;
        for (std::size_t q = 0; q < b.size(); ++q) s += b[q];
        out.sums[id] = static_cast<float>(K) * s;
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        this->declare_input(inputs.pts, "pts");
        this->declare_output(outputs.sums, "sums", cs::unspecified);
    }
    void pre_forward() override { this->set_shape(outputs.sums, this->shape_of(inputs.pts).n); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(outputs.sums.shape()));
    }
};

class TieInTop : public cs::System {
    Counts init_;

public:
    explicit TieInTop(Counts init) : init_{std::move(init)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<CProd>("sz", init_));
        add_subsystem(cs::subsystem_desc<Gen>("gen"));
        add_subsystem(cs::subsystem_desc<SumK<1>>("sa"));
        add_subsystem(cs::subsystem_desc<SumK<2>>("sb"));
        connect("sz.counts", "gen.counts");
        shape_hook("gen.layout", cs::ragged::counts_from("sz.counts"));
        tie_inputs<DType::f32, 2>({"sa.pts", "sb.pts"}, "pts");
        connect("gen.pts", "pts");
    }
};

// ---- a CONNECTED tie_inputs alias that declares its counts ---------------------------------------------------------------------
/// sz -> gen (ragged [N, counts[i]]) -> two RaggedInputs tied with cs::shape(cs::counts(declared)), connected from gen.pts: the
/// connect merges the declared counts into gen.pts's definition, and Gen's set_shape is checked against it at the forward.
class TieInDeclTop : public cs::System {
    Counts init_;
    Sizes  declared_;
    int    gen_mode_;

public:
    TieInDeclTop(Counts init, Sizes declared, int gen_mode = 1)
        : init_{std::move(init)}, declared_{std::move(declared)}, gen_mode_{gen_mode} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<CProd>("sz", init_));
        add_subsystem(cs::subsystem_desc<Gen>("gen", gen_mode_));
        add_subsystem(cs::subsystem_desc<SumK<1>>("sa"));
        add_subsystem(cs::subsystem_desc<SumK<2>>("sb"));
        connect("sz.counts", "gen.counts");
        shape_hook("gen.layout", cs::ragged::counts_from("sz.counts"));
        tie_inputs<DType::f32, 2>({"sa.pts", "sb.pts"}, "pts", cs::shape(cs::counts(declared_)));
        connect("gen.pts", "pts");
    }
};

// ---- Src: a plain source, x[i] = 0.5 i + 1 -----------------------------------------------------------------------

class Src : public cs::Component<Src> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> x;
    };
    static void compute_impl(sycl::id<1> i, const Inputs&, const Outputs& o) { o.x[i] = 0.5f * static_cast<float>(i[0]) + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

private:
    std::size_t               n_;

public:
    explicit Src(std::size_t n) : n_{n} {}
    void setup() override { declare_output(outputs.x, "x", n_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.x.shape()));
    }
};

/// BuildX: t = in[i] (q + 1), a tied output; reads the outside input `in`.

class BuildX : public cs::Component<BuildX> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> t;
    };
    static void compute_impl(sycl::id<1> id, const Inputs& a, const Outputs& o) {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = a.in[id] * static_cast<float>(q + 1);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.in, "in", cs::unspecified);
        declare_output(outputs.t, "t");
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{outputs.t.size()}));
    }
};

class Cons : public cs::Component<Cons> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& a, const Outputs& o) { o.out[i] = 2.0f * a.in[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.in, "in", cs::unspecified);
        declare_output(outputs.out, "out", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.out, shape_of(inputs.in)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.out.shape()));
    }
};

/// g: build (t = in (q+1)) -> factor (t = 2 t + 1, in place) -> sum; the tie's alias is cs::shape(cs::counts(sizes)).
class G : public cs::System {
    Sizes sizes_;

public:
    explicit G(Sizes sizes) : sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<BuildX>("p"));
        add_subsystem(cs::subsystem_desc<Factor>("q"));
        add_subsystem(cs::subsystem_desc<SumT>("s"));
        tie_outputs<DType::f32, 2>({"p.t", "q.t"}, "t", cs::shape(cs::counts(sizes_)), cs::TieOutputsPolicy::allow_plain);
        depends_on("q", "p");  // tied writers never wait on each other (section 3): the in-place transform needs the build
        connect("t", "s.pts");
    }
};
/// The interface of the nested System: its promotions.
class Nested : public cs::System {
    Sizes sizes_;

public:
    explicit Nested(Sizes sizes) : sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<G>("g", sizes_));
        promote<DType::f32, 1>("g.p.in", "in");
        promote<DType::f32, 1>("g.s.sums", "sums");
    }
};
class NestedTop : public cs::System {
    Sizes sizes_;

public:
    explicit NestedTop(Sizes sizes) : sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("src", sizes_.size()));
        add_subsystem(cs::subsystem_desc<Nested>("inner", sizes_));
        add_subsystem(cs::subsystem_desc<Cons>("cons"));
        connect("src.x", "inner.in");
        connect("inner.sums", "cons.in");
    }
};

/// One forward, then cons.out (synced).
std::vector<float> forward_out(cs::Model& m) {
    m.run();
    m.wait();
    const auto v = cst::synced_output<DType::f32, 1>(m, "cons.out");
    return std::vector<float>(v.span().begin(), v.span().end());
}

std::vector<float> nested_ref(const Sizes& sizes) {
    std::vector<float> out;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const float x = 0.5f * static_cast<float>(i) + 1.0f;
        float       s = 0.0f;
        for (std::size_t q = 0; q < sizes[i]; ++q) s += 2.0f * (x * static_cast<float>(q + 1)) + 1.0f;
        out.push_back(2.0f * s + 1.0f);
    }
    return out;
}

// ---- tie shapes from INPUTS of a nested System --------------------------------------------------------------------------
/// BuildS: t = in (q + 1); also has the u32 input `sizes` (promoted by G; the alias hook reads its values).

class BuildS : public cs::Component<BuildS> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> in;
        cs::Input<DType::u32, 1> sizes;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> t;
    };
    static void compute_impl(sycl::id<1> id, const Inputs& a, const Outputs& o) {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = a.in[id] * static_cast<float>(q + 1);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.in, "in", cs::unspecified);
        declare_input(inputs.sizes, "sizes", cs::unspecified);
        declare_output(outputs.t, "t");
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{outputs.t.size()}));
    }
};

/// AccX: an ATOMIC ragged tie member: t += in (q + 1).

class AccX : public cs::Component<AccX> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::RaggedOutput<DType::atomic_f32, 2> t;
    };
    static void compute_impl(sycl::id<1> id, const Inputs& a, const Outputs& o) {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q].fetch_add(a.in[id] * static_cast<float>(q + 1));
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.in, "in", cs::unspecified);
        declare_output(outputs.t, "t");
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{outputs.t.size()}));
    }
};

/// The system of one variant (v):
///   0  the alias is hooked with counts_from of the promoted input `sizes` (u32 [N]); plain members (build, factor)
///   1  the alias declares cs::counts(sizes); plain members (build, factor)
///   2  ATOMIC members (zero, then AccX), the alias declares cs::counts(sizes); require_atomic (the default)
class GV : public cs::System {
    int   v_;
    Sizes sizes_;

public:
    GV(int v, Sizes sizes) : v_{v}, sizes_{std::move(sizes)} {}
    void setup() override {
        const auto policy = cs::TieOutputsPolicy::allow_plain;
        if (v_ == 0) {
            add_subsystem(cs::subsystem_desc<BuildS>("p"));
            promote<DType::u32, 1>("p.sizes", "sizes");
        } else if (v_ == 1) {
            add_subsystem(cs::subsystem_desc<BuildX>("p"));
        } else {
            add_subsystem(cs::subsystem_desc<ZeroA>("p"));
        }
        if (v_ == 2)
            add_subsystem(cs::subsystem_desc<AccX>("q"));
        else
            add_subsystem(cs::subsystem_desc<Factor>("q"));
        add_subsystem(cs::subsystem_desc<SumT>("s"));
        depends_on("q", "p");  // tied writers never wait on each other (section 3): q needs what p wrote
        switch (v_) {
            case 0:
                tie_outputs<DType::f32, 2>({"p.t", "q.t"}, "t", cs::shape(), policy);
                shape_hook("t", cs::ragged::counts_from("sizes"));
                break;
            case 1: tie_outputs<DType::f32, 2>({"p.t", "q.t"}, "t", cs::shape(cs::counts(sizes_)), policy); break;
            default: tie_outputs<DType::atomic_f32, 2>({"p.t", "q.t"}, "t", cs::shape(cs::counts(sizes_)));
        }
        connect("t", "s.pts");
    }
};
/// The interface of the variant's System: the promotions its wiring needs.
void promote_v(cs::System& s, int v) {
    s.promote<DType::f32, 1>(v == 2 ? "g.q.in" : "g.p.in", "in");
    if (v == 0) s.promote<DType::u32, 1>("g.sizes", "sizes");
    s.promote<DType::f32, 1>("g.s.sums", "sums");
}
class InnerV : public cs::System {
    int   v_;
    Sizes sizes_;

public:
    InnerV(int v, Sizes sizes) : v_{v}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<GV>("g", v_, sizes_));
        promote_v(*this, v_);
    }
};
class VTop : public cs::System {
    Counts counts_;
    int    v_;

public:
    VTop(Counts counts, int v) : counts_{std::move(counts)}, v_{v} {}
    void setup() override {
        const std::size_t n = counts_.size();
        Sizes             sizes(counts_.begin(), counts_.end());
        add_subsystem(cs::subsystem_desc<CProd>("sz", counts_));
        add_subsystem(cs::subsystem_desc<Src>("src", n));
        add_subsystem(cs::subsystem_desc<InnerV>("inner", v_, sizes));
        add_subsystem(cs::subsystem_desc<Cons>("cons"));
        connect("src.x", "inner.in");
        if (v_ == 0) connect("sz.counts", "inner.sizes");
        connect("inner.sums", "cons.in");
    }
};

/// forward -> cons.out of the variant: block i holds x_i (q + 1), then (variants 0, 1) 2 t + 1, summed; cons: 2 s + 1.
std::vector<float> variant_ref(const Sizes& sizes, int v) {
    std::vector<float> out;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const float x = 0.5f * static_cast<float>(i) + 1.0f;
        float       s = 0.0f;
        for (std::size_t q = 0; q < sizes[i]; ++q) {
            const float base = x * static_cast<float>(q + 1);
            s += v == 2 ? base : 2.0f * base + 1.0f;
        }
        out.push_back(2.0f * s + 1.0f);
    }
    return out;
}

// ---- errors ----------------------------------------------------------------------------------------------------------------
class ErrTop : public cs::System {
    int variant_;

public:
    explicit ErrTop(int variant) : variant_{variant} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<CProd>("sz", Counts{1, 2, 3, 4}));
        add_subsystem(cs::subsystem_desc<Src>("src", 4));
        if (variant_ == 0)
            add_subsystem(cs::subsystem_desc<TieMemberLead>("p"));
        else
            add_subsystem(cs::subsystem_desc<Zero>("p"));
        add_subsystem(cs::subsystem_desc<Acc>("q"));
        const auto policy = cs::TieOutputsPolicy::allow_plain;
        if (variant_ == 0) {  // the alias declares counts {1, 2} (N = 2), the member a leading dimension of 4
            tie_outputs<DType::f32, 2>({"p.t", "q.t"}, "t", cs::shape(cs::counts(Sizes{1, 2})), policy);
        } else if (variant_ == 1) {  // a hook on a dimension the alias declares
            tie_outputs<DType::f32, 2>({"p.t", "q.t"}, "t", cs::shape(cs::counts(Sizes{1, 2})), policy);
            shape_hook("t", cs::ragged::counts_from("sz.counts"));
        } else if (variant_ == 2) {  // counts_from names a float variable
            tie_outputs<DType::f32, 2>({"p.t", "q.t"}, "t", cs::shape(), policy);
            shape_hook("t", cs::ragged::counts_from("src.x"));
        } else if (variant_ == 3) {  // a source produced by a LATER child
            add_subsystem(cs::subsystem_desc<CProd>("late", Counts{1, 2, 3, 4}));
            tie_outputs<DType::f32, 2>({"p.t", "q.t"}, "t", cs::shape(), policy);
            shape_hook("t", cs::ragged::counts_from("late.counts"));
        } else {  // a FREE input whose owner (late) is registered after the tie's first member: no producer, no ordering
            add_subsystem(cs::subsystem_desc<CProd>("late", Counts{1, 2, 3, 4}));
            tie_outputs<DType::f32, 2>({"p.t", "q.t"}, "t", cs::shape(), policy);
            shape_hook("t", cs::ragged::counts_from("late.raw"));
        }
    }
};

/// AccIn: t += in[i] (q + 1); `t` is a tied ragged output, `in` a plain input.

class AccIn : public cs::Component<AccIn> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> t;
    };
    static void compute_impl(sycl::id<1> id, const Inputs& a, const Outputs& o) {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] += a.in[id] * static_cast<float>(q + 1);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.in, "in", cs::unspecified);
        declare_output(outputs.t, "t");
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{outputs.t.size()}));
    }
};

// ---- an outside tie over PROMOTIONS of inside ports with unspecified dims ----------------------------------------------------------------
class OtInner : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<AccIn>("q"));
        promote<DType::f32, 1>("q.in", "in");
        promote<DType::f32, 2>("q.t", "qt");  // a promotion of an inside output with unspecified dims: the outside tie's member
    }
};
/// tie_outputs over {outside writer w.t (zeroes), inner.qt (the inside accumulator)}, hooked with counts_from(sz.counts).
class OtTop : public cs::System {
    Counts init_;

public:
    explicit OtTop(Counts init) : init_{std::move(init)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<CProd>("sz", init_));
        add_subsystem(cs::subsystem_desc<Src>("src", init_.size()));
        add_subsystem(cs::subsystem_desc<Zero>("w"));
        add_subsystem(cs::subsystem_desc<OtInner>("inner"));
        add_subsystem(cs::subsystem_desc<SumT>("sum"));
        add_subsystem(cs::subsystem_desc<Cons>("cons"));
        tie_outputs<DType::f32, 2>({"w.t", "inner.qt"}, "t", cs::shape(), cs::TieOutputsPolicy::allow_plain);
        shape_hook("t", cs::ragged::counts_from("sz.counts"));
        // Tied writers never wait on each other (section 3): the accumulating System needs the zeroing writer first.
        depends_on("inner", "w");
        connect("src.x", "inner.in");
        connect("t", "sum.pts");
        connect("sum.sums", "cons.in");
    }
};

/// tie_inputs over {inner.pts (a promotion of the inside tied input sa.pts), sb.pts}, connected from gen.pts.
class TiInner : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<SumK<1>>("sa"));
        promote<DType::f32, 2>("sa.pts", "pts");
        promote<DType::f32, 1>("sa.sums", "sums");
    }
};
class TiTop : public cs::System {
    Counts init_;

public:
    explicit TiTop(Counts init) : init_{std::move(init)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<CProd>("sz", init_));
        add_subsystem(cs::subsystem_desc<Gen>("gen"));
        add_subsystem(cs::subsystem_desc<TiInner>("inner"));
        add_subsystem(cs::subsystem_desc<SumK<1>>("sb"));
        add_subsystem(cs::subsystem_desc<Cons>("c1"));
        add_subsystem(cs::subsystem_desc<Cons>("c2"));
        connect("sz.counts", "gen.counts");
        shape_hook("gen.layout", cs::ragged::counts_from("sz.counts"));
        tie_inputs<DType::f32, 2>({"inner.pts", "sb.pts"}, "tp");
        connect("gen.pts", "tp");
        connect("inner.sums", "c1.in");
        connect("sb.sums", "c2.in");
    }
};

}  // namespace trt

TEST_CASE("Ragged ties: tie_outputs of two ragged members (zero, then accumulate), sizes from an output, a free input or a connected input",
          "[ragged][tie]") {
    for (const int shape_mode : {0, 1, 2}) {
        INFO("shape_mode " << shape_mode);
        cs::Model m = cs::create_model<trt::TieOutTop<trt::Zero, trt::Acc>>(default_config(), trt::Counts{3, 0, 2, 4}, shape_mode);
        m.setup();
        trt::run_tie_out(m, {3, 0, 2, 4}, false);  // a zero in the middle
        trt::run_tie_out(m, {0, 5, 1, 2}, false);  // an empty first block, growth
        trt::run_tie_out(m, {0, 0, 0, 0}, false);  // all empty
        trt::run_tie_out(m, {2, 2, 2, 1}, false);  // and back
    }
}

TEST_CASE("Ragged ties: a plain tie whose second member transforms the first one's output in place", "[ragged][tie]") {
    for (const int shape_mode : {0, 1}) {
        INFO("shape_mode " << shape_mode);
        cs::Model m = cs::create_model<trt::TieOutTop<trt::Build0, trt::Factor>>(default_config(), trt::Counts{1, 3, 0, 2}, shape_mode);
        m.setup();
        trt::run_tie_out(m, {1, 3, 0, 2}, true);
        trt::run_tie_out(m, {4, 0, 2, 2}, true);
    }
}

TEST_CASE("Ragged ties: a connected tie_inputs alias that declares cs::counts is checked against the source's counts",
          "[ragged][tie]") {
    // was: alias::dims(input_dim, input_values, 3) read from the values of an input column. The alias now declares its counts
    // (a definition), the connect merges them into gen.pts's variable, and Gen's set_shape is checked against them. The declared
    // counts are fixed, so the old second forward with CHANGING sizes does not exist any more.
    SECTION("matching counts") {
        const trt::Counts sizes{2, 2, 0, 3};
        cs::Model         m = cs::create_model<trt::TieInDeclTop>(default_config(), sizes, trt::Sizes{2, 2, 0, 3});
        m.setup();
        m.run();
        m.wait();
        auto sa = cst::synced_output<DType::f32, 1>(m, "sa.sums");
        auto sb = cst::synced_output<DType::f32, 1>(m, "sb.sums");
        REQUIRE(sa.size() == 4);
        for (std::size_t i = 0; i < 4; ++i) {
            float want = 0.0f;
            for (std::size_t q = 0; q < sizes[i]; ++q) want += static_cast<float>(i * 1000 + q);
            CHECK(sa.span()[i] == want);
            CHECK(sb.span()[i] == 2.0f * want);
        }
        auto pa = m.view_ragged_input<DType::f32, 2>("sa.pts");
        pa.sync();
        REQUIRE(pa.n() == 4);
        for (std::size_t i = 0; i < 4; ++i) CHECK(pa.block(i).size() == sizes[i]);
    }
    SECTION("a producer that sets the shape of a declared-counts output is an error at the forward, even when they match") {
        // P5: a number on a specified dimension throws (the ragged copy form sets n and counts)
        for (const trt::Counts& c : {trt::Counts{2, 0, 1, 1}, trt::Counts{2, 0, 3, 1}}) {
            cs::Model m = cs::create_model<trt::TieInDeclTop>(default_config(), c, trt::Sizes{2, 0, 3, 1}, 2);
            m.setup();
            trt::expect_throws([&] { m.run(); }, {"gen.pts", "takes a number only on an unspecified dimension"});
            m.wait();
        }
    }
}

TEST_CASE("Ragged ties: tie_inputs of two RaggedInputs, connected from a ragged output", "[ragged][tie]") {
    cs::Model m = cs::create_model<trt::TieInTop>(default_config(), trt::Counts{2, 0, 3, 1});
    m.setup();
    for (const trt::Counts& counts : {trt::Counts{2, 0, 3, 1}, trt::Counts{0, 4, 0, 2}, trt::Counts{1, 1, 1, 1}}) {
        m.update_input("sz.raw", cs::init::from_host(counts));
        m.run();
        m.wait();
        auto sa = cst::synced_output<DType::f32, 1>(m, "sa.sums");
        auto sb = cst::synced_output<DType::f32, 1>(m, "sb.sums");
        REQUIRE(sa.size() == counts.size());
        REQUIRE(sb.size() == counts.size());
        for (std::size_t i = 0; i < counts.size(); ++i) {
            float want = 0.0f;
            for (std::size_t q = 0; q < counts[i]; ++q) want += static_cast<float>(i * 1000 + q);
            CHECK(sa.span()[i] == want);
            CHECK(sb.span()[i] == 2.0f * want);
        }
        auto pa = m.view_ragged_input<DType::f32, 2>("sa.pts");
        pa.sync();
        REQUIRE(pa.n() == counts.size());
        for (std::size_t i = 0; i < counts.size(); ++i) CHECK(pa.block(i).size() == counts[i]);  // the per-index dim, as sizes
    }
}

TEST_CASE("Ragged ties: a tie with cs::counts host arrays in a nested System, values and a second forward",
          "[ragged][tie]") {
    // The plain half of the old "chunked vs plain" cases; the chunked half (budgets, allocation counts) is deleted.
    const trt::Sizes sizes{1, 3, 2, 3, 0, 3, 1, 2};
    const std::size_t n    = sizes.size();
    const auto        want = trt::nested_ref(sizes);
    cs::Model         m    = cs::create_model<trt::NestedTop>(default_config(), sizes);
    m.setup();
    for (int rep = 0; rep < 2; ++rep) {
        INFO("forward " << rep + 1);
        const auto out = trt::forward_out(m);
        REQUIRE(out.size() == n);
        for (std::size_t i = 0; i < n; ++i) CHECK(out[i] == want[i]);
    }
}

TEST_CASE("Ragged ties: ATOMIC ragged members, zero then accumulate (require_atomic), several forwards", "[ragged][tie][atomic]") {
    for (const int shape_mode : {0, 1}) {
        INFO("shape_mode " << shape_mode);
        cs::Model m = cs::create_model<trt::TieOutTop<trt::ZeroA, trt::AccA, true>>(default_config(), trt::Counts{3, 0, 2, 4}, shape_mode);
        m.setup();
        trt::run_tie_out(m, {3, 0, 2, 4}, false);
        trt::run_tie_out(m, {0, 5, 1, 2}, false);
        trt::run_tie_out(m, {2, 2, 2, 1}, false);
    }
}

TEST_CASE("Ragged ties: tie shapes from a promoted input's values and from declared counts (variants 0-2), in a nested System",
          "[ragged][tie]") {
    // The plain half of the old "alias shapes ... inside a ChunkedSystem" and "Chunked vs plain: tie aliases" cases. The old
    // variants 1-3 (shape inherited from a ragged input or an internal ragged output) are deleted, see the header.
    const trt::Counts counts{1, 3, 2, 3, 0, 3, 1, 2};
    const trt::Sizes  sizes(counts.begin(), counts.end());
    for (int v = 0; v <= 2; ++v) {
        INFO("variant " << v);
        const auto want = trt::variant_ref(sizes, v);
        cs::Model  m    = cs::create_model<trt::VTop>(default_config(), counts, v);
        m.setup();
        for (int rep = 0; rep < 2; ++rep) {
            INFO("forward " << rep + 1);
            const auto out = trt::forward_out(m);
            REQUIRE(out.size() == counts.size());
            for (std::size_t i = 0; i < out.size(); ++i) CHECK(out[i] == want[i]);
        }
    }
}

TEST_CASE("Ragged ties: errors", "[ragged][tie]") {
    SECTION("a member's leading dimension that does not match the alias's counts (was: a per-index array against a fixed dim 0)") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 0);
        trt::expect_throws([&] { m.setup(); }, {"tie_outputs", "dimension 0", "declares 2", "declares 4", "the extents must match"});
    }
    SECTION("a hook on a dimension the alias declares (was: a per-index array that does not match dimension 0 at run time)") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 1);
        trt::expect_throws([&] { m.setup(); }, {"shape_hook", "'t'", "dimension 1", "fixed counts [1,2]", "a hook can only own an unspecified dimension"});
    }
    SECTION("counts_from naming a float variable (was: input_values naming an output)") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 2);
        trt::expect_throws([&] { m.setup(); }, {"shape_hook", "'src.x'", "must have an integer dtype"});
    }
    SECTION("a source produced by a later child is no setup error any more (P5: no source-ordering check)") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 3);
        CHECK_NOTHROW(m.setup());
    }
    SECTION("a free input whose owner is registered after the alias's first member is fine: it has no producer to order") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 4);
        CHECK_NOTHROW(m.setup());
    }
}

TEST_CASE("Ragged ties: an outside tie_outputs over a promotion of an inside output plus an outside writer", "[ragged][tie]") {
    // The plain half of the old "Chunked vs plain: an outside tie_outputs over a promotion ..." case, with a host reference.
    // The accumulating System `inner` is ordered after the zeroing writer `w` by depends_on("inner", "w") on a System child:
    // depends_on hands the events on to every child's submissions for one forward (system.hpp).
    const trt::Counts counts{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    cs::Model         m = cs::create_model<trt::OtTop>(default_config(), counts);
    m.setup();
    m.run();
    m.wait();
    std::vector<float>         want_flat, want_cons;
    std::vector<std::uint64_t> want_offs{0};
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const float x = 0.5f * static_cast<float>(i) + 1.0f;  // Src's x[i]
        float       s = 0.0f;
        for (std::size_t q = 0; q < counts[i]; ++q) {
            want_flat.push_back(x * static_cast<float>(q + 1));
            s += want_flat.back();
        }
        want_offs.push_back(want_flat.size());
        want_cons.push_back(2.0f * s + 1.0f);
    }
    const auto cons = cst::to_vector(cst::synced_output<DType::f32, 1>(m, "cons.out"));
    CHECK(cst::all_close(cons, want_cons));
    auto t = m.view_ragged_input<DType::f32, 2>("sum.pts");  // the tie alias: the full root
    t.sync();
    CHECK(t.block_offsets() == want_offs);
    CHECK(cst::all_close(trt::flat_of(t), want_flat));
}

TEST_CASE("Ragged ties: an outside tie_inputs over a promotion of an inside tied input", "[ragged][tie]") {
    // The plain half of the old "Chunked vs plain: an outside tie_inputs over a promotion ..." case, with a host reference.
    const trt::Counts first{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    const trt::Counts second{2, 0, 5, 5, 1, 1, 3, 4, 0, 2};
    cs::Model         m = cs::create_model<trt::TiTop>(default_config(), first);
    m.setup();
    for (const trt::Counts* counts : {&first, &second}) {
        m.update_input("sz.raw", cs::init::from_host(*counts));
        m.run();
        m.wait();
        for (const char* out : {"c1.out", "c2.out"}) {
            const auto o = cst::to_vector(cst::synced_output<DType::f32, 1>(m, out));
            REQUIRE(o.size() == counts->size());
            for (std::size_t i = 0; i < counts->size(); ++i) {
                float s = 0.0f;
                for (std::size_t q = 0; q < (*counts)[i]; ++q) s += static_cast<float>(i * 1000 + q);
                CHECK(cst::close(o[i], 2.0f * s + 1.0f));  // Cons: 2 s + 1, both members read the same alias
            }
        }
    }
}
