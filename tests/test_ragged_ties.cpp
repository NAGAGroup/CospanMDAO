// Ragged tie aliases: a tie alias takes the same per-index shape rules as a port.
//
//   alias::dims(spec0, spec1, ...) with an integer, per_index, output_value, output_values, output_dim;
//   from_output_shape copies a ragged shape, and so does a connection (an unspecified dim takes its source's).
//
// Members are RaggedOutput / RaggedInput declared with all dims unspecified. Plain members use TieOutputsPolicy::allow_plain
// (the producers write one after the other); atomic ragged members take the require_atomic default.
// Alias sources may also be INPUTS of the declaring System (input_value / input_values / input_dim / from_input_shape).
//
// Design review notes for this port:
//  - section 3: execution order is data dependencies only, and tied writers never wait on each other, so every tie whose
//    members must write in an order (a zeroing producer, then an accumulator; a builder, then an in-place transform)
//    declares depends_on("q", "p");
//  - section 5: views are taken after the forward and sync()ed; a ragged view is n() / block_offsets() / block(i) / span();
//  - section 6: value-driven ragged shapes come from the device column, so the old assertions on the host's per-index
//    arrays (extents(d), is_per_index(d)) are replaced by the blocks' sizes;
//  - ChunkedSystem is dead code: every chunked case (the chunk window, full_passes, resolve_shapes, ragged_uploads, the
//    opacity routes, the topology cache) is deleted. The plain half of the "chunked vs plain" comparisons is kept where it
//    checks a value the other cases do not.
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
struct CProdBody {
    struct Inputs {
        cs::Input<DType::u32, 1> raw;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> counts;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.counts[i] = a.raw[i]; }
};
class CProd : public cs::Component<CProdBody> {
    Counts                    init_;
    cs::Input<DType::u32, 1>  raw_;
    cs::Output<DType::u32, 1> counts_;

public:
    explicit CProd(Counts init) : init_{std::move(init)} {}
    void setup() override {
        declare_input(raw_, "raw", init_.size());
        init_input("raw", cs::init::from_host(init_));
        declare_output(counts_, "counts", init_.size());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {raw_}, {counts_}, {});
    }
};

// ---- Gen: pts [N, c_i, 3], p[i][q] = i * 1000 + q ---------------------------------------------------------------
struct GenBody {
    struct Inputs {
        cs::Input<DType::u32, 1> counts;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> pts;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        auto b = o.pts.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = static_cast<float>(id[0] * 1000 + q);
    }
};
class Gen : public cs::Component<GenBody> {
    cs::Input<DType::u32, 1>        counts_;
    cs::RaggedOutput<DType::f32, 3> pts_;

public:
    void setup() override {
        declare_input(counts_, "counts", cs::unspecified);
        declare_output(pts_, "pts", cs::FromInputDim{"counts", 0}, cs::FromInputValues{"counts"}, 3);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {counts_}, {pts_}, {});
    }
};

// ---- tie members: a rank-3 ragged output with all dims unspecified -------------------------------------------------------------
/// Op: 0 zero, 1 accumulate (t += (i+1)(q+1)), 2 build (t = (i+1)(q+1)), 3 factor (t = 2 t + 1, reads its own output).
template <int Op>
struct TieBody {
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        auto        b    = o.t.block(id[0]);
        const float base = static_cast<float>(id[0] + 1);
        for (std::size_t q = 0; q < b.size(); ++q) {
            if constexpr (Op == 0) b[q] = 0.0f;
            if constexpr (Op == 1) b[q] += base * static_cast<float>(q + 1);
            if constexpr (Op == 2) b[q] = base * static_cast<float>(q + 1);
            if constexpr (Op == 3) b[q] = 2.0f * b[q] + 1.0f;
        }
    }
};
template <int Op>
class TieMember : public cs::Component<TieBody<Op>> {
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override { this->declare_output(t_, "t", cs::unspecified_shape<3>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(sycl::range<1>{t_.size()}, {}, {t_}, {});
    }
};
// ---- atomic ragged members: zero (store), accumulate (fetch_add) ------------------------------------------------
template <int Op>
struct TieBodyA {
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::atomic_f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        auto        b    = o.t.block(id[0]);
        const float base = static_cast<float>(id[0] + 1);
        for (std::size_t q = 0; q < b.size(); ++q) {
            if constexpr (Op == 0) b[q].store(0.0f);
            if constexpr (Op == 1) b[q].fetch_add(base * static_cast<float>(q + 1));
        }
    }
};
template <int Op>
class TieMemberA : public cs::Component<TieBodyA<Op>> {
    cs::RaggedOutput<DType::atomic_f32, 3> t_;

public:
    void setup() override { this->declare_output(t_, "t", cs::unspecified_shape<3>()); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(sycl::range<1>{t_.size()}, {}, {t_}, {});
    }
};
using ZeroA = TieMemberA<0>;
using AccA  = TieMemberA<1>;

using Zero   = TieMember<0>;
using Acc    = TieMember<1>;
using Build0 = TieMember<2>;
using Factor = TieMember<3>;

// ---- SumT: sums[i] = the sum of block i of a ragged [N, a, b] input ---------------------------------------------------------
struct SumTBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 3> pts;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    void operator()(sycl::id<1> id, Inputs in, Outputs out) const {
        auto  b = in.pts.block(id[0]);
        float s = 0.0f;
        for (std::size_t q = 0; q < b.size(); ++q) s += b[q];
        out.sums[id] = s;
    }
};
class SumT : public cs::Component<SumTBody> {
    cs::RaggedInput<DType::f32, 3> pts_;
    cs::Output<DType::f32, 1>      sums_;

public:
    void setup() override {
        declare_input(pts_, "pts", cs::unspecified, cs::unspecified, cs::unspecified);
        declare_output(sums_, "sums", cs::FromInputDim{"pts", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sums_.shape(), {pts_}, {sums_}, {});
    }
};

// ---- tie_outputs top: sz -> gen, then the tie (zero/acc or build/factor), then a sum ---------------------------------
/// shape_mode 0: alias::dims(output_dim(sz.counts, 0), output_values(sz.counts), output_values(sz.counts)) -> [N, m, m]
/// shape_mode 1: alias::from_output_shape(gen.pts) -> [N, m, 3]
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
        cs::AliasShape shape;
        switch (shape_mode_) {
            case 0:  // [N, m, m] from an upstream output
                shape = cs::alias::dims(cs::alias::output_dim("sz.counts", 0), cs::alias::output_values("sz.counts"),
                                        cs::alias::output_values("sz.counts"));
                break;
            case 1: shape = cs::alias::from_output_shape("gen.pts"); break;  // [N, m, 3], ragged upstream shape
            case 2:  // [N, m, m] from a FREE input (its owner, sz, is registered first)
                shape = cs::alias::dims(cs::alias::input_dim("sz.raw", 0), cs::alias::input_values("sz.raw"),
                                        cs::alias::input_values("sz.raw"));
                break;
            default:  // [N, m, m] from a CONNECTED input (gen.counts, connected from sz.counts)
                shape = cs::alias::dims(cs::alias::input_dim("gen.counts", 0), cs::alias::input_values("gen.counts"),
                                        cs::alias::input_values("gen.counts"));
        }
        if constexpr (Atomic)
            tie_outputs<DType::atomic_f32, 3>({"p.t", "q.t"}, "t", std::move(shape));  // require_atomic (the default)
        else
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", std::move(shape), cs::TieOutputsPolicy::allow_plain);
        connect("t", "sum.pts");
    }
};

/// The reference of the tie: block i has counts[i] rows and (shape_mode 0: counts[i], 1: 3) columns.
struct TieRef {
    std::vector<float>         data, sums;
    std::vector<std::uint64_t> offsets{0};
};
TieRef tie_ref(const Counts& counts, int shape_mode, bool factor) {
    TieRef r;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const std::size_t size = counts[i] * (shape_mode == 1 ? 3u : counts[i]);
        float             s    = 0.0f;
        for (std::size_t q = 0; q < size; ++q) {
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

void run_tie_out(cs::Model& m, const Counts& counts, int shape_mode, bool factor) {
    m.update_input("sz.raw", cs::init::from_host(counts));
    m.forward();
    m.wait();
    const TieRef ref  = tie_ref(counts, shape_mode, factor);
    auto         sums = cst::synced_output<DType::f32, 1>(m, "sum.sums");
    REQUIRE(sums.size() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) CHECK(sums.span()[i] == ref.sums[i]);
    auto pts = m.view_ragged_input<DType::f32, 3>("sum.pts");  // the alias's data, through a peer
    pts.sync();
    CHECK(pts.block_offsets() == ref.offsets);
    CHECK(flat_of(pts) == ref.data);
    CHECK(pts.n() == counts.size());
    // block i has counts[i] rows and (shape_mode 1: 3, else counts[i]) columns
    for (std::size_t i = 0; i < counts.size(); ++i) CHECK(pts.block(i).size() == counts[i] * (shape_mode == 1 ? 3u : counts[i]));
}


// ---- tie_inputs top: gen -> (two ragged inputs tied, connected from a ragged output) -----------------------------------------------
template <int K>
struct SumKBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 3> pts;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    void operator()(sycl::id<1> id, Inputs in, Outputs out) const {
        auto  b = in.pts.block(id[0]);
        float s = 0.0f;
        for (std::size_t q = 0; q < b.size(); ++q) s += b[q];
        out.sums[id] = static_cast<float>(K) * s;
    }
};
template <int K>
class SumK : public cs::Component<SumKBody<K>> {
    cs::RaggedInput<DType::f32, 3> pts_;
    cs::Output<DType::f32, 1>      sums_;

public:
    void setup() override {
        this->declare_input(pts_, "pts", cs::unspecified_shape<3>());
        this->declare_output(sums_, "sums", cs::FromInputDim{"pts", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(sums_.shape(), {pts_}, {sums_}, {});
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
        tie_inputs<DType::f32, 3>({"sa.pts", "sb.pts"}, "pts");
        connect("gen.pts", "pts");
    }
};

// ---- a CONNECTED tie_inputs alias with per-index dims read from input values -----------------------------------------------
/// Sz2: sizes [N, 2] (u32), a free input (init from the host), copied.
struct Sz2Body {
    struct Inputs {
        cs::Input<DType::u32, 2> raw;
    };
    struct Outputs {
        cs::Output<DType::u32, 2> out;
    };
    void operator()(sycl::id<2> i, Inputs a, Outputs o) const { o.out[i] = a.raw[i]; }
};
class Sz2 : public cs::Component<Sz2Body> {
    Counts                    init_;
    cs::Input<DType::u32, 2>  raw_;
    cs::Output<DType::u32, 2> out_;

public:
    explicit Sz2(Counts init) : init_{std::move(init)} {}
    void setup() override {
        declare_input(raw_, "raw", init_.size() / 2, 2);
        init_input("raw", cs::init::from_host(init_));
        declare_output(out_, "out", cs::FromInputDim{"raw", 0}, 2);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {raw_}, {out_}, {});
    }
};

/// GenCols: pts [N, sizes[i, 1], 3], pts.block(i)[q] = i * 1000 + q.
struct GenColsBody {
    struct Inputs {
        cs::Input<DType::u32, 2> sizes;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> pts;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs o) const {
        auto b = o.pts.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = static_cast<float>(id[0] * 1000 + q);
    }
};
class GenCols : public cs::Component<GenColsBody> {
    cs::Input<DType::u32, 2>        sizes_;
    cs::RaggedOutput<DType::f32, 3> pts_;

public:
    void setup() override {
        declare_input(sizes_, "sizes", cs::unspecified, cs::unspecified);
        declare_output(pts_, "pts", cs::FromInputDim{"sizes", 0}, cs::FromInputValues{"sizes", 1}, 3);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{sizes_.shape()[0]}, {sizes_}, {pts_}, {});
    }
};

/// sz -> gen (ragged [N, sizes[i,1], 3]) -> two RaggedInputs tied, whose alias is
/// dims(input_dim(gen.sizes, 0), input_values(gen.sizes, `col`), 3), connected from gen.pts.
class TieInColTop : public cs::System {
    Counts      init_;
    std::size_t col_;

public:
    TieInColTop(Counts init, std::size_t col) : init_{std::move(init)}, col_{col} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Sz2>("sz", init_));
        add_subsystem(cs::subsystem_desc<GenCols>("gen"));
        add_subsystem(cs::subsystem_desc<SumK<1>>("sa"));
        add_subsystem(cs::subsystem_desc<SumK<2>>("sb"));
        connect("sz.out", "gen.sizes");
        tie_inputs<DType::f32, 3>({"sa.pts", "sb.pts"}, "pts",
                                  cs::alias::dims(cs::alias::input_dim("gen.sizes", 0), cs::alias::input_values("gen.sizes", col_), 3));
        connect("gen.pts", "pts");
    }
};

// ---- Src: a plain source, x[i] = 0.5 i + 1 -----------------------------------------------------------------------
struct SrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> x;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const { o.x[i] = 0.5f * static_cast<float>(i[0]) + 1.0f; }
};
class Src : public cs::Component<SrcBody> {
    std::size_t               n_;
    cs::Output<DType::f32, 1> x_;

public:
    explicit Src(std::size_t n) : n_{n} {}
    void setup() override { declare_output(x_, "x", n_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {}, {x_}, {});
    }
};

/// BuildX: t = in[i] (q + 1), a tied output; reads the outside input `in`.
struct BuildXBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = a.in[id] * static_cast<float>(q + 1);
    }
};
class BuildX : public cs::Component<BuildXBody> {
    cs::Input<DType::f32, 1>        in_;
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_output(t_, "t", cs::unspecified_shape<3>());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{t_.size()}, {in_}, {t_}, {});
    }
};

struct ConsBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.out[i] = 2.0f * a.in[i] + 1.0f; }
};
class Cons : public cs::Component<ConsBody> {
    cs::Input<DType::f32, 1>  in_;
    cs::Output<DType::f32, 1> out_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_output(out_, "out", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {in_}, {out_}, {});
    }
};

/// g: build (t = in (q+1)) -> factor (t = 2 t + 1, in place) -> sum; the tie's alias is [N, per_index(m), per_index(m)].
class G : public cs::System {
    std::size_t n_;
    Sizes       sizes_;

public:
    G(std::size_t n, Sizes sizes) : n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<BuildX>("p"));
        add_subsystem(cs::subsystem_desc<Factor>("q"));
        add_subsystem(cs::subsystem_desc<SumT>("s"));
        tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(n_, cs::per_index(sizes_), cs::per_index(sizes_)),
                                   cs::TieOutputsPolicy::allow_plain);
        depends_on("q", "p");  // tied writers never wait on each other (section 3): the in-place transform needs the build
        connect("t", "s.pts");
    }
};
/// The interface of the nested System: its promotions.
class Nested : public cs::System {
    std::size_t n_;
    Sizes       sizes_;

public:
    Nested(std::size_t n, Sizes sizes) : n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<G>("g", n_, sizes_));
        promote<DType::f32, 1>("g.p.in", "in");
        promote<DType::f32, 1>("g.s.sums", "sums");
    }
};
class NestedTop : public cs::System {
    std::size_t n_;
    Sizes       sizes_;

public:
    NestedTop(std::size_t n, Sizes sizes) : n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Src>("src", n_));
        add_subsystem(cs::subsystem_desc<Nested>("inner", n_, sizes_));
        add_subsystem(cs::subsystem_desc<Cons>("cons"));
        connect("src.x", "inner.in");
        connect("inner.sums", "cons.in");
    }
};

/// One forward, then cons.out (synced).
std::vector<float> forward_out(cs::Model& m) {
    m.forward();
    m.wait();
    const auto v = cst::synced_output<DType::f32, 1>(m, "cons.out");
    return std::vector<float>(v.span().begin(), v.span().end());
}

std::vector<float> nested_ref(const Sizes& sizes) {
    std::vector<float> out;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const float x = 0.5f * static_cast<float>(i) + 1.0f;
        float       s = 0.0f;
        for (std::size_t q = 0; q < sizes[i] * sizes[i]; ++q) s += 2.0f * (x * static_cast<float>(q + 1)) + 1.0f;
        out.push_back(2.0f * s + 1.0f);
    }
    return out;
}

// ---- alias shapes from INPUTS, and shape inheritance from internal entries, in a nested System ------------------------
/// BuildS: t = in (q + 1); also has the u32 input `sizes` (promoted by G; the alias reads its values).
struct BuildSBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
        cs::Input<DType::u32, 1> sizes;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = a.in[id] * static_cast<float>(q + 1);
    }
};
class BuildS : public cs::Component<BuildSBody> {
    cs::Input<DType::f32, 1>        in_;
    cs::Input<DType::u32, 1>        sizes_;
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_input(sizes_, "sizes", cs::unspecified);
        declare_output(t_, "t", cs::unspecified_shape<3>());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{t_.size()}, {in_, sizes_}, {t_}, {});
    }
};

/// BuildP: t = in (q + 1); also has a RAGGED input `pts` (promoted by G; the alias inherits its shape).
struct BuildPBody {
    struct Inputs {
        cs::Input<DType::f32, 1>       in;
        cs::RaggedInput<DType::f32, 3> pts;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = a.in[id] * static_cast<float>(q + 1);
    }
};
class BuildP : public cs::Component<BuildPBody> {
    cs::Input<DType::f32, 1>        in_;
    cs::RaggedInput<DType::f32, 3>  pts_;
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_input(pts_, "pts", cs::unspecified, cs::unspecified, cs::unspecified);
        declare_output(t_, "t", cs::unspecified_shape<3>());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{t_.size()}, {in_, pts_}, {t_}, {});
    }
};

/// Mk: an INTERNAL ragged output r [N, sizes[i], 3] (shape from the values of a boundary input), r = sizes[i].
struct MkBody {
    struct Inputs {
        cs::Input<DType::u32, 1> sizes;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> r;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto b = o.r.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] = static_cast<float>(a.sizes[id]);
    }
};
class Mk : public cs::Component<MkBody> {
    cs::Input<DType::u32, 1>        sizes_;
    cs::RaggedOutput<DType::f32, 3> r_;

public:
    void setup() override {
        declare_input(sizes_, "sizes", cs::unspecified);
        declare_output(r_, "r", cs::FromInputDim{"sizes", 0}, cs::FromInputValues{"sizes"}, 3);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sizes_.shape(), {sizes_}, {r_}, {});
    }
};

/// AccX: an ATOMIC ragged tie member: t += in (q + 1).
struct AccXBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::RaggedOutput<DType::atomic_f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q].fetch_add(a.in[id] * static_cast<float>(q + 1));
    }
};
class AccX : public cs::Component<AccXBody> {
    cs::Input<DType::f32, 1>               in_;
    cs::RaggedOutput<DType::atomic_f32, 3> t_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_output(t_, "t", cs::unspecified_shape<3>());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{t_.size()}, {in_}, {t_}, {});
    }
};

/// The system of one variant (v):
///   0  alias [N, m, m] from alias::input_dim / input_values of the promoted input `sizes` (u32 [N])
///   1  alias [N, m, 3] from alias::from_input_shape of the promoted RAGGED input `pts`
///   2  alias [N, m, m] from alias::input_dim of `pts`
///   3  alias [N, m, m] from alias::output_dim of the INTERNAL ragged output mk.r (shape inheritance)
///   4  ATOMIC members (zero, then AccX), alias from per_index; require_atomic (the default)
class GV : public cs::System {
    int         v_;
    std::size_t n_;
    Sizes       sizes_;

public:
    GV(int v, std::size_t n, Sizes sizes) : v_{v}, n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        const auto policy = cs::TieOutputsPolicy::allow_plain;
        if (v_ == 0) {
            add_subsystem(cs::subsystem_desc<BuildS>("p"));
            promote<DType::u32, 1>("p.sizes", "sizes");
        } else if (v_ == 1 || v_ == 2) {
            add_subsystem(cs::subsystem_desc<BuildP>("p"));
            promote<DType::f32, 3>("p.pts", "pts");
        } else if (v_ == 3) {
            add_subsystem(cs::subsystem_desc<Mk>("mk"));
            add_subsystem(cs::subsystem_desc<BuildX>("p"));
        } else {
            add_subsystem(cs::subsystem_desc<ZeroA>("p"));
        }
        if (v_ == 4)
            add_subsystem(cs::subsystem_desc<AccX>("q"));
        else
            add_subsystem(cs::subsystem_desc<Factor>("q"));
        add_subsystem(cs::subsystem_desc<SumT>("s"));
        depends_on("q", "p");  // tied writers never wait on each other (section 3): q needs what p wrote
        switch (v_) {
            case 0:
                tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t",
                                           cs::alias::dims(cs::alias::input_dim("sizes", 0), cs::alias::input_values("sizes"),
                                                           cs::alias::input_values("sizes")),
                                           policy);
                break;
            case 1: tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::from_input_shape("pts"), policy); break;
            case 2:
                tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t",
                                           cs::alias::dims(cs::alias::input_dim("pts", 0), cs::alias::input_dim("pts", 1),
                                                           cs::alias::input_dim("pts", 1)),
                                           policy);
                break;
            case 3:
                tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t",
                                           cs::alias::dims(cs::alias::output_dim("mk.r", 0), cs::alias::output_dim("mk.r", 1),
                                                           cs::alias::output_dim("mk.r", 1)),
                                           policy);
                break;
            default:
                tie_outputs<DType::atomic_f32, 3>({"p.t", "q.t"}, "t",
                                                  cs::alias::dims(n_, cs::per_index(sizes_), cs::per_index(sizes_)));
        }
        connect("t", "s.pts");
    }
};
/// The interface of the variant's System: the promotions its wiring needs.
void promote_v(cs::System& s, int v) {
    s.promote<DType::f32, 1>(v == 4 ? "g.q.in" : "g.p.in", "in");
    if (v == 0) s.promote<DType::u32, 1>("g.sizes", "sizes");
    if (v == 1 || v == 2) s.promote<DType::f32, 3>("g.pts", "pts");
    if (v == 3) s.promote<DType::u32, 1>("g.mk.sizes", "mk_sizes");
    s.promote<DType::f32, 1>("g.s.sums", "sums");
}
class InnerV : public cs::System {
    int         v_;
    std::size_t n_;
    Sizes       sizes_;

public:
    InnerV(int v, std::size_t n, Sizes sizes) : v_{v}, n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<GV>("g", v_, n_, sizes_));
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
        add_subsystem(cs::subsystem_desc<Gen>("gen"));
        add_subsystem(cs::subsystem_desc<Src>("src", n));
        add_subsystem(cs::subsystem_desc<InnerV>("inner", v_, n, sizes));
        add_subsystem(cs::subsystem_desc<Cons>("cons"));
        connect("sz.counts", "gen.counts");
        connect("src.x", "inner.in");
        if (v_ == 0) connect("sz.counts", "inner.sizes");
        if (v_ == 1 || v_ == 2) connect("gen.pts", "inner.pts");
        if (v_ == 3) connect("sz.counts", "inner.mk_sizes");
        connect("inner.sums", "cons.in");
    }
};

/// forward -> cons.out of the variant: block i holds x_i (q + 1), then (variants 0-3) 2 t + 1, summed; cons: 2 s + 1.
std::vector<float> variant_ref(const Sizes& sizes, int v) {
    std::vector<float> out;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const float       x    = 0.5f * static_cast<float>(i) + 1.0f;
        const std::size_t size = sizes[i] * (v == 1 ? 3u : sizes[i]);
        float             s    = 0.0f;
        for (std::size_t q = 0; q < size; ++q) {
            const float base = x * static_cast<float>(q + 1);
            s += v == 4 ? base : 2.0f * base + 1.0f;
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
        add_subsystem(cs::subsystem_desc<Zero>("p"));
        add_subsystem(cs::subsystem_desc<Acc>("q"));
        const auto policy = cs::TieOutputsPolicy::allow_plain;
        if (variant_ == 0)  // a per-index spec on dim 0
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(cs::per_index(Sizes{1, 2}), 2, 2), policy);
        else if (variant_ == 1)  // a fixed dim 0 of 4 and a per-index array of 2
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(4, cs::per_index(Sizes{1, 2}), 2), policy);
        else if (variant_ == 2)  // dim 0 is 4 (from sz.counts) at run time; the per-index array has 2
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t",
                                       cs::alias::dims(cs::alias::output_dim("sz.counts", 0), cs::per_index(Sizes{1, 2}), 2), policy);
        else if (variant_ == 3)  // input_values names an OUTPUT
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(4, cs::alias::input_values("sz.counts"), 2), policy);
        else {  // a FREE input whose owner (late) is registered after the tie's first member
            add_subsystem(cs::subsystem_desc<CProd>("late", Counts{1, 2, 3, 4}));
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(4, cs::alias::input_values("late.raw"), 2), policy);
        }
    }
};

/// AccIn: t += in[i] (q + 1); `t` is a tied ragged output, `in` a plain input.
struct AccInBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs a, Outputs o) const {
        auto b = o.t.block(id[0]);
        for (std::size_t q = 0; q < b.size(); ++q) b[q] += a.in[id] * static_cast<float>(q + 1);
    }
};
class AccIn : public cs::Component<AccInBody> {
    cs::Input<DType::f32, 1>        in_;
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_output(t_, "t", cs::unspecified_shape<3>());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{t_.size()}, {in_}, {t_}, {});
    }
};

// ---- an outside tie over PROMOTIONS of inside ports with unspecified dims ----------------------------------------------------------------
class OtInner : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<AccIn>("q"));
        promote<DType::f32, 1>("q.in", "in");
        promote<DType::f32, 3>("q.t", "qt");  // a promotion of an inside output with unspecified dims: the outside tie's member
    }
};
/// tie_outputs over {outside writer w.t (zeroes), inner.qt (the inside accumulator)}, sized from sz.counts.
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
        tie_outputs<DType::f32, 3>({"w.t", "inner.qt"}, "t",
                                   cs::alias::dims(cs::alias::output_dim("sz.counts", 0), cs::alias::output_values("sz.counts"),
                                                   cs::alias::output_values("sz.counts")),
                                   cs::TieOutputsPolicy::allow_plain);
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
        promote<DType::f32, 3>("sa.pts", "pts");
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
        tie_inputs<DType::f32, 3>({"inner.pts", "sb.pts"}, "tp");
        connect("gen.pts", "tp");
        connect("inner.sums", "c1.in");
        connect("sb.sums", "c2.in");
    }
};

}  // namespace trt

TEST_CASE("Ragged ties: tie_outputs of two ragged members (zero, then accumulate), sizes from an upstream output or an input",
          "[ragged][tie]") {
    for (const int shape_mode : {0, 1, 2, 3}) {
        INFO("shape_mode " << shape_mode);
        cs::Model m = cs::create_model<trt::TieOutTop<trt::Zero, trt::Acc>>(default_config(), trt::Counts{3, 0, 2, 4}, shape_mode);
        m.setup();
        trt::run_tie_out(m, {3, 0, 2, 4}, shape_mode, false);  // a zero in the middle
        trt::run_tie_out(m, {0, 5, 1, 2}, shape_mode, false);  // an empty first block, growth
        trt::run_tie_out(m, {0, 0, 0, 0}, shape_mode, false);  // all empty
        trt::run_tie_out(m, {2, 2, 2, 1}, shape_mode, false);  // and back
    }
}

TEST_CASE("Ragged ties: a plain tie whose second member transforms the first one's output in place", "[ragged][tie]") {
    for (const int shape_mode : {0, 1}) {
        INFO("shape_mode " << shape_mode);
        cs::Model m = cs::create_model<trt::TieOutTop<trt::Build0, trt::Factor>>(default_config(), trt::Counts{1, 3, 0, 2}, shape_mode);
        m.setup();
        trt::run_tie_out(m, {1, 3, 0, 2}, shape_mode, true);
        trt::run_tie_out(m, {4, 0, 2, 2}, shape_mode, true);
    }
}

TEST_CASE("Ragged ties: a connected tie_inputs alias with dims(input_dim, input_values, 3) checks the per-index arrays",
          "[ragged][tie]") {
    SECTION("matching per-index extents, two forwards with changing sizes") {
        cs::Model m = cs::create_model<trt::TieInColTop>(default_config(), trt::Counts{2, 2, 0, 0, 3, 3, 1, 1}, std::size_t{1});
        m.setup();
        for (const trt::Counts& sizes : {trt::Counts{2, 2, 0, 0, 3, 3, 1, 1}, trt::Counts{1, 4, 0, 5, 2, 2, 3, 0}}) {
            m.update_input("sz.raw", cs::init::from_host(sizes));
            m.forward();
            m.wait();
            auto sa = cst::synced_output<DType::f32, 1>(m, "sa.sums");
            auto sb = cst::synced_output<DType::f32, 1>(m, "sb.sums");
            REQUIRE(sa.size() == 4);
            for (std::size_t i = 0; i < 4; ++i) {
                float want = 0.0f;
                for (std::size_t q = 0; q < sizes[2 * i + 1] * 3u; ++q) want += static_cast<float>(i * 1000 + q);
                CHECK(sa.span()[i] == want);
                CHECK(sb.span()[i] == 2.0f * want);
            }
            auto pa = m.view_ragged_input<DType::f32, 3>("sa.pts");
            pa.sync();
            REQUIRE(pa.n() == 4);
            for (std::size_t i = 0; i < 4; ++i) CHECK(pa.block(i).size() == sizes[2 * i + 1] * 3u);  // column 1 of sizes, times 3
        }
    }
    SECTION("a mismatching per-index array is an error at forward") {
        // the alias reads column 0 ([2,0,3,1]); gen.pts has column 1 ([2,0,1,1]): they differ first at index 2. The connect
        // makes gen.pts's variable the survivor and the tie's rule a constraint (design review, section 9); the mismatch
        // names both shapes, per-index extents included.
        cs::Model m = cs::create_model<trt::TieInColTop>(default_config(), trt::Counts{2, 2, 0, 0, 3, 1, 1, 1}, std::size_t{0});
        m.setup();
        trt::expect_throws([&] { m.forward(); }, {"tie alias 'pts' declares the shape 4x[2,0,3,1]x3",
                                                  "but is connected to a source whose shape is 4x[2,0,1,1]x3"});
        m.wait();
    }
}

TEST_CASE("Ragged ties: tie_inputs of two RaggedInputs, connected from a ragged output", "[ragged][tie]") {
    cs::Model m = cs::create_model<trt::TieInTop>(default_config(), trt::Counts{2, 0, 3, 1});
    m.setup();
    for (const trt::Counts& counts : {trt::Counts{2, 0, 3, 1}, trt::Counts{0, 4, 0, 2}, trt::Counts{1, 1, 1, 1}}) {
        m.update_input("sz.raw", cs::init::from_host(counts));
        m.forward();
        m.wait();
        auto sa = cst::synced_output<DType::f32, 1>(m, "sa.sums");
        auto sb = cst::synced_output<DType::f32, 1>(m, "sb.sums");
        REQUIRE(sa.size() == counts.size());
        REQUIRE(sb.size() == counts.size());
        for (std::size_t i = 0; i < counts.size(); ++i) {
            float want = 0.0f;
            for (std::size_t q = 0; q < counts[i] * 3u; ++q) want += static_cast<float>(i * 1000 + q);
            CHECK(sa.span()[i] == want);
            CHECK(sb.span()[i] == 2.0f * want);
        }
        auto pa = m.view_ragged_input<DType::f32, 3>("sa.pts");
        pa.sync();
        REQUIRE(pa.n() == counts.size());
        for (std::size_t i = 0; i < counts.size(); ++i) CHECK(pa.block(i).size() == counts[i] * 3u);  // the per-index dim, as sizes
    }
}

TEST_CASE("Ragged ties: a tie with cs::per_index host arrays in a nested System, values and a second forward",
          "[ragged][tie]") {
    // The plain half of the old "chunked vs plain" cases; the chunked half (budgets, allocation counts) is deleted.
    const trt::Sizes sizes{1, 3, 2, 3, 0, 3, 1, 2};
    const std::size_t n    = sizes.size();
    const auto        want = trt::nested_ref(sizes);
    cs::Model         m    = cs::create_model<trt::NestedTop>(default_config(), n, sizes);
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
        trt::run_tie_out(m, {3, 0, 2, 4}, shape_mode, false);
        trt::run_tie_out(m, {0, 5, 1, 2}, shape_mode, false);
        trt::run_tie_out(m, {2, 2, 2, 1}, shape_mode, false);
    }
}

TEST_CASE("Ragged ties: alias shapes from inputs and from internal entries (variants 0-4), in a nested System", "[ragged][tie]") {
    // The plain half of the old "alias shapes ... inside a ChunkedSystem" and "Chunked vs plain: tie aliases" cases.
    const trt::Counts counts{1, 3, 2, 3, 0, 3, 1, 2};
    const trt::Sizes  sizes(counts.begin(), counts.end());
    for (int v = 0; v <= 4; ++v) {
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
    SECTION("a per-index spec on dimension 0") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 0);
        trt::expect_throws([&] { m.setup(); }, {"alias::dims, dimension 0", "per-index spec is for dimensions >= 1"});
    }
    SECTION("a per-index array that does not match a fixed dimension 0") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 1);
        trt::expect_throws([&] { m.setup(); }, {"per_index has 2 extents", "dimension 0 is 4"});
    }
    SECTION("a per-index array that does not match dimension 0 at run time") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 2);
        m.setup();
        trt::expect_throws([&] { m.forward(); }, {"has 2 extents", "dimension 0 has 4"});
        m.wait();  // earlier components of the failed forward were submitted: wait before the Model is destroyed
    }
    SECTION("input_values naming an output") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 3);
        trt::expect_throws([&] { m.setup(); }, {"alias::dims, dimension 1", "is not an input"});
    }
    SECTION("a free input whose owner is registered after the alias's first member is fine: it is materialized on need") {
        // the old 'registered after' setup error is gone with the owner concept: whoever reads the input first materializes it
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 4);
        CHECK_NOTHROW(m.setup());
    }
}

TEST_CASE("Ragged ties: an outside tie_outputs over a promotion of an inside output plus an outside writer", "[ragged][tie]") {
    // The plain half of the old "Chunked vs plain: an outside tie_outputs over a promotion ..." case, with a host reference.
    // REVIEW: the accumulating System `inner` is ordered after the zeroing writer `w` by depends_on("inner", "w") on a System
    // child (not a component). system.hpp's depends_on and Subsystem::add_dependencies are assumed to forward to the System's
    // children; no header here states it.
    const trt::Counts counts{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    cs::Model         m = cs::create_model<trt::OtTop>(default_config(), counts);
    m.setup();
    m.forward();
    m.wait();
    std::vector<float>         want_flat, want_cons;
    std::vector<std::uint64_t> want_offs{0};
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const float x = 0.5f * static_cast<float>(i) + 1.0f;  // Src's x[i]
        float       s = 0.0f;
        for (std::size_t q = 0; q < counts[i] * counts[i]; ++q) {
            want_flat.push_back(x * static_cast<float>(q + 1));
            s += want_flat.back();
        }
        want_offs.push_back(want_flat.size());
        want_cons.push_back(2.0f * s + 1.0f);
    }
    const auto cons = cst::to_vector(cst::synced_output<DType::f32, 1>(m, "cons.out"));
    CHECK(cst::all_close(cons, want_cons));
    auto t = m.view_ragged_input<DType::f32, 3>("sum.pts");  // the tie alias: the full root
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
        m.forward();
        m.wait();
        for (const char* out : {"c1.out", "c2.out"}) {
            const auto o = cst::to_vector(cst::synced_output<DType::f32, 1>(m, out));
            REQUIRE(o.size() == counts->size());
            for (std::size_t i = 0; i < counts->size(); ++i) {
                float s = 0.0f;
                for (std::size_t q = 0; q < (*counts)[i] * 3u; ++q) s += static_cast<float>(i * 1000 + q);
                CHECK(cst::close(o[i], 2.0f * s + 1.0f));  // Cons: 2 s + 1, both members read the same alias
            }
        }
    }
}
