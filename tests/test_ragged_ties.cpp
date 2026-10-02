// Ragged tie aliases (D140): a tie alias takes the same per-index shape rules as a port.
//
//   alias::dims(spec0, spec1, ...) with an integer, per_index, output_value, output_values, output_dim;
//   from_output_shape copies a ragged shape, and so does a connection (an unspecified dim takes its source's).
//
// Members are RaggedOutput / RaggedInput declared with cs::tied. Plain members use TieOutputsPolicy::allow_plain
// (the producers run in order, they do not race); atomic ragged members take the require_atomic default.
// Alias sources may also be INPUTS of the declaring System (input_value / input_values / input_dim /
// from_input_shape), and a ChunkedSystem refuses only VALUE reads from its internal roots.
//
// Everything is checked against host references.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "chunk_capture.hpp"
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

// ---- tie members: a rank-3 ragged output declared cs::tied -------------------------------------------------------------
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
        add_subsystem<CProd>("sz", init_);
        add_subsystem<Gen>("gen");
        add_subsystem<P>("p");
        add_subsystem<Q>("q");
        add_subsystem<SumT>("sum");
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
    m.init_input("sz.raw", cs::init::from_host(counts));
    m.forward();
    m.wait();
    const TieRef ref  = tie_ref(counts, shape_mode, factor);
    auto         sums = m.view_output<DType::f32, 1>("sum.sums");
    REQUIRE(sums.size() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) CHECK(sums.span()[i] == ref.sums[i]);
    auto pts = m.view_ragged_input<DType::f32, 3>("sum.pts");  // the alias's data, through a peer
    CHECK(pts.offsets == ref.offsets);
    CHECK(pts.data == ref.data);
    CHECK(pts.size() == counts.size());
    CHECK(pts.is_per_index(1));
    if (shape_mode != 1)
        CHECK(pts.is_per_index(2));
    else
        CHECK_FALSE(pts.is_per_index(2));
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
        add_subsystem<CProd>("sz", init_);
        add_subsystem<Gen>("gen");
        add_subsystem<SumK<1>>("sa");
        add_subsystem<SumK<2>>("sb");
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
        add_subsystem<Sz2>("sz", init_);
        add_subsystem<GenCols>("gen");
        add_subsystem<SumK<1>>("sa");
        add_subsystem<SumK<2>>("sb");
        connect("sz.out", "gen.sizes");
        tie_inputs<DType::f32, 3>({"sa.pts", "sb.pts"}, "pts",
                                  cs::alias::dims(cs::alias::input_dim("gen.sizes", 0), cs::alias::input_values("gen.sizes", col_), 3));
        connect("gen.pts", "pts");
    }
};

// ---- a plain tie inside (and outside) a ChunkedSystem ----------------------------------------------------------------------
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
        add_subsystem<BuildX>("p");
        add_subsystem<Factor>("q");
        add_subsystem<SumT>("s");
        tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(n_, cs::per_index(sizes_), cs::per_index(sizes_)),
                                   cs::TieOutputsPolicy::allow_plain);
        connect("t", "s.pts");
    }
};
class InnerPlain : public cs::System {
    std::size_t n_;
    Sizes       sizes_;

public:
    InnerPlain(std::size_t, std::size_t n, Sizes sizes) : n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem<G>("g", n_, sizes_);
        promote<DType::f32, 1>("g.p.in", "in");  // the interface of an opaque chunked system: its promotions
        promote<DType::f32, 1>("g.s.sums", "sums");
    }
};
class InnerChunk : public cs::ChunkedSystem {
    std::size_t n_;
    Sizes       sizes_;

public:
    InnerChunk(std::size_t budget, std::size_t n, Sizes sizes) : cs::ChunkedSystem(budget), n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem<G>("g", n_, sizes_);
        promote<DType::f32, 1>("g.p.in", "in");
        promote<DType::f32, 1>("g.s.sums", "sums");
    }
};
template <class Inner>
class ChunkTop : public cs::System {
    std::size_t n_, budget_;
    Sizes       sizes_;

public:
    ChunkTop(std::size_t n, std::size_t budget, Sizes sizes) : n_{n}, budget_{budget}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem<Src>("src", n_);
        add_subsystem<Inner>("inner", budget_, n_, sizes_);
        add_subsystem<Cons>("cons");
        connect("src.x", "inner.in");
        connect("inner.sums", "cons.in");
    }
};

/// An entry INSIDE the chunked system 'inner', resolved by the chunked system itself (from outside only its promotions
/// are visible).
cs::detail::IOEntry& inner_entry(cs::Model& m, const char* path) {
    return dynamic_cast<cs::ChunkedSystem&>(m.get_subsystem("inner")).entry_at(path);
}

std::vector<float> forward_out(cs::Model& m) {
    m.forward();
    m.wait();
    auto v = m.view_output<DType::f32, 1>("cons.out");
    return std::vector<float>(v.span().begin(), v.span().end());
}

std::vector<float> chunk_ref(const Sizes& sizes) {
    std::vector<float> out;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const float x = 0.5f * static_cast<float>(i) + 1.0f;
        float       s = 0.0f;
        for (std::size_t q = 0; q < sizes[i] * sizes[i]; ++q) s += 2.0f * (x * static_cast<float>(q + 1)) + 1.0f;
        out.push_back(2.0f * s + 1.0f);
    }
    return out;
}

// ---- a tie_inputs alias (tied RaggedInput members) INSIDE a ChunkedSystem, connected from an outside ragged output --------
/// Copy3: cp = 2 pts, an INTERNAL ragged output of a chunked system; `pts` is a tied member.
struct Copy3Body {
    struct Inputs {
        cs::RaggedInput<DType::f32, 3> pts;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> cp;
    };
    void operator()(sycl::id<1> id, Inputs in, Outputs out) const {
        auto a = in.pts.block(id[0]);
        auto b = out.cp.block(id[0]);
        for (std::size_t q = 0; q < a.size(); ++q) b[q] = 2.0f * a[q];
    }
};
class Copy3 : public cs::Component<Copy3Body> {
    cs::RaggedInput<DType::f32, 3>  pts_;
    cs::RaggedOutput<DType::f32, 3> cp_;

public:
    void setup() override {
        declare_input(pts_, "pts", cs::unspecified_shape<3>());
        declare_output(cp_, "cp", cs::FromInputDim{"pts", 0}, cs::FromInputDim{"pts", 1}, cs::FromInputDim{"pts", 2});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{pts_.size()}, {pts_}, {cp_}, {});
    }
};
/// sa (sums, a tied member) and cp (a tied member) share the alias `pts`; cp's output feeds st.
class TieChunkG : public cs::System {
public:
    void setup() override {
        add_subsystem<SumK<1>>("sa");
        add_subsystem<Copy3>("cp");
        add_subsystem<SumT>("st");
        tie_inputs<DType::f32, 3>({"sa.pts", "cp.pts"}, "pts");
        connect("cp.cp", "st.pts");
    }
};
class TieChunkPlain : public cs::System {
public:
    explicit TieChunkPlain(std::size_t) {}
    void setup() override {
        add_subsystem<TieChunkG>("g");
        promote<DType::f32, 3>("g.pts", "pts");  // the tie alias, the interface of an opaque chunked system
        promote<DType::f32, 1>("g.sa.sums", "sa_sums");
        promote<DType::f32, 1>("g.st.sums", "st_sums");
    }
};
class TieChunkChunked : public cs::ChunkedSystem {
public:
    explicit TieChunkChunked(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override {
        add_subsystem<TieChunkG>("g");
        promote<DType::f32, 3>("g.pts", "pts");
        promote<DType::f32, 1>("g.sa.sums", "sa_sums");
        promote<DType::f32, 1>("g.st.sums", "st_sums");
    }
};
template <class Inner>
class TieChunkTop : public cs::System {
    Counts      init_;
    std::size_t budget_;

public:
    TieChunkTop(Counts init, std::size_t budget) : init_{std::move(init)}, budget_{budget} {}
    void setup() override {
        add_subsystem<CProd>("sz", init_);
        add_subsystem<Gen>("gen");
        add_subsystem<Inner>("inner", budget_);
        add_subsystem<Cons>("c1");
        add_subsystem<Cons>("c2");
        connect("sz.counts", "gen.counts");
        connect("gen.pts", "inner.pts");
        connect("inner.sa_sums", "c1.in");
        connect("inner.st_sums", "c2.in");
    }
};

// ---- alias shapes from INPUTS, and shape inheritance from internal entries, inside a ChunkedSystem ------------------------
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
            add_subsystem<BuildS>("p");
            promote<DType::u32, 1>("p.sizes", "sizes");
        } else if (v_ == 1 || v_ == 2) {
            add_subsystem<BuildP>("p");
            promote<DType::f32, 3>("p.pts", "pts");
        } else if (v_ == 3) {
            add_subsystem<Mk>("mk");
            add_subsystem<BuildX>("p");
        } else {
            add_subsystem<ZeroA>("p");
        }
        if (v_ == 4)
            add_subsystem<AccX>("q");
        else
            add_subsystem<Factor>("q");
        add_subsystem<SumT>("s");
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
/// The interface of an opaque chunked system (and of its plain twin): the promotions the variant's wiring needs.
void promote_v(cs::System& s, int v) {
    s.promote<DType::f32, 1>(v == 4 ? "g.q.in" : "g.p.in", "in");
    if (v == 0) s.promote<DType::u32, 1>("g.sizes", "sizes");
    if (v == 1 || v == 2) s.promote<DType::f32, 3>("g.pts", "pts");
    if (v == 3) s.promote<DType::u32, 1>("g.mk.sizes", "mk_sizes");
    s.promote<DType::f32, 1>("g.s.sums", "sums");
}
class InnerPlainV : public cs::System {
    int         v_;
    std::size_t n_;
    Sizes       sizes_;

public:
    InnerPlainV(std::size_t, int v, std::size_t n, Sizes sizes) : v_{v}, n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem<GV>("g", v_, n_, sizes_);
        promote_v(*this, v_);
    }
};
class InnerChunkV : public cs::ChunkedSystem {
    int         v_;
    std::size_t n_;
    Sizes       sizes_;

public:
    InnerChunkV(std::size_t budget, int v, std::size_t n, Sizes sizes)
        : cs::ChunkedSystem(budget), v_{v}, n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        add_subsystem<GV>("g", v_, n_, sizes_);
        promote_v(*this, v_);
    }
};
template <class Inner>
class VTop : public cs::System {
    Counts      counts_;
    std::size_t budget_;
    int         v_;

public:
    VTop(Counts counts, std::size_t budget, int v) : counts_{std::move(counts)}, budget_{budget}, v_{v} {}
    void setup() override {
        const std::size_t n = counts_.size();
        Sizes             sizes(counts_.begin(), counts_.end());
        add_subsystem<CProd>("sz", counts_);
        add_subsystem<Gen>("gen");
        add_subsystem<Src>("src", n);
        add_subsystem<Inner>("inner", budget_, v_, n, sizes);
        add_subsystem<Cons>("cons");
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
        add_subsystem<trt::CProd>("sz", trt::Counts{1, 2, 3, 4});
        add_subsystem<Zero>("p");
        add_subsystem<Acc>("q");
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
            add_subsystem<CProd>("late", Counts{1, 2, 3, 4});
            tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(4, cs::alias::input_values("late.raw"), 2), policy);
        }
    }
};

struct SrcUBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::u32, 1> x;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const { o.x[i] = static_cast<std::uint32_t>(i[0] % 3 + 1); }
};
class SrcU : public cs::Component<SrcUBody> {
    cs::Output<DType::u32, 1> x_;

public:
    void setup() override { declare_output(x_, "x", 6); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {}, {x_}, {});
    }
};
struct CopyUBody {
    struct Inputs {
        cs::Input<DType::u32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> counts;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.counts[i] = a.in[i]; }
};
class CopyU : public cs::Component<CopyUBody> {
    cs::Input<DType::u32, 1>  in_;
    cs::Output<DType::u32, 1> counts_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified);
        declare_output(counts_, "counts", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {in_}, {counts_}, {});
    }
};
/// The tie's per-index source (cc.counts) is computed INSIDE the chunked system.
class GBad : public cs::System {
public:
    void setup() override {
        add_subsystem<CopyU>("cc");
        add_subsystem<Zero>("p");
        add_subsystem<Acc>("q");
        tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t", cs::alias::dims(6, cs::alias::output_values("cc.counts"), 2),
                                   cs::TieOutputsPolicy::allow_plain);
    }
};
class BadInner : public cs::ChunkedSystem {
public:
    BadInner() : cs::ChunkedSystem(1000) {}
    void setup() override {
        add_subsystem<GBad>("g");
        promote<DType::u32, 1>("g.cc.in", "in");
    }
};
class BadTop : public cs::System {
public:
    void setup() override {
        add_subsystem<SrcU>("src");
        add_subsystem<BadInner>("inner");
        connect("src.x", "inner.in");
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
            m.init_input("sz.raw", cs::init::from_host(sizes));
            m.forward();
            m.wait();
            auto sa = m.view_output<DType::f32, 1>("sa.sums");
            auto sb = m.view_output<DType::f32, 1>("sb.sums");
            REQUIRE(sa.size() == 4);
            for (std::size_t i = 0; i < 4; ++i) {
                float want = 0.0f;
                for (std::size_t q = 0; q < sizes[2 * i + 1] * 3u; ++q) want += static_cast<float>(i * 1000 + q);
                CHECK(sa.span()[i] == want);
                CHECK(sb.span()[i] == 2.0f * want);
            }
            auto pa = m.view_ragged_input<DType::f32, 3>("sa.pts");
            trt::Sizes col1;
            for (std::size_t i = 0; i < 4; ++i) col1.push_back(sizes[2 * i + 1]);
            CHECK(pa.extents(1) == col1);
        }
    }
    SECTION("a mismatching per-index array names the index") {
        // the alias reads column 0 ([2,0,3,1]); gen.pts has column 1 ([2,0,1,1]): they differ first at index 2
        cs::Model m = cs::create_model<trt::TieInColTop>(default_config(), trt::Counts{2, 2, 0, 0, 3, 1, 1, 1}, std::size_t{0});
        m.setup();
        trt::expect_throws([&] { m.forward(); }, {"tie alias 'pts'", "per-index dimension 1 differs at index 2",
                                                  "the alias declares 3", "the connection has 1"});
        m.wait();
    }
}

TEST_CASE("Ragged ties: tie_inputs of two RaggedInputs, connected from a ragged output", "[ragged][tie]") {
    cs::Model m = cs::create_model<trt::TieInTop>(default_config(), trt::Counts{2, 0, 3, 1});
    m.setup();
    for (const trt::Counts& counts : {trt::Counts{2, 0, 3, 1}, trt::Counts{0, 4, 0, 2}, trt::Counts{1, 1, 1, 1}}) {
        m.init_input("sz.raw", cs::init::from_host(counts));
        m.forward();
        m.wait();
        auto sa = m.view_output<DType::f32, 1>("sa.sums");
        auto sb = m.view_output<DType::f32, 1>("sb.sums");
        REQUIRE(sa.size() == counts.size());
        REQUIRE(sb.size() == counts.size());
        for (std::size_t i = 0; i < counts.size(); ++i) {
            float want = 0.0f;
            for (std::size_t q = 0; q < counts[i] * 3u; ++q) want += static_cast<float>(i * 1000 + q);
            CHECK(sa.span()[i] == want);
            CHECK(sb.span()[i] == 2.0f * want);
        }
        auto pa = m.view_ragged_input<DType::f32, 3>("sa.pts");
        CHECK(pa.extents(1) == trt::Sizes(counts.begin(), counts.end()));
        CHECK(pa.is_per_index(1));
    }
}

TEST_CASE("Ragged ties: a tie inside a ChunkedSystem, bitwise equal to unchunked, allocated once",
          "[ragged][tie][chunked]") {
    const trt::Sizes sizes{1, 3, 2, 3, 0, 3, 1, 2};
    const std::size_t n = sizes.size();
    const auto        want = trt::chunk_ref(sizes);
    cs::Model plain = cs::create_model<trt::ChunkTop<trt::InnerPlain>>(default_config(), n, std::size_t{0}, sizes);
    plain.setup();
    const auto pout = trt::forward_out(plain);
    REQUIRE(pout.size() == n);
    for (std::size_t i = 0; i < n; ++i) CHECK(pout[i] == want[i]);
    for (const std::size_t budget : {std::size_t{36}, std::size_t{50}, std::size_t{100}, std::size_t{100000}}) {
        INFO("budget " << budget);
        cs::Model m = cs::create_model<trt::ChunkTop<trt::InnerChunk>>(default_config(), n, budget, sizes);
        m.setup();
        cs::detail::IOEntry& t = trt::inner_entry(m, "g.t");  // the internal ragged tie alias root
        for (int rep = 0; rep < 2; ++rep) {
            const std::size_t before = t.allocation_count();
            const auto        out    = trt::forward_out(m);
            CHECK(out == pout);
            CHECK(t.allocation_count() - before == (rep == 0 ? 1u : 0u));  // eternal (D141): the first forward allocates it
            CHECK_FALSE(t.held());
        }
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

TEST_CASE("Ragged ties: alias shapes from inputs and from internal entries inside a ChunkedSystem", "[ragged][tie][chunked]") {
    const trt::Counts counts{1, 3, 2, 3, 0, 3, 1, 2};
    const trt::Sizes  sizes(counts.begin(), counts.end());
    for (int v = 0; v <= 4; ++v) {
        INFO("variant " << v);
        const auto want = trt::variant_ref(sizes, v);
        cs::Model  plain = cs::create_model<trt::VTop<trt::InnerPlainV>>(default_config(), counts, std::size_t{0}, v);
        plain.setup();
        const auto pout = trt::forward_out(plain);
        REQUIRE(pout.size() == counts.size());
        for (std::size_t i = 0; i < pout.size(); ++i) CHECK(pout[i] == want[i]);
        for (const std::size_t budget : {std::size_t{72}, std::size_t{100}, std::size_t{150}, std::size_t{100000}}) {
            INFO("budget " << budget);
            cs::Model m = cs::create_model<trt::VTop<trt::InnerChunkV>>(default_config(), counts, budget, v);
            m.setup();
            cs::detail::IOEntry& t = trt::inner_entry(m, "g.t");  // the internal ragged tie alias root
            for (int rep = 0; rep < 2; ++rep) {
                const std::size_t before = t.allocation_count();
                const auto        out    = trt::forward_out(m);
                CHECK(out == pout);
                CHECK(t.allocation_count() - before == (rep == 0 ? 1u : 0u));
                if (v == 3) {  // the internal ragged OUTPUT the alias inherits its shape from
                    cs::detail::IOEntry& r = trt::inner_entry(m, "g.mk.r");
                    CHECK(r.allocation_count() == 1);
                }
            }
        }
    }
}

TEST_CASE("Ragged ties: tied ragged members of an alias inside a ChunkedSystem slice the per-index arrays per chunk",
          "[ragged][tie][chunked]") {
    // internal bytes per index: cp is [c, 3] floats = 12 c; the greedy scan of the framework
    const auto chunks_for = [](const trt::Counts& counts, std::size_t budget) {
        std::size_t chunks = 0, acc = 0;
        for (const std::uint32_t c : counts) {
            if (acc + 12u * c > budget) {
                ++chunks;
                acc = 0;
            }
            acc += 12u * c;
        }
        return chunks + 1;
    };
    const trt::Counts first{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    const trt::Counts second{2, 0, 5, 5, 1, 1, 3, 4, 0, 2};
    // total 300 B (1 chunk), 150 (about 2), 70 and 60 (many; 60 fits the largest group, c = 5)
    for (const std::size_t budget : {std::size_t{300}, std::size_t{150}, std::size_t{70}, std::size_t{60}}) {
        INFO("budget " << budget);
        cs::Model plain = cs::create_model<trt::TieChunkTop<trt::TieChunkPlain>>(default_config(), first, std::size_t{0});
        plain.setup();
        cs::Model m = cs::create_model<trt::TieChunkTop<trt::TieChunkChunked>>(default_config(), first, budget);
        m.setup();
        for (const trt::Counts* counts : {&first, &second}) {
            plain.init_input("sz.raw", cs::init::from_host(*counts));
            m.init_input("sz.raw", cs::init::from_host(*counts));
            plain.forward();
            plain.wait();
            m.forward();
            m.wait();
            for (const char* out : {"c1.out", "c2.out"}) {
                const auto p = plain.view_output<DType::f32, 1>(out);
                const auto c = m.view_output<DType::f32, 1>(out);
                REQUIRE(p.size() == counts->size());
                REQUIRE(c.size() == counts->size());
                for (std::size_t i = 0; i < counts->size(); ++i) CHECK(c.span()[i] == p.span()[i]);  // bitwise
            }
            // host reference: c1 = 2 sum(pts) + 1, c2 = 2 sum(2 pts) + 1
            const auto c1 = m.view_output<DType::f32, 1>("c1.out");
            const auto c2 = m.view_output<DType::f32, 1>("c2.out");
            for (std::size_t i = 0; i < counts->size(); ++i) {
                float s = 0.0f;
                for (std::size_t q = 0; q < (*counts)[i] * 3u; ++q) s += static_cast<float>(i * 1000 + q);
                CHECK(c1.span()[i] == 2.0f * s + 1.0f);
                CHECK(c2.span()[i] == 2.0f * (2.0f * s) + 1.0f);
            }
            CHECK(dynamic_cast<cs::ChunkedSystem&>(m.get_subsystem("inner")).last_chunk_count() == chunks_for(*counts, budget));
            std::cout << "[obs] tied ragged members chunked: budget " << budget << ", chunks "
                      << dynamic_cast<cs::ChunkedSystem&>(m.get_subsystem("inner")).last_chunk_count() << "\n";
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
        m.wait();  // earlier components of the failed forward were submitted: wait before the Model is destroyed (D135)
    }
    SECTION("input_values naming an output") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 3);
        trt::expect_throws([&] { m.setup(); }, {"alias::dims, dimension 1", "is not an input"});
    }
    SECTION("a free input whose owner is registered after the alias's first member") {
        cs::Model m = cs::create_model<trt::ErrTop>(default_config(), 4);
        trt::expect_throws([&] { m.setup(); }, {"free input", "late.raw", "registered after"});
    }
    SECTION("a per-index source computed inside a chunked system") {
        cs::Model m = cs::create_model<trt::BadTop>(default_config());
        m.setup();
        trt::expect_throws([&] { m.forward(); }, {"is sized by a value computed inside chunked system", "inner"});
        m.wait();
    }
}


// =============================================================================================================================
// The tie aliases inside a ChunkedSystem, against the same models in a plain System.
// =============================================================================================================================
TEST_CASE("Chunked vs plain: tie aliases (input_values, from_input_shape, input_dim, output_dim, per_index) inside a ChunkedSystem",
          "[ragged][tie][chunked][replay]") {
    const trt::Counts first{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    const trt::Counts second{2, 0, 5, 5, 1, 1, 3, 4, 0, 2};
    for (int v = 0; v <= 4; ++v)
        for (const std::size_t budget : {std::size_t{1000}, std::size_t{200}}) {
            INFO("variant " << v << ", budget " << budget);
            tcc::against_plain("inner", [&] { return cs::create_model<trt::VTop<trt::InnerChunkV>>(default_config(), first, budget, v); },
                        [&] { return cs::create_model<trt::VTop<trt::InnerPlainV>>(default_config(), first, std::size_t{0}, v); },
                        [](cs::Model& m) {
                            auto o = m.view_output<DType::f32, 1>("cons.out");
                            return std::vector<float>(o.span().begin(), o.span().end());
                        },
                        3, [&](cs::Model& m, int f) {
                            // variant 4 declares its per-index extents on the host: they cannot change between forwards
                            if (v != 4) m.init_input("sz.raw", cs::init::from_host(f == 1 ? second : first));
                        });
        }
}

TEST_CASE("Chunked vs plain: tied ragged members connected from an outside ragged output",
          "[ragged][tie][chunked][replay]") {
    const trt::Counts first{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    const trt::Counts second{2, 0, 5, 5, 1, 1, 3, 4, 0, 2};
    for (const std::size_t budget : {std::size_t{300}, std::size_t{150}, std::size_t{70}, std::size_t{60}}) {
        INFO("budget " << budget);
        tcc::against_plain("inner", [&] { return cs::create_model<trt::TieChunkTop<trt::TieChunkChunked>>(default_config(), first, budget); },
                    [&] { return cs::create_model<trt::TieChunkTop<trt::TieChunkPlain>>(default_config(), first, std::size_t{0}); },
                    [](cs::Model& m) {
                        std::vector<float> out;
                        for (const char* p : {"c1.out", "c2.out"}) {
                            auto o = m.view_output<DType::f32, 1>(p);
                            out.insert(out.end(), o.span().begin(), o.span().end());
                        }
                        return out;
                    },
                    3, [&](cs::Model& m, int f) { m.init_input("sz.raw", cs::init::from_host(f == 1 ? second : first)); });
    }
}

TEST_CASE("Topology cache: a changed SHAPE of an outside boundary root rebuilds (no value is read inside)",
          "[ragged][tie][chunked][replay][cache]") {
    const trt::Counts first{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    const trt::Counts second{2, 0, 5, 5, 1, 1, 3, 4, 0, 2};
    cs::Model         m = cs::create_model<trt::TieChunkTop<trt::TieChunkChunked>>(default_config(), first, std::size_t{70});
    m.setup();
    const auto run = [&] {
        const auto b = cs::diag::snapshot();
        m.forward();
        m.wait();
        return std::make_pair(cs::diag::snapshot() - b, cs::diag::last_window());
    };
    const auto f1 = run();
    CHECK(f1.first.full_passes == 1);
    const auto f2 = run();  // gen.pts is rewritten (same shape, same data): the producer ran again, nothing else changed
    CHECK(f2.first.full_passes == 0);
    CHECK(f2.second.resolve_shapes == 0);
    CHECK(f2.second.ragged_uploads == 0);
    m.init_input("sz.raw", cs::init::from_host(second));  // gen.pts changes SHAPE
    const auto f3 = run();
    CHECK(f3.first.full_passes == 1);
    CHECK(run().first.full_passes == 0);
}


// ---- aliases declared DIRECTLY in a ChunkedSystem's own setup() (C2c) -----------------------------------------------------
namespace trt {
/// The variant-0 shape of GV, but the promote and the tie live in the chunked system itself (no child system g).
class DirectChunk : public cs::ChunkedSystem {
public:
    explicit DirectChunk(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override {
        add_subsystem<BuildS>("p");
        promote<DType::u32, 1>("p.sizes", "sizes");
        add_subsystem<Factor>("q");
        add_subsystem<SumT>("s");
        tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t",
                                   cs::alias::dims(cs::alias::input_dim("sizes", 0), cs::alias::input_values("sizes"),
                                                   cs::alias::input_values("sizes")),
                                   cs::TieOutputsPolicy::allow_plain);
        connect("t", "s.pts");
        promote<DType::f32, 1>("p.in", "in");
        promote<DType::f32, 1>("s.sums", "sums");
    }
};
class DirectPlain : public cs::System {
public:
    explicit DirectPlain(std::size_t) {}
    void setup() override {
        add_subsystem<BuildS>("p");
        promote<DType::u32, 1>("p.sizes", "sizes");
        add_subsystem<Factor>("q");
        add_subsystem<SumT>("s");
        tie_outputs<DType::f32, 3>({"p.t", "q.t"}, "t",
                                   cs::alias::dims(cs::alias::input_dim("sizes", 0), cs::alias::input_values("sizes"),
                                                   cs::alias::input_values("sizes")),
                                   cs::TieOutputsPolicy::allow_plain);
        connect("t", "s.pts");
        promote<DType::f32, 1>("p.in", "in");
        promote<DType::f32, 1>("s.sums", "sums");
    }
};
template <class Inner>
class DirectTop : public cs::System {
    Counts      counts_;
    std::size_t budget_;

public:
    DirectTop(Counts counts, std::size_t budget) : counts_{std::move(counts)}, budget_{budget} {}
    void setup() override {
        add_subsystem<CProd>("sz", counts_);
        add_subsystem<Src>("src", counts_.size());
        add_subsystem<Inner>("inner", budget_);
        add_subsystem<Cons>("cons");
        connect("src.x", "inner.in");
        connect("sz.counts", "inner.sizes");
        connect("inner.sums", "cons.in");
    }
};
}  // namespace trt

TEST_CASE("Chunked: a tie declared in the chunked system's own setup is internal (input_values on a promoted input)",
          "[chunked-level-tie][ragged][tie][chunked]") {
    const trt::Counts counts{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    cs::Model         plain = cs::create_model<trt::DirectTop<trt::DirectPlain>>(default_config(), counts, std::size_t{0});
    plain.setup();
    const auto want = trt::forward_out(plain);
    cs::Model  m    = cs::create_model<trt::DirectTop<trt::DirectChunk>>(default_config(), counts, std::size_t{200});
    m.setup();
    CHECK(trt::forward_out(m) == want);
    // internal and eternal (its holders answer -1, D141): the tie stays allocated, at a chunk's capacity
    CHECK(trt::inner_entry(m, "t").has_storage());
    CHECK(trt::inner_entry(m, "t").held() == false);
}

TEST_CASE("Chunked vs plain: a tie declared in the chunked system's own setup", "[chunked-level-tie][ragged][tie][chunked][replay]") {
    const trt::Counts first{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    const trt::Counts second{2, 0, 5, 5, 1, 1, 3, 4, 0, 2};
    for (const std::size_t budget : {std::size_t{1000}, std::size_t{200}, std::size_t{100}}) {
        INFO("budget " << budget);
        tcc::against_plain("inner", [&] { return cs::create_model<trt::DirectTop<trt::DirectChunk>>(default_config(), first, budget); },
                    [&] { return cs::create_model<trt::DirectTop<trt::DirectPlain>>(default_config(), first, std::size_t{0}); },
                    [](cs::Model& m) {
                        auto o = m.view_output<DType::f32, 1>("cons.out");
                        return std::vector<float>(o.span().begin(), o.span().end());
                    },
                    3, [&](cs::Model& m, int f) { m.init_input("sz.raw", cs::init::from_host(f == 1 ? second : first)); });
    }
}

// =============================================================================================================================
// OPACITY (C2e): a chunked system's only interface is its own promotions.
// =============================================================================================================================
namespace trt {

/// A top with ONE route into the chunked system 'inner' (ChunkTop's InnerChunk: g.p.in, g.t, g.s.sums inside; `in` and
/// `sums` promoted). Every route names something inside.
class RouteTop : public cs::System {
    int route_;

public:
    explicit RouteTop(int route) : route_{route} {}
    void setup() override {
        const Sizes sizes{1, 2, 3, 1};
        add_subsystem<Src>("src", 4);
        add_subsystem<InnerChunk>("inner", std::size_t{1000}, std::size_t{4}, sizes);
        add_subsystem<Zero>("z1");
        add_subsystem<Acc>("z2");
        switch (route_) {
            case 0: connect("src.x", "inner.g.p.in"); break;
            case 1:
                tie_outputs<DType::f32, 3>({"inner.g.p.t", "z2.t"}, "tt", cs::alias::dims(4, 2, 2), cs::TieOutputsPolicy::allow_plain);
                break;
            case 2:
                tie_outputs<DType::f32, 3>({"z1.t", "z2.t"}, "tt",
                                           cs::alias::dims(cs::alias::input_dim("inner.g.p.in", 0), 2, 2),
                                           cs::TieOutputsPolicy::allow_plain);
                break;
            case 3: promote<DType::f32, 1>("inner.g.p.in", "pp"); break;
            case 4: init_input("inner.g.p.in", cs::init::fill(1.0f)); break;
            case 5: add_design_var("inner.g.p.in"); break;
            default: break;
        }
    }
};

}  // namespace trt

TEST_CASE("Opaque chunked system: every route into it from outside is refused at setup", "[chunked][opaque]") {
    const std::vector<std::pair<int, const char*>> routes{
        {0, "connect"}, {1, "tie member path"}, {2, "alias source path"}, {3, "promote from above"}, {4, "init_input"}, {5, "mark"}};
    for (const auto& [route, what] : routes) {
        INFO("route: " << what);
        cs::Model m = cs::create_model<trt::RouteTop>(default_config(), route);
        trt::expect_throws([&] { m.setup(); }, {"lives inside chunked system 'inner'", "only its promotions are visible outside it",
                                                 "promote it to use it here"});
    }
}

TEST_CASE("Opaque chunked system: a Model reaches its promotions only; the chunked system reaches its own subtree",
          "[chunked][opaque]") {
    const trt::Sizes sizes{1, 3, 2, 3, 0, 3, 1, 2};
    cs::Model        m = cs::create_model<trt::ChunkTop<trt::InnerChunk>>(default_config(), sizes.size(), std::size_t{100}, sizes);
    m.setup();
    auto exp_open = [&](auto&& f) { trt::expect_throws(f, {"lives inside chunked system 'inner'", "promote it to use it here"}); };
    exp_open([&] { (void)m.root().entry_at("inner.g.p.in"); });
    exp_open([&] { (void)m.root().entry_at("inner.g.t"); });
    exp_open([&] { m.init_input("inner.g.p.in", cs::init::fill(1.0f)); });
    exp_open([&] { (void)m.view_input<DType::f32, 1>("inner.g.p.in"); });
    exp_open([&] { (void)m.view_output<DType::f32, 1>("inner.g.s.sums"); });
    exp_open([&] { (void)m.get_subsystem("inner.g"); });
    exp_open([&] { (void)m.get_output<DType::f32, 1>("inner.g.s.sums"); });
    // a path ending AT its promotion is legal
    CHECK_NOTHROW((void)m.root().entry_at("inner.sums"));
    CHECK_NOTHROW((void)m.root().entry_at("inner.in"));
    // the chunked system itself, and the systems inside it, reach their own subtrees
    auto& x = dynamic_cast<cs::ChunkedSystem&>(m.get_subsystem("inner"));
    CHECK_NOTHROW((void)x.entry_at("g.p.in"));
    CHECK_NOTHROW((void)x.entry_at("g.t"));
    CHECK_NOTHROW((void)x.get_subsystem("g"));
    auto& g = dynamic_cast<cs::System&>(x.get_subsystem("g"));
    CHECK_NOTHROW((void)g.entry_at("s.sums"));
    CHECK_NOTHROW((void)g.entry_at("t"));
    // and the model still runs
    CHECK(trt::forward_out(m) == trt::chunk_ref(sizes));
}

TEST_CASE("Opaque chunked system: a Model whose ROOT is a chunked system sees its promotions only", "[chunked][opaque]") {
    const trt::Sizes sizes{1, 3, 2, 3, 0, 3, 1, 2};
    cs::Model        m = cs::create_model<trt::InnerChunk>(default_config(), std::size_t{100}, sizes.size(), sizes);
    m.setup();
    auto exp_open = [&](auto&& f) { trt::expect_throws(f, {"lives inside chunked system", "promote it to use it here"}); };
    exp_open([&] { m.init_input("g.p.in", cs::init::fill(1.0f)); });
    exp_open([&] { (void)m.view_output<DType::f32, 1>("g.s.sums"); });
    exp_open([&] { (void)m.get_subsystem("g"); });
    CHECK_NOTHROW(m.init_input("in", cs::init::fill(1.0f)));  // its own promotion
    // the chunked root itself still reaches its subtree
    CHECK_NOTHROW((void)m.root().entry_at("g.p.in"));
}

TEST_CASE("Opaque chunked system: a tie declared above, with members inside, fails with the opacity error",
          "[outside-tie-inside-members][ragged][tie][chunked][opaque]") {
    class OvInnerBad : public cs::ChunkedSystem {
    public:
        explicit OvInnerBad(std::size_t budget) : cs::ChunkedSystem(budget) {}
        void setup() override {
            add_subsystem<trt::Zero>("p");
            add_subsystem<trt::Acc>("q");
        }
    };
    class OvTopBad : public cs::System {
    public:
        void setup() override {
            add_subsystem<trt::CProd>("sz", trt::Counts{1, 2, 3, 4});
            add_subsystem<OvInnerBad>("inner", std::size_t{1000});
            tie_outputs<DType::f32, 3>({"inner.p.t", "inner.q.t"}, "t",
                                       cs::alias::dims(cs::alias::output_dim("sz.counts", 0), cs::alias::output_values("sz.counts"),
                                                       cs::alias::output_values("sz.counts")),
                                       cs::TieOutputsPolicy::allow_plain);
        }
    };
    cs::Model m = cs::create_model<OvTopBad>(default_config());
    trt::expect_throws([&] { m.setup(); }, {"lives inside chunked system 'inner'", "promote it to use it here"});
}

// ---- an outside tie over PROMOTIONS of inside cs::tied ports ----------------------------------------------------------------
namespace trt {

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

void fill_ot(cs::System& s) {
    s.add_subsystem<AccIn>("q");
    s.promote<DType::f32, 1>("q.in", "in");
    s.promote<DType::f32, 3>("q.t", "qt");  // a promotion of an inside cs::tied output: the outside tie's member
}
class OtPlain : public cs::System {
public:
    explicit OtPlain(std::size_t) {}
    void setup() override { fill_ot(*this); }
};
class OtChunk : public cs::ChunkedSystem {
public:
    explicit OtChunk(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override { fill_ot(*this); }
};
/// tie_outputs over {outside writer w.t (zeroes), inner.qt (the chunks accumulate their slices)}, sized from sz.counts.
template <class Inner>
class OtTop : public cs::System {
    Counts      init_;
    std::size_t budget_;

public:
    OtTop(Counts init, std::size_t budget) : init_{std::move(init)}, budget_{budget} {}
    void setup() override {
        add_subsystem<CProd>("sz", init_);
        add_subsystem<Src>("src", init_.size());
        add_subsystem<Zero>("w");
        add_subsystem<Inner>("inner", budget_);
        add_subsystem<SumT>("sum");
        add_subsystem<Cons>("cons");
        tie_outputs<DType::f32, 3>({"w.t", "inner.qt"}, "t",
                                   cs::alias::dims(cs::alias::output_dim("sz.counts", 0), cs::alias::output_values("sz.counts"),
                                                   cs::alias::output_values("sz.counts")),
                                   cs::TieOutputsPolicy::allow_plain);
        connect("src.x", "inner.in");
        connect("t", "sum.pts");
        connect("sum.sums", "cons.in");
    }
};

/// tie_inputs over {inner.pts (a promotion of the inside tied input sa.pts), sb.pts}, connected from gen.pts.
void fill_ti(cs::System& s) {
    s.add_subsystem<SumK<1>>("sa");
    s.promote<DType::f32, 3>("sa.pts", "pts");
    s.promote<DType::f32, 1>("sa.sums", "sums");
}
class TiPlain : public cs::System {
public:
    explicit TiPlain(std::size_t) {}
    void setup() override { fill_ti(*this); }
};
class TiChunk : public cs::ChunkedSystem {
public:
    explicit TiChunk(std::size_t budget) : cs::ChunkedSystem(budget) {}
    void setup() override { fill_ti(*this); }
};
template <class Inner>
class TiTop : public cs::System {
    Counts      init_;
    std::size_t budget_;

public:
    TiTop(Counts init, std::size_t budget) : init_{std::move(init)}, budget_{budget} {}
    void setup() override {
        add_subsystem<CProd>("sz", init_);
        add_subsystem<Gen>("gen");
        add_subsystem<Inner>("inner", budget_);
        add_subsystem<SumK<1>>("sb");
        add_subsystem<Cons>("c1");
        add_subsystem<Cons>("c2");
        connect("sz.counts", "gen.counts");
        tie_inputs<DType::f32, 3>({"inner.pts", "sb.pts"}, "tp");
        connect("gen.pts", "tp");
        connect("inner.sums", "c1.in");
        connect("sb.sums", "c2.in");
    }
};
}  // namespace trt

TEST_CASE("Chunked vs plain: an outside tie_outputs over a promotion of an inside tied output plus an outside writer",
          "[outside-tie-inside-members][ragged][tie][chunked][replay]") {
    const trt::Counts first{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    const trt::Counts second{2, 0, 5, 5, 1, 1, 3, 4, 0, 2};
    const auto        read = [](cs::Model& m) {
        auto o = m.view_output<DType::f32, 1>("cons.out");
        std::vector<float> out(o.span().begin(), o.span().end());
        auto               t = m.view_ragged_input<DType::f32, 3>("sum.pts");  // the tie alias: the FULL root
        out.insert(out.end(), t.data.begin(), t.data.end());
        return out;
    };
    cs::Model plain = cs::create_model<trt::OtTop<trt::OtPlain>>(default_config(), first, std::size_t{0});
    plain.setup();
    const auto pout = (plain.forward(), plain.wait(), read(plain));
    for (const std::size_t budget : {std::size_t{1000}, std::size_t{150}, std::size_t{70}}) {
        INFO("budget " << budget);
        cs::Model m = cs::create_model<trt::OtTop<trt::OtChunk>>(default_config(), first, budget);
        m.setup();
        m.forward();
        m.wait();
        CHECK(read(m) == pout);  // matches the unchunked run (the alias is the full root, the chunks wrote their slices)
        tcc::against_plain("inner", [&] { return cs::create_model<trt::OtTop<trt::OtChunk>>(default_config(), first, budget); },
                    [&] { return cs::create_model<trt::OtTop<trt::OtPlain>>(default_config(), first, std::size_t{0}); }, read, 3,
                    [&](cs::Model& mm, int f) { mm.init_input("sz.raw", cs::init::from_host(f == 1 ? second : first)); });
    }
}

TEST_CASE("Chunked vs plain: an outside tie_inputs over a promotion of an inside tied input", "[outside-tie-inside-members][ragged][tie][chunked][replay]") {
    const trt::Counts first{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    const trt::Counts second{2, 0, 5, 5, 1, 1, 3, 4, 0, 2};
    const auto        read = [](cs::Model& m) {
        std::vector<float> out;
        for (const char* p : {"c1.out", "c2.out"}) {
            auto o = m.view_output<DType::f32, 1>(p);
            out.insert(out.end(), o.span().begin(), o.span().end());
        }
        return out;
    };
    cs::Model plain = cs::create_model<trt::TiTop<trt::TiPlain>>(default_config(), first, std::size_t{0});
    plain.setup();
    const auto pout = (plain.forward(), plain.wait(), read(plain));
    for (const std::size_t budget : {std::size_t{1000}, std::size_t{100}}) {
        INFO("budget " << budget);
        cs::Model m = cs::create_model<trt::TiTop<trt::TiChunk>>(default_config(), first, budget);
        m.setup();
        m.forward();
        m.wait();
        CHECK(read(m) == pout);
        tcc::against_plain("inner", [&] { return cs::create_model<trt::TiTop<trt::TiChunk>>(default_config(), first, budget); },
                    [&] { return cs::create_model<trt::TiTop<trt::TiPlain>>(default_config(), first, std::size_t{0}); }, read, 3,
                    [&](cs::Model& mm, int f) { mm.init_input("sz.raw", cs::init::from_host(f == 1 ? second : first)); });
    }
}

TEST_CASE("Chunked vs plain: cs::per_index host arrays on a tie alias, several budgets", "[ragged][tie][chunked][replay]") {
    const trt::Sizes sizes{3, 1, 4, 0, 2, 5, 1, 3, 2, 4};
    for (const std::size_t budget : {std::size_t{1000}, std::size_t{400}, std::size_t{200}, std::size_t{100}}) {
        INFO("budget " << budget);
        tcc::against_plain("inner", [&] { return cs::create_model<trt::ChunkTop<trt::InnerChunk>>(default_config(), sizes.size(), budget, sizes); },
                    [&] { return cs::create_model<trt::ChunkTop<trt::InnerPlain>>(default_config(), sizes.size(), std::size_t{0}, sizes); },
                    [](cs::Model& m) {
                        auto o = m.view_output<DType::f32, 1>("cons.out");
                        return std::vector<float>(o.span().begin(), o.span().end());
                    });

        // by hand: the tie member g.p.t is [N, m_i, m_i]; in each chunk [a, b) its dim 0 is b - a and both per-index arrays are
        // sizes[a..b)
        cs::Model m = cs::create_model<trt::ChunkTop<trt::InnerChunk>>(default_config(), sizes.size(), budget, sizes);
        m.setup();
        auto&         sys = dynamic_cast<cs::ChunkedSystem&>(m.get_subsystem("inner"));
        tcc::Recorder rec;
        rec.attach(sys, {"g.p.in", "g.p.t"});
        m.forward();
        m.wait();
        std::size_t next = 0;
        for (const auto& row : rec.chunks) {
            REQUIRE(row[0].range.has_value());
            const std::size_t a = row[0].range->first, b = row[0].range->second;
            INFO("chunk [" << a << ", " << b << ")");
            CHECK(a == next);
            next = b;
            const trt::Sizes want(sizes.begin() + static_cast<std::ptrdiff_t>(a), sizes.begin() + static_cast<std::ptrdiff_t>(b));
            CHECK(row[1].dim0 == b - a);
            CHECK(row[1].per1 == want);
            CHECK(row[1].per2 == want);
        }
        CHECK(next == sizes.size());
        if (budget == 1000) {  // everything fits: one chunk, the whole arrays
            REQUIRE(rec.chunks.size() == 1);
            CHECK(rec.chunks[0][1].per1 == sizes);
        }
        if (budget <= 200) CHECK(rec.chunks.size() > 1);  // the tie root alone is 4 * (9+1+16+0+4+25+1+9+4+16) = 340 B
    }
}
