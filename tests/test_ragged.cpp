// Ragged dimensions: the full test set.
//
//   Counts   u32 [N] counts (a copy of a free input)
//   Fill     RaggedOutput<f32,3> "pts": N x counts[i] x 3, extents from the VALUES of `counts`
//            (FromInputValues); pts.block(i)[j,k] = i*1000 + j*10 + k
//   Scale    RaggedInput -> RaggedOutput (inherits the input's per-index dims with FromInputDim): x2
//   Sum      RaggedInput (inherits through an unspecified dim) -> a dense [N] output of block sums
//
// Everything is checked against host references, through view_output (dense) and view_ragged_* (n(), block_offsets(),
// block(i), span()), synced after the forward (docs/design-review-2026-10-03.md, section 5). Value-driven ragged shapes come
// from the device column (section 6): the host's per-index arrays of those no longer exist, so the old assertions on
// extents(d) and is_per_index(d) are dropped; the shape is asserted through the blocks' offsets and sizes instead.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
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

namespace trag {

using Sizes = std::vector<std::size_t>;

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

// ---- Counts: counts[i] = raw[i] ---------------------------------------------------------------
struct CountsBody {
    struct Inputs {
        cs::Input<DType::u32, 1> raw;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> counts;
    };
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const { out.counts[i] = in.raw[i]; }
};
// N comes from a free input `n` (so `raw` is value-driven: a fixed size-0 root would have no arena).
class Counts : public cs::Component<CountsBody> {
    std::size_t               n_;
    cs::Input<DType::u32, 1>  nin_;
    cs::Input<DType::u32, 1>  raw_;
    cs::Output<DType::u32, 1> counts_;

public:
    explicit Counts(std::size_t n) : n_{n} {}
    void setup() override {
        declare_input(nin_, "n", 1);
        init_input("n", cs::init::from_host(std::vector<std::uint32_t>{static_cast<std::uint32_t>(n_)}));
        declare_input(raw_, "raw", cs::FromInput{"n", 0});
        declare_output(counts_, "counts", cs::FromInputDim{"raw", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(counts_.shape(), {raw_}, {counts_}, {});
    }
};

// ---- Fill: pts.block(i)[j,k] = i*1000 + j*10 + k ---------------------------------------------------
struct FillBody {
    struct Inputs {
        cs::Input<DType::u32, 1> counts;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> pts;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs out) const {
        const std::size_t i = id[0];
        auto              b = out.pts.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j)
            for (std::size_t k = 0; k < b.shape()[1]; ++k)
                b[sycl::id<2>{j, k}] = static_cast<float>(i * 1000 + j * 10 + k);
    }
};
class Fill : public cs::Component<FillBody> {
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

// ---- Scale: out = 2 * in, the output inherits the input's per-index dims (FromInputDim) -----------------
struct ScaleBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 3> in;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> out;
    };
    void operator()(sycl::id<1> id, Inputs in, Outputs out) const {
        auto bi = in.in.block(id[0]);
        auto bo = out.out.block(id[0]);
        for (std::size_t q = 0; q < bi.size(); ++q) bo[q] = 2.0f * bi[q];
    }
};
class Scale : public cs::Component<ScaleBody> {
    cs::RaggedInput<DType::f32, 3>  in_;
    cs::RaggedOutput<DType::f32, 3> out_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified, cs::unspecified, cs::unspecified);
        declare_output(out_, "out", cs::FromInputDim{"in", 0}, cs::FromInputDim{"in", 1}, cs::FromInputDim{"in", 2});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{in_.size()}, {in_}, {out_}, {});
    }
};

// ---- Sum: sums[i] = the sum of every element of block i (rank-agnostic, flat access) -----------------
template <int R>
struct SumBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, R> pts;
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
template <int R>
class Sum : public cs::Component<SumBody<R>> {
    cs::RaggedInput<DType::f32, R> pts_;
    cs::Output<DType::f32, 1>      sums_;

public:
    void setup() override {
        if constexpr (R == 3)
            this->declare_input(pts_, "pts", cs::unspecified, cs::unspecified, cs::unspecified);
        else
            this->declare_input(pts_, "pts", cs::unspecified, cs::unspecified);
        this->declare_output(sums_, "sums", cs::FromInputDim{"pts", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(sums_.shape(), {pts_}, {sums_}, {});
    }
};

/// counts -> fill -> scale -> sum (n counts, free input "counts.raw").
class RaggedTop : public cs::System {
    std::size_t n_;

public:
    explicit RaggedTop(std::size_t n) : n_{n} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Counts>("counts", n_));
        add_subsystem(cs::subsystem_desc<Fill>("fill"));
        add_subsystem(cs::subsystem_desc<Scale>("scale"));
        add_subsystem(cs::subsystem_desc<Sum<3>>("sum"));
        connect("counts.counts", "fill.counts");
        connect("fill.pts", "scale.in");
        connect("scale.out", "sum.pts");
    }
};

// ---- Cols: [N,2] input; BOTH trailing dims per-index, from the two columns ----------------------------------
struct ColsBody {
    struct Inputs {
        cs::Input<DType::u32, 2> ab;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 3> t;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs out) const {
        const std::size_t i = id[0];
        auto              b = out.t.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j)
            for (std::size_t k = 0; k < b.shape()[1]; ++k)
                b[sycl::id<2>{j, k}] = static_cast<float>(i * 1000 + j * 10 + k);
    }
};
class Cols : public cs::Component<ColsBody> {
    cs::Input<DType::u32, 2>        ab_;
    cs::RaggedOutput<DType::f32, 3> t_;

public:
    void setup() override {
        declare_input(ab_, "ab", 4, 2);
        declare_output(t_, "t", cs::FromInputDim{"ab", 0}, cs::FromInputValues{"ab", 0}, cs::FromInputValues{"ab", 1});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{ab_.shape()[0]}, {ab_}, {t_}, {});
    }
};

// ---- Col1: [N,2] input, the extents are COLUMN 1 -------------------------------------------------------------
struct Col1Body {
    struct Inputs {
        cs::Input<DType::u32, 2> cnt;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> v;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs out) const {
        const std::size_t i = id[0];
        auto              b = out.v.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j) b[j] = static_cast<float>(i * 100 + j);
    }
};
class Col1 : public cs::Component<Col1Body> {
    cs::Input<DType::u32, 2>        cnt_;
    cs::RaggedOutput<DType::f32, 2> v_;

public:
    void setup() override {
        declare_input(cnt_, "cnt", 4, 2);
        declare_output(v_, "v", cs::FromInputDim{"cnt", 0}, cs::FromInputValues{"cnt", 1});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{cnt_.shape()[0]}, {cnt_}, {v_}, {});
    }
};

// ---- Neg: an i32 input of extents; a negative one is an error ---------------------------------------------------
struct NegBody {
    struct Inputs {
        cs::Input<DType::i32, 1> cnt;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> v;
    };
    void operator()(sycl::id<1>, Inputs, Outputs) const {}
};
class Neg : public cs::Component<NegBody> {
    cs::Input<DType::i32, 1>        cnt_;
    cs::RaggedOutput<DType::f32, 2> v_;

public:
    void setup() override {
        declare_input(cnt_, "cnt", 3);
        declare_output(v_, "v", cs::FromInputDim{"cnt", 0}, cs::FromInputValues{"cnt"});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cnt_.shape(), {cnt_}, {v_}, {});
    }
};

// ---- Make: a PerIndex host-array port of n blocks; m.block(i)[j] = i*100 + j -------------------------------------
struct MakeBody {
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> m;
    };
    void operator()(sycl::id<1> id, Inputs, Outputs out) const {
        const std::size_t i = id[0];
        auto              b = out.m.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j) b[j] = static_cast<float>(i * 100 + j);
    }
};
class Make : public cs::Component<MakeBody> {
    std::size_t                     n_;
    Sizes                           sizes_;
    cs::RaggedOutput<DType::f32, 2> m_;

public:
    Make(std::size_t n, Sizes sizes) : n_{n}, sizes_{std::move(sizes)} {}
    void setup() override { declare_output(m_, "m", n_, cs::per_index(sizes_)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{n_}, {}, {m_}, {});
    }
};

/// Sums a rank-2 ragged input that DECLARES its own per-index host array (checked against the source).
class TakeSum : public cs::Component<SumBody<2>> {
    std::size_t                    n_;
    Sizes                          sizes_;
    cs::RaggedInput<DType::f32, 2> pts_;
    cs::Output<DType::f32, 1>      sums_;

public:
    TakeSum(std::size_t n, Sizes sizes) : n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        declare_input(pts_, "pts", n_, cs::per_index(sizes_));
        declare_output(sums_, "sums", cs::FromInputDim{"pts", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sums_.shape(), {pts_}, {sums_}, {});
    }
};

/// make (n blocks of `make_sizes`) -> take (declares `take_sizes`).
class PerIndexTop : public cs::System {
    std::size_t n_;
    Sizes       make_, take_;

public:
    PerIndexTop(std::size_t n, Sizes make_sizes, Sizes take_sizes)
        : n_{n}, make_{std::move(make_sizes)}, take_{std::move(take_sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Make>("make", n_, make_));
        add_subsystem(cs::subsystem_desc<TakeSum>("take", n_, take_));
        connect("make.m", "take.pts");
    }
};

// ---- a RaggedInput reading a DENSE port ----------------------------------------------------------------------------
struct DenseGenBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> d;
    };
    void operator()(sycl::id<2> i, Inputs, Outputs out) const { out.d[i] = static_cast<float>(i[0] * 10 + i[1]); }
};
class DenseGen : public cs::Component<DenseGenBody> {
    cs::Output<DType::f32, 2> d_;

public:
    void setup() override { declare_output(d_, "d", 3, 4); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(d_.shape(), {}, {d_}, {});
    }
};
class DenseTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<DenseGen>("gen"));
        add_subsystem(cs::subsystem_desc<Sum<2>>("sum"));
        connect("gen.d", "sum.pts");
    }
};

// ---- errors ----------------------------------------------------------------------------------------------------------
struct NoopBody {
    struct Inputs {};
    struct Outputs {};
    void operator()(sycl::id<1>, Inputs, Outputs) const {}
};
/// A per-index spec on dimension 0.
class Dim0Bad : public cs::Component<MakeBody> {
    cs::RaggedOutput<DType::f32, 2> m_;

public:
    void setup() override { declare_output(m_, "m", cs::per_index(Sizes{1, 2}), 2); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{2}, {}, {m_}, {});
    }
};
/// A DENSE wrapper declared with a per-index spec.
struct DenseBadBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> m;
    };
    void operator()(sycl::id<1>, Inputs, Outputs) const {}
};
class DenseBad : public cs::Component<DenseBadBody> {
    cs::Output<DType::f32, 2> m_;

public:
    void setup() override { declare_output(m_, "m", std::size_t{4}, cs::per_index(Sizes{1, 0, 3, 2})); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{4}, {}, {m_}, {});
    }
};
/// A dense INPUT that inherits a ragged source (through an unspecified dim).
struct DenseInBody {
    struct Inputs {
        cs::Input<DType::f32, 2> in;
    };
    struct Outputs {};
    void operator()(sycl::id<1>, Inputs, Outputs) const {}
};
class DenseIn : public cs::Component<DenseInBody> {
    cs::Input<DType::f32, 2> in_;

public:
    void setup() override { declare_input(in_, "in", cs::unspecified, cs::unspecified); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{4}, {in_}, {}, {});
    }
};
/// A dense OUTPUT that inherits a ragged input's per-index dim (FromInputDim).
struct DenseOutBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, 2> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> o;
    };
    void operator()(sycl::id<1>, Inputs, Outputs) const {}
};
class DenseOut : public cs::Component<DenseOutBody> {
    cs::RaggedInput<DType::f32, 2> in_;
    cs::Output<DType::f32, 2>      o_;

public:
    void setup() override {
        declare_input(in_, "in", cs::unspecified, cs::unspecified);
        declare_output(o_, "o", cs::FromInputDim{"in", 0}, cs::FromInputDim{"in", 1});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{4}, {in_}, {o_}, {});
    }
};
template <class C>
class InheritTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Make>("make", std::size_t{4}, Sizes{1, 0, 3, 2}));
        add_subsystem(cs::subsystem_desc<C>("c"));
        connect("make.m", "c.in");
    }
};
template <class C>
class SingleTop : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<C>("c")); }
};
/// A fused component (a helper-built descriptor) whose only stage has a ragged output.
template <cs::fuse_strategy S>
class FusedRagged : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::fuse_components<S>("w", cs::subsystem_desc<Make>("make", std::size_t{4}, Sizes{1, 0, 3, 2})));
    }
};

// ---- references ------------------------------------------------------------------------------------------------------
/// The flat data of Fill for `counts`, and its offsets.
struct FillRef {
    std::vector<float>         data;
    std::vector<std::uint64_t> offsets{0};
};
FillRef fill_ref(const std::vector<std::uint32_t>& counts) {
    FillRef r;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        for (std::size_t j = 0; j < counts[i]; ++j)
            for (std::size_t k = 0; k < 3; ++k) r.data.push_back(static_cast<float>(i * 1000 + j * 10 + k));
        r.offsets.push_back(r.data.size());
    }
    return r;
}

/// A synced host copy of a dense rank-1 output (the model has run).
std::vector<float> dense1(cs::Model& model, const char* path) { return cst::to_vector(cst::synced_output<DType::f32, 1>(model, path)); }

/// The elements of a synced ragged view, end to end.
template <class View>
std::vector<float> flat_of(const View& v) {
    return std::vector<float>(v.span().begin(), v.span().end());
}

/// Run one forward with `counts` and check every view against the references.
void run_counts(cs::Model& model, const std::vector<std::uint32_t>& counts) {
    if (!counts.empty()) model.update_input("counts.raw", cs::init::from_host(counts));
    model.forward();
    model.wait();
    const FillRef ref = fill_ref(counts);

    // the dense sums of the scaled blocks
    auto sums = cst::synced_output<DType::f32, 1>(model, "sum.sums");
    REQUIRE(sums.size() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        float want = 0.0f;
        for (std::uint64_t q = ref.offsets[i]; q < ref.offsets[i + 1]; ++q) want += 2.0f * ref.data[q];
        INFO("block " << i << " (count " << counts[i] << ")");
        CHECK(sums.span()[i] == want);
    }

    // the ragged output itself, and its peer
    auto pts = model.view_ragged_output<DType::f32, 3>("fill.pts");
    pts.sync();
    CHECK(pts.n() == counts.size());
    CHECK(pts.block_offsets() == ref.offsets);
    CHECK(flat_of(pts) == ref.data);
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const auto b = pts.block(i);  // a flat span: counts[i] rows of 3
        REQUIRE(b.size() == counts[i] * 3);
        for (std::size_t j = 0; j < counts[i]; ++j)
            for (std::size_t k = 0; k < 3; ++k) CHECK(b[j * 3 + k] == static_cast<float>(i * 1000 + j * 10 + k));
    }
    auto scaled = model.view_ragged_output<DType::f32, 3>("scale.out");
    scaled.sync();
    CHECK(scaled.block_offsets() == ref.offsets);
    REQUIRE(scaled.size() == ref.data.size());
    for (std::size_t q = 0; q < ref.data.size(); ++q) CHECK(scaled.span()[q] == 2.0f * ref.data[q]);
    auto peer = model.view_ragged_input<DType::f32, 3>("scale.in");  // an input peer of the root
    peer.sync();
    CHECK(flat_of(peer) == ref.data);
    CHECK(peer.block_offsets() == ref.offsets);
}

}  // namespace trag

TEST_CASE("Ragged: fill / scale / sum over forwards with changing counts", "[ragged]") {
    cs::Model model = cs::create_model<trag::RaggedTop>(default_config(), std::size_t{6});
    model.setup();
    trag::run_counts(model, {3, 0, 2, 4, 0, 1});  // zeros, an empty block in the middle and one last
    trag::run_counts(model, {0, 5, 1, 0, 2, 2});  // growth (10 -> 10 blocks differ), an empty FIRST block
    trag::run_counts(model, {1, 0, 0, 0, 0, 0});  // shrink to a single block
    trag::run_counts(model, {0, 0, 0, 0, 0, 0});  // all blocks empty
    trag::run_counts(model, {2, 2, 2, 2, 2, 7});  // and back up
}

TEST_CASE("Ragged: N = 0 stays ragged-capable", "[ragged]") {
    // REVIEW: n = 0 makes zero-size variables (counts, raw, pts); the headers do not say how a zero-element allocation is
    // handled now that the old arena (and its "no arena for a size-0 root" comment) is gone. The old assertions are kept.
    cs::Model model = cs::create_model<trag::RaggedTop>(default_config(), std::size_t{0});
    model.setup();
    trag::run_counts(model, {});
    auto pts = model.view_ragged_output<DType::f32, 3>("fill.pts");
    pts.sync();
    CHECK(pts.n() == 0);
    CHECK(pts.block_offsets() == std::vector<std::uint64_t>{0});
    CHECK(pts.size() == 0);
}

TEST_CASE("Ragged: FromInputValues on a rank-2 [N,k] input, column 1", "[ragged]") {
    cs::Model model = cs::create_model<trag::SingleTop<trag::Col1>>(default_config());
    model.setup();
    model.update_input("c.cnt", cs::init::from_host(std::vector<std::uint32_t>{9, 2, 9, 0, 9, 3, 9, 1}));
    model.forward();
    model.wait();
    auto v = model.view_ragged_output<DType::f32, 2>("c.v");
    v.sync();
    CHECK(v.n() == 4);
    CHECK(v.block_offsets() == std::vector<std::uint64_t>{0, 2, 2, 5, 6});
    const trag::Sizes extents{2, 0, 3, 1};  // the blocks' sizes (the per-index extents of dimension 1)
    for (std::size_t i = 0; i < 4; ++i) CHECK(v.block(i).size() == extents[i]);
    CHECK(trag::flat_of(v) == std::vector<float>{0, 1, 200, 201, 202, 300});
}

TEST_CASE("Ragged: both trailing dims per-index, from two columns", "[ragged]") {
    cs::Model model = cs::create_model<trag::SingleTop<trag::Cols>>(default_config());
    model.setup();
    const auto run = [&](const std::vector<std::uint32_t>& ab) {
        model.update_input("c.ab", cs::init::from_host(ab));
        model.forward();
        model.wait();
        auto t = model.view_ragged_output<DType::f32, 3>("c.t");
        t.sync();
        REQUIRE(t.n() == 4);
        trag::Sizes                e1, e2;
        std::vector<float>         data;
        std::vector<std::uint64_t> offs{0};
        for (std::size_t i = 0; i < 4; ++i) {
            e1.push_back(ab[2 * i]);
            e2.push_back(ab[2 * i + 1]);
            for (std::size_t j = 0; j < ab[2 * i]; ++j)
                for (std::size_t k = 0; k < ab[2 * i + 1]; ++k) data.push_back(static_cast<float>(i * 1000 + j * 10 + k));
            offs.push_back(data.size());
        }
        CHECK(t.block_offsets() == offs);
        CHECK(trag::flat_of(t) == data);
        for (std::size_t i = 0; i < 4; ++i) CHECK(t.block(i).size() == e1[i] * e2[i]);  // each block is e1[i] x e2[i]
    };
    run({2, 3, 0, 4, 3, 0, 1, 1});  // empty blocks in the middle (a zero in either column)
    run({0, 5, 2, 2, 3, 1, 0, 0});  // empty first and last blocks, then growth
}

TEST_CASE("Ragged: a ragged output connected to a ragged input with matching extents", "[ragged]") {
    const trag::Sizes sizes{1, 0, 3, 2};
    cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), std::size_t{4}, sizes, sizes);
    model.setup();
    model.forward();
    model.wait();
    auto sums = cst::synced_output<DType::f32, 1>(model, "take.sums");
    REQUIRE(sums.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        float want = 0.0f;
        for (std::size_t j = 0; j < sizes[i]; ++j) want += static_cast<float>(i * 100 + j);
        CHECK(sums.span()[i] == want);
    }
    auto m = model.view_ragged_output<DType::f32, 2>("make.m");
    m.sync();
    CHECK(m.block_offsets() == std::vector<std::uint64_t>{0, 1, 1, 4, 6});
    for (std::size_t i = 0; i < 4; ++i) CHECK(m.block(i).size() == sizes[i]);
    CHECK(trag::flat_of(m) == std::vector<float>{0, 200, 201, 202, 300, 301});
    // N = 0 with a PerIndex array of length 0 is ragged-capable too
    cs::Model zero = cs::create_model<trag::PerIndexTop>(default_config(), std::size_t{0}, trag::Sizes{}, trag::Sizes{});
    zero.setup();
    zero.forward();
    zero.wait();
    auto z = zero.view_ragged_output<DType::f32, 2>("make.m");
    z.sync();
    CHECK(z.n() == 0);
    CHECK(z.block_offsets() == std::vector<std::uint64_t>{0});
}

TEST_CASE("Ragged: mismatching declared per-index arrays error at forward", "[ragged][errors][shape]") {
    // REVIEW: the old needles ("dimension 1 is PerIndex", "'make.m'") are gone. component.hpp check_expected now reports
    // "shape mismatch: dimension 1 of '<port>' is PerIndex = [..] but the variable's dimension 1 is [..]" for the instance
    // whose expectation fails; which side (take.pts or make.m) it names depends on the order the checks run, so the needles
    // are the rule text and the two extents arrays.
    SECTION("same total, different extents") {
        cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), std::size_t{4}, trag::Sizes{1, 0, 3, 2},
                                                              trag::Sizes{2, 1, 2, 1});
        model.setup();
        trag::expect_throws([&] { model.forward(); }, {"shape mismatch", "dimension 1", "PerIndex", "[2,1,2,1]", "[1,0,3,2]"});
        model.wait();  // the source already launched
    }
    SECTION("a different total") {
        cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), std::size_t{4}, trag::Sizes{1, 0, 3, 2},
                                                              trag::Sizes{1, 1, 1, 1});
        model.setup();
        trag::expect_throws([&] { model.forward(); }, {"shape mismatch", "dimension 1", "PerIndex", "[1,1,1,1]", "[1,0,3,2]"});
        model.wait();
    }
}

TEST_CASE("Ragged: a RaggedInput reading a DENSE port", "[ragged]") {
    cs::Model model = cs::create_model<trag::DenseTop>(default_config());
    model.setup();
    model.forward();
    model.wait();
    auto sums = cst::synced_output<DType::f32, 1>(model, "sum.sums");
    REQUIRE(sums.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) CHECK(sums.span()[i] == static_cast<float>(40 * i + 6));
    // view_ragged_* on a dense entry: offsets are the row starts
    auto d = model.view_ragged_output<DType::f32, 2>("gen.d");
    d.sync();
    CHECK(d.n() == 3);
    CHECK(d.block_offsets() == std::vector<std::uint64_t>{0, 4, 8, 12});
    for (std::size_t i = 0; i < 3; ++i) CHECK(d.block(i).size() == 4);
    CHECK(d.block(2)[std::size_t{3}] == 23.0f);
    auto in = model.view_ragged_input<DType::f32, 2>("sum.pts");
    in.sync();
    CHECK(trag::flat_of(in) == trag::flat_of(d));
    CHECK(in.block_offsets() == d.block_offsets());
    // the dense view of a dense entry is unchanged
    CHECK(cst::synced_output<DType::f32, 2>(model, "gen.d").span()[5] == 11.0f);
}

TEST_CASE("Ragged: the dense host views of a ragged entry point to view_ragged_*", "[ragged][errors]") {
    cs::Model model = cs::create_model<trag::RaggedTop>(default_config(), std::size_t{6});
    model.setup();
    trag::run_counts(model, {3, 0, 2, 4, 0, 1});
    trag::expect_throws([&] { (void)model.view_output<DType::f32, 3>("fill.pts"); }, {"fill.pts", "view_ragged_output"});
    trag::expect_throws([&] { (void)model.view_input<DType::f32, 3>("sum.pts"); }, {"sum.pts", "view_ragged_input"});
    // the wrong dtype or rank still names the view
    trag::expect_throws([&] { (void)model.view_ragged_output<DType::u32, 3>("fill.pts"); }, {"wrong dtype or rank"});
}

TEST_CASE("Ragged: declaration and binding errors", "[ragged][errors]") {
    SECTION("a per-index spec on dimension 0") {
        cs::Model model = cs::create_model<trag::SingleTop<trag::Dim0Bad>>(default_config());
        trag::expect_throws([&] { model.setup(); }, {"dimension 0"});
    }
    SECTION("a dense wrapper declared with a per-index spec") {
        cs::Model model = cs::create_model<trag::SingleTop<trag::DenseBad>>(default_config());
        trag::expect_throws([&] { model.setup(); }, {"'m'", "per-index dims need RaggedInput/RaggedOutput"});
    }
    SECTION("a dense input connected to a ragged source: the connect throws at setup") {
        // Was a forward-time error. A dense Input cannot see per-index extents, and that is a wiring rule: connect checks it
        // in its own call (section 2; wiring.hpp, connect).
        cs::Model model = cs::create_model<trag::InheritTop<trag::DenseIn>>(default_config());
        trag::expect_throws([&] { model.setup(); },
                            {"connect('make.m', 'c.in')", "'make.m' is ragged", "a dense Input cannot see per-index extents", "RaggedInput"});
    }
    SECTION("a dense output inheriting a per-index dim of a ragged input") {
        cs::Model model = cs::create_model<trag::InheritTop<trag::DenseOut>>(default_config());
        model.setup();
        trag::expect_throws([&] { model.forward(); }, {"'o' is ragged; declare it with RaggedInput/RaggedOutput"});
        model.wait();
    }
    SECTION("a per-index length different from N") {
        // REVIEW: the old message is kept as the needle; the header that reports a host per-index array of the wrong length
        // for a produced variable (finish_ragged_total / resolve_free_shape) is only confirmed for the "per-index dimension
        // d has k extents but dimension 0 has n" text, not for which path of a produced shape fires it.
        cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), std::size_t{4}, trag::Sizes{1, 2, 3},
                                                              trag::Sizes{1, 2, 3, 4});
        model.setup();
        trag::expect_throws([&] { model.forward(); }, {"per-index dimension 1 has 3 extents but dimension 0 has 4"});
    }
    SECTION("negative extents") {
        cs::Model model = cs::create_model<trag::SingleTop<trag::Neg>>(default_config());
        model.setup();
        model.update_input("c.cnt", cs::init::from_host(std::vector<std::int32_t>{1, -1, 2}));
        // (the old needle was "negative extent"; driving_values now says "holds a negative value; a size cannot be negative")
        trag::expect_throws([&] { model.forward(); }, {"holds a negative value"});
    }
    SECTION("a ragged port in a fused component: refused at setup") {
        cs::Model model = cs::create_model<trag::FusedRagged<cs::fuse_strategy::seq>>(default_config());
        trag::expect_throws([&] { model.setup(); }, {"ragged ports are not supported in fused components yet"});
    }
    SECTION("a ragged port in a parallel fused component: refused at setup") {
        cs::Model model = cs::create_model<trag::FusedRagged<cs::fuse_strategy::par>>(default_config());
        trag::expect_throws([&] { model.setup(); }, {"ragged ports are not supported in fused components yet"});
    }
}
