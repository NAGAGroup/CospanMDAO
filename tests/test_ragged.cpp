// Ragged dimensions (D140 part 1): the full test set.
//
//   Counts   u32 [N] counts (a copy of a free input)
//   Fill     RaggedOutput<f32,3> "pts": N x counts[i] x 3, extents from the VALUES of `counts`
//            (FromInputValues); pts.block(i)[j,k] = i*1000 + j*10 + k
//   Scale    RaggedInput -> RaggedOutput (inherits the input's per-index dims with FromInputDim): x2
//   Sum      RaggedInput (inherits with FromSourceDim) -> a dense [N] output of block sums
//
// Everything is checked against host references, through view_output (dense) and view_ragged_*.

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
        declare_input(counts_, "counts", cs::FromSourceDim{0});
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
        declare_input(in_, "in", cs::FromSourceDim{0}, cs::FromSourceDim{1}, cs::FromSourceDim{2});
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
            this->declare_input(pts_, "pts", cs::FromSourceDim{0}, cs::FromSourceDim{1}, cs::FromSourceDim{2});
        else
            this->declare_input(pts_, "pts", cs::FromSourceDim{0}, cs::FromSourceDim{1});
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
        add_subsystem<Counts>("counts", n_);
        add_subsystem<Fill>("fill");
        add_subsystem<Scale>("scale");
        add_subsystem<Sum<3>>("sum");
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
        add_subsystem<Make>("make", n_, make_);
        add_subsystem<TakeSum>("take", n_, take_);
        connect("make.m", "take.pts");
    }
};

/// make -> sum<2> (inherits with FromSourceDim), the Slice of `slice` between them.
class SliceTop : public cs::System {
public:
    void setup() override {
        add_subsystem<Make>("make", std::size_t{4}, Sizes{1, 0, 3, 2});
        add_subsystem<Sum<2>>("sum");
        connect("make.m", "sum.pts", cs::Slice{cs::subarray<2>({0, 0}, {2, 1})});
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
        add_subsystem<DenseGen>("gen");
        add_subsystem<Sum<2>>("sum");
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
/// A dense INPUT that inherits a ragged source (FromSourceDim).
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
    void setup() override { declare_input(in_, "in", cs::FromSourceDim{0}, cs::FromSourceDim{1}); }
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
        declare_input(in_, "in", cs::FromSourceDim{0}, cs::FromSourceDim{1});
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
        add_subsystem<Make>("make", std::size_t{4}, Sizes{1, 0, 3, 2});
        add_subsystem<C>("c");
        connect("make.m", "c.in");
    }
};
template <class C>
class SingleTop : public cs::System {
public:
    void setup() override { add_subsystem<C>("c"); }
};
class WaveTop : public cs::FusedSystem {
public:
    WaveTop() : cs::FusedSystem(Mode::sequential, {}) {}
    void setup() override { add_subsystem<Make>("make", std::size_t{4}, Sizes{1, 0, 3, 2}); }
};
class ParTop : public cs::ParallelSystem {
public:
    void setup() override { add_subsystem<Make>("make", std::size_t{4}, Sizes{1, 0, 3, 2}); }
};
template <class W>
class HostTop : public cs::System {
public:
    void setup() override { add_subsystem<W>("w"); }
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

/// Run one forward with `counts` and check every view against the references.
void run_counts(cs::Model& model, const std::vector<std::uint32_t>& counts) {
    if (!counts.empty()) model.init_input("counts.raw", cs::init::from_host(counts));
    model.forward();
    model.wait();
    const FillRef ref = fill_ref(counts);

    // the dense sums of the scaled blocks
    auto sums = model.view_output<DType::f32, 1>("sum.sums");
    REQUIRE(sums.size() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        float want = 0.0f;
        for (std::uint64_t q = ref.offsets[i]; q < ref.offsets[i + 1]; ++q) want += 2.0f * ref.data[q];
        INFO("block " << i << " (count " << counts[i] << ")");
        CHECK(sums.span()[i] == want);
    }

    // the ragged output itself, and its peer
    auto pts = model.view_ragged_output<DType::f32, 3>("fill.pts");
    CHECK(pts.size() == counts.size());
    CHECK(pts.offsets == ref.offsets);
    CHECK(pts.data == ref.data);
    CHECK(pts.extents(1) == Sizes(counts.begin(), counts.end()));
    CHECK(pts.extents(2) == Sizes(counts.size(), 3));
    CHECK(pts.is_per_index(1));
    CHECK_FALSE(pts.is_per_index(2));
    for (std::size_t i = 0; i < counts.size(); ++i) {
        auto b = pts.block(i);
        REQUIRE(b.shape()[0] == counts[i]);
        REQUIRE(b.shape()[1] == 3);
        for (std::size_t j = 0; j < counts[i]; ++j)
            for (std::size_t k = 0; k < 3; ++k) CHECK(b[sycl::id<2>{j, k}] == static_cast<float>(i * 1000 + j * 10 + k));
    }
    auto scaled = model.view_ragged_output<DType::f32, 3>("scale.out");
    CHECK(scaled.offsets == ref.offsets);
    REQUIRE(scaled.data.size() == ref.data.size());
    for (std::size_t q = 0; q < ref.data.size(); ++q) CHECK(scaled.data[q] == 2.0f * ref.data[q]);
    auto peer = model.view_ragged_input<DType::f32, 3>("scale.in");  // an input peer of the root
    CHECK(peer.data == ref.data);
    CHECK(peer.extents(1) == pts.extents(1));
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
    cs::Model model = cs::create_model<trag::RaggedTop>(default_config(), std::size_t{0});
    model.setup();
    trag::run_counts(model, {});
    auto pts = model.view_ragged_output<DType::f32, 3>("fill.pts");
    CHECK(pts.size() == 0);
    CHECK(pts.offsets == std::vector<std::uint64_t>{0});
    CHECK(pts.data.empty());
}

TEST_CASE("Ragged: FromInputValues on a rank-2 [N,k] input, column 1", "[ragged]") {
    cs::Model model = cs::create_model<trag::SingleTop<trag::Col1>>(default_config());
    model.setup();
    model.init_input("c.cnt", cs::init::from_host(std::vector<std::uint32_t>{9, 2, 9, 0, 9, 3, 9, 1}));
    model.forward();
    model.wait();
    auto v = model.view_ragged_output<DType::f32, 2>("c.v");
    CHECK(v.size() == 4);
    CHECK(v.offsets == std::vector<std::uint64_t>{0, 2, 2, 5, 6});
    CHECK(v.extents(1) == trag::Sizes{2, 0, 3, 1});
    CHECK(v.data == std::vector<float>{0, 1, 200, 201, 202, 300});
}

TEST_CASE("Ragged: both trailing dims per-index, from two columns", "[ragged]") {
    cs::Model model = cs::create_model<trag::SingleTop<trag::Cols>>(default_config());
    model.setup();
    const auto run = [&](const std::vector<std::uint32_t>& ab) {
        model.init_input("c.ab", cs::init::from_host(ab));
        model.forward();
        model.wait();
        auto t = model.view_ragged_output<DType::f32, 3>("c.t");
        REQUIRE(t.size() == 4);
        trag::Sizes        e1, e2;
        std::vector<float> data;
        std::vector<std::uint64_t> offs{0};
        for (std::size_t i = 0; i < 4; ++i) {
            e1.push_back(ab[2 * i]);
            e2.push_back(ab[2 * i + 1]);
            for (std::size_t j = 0; j < ab[2 * i]; ++j)
                for (std::size_t k = 0; k < ab[2 * i + 1]; ++k) data.push_back(static_cast<float>(i * 1000 + j * 10 + k));
            offs.push_back(data.size());
        }
        CHECK(t.extents(1) == e1);
        CHECK(t.extents(2) == e2);
        CHECK(t.is_per_index(1));
        CHECK(t.is_per_index(2));
        CHECK(t.offsets == offs);
        CHECK(t.data == data);
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(t.block(i).shape()[0] == e1[i]);
            CHECK(t.block(i).shape()[1] == e2[i]);
        }
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
    auto sums = model.view_output<DType::f32, 1>("take.sums");
    REQUIRE(sums.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        float want = 0.0f;
        for (std::size_t j = 0; j < sizes[i]; ++j) want += static_cast<float>(i * 100 + j);
        CHECK(sums.span()[i] == want);
    }
    auto m = model.view_ragged_output<DType::f32, 2>("make.m");
    CHECK(m.offsets == std::vector<std::uint64_t>{0, 1, 1, 4, 6});
    CHECK(m.extents(1) == sizes);
    CHECK(m.data == std::vector<float>{0, 200, 201, 202, 300, 301});
    // N = 0 with a PerIndex array of length 0 is ragged-capable too
    cs::Model zero = cs::create_model<trag::PerIndexTop>(default_config(), std::size_t{0}, trag::Sizes{}, trag::Sizes{});
    zero.setup();
    zero.forward();
    zero.wait();
    auto z = zero.view_ragged_output<DType::f32, 2>("make.m");
    CHECK(z.size() == 0);
    CHECK(z.offsets == std::vector<std::uint64_t>{0});
}

TEST_CASE("Ragged: mismatching declared per-index arrays error at forward", "[ragged][errors]") {
    SECTION("same total, different extents") {
        cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), std::size_t{4}, trag::Sizes{1, 0, 3, 2},
                                                              trag::Sizes{2, 1, 2, 1});
        model.setup();
        trag::expect_throws([&] { model.forward(); }, {"'pts'", "per-index extents of dimension 1"});
        model.wait();  // the source already launched
    }
    SECTION("a different total") {
        cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), std::size_t{4}, trag::Sizes{1, 0, 3, 2},
                                                              trag::Sizes{1, 1, 1, 1});
        model.setup();
        trag::expect_throws([&] { model.forward(); }, {"'pts'", "per-index extents"});
        model.wait();
    }
}

TEST_CASE("Ragged: a RaggedInput reading a DENSE port", "[ragged]") {
    cs::Model model = cs::create_model<trag::DenseTop>(default_config());
    model.setup();
    model.forward();
    model.wait();
    auto sums = model.view_output<DType::f32, 1>("sum.sums");
    REQUIRE(sums.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) CHECK(sums.span()[i] == static_cast<float>(40 * i + 6));
    // view_ragged_* on a dense entry: offsets are the row starts
    auto d = model.view_ragged_output<DType::f32, 2>("gen.d");
    CHECK(d.size() == 3);
    CHECK(d.offsets == std::vector<std::uint64_t>{0, 4, 8, 12});
    CHECK(d.extents(1) == trag::Sizes{4, 4, 4});
    CHECK_FALSE(d.is_per_index(1));
    CHECK(d.block(2)[std::size_t{3}] == 23.0f);
    auto in = model.view_ragged_input<DType::f32, 2>("sum.pts");
    CHECK(in.data == d.data);
    CHECK(in.offsets == d.offsets);
    // the dense view of a dense entry is unchanged
    CHECK(model.view_output<DType::f32, 2>("gen.d").span()[5] == 11.0f);
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
    SECTION("a dense input inheriting a ragged source") {
        cs::Model model = cs::create_model<trag::InheritTop<trag::DenseIn>>(default_config());
        model.setup();
        trag::expect_throws([&] { model.forward(); }, {"'in' is ragged; declare it with RaggedInput/RaggedOutput"});
        model.wait();
    }
    SECTION("a dense output inheriting a per-index dim of a ragged input") {
        cs::Model model = cs::create_model<trag::InheritTop<trag::DenseOut>>(default_config());
        model.setup();
        trag::expect_throws([&] { model.forward(); }, {"'o' is ragged; declare it with RaggedInput/RaggedOutput"});
        model.wait();
    }
    SECTION("a per-index length different from N") {
        cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), std::size_t{4}, trag::Sizes{1, 2, 3},
                                                              trag::Sizes{1, 2, 3, 4});
        model.setup();
        trag::expect_throws([&] { model.forward(); },
                            {"make.m", "per-index dimension 1 has 3 extents but dimension 0 has 4"});
    }
    SECTION("negative extents") {
        cs::Model model = cs::create_model<trag::SingleTop<trag::Neg>>(default_config());
        model.setup();
        model.init_input("c.cnt", cs::init::from_host(std::vector<std::int32_t>{1, -1, 2}));
        trag::expect_throws([&] { model.forward(); }, {"negative extent"});
    }
    SECTION("a Slice onto a ragged port") {
        cs::Model model = cs::create_model<trag::SliceTop>(default_config());
        trag::expect_throws([&] { model.setup(); }, {"ragged", "Slice"});
    }
    SECTION("a ragged port in a FusedSystem") {
        cs::Model model = cs::create_model<trag::HostTop<trag::WaveTop>>(default_config());
        trag::expect_throws([&] { model.setup(); },
                            {"ragged ports are not supported in fused/parallel systems yet"});
    }
    SECTION("a ragged port in a ParallelSystem") {
        cs::Model model = cs::create_model<trag::HostTop<trag::ParTop>>(default_config());
        trag::expect_throws([&] { model.setup(); },
                            {"ragged ports are not supported in fused/parallel systems yet"});
    }
}
