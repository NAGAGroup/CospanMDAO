// Ragged dimensions: the full test set, on the section 10 shape model.
//
//   Counts   u32 [N] counts (a copy of a free input of fixed size N)
//   Fill     RaggedOutput<f32,2> "pts": N blocks of counts[i] elements, pts.block(i)[j] = i*1000 + j. Its shape comes from
//            a hooked "layout" input (a System shape hook, cs::ragged::counts_from("counts.counts")) that Fill copies in
//            pre_forward with set_shape(pts, shape_of(layout))
//   Scale    RaggedInput -> RaggedOutput (set_shape(out, shape_of(in))): x2
//   Sum      RaggedInput -> a dense [N] output of block sums
//
// Everything is checked against host references, through view_output (dense) and view_ragged_* (n(), block_offsets(),
// block(i), span()), synced after the forward (docs/design-review-2026-10-03.md, section 5).
//
// PORT NOTES (design review, section 10). Ragged is RANK 2 only, a per-index dimension is declared with cs::counts / cs::offsets
// (the length of the counts is N), there are no derived dimensions (FromInputDim, FromInputValues, ...), and an output is
// reshaped only by its producing component's pre_forward. DELETED: the rank-3 cases (Fill's 3-wide rows, "both trailing dims
// per-index, from two columns", the Cols component) and the declaration errors that can no longer be written ("a per-index spec
// on dimension 0", "a dense wrapper declared with a per-index spec": cs::counts is not a DimSpec, so they do not compile) and
// "a dense output inheriting a per-index dim" (a dense set_shape carries no per-index extents). The old "a per-index length
// different from N" is converted into a connect-time mismatch of N.

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

// N is fixed by the declaration of the free input `raw` (the old free input `n` that sized it by value is gone: a size
// from a value is a shape hook now, covered in test_shapes.cpp).
class Counts : public cs::Component<Counts> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> raw;
    };
    struct Outputs {
        cs::Output<DType::u32, 1> counts;
    };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.counts[i] = in.raw[i]; }

    Inputs  inputs;
    Outputs outputs;

private:
    std::size_t               n_;

public:
    explicit Counts(std::size_t n) : n_{n} {}
    void setup() override {
        declare_input(inputs.raw, "raw", n_);
        declare_output(outputs.counts, "counts", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.counts, shape_of(inputs.raw)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.counts.shape()));
    }
};

// ---- Fill: pts.block(i)[j] = i*1000 + j ----------------------------------------------------------

// Fill tests a hook on a ragged INPUT: a component reads no values, so the size comes in through a free ragged input
// `layout` that the System hooks with counts_from, and Fill copies its shape. (Hooks on ragged outputs are tested in
// test_shapes.cpp.)
class Fill : public cs::Component<Fill> {
public:
    struct Inputs {
        cs::RaggedInput<DType::f32, 2> layout;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> pts;
    };
    static void compute_impl(sycl::id<1> id, const Inputs&, const Outputs& out) {
        const std::size_t i = id[0];
        auto              b = out.pts.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j) b[j] = static_cast<float>(i * 1000 + j);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.layout, "layout");
        declare_output(outputs.pts, "pts");
    }
    void pre_forward() override { set_shape(outputs.pts, shape_of(inputs.layout)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{inputs.layout.size()}));
    }
};

// ---- Scale: out = 2 * in, the output takes the input's ragged shape ---------------------------------------

class Scale : public cs::Component<Scale> {
public:
    struct Inputs {
        cs::RaggedInput<DType::f32, 2> in;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> out;
    };
    static void compute_impl(sycl::id<1> id, const Inputs& in, const Outputs& out) {
        auto bi = in.in.block(id[0]);
        auto bo = out.out.block(id[0]);
        for (std::size_t q = 0; q < bi.size(); ++q) bo[q] = 2.0f * bi[q];
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.in, "in");
        declare_output(outputs.out, "out");
    }
    void pre_forward() override { set_shape(outputs.out, shape_of(inputs.in)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{inputs.in.size()}));
    }
};

// ---- Sum: sums[i] = the sum of every element of block i ---------------------------------------------------
template <int R>
class Sum : public cs::Component<Sum<R>> {
public:
    struct Inputs {
        cs::RaggedInput<DType::f32, R> pts;
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
        this->declare_input(inputs.pts, "pts");
        this->declare_output(outputs.sums, "sums", cs::unspecified);
    }
    void pre_forward() override { this->set_shape(outputs.sums, this->shape_of(inputs.pts).n); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(cs::range_desc(outputs.sums.shape()));
    }
};

/// counts -> fill -> scale -> sum (n counts, free input "counts.raw"); fill's layout is hooked from counts.counts.
class RaggedTop : public cs::System {
    std::size_t n_;

public:
    explicit RaggedTop(std::size_t n) : n_{n} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Counts>("counts", n_));
        add_subsystem(cs::subsystem_desc<Fill>("fill"));
        add_subsystem(cs::subsystem_desc<Scale>("scale"));
        add_subsystem(cs::subsystem_desc<Sum<2>>("sum"));
        shape_hook("fill.layout", cs::ragged::counts_from("counts.counts"));
        connect("fill.pts", "scale.in");
        connect("scale.out", "sum.pts");
    }
};

// ---- Col0: [N,2] input; the counts are COLUMN 0 (counts_from reads column 0 of a rank-2 source) --------------

class Col0 : public cs::Component<Col0> {
public:
    struct Inputs {
        cs::Input<DType::u32, 2>       cnt;
        cs::RaggedInput<DType::f32, 2> layout;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> v;
    };
    static void compute_impl(sycl::id<1> id, const Inputs&, const Outputs& out) {
        const std::size_t i = id[0];
        auto              b = out.v.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j) b[j] = static_cast<float>(i * 100 + j);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.cnt, "cnt", 4, 2);
        declare_input(inputs.layout, "layout");
        declare_output(outputs.v, "v");
    }
    void pre_forward() override { set_shape(outputs.v, shape_of(inputs.layout)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{inputs.cnt.shape()[0]}));
    }
};

// ---- Neg: an i32 input of extents; a negative one is an error ---------------------------------------------------

class Neg : public cs::Component<Neg> {
public:
    struct Inputs {
        cs::Input<DType::i32, 1>       cnt;
        cs::RaggedInput<DType::f32, 2> layout;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> v;
    };
    static void compute_impl(sycl::id<1>, const Inputs&, const Outputs&) {}

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.cnt, "cnt", 3);
        declare_input(inputs.layout, "layout");
        declare_output(outputs.v, "v");
    }
    void pre_forward() override { set_shape(outputs.v, shape_of(inputs.layout)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(inputs.cnt.shape()));
    }
};

/// One child "c" whose ragged input `layout` is hooked from its own input `cnt`.
template <class C>
class HookedTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<C>("c"));
        shape_hook("c.layout", cs::ragged::counts_from("c.cnt"));
    }
};

// ---- Make: a cs::counts host-array port of n blocks; m.block(i)[j] = i*100 + j -----------------------------------

class Make : public cs::Component<Make> {
public:
    struct Inputs {};
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> m;
    };
    static void compute_impl(sycl::id<1> id, const Inputs&, const Outputs& out) {
        const std::size_t i = id[0];
        auto              b = out.m.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j) b[j] = static_cast<float>(i * 100 + j);
    }

    Inputs  inputs;
    Outputs outputs;

private:
    Sizes                           sizes_;

public:
    explicit Make(Sizes sizes) : sizes_{std::move(sizes)} {}
    void setup() override { declare_output(outputs.m, "m", cs::counts(sizes_)); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{sizes_.size()}));
    }
};

/// Sums a rank-2 ragged input that DECLARES its own counts (merged with the source's at connect).
class TakeSum : public Sum<2> {
    Sizes sizes_;

public:
    explicit TakeSum(Sizes sizes) : sizes_{std::move(sizes)} {}
    void setup() override {
        declare_input(inputs.pts, "pts", cs::counts(sizes_));
        declare_output(outputs.sums, "sums", cs::unspecified);
    }
};

/// make (blocks of `make_sizes`) -> take (declares `take_sizes`).
class PerIndexTop : public cs::System {
    Sizes make_, take_;

public:
    PerIndexTop(Sizes make_sizes, Sizes take_sizes) : make_{std::move(make_sizes)}, take_{std::move(take_sizes)} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Make>("make", make_));
        add_subsystem(cs::subsystem_desc<TakeSum>("take", take_));
        connect("make.m", "take.pts");
    }
};

// ---- a RaggedInput reading a DENSE port ----------------------------------------------------------------------------

class DenseGen : public cs::Component<DenseGen> {
public:
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> d;
    };
    static void compute_impl(sycl::id<2> i, const Inputs&, const Outputs& out) { out.d[i] = static_cast<float>(i[0] * 10 + i[1]); }

    Inputs  inputs;
    Outputs outputs;

    void setup() override { declare_output(outputs.d, "d", 3, 4); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(outputs.d.shape()));
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
/// A dense INPUT that would take a ragged source (through unspecified dims).

class DenseIn : public cs::Component<DenseIn> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> in;
    };
    struct Outputs {};
    static void compute_impl(sycl::id<1>, const Inputs&, const Outputs&) {}

    Inputs  inputs;
    Outputs outputs;

    void setup() override { declare_input(inputs.in, "in", cs::unspecified, cs::unspecified); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{4}));
    }
};
template <class C>
class InheritTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Make>("make", Sizes{1, 0, 3, 2}));
        add_subsystem(cs::subsystem_desc<C>("c"));
        connect("make.m", "c.in");
    }
};
/// A fused component (a helper-built descriptor) whose only stage has a ragged output.
template <cst::FuseMode S>
class FusedRagged : public cs::System {
public:
    void setup() override { add_subsystem(cst::fuse<S>("w", cs::subsystem_desc<Make>("make", Sizes{1, 0, 3, 2}))); }
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
        for (std::size_t j = 0; j < counts[i]; ++j) r.data.push_back(static_cast<float>(i * 1000 + j));
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
    model.run();
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
    auto pts = model.view_ragged_output<DType::f32, 2>("fill.pts");
    pts.sync();
    CHECK(pts.n() == counts.size());
    CHECK(pts.block_offsets() == ref.offsets);
    CHECK(flat_of(pts) == ref.data);
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const auto b = pts.block(i);  // a flat span: counts[i] elements
        REQUIRE(b.size() == counts[i]);
        for (std::size_t j = 0; j < counts[i]; ++j) CHECK(b[j] == static_cast<float>(i * 1000 + j));
    }
    auto scaled = model.view_ragged_output<DType::f32, 2>("scale.out");
    scaled.sync();
    CHECK(scaled.block_offsets() == ref.offsets);
    REQUIRE(scaled.size() == ref.data.size());
    for (std::size_t q = 0; q < ref.data.size(); ++q) CHECK(scaled.span()[q] == 2.0f * ref.data[q]);
    auto peer = model.view_ragged_input<DType::f32, 2>("scale.in");  // an input peer of the root
    peer.sync();
    CHECK(flat_of(peer) == ref.data);
    CHECK(peer.block_offsets() == ref.offsets);
}

// ---- HookedOut: a ragged OUTPUT whose shape is owned by a hook (counts_from its own free input) -----------------------
// No set_shape: the hook gives the output its n, its counts and its offsets (a device column built on the source's device).
// v.block(i)[j] = i*100 + j.

class HookedOut : public cs::Component<HookedOut> {
public:
    struct Inputs {
        cs::Input<DType::u32, 1> cnt;
    };
    struct Outputs {
        cs::RaggedOutput<DType::f32, 2> v;
    };
    static void compute_impl(sycl::id<1> id, const Inputs&, const Outputs& out) {
        const std::size_t i = id[0];
        auto              b = out.v.block(i);
        for (std::size_t j = 0; j < b.shape()[0]; ++j) b[j] = static_cast<float>(i * 100 + j);
    }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.cnt, "cnt", cs::unspecified);
        declare_output(outputs.v, "v");
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{inputs.cnt.shape()[0]}));
    }
};
class HookedOutTop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<HookedOut>("c"));
        shape_hook("c.v", cs::ragged::counts_from("c.cnt"));
    }
};

}  // namespace trag

// A hook on a ragged OUTPUT: when the counts grow past the column's capacity the hook's column reallocates, and the event of
// that resize must reach the TARGET variable's resize-event slot (the producer, and any other writer of the target, waits on
// it). The producer's values and offsets stay right over forwards with growing and shrinking counts.
TEST_CASE("Ragged: a hook on a ragged output hands its column's resize events to the target's slot", "[ragged][hook][events]") {
    cs::Model model = cs::create_model<trag::HookedOutTop>(default_config());
    model.setup();

    const std::vector<std::vector<std::uint32_t>> runs{
        {1, 2},                                // the column is made (n = 2)
        {3, 1, 0, 2},                          // n = 4 > capacity 2: the column reallocates with data
        {1, 1, 1, 1, 1, 1, 1, 1, 1},           // n = 9
        {2, 0},                                // shrinks: within capacity, no event
        {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},  // n = 12 > capacity 9
    };
    const std::vector<bool> grows{false, true, true, false, true};
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const auto& counts = runs[r];
        INFO("forward " << r + 1 << ", n = " << counts.size());
        const auto        meta   = cs::detail::ModelAccess::instance(model, "c.v")->meta;
        const std::size_t before = meta->hook_resize_notes;

        model.update_input("c.cnt", cs::init::from_host(counts));
        model.run();
        model.wait();

        // the slot received the column's resize event exactly when the column reallocated with data
        if (grows[r]) CHECK(meta->hook_resize_notes > before);
        else CHECK(meta->hook_resize_notes == before);

        trag::FillRef ref;
        for (std::size_t i = 0; i < counts.size(); ++i) {
            for (std::size_t j = 0; j < counts[i]; ++j) ref.data.push_back(static_cast<float>(i * 100 + j));
            ref.offsets.push_back(ref.data.size());
        }
        auto v = model.view_ragged_output<DType::f32, 2>("c.v");
        v.sync();
        CHECK(v.n() == counts.size());
        CHECK(v.block_offsets() == ref.offsets);
        CHECK(trag::flat_of(v) == ref.data);
        for (std::size_t i = 0; i < counts.size(); ++i) CHECK(v.block(i).size() == counts[i]);
    }
}

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
    // n = 0 makes zero-size variables (raw, counts, layout, pts): each is present, with size 0 and a host copy of size 0.
    cs::Model model = cs::create_model<trag::RaggedTop>(default_config(), std::size_t{0});
    model.setup();
    trag::run_counts(model, {});
    auto pts = model.view_ragged_output<DType::f32, 2>("fill.pts");
    pts.sync();
    CHECK(pts.n() == 0);
    CHECK(pts.block_offsets() == std::vector<std::uint64_t>{0});
    CHECK(pts.size() == 0);
}

TEST_CASE("Ragged: counts_from on a rank-2 [N,k] input reads column 0", "[ragged]") {
    // counts_from takes no column argument: it reads column 0 of a rank-2 [N, k] input (shape_hook.hpp).
    cs::Model model = cs::create_model<trag::HookedTop<trag::Col0>>(default_config());
    model.setup();
    model.update_input("c.cnt", cs::init::from_host(std::vector<std::uint32_t>{2, 9, 0, 9, 3, 9, 1, 9}));
    model.run();
    model.wait();
    auto v = model.view_ragged_output<DType::f32, 2>("c.v");
    v.sync();
    CHECK(v.n() == 4);
    CHECK(v.block_offsets() == std::vector<std::uint64_t>{0, 2, 2, 5, 6});
    const trag::Sizes extents{2, 0, 3, 1};  // the blocks' sizes (the per-index extents of dimension 1)
    for (std::size_t i = 0; i < 4; ++i) CHECK(v.block(i).size() == extents[i]);
    CHECK(trag::flat_of(v) == std::vector<float>{0, 1, 200, 201, 202, 300});
}

TEST_CASE("Ragged: a ragged output connected to a ragged input with matching counts", "[ragged]") {
    const trag::Sizes sizes{1, 0, 3, 2};
    cs::Model         model = cs::create_model<trag::PerIndexTop>(default_config(), sizes, sizes);
    model.setup();
    model.run();
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
    // N = 0 with a counts array of length 0 is ragged-capable too
    cs::Model zero = cs::create_model<trag::PerIndexTop>(default_config(), trag::Sizes{}, trag::Sizes{});
    zero.setup();
    zero.run();
    zero.wait();
    auto z = zero.view_ragged_output<DType::f32, 2>("make.m");
    z.sync();
    CHECK(z.n() == 0);
    CHECK(z.block_offsets() == std::vector<std::uint64_t>{0});
}

TEST_CASE("Ragged: mismatching declared counts error at connect (setup)", "[ragged][errors][shape]") {
    // Design review, section 10: declared counts are definitions and merge at the connect: two specified values that
    // disagree are an error AT THE CALL (it was a forward-time error before).
    SECTION("same total, different counts") {
        cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), trag::Sizes{1, 0, 3, 2}, trag::Sizes{2, 1, 2, 1});
        trag::expect_throws([&] { model.setup(); },
                            {"connect('make.m', 'take.pts')", "dimension 1", "the counts must match", "[1,0,3,2]", "[2,1,2,1]"});
    }
    SECTION("a different total") {
        cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), trag::Sizes{1, 0, 3, 2}, trag::Sizes{1, 1, 1, 1});
        trag::expect_throws([&] { model.setup(); },
                            {"connect('make.m', 'take.pts')", "the counts must match", "[1,1,1,1]", "[1,0,3,2]"});
    }
    SECTION("a different number of indices") {
        // was "a per-index length different from N" (a forward-time error): N is the length of the counts, so the leading
        // dimensions disagree
        cs::Model model = cs::create_model<trag::PerIndexTop>(default_config(), trag::Sizes{1, 2, 3, 4}, trag::Sizes{1, 2, 3});
        trag::expect_throws([&] { model.setup(); },
                            {"connect('make.m', 'take.pts')", "dimension 0", "declares 4", "declares 3", "the extents must match"});
    }
}

TEST_CASE("Ragged: a RaggedInput reading a DENSE port", "[ragged]") {
    cs::Model model = cs::create_model<trag::DenseTop>(default_config());
    model.setup();
    model.run();
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
    trag::expect_throws([&] { (void)model.view_output<DType::f32, 2>("fill.pts"); }, {"fill.pts", "view_ragged_output"});
    trag::expect_throws([&] { (void)model.view_input<DType::f32, 2>("sum.pts"); }, {"sum.pts", "view_ragged_input"});
    // the wrong dtype or rank still names the view
    trag::expect_throws([&] { (void)model.view_ragged_output<DType::u32, 2>("fill.pts"); }, {"wrong dtype or rank"});
}

TEST_CASE("Ragged: declaration and binding errors", "[ragged][errors]") {
    SECTION("a dense input connected to a ragged source: the connect throws at setup") {
        // A dense Input cannot see per-index extents, and that is a wiring rule: connect checks it in its own call
        // (section 2; wiring.hpp, connect).
        cs::Model model = cs::create_model<trag::InheritTop<trag::DenseIn>>(default_config());
        trag::expect_throws([&] { model.setup(); },
                            {"connect('make.m', 'c.in')", "'make.m' is ragged", "a dense Input cannot see per-index extents", "RaggedInput"});
    }
    SECTION("negative extents") {
        cs::Model model = cs::create_model<trag::HookedTop<trag::Neg>>(default_config());
        model.setup();
        model.update_input("c.cnt", cs::init::from_host(std::vector<std::int32_t>{1, -1, 2}));
        // (the old needle was "negative extent"; the hook says "holds a negative value; a size cannot be negative")
        trag::expect_throws([&] { model.run(); }, {"shape hook on 'c.layout'", "holds a negative value"});
        model.wait();
    }
    SECTION("a ragged port in a fused component: refused at setup") {
        cs::Model model = cs::create_model<trag::FusedRagged<cst::FuseMode::seq>>(default_config());
        trag::expect_throws([&] { model.setup(); }, {"ragged ports are not supported in fused components yet"});
    }
    SECTION("a ragged port in a parallel fused component: refused at setup") {
        cs::Model model = cs::create_model<trag::FusedRagged<cst::FuseMode::par>>(default_config());
        trag::expect_throws([&] { model.setup(); }, {"ragged ports are not supported in fused components yet"});
    }
}
