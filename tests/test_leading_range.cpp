// D140 B1: runtime leading-range views (the plumbing the chunked system will use), and size-0 fixed roots.
//
// A consumer's connected input is pointed, between forwards, at rows / blocks [a, b) of its root through
// detail::LeadingRangeAccess (the framework-only hook); the consumer's results are checked against host
// references. Upstream roots: dense [N, 3] and ragged [N, c_i, 3]; consumers: a dense Input, and a
// RaggedInput reading a dense port or a ragged one.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "replay_oracle.hpp"
#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace tlr {

using Sizes = std::vector<std::size_t>;

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

// ---- upstream roots ---------------------------------------------------------------------------------
/// Dense [n, 3]: out[i, j] = 10 i + j.
struct DenseSrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> out;
    };
    void operator()(sycl::id<2> i, Inputs, Outputs o) const { o.out[i] = static_cast<float>(i[0] * 10 + i[1]); }
};
class DenseSrc : public cs::Component<DenseSrcBody> {
    std::size_t               n_;
    cs::Output<DType::f32, 2> out_;

public:
    explicit DenseSrc(std::size_t n) : n_{n} {}
    void setup() override { declare_output(out_, "out", n_, 3); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {}, {out_}, {});
    }
};

/// Ragged [n, c_i, 3] (c = a host array): out.block(i)[j, k] = 1000 i + 10 j + k.
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
    Sizes                           sizes_;
    cs::RaggedOutput<DType::f32, 3> out_;

public:
    explicit RaggedSrc(Sizes s) : sizes_{std::move(s)} {}
    void setup() override { declare_output(out_, "out", sizes_.size(), cs::per_index(sizes_), 3); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{out_.size()}, {}, {out_}, {});  // the chunk's blocks inside a ChunkedSystem
    }
};

// ---- consumers: sums[r] = the sum of row / block r of the range their input sees ---------------------
struct DenseSumBody {
    struct Inputs {
        cs::Input<DType::f32, 2> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> sums;
    };
    void operator()(sycl::id<1> id, Inputs i, Outputs o) const {
        float s = 0.0f;
        for (std::size_t j = 0; j < i.in.shape()[1]; ++j) s += i.in[sycl::id<2>{id[0], j}];
        o.sums[id] = s;
    }
};
class DenseSum : public cs::Component<DenseSumBody> {
    cs::Input<DType::f32, 2>  in_;
    cs::Output<DType::f32, 1> sums_;

public:
    void setup() override {
        declare_input(in_, "in", cs::FromSourceDim{0}, cs::FromSourceDim{1});
        declare_output(sums_, "sums", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sums_.shape(), {in_}, {sums_}, {});
    }
};

template <int R>
struct RSumBody {
    struct Inputs {
        cs::RaggedInput<DType::f32, R> in;
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
template <int R>
class RSum : public cs::Component<RSumBody<R>> {
    cs::RaggedInput<DType::f32, R> in_;
    cs::Output<DType::f32, 1>      sums_;

public:
    void setup() override {
        if constexpr (R == 3)
            this->declare_input(in_, "in", cs::FromSourceDim{0}, cs::FromSourceDim{1}, cs::FromSourceDim{2});
        else
            this->declare_input(in_, "in", cs::FromSourceDim{0}, cs::FromSourceDim{1});
        this->declare_output(sums_, "sums", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return this->create_launch_config(sums_.shape(), {in_}, {sums_}, {});
    }
};

enum class Kind { dense_dense, dense_ragged, ragged_ragged };

class LRTop : public cs::System {
    Kind        kind_;
    std::size_t n_;
    Sizes       sizes_;

public:
    LRTop(Kind k, std::size_t n, Sizes sizes) : kind_{k}, n_{n}, sizes_{std::move(sizes)} {}
    void setup() override {
        switch (kind_) {
            case Kind::dense_dense:
                add_subsystem<DenseSrc>("src", n_);
                add_subsystem<DenseSum>("cons");
                break;
            case Kind::dense_ragged:
                add_subsystem<DenseSrc>("src", n_);
                add_subsystem<RSum<2>>("cons");
                break;
            case Kind::ragged_ragged:
                add_subsystem<RaggedSrc>("src", sizes_);
                add_subsystem<RSum<3>>("cons");
                break;
        }
        connect("src.out", "cons.in");
    }
};

// ---- references ------------------------------------------------------------------------------------------
float ref_sum(Kind kind, const Sizes& sizes, std::size_t r) {
    float s = 0.0f;
    if (kind == Kind::ragged_ragged) {
        for (std::size_t j = 0; j < sizes[r]; ++j)
            for (std::size_t k = 0; k < 3; ++k) s += static_cast<float>(r * 1000 + j * 10 + k);
    } else {
        for (std::size_t j = 0; j < 3; ++j) s += static_cast<float>(r * 10 + j);
    }
    return s;
}

/// The flat data of blocks [a, b) of the ragged source, and their rebased offsets.
void ref_ragged_range(const Sizes& sizes, std::size_t a, std::size_t b, std::vector<float>& data,
                      std::vector<std::uint64_t>& offs) {
    data.clear();
    offs = {0};
    for (std::size_t i = a; i < b; ++i) {
        for (std::size_t j = 0; j < sizes[i]; ++j)
            for (std::size_t k = 0; k < 3; ++k) data.push_back(static_cast<float>(i * 1000 + j * 10 + k));
        offs.push_back(data.size());
    }
}

/// Point the consumer's input at [a, b) (or clear it), run one forward, check the sums and the host views.
void run_range(cs::Model& m, Kind kind, const Sizes& sizes, std::size_t a, std::size_t b, bool clear = false) {
    cs::detail::IOEntry& in = m.root().entry_at("cons.in");
    if (clear)
        cs::detail::LeadingRangeAccess::clear(in);
    else
        cs::detail::LeadingRangeAccess::set(in, a, b);
    m.forward();
    m.wait();
    INFO("range [" << a << ", " << b << ")" << (clear ? " (cleared)" : ""));
    auto sums = m.view_output<DType::f32, 1>("cons.sums");
    REQUIRE(sums.size() == b - a);
    for (std::size_t r = a; r < b; ++r) CHECK(sums.span()[r - a] == ref_sum(kind, sizes, r));

    // the consumer's own view of its range
    if (kind == Kind::ragged_ragged) {
        auto v = m.view_ragged_input<DType::f32, 3>("cons.in");
        std::vector<float>         data;
        std::vector<std::uint64_t> offs;
        ref_ragged_range(sizes, a, b, data, offs);
        CHECK(v.size() == b - a);
        CHECK(v.offsets == offs);
        CHECK(v.data == data);
        Sizes want(sizes.begin() + static_cast<std::ptrdiff_t>(a), sizes.begin() + static_cast<std::ptrdiff_t>(b));
        CHECK(v.extents(1) == want);
    } else if (kind == Kind::dense_dense) {
        auto v = m.view_input<DType::f32, 2>("cons.in");
        REQUIRE(v.size() == (b - a) * 3);
        for (std::size_t r = a; r < b; ++r)
            for (std::size_t j = 0; j < 3; ++j) CHECK(v.span()[(r - a) * 3 + j] == static_cast<float>(r * 10 + j));
    } else {
        auto v = m.view_ragged_input<DType::f32, 2>("cons.in");
        CHECK(v.size() == b - a);
        CHECK(v.offsets.back() == (b - a) * 3);
    }
}

void run_all_ranges(Kind kind, const Sizes& sizes, std::size_t n) {
    cs::Model m = cs::create_model<LRTop>(default_config(), kind, n, sizes);
    m.setup();
    m.forward();  // the whole range first: the producer has run, the consumer is unranged
    m.wait();
    const std::size_t k = n / 2;
    const std::vector<std::pair<std::size_t, std::size_t>> ranges{{0, n},     {0, 1},     {2, 5},     {n - 1, n},
                                                                  {k, k},     {4, 5},     {1, n - 1}, {0, 0},
                                                                  {n, n},     {0, n}};
    for (const auto& [a, b] : ranges) run_range(m, kind, sizes, a, b);
    // clearing restores the whole range
    run_range(m, kind, sizes, 0, n, /*clear=*/true);
    cs::detail::LeadingRangeAccess::set(m.root().entry_at("cons.in"), 2, 4);
    cs::detail::LeadingRangeAccess::clear(m.root().entry_at("cons.in"));
    CHECK_FALSE(m.root().entry_at("cons.in").leading_range().has_value());
    run_range(m, kind, sizes, 0, n, /*clear=*/true);

    // validation errors name the input and leave the previous range in place
    cs::detail::IOEntry& in = m.root().entry_at("cons.in");
    expect_throws([&] { cs::detail::LeadingRangeAccess::set(in, 4, 2); }, {"'in'", "a > b"});
    expect_throws([&] { cs::detail::LeadingRangeAccess::set(in, 0, n + 1); }, {"'in'", "past the"});
    expect_throws([&] { cs::detail::LeadingRangeAccess::set(in, n + 1, n + 1); }, {"'in'", "past the"});
    CHECK_FALSE(in.leading_range().has_value());
    // a ROOT is not a connected input
    expect_throws([&] { cs::detail::LeadingRangeAccess::set(m.root().entry_at("src.out"), 0, 1); }, {"connected input"});
    run_range(m, kind, sizes, 1, 3);  // still usable after the errors
}

// ---- a fixed [0] root --------------------------------------------------------------------------------------
struct ZeroSrcBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs o) const { o.out[i] = 1.0f; }
};
class ZeroSrc : public cs::Component<ZeroSrcBody> {
    cs::Output<DType::f32, 1> out_;

public:
    void setup() override { declare_output(out_, "out", 0); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {}, {out_}, {});
    }
};
struct CopyBody {
    struct Inputs {
        cs::Input<DType::f32, 1> in;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> out;
    };
    void operator()(sycl::id<1> i, Inputs a, Outputs o) const { o.out[i] = 2.0f * a.in[i]; }
};
class Copy : public cs::Component<CopyBody> {
    cs::Input<DType::f32, 1>  in_;
    cs::Output<DType::f32, 1> out_;

public:
    void setup() override {
        declare_input(in_, "in", cs::FromSourceDim{0});
        declare_output(out_, "out", cs::FromInputDim{"in", 0});
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(out_.shape(), {in_}, {out_}, {});
    }
};
class ZeroTop : public cs::System {
public:
    void setup() override {
        add_subsystem<ZeroSrc>("src");
        add_subsystem<Copy>("cons");
        connect("src.out", "cons.in");
    }
};

}  // namespace tlr

TEST_CASE("Leading range: dense [N,3] root, dense consumer", "[leading_range]") {
    tlr::run_all_ranges(tlr::Kind::dense_dense, {}, 6);
}

TEST_CASE("Leading range: dense [N,3] root, RaggedInput consumer", "[leading_range]") {
    tlr::run_all_ranges(tlr::Kind::dense_ragged, {}, 6);
}

TEST_CASE("Leading range: ragged [N,c_i,3] root, RaggedInput consumer", "[leading_range]") {
    // empty blocks first, in the middle and last
    tlr::run_all_ranges(tlr::Kind::ragged_ragged, {0, 3, 0, 2, 4, 0}, 6);
}

TEST_CASE("Leading range: ragged root with all-empty blocks and with no empty ones", "[leading_range]") {
    tlr::run_all_ranges(tlr::Kind::ragged_ragged, {0, 0, 0, 0, 0, 0}, 6);
    tlr::run_all_ranges(tlr::Kind::ragged_ragged, {1, 2, 3, 1, 2, 3}, 6);
}

TEST_CASE("Size-0 fixed roots: an empty root consumed downstream; the views are empty", "[ragged][empty]") {
    cs::Model m = cs::create_model<tlr::ZeroTop>(default_config());
    m.setup();
    m.forward();
    m.wait();
    CHECK(m.view_output<DType::f32, 1>("src.out").size() == 0);
    CHECK(m.view_input<DType::f32, 1>("cons.in").size() == 0);
    CHECK(m.view_output<DType::f32, 1>("cons.out").size() == 0);
    m.forward();  // and again
    m.wait();
    CHECK(m.view_output<DType::f32, 1>("cons.out").size() == 0);
}


// =============================================================================================================================
// The replay ORACLE (C2c) on the leading-range models wrapped in a ChunkedSystem: a boundary input that is a dense or a ragged
// OUTSIDE root (the consumer is pointed at the chunk's range), an inside output consumed outside (RE-ROOTED), and a ragged
// root with host per-index arrays declared inside (cs::per_index on a component port). (A re-rooted free INPUT is the promoted
// free input of "Replay oracle: a promoted free input", in test_chunked.cpp.)
// =============================================================================================================================
namespace tlr {

/// cons (the model's consumer) -> cp (an inside output, re-rooted: consumed outside). `local`: the ragged source is inside too.
class LRInner : public cs::ChunkedSystem {
    Kind  kind_;
    bool  local_;
    Sizes sizes_;

public:
    LRInner(std::size_t budget, Kind k, bool local, Sizes sizes)
        : cs::ChunkedSystem(budget), kind_{k}, local_{local}, sizes_{std::move(sizes)} {}
    void setup() override {
        if (local_) {
            add_subsystem<RaggedSrc>("rs", sizes_);
            add_subsystem<RSum<3>>("cons");
            connect("rs.out", "cons.in");
            add_subsystem<DenseSum>("ds");  // reads the outside dense source: the chunk length comes from a boundary root
        } else if (kind_ == Kind::dense_dense) {
            add_subsystem<DenseSum>("cons");
        } else if (kind_ == Kind::dense_ragged) {
            add_subsystem<RSum<2>>("cons");
        } else {
            add_subsystem<RSum<3>>("cons");
        }
        add_subsystem<Copy>("cp");
        connect("cons.sums", "cp.in");
    }
};
class LRChunkTop : public cs::System {
    Kind        kind_;
    bool        local_;
    std::size_t n_, budget_;
    Sizes       sizes_;

public:
    LRChunkTop(Kind k, bool local, std::size_t n, Sizes sizes, std::size_t budget)
        : kind_{k}, local_{local}, n_{n}, budget_{budget}, sizes_{std::move(sizes)} {}
    void setup() override {
        if (local_) {
            add_subsystem<DenseSrc>("src", n_);
        } else {
            if (kind_ == Kind::ragged_ragged)
                add_subsystem<RaggedSrc>("src", sizes_);
            else
                add_subsystem<DenseSrc>("src", n_);
        }
        add_subsystem<LRInner>("inner", budget_, kind_, local_, sizes_);
        add_subsystem<Copy>("fin");
        connect("src.out", local_ ? "inner.ds.in" : "inner.cons.in");
        connect("inner.cp.out", "fin.in");
    }
};
}  // namespace tlr

TEST_CASE("Replay oracle: leading-range models in a ChunkedSystem (dense / ragged outside roots, re-rooted output)",
          "[leading_range][chunked][replay]") {
    const tlr::Sizes sizes{0, 3, 0, 2, 4, 0};
    const auto       read = [](cs::Model& m) {
        auto v = m.view_output<DType::f32, 1>("fin.out");
        return std::vector<float>(v.span().begin(), v.span().end());
    };
    for (const std::size_t budget : {std::size_t{4}, std::size_t{12}, std::size_t{1000}}) {
        INFO("budget " << budget);
        tor::oracle("inner", [&] { return cs::create_model<tlr::LRChunkTop>(default_config(), tlr::Kind::dense_dense, false, std::size_t{6}, sizes, budget); }, read);
        tor::oracle("inner", [&] { return cs::create_model<tlr::LRChunkTop>(default_config(), tlr::Kind::dense_ragged, false, std::size_t{6}, sizes, budget); }, read);
        tor::oracle("inner", [&] { return cs::create_model<tlr::LRChunkTop>(default_config(), tlr::Kind::ragged_ragged, false, std::size_t{6}, sizes, budget); }, read);
    }
    // all-empty and no-empty ragged blocks
    for (const tlr::Sizes& s : {tlr::Sizes{0, 0, 0, 0, 0, 0}, tlr::Sizes{1, 2, 3, 1, 2, 3}})
        tor::oracle("inner", [&] { return cs::create_model<tlr::LRChunkTop>(default_config(), tlr::Kind::ragged_ragged, false, std::size_t{6}, s, std::size_t{8}); }, read);
}

TEST_CASE("Replay oracle: an internal ragged port with host per-index arrays (cs::per_index on a component port)",
          "[port_per_index][leading_range][chunked][replay]") {
    // The full-length host array of rs.out is cut to each chunk [start, start + dim0) (C2d), as a tie alias''s is.
    const auto read = [](cs::Model& m) {
        auto v = m.view_output<DType::f32, 1>("fin.out");
        return std::vector<float>(v.span().begin(), v.span().end());
    };
    // per index: 4 B (cons.sums) + 4 B (ds.sums) + 12 B per row of rs.out (at most 4 rows: 56 B)
    for (const tlr::Sizes& s : {tlr::Sizes{0, 3, 0, 2, 4, 0}, tlr::Sizes{1, 2, 3, 1, 2, 3}})
        for (const std::size_t budget : {std::size_t{56}, std::size_t{100}, std::size_t{1000}}) {
            INFO("budget " << budget);
            tor::oracle("inner", [&] { return cs::create_model<tlr::LRChunkTop>(default_config(), tlr::Kind::ragged_ragged, true, s.size(), s, budget); }, read);
        }
}
