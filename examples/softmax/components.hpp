#pragma once

// =============================================================================
// examples/softmax/components.hpp — row-wise softmax as CoSpan components
// =============================================================================
//
//   y[i][j] = exp(x[i][j] - max_i) / sum_j exp(x[i][j] - max_i)        x, y: n rows x m columns of f32
//
// Every kernel is rank 1, work-group size 256 (kWg), one work-group per row. All bodies are GROUP bodies.
//
// UNFUSED (three global-memory components, three launches; five passes over the matrix):
//   RowMax     reads x             writes mx[n]
//   ExpSum     reads x and mx      writes e = exp(x - mx) [n x m] and sum[n]
//   Normalize  reads e and sum     writes y
//
// FUSED (three stages of ONE launch, a single segment<local_mem::shared>; two passes over the matrix: read x, write y):
//   LoadMax        loads the row into local memory and reduces its max
//   ExpSumLocal    exponentiates the row IN LOCAL MEMORY and reduces the sum
//   NormalizeWrite divides the row in local memory by the sum and writes y
// The stages run one after the other inside a work-group, so the row stays in local memory from one to the next, and
// a stage that took several rows (a grid-stride loop over rows) would leave only the last one there: the fused
// launch has exactly one work-group per row, and each stage processes row = its group index. The stages call a group
// barrier before they read what the previous stage's other work-items wrote (the row's scalars: the max, the sum).
// The two scalars live in the same local block as the row, after it. LoadMax and ExpSumLocal also write their scalar
// to a small global output (mx[n], sum[n], 4 bytes per row, 1/m of the matrix traffic) so every stage is an ordinary
// component with a port; nothing reads those outputs.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace sm {

namespace cs = naga::cospan;
using cs::DType;

constexpr std::size_t kWg = 256;  // the work-group size of every kernel

inline std::uint32_t u32(std::size_t v) { return static_cast<std::uint32_t>(v); }

// =============================================================================
// Unfused
// =============================================================================
struct RowMaxBody {
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> mx;
    };
    std::uint32_t n, m;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const std::uint32_t l  = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t wg = static_cast<std::uint32_t>(it.get_local_range(0));
        const std::uint32_t ng = static_cast<std::uint32_t>(it.get_group_range(0));
        for (std::uint32_t i = static_cast<std::uint32_t>(it.get_group(0)); i < n; i += ng) {
            float acc = std::numeric_limits<float>::lowest();
            for (std::uint32_t j = l; j < m; j += wg) acc = sycl::fmax(acc, in.x[sycl::id<2>{i, j}]);
            const float mx = sycl::reduce_over_group(it.get_group(), acc, sycl::maximum<float>());
            if (l == 0) out.mx[sycl::id<1>{i}] = mx;
        }
    }
};

class RowMax : public cs::Component<RowMaxBody> {
    cs::Input<DType::f32, 2>  x_;
    cs::Output<DType::f32, 1> mx_;
    int                       n_, m_;
    bool                      tied_x_;  // x is rank-only: a tie_inputs member (the tie alias owns its shape)

public:
    RowMax(int n, int m, bool tied_x = false) : n_{n}, m_{m}, tied_x_{tied_x} {}
    void setup() override {
        if (tied_x_)
            declare_input(x_, "x", cs::unspecified_shape<2>());
        else
            declare_input(x_, "x", n_, m_);
        declare_output(mx_, "mx", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{static_cast<std::size_t>(n_) * kWg}, sycl::range<1>{kWg}, RowMaxBody::Inputs{x_},
                                    RowMaxBody::Outputs{mx_}, RowMaxBody{u32(n_), u32(m_)});
    }
};

struct ExpSumBody {
    struct Inputs {
        cs::Input<DType::f32, 2> x;
        cs::Input<DType::f32, 1> mx;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> e;
        cs::Output<DType::f32, 1> sum;
    };
    std::uint32_t n, m;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const std::uint32_t l  = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t wg = static_cast<std::uint32_t>(it.get_local_range(0));
        const std::uint32_t ng = static_cast<std::uint32_t>(it.get_group_range(0));
        for (std::uint32_t i = static_cast<std::uint32_t>(it.get_group(0)); i < n; i += ng) {
            const float mx  = in.mx[sycl::id<1>{i}];
            float       acc = 0.0f;
            for (std::uint32_t j = l; j < m; j += wg) {
                const float v = sycl::exp(in.x[sycl::id<2>{i, j}] - mx);
                out.e[sycl::id<2>{i, j}] = v;
                acc += v;
            }
            const float s = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            if (l == 0) out.sum[sycl::id<1>{i}] = s;
        }
    }
};

class ExpSum : public cs::Component<ExpSumBody> {
    cs::Input<DType::f32, 2>  x_;
    cs::Input<DType::f32, 1>  mx_;
    cs::Output<DType::f32, 2> e_;
    cs::Output<DType::f32, 1> sum_;
    int                       n_, m_;
    bool                      tied_x_;

public:
    ExpSum(int n, int m, bool tied_x = false) : n_{n}, m_{m}, tied_x_{tied_x} {}
    void setup() override {
        if (tied_x_)
            declare_input(x_, "x", cs::unspecified_shape<2>());
        else
            declare_input(x_, "x", n_, m_);
        declare_input(mx_, "mx", n_);
        declare_output(e_, "e", n_, m_);
        declare_output(sum_, "sum", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{static_cast<std::size_t>(n_) * kWg}, sycl::range<1>{kWg},
                                    ExpSumBody::Inputs{x_, mx_}, ExpSumBody::Outputs{e_, sum_}, ExpSumBody{u32(n_), u32(m_)});
    }
};

struct NormalizeBody {
    struct Inputs {
        cs::Input<DType::f32, 2> e;
        cs::Input<DType::f32, 1> sum;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    std::uint32_t n, m;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const std::uint32_t l  = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t wg = static_cast<std::uint32_t>(it.get_local_range(0));
        const std::uint32_t ng = static_cast<std::uint32_t>(it.get_group_range(0));
        for (std::uint32_t i = static_cast<std::uint32_t>(it.get_group(0)); i < n; i += ng) {
            const float s = in.sum[sycl::id<1>{i}];
            for (std::uint32_t j = l; j < m; j += wg) out.y[sycl::id<2>{i, j}] = in.e[sycl::id<2>{i, j}] / s;
        }
    }
};

class Normalize : public cs::Component<NormalizeBody> {
    cs::Input<DType::f32, 2>  e_;
    cs::Input<DType::f32, 1>  sum_;
    cs::Output<DType::f32, 2> y_;
    int                       n_, m_;

public:
    Normalize(int n, int m) : n_{n}, m_{m} {}
    void setup() override {
        declare_input(e_, "e", n_, m_);
        declare_input(sum_, "sum", n_);
        declare_output(y_, "y", n_, m_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{static_cast<std::size_t>(n_) * kWg}, sycl::range<1>{kWg},
                                    NormalizeBody::Inputs{e_, sum_}, NormalizeBody::Outputs{y_}, NormalizeBody{u32(n_), u32(m_)});
    }
};

// =============================================================================
// Fused: the row and its two scalars live in local memory (the same block for every stage of a `shared` segment)
// =============================================================================
/// The local layout every fused stage allocates in the same order: m floats of row, then [max, sum].
struct LocalRow {
    float* row;
    float* sc;  // sc[0] = the row max, sc[1] = the row sum
};
inline LocalRow local_row(cs::GroupItem<1>& it, std::uint32_t m) {
    float* row = it.local<float>(m);
    float* sc  = it.local<float>(2);
    return LocalRow{row, sc};
}
/// Bytes of local memory a fused stage requests (the framework rounds up to 16).
inline std::size_t local_row_bytes(int m) { return (static_cast<std::size_t>(m) + 2) * sizeof(float); }

struct LoadMaxBody {
    struct Inputs {
        cs::Input<DType::f32, 2> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> mx;
    };
    std::uint32_t n, m;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const LocalRow      lr = local_row(it, m);
        const std::uint32_t l  = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t wg = static_cast<std::uint32_t>(it.get_local_range(0));
        const std::uint32_t i  = static_cast<std::uint32_t>(it.get_group(0));  // one row per group
        float               acc = std::numeric_limits<float>::lowest();
        for (std::uint32_t j = l; j < m; j += wg) {
            const float v = in.x[sycl::id<2>{i, j}];
            lr.row[j]     = v;
            acc           = sycl::fmax(acc, v);
        }
        const float mx = sycl::reduce_over_group(it.get_group(), acc, sycl::maximum<float>());
        if (l == 0) {
            lr.sc[0]               = mx;
            out.mx[sycl::id<1>{i}] = mx;
        }
    }
};

class LoadMax : public cs::Component<LoadMaxBody> {
    cs::Input<DType::f32, 2>  x_;
    cs::Output<DType::f32, 1> mx_;
    int                       n_, m_;

public:
    LoadMax(int n, int m) : n_{n}, m_{m} {}
    void setup() override {
        declare_input(x_, "x", n_, m_);
        declare_output(mx_, "mx", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = create_launch_config(sycl::range<1>{static_cast<std::size_t>(n_) * kWg}, sycl::range<1>{kWg}, LoadMaxBody::Inputs{x_},
                                      LoadMaxBody::Outputs{mx_}, LoadMaxBody{u32(n_), u32(m_)});
        d->request_local_bytes(local_row_bytes(m_));
        return d;
    }
};

struct ExpSumLocalBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 1> sum;
    };
    std::uint32_t n, m;

    void operator()(cs::GroupItem<1> it, Inputs, Outputs out) const {
        const LocalRow      lr = local_row(it, m);
        const std::uint32_t l  = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t wg = static_cast<std::uint32_t>(it.get_local_range(0));
        const std::uint32_t i  = static_cast<std::uint32_t>(it.get_group(0));
        sycl::group_barrier(it.get_group());  // LoadMax's max is in sc[0]
        const float mx  = lr.sc[0];
        float       acc = 0.0f;
        for (std::uint32_t j = l; j < m; j += wg) {
            const float v = sycl::exp(lr.row[j] - mx);
            lr.row[j]     = v;
            acc += v;
        }
        const float s = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        if (l == 0) {
            lr.sc[1]                = s;
            out.sum[sycl::id<1>{i}] = s;
        }
    }
};

class ExpSumLocal : public cs::Component<ExpSumLocalBody> {
    cs::Output<DType::f32, 1> sum_;
    int                       n_, m_;

public:
    ExpSumLocal(int n, int m) : n_{n}, m_{m} {}
    void setup() override { declare_output(sum_, "sum", n_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = create_launch_config(sycl::range<1>{static_cast<std::size_t>(n_) * kWg}, sycl::range<1>{kWg}, ExpSumLocalBody::Inputs{},
                                      ExpSumLocalBody::Outputs{sum_}, ExpSumLocalBody{u32(n_), u32(m_)});
        d->request_local_bytes(local_row_bytes(m_));
        return d;
    }
};

struct NormalizeWriteBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    std::uint32_t n, m;

    void operator()(cs::GroupItem<1> it, Inputs, Outputs out) const {
        const LocalRow      lr = local_row(it, m);
        const std::uint32_t l  = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t wg = static_cast<std::uint32_t>(it.get_local_range(0));
        const std::uint32_t i  = static_cast<std::uint32_t>(it.get_group(0));
        sycl::group_barrier(it.get_group());  // ExpSumLocal's sum is in sc[1]
        const float s = lr.sc[1];
        for (std::uint32_t j = l; j < m; j += wg) out.y[sycl::id<2>{i, j}] = lr.row[j] / s;
    }
};

class NormalizeWrite : public cs::Component<NormalizeWriteBody> {
    cs::Output<DType::f32, 2> y_;
    int                       n_, m_;

public:
    NormalizeWrite(int n, int m) : n_{n}, m_{m} {}
    void setup() override { declare_output(y_, "y", n_, m_); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        auto d = create_launch_config(sycl::range<1>{static_cast<std::size_t>(n_) * kWg}, sycl::range<1>{kWg},
                                      NormalizeWriteBody::Inputs{}, NormalizeWriteBody::Outputs{y_}, NormalizeWriteBody{u32(n_), u32(m_)});
        d->request_local_bytes(local_row_bytes(m_));
        return d;
    }
};

}  // namespace sm
