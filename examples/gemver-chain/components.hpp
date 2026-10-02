#pragma once

// =============================================================================
// examples/gemver-chain/components.hpp — the kernels of the GEMVER -> BiCGK -> GEMVER chain
// =============================================================================
//
// Rules every component follows:
//   - Every kernel is rank-1 with work-group size 256 (kWg). All bodies are GROUP
//     bodies (they take a cs::GroupItem<1>), because a plain body cannot ask for a
//     work-group size and an unfused launch would then use the framework default (128).
//   - Matrices are Input/Output<f32, 2> and are indexed ONLY by sycl::id<2>{i, j}
//     (the flat operator does a div/mod on every access). Vectors use sycl::id<1>
//     for the same reason.
//   - Every element-wise body is grid-stride: `for (t = id; t < total; t += stride)`,
//     `stride` being a body field equal to the launch range.
//   - The launch range is the full work by default. Each component takes an optional
//     `resident_items` cap; when it is non-zero the range is exactly that value
//     (G * 256), which is how the barrier variant keeps every group co-resident.
//
// The components:
//   Rank2Update   B = A + u1 v1^T + u2 v2^T, in strips of R rows
//   GemvTPartial  P[c][j] = sum_{k<R} B[c R + k][j] * y[c R + k]   (same item mapping)
//   ColReduce<Z>  x[j] = (Z ? z[j] : 0) + scale * sum_c P[c][j]
//   Gemv          out[i] = scale * (B[i] . x), a group per row (grid-stride over rows)
//   GridBarrier   a software grid barrier over atomics (needs every group co-resident)

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace gc {

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;

constexpr std::size_t kWg = 256;  // the work-group size of every kernel

/// The launch range: the full work, or exactly `resident` items when a cap is set.
inline std::size_t launch_items(std::size_t full, std::size_t resident) { return resident != 0 ? resident : full; }

inline std::uint32_t u32(std::size_t v) { return static_cast<std::uint32_t>(v); }

// =============================================================================
// Rank2Update: t = (c = t / n, j = t % n) over (n / R) * n items
// =============================================================================
struct Rank2UpdateBody {
    struct Inputs {
        cs::Input<DType::f32, 2> A;
        cs::Input<DType::f32, 1> u1, v1, u2, v2;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> B;
    };
    std::uint32_t n, R, total, stride;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        for (std::uint32_t t = static_cast<std::uint32_t>(it.get_global_id(0)); t < total; t += stride) {
            const std::uint32_t c = t / n, j = t - c * n;
            const float         v1j = in.v1[sycl::id<1>{j}], v2j = in.v2[sycl::id<1>{j}];
            for (std::uint32_t k = 0; k < R; ++k) {
                const std::uint32_t i = c * R + k;
                out.B[sycl::id<2>{i, j}] =
                    in.A[sycl::id<2>{i, j}] + in.u1[sycl::id<1>{i}] * v1j + in.u2[sycl::id<1>{i}] * v2j;
            }
        }
    }
};

class Rank2Update : public cs::Component<Rank2UpdateBody> {
    cs::Input<DType::f32, 2>  A_;
    cs::Input<DType::f32, 1>  u1_, v1_, u2_, v2_;
    cs::Output<DType::f32, 2> B_;
    int                       n_, R_;
    std::size_t               resident_;

public:
    Rank2Update(int n, int R, std::size_t resident_items = 0) : n_{n}, R_{R}, resident_{resident_items} {}
    void setup() override {
        declare_input(A_, "A", n_, n_);
        declare_input(u1_, "u1", n_);
        declare_input(v1_, "v1", n_);
        declare_input(u2_, "u2", n_);
        declare_input(v2_, "v2", n_);
        declare_output(B_, "B", n_, n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t work  = static_cast<std::size_t>(n_ / R_) * n_;
        const std::size_t range = launch_items(work, resident_);
        return create_launch_config(sycl::range<1>{range}, sycl::range<1>{kWg},
                                    Rank2UpdateBody::Inputs{A_, u1_, v1_, u2_, v2_}, Rank2UpdateBody::Outputs{B_},
                                    Rank2UpdateBody{u32(n_), u32(R_), u32(work), u32(range)});
    }
};

// =============================================================================
// GemvTPartial: the same item mapping as Rank2Update; one partial column sum per strip
// =============================================================================
struct GemvTPartialBody {
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> y;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> P;
    };
    std::uint32_t n, R, total, stride;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        for (std::uint32_t t = static_cast<std::uint32_t>(it.get_global_id(0)); t < total; t += stride) {
            const std::uint32_t c = t / n, j = t - c * n;
            float               acc = 0.0f;
            for (std::uint32_t k = 0; k < R; ++k) {
                const std::uint32_t i = c * R + k;
                acc += in.B[sycl::id<2>{i, j}] * in.y[sycl::id<1>{i}];
            }
            out.P[sycl::id<2>{c, j}] = acc;
        }
    }
};

class GemvTPartial : public cs::Component<GemvTPartialBody> {
    cs::Input<DType::f32, 2>  B_;
    cs::Input<DType::f32, 1>  y_;
    cs::Output<DType::f32, 2> P_;
    int                       n_, R_;
    std::size_t               resident_;
    bool                      tied_b_;  // B is rank-only: a tie_inputs member (the tie alias owns its shape, D139)

public:
    GemvTPartial(int n, int R, std::size_t resident_items = 0, bool tied_b = false)
        : n_{n}, R_{R}, resident_{resident_items}, tied_b_{tied_b} {}
    void setup() override {
        if (tied_b_)
            declare_input(B_, "B", cs::unspecified_shape<2>());
        else
            declare_input(B_, "B", n_, n_);
        declare_input(y_, "y", n_);
        declare_output(P_, "P", n_ / R_, n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t work  = static_cast<std::size_t>(n_ / R_) * n_;
        const std::size_t range = launch_items(work, resident_);
        return create_launch_config(sycl::range<1>{range}, sycl::range<1>{kWg}, GemvTPartialBody::Inputs{B_, y_},
                                    GemvTPartialBody::Outputs{P_},
                                    GemvTPartialBody{u32(n_), u32(R_), u32(work), u32(range)});
    }
};

// =============================================================================
// ColReduce<Z>: x[j] = (Z ? z[j] : 0) + scale * sum_c P[c][j]
// =============================================================================
// Z = true is GEMVER's x (scale beta, with z); Z = false is BiCGK's s (scale 1, no z
// input at all, not a dummy one).
template <bool Z>
struct ColReduceBody {
    struct InputsZ {
        cs::Input<DType::f32, 2> P;
        cs::Input<DType::f32, 1> z;
    };
    struct InputsNoZ {
        cs::Input<DType::f32, 2> P;
    };
    using Inputs = std::conditional_t<Z, InputsZ, InputsNoZ>;
    struct Outputs {
        cs::Output<DType::f32, 1> x;
    };
    std::uint32_t n, strips, stride;
    float         scale;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        for (std::uint32_t j = static_cast<std::uint32_t>(it.get_global_id(0)); j < n; j += stride) {
            float acc = 0.0f;
            for (std::uint32_t c = 0; c < strips; ++c) acc += in.P[sycl::id<2>{c, j}];
            if constexpr (Z)
                out.x[sycl::id<1>{j}] = in.z[sycl::id<1>{j}] + scale * acc;
            else
                out.x[sycl::id<1>{j}] = scale * acc;
        }
    }
};

template <bool Z>
class ColReduce : public cs::Component<ColReduceBody<Z>> {
    using Body = ColReduceBody<Z>;
    cs::Input<DType::f32, 2>  P_;
    cs::Input<DType::f32, 1>  z_;  // only declared when Z
    cs::Output<DType::f32, 1> x_;
    int                       n_, R_;
    float                     scale_;
    std::size_t               resident_;

public:
    ColReduce(int n, int R, float scale, std::size_t resident_items = 0)
        : n_{n}, R_{R}, scale_{scale}, resident_{resident_items} {}
    void setup() override {
        this->declare_input(P_, "P", n_ / R_, n_);
        if constexpr (Z) this->declare_input(z_, "z", n_);
        this->declare_output(x_, "x", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t range = launch_items(static_cast<std::size_t>(n_), resident_);
        typename Body::Inputs in{};
        in.P = P_;
        if constexpr (Z) in.z = z_;
        return this->create_launch_config(sycl::range<1>{range}, sycl::range<1>{kWg}, in, typename Body::Outputs{x_},
                                          Body{u32(n_), u32(n_ / R_), u32(range), scale_});
    }
};

// =============================================================================
// Gemv: out[i] = scale * (B[i][:] . x). A GROUP body: group g takes rows g, g + ngroups, ...
// =============================================================================
struct GemvBody {
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    std::uint32_t n;
    float         scale;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const std::uint32_t l = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t wg = static_cast<std::uint32_t>(it.get_local_range(0));
        const std::uint32_t g = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t ng = static_cast<std::uint32_t>(it.get_group_range(0));  // this stage's groups
        for (std::uint32_t i = g; i < n; i += ng) {
            float acc = 0.0f;
            for (std::uint32_t j = l; j < n; j += wg) acc += in.B[sycl::id<2>{i, j}] * in.x[sycl::id<1>{j}];
            const float sum = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            if (l == 0) out.y[sycl::id<1>{i}] = scale * sum;
        }
    }
};

class Gemv : public cs::Component<GemvBody> {
    cs::Input<DType::f32, 2>  B_;
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
    int                       n_;
    float                     scale_;
    std::size_t               resident_;
    bool                      tied_b_;  // B is rank-only: a tie_inputs member (the tie alias owns its shape, D139)

public:
    Gemv(int n, float scale, std::size_t resident_items = 0, bool tied_b = false)
        : n_{n}, scale_{scale}, resident_{resident_items}, tied_b_{tied_b} {}
    void setup() override {
        if (tied_b_)
            declare_input(B_, "B", cs::unspecified_shape<2>());
        else
            declare_input(B_, "B", n_, n_);
        declare_input(x_, "x", n_);
        declare_output(y_, "y", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t range = launch_items(static_cast<std::size_t>(n_) * kWg, resident_);  // a group per row
        return create_launch_config(sycl::range<1>{range}, sycl::range<1>{kWg}, GemvBody::Inputs{B_, x_},
                                    GemvBody::Outputs{y_}, GemvBody{u32(n_), scale_});
    }
};

// =============================================================================
// BicgkOnePass: q = B p and s = B^T r in ONE pass over B
// =============================================================================
// A group body over (strip c, 256-column block blk): work-item l owns column j = blk*256 + l.
//   Ps[c][j]   = sum_{k<R} r[i] * B[i][j]                     (i = c*R + k)
//   Pq[i][blk] = reduce_over_group(B[i][j] * p[j])            (per row i of the strip)
// RowReduce then sums Pq over the blocks (q), and the scale-1 ColReduce sums Ps (s).
struct BicgkOnePassBody {
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> p, r;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> Ps, Pq;
    };
    std::uint32_t n, R, blocks, total_groups;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        const std::uint32_t l  = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t ng = static_cast<std::uint32_t>(it.get_group_range(0));
        for (std::uint32_t g = static_cast<std::uint32_t>(it.get_group(0)); g < total_groups; g += ng) {
            const std::uint32_t c = g / blocks, blk = g - c * blocks;
            const std::uint32_t j = blk * static_cast<std::uint32_t>(it.get_local_range(0)) + l;
            const float         pj = in.p[sycl::id<1>{j}];
            float               ps = 0.0f;
            for (std::uint32_t k = 0; k < R; ++k) {
                const std::uint32_t i = c * R + k;
                const float         b = in.B[sycl::id<2>{i, j}];
                ps += in.r[sycl::id<1>{i}] * b;
                const float sum = sycl::reduce_over_group(it.get_group(), b * pj, sycl::plus<float>());
                if (l == 0) out.Pq[sycl::id<2>{i, blk}] = sum;
            }
            out.Ps[sycl::id<2>{c, j}] = ps;
        }
    }
};

class BicgkOnePass : public cs::Component<BicgkOnePassBody> {
    cs::Input<DType::f32, 2>  B_;
    cs::Input<DType::f32, 1>  p_, r_;
    cs::Output<DType::f32, 2> Ps_, Pq_;
    int                       n_, R_;

public:
    BicgkOnePass(int n, int R) : n_{n}, R_{R} {}
    void setup() override {
        declare_input(B_, "B", n_, n_);
        declare_input(p_, "p", n_);
        declare_input(r_, "r", n_);
        declare_output(Ps_, "Ps", n_ / R_, n_);
        declare_output(Pq_, "Pq", n_, n_ / static_cast<int>(kWg));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / R_) * blocks;
        return create_launch_config(sycl::range<1>{groups * kWg}, sycl::range<1>{kWg},
                                    BicgkOnePassBody::Inputs{B_, p_, r_}, BicgkOnePassBody::Outputs{Ps_, Pq_},
                                    BicgkOnePassBody{u32(n_), u32(R_), u32(blocks), u32(groups)});
    }
};

// RowReduce: q[i] = sum_blk Pq[i][blk]
struct RowReduceBody {
    struct Inputs {
        cs::Input<DType::f32, 2> Pq;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> q;
    };
    std::uint32_t n, blocks, stride;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        for (std::uint32_t i = static_cast<std::uint32_t>(it.get_global_id(0)); i < n; i += stride) {
            float acc = 0.0f;
            for (std::uint32_t b = 0; b < blocks; ++b) acc += in.Pq[sycl::id<2>{i, b}];
            out.q[sycl::id<1>{i}] = acc;
        }
    }
};

class RowReduce : public cs::Component<RowReduceBody> {
    cs::Input<DType::f32, 2>  Pq_;
    cs::Output<DType::f32, 1> q_;
    int                       n_;

public:
    explicit RowReduce(int n) : n_{n} {}
    void setup() override {
        declare_input(Pq_, "Pq", n_, n_ / static_cast<int>(kWg));
        declare_output(q_, "q", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t range = static_cast<std::size_t>(n_);
        return create_launch_config(sycl::range<1>{range}, sycl::range<1>{kWg}, RowReduceBody::Inputs{Pq_},
                                    RowReduceBody::Outputs{q_},
                                    RowReduceBody{u32(n_), u32(n_ / static_cast<int>(kWg)), u32(range)});
    }
};

// =============================================================================
// GridBarrier: a software grid barrier. Only valid when EVERY group of the launch is
// co-resident, which is why the whole chain runs on a capped, resident number of groups.
// =============================================================================
// ZeroCounters (below), submitted as its OWN launch before the fused wave (never a stage inside it),
// zeroes `count` and `timeout` at the start of every forward (outputs are never initialized, D138).
// `count` then grows by G at the one crossing of the forward, which waits for count >= G.
// A group that waits `spin_limit` polls without seeing the target gives up and sets `timeout`.
struct GridBarrierBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_u32, 1> count, timeout;
    };
    std::uint32_t target, spin_limit;

    void operator()(cs::GroupItem<1> it, Inputs, Outputs out) const {
        // Publish this work-item's earlier global writes at device scope, then gather the group.
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        sycl::group_barrier(it.get_group());
        if (it.get_local_id(0) == 0) {
            auto count = out.count[sycl::id<1>{std::size_t{0}}];
            count.fetch_add(1u);
            std::uint32_t spins = 0;
            while (count.load() < target && spins < spin_limit) ++spins;
            if (count.load() < target) out.timeout[sycl::id<1>{std::size_t{0}}].store(1u);
        }
        sycl::group_barrier(it.get_group());
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
    }
};

class GridBarrier : public cs::Component<GridBarrierBody> {
    cs::Output<DType::atomic_u32, 1> count_, timeout_;
    std::size_t                      groups_;
    std::uint32_t                    spin_limit_;

public:
    GridBarrier(std::size_t groups, std::uint32_t spin_limit) : groups_{groups}, spin_limit_{spin_limit} {}
    void setup() override {
        declare_output(count_, "count", cs::unspecified_shape<1>());  // BarrierTop's tie_outputs owns the shape (D139)
        declare_output(timeout_, "timeout", cs::unspecified_shape<1>());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        // The counters were zeroed at the start of this forward: one crossing waits for count >= G.
        return create_launch_config(sycl::range<1>{groups_ * kWg}, sycl::range<1>{kWg}, GridBarrierBody::Inputs{},
                                    GridBarrierBody::Outputs{count_, timeout_},
                                    GridBarrierBody{static_cast<std::uint32_t>(groups_), spin_limit_});
    }
};

// =============================================================================
// ZeroCounters: the zeroing producer of ALL the barriers' counters, in ONE launch of one item
// =============================================================================
// Outputs are never initialized (D138), so the barriers' `count` and `timeout` come from a
// producer that computes zero. It is its own component, submitted before the fused wave (never
// a stage inside it); output `count<i>` / `timeout<i>` is tied to barrier i's by tie_outputs.
struct ZeroCountersBody {
    struct Inputs {};
    struct Outputs {
        cs::Output<DType::atomic_u32, 1> count1, count2, count3, count4, count5, count6, count7, timeout1, timeout2, timeout3, timeout4, timeout5, timeout6, timeout7;
    };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const {
        out.count1[i].store(0u);
        out.count2[i].store(0u);
        out.count3[i].store(0u);
        out.count4[i].store(0u);
        out.count5[i].store(0u);
        out.count6[i].store(0u);
        out.count7[i].store(0u);
        out.timeout1[i].store(0u);
        out.timeout2[i].store(0u);
        out.timeout3[i].store(0u);
        out.timeout4[i].store(0u);
        out.timeout5[i].store(0u);
        out.timeout6[i].store(0u);
        out.timeout7[i].store(0u);
    }
};

class ZeroCounters : public cs::Component<ZeroCountersBody> {
    cs::Output<DType::atomic_u32, 1> count1_, count2_, count3_, count4_, count5_, count6_, count7_, timeout1_, timeout2_, timeout3_, timeout4_, timeout5_, timeout6_, timeout7_;

public:
    void setup() override {
        declare_output(count1_, "count1", cs::unspecified_shape<1>());
        declare_output(count2_, "count2", cs::unspecified_shape<1>());
        declare_output(count3_, "count3", cs::unspecified_shape<1>());
        declare_output(count4_, "count4", cs::unspecified_shape<1>());
        declare_output(count5_, "count5", cs::unspecified_shape<1>());
        declare_output(count6_, "count6", cs::unspecified_shape<1>());
        declare_output(count7_, "count7", cs::unspecified_shape<1>());
        declare_output(timeout1_, "timeout1", cs::unspecified_shape<1>());
        declare_output(timeout2_, "timeout2", cs::unspecified_shape<1>());
        declare_output(timeout3_, "timeout3", cs::unspecified_shape<1>());
        declare_output(timeout4_, "timeout4", cs::unspecified_shape<1>());
        declare_output(timeout5_, "timeout5", cs::unspecified_shape<1>());
        declare_output(timeout6_, "timeout6", cs::unspecified_shape<1>());
        declare_output(timeout7_, "timeout7", cs::unspecified_shape<1>());
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(sycl::range<1>{1}, {}, {count1_, count2_, count3_, count4_, count5_, count6_, count7_, timeout1_, timeout2_, timeout3_, timeout4_, timeout5_, timeout6_, timeout7_}, {});
    }
};

}  // namespace gc
