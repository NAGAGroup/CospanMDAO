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
//   - The launch range is the full work.
//
// The components:
//   Rank2Update   B = A + u1 v1^T + u2 v2^T, in strips of R rows
//   GemvTPartial  P[c][j] = sum_{k<R} B[c R + k][j] * y[c R + k]   (same item mapping)
//   ColReduce<Z>  x[j] = (Z ? z[j] : 0) + scale * sum_c P[c][j]
//   Gemv          out[i] = scale * (B[i] . x), a group per row (grid-stride over rows)
//   BicgkOnePass, RowReduce   q and s from one pass over B, and the row reducer of its q
//
// The "crafted" components (bottom of this file) are written to be fused: a work-group owns a TILE of kRt rows by 256
// columns of B and keeps it in local memory between the stages of one segment (local_mem::shared), so B is read from
// global memory once where the plain components read it twice:
//   Rank2UpdateLocal      B tile = A + u1 v1^T + u2 v2^T; writes it to global B AND leaves it in local memory
//   GemvTPartialLocal     P[c][j] = sum_k tile[k][j] y[c kRt + k], reading the tile from local memory
//   BicgkTileLocal        loads the B1 tile into local memory; the q and s partials of BicgkOnePass from it
//   Rank2UpdateFromLocal  B2 tile = B1 tile (from local memory) + u1b v1b^T + u2b v2b^T, written to global B2
// These stages process EXACTLY ONE tile per work-group (the launch has one group per tile and no grid-stride loop): the
// stages of a fused group run one after the other inside a work-group, so a stage that looped over several tiles would
// leave only the last one in local memory for the next stage.

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

public:
    Rank2Update(int n, int R) : n_{n}, R_{R} {}
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
        const std::size_t range = work;
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
    bool                      tied_b_;  // B is rank-only: a tie_inputs member (the tie alias owns its shape, D139)

public:
    GemvTPartial(int n, int R, bool tied_b = false) : n_{n}, R_{R}, tied_b_{tied_b} {}
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
        const std::size_t range = work;
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

public:
    ColReduce(int n, int R, float scale) : n_{n}, R_{R}, scale_{scale} {}
    void setup() override {
        this->declare_input(P_, "P", n_ / R_, n_);
        if constexpr (Z) this->declare_input(z_, "z", n_);
        this->declare_output(x_, "x", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t range = static_cast<std::size_t>(n_);
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
    bool                      tied_b_;  // B is rank-only: a tie_inputs member (the tie alias owns its shape, D139)

public:
    Gemv(int n, float scale, bool tied_b = false) : n_{n}, scale_{scale}, tied_b_{tied_b} {}
    void setup() override {
        if (tied_b_)
            declare_input(B_, "B", cs::unspecified_shape<2>());
        else
            declare_input(B_, "B", n_, n_);
        declare_input(x_, "x", n_);
        declare_output(y_, "y", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t range = static_cast<std::size_t>(n_) * kWg;  // a group per row
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
// The crafted components: one tile of kRt x 256 floats of B per work-group, kept in local memory
// =============================================================================
// Work-group g of a launch over (n / kRt) * (n / 256) groups owns strip c = g / blocks (rows c kRt .. c kRt + kRt - 1) and
// column block blk = g % blocks (columns blk 256 .. blk 256 + 255); work-item l owns column j = blk 256 + l. The tile is
// tile[k * 256 + l] = B[c kRt + k][j]. 32 rows * 256 columns * 4 B = 32 KB, inside the 48 KB of local memory.
constexpr std::uint32_t kRt        = 32;
constexpr std::size_t   kTileBytes = static_cast<std::size_t>(kRt) * kWg * sizeof(float);

/// Rank2UpdateLocal: B = A + u1 v1^T + u2 v2^T for the tile; B goes to global memory (B1 is an output) and stays in the tile.
struct Rank2UpdateLocalBody {
    struct Inputs {
        cs::Input<DType::f32, 2> A;
        cs::Input<DType::f32, 1> u1, v1, u2, v2;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> B;
    };
    std::uint32_t blocks;  // n / 256 column blocks

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t c = g / blocks, blk = g - c * blocks;
        const std::uint32_t j = blk * static_cast<std::uint32_t>(kWg) + l;
        const float         v1j = in.v1[sycl::id<1>{j}], v2j = in.v2[sycl::id<1>{j}];
        for (std::uint32_t k = 0; k < kRt; ++k) {
            const std::uint32_t i   = c * kRt + k;
            const float         val = in.A[sycl::id<2>{i, j}] + in.u1[sycl::id<1>{i}] * v1j + in.u2[sycl::id<1>{i}] * v2j;
            out.B[sycl::id<2>{i, j}] = val;
            tile[k * kWg + l]        = val;
        }
    }
};

class Rank2UpdateLocal : public cs::Component<Rank2UpdateLocalBody> {
    cs::Input<DType::f32, 2>  A_;
    cs::Input<DType::f32, 1>  u1_, v1_, u2_, v2_;
    cs::Output<DType::f32, 2> B_;
    int                       n_;

public:
    explicit Rank2UpdateLocal(int n) : n_{n} {}
    void setup() override {
        declare_input(A_, "A", n_, n_);
        declare_input(u1_, "u1", n_);
        declare_input(v1_, "v1", n_);
        declare_input(u2_, "u2", n_);
        declare_input(v2_, "v2", n_);
        declare_output(B_, "B", n_, n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / static_cast<int>(kRt)) * blocks;
        auto              d      = create_launch_config(sycl::range<1>{groups * kWg}, sycl::range<1>{kWg},
                                                        Rank2UpdateLocalBody::Inputs{A_, u1_, v1_, u2_, v2_},
                                                        Rank2UpdateLocalBody::Outputs{B_}, Rank2UpdateLocalBody{u32(blocks)});
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

/// GemvTPartialLocal: P[c][j] = sum_k tile[k][j] * y[c kRt + k]. No global B input: the tile comes from Rank2UpdateLocal.
struct GemvTPartialLocalBody {
    struct Inputs {
        cs::Input<DType::f32, 1> y;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> P;
    };
    std::uint32_t blocks;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);  // the same block as Rank2UpdateLocal's
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t c = g / blocks, blk = g - c * blocks;
        const std::uint32_t j = blk * static_cast<std::uint32_t>(kWg) + l;
        sycl::group_barrier(it.get_group());  // the tile is complete
        float acc = 0.0f;
        for (std::uint32_t k = 0; k < kRt; ++k) acc += tile[k * kWg + l] * in.y[sycl::id<1>{c * kRt + k}];
        out.P[sycl::id<2>{c, j}] = acc;
    }
};

class GemvTPartialLocal : public cs::Component<GemvTPartialLocalBody> {
    cs::Input<DType::f32, 1>  y_;
    cs::Output<DType::f32, 2> P_;
    int                       n_;

public:
    explicit GemvTPartialLocal(int n) : n_{n} {}
    void setup() override {
        declare_input(y_, "y", n_);
        declare_output(P_, "P", n_ / static_cast<int>(kRt), n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / static_cast<int>(kRt)) * blocks;
        auto              d      = create_launch_config(sycl::range<1>{groups * kWg}, sycl::range<1>{kWg},
                                                        GemvTPartialLocalBody::Inputs{y_}, GemvTPartialLocalBody::Outputs{P_},
                                                        GemvTPartialLocalBody{u32(blocks)});
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

/// BicgkTileLocal: loads the B tile into local memory and produces BicgkOnePass's partials from it:
///   Ps[c][j]   = sum_k r[i] B[i][j]                  Pq[i][blk] = reduce_over_group(B[i][j] p[j])   (i = c kRt + k)
struct BicgkTileLocalBody {
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> p, r;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> Ps, Pq;
    };
    std::uint32_t blocks;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t c = g / blocks, blk = g - c * blocks;
        const std::uint32_t j  = blk * static_cast<std::uint32_t>(kWg) + l;
        const float         pj = in.p[sycl::id<1>{j}];
        float               ps = 0.0f;
        for (std::uint32_t k = 0; k < kRt; ++k) {
            const std::uint32_t i = c * kRt + k;
            const float         b = in.B[sycl::id<2>{i, j}];  // the ONE read of this element of B1
            tile[k * kWg + l]     = b;
            ps += in.r[sycl::id<1>{i}] * b;
            const float sum = sycl::reduce_over_group(it.get_group(), b * pj, sycl::plus<float>());
            if (l == 0) out.Pq[sycl::id<2>{i, blk}] = sum;
        }
        out.Ps[sycl::id<2>{c, j}] = ps;
    }
};

class BicgkTileLocal : public cs::Component<BicgkTileLocalBody> {
    cs::Input<DType::f32, 2>  B_;
    cs::Input<DType::f32, 1>  p_, r_;
    cs::Output<DType::f32, 2> Ps_, Pq_;
    int                       n_;

public:
    explicit BicgkTileLocal(int n) : n_{n} {}
    void setup() override {
        declare_input(B_, "B", n_, n_);
        declare_input(p_, "p", n_);
        declare_input(r_, "r", n_);
        declare_output(Ps_, "Ps", n_ / static_cast<int>(kRt), n_);
        declare_output(Pq_, "Pq", n_, n_ / static_cast<int>(kWg));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / static_cast<int>(kRt)) * blocks;
        auto              d      = create_launch_config(sycl::range<1>{groups * kWg}, sycl::range<1>{kWg},
                                                        BicgkTileLocalBody::Inputs{B_, p_, r_},
                                                        BicgkTileLocalBody::Outputs{Ps_, Pq_}, BicgkTileLocalBody{u32(blocks)});
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

/// Rank2UpdateFromLocal: G2's B2 = B1 + u1b v1b^T + u2b v2b^T, with B1 taken from the tile BicgkTileLocal loaded (no global
/// read of B1 here, and no dependency on q or s, which is why it can share BiCGK's read of B1).
struct Rank2UpdateFromLocalBody {
    struct Inputs {
        cs::Input<DType::f32, 1> u1, v1, u2, v2;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> B;
    };
    std::uint32_t blocks;

    void operator()(cs::GroupItem<1> it, Inputs in, Outputs out) const {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t c = g / blocks, blk = g - c * blocks;
        const std::uint32_t j = blk * static_cast<std::uint32_t>(kWg) + l;
        sycl::group_barrier(it.get_group());  // the tile is complete
        const float v1j = in.v1[sycl::id<1>{j}], v2j = in.v2[sycl::id<1>{j}];
        for (std::uint32_t k = 0; k < kRt; ++k) {
            const std::uint32_t i = c * kRt + k;
            out.B[sycl::id<2>{i, j}] = tile[k * kWg + l] + in.u1[sycl::id<1>{i}] * v1j + in.u2[sycl::id<1>{i}] * v2j;
        }
    }
};

class Rank2UpdateFromLocal : public cs::Component<Rank2UpdateFromLocalBody> {
    cs::Input<DType::f32, 1>  u1_, v1_, u2_, v2_;
    cs::Output<DType::f32, 2> B_;
    int                       n_;

public:
    explicit Rank2UpdateFromLocal(int n) : n_{n} {}
    void setup() override {
        declare_input(u1_, "u1", n_);
        declare_input(v1_, "v1", n_);
        declare_input(u2_, "u2", n_);
        declare_input(v2_, "v2", n_);
        declare_output(B_, "B", n_, n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / static_cast<int>(kRt)) * blocks;
        auto              d      = create_launch_config(sycl::range<1>{groups * kWg}, sycl::range<1>{kWg},
                                                        Rank2UpdateFromLocalBody::Inputs{u1_, v1_, u2_, v2_},
                                                        Rank2UpdateFromLocalBody::Outputs{B_}, Rank2UpdateFromLocalBody{u32(blocks)});
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

}  // namespace gc
