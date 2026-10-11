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

// Runtime values (the matrix order n, the strip height R, the scale) are free inputs of extent 1 (u32 / f32) that each component
// declares and initializes from its constructor's value; a stride is read from the item (groups x group size).
inline std::uint32_t first(const cs::Input<DType::u32, 1>& v) { return v[std::size_t{0}]; }
inline float         first(const cs::Input<DType::f32, 1>& v) { return v[std::size_t{0}]; }

// =============================================================================
// Rank2Update: t = (c = t / n, j = t % n) over (n / R) * n items
// =============================================================================

class Rank2Update : public cs::Component<Rank2Update> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> A;
        cs::Input<DType::f32, 1> u1, v1, u2, v2;
        cs::Input<DType::u32, 1> n, R;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> B;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = first(in.n), R = first(in.R), total = (n / R) * n;
        const std::uint32_t stride = static_cast<std::uint32_t>(it.get_group_range(0) * it.get_local_range(0));
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

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_, R_;

public:
    Rank2Update(int n, int R) : n_{n}, R_{R} {}
    void setup() override {
        declare_input(inputs.n, "n", 1);
        declare_input(inputs.R, "R", 1);
        init_input("n", cs::init::fill(u32(n_)));
        init_input("R", cs::init::fill(u32(R_)));
        declare_input(inputs.A, "A", n_, n_);
        declare_input(inputs.u1, "u1", n_);
        declare_input(inputs.v1, "v1", n_);
        declare_input(inputs.u2, "u2", n_);
        declare_input(inputs.v2, "v2", n_);
        declare_output(outputs.B, "B", n_, n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t work  = static_cast<std::size_t>(n_ / R_) * n_;
        const std::size_t range = work;
        return create_launch_config(cs::range_desc(sycl::range<1>{range}), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// =============================================================================
// GemvTPartial: the same item mapping as Rank2Update; one partial column sum per strip
// =============================================================================

class GemvTPartial : public cs::Component<GemvTPartial> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> y;
        cs::Input<DType::u32, 1> n, R;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> P;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = first(in.n), R = first(in.R), total = (n / R) * n;
        const std::uint32_t stride = static_cast<std::uint32_t>(it.get_group_range(0) * it.get_local_range(0));
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

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_, R_;
    bool                      tied_b_;  // B is rank-only: a tie_inputs member (the tie alias owns its shape, D139)

public:
    GemvTPartial(int n, int R, bool tied_b = false) : n_{n}, R_{R}, tied_b_{tied_b} {}
    void setup() override {
        declare_input(inputs.n, "n", 1);
        declare_input(inputs.R, "R", 1);
        init_input("n", cs::init::fill(u32(n_)));
        init_input("R", cs::init::fill(u32(R_)));
        if (tied_b_)
            declare_input(inputs.B, "B", cs::unspecified_shape<2>());
        else
            declare_input(inputs.B, "B", n_, n_);
        declare_input(inputs.y, "y", n_);
        declare_output(outputs.P, "P", n_ / R_, n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t work  = static_cast<std::size_t>(n_ / R_) * n_;
        const std::size_t range = work;
        return create_launch_config(cs::range_desc(sycl::range<1>{range}), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// =============================================================================
// ColReduce<Z>: x[j] = (Z ? z[j] : 0) + scale * sum_c P[c][j]
// =============================================================================
// Z = true is GEMVER's x (scale beta, with z); Z = false is BiCGK's s (scale 1, no z
// input at all, not a dummy one).

template <bool Z>
class ColReduce : public cs::Component<ColReduce<Z>> {
public:
    struct InputsZ {
        cs::Input<DType::f32, 2> P;
        cs::Input<DType::f32, 1> z;
        cs::Input<DType::u32, 1> n, R;
        cs::Input<DType::f32, 1> scale;
    };
    struct InputsNoZ {
        cs::Input<DType::f32, 2> P;
        cs::Input<DType::u32, 1> n, R;
        cs::Input<DType::f32, 1> scale;
    };
    using Inputs = std::conditional_t<Z, InputsZ, InputsNoZ>;
    struct Outputs {
        cs::Output<DType::f32, 1> x;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = first(in.n), strips = n / first(in.R);
        const float         scale  = first(in.scale);
        const std::uint32_t stride = static_cast<std::uint32_t>(it.get_group_range(0) * it.get_local_range(0));
        for (std::uint32_t j = static_cast<std::uint32_t>(it.get_global_id(0)); j < n; j += stride) {
            float acc = 0.0f;
            for (std::uint32_t c = 0; c < strips; ++c) acc += in.P[sycl::id<2>{c, j}];
            if constexpr (Z)
                out.x[sycl::id<1>{j}] = in.z[sycl::id<1>{j}] + scale * acc;
            else
                out.x[sycl::id<1>{j}] = scale * acc;
        }
    }

    Inputs  inputs;
    Outputs outputs;

private:
    int   n_, R_;
    float scale_;

public:
    ColReduce(int n, int R, float scale) : n_{n}, R_{R}, scale_{scale} {}
    void setup() override {
        this->declare_input(inputs.P, "P", n_ / R_, n_);
        if constexpr (Z) this->declare_input(inputs.z, "z", n_);  // only declared when Z
        this->declare_input(inputs.n, "n", 1);
        this->declare_input(inputs.R, "R", 1);
        this->declare_input(inputs.scale, "scale", 1);
        this->init_input("n", cs::init::fill(u32(n_)));
        this->init_input("R", cs::init::fill(u32(R_)));
        this->init_input("scale", cs::init::fill(scale_));
        this->declare_output(outputs.x, "x", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t range = static_cast<std::size_t>(n_);
        return this->create_launch_config(cs::range_desc(sycl::range<1>{range}), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// =============================================================================
// Gemv: out[i] = scale * (B[i][:] . x). A GROUP body: group g takes rows g, g + ngroups, ...
// =============================================================================

class Gemv : public cs::Component<Gemv> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::u32, 1> n;
        cs::Input<DType::f32, 1> scale;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = first(in.n);
        const float         scale = first(in.scale);
        const std::uint32_t l = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t wg = static_cast<std::uint32_t>(it.get_local_range(0));
        const std::uint32_t g = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t ng = static_cast<std::uint32_t>(it.get_group_range(0));  // this stage's groups
        for (std::uint32_t i = g; i < n; i += ng) {
            float acc = 0.0f;
            for (std::uint32_t j = l; j < n; j += wg) acc += in.B[sycl::id<2>{i, j}] * in.x[sycl::id<1>{j}];
            const float sum = cs::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            if (l == 0) out.y[sycl::id<1>{i}] = scale * sum;
        }
    }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_;
    float                     scale_;
    bool                      tied_b_;  // B is rank-only: a tie_inputs member (the tie alias owns its shape, D139)

public:
    Gemv(int n, float scale, bool tied_b = false) : n_{n}, scale_{scale}, tied_b_{tied_b} {}
    void setup() override {
        declare_input(inputs.n, "n", 1);
        declare_input(inputs.scale, "scale", 1);
        init_input("n", cs::init::fill(u32(n_)));
        init_input("scale", cs::init::fill(scale_));
        if (tied_b_)
            declare_input(inputs.B, "B", cs::unspecified_shape<2>());
        else
            declare_input(inputs.B, "B", n_, n_);
        declare_input(inputs.x, "x", n_);
        declare_output(outputs.y, "y", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t range = static_cast<std::size_t>(n_) * kWg;  // a group per row
        return create_launch_config(cs::range_desc(sycl::range<1>{range}), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// =============================================================================
// BicgkOnePass: q = B p and s = B^T r in ONE pass over B
// =============================================================================
// A group body over (strip c, 256-column block blk): work-item l owns column j = blk*256 + l.
//   Ps[c][j]   = sum_{k<R} r[i] * B[i][j]                     (i = c*R + k)
//   Pq[i][blk] = reduce_over_group(B[i][j] * p[j])            (per row i of the strip)
// RowReduce then sums Pq over the blocks (q), and the scale-1 ColReduce sums Ps (s).

class BicgkOnePass : public cs::Component<BicgkOnePass> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> p, r;
        cs::Input<DType::u32, 1> n, R;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> Ps, Pq;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = first(in.n), R = first(in.R);
        const std::uint32_t blocks = n / static_cast<std::uint32_t>(it.get_local_range(0)), total_groups = (n / R) * blocks;
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
                const float sum = cs::reduce_over_group(it.get_group(), b * pj, sycl::plus<float>());
                if (l == 0) out.Pq[sycl::id<2>{i, blk}] = sum;
            }
            out.Ps[sycl::id<2>{c, j}] = ps;
        }
    }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_, R_;

public:
    BicgkOnePass(int n, int R) : n_{n}, R_{R} {}
    void setup() override {
        declare_input(inputs.n, "n", 1);
        declare_input(inputs.R, "R", 1);
        init_input("n", cs::init::fill(u32(n_)));
        init_input("R", cs::init::fill(u32(R_)));
        declare_input(inputs.B, "B", n_, n_);
        declare_input(inputs.p, "p", n_);
        declare_input(inputs.r, "r", n_);
        declare_output(outputs.Ps, "Ps", n_ / R_, n_);
        declare_output(outputs.Pq, "Pq", n_, n_ / static_cast<int>(kWg));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / R_) * blocks;
        return create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// RowReduce: q[i] = sum_blk Pq[i][blk]

class RowReduce : public cs::Component<RowReduce> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> Pq;
        cs::Input<DType::u32, 1> n;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> q;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = first(in.n), blocks = n / static_cast<std::uint32_t>(kWg);
        const std::uint32_t stride = static_cast<std::uint32_t>(it.get_group_range(0) * it.get_local_range(0));
        for (std::uint32_t i = static_cast<std::uint32_t>(it.get_global_id(0)); i < n; i += stride) {
            float acc = 0.0f;
            for (std::uint32_t b = 0; b < blocks; ++b) acc += in.Pq[sycl::id<2>{i, b}];
            out.q[sycl::id<1>{i}] = acc;
        }
    }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_;

public:
    explicit RowReduce(int n) : n_{n} {}
    void setup() override {
        declare_input(inputs.n, "n", 1);
        init_input("n", cs::init::fill(u32(n_)));
        declare_input(inputs.Pq, "Pq", n_, n_ / static_cast<int>(kWg));
        declare_output(outputs.q, "q", n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t range = static_cast<std::size_t>(n_);
        return create_launch_config(cs::range_desc(sycl::range<1>{range}), cs::wg_desc(sycl::range<1>{kWg}));
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

class Rank2UpdateLocal : public cs::Component<Rank2UpdateLocal> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> A;
        cs::Input<DType::f32, 1> u1, v1, u2, v2;
        cs::Input<DType::u32, 1> n;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> B;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t blocks = first(in.n) / static_cast<std::uint32_t>(kWg);
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

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_;

public:
    explicit Rank2UpdateLocal(int n) : n_{n} {}
    void setup() override {
        declare_input(inputs.n, "n", 1);
        init_input("n", cs::init::fill(u32(n_)));
        declare_input(inputs.A, "A", n_, n_);
        declare_input(inputs.u1, "u1", n_);
        declare_input(inputs.v1, "v1", n_);
        declare_input(inputs.u2, "u2", n_);
        declare_input(inputs.v2, "v2", n_);
        declare_output(outputs.B, "B", n_, n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / static_cast<int>(kRt)) * blocks;
        auto              d      = create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

/// GemvTPartialLocal: P[c][j] = sum_k tile[k][j] * y[c kRt + k]. No global B input: the tile comes from Rank2UpdateLocal.

class GemvTPartialLocal : public cs::Component<GemvTPartialLocal> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> y;
        cs::Input<DType::u32, 1> n;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> P;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);  // the same block as Rank2UpdateLocal's
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t blocks = first(in.n) / static_cast<std::uint32_t>(kWg);
        const std::uint32_t c = g / blocks, blk = g - c * blocks;
        const std::uint32_t j = blk * static_cast<std::uint32_t>(kWg) + l;
        cs::group_barrier(it.get_group());  // the tile is complete
        float acc = 0.0f;
        for (std::uint32_t k = 0; k < kRt; ++k) acc += tile[k * kWg + l] * in.y[sycl::id<1>{c * kRt + k}];
        out.P[sycl::id<2>{c, j}] = acc;
    }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_;

public:
    explicit GemvTPartialLocal(int n) : n_{n} {}
    void setup() override {
        declare_input(inputs.n, "n", 1);
        init_input("n", cs::init::fill(u32(n_)));
        declare_input(inputs.y, "y", n_);
        declare_output(outputs.P, "P", n_ / static_cast<int>(kRt), n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / static_cast<int>(kRt)) * blocks;
        auto              d      = create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

/// BicgkTileLocal: loads the B tile into local memory and produces BicgkOnePass's partials from it:
///   Ps[c][j]   = sum_k r[i] B[i][j]                  Pq[i][blk] = reduce_over_group(B[i][j] p[j])   (i = c kRt + k)

class BicgkTileLocal : public cs::Component<BicgkTileLocal> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> p, r;
        cs::Input<DType::u32, 1> n;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> Ps, Pq;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t blocks = first(in.n) / static_cast<std::uint32_t>(kWg);
        const std::uint32_t c = g / blocks, blk = g - c * blocks;
        const std::uint32_t j  = blk * static_cast<std::uint32_t>(kWg) + l;
        const float         pj = in.p[sycl::id<1>{j}];
        float               ps = 0.0f;
        for (std::uint32_t k = 0; k < kRt; ++k) {
            const std::uint32_t i = c * kRt + k;
            const float         b = in.B[sycl::id<2>{i, j}];  // the ONE read of this element of B1
            tile[k * kWg + l]     = b;
            ps += in.r[sycl::id<1>{i}] * b;
            const float sum = cs::reduce_over_group(it.get_group(), b * pj, sycl::plus<float>());
            if (l == 0) out.Pq[sycl::id<2>{i, blk}] = sum;
        }
        out.Ps[sycl::id<2>{c, j}] = ps;
    }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_;

public:
    explicit BicgkTileLocal(int n) : n_{n} {}
    void setup() override {
        declare_input(inputs.n, "n", 1);
        init_input("n", cs::init::fill(u32(n_)));
        declare_input(inputs.B, "B", n_, n_);
        declare_input(inputs.p, "p", n_);
        declare_input(inputs.r, "r", n_);
        declare_output(outputs.Ps, "Ps", n_ / static_cast<int>(kRt), n_);
        declare_output(outputs.Pq, "Pq", n_, n_ / static_cast<int>(kWg));
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / static_cast<int>(kRt)) * blocks;
        auto              d      = create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

/// Rank2UpdateFromLocal: G2's B2 = B1 + u1b v1b^T + u2b v2b^T, with B1 taken from the tile BicgkTileLocal loaded (no global
/// read of B1 here, and no dependency on q or s, which is why it can share BiCGK's read of B1).

class Rank2UpdateFromLocal : public cs::Component<Rank2UpdateFromLocal> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> u1, v1, u2, v2;
        cs::Input<DType::u32, 1> n;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> B;
    };
    static void compute_impl(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t blocks = first(in.n) / static_cast<std::uint32_t>(kWg);
        const std::uint32_t c = g / blocks, blk = g - c * blocks;
        const std::uint32_t j = blk * static_cast<std::uint32_t>(kWg) + l;
        cs::group_barrier(it.get_group());  // the tile is complete
        const float v1j = in.v1[sycl::id<1>{j}], v2j = in.v2[sycl::id<1>{j}];
        for (std::uint32_t k = 0; k < kRt; ++k) {
            const std::uint32_t i = c * kRt + k;
            out.B[sycl::id<2>{i, j}] = tile[k * kWg + l] + in.u1[sycl::id<1>{i}] * v1j + in.u2[sycl::id<1>{i}] * v2j;
        }
    }

    Inputs  inputs;
    Outputs outputs;

private:
    int                       n_;

public:
    explicit Rank2UpdateFromLocal(int n) : n_{n} {}
    void setup() override {
        declare_input(inputs.n, "n", 1);
        init_input("n", cs::init::fill(u32(n_)));
        declare_input(inputs.u1, "u1", n_);
        declare_input(inputs.v1, "v1", n_);
        declare_input(inputs.u2, "u2", n_);
        declare_input(inputs.v2, "v2", n_);
        declare_output(outputs.B, "B", n_, n_);
    }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        const std::size_t blocks = static_cast<std::size_t>(n_) / kWg;
        const std::size_t groups = static_cast<std::size_t>(n_ / static_cast<int>(kRt)) * blocks;
        auto              d      = create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

}  // namespace gc
