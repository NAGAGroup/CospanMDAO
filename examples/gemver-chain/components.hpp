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
//     `stride` being the launch size, read from the item.
//   - The launch range is the full work.
//   - SIZES COME FROM THE PORTS. A kernel reads n, the strip height and the block count from the shapes of its ports
//     (`in.B.shape()`, `out.Ps.shape()`, ...), and a launch range comes from the shapes of the ports too. Every data port is
//     declared with an unspecified shape: the owning System (systems.hpp) gives the free inputs and the partial-sum outputs
//     their shapes with shape hooks, every other output takes its shape in prepare_forward from the shapes of its inputs.
//     The one value no shape carries is Rank2Update's strip height R: it is a constructor argument (the host launch range)
//     and a u32 input of extent 1 that Rank2Update initializes itself from that argument.
//   - `scale` is an f32 input of extent 1; the System ties the users of one value into an alias and initializes it.
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

inline std::uint32_t first(const cs::Input<DType::u32, 1>& v) { return v[std::size_t{0}]; }
inline float         first(const cs::Input<DType::f32, 1>& v) { return v[std::size_t{0}]; }

// =============================================================================
// Rank2Update: t = (c = t / n, j = t % n) over (n / R) * n items
// =============================================================================
// n is A's shape. R is the one size no shape carries: the constructor argument (the host launch range) and an extent-1 u32
// input that setup() initializes from it.

class Rank2Update : public cs::Component<Rank2Update> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> A;
        cs::Input<DType::f32, 1> u1, v1, u2, v2;
        cs::Input<DType::u32, 1> R;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> B;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = u32(in.A.shape()[1]), R = first(in.R), total = (n / R) * n;
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
    int R_;

public:
    explicit Rank2Update(int R) : R_{R} {}
    void declare() override {
        declare_input(inputs.A, "A", cs::unspecified_shape<2>());
        declare_input(inputs.u1, "u1", cs::unspecified_shape<1>());
        declare_input(inputs.v1, "v1", cs::unspecified_shape<1>());
        declare_input(inputs.u2, "u2", cs::unspecified_shape<1>());
        declare_input(inputs.v2, "v2", cs::unspecified_shape<1>());
        declare_input(inputs.R, "R", 1);
        init_input("R", cs::init::fill(u32(R_)));
        declare_output(outputs.B, "B", cs::unspecified_shape<2>());
    }
    void prepare_forward() override {
        const sycl::range<2> a = shape_of(inputs.A);
        set_shape(outputs.B, a[0], a[1]);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t n    = inputs.A.shape()[1];
        const std::size_t work = (n / static_cast<std::size_t>(R_)) * n;
        return create_launch_config(cs::range_desc(sycl::range<1>{work}), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// =============================================================================
// GemvTPartial: the same item mapping as Rank2Update; one partial column sum per strip
// =============================================================================
// n = B's columns, R = B's rows / P's rows (P is hooked to (n / R, n) by the owning System).

class GemvTPartial : public cs::Component<GemvTPartial> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> y;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> P;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = u32(in.B.shape()[1]), R = u32(in.B.shape()[0] / out.P.shape()[0]), total = (n / R) * n;
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

    void declare() override {
        declare_input(inputs.B, "B", cs::unspecified_shape<2>());
        declare_input(inputs.y, "y", cs::unspecified_shape<1>());
        declare_output(outputs.P, "P", cs::unspecified_shape<2>());  // hooked by the owning System: (n / R, n)
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t range = outputs.P.shape()[0] * outputs.P.shape()[1];  // (n / R) * n
        return create_launch_config(cs::range_desc(sycl::range<1>{range}), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// =============================================================================
// ColReduce<Z>: x[j] = (Z ? z[j] : 0) + scale * sum_c P[c][j]
// =============================================================================
// Z = true is GEMVER's x (scale beta, with z); Z = false is BiCGK's s (scale 1, no z
// input at all, not a dummy one). n = P's columns, strips = P's rows.

template <bool Z>
class ColReduce : public cs::Component<ColReduce<Z>> {
public:
    struct InputsZ {
        cs::Input<DType::f32, 2> P;
        cs::Input<DType::f32, 1> z;
        cs::Input<DType::f32, 1> scale;
    };
    struct InputsNoZ {
        cs::Input<DType::f32, 2> P;
        cs::Input<DType::f32, 1> scale;
    };
    using Inputs = std::conditional_t<Z, InputsZ, InputsNoZ>;
    struct Outputs {
        cs::Output<DType::f32, 1> x;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = u32(in.P.shape()[1]), strips = u32(in.P.shape()[0]);
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

    void declare() override {
        this->declare_input(inputs.P, "P", cs::unspecified_shape<2>());
        if constexpr (Z) this->declare_input(inputs.z, "z", cs::unspecified_shape<1>());  // only declared when Z
        this->declare_input(inputs.scale, "scale", cs::unspecified_shape<1>());  // a tie_inputs member; the System sets the alias
        this->declare_output(outputs.x, "x", cs::unspecified_shape<1>());
    }
    void prepare_forward() override { this->set_shape(outputs.x, this->shape_of(inputs.P)[1]); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t range = inputs.P.shape()[1];
        return this->create_launch_config(cs::range_desc(sycl::range<1>{range}), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// =============================================================================
// Gemv: out[i] = scale * (B[i][:] . x). A GROUP body: group g takes rows g, g + ngroups, ...
// =============================================================================
// n = B's rows.

class Gemv : public cs::Component<Gemv> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> x;
        cs::Input<DType::f32, 1> scale;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = u32(in.B.shape()[0]);
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

    void declare() override {
        declare_input(inputs.B, "B", cs::unspecified_shape<2>());  // the rank-only B of a tie_inputs member, or connected
        declare_input(inputs.x, "x", cs::unspecified_shape<1>());
        declare_input(inputs.scale, "scale", cs::unspecified_shape<1>());  // a tie_inputs member; the System sets the alias
        declare_output(outputs.y, "y", cs::unspecified_shape<1>());
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.B)[0]); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t range = inputs.B.shape()[0] * kWg;  // a group per row
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
// n = B's rows, R = B's rows / Ps's rows (Ps is hooked to (n / R, n) by the owning System); Pq is (n, n / kWg).

class BicgkOnePass : public cs::Component<BicgkOnePass> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> p, r;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> Ps, Pq;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = u32(in.B.shape()[0]), R = u32(in.B.shape()[0] / out.Ps.shape()[0]);
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

    void declare() override {
        declare_input(inputs.B, "B", cs::unspecified_shape<2>());
        declare_input(inputs.p, "p", cs::unspecified_shape<1>());
        declare_input(inputs.r, "r", cs::unspecified_shape<1>());
        declare_output(outputs.Ps, "Ps", cs::unspecified_shape<2>());  // hooked by the owning System: (n / R, n)
        declare_output(outputs.Pq, "Pq", cs::unspecified_shape<2>());
    }
    void prepare_forward() override {
        const sycl::range<2> b = shape_of(inputs.B);
        set_shape(outputs.Pq, b[0], b[0] / kWg);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t blocks = inputs.B.shape()[0] / kWg;
        const std::size_t groups = outputs.Ps.shape()[0] * blocks;  // (n / R) * blocks
        return create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
    }
};

// RowReduce: q[i] = sum_blk Pq[i][blk]      (n = Pq's rows, blocks = Pq's columns)

class RowReduce : public cs::Component<RowReduce> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> Pq;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> q;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        const std::uint32_t n = u32(in.Pq.shape()[0]), blocks = u32(in.Pq.shape()[1]);
        const std::uint32_t stride = static_cast<std::uint32_t>(it.get_group_range(0) * it.get_local_range(0));
        for (std::uint32_t i = static_cast<std::uint32_t>(it.get_global_id(0)); i < n; i += stride) {
            float acc = 0.0f;
            for (std::uint32_t b = 0; b < blocks; ++b) acc += in.Pq[sycl::id<2>{i, b}];
            out.q[sycl::id<1>{i}] = acc;
        }
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.Pq, "Pq", cs::unspecified_shape<2>());
        declare_output(outputs.q, "q", cs::unspecified_shape<1>());
    }
    void prepare_forward() override { set_shape(outputs.q, shape_of(inputs.Pq)[0]); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t range = inputs.Pq.shape()[0];
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
/// n = A's rows.

class Rank2UpdateLocal : public cs::Component<Rank2UpdateLocal> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> A;
        cs::Input<DType::f32, 1> u1, v1, u2, v2;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> B;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t blocks = u32(in.A.shape()[0]) / static_cast<std::uint32_t>(kWg);
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

    void declare() override {
        declare_input(inputs.A, "A", cs::unspecified_shape<2>());
        declare_input(inputs.u1, "u1", cs::unspecified_shape<1>());
        declare_input(inputs.v1, "v1", cs::unspecified_shape<1>());
        declare_input(inputs.u2, "u2", cs::unspecified_shape<1>());
        declare_input(inputs.v2, "v2", cs::unspecified_shape<1>());
        declare_output(outputs.B, "B", cs::unspecified_shape<2>());
    }
    void prepare_forward() override {
        const sycl::range<2> a = shape_of(inputs.A);
        set_shape(outputs.B, a[0], a[1]);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t n      = inputs.A.shape()[0];
        const std::size_t blocks = n / kWg;
        const std::size_t groups = (n / kRt) * blocks;
        auto              d      = create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

/// GemvTPartialLocal: P[c][j] = sum_k tile[k][j] * y[c kRt + k]. No global B input: the tile comes from Rank2UpdateLocal.
/// n = y's length; P is hooked to (n / kRt, n) by the owning System.

class GemvTPartialLocal : public cs::Component<GemvTPartialLocal> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> y;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> P;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);  // the same block as Rank2UpdateLocal's
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t blocks = u32(in.y.shape()[0]) / static_cast<std::uint32_t>(kWg);
        const std::uint32_t c = g / blocks, blk = g - c * blocks;
        const std::uint32_t j = blk * static_cast<std::uint32_t>(kWg) + l;
        cs::group_barrier(it.get_group());  // the tile is complete
        float acc = 0.0f;
        for (std::uint32_t k = 0; k < kRt; ++k) acc += tile[k * kWg + l] * in.y[sycl::id<1>{c * kRt + k}];
        out.P[sycl::id<2>{c, j}] = acc;
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.y, "y", cs::unspecified_shape<1>());
        declare_output(outputs.P, "P", cs::unspecified_shape<2>());  // hooked by the owning System
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t n      = inputs.y.shape()[0];
        const std::size_t blocks = n / kWg;
        const std::size_t groups = (n / kRt) * blocks;
        auto              d      = create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

/// BicgkTileLocal: loads the B tile into local memory and produces BicgkOnePass's partials from it:
///   Ps[c][j]   = sum_k r[i] B[i][j]                  Pq[i][blk] = reduce_over_group(B[i][j] p[j])   (i = c kRt + k)
/// n = B's rows; Ps is hooked to (n / kRt, n) by the owning System, Pq is (n, n / kWg).

class BicgkTileLocal : public cs::Component<BicgkTileLocal> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> B;
        cs::Input<DType::f32, 1> p, r;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> Ps, Pq;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t blocks = u32(in.B.shape()[0]) / static_cast<std::uint32_t>(kWg);
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

    void declare() override {
        declare_input(inputs.B, "B", cs::unspecified_shape<2>());
        declare_input(inputs.p, "p", cs::unspecified_shape<1>());
        declare_input(inputs.r, "r", cs::unspecified_shape<1>());
        declare_output(outputs.Ps, "Ps", cs::unspecified_shape<2>());  // hooked by the owning System
        declare_output(outputs.Pq, "Pq", cs::unspecified_shape<2>());
    }
    void prepare_forward() override {
        const sycl::range<2> b = shape_of(inputs.B);
        set_shape(outputs.Pq, b[0], b[0] / kWg);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t n      = inputs.B.shape()[0];
        const std::size_t blocks = n / kWg;
        const std::size_t groups = (n / kRt) * blocks;
        auto              d      = create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

/// Rank2UpdateFromLocal: G2's B2 = B1 + u1b v1b^T + u2b v2b^T, with B1 taken from the tile BicgkTileLocal loaded (no global
/// read of B1 here, and no dependency on q or s, which is why it can share BiCGK's read of B1). n = u1's length.

class Rank2UpdateFromLocal : public cs::Component<Rank2UpdateFromLocal> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> u1, v1, u2, v2;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> B;
    };
    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out) {
        float*              tile = it.local<float>(static_cast<std::size_t>(kRt) * kWg);
        const std::uint32_t l    = static_cast<std::uint32_t>(it.get_local_id(0));
        const std::uint32_t g    = static_cast<std::uint32_t>(it.get_group(0));
        const std::uint32_t blocks = u32(in.u1.shape()[0]) / static_cast<std::uint32_t>(kWg);
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

    void declare() override {
        declare_input(inputs.u1, "u1", cs::unspecified_shape<1>());
        declare_input(inputs.v1, "v1", cs::unspecified_shape<1>());
        declare_input(inputs.u2, "u2", cs::unspecified_shape<1>());
        declare_input(inputs.v2, "v2", cs::unspecified_shape<1>());
        declare_output(outputs.B, "B", cs::unspecified_shape<2>());
    }
    void prepare_forward() override {
        const std::size_t n = shape_of(inputs.u1)[0];
        set_shape(outputs.B, n, n);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        const std::size_t n      = inputs.u1.shape()[0];
        const std::size_t blocks = n / kWg;
        const std::size_t groups = (n / kRt) * blocks;
        auto              d      = create_launch_config(cs::range_desc(sycl::range<1>{groups * kWg}), cs::wg_desc(sycl::range<1>{kWg}));
        d->request_local_bytes(kTileBytes);
        return d;
    }
};

}  // namespace gc
