// =============================================================================
// include/naga/core/primitives.hpp — framework compute primitives
// =============================================================================
//
// FRAMEWORK_CLEANUP S5c. Pre-built, tuned kernel primitives so applications
// never hand-roll (or worse, serialize) common parallel patterns. Everything
// here is application-agnostic: semantics arrive through user functors,
// storage arrives through the view concepts (concepts.hpp) or documented raw
// contracts, and execution geometry is configurable — the defaults are
// starting points measured on one machine, not truths (re-measure with
// NAGA_KERNEL_PROF=1 on your target hardware).
//
// All launches flow through the execute_compute choke point (kernel.hpp), so
// NAGA_KERNEL_PROF profiles report each primitive instantiation under its
// user-functor type name.
//
// Accumulation-order guarantee: NONE. Reductions use work-group tree
// reductions (sycl::reduce_over_group) plus grid striding; float results are
// run-to-run deterministic only insofar as the backend's group algorithms
// are. If you need a bitwise-reproducible sum, this is not the tool.
//
// Contents:
//   reduce_sum      — sum a ReadableView into a 1-element WritableView.
//   map_reduce_sum  — Σ transform(i) over [0, n), then finalize(sum) written
//                     to a WritableView (no temporary buffer).
//   map_reduce_apply— Σ transform(i) over [0, n), handed to a user Sink on
//                     exactly one thread. For destinations that are not
//                     WritableViews (e.g. atomic gradient accumulation).
//   csr_row_reduce  — per-row gather-reduce over a CSR structure with a
//                     user row-operator (multi-accumulator, per-edge side
//                     effects, per-row epilogue). The workhorse for
//                     neighbor-list / sparse-operator physics.
//   sell_row_reduce — per-row gather-reduce over a SELL-C-σ structure,
//                     thread-per-row with private register accumulation.
//                     Same RowOp contract as csr_row_reduce; the right shape
//                     for ~uniform-degree rows (padded slice-local
//                     column-major storage → lane-adjacent coalesced loads).
// =============================================================================

#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <naga/core/concepts.hpp>
#include <naga/core/kernel.hpp>

namespace naga::core {

// ── reduce_sum / map_reduce_sum ──────────────────────────────────────────────
//
// Single work-group, grid-stride: one WG of `wg_size` threads strides the
// index space, then one tree reduction. This shape is intentionally simple —
// it needs no scratch buffer and no zero-initialized destination (the result
// is WRITTEN, not accumulated), and it is bandwidth-bound and fast for the
// n ≲ 10⁶ range. If a profile ever shows a reduction of that size on the hot
// path, add a two-pass multi-group variant rather than growing this one.

struct ReduceConfig {
    // Work-group size. Powers of two recommended. Default measured on a
    // 16k-element reduction (RTX 3070); re-measure on target hardware.
    uint32_t wg_size = 256;
};

namespace detail {

template <class Transform, class Finalize, class OutV>
struct MapReduceSumKernel {
    Transform transform;   // float operator()(size_t i) const
    Finalize  finalize;    // float operator()(float sum) const
    OutV      out;         // 1-element destination, written at out_index
    size_t    n, out_index;
    uint32_t  wg;

    sycl::nd_range<1> nd_range() const {
        return sycl::nd_range<1>{sycl::range<1>{wg}, sycl::range<1>{wg}};
    }

    void operator()(sycl::nd_item<1> it) const {
        const uint32_t lid = static_cast<uint32_t>(it.get_local_id(0));
        float part = 0.0f;
        for (size_t i = lid; i < n; i += wg)
            part += transform(i);
        const float total = sycl::reduce_over_group(
            it.get_group(), part, sycl::plus<float>());
        if (lid == 0)
            out.write(out_index, finalize(total));
    }
};

// Identity pieces for the plain-view reduce_sum form.
template <class InV>
struct ViewReadTransform {
    InV in;
    float operator()(size_t i) const { return static_cast<float>(in.get(i)); }
};
struct IdentityFinalize {
    float operator()(float x) const { return x; }
};

template <class Transform, class Sink>
struct MapReduceApplyKernel {
    Transform transform;   // float operator()(size_t i) const
    Sink      sink;        // void operator()(float sum) const — runs ONCE
    size_t    n;
    uint32_t  wg;

    sycl::nd_range<1> nd_range() const {
        return sycl::nd_range<1>{sycl::range<1>{wg}, sycl::range<1>{wg}};
    }

    void operator()(sycl::nd_item<1> it) const {
        const uint32_t lid = static_cast<uint32_t>(it.get_local_id(0));
        float part = 0.0f;
        for (size_t i = lid; i < n; i += wg)
            part += transform(i);
        const float total = sycl::reduce_over_group(
            it.get_group(), part, sycl::plus<float>());
        if (lid == 0)
            sink(total);
    }
};

} // namespace detail

// Σ_{i<n} transform(i) → out.write(out_index, finalize(sum)).
// transform/finalize must be trivially copyable device-callable functors.
template <class Transform, class Finalize, WritableView OutV>
inline void map_reduce_sum(sycl::queue& q, size_t n, Transform transform,
                           Finalize finalize, OutV out, size_t out_index = 0,
                           ReduceConfig cfg = {}) {
    static_assert(std::is_trivially_copyable_v<Transform> &&
                  std::is_trivially_copyable_v<Finalize>,
                  "map_reduce_sum: functors must be trivially copyable");
    execute_compute(q, detail::MapReduceSumKernel<Transform, Finalize, OutV>{
        transform, finalize, out, n, out_index,
        cfg.wg_size ? cfg.wg_size : 256u});
}

// Sum of an entire ReadableView → out.write(out_index, sum).
template <ReadableView InV, WritableView OutV>
inline void reduce_sum(sycl::queue& q, InV in, OutV out, size_t out_index = 0,
                       ReduceConfig cfg = {}) {
    map_reduce_sum(q, in.size(), detail::ViewReadTransform<InV>{in},
                   detail::IdentityFinalize{}, out, out_index, cfg);
}

// Σ_{i<n} transform(i) → sink(sum), where Sink is an arbitrary user functor
// called on exactly ONE thread after the group tree reduction. Use when the
// destination is not a WritableView — the canonical case is a single atomic
// accumulation into a gradient (e.g. Input::accum_grad on a shared design
// variable), turning per-contributor atomic storms into one atomic per call.
// Sink side effects run exactly once per invocation; atomics inside the sink
// are explicitly allowed.
template <class Transform, class Sink>
inline void map_reduce_apply(sycl::queue& q, size_t n, Transform transform,
                             Sink sink, ReduceConfig cfg = {}) {
    static_assert(std::is_trivially_copyable_v<Transform> &&
                  std::is_trivially_copyable_v<Sink>,
                  "map_reduce_apply: functors must be trivially copyable");
    execute_compute(q, detail::MapReduceApplyKernel<Transform, Sink>{
        transform, sink, n, cfg.wg_size ? cfg.wg_size : 256u});
}

// ── csr_row_reduce ───────────────────────────────────────────────────────────
//
// Per-row gather-reduce over a CSR sparse structure.
//
// CSR contract (textbook): row_ptr has n_rows_total+1 monotonically
// non-decreasing entries; the edges of row r are the index range
// [row_ptr[r], row_ptr[r+1]) into col_ind (and into any parallel per-edge
// arrays the row operator captures). This primitive processes rows
// [row_begin, row_begin + n_rows).
//
// Row operator contract (RowOp, user-supplied, trivially copyable):
//
//   int   acc_count() const;
//       Number of float accumulators in use (uniform across rows,
//       1 ≤ acc_count() ≤ MaxAcc).
//
//   State begin(size_t row) const;
//       Per-row prologue. Runs on EVERY thread assigned to the row (compute
//       redundantly in registers or keep State small and recompute heavy
//       values in end()). State must be trivially copyable.
//
//   void edge(State&, float* acc, size_t row, IndexT e, IndexT col) const;
//       Called once per edge (e = edge index, col = col_ind[e]), distributed
//       across the row's threads. Add contributions into acc[0..acc_count);
//       per-edge side effects (e.g. atomic gradient scatter through a view's
//       accum_grad) are explicitly allowed.
//
//   void end(const State&, const float* acc, size_t row) const;
//       Per-row epilogue, called on ONE thread with the group-summed
//       accumulators. Write row outputs / accumulate row-level gradients
//       here (atomic epilogues via AtomicAddView/GradAccumView views are the
//       intended pattern when rows share targets).
//
// Accumulation order across a row's edges is NOT specified (tree reduction +
// thread striding). Per-edge side effects run exactly once per edge.

struct CsrRowReduceConfig {
    // Threads cooperating on one row (the work-group size; one WG per row).
    // Powers of two recommended. Default measured on rows of mean degree ~64
    // (RTX 3070); re-measure with NAGA_KERNEL_PROF=1 on target hardware and
    // adjust to your degree distribution.
    uint32_t threads_per_row = 32;
};

namespace detail {

template <int MaxAcc, class IndexT, class RowOp>
struct CsrRowReduceKernel {
    RowOp         op;
    const IndexT* row_ptr;
    const IndexT* col_ind;
    size_t        row_begin, n_rows;
    uint32_t      tpr;

    sycl::nd_range<1> nd_range() const {
        return sycl::nd_range<1>{sycl::range<1>{n_rows * tpr},
                                 sycl::range<1>{tpr}};
    }

    void operator()(sycl::nd_item<1> it) const {
        const size_t   row = row_begin + it.get_group(0);
        const uint32_t lid = static_cast<uint32_t>(it.get_local_id(0));
        const int      na  = op.acc_count();

        auto state = op.begin(row);

        float acc[MaxAcc];
        for (int a = 0; a < na; ++a) acc[a] = 0.0f;

        const IndexT rs = row_ptr[row];
        const IndexT re = row_ptr[row + 1];
        for (IndexT e = rs + static_cast<IndexT>(lid); e < re;
             e += static_cast<IndexT>(tpr))
            op.edge(state, acc, row, e, col_ind[e]);

        // Group-sum each accumulator (uniform loop bound — group algorithm
        // requirement).
        for (int a = 0; a < na; ++a)
            acc[a] = sycl::reduce_over_group(it.get_group(), acc[a],
                                             sycl::plus<float>());

        if (lid == 0)
            op.end(state, acc, row);
    }
};

} // namespace detail

template <int MaxAcc, class IndexT, class RowOp>
inline void csr_row_reduce(sycl::queue& q, const IndexT* row_ptr,
                           const IndexT* col_ind, size_t row_begin,
                           size_t n_rows, RowOp op,
                           CsrRowReduceConfig cfg = {}) {
    static_assert(MaxAcc >= 1, "csr_row_reduce: MaxAcc must be >= 1");
    static_assert(std::is_trivially_copyable_v<RowOp>,
                  "csr_row_reduce: RowOp must be trivially copyable");
    static_assert(
        std::is_trivially_copyable_v<decltype(op.begin(size_t{0}))>,
        "csr_row_reduce: RowOp::State must be trivially copyable");
    if (n_rows == 0) return;
    execute_compute(q, detail::CsrRowReduceKernel<MaxAcc, IndexT, RowOp>{
        op, row_ptr, col_ind, row_begin, n_rows,
        cfg.threads_per_row ? cfg.threads_per_row : 32u});
}

// ── sell_row_reduce ──────────────────────────────────────────────────────────
//
// Per-row gather-reduce over a SELL-C-σ sparse structure.
//
// SELL contract: rows are grouped into slices of `chunk` (= C) consecutive
// rows; row r occupies lane r % C of slice r / C. slice_base[s] is the first
// slot of slice s (slice_base has n_slices+1 entries); the j-th REAL edge of
// row r (j < row_ptr[r+1] − row_ptr[r]; padding is never enumerated) lives at
// slot slice_base[r/C] + j·C + r%C — slice-local column-major, so the
// per-row threads of a slice read lane-adjacent (coalesced) slots at each j.
//
// Execution shape: THREAD-PER-ROW, private register accumulators — no group
// reduction. Right for ~uniform row degrees (e.g. adaptive-radius stencil
// constructions targeting a fixed neighbor count per row); for wildly
// skewed degrees prefer csr_row_reduce's cooperative rows.
//
// RowOp contract: identical to csr_row_reduce (acc_count/begin/edge/end),
// with the simplification that begin/edge/end all run on the row's single
// thread (State need not be recomputed anywhere; end sees the private acc).
// Edge indices passed to edge() are SLOT indices into the SELL arrays.

struct SellRowReduceConfig {
    // Work-group size (thread-per-row → rows per work-group). S9 ruling:
    // 256, defaults everywhere until measurement earns boutique tuning.
    uint32_t wg_size = 256;
};

namespace detail {

template <int MaxAcc, class RowOp>
struct SellRowReduceKernel {
    RowOp           op;
    const uint32_t* slice_base;
    const uint32_t* sell_col;
    const uint32_t* row_ptr;   // REAL degrees: deg(r) = row_ptr[r+1]-row_ptr[r]
    uint32_t        chunk;
    size_t          row_begin, n_rows;
    uint32_t        wg;

    sycl::nd_range<1> nd_range() const {
        const size_t global = (n_rows + wg - 1) / wg * wg;
        return sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{wg}};
    }

    void operator()(sycl::nd_item<1> it) const {
        const size_t i = it.get_global_id(0);
        if (i >= n_rows) return;
        const size_t   row   = row_begin + i;
        const uint32_t slice = static_cast<uint32_t>(row) / chunk;
        const uint32_t lane  = static_cast<uint32_t>(row) % chunk;
        const uint32_t base  = slice_base[slice];
        const uint32_t deg   = row_ptr[row + 1] - row_ptr[row];

        auto state = op.begin(row);
        float acc[MaxAcc];
        const int na = op.acc_count();
        for (int a = 0; a < na; ++a) acc[a] = 0.0f;

        for (uint32_t j = 0; j < deg; ++j) {
            const uint32_t e = base + j * chunk + lane;
            op.edge(state, acc, row, e, sell_col[e]);
        }
        op.end(state, acc, row);
    }
};

} // namespace detail

template <int MaxAcc, class RowOp>
inline void sell_row_reduce(sycl::queue& q, const uint32_t* slice_base,
                            const uint32_t* sell_col, const uint32_t* row_ptr,
                            uint32_t chunk, size_t row_begin, size_t n_rows,
                            RowOp op, SellRowReduceConfig cfg = {}) {
    static_assert(MaxAcc >= 1, "sell_row_reduce: MaxAcc must be >= 1");
    static_assert(std::is_trivially_copyable_v<RowOp>,
                  "sell_row_reduce: RowOp must be trivially copyable");
    static_assert(
        std::is_trivially_copyable_v<decltype(op.begin(size_t{0}))>,
        "sell_row_reduce: RowOp::State must be trivially copyable");
    if (n_rows == 0) return;
    execute_compute(q, detail::SellRowReduceKernel<MaxAcc, RowOp>{
        op, slice_base, sell_col, row_ptr, chunk, row_begin, n_rows,
        cfg.wg_size ? cfg.wg_size : 256u});
}

} // namespace naga::core
