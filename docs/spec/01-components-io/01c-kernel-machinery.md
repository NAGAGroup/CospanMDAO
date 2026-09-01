# Spec 01.C — kernel machinery

Part of the phase-1 spec (`docs/spec/01-components-io/`). Binding contract;
declarations only. Wrappers: 01.A. Metadata/shapes: 01.B. Component base
and provided kernels: 01.D. System-side execution: 01.E.

## The map taxonomy is doctrine, not metadata

Bijective, injective, surjective: the completeness argument — each class
being implementable guarantees Components can express any function — and
the roadmap for provided kernels (reductions, scatters, scans, sorts).
Kernels are NEVER classified by their authors; no class field exists.
The general case is guaranteed by construction: indexed access plus atomic
dtypes (01.A), available in every kernel body.

## The KernelImpl contract

```cpp
namespace naga::cospan {

/// THE KERNELIMPL CONTRACT — the user's one kernel, as a type.
///   * A trivially copyable struct; members are I/O wrappers (01.A) plus
///     POD configuration. It is copied byte-wise into its param block.
///   * static constexpr int kernel_rank;   // iteration-space rank, 1..3
///   * exactly ONE body, in exactly one of two signatures:
///       void operator()(sycl::id<kernel_rank>) const;
///       void operator()(sycl::nd_item<kernel_rank>,
///                       sycl::id<kernel_rank>) const;
///     THE SIGNATURE IS THE DECLARATION: a body without the nd_item
///     cannot touch group abstractions (it has no group object). Bodies
///     wanting group semantics take the nd_item form — at the rank they
///     mean it: an nd_item body's rank IS its group-geometry rank.
///   * REQUIREMENT (phase-8 keystone): the struct SHALL be a template
///     over its value mode, using wrapper `value_type` exclusively in
///     the body. Opt-out: components supplying an analytical adjoint.
template <class KI>
concept KernelImplC =
    std::is_trivially_copyable_v<KI> &&
    requires { { KI::kernel_rank } -> std::convertible_to<int>; } &&
    (PlainBody<KI> || GroupBody<KI>);   // exactly one of the signatures

/// Device-side positional reader of a param block region. Position IS
/// identity: each stage in a fused sequence consumes its own block in
/// arena order — no stage ids exist anywhere (which is also what makes
/// the same KernelImpl type legal twice in one wave).
struct ParamCursor {
    const std::byte* pos;
    /// Align up to alignof(Block), return the block, advance past it.
    template <class Block> const Block& next();
};

/// The typed param block. FIELD OWNERSHIP IS SPLIT:
///   component-bound (written by bind(), 01.D): impl, range, flat_extent
///   System-stamped (written at launch preparation): the GROUP-INDEX
///     slice {group_offset, group_count} — slices are counted in whole
///     work-groups, so a group never straddles components under ANY
///     geometry. For an individual launch the stamp is the degenerate
///     {0, all_groups}.
template <KernelImplC KI>
struct KernelImplWrapper {
    KI                            impl;
    sycl::range<KI::kernel_rank>  range;        ///< LOGICAL iteration space
    size_t                        flat_extent;  ///< product of range
    size_t                        group_offset; ///< System-stamped
    size_t                        group_count;  ///< System-stamped

    /// Masking. THE INVARIANT IS TWO LINES:
    ///   1. Slices are group-index slices — the wrapper's check
    ///      (group_offset <= get_group_linear_id() < group_offset +
    ///      group_count) is GROUP-UNIFORM by construction, under any
    ///      launch geometry. Group barriers are group-scoped in SYCL —
    ///      groups owe each other nothing — so nothing further exists to
    ///      check at orchestration level.
    ///   2. Each KernelImpl is correct from its own scoped view, exactly
    ///      as if launched standalone.
    /// KERNELS ARE NEVER WRITTEN DIFFERENTLY FOR FUSION. Concretely:
    ///   * id-only bodies: the wrapper converts the launch nd_item to the
    ///     impl's declared-rank global sycl::id (stage-local flat index =
    ///     (group_linear_id - group_offset) * prod(local_range) +
    ///     local_linear_id, then row-major divmod against `range`;
    ///     sycl::id is constructible — verified, unlike nd_item) and ALSO
    ///     masks the padding tail (return when stage-local flat >=
    ///     flat_extent) — legal because the signature proves the body
    ///     group-free. Such bodies contain ZERO guards, see exactly
    ///     their logical range, and run under ANY launch rank. THE POINT
    ///     OF THE id-ONLY FORM IS UX AND NOTHING ELSE: work-groups are
    ///     opt-in for users while the root kernel always operates via
    ///     work-groups — the wrapper hands the impl its component-scoped
    ///     global id and returns before the impl ever sees a thread
    ///     index beyond what it asked for. Clean group boundaries stay
    ///     enforced regardless: not splitting groups among tasks is good
    ///     practice even where the spec minimally permits it, and an
    ///     early-returning tail is the absence of a task, not a task.
    ///   * nd_item bodies: receive their full padded slice, tail
    ///     included — the wrapper cannot see whether the body barriers,
    ///     so it must not exit items on the body's behalf. The body's
    ///     own freedom is bounded by SYCL 2020 §4.17.3, verbatim: group
    ///     functions "must be encountered in converged control flow by
    ///     all work-items in the group ... under the same set of
    ///     conditions" — skipping one via early exit, OR calling it from
    ///     divergent branches/iterations, is undefined behavior.
    ///     Divergence is legal in code free of group functions. We spec
    ///     to the spec, not to hardware tolerance.
    ///     Barrier-free nd_item bodies may early-return their tail;
    ///     barrier-bearing bodies guard the loads, never the
    ///     participation. Handling ragged n % wg is the standard
    ///     obligation of any group-using nd_range kernel in any
    ///     framework, identical fused or standalone.
    void operator()(sycl::nd_item<KI::kernel_rank> it) const; // nd_item rank
                                    // = launch rank; see rank rules below
};

/// The placeholders — the dynamic-function call sites, ONE PER LAUNCH
/// RANK (the nd_item in the signature fixes the rank):
void stage_r1(ParamCursor& cur, sycl::nd_item<1> it);
void stage_r2(ParamCursor& cur, sycl::nd_item<2> it);
void stage_r3(ParamCursor& cur, sycl::nd_item<3> it);
/// This split costs nothing at rest: the SSCP flow lowers a kernel to
/// device code only when its codepath is first invoked — an unhit
/// placeholder rank never becomes device code.

/// Kernel registration and geometry. Called from the component's setup
/// (01.D), once per KernelImpl type.
///
///   createKernelWrapper<KI>(sycl::range<R> global_range)
///     — R must equal KI's argument rank (compile-checked). global_range
///     is the kernel's LOGICAL extent; the framework pads to work-group
///     multiples in all paths. Stored internally as nd_range<3> + rank.
///     A rank-appropriate default work-group shape applies
///     (implementation-tunable, not a contract constant).
///
///   createKernelWrapper<KI>(sycl::range<R> global_range,
///                           sycl::range<R> wg_shape)
///     — replaces the default work-group shape. Available ONLY for
///     nd_item-taking KernelImpls (same rank, compile-checked).
///
/// The generated SYCL_EXTERNAL wrapper is registered against the
/// placeholder of its launch rank:
///   { auto& blk = cur.next<KernelImplWrapper<KI>>(); blk(it); }
struct KernelRegistration {
    sycl::AdaptiveCpp_jit::dynamic_function_id id;
    int             launch_rank;
    sycl::nd_range<3> geometry;   ///< logical global + wg shape, 1-padded
    size_t          block_size;
    size_t          block_align;
};

} // namespace naga::cospan
```

## The pipeline

Three outer kernels — one per launch rank, each the single call site of
its placeholder. The full launch mechanism, end to end:

```cpp
// Once at startup, per rank: the placeholder's type-erased handle.
//   auto stage_rR_id = sycl::AdaptiveCpp_jit::dynamic_function<
//       void, ParamCursor&, sycl::nd_item<R>>{&stage_rR}.id();
// (Wrapper ids come from createKernelWrapper<KI>() at setup.)

// Per launch: obtain the config — from the System's cache, keyed by
// (rank, ordered wrapper-id sequence); construct + define only on miss.
// acpp laws: one config per launch; construction has overhead (reuse);
// the config object must outlive kernel completion.
sycl::AdaptiveCpp_jit::dynamic_function_config& cfg =
    config_cache.get_or_create(key, [&](auto& c){
        // individual launch — one stage:
        c.define(stage_rR_id, wrapper_KI_id);
        // fused launch — the wave, in execution order:
        // c.define_as_call_sequence(stage_rR_id, {w1_id, ..., wN_id});
    });

// The launch — identical shape for individual and fused; the JIT
// expands the placeholder call site per cfg (a single wrapper call, or
// the sequence inlined in order):
q.parallel_for(sycl::nd_range<R>{padded_global, wg_shape},
    cfg.apply([=](sycl::nd_item<R> it){
        ParamCursor cur{params_base};
        stage_rR(cur, it);
    }));
```

- **Rank dispatch**: the System executor switches on the stored rank and
  issues the correctly-typed launch. For id-only stages the choice of
  launch rank is immaterial (the wrapper converts); for nd_item stages
  the launch rank is the body's declared rank.
- **Individual launches** (all Components outside a FusedSystem, whether
  under plain Systems or ParallelSystems): the Component uploads its OWN
  single param block at its own turn — Component-local, no shared arena —
  and the launch uses `cfg.define(&stage_rR, wrapper_KI)` with the
  component's registered geometry (default or overridden).
- **FusedSystem launches** are the ONLY multi-block case. Rank and
  work-group shape are specified AT SYSTEM LEVEL; component wg-shape
  overrides are ignored inside a fuse. Rank agreement: an nd_item stage
  whose rank differs from the fuse's rank is a hard error at fuse
  assembly; id-only stages are rank-exempt. The wave's blocks are packed
  into one arena in execution order, once per wave (legal because every
  stage's extents are already resolved — see the dynamic-shape rules),
  and the launch uses `cfg.define_as_call_sequence(&stage_rR, {w1..wN})`.
  The single call site expands inline at JIT time; every expanded call
  receives the same `(cur, it)`; cursor consumption assigns blocks
  positionally. Per-element cost is N slice guards plus only the work
  that survives them — part of FusedSystem's documented cost model
  (spec 05).
- **Slices and extents are counted in whole work-groups**: per-stage
  padding rounds the logical extent up to groups of the launch geometry;
  individual = the stage's own group count; fused-sequential = max stage
  group count; fused-parallel = Σ stage group counts, with group_offset
  the prefix sum. A group never straddles components, under any geometry.
- **One dynamic_function_config per launch** (acpp law). The System
  caches configs keyed by the ordered dynamic_function_id sequence; acpp
  caches the JIT result per configuration, persistently across runs.

## Dynamic shapes and the pipeline

- **Resolution is just-in-time.** A Component resolves shapes in its
  pre-forward codepath at its own turn in depth-first order — upstream
  has completed by construction. Static models pay nothing anywhere in
  this section.
- **FromInput reads never flush the queue.** Each DimRecord carries a
  tiny always-valid host destination (`dim_scratch`, 01.B). The PRODUCING
  Component, right after enqueueing its kernel, enqueues the
  device-to-host copy into the consumer's dim_scratch (the link is known
  at finalize from the metadata chains). The consumer applies the value
  to its extents at sizing time after the copy's event completes — by
  which point it almost always already has. The wait, when it happens at
  all, is an event wait, never a queue drain.
- **FusedSystem: no intra-fuse shape dependencies — hard error.** A
  wave's geometry needs every stage's extent before the wave launches; a
  stage whose shape derives, through any determiner-or-connection chain,
  from data computed by another stage of the same fuse is an
  impossibility, not a trust question. Detected at fuse assembly from
  metadata; the error names the crossing chain. USERS SPLIT THEIR
  FUSEDSYSTEMS AT DYNAMIC-SHAPE BOUNDARIES.
- **ParallelSystem: no cross-sibling shape dependencies — hard error.** A
  child's shape reading a sibling's output (by value or shape) is a data
  dependency, contradicting the independence the user asserted by
  wrapping in ParallelSystem. Detected from the same metadata chains.
  General data-independence of parallel children remains the user's
  responsibility, unchanged.
