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
///   * static constexpr int kernel_rank;   // iteration-space rank
///   * exactly ONE body, in exactly one of two signatures:
///       void operator()(sycl::id<kernel_rank>) const;
///       void operator()(sycl::nd_item<1>, sycl::id<kernel_rank>) const;
///     THE SIGNATURE IS THE DECLARATION: a body without the nd_item
///     cannot touch group abstractions (it has no group object), so the
///     framework derives the masking discipline structurally — no flags,
///     no classification. Bodies wanting group collectives take the
///     nd_item form and use group abstractions as they wish.
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
///   System-stamped (written at launch preparation): slice_offset,
///     padded_extent. For an individual launch the stamp is the
///     degenerate {0, padded_launch}; sequential-mode fusion stamps
///     {0, launch_extent} for every stage; parallel-mode stamps true
///     padded slices. Components never know their slice.
template <KernelImplC KI>
struct KernelImplWrapper {
    KI                            impl;
    sycl::range<KI::kernel_rank>  range;         ///< logical iteration space
    size_t                        flat_extent;   ///< product of range
    size_t                        slice_offset;  ///< System-stamped
    size_t                        padded_extent; ///< System-stamped

    /// Masking. THE INVARIANT IS TWO LINES:
    ///   1. Slices are work-group multiples — a group never straddles
    ///      components. The wrapper's slice check
    ///      (slice_offset <= g < slice_offset + padded_extent) is
    ///      therefore GROUP-UNIFORM: whole groups enter or skip together.
    ///      Group barriers are group-scoped in SYCL — groups owe each
    ///      other nothing — so nothing further exists to check at
    ///      orchestration level.
    ///   2. Each KernelImpl is correct from its own scoped view, exactly
    ///      as if launched standalone.
    /// KERNELS ARE NEVER WRITTEN DIFFERENTLY FOR FUSION. Concretely:
    ///   * id-only bodies: the wrapper ALSO masks the padding tail
    ///     (return when local >= flat_extent) — legal because the
    ///     signature proves the body group-free. Such bodies contain
    ///     ZERO guards and see exactly their logical range.
    ///   * nd_item bodies: receive their full padded slice, tail
    ///     included — the wrapper cannot see whether the body barriers,
    ///     so it must not exit items on the body's behalf. The body's
    ///     own freedom is full: divergence is always legal; early
    ///     return is legal unless it would skip a group barrier or
    ///     collective that groupmates execute (the one UB, per the SYCL
    ///     spec — we spec to the spec, not to hardware tolerance).
    ///     Barrier-free nd_item bodies may early-return their tail;
    ///     barrier-bearing bodies guard the loads, never the
    ///     participation. Handling ragged n % wg is the standard
    ///     obligation of any group-using nd_range kernel in any
    ///     framework, identical fused or standalone.
    /// Localization: local = g - slice_offset; id<kernel_rank> by
    /// row-major divmod against range. Bodies never see flattening,
    /// slices, or fusion.
    void operator()(sycl::nd_item<1> it) const;
};

/// The placeholder — THE single dynamic-function call site in the entire
/// framework. Declaration only; definitions replace it at JIT time.
void stage(ParamCursor& cur, sycl::nd_item<1> it);

/// Per-KernelImpl-TYPE generated wrapper (SYCL_EXTERNAL free function):
///   { auto& blk = cur.next<KernelImplWrapper<KI>>(); blk(it); }
/// createKernelWrapper<KI>() emits it and returns the registration
/// record; called from the component's setup (01.D). The type-erased
/// dynamic_function_id is stored centrally by the owning System.
struct KernelRegistration {
    sycl::AdaptiveCpp_jit::dynamic_function_id id;
    size_t block_size;
    size_t block_align;
};
template <KernelImplC KI> KernelRegistration createKernelWrapper();

} // namespace naga::cospan
```

## The pipeline

ONE outer kernel in the whole framework, nd_range<1>:

```cpp
q.parallel_for(sycl::nd_range<1>{padded_launch, wg},
    cfg.apply([=](sycl::nd_item<1> it){
        ParamCursor cur{params_base};
        stage(cur, it);
    }));
```

- **Individual launches** (all Components outside a FusedSystem, whether
  under plain Systems or ParallelSystems): the Component uploads its OWN
  single param block at its own turn — Component-local, no shared arena,
  no packing machinery — and the launch uses
  `cfg.define(&stage, wrapper_KI)`.
- **FusedSystem launches** are the ONLY multi-block case: the wave's
  blocks are packed into one arena in execution order, once per wave
  (legal because every stage's extents are already resolved — see the
  dynamic-shape rules), and the launch uses
  `cfg.define_as_call_sequence(&stage, {w1..wN})`. The single call site
  expands inline at JIT time; every expanded call receives the same
  `(cur, it)`; cursor consumption assigns blocks positionally.
  Per-element cost is N slice guards plus only the work that survives
  them — part of FusedSystem's documented cost model (spec 05).
- **Launch extents**: individual = the stage's padded extent;
  fused-sequential = max stage padded extent; fused-parallel = Σ padded
  extents. All padding is to work-group multiples: whole-group slice
  membership is the collective-safety guarantee.
- **Work-group size** is System-owned per launch; stages are
  WG-size-agnostic by contract; group-using bodies adapt to any size.
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
