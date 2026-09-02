# Spec 01.C — kernel machinery

Part of the phase-1 spec (`docs/spec/01-components-io/`). Binding contract;
declarations only. Wrappers: 01.A. Metadata/shapes: 01.B. Component: 01.D.
System-side execution: 01.E.

## The map taxonomy is doctrine, not metadata

Bijective, injective, surjective: the completeness argument — each class
being implementable guarantees Components can express any function — and
the roadmap for provided kernels (reductions, scatters, scans, sorts).
Kernels are NEVER classified by their authors; no class field exists.
The general case is guaranteed by construction: indexed access plus atomic
dtypes (01.A), available in every kernel body.

## 1. The KernelBody contract

```cpp
namespace naga::cospan {

/// A kernel is a TYPE with nested argument structs and one call operator:
///
///   struct MyBody {                      // POD config lives as members
///       struct Inputs  { Input<...>  a; Input<...> b; };
///       struct Outputs { Output<...> y; };
///       struct Scratch { Local<...>  tile; };     // OPTIONAL (local mem)
///       void operator()(sycl::id<R> idx, Inputs, Outputs) const;
///          // or (sycl::nd_item<R>, Inputs, Outputs)
///          // or (sycl::nd_item<R>, Inputs, Outputs, Scratch)
///          // or (CustomOp&, Inputs, Outputs)        — CUSTOM OPERATION
///   };
///
/// CUSTOM-OPERATION BODIES are never compiled to a device function or
/// launched: the owner submits them via AdaptiveCpp_enqueue_custom_
/// operation on the SAME queue, with the same dependencies, retarget
/// and lane — so ordering, parallel lanes and gating apply unchanged.
/// Inputs/Outputs arrive as the bare wrappers with CURRENT device
/// pointers. What the body does is open: host work, or handing the
/// pointers to a native library on the native stream (CustomOp::h is
/// the interop handle; CustomOp::to_host/from_host move data through
/// the native backend for pure host work). Not fusable (no device
/// function exists). Phase 7: opaque to the engine — analytical adjoint
/// or FD/CS required when derivatives are requested.
///
/// THE SIGNATURE IS THE DECLARATION, four ways:
///   * first parameter id<R> vs nd_item<R>  => group semantics opt-in
///     (a body without the nd_item cannot touch group abstractions);
///   * R                                    => kernel rank, deduced
///     (kernel_rank_v<KB>; no static member);
///   * trailing Scratch parameter           => local-memory opt-in
///     (nd_item bodies only — local memory without a group is
///     meaningless).
/// Inputs, Outputs, KB itself, and Scratch (if present) SHALL be
/// trivially copyable — concept-enforced; they are copied byte-wise
/// into the launch's device streams.
/// REQUIREMENT (phase-8 keystone): KB SHALL be a template over its value
/// mode, using wrapper `value_type` exclusively in the body. Opt-out:
/// components supplying an analytical adjoint.
template <class KB> concept PlainBody   = /* operator()(id<R>, In, Out) */;
template <class KB> concept GroupBody   = /* operator()(nd_item<R>, In, Out [, Scratch]) */;
template <class KB> concept HasScratch  = /* trailing Scratch parameter */;
template <class KB> concept CustomOpBody= /* operator()(CustomOp&, In, Out) */;
template <class KB> concept KernelBodyC =
    std::is_trivially_copyable_v<KB> &&
    std::is_trivially_copyable_v<typename KB::Inputs> &&
    std::is_trivially_copyable_v<typename KB::Outputs> &&
    (int(PlainBody<KB>) + int(GroupBody<KB>) + int(CustomOpBody<KB>) == 1);
template <class KB> inline constexpr int kernel_rank_v = /* from signature */;

## 2. Local<DT, Rank> — local memory with wrapper ergonomics

/// Same accessor discipline as Output (bare references; plain dtypes
/// only in v1), but the storage is GROUP-LOCAL and the wrapper carries a
/// byte OFFSET instead of a pointer — local memory has no host address.
/// The user shapes it host-side in compute_forward() like any other
/// launch parameter:
///     MyBody::Scratch s;  s.tile.reshape({17, 16});
/// create_launch_config() lays out all Locals in a Scratch (aligned,
/// summed -> the stage's scratch_bytes); the device function fixes up
/// each Local's pointer as local_base + offset before the body runs.
/// Per-group instance semantics are local memory's own; the framework
/// adds nothing.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
struct Local {
    using value_type = cpp_type_t<DT>;

    void reshape(sycl::range<Rank> shape);        // host, compute_forward
    value_type& operator[](sycl::id<Rank> idx) const;  // device, post-fixup
    value_type& operator[](size_t flat) const;
    sycl::range<Rank> shape() const;
    size_t size() const;

private:
    sycl::range<Rank> shape_;
    size_t            offset_;   // into the launch's local allocation
    value_type*       ptr_;      // null until device-side fixup
    friend struct detail::ScratchLayout;
};

## 3. StageRecord and FnOffsetInfo

/// One StageRecord per stage per launch, fixed-size, packed into a
/// device array in EXECUTION ORDER. Field ownership:
///   component-bound (create_launch_config): rank, range, flat_extent,
///     stream offsets, scratch_off/scratch_bytes;
///   System-stamped (launch pack): group_offset, group_count — the
///     GROUP-INDEX slice: slices are counted in whole work-groups, so a
///     group never straddles stages under ANY geometry.
struct StageRecord {
    sycl::range<3> range;         // logical iteration space, 1-padded
    size_t   flat_extent;         // product of range
    size_t   group_offset;        // System-stamped
    size_t   group_count;         // System-stamped
    uint32_t input_off;           // into the inputs stream
    uint32_t output_off;          // into the outputs stream
    uint32_t body_off;            // into the bodies stream
    uint32_t scratch_off;         // into the bodies stream (Scratch bytes)
    uint32_t scratch_bytes;       // this stage's local-memory requirement
    uint8_t  rank;
};

/// FnOffsetInfo — position-as-identity for the chained executor. Every
/// definition in an acpp call sequence receives IDENTICAL arguments (one
/// call site, expanded), so a stage can only know WHICH stage it is by
/// advancing shared mutable state in call order. This is that state.
struct FnOffsetInfo {
    const StageRecord* records;   // device array, execution order
    uint32_t           i;         // per-work-item cursor, starts at 0
    const StageRecord& next() { return records[i++]; }
};

## 4. The generated device function — one per (KernelBody type, rank)

/// compile_kernel() (01.D) emits, for KB at launch rank R:
SYCL_EXTERNAL void kb_fn(sycl::nd_item<R> it, FnOffsetInfo& info,
                         const std::byte* inputs, std::byte* outputs,
                         const std::byte* bodies, std::byte* local_base)
{
    // LAW: THE CURSOR ADVANCE IS THE FIRST STATEMENT AND UNCONDITIONAL.
    // Every stage function consumes exactly one record before any
    // possible return; a masked-out stage still advances. Violating
    // this desynchronizes every stage after it.
    const StageRecord& r = info.next();

    // Group-index slice check — group-uniform by construction: whole
    // groups skip together; SYCL group barriers are group-scoped, so
    // skipped groups violate nothing (verified: SYCL 2020 §4.17.3 and
    // the execution model's weakly-parallel work-item guarantee).
    const size_t glid = it.get_group_linear_id();
    if (glid < r.group_offset || glid - r.group_offset >= r.group_count)
        return;

    // Stage-local flat index — contiguous per stage under any geometry:
    const size_t flat = (glid - r.group_offset) * it.get_local_range().size()
                      + it.get_local_linear_id();

    // id-only bodies: tail mask HERE. The body sees only its logical
    // range: ZERO guards in user code, fusion invisible (the opacity
    // contract — kernels are never written differently for fusion).
    if constexpr (PlainBody<KB>)
        if (flat >= r.flat_extent) return;
    // nd_item bodies: receive the full padded slice; ragged-tail
    // handling is the standard obligation of any group-using nd_range
    // kernel. Their freedom is bounded by SYCL 2020 §4.17.3, verbatim:
    // group functions "must be encountered in converged control flow by
    // all work-items in the group ... under the same set of conditions"
    // — skipping one via early exit, or calling it from divergent
    // branches, is UB. Divergence is legal in code free of group
    // functions. Guard the loads, never the participation. We spec to
    // the spec, not to hardware tolerance.

    const auto idx = detail::delinearize<kernel_rank_v<KB>>(flat, r.range);
    auto in   = *reinterpret_cast<const typename KB::Inputs*>(inputs  + r.input_off);
    auto out  = *reinterpret_cast<const typename KB::Outputs*>(outputs + r.output_off);
    auto body = *reinterpret_cast<const KB*>(bodies + r.body_off);

    if constexpr (HasScratch<KB>) {
        auto s = *reinterpret_cast<const typename KB::Scratch*>(bodies + r.scratch_off);
        detail::fixup_scratch(s, local_base);  // Local.ptr_ = base + offset_
        body(it, in, out, s);
    } else if constexpr (GroupBody<KB>) {
        body(it, in, out);
    } else {
        body(idx, in, out);
    }
}

## 5. The chained executor

/// Placeholders — one per launch rank; declarations only, JIT replaces.
/// Three families cost nothing at rest: SSCP lowers a codepath on first
/// invocation only.
void kb_entry_r1(sycl::nd_item<1>, FnOffsetInfo&, const std::byte*,
                 std::byte*, const std::byte*, std::byte*);
void kb_entry_r2(sycl::nd_item<2>, /* same trailing params */);
void kb_entry_r3(sycl::nd_item<3>, /* same trailing params */);

/// The outer kernel, one per rank — THE single call site:
q.submit([&](sycl::handler& cgh){
    sycl::local_accessor<std::byte, 1> local{launch_scratch_bytes, cgh};
                                 // max over stages; zero when unused
    cgh.parallel_for(sycl::nd_range<R>{padded_global, wg},
        cfg.apply([=](sycl::nd_item<R> it){
            FnOffsetInfo info{records_dev, 0};      // fresh per work-item
            kb_entry_rR(it, info, inputs_dev, outputs_dev, bodies_dev,
                        local.get_pointer().get());
        }));
});

/// MECHANISM. cfg is built ONCE, at setup — there is NO runtime config
/// cache: an individual component's config (define(entry_id, fn_id))
/// never changes, and a FusedSystem's sequence config is built at ITS
/// setup because fuse membership is static.
///   individual:  cfg.define(kb_entry_rR_id, kb_fn_id);
///   fused wave:  cfg.define_as_call_sequence(kb_entry_rR_id,
///                    {fn_A_id, fn_B_id, barrier_id, fn_C_id, ...});
/// At JIT time the ONE kb_entry_rR call expands, inline and in order:
///     kb_fn_A(it, info, inputs, outputs, bodies, local);
///     kb_fn_B(it, info, inputs, outputs, bodies, local);
///     ...
/// Every expanded call receives the SAME arguments — acpp's sequence
/// semantics. `info` is a mutable reference to the work-item's private
/// cursor: A's unconditional first-statement next() returns records[0]
/// and leaves i==1; B's returns records[1]; and so on.
///     CALL ORDER = SEQUENCE ORDER = RECORD ORDER = PACK ORDER
/// — one identity, maintained by the System's packer writing records in
/// execution order and by the first-statement law. No stage ids exist;
/// the same KernelBody type twice in one wave reads two different
/// records correctly by position alone.
///
/// Barrier stages: when any stage of a SEQUENTIAL-mode wave declares
/// Scratch, the System interleaves the provided
///     void kb_barrier(...same signature...) {
///         sycl::group_barrier(it.get_group());   // does NOT call next()
///     }
/// between consecutive scratch-using stages: the call site is
/// unconditional, so every work-item reaches it in converged control
/// flow (legal), and stage N+1 cannot touch local bytes stage N still
/// reads (necessary). kb_barrier consumes NO record — the one deliberate
/// exception to one-call-one-record; the packer writes records only for
/// real stages. Parallel-mode waves need no barriers: each group
/// executes exactly one stage's body, so scratch never crosses stages.

} // namespace naga::cospan
```

## 6. Geometry and slices

- All launches are typed by rank (the executor's switch); the default
  geometry is flat: `{{padded_flat}, {default_wg}}` at rank 1, with
  rank-appropriate defaults at ranks 2/3 (implementation-tunable, not
  contract constants). The wg overloads of create_launch_config (01.D)
  override per launch; FUSED rank and wg are set at SYSTEM level,
  component wg intent ignored inside a fuse.
- Rank agreement in a fuse: an nd_item stage whose rank differs from the
  wave's rank is a hard error at fuse assembly; id-only stages are
  rank-exempt (their device function is generated lazily for the wave's
  rank; SSCP lowers only what is invoked).
- Padding and extents are counted in WHOLE WORK-GROUPS: individual = the
  stage's own group count; fused-sequential = max stage group count;
  fused-parallel = Σ stage group counts with group_offset the prefix
  sum. A group never straddles stages, under any geometry — this is
  what makes the slice check group-uniform and collectives safe.
- One dynamic_function_config per launch; the config object outlives
  kernel completion; acpp JIT-caches per configuration, persistently
  across runs (acpp laws).

## 7. Dynamic shapes and the pipeline

- **Resolution is just-in-time.** A Component resolves shapes in
  run_pre_forward at its own turn in depth-first order — upstream has
  completed by construction. Static models pay nothing anywhere in this
  section.
- **FromInput resizes are the consumer's own business.** In its
  `run_pre_compute`, the consuming Component enqueues the D2H of the
  driving input into the entry's `dim_scratch`, then a custom operation
  (depending on that event) that applies the resize on the host —
  reallocating through the OS route from inside the custom-op body if
  grown — and waits on that custom op's event: the one forced sync, so
  the launch geometry is known at enqueue. Nothing is registered
  anywhere; no producer knows its consumers.
- **FusedSystem: no intra-fuse shape dependencies — hard error.** A
  wave's geometry needs every stage's extent before the wave launches; a
  stage whose shape derives, through any determiner-or-connection chain,
  from data computed by another stage of the same fuse is an
  impossibility, not a trust question. Detected at fuse assembly from
  metadata; the error names the crossing chain. USERS SPLIT THEIR
  FUSEDSYSTEMS AT DYNAMIC-SHAPE BOUNDARIES.
- **ParallelSystem: no cross-sibling shape dependencies — hard error.**
  A child's shape reading a sibling's output (by value or shape) is a
  data dependency, contradicting the independence the user asserted by
  wrapping in ParallelSystem. Detected from the same metadata chains.
  General data-independence of parallel children remains the user's
  responsibility, unchanged.
