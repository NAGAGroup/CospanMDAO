# CoSpan roadmap

Messy by design. Checkboxes, notes, strike-throughs welcome. Each item points at the spec revision that will absorb it and the date of the project-knowledge notes it came from (2026-09-28 unless stated).

## Before any code (spec drafts still needed)

- [x] Chunked systems — a system flavour with a scratch budget; nesting and sub-chunking are required; the chunk axis and the sub-chunk declaration are open. (absorbed by: r02)
- [ ] Cycles, solvers and implicit components — cycles by registration order; an implicit component is a self-connection with its own buffer; a balance helper. (absorbed by: r02)
- [ ] A "proven by" list per r01 section — the order-sensitive tests each contract needs. (absorbed by: r02)

## Build order (forward-complete first)

1. [ ] SlabKit on the fork — repin, six gap tests, spec/README refresh.
2. [x] CoSpan skeleton, including chunked systems — forward passes for every non-cyclic pattern.
3. [ ] Forward time stepping (MLS-MPM).
4. [ ] Cycles, solvers, implicit components — Sellar converges.
5. [ ] Implicit time stepping (FE heat conduction; steady state matches the solver).
6. [ ] AD partials with a compute_partials override; total derivatives redesigned.
7. [ ] SLSQP, then IPOPT; constraints and objectives.

## Phase 1 follow-ups (low-hanging, agreed 2026-09-29)

Principle: fusion is always the user's explicit decision. The framework never guesses heuristically when to fuse. What it offers is abstractions and guarantees that make fusing well possible. When fusion pays off, and how to profile it, belongs in the docs.

- [ ] Restrict pointers: pass component input/output pointers as `AdaptiveCpp_restrict_ptr` (ACPP_EXT_RESTRICT_PTR, generic JIT only). The framework owns the storage, so it knows the pointers don't alias. Open question: user slices that overlap inside one component.
- [ ] Ticket-ordered groups for single-pass cross-group patterns (decoupled lookback). An atomic ticket gives each group its id in start order, so a group may wait on lower ids without co-residency. AdaptiveCpp's own scan does this: `include/hipSYCL/algorithms/scan/decoupled_lookback_scan.hpp` in the fork. It is a framework guarantee that users build single-pass scans, compaction and running sums on. Chosen over persistent threads with a grid barrier.
- [ ] JIT-constant stage records via `sycl::specialized` (ACPP_EXT_SPECIALIZED). Ranges and offsets become constants, so the per-work-item index division disappears. The records live in USM today and would need to be passed by value; each new shape set is a new cached kernel. Also try ACPP_ADAPTIVITY_LEVEL=2, which hardwires invariant kernel arguments.
- Not pursued: scoped parallelism. It is moot in the generic flow.

## Examples

- [ ] Sellar.
- [ ] LRPIM module + nonlocal gradient over it.
- [ ] MLS-MPM.
- [ ] FE heat conduction (a client demonstration, not only a test).
- [x] GEMVER→BiCGK→GEMVER chain (examples/gemver-chain)

## User documentation (tutorial-style; before cycles and optimizers)

Topics learned while building the light-transport forcing case (2026-09-30). Each becomes a tutorial section with a worked example.

- [ ] The functional model: outputs are computed fresh and are never initialized; no global scratch; multi-step work is several components, producer to consumer.
- [ ] Accumulation patterns: group work with local memory inside one kernel; or a zeroing producer component plus the accumulators, joined with tie_outputs (atomic by default, allow_plain when writes are disjoint).
- [ ] Atomics are relaxed; use sycl::atomic_fence(acq_rel, device) where ordering matters (e.g. bottom-up tree builds).
- [ ] Namespaces, promotions and ties: subsystems get their own namespaces; Systems declare no I/O of their own; promote exposes one entry as a public alias (optional, only for what matters outside); ties are how inputs or outputs share a root; connect makes data links; names at one level are unique.
- [ ] Registration order is execution order: add children in dataflow order.
- [ ] Value-driven sizes: a size computed by an earlier sibling in the same forward drives a later sibling's shape (count → scan → fill); not inside Fused or Parallel systems.
- [ ] Run policy as a property of add_subsystem: eternal (the default), run_n, do_once, or transient (it exists for one forward of its parent: built at the start, deleted at the end). A variable lives while something that uses it exists. rearm(name) on the parent rebuilds a child.
- [ ] Rearm: the parent rebuilds a child by name; its wiring is replayed.
- [ ] Initialization as metadata: init_input on free inputs, the highest level wins, ignored when the input is connected, zero by default; applied only when the variable is materialized; Model::init_input applies at the next run.
- [ ] Cycles: expressed by registration order; they warn at setup until solvers exist.
- [ ] Packaging a CoSpan consumer with pixi: depend on cospanmdao as a host AND run dependency (SlabKit's shared library comes with it).

## Later, wanted

- [ ] Parallel Systems (beyond Components).
- [ ] Serialization.
- [ ] A derivative-aware paragraph per forward feature.

## Housekeeping

- [x] CoSpan pixi manifest repinned to the naga toolchain (full previous manifest at dbebc3a:pixi.toml).
- [ ] README.md rewrite (it still describes the first-pass framework).
- [x] Tests from the first code commit.
- [ ] Clean up tests/smoke and tests/spike: standalone, unbuilt, and stale after D138 (they use init() hooks, initializers, EntryAccess/RootDriver). Fold anything still worth checking into the Catch2 suite, then remove them (Jack decides at cleanup time).
- [ ] Internal naming: the code still calls a variable's storage record 'root' or 'shadow' (IOEntry::root_, is_root). Rename to variable terms in a dedicated pass.

## After the 2026-10-01 checkpoint (D140 rework)

Found while porting LRPIM onto opaque chunked systems (C2b to C2e) and measuring it. Notes dated 2026-10-01.

- [x] Hold internal roots across forwards when the topology is unchanged. Why: removes the one release wait per chunked forward. Done: chunked internal variables are kept across chunks and forwards while the chunked system exists; D141 is superseded by D143.
- [x] D135's one stream wait per releasing component in plain systems. Why: pipeline stalls outside chunked systems. Superseded by D143: frees are ordered by events registered on the storage (SlabKit 572b581); no queue waits.
- [x] Registration-order diagnostics that name the offending registration. Done (D143): shape errors are reactive and name the needer, the dimension and what it waited on; a connection that forms a cycle warns at setup.
- [x] Audit of alias kinds and declaration forms for overlap. Done: D142 (FromSourceDim, cs::tied and alias::from_connection collapse into unspecified dims), and variables and ports under D143. Why: several forms now express nearly the same sharing; one should be canonical.
- [ ] Ragged ports in FusedSystem and ParallelSystem. Why: ragged data is limited to plain systems and chunked systems.
- [ ] Device-side scan for extents known only on the device. Why: value-driven sizes currently need a host read.
- [ ] SlabKit `from_host` without the `async_host` keep-alive task (a general fix). Why: host tasks fence the stream.
- [ ] Per-backend launch configuration (e.g. work-group size). Why: group bodies with many barriers are fast on CUDA but about 5x slower than a one-item-per-group body on the omp backend (LRPIM measurement).
- [ ] Sparse bundles over ragged ports (CSR, block CSR, SELL); memoryless functional ports; restrict_ptr (see Phase 1 follow-ups). Why: sparse data is what the ragged ports are used for, and each bundle would carry its own invariants.
- [ ] Investigate: wave kernels JIT only in the translation unit that defines the stages. Why: a FusedSystem of kernels defined in another TU aborts with "Could not find function call specialization target".
- [ ] Housekeeping: tests/smoke is stale against the D139/D140 APIs (HostViewMut semantics changed in C2c). Why: it is unbuilt and no longer matches the API.
- [ ] Housekeeping: a clean gemver timing rerun, not concurrent with other GPU work. Why: earlier timings may have been disturbed by other GPU jobs.

## Orchestration redesign (rough shape; specifics settled item by item)

CoSpan's differentiator is orchestration on top of AdaptiveCpp's JIT and fusion. This phase makes CoSpan model anything, fast and accurate, with good ergonomics, and everything after builds on the principles settled here.

- [ ] SlabKit queue: one queue, one mental model. In-order items (single kernels, single host items, parallel task graphs) form the backbone. A parallel task graph's members run concurrently and may themselves be task graphs of sequential and parallel steps (nesting). Why: users shouldn't manage several streams with different ordering guarantees.
- [ ] SlabKit out-of-order lane: async_host is fully asynchronous again (unordered except by the events passed in; async_host_in_order is the ordered form), and the lane also takes device (non-host) work. In-order items accept extra event dependencies, and the global wait covers the lane. Why: the primitive for the push hooks below.
- [ ] SlabKit: remove the async_host keep-alive task in from_host. Why: it puts a host callback in the stream only to keep a buffer alive.
- [x] usm_vector operations (assignment, move, copy, resize, from_host) accept dependent events. Done: SlabKit 572b581 (register_events and event-taking mutations).
- [ ] Push pattern: consumer-side work is registered on a root at setup (cross-device copies for connections, initializers, a device-side scan for chunked topology) and is submitted from the producer's event as soon as the producer is submitted. Why: that work finishes long before the consumer is reached.
- [ ] Remove ParallelSystem: independent sibling subsystems (including whole Systems) are detected and submitted as a parallel task graph. FusedSystem stays as the one special flavor (components only, sequential and parallel waves). Why: concurrency follows from the dependency graph, not from a user-chosen flavor.
- [x] Lifetime is scope: shared ownership of variables (a variable lives while some port that uses it exists); parents own their children, delete them when their run policy ends, and rebuild them by name (D143).
- [ ] Visibility for those trade-offs: a user-facing per-forward report of allocations, frees and waits per subsystem (from memory_report and the diag counters). Why: the modeler can only choose scopes from a profile.
- [ ] Design guidance to document: prefer shapes inherited from connections, keep connected components on one device, and treat repeatedly resizing value-driven shapes as a cost. Why: these avoid reallocation churn.
- [ ] Ergonomics pass over model building (declaration forms, aliases, task-graph construction). Why: the demo has to look as good to write as it runs.
- [ ] Demo target: meshless LBM fluid (LRPIM) with moving mesh boundaries at large node counts. Why: it shows the framework and the method together.
