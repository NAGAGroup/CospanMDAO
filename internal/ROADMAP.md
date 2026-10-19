# CoSpan roadmap

Messy by design. Checkboxes, notes, strike-throughs welcome. Each item points at the spec revision that will absorb it and the date of the project-knowledge notes it came from (2026-09-28 unless stated).

## Status and next

- [x] **The 2026-10-03 rework is done**: branch `rework-2026-10-03`, last commit before the docs commit `2e8e223` (default and cuda suites green; spec `internal/spec/revisions/r02-2026-10-03-the-design-as-built.md`, decision D144, migration `internal/rework-migration-2026-10-03.md`).
- [x] **Design review section 11 implemented on main**; spec r03 (`internal/spec/revisions/r03-2026-10-06-section-11-as-built.md`).
- [ ] **The chunked-systems redesign is deferred until after the public release** (decided with Jack, 2026-10-11). Chunked systems, slices and leading ranges were not ported; the old code is dead code in `include/naga/cospan/chunked_system.hpp` and in git history (`c8553f6`).
- [ ] **Path to a public release (agreed 2026-10-11)**
  1. [ ] Documentation infrastructure, then documentation and tutorials. CoSpan gets polished along the way as rough spots show up. Tutorials target the default device.
  2. [ ] Lay of the land for the web presence. This is discussion only, with nothing built: what goes on the NAGA Labs site, whether CoSpan lives there or at cospan.dev, pointing Jack's personal site at NAGA Labs, and the LinkedIn update.
  3. [ ] SlabKit polish, plus any CoSpan polish the tutorials didn't cover.
  4. [ ] Toolchain and OpenMP work: the fork's OpenMP dependency-ordering bug (must be fixed before anything is published), the gemver onepass OpenMP bug, and the toolchain CI failures. Also: exceptions thrown by host tasks never reach the queue's async handler. The AdaptiveCpp fork doesn't route them there, which is why SlabKit catches them itself (HostErrors) and rethrows from wait_and_throw; the fix spans the fork and SlabKit, specifics later (2026-10-11).
  5. [ ] CI/CD for SlabKit and CoSpan.
  6. [ ] Building the site, publishing and promoting. Not defined yet.
  7. [ ] C++20 audit: use concepts and constraints instead of static_assert throughout.

Open items after the rework:

- [ ] gemver-chain `onepass` gives a wrong `q` on the OpenMP backend (pre-existing; `--variant all` skips it there, `--variant onepass` shows it; correct on CUDA). The `naive` gemver variant also fails on OpenMP because it contains `BicgkOnePass`; `--variant all` skips it there too.
- [ ] A user-docs example of a host view kept as a component member to size outputs from values (design review section 10: there is no framework helper for it). Note: `view_input` now exists (P6). Stays open as a docs item.
- [ ] Performance passes: the device scan (`device_scan.hpp`, behind `counts_from` and `offsets_from`) and ~~the staging of fused components (packing buffers)~~ (the packing buffers were removed in P7a).
- [x] Done in P8/P9; none remain. The `REVIEW` markers left in the tests, to be resolved or removed: `tests/test_component.cpp`, `test_epochs.cpp`, `test_flavors.cpp`, `test_model.cpp`, `test_ragged.cpp`, `test_ragged_ties.cpp`, `test_rework.cpp`, `test_shapes.cpp`, `test_system.cpp` (and two in `examples/gemver-chain`: `main.cpp`, `systems.hpp`).

## Before any code (spec drafts still needed)

- [ ] Chunked systems — a system flavour with a scratch budget; nesting and sub-chunking are required; the chunk axis and the sub-chunk declaration are open. Not ported in the 2026-10-03 rework: redesign deferred until after the public release (see Status and next).
- [ ] Cycles, solvers and implicit components — cycles by registration order; an implicit component is a self-connection with its own buffer; a balance helper. (absorbed by: r02)
- [ ] A "proven by" list per r01 section — the order-sensitive tests each contract needs. (absorbed by: r02)

## Build order (forward-complete first)

1. [ ] SlabKit on the fork — repin, six gap tests, spec/README refresh.
2. [x] CoSpan skeleton — forward passes for every non-cyclic pattern. (~~including chunked systems~~: not ported by the 2026-10-03 rework.)
3. [ ] Forward time stepping (MLS-MPM).
4. [ ] Cycles, solvers, implicit components — Sellar converges.
5. [ ] Implicit time stepping (FE heat conduction; steady state matches the solver).
6. [ ] AD partials with a compute_partials override; total derivatives redesigned.
7. [ ] SLSQP, then IPOPT; constraints and objectives.

## Phase 1 follow-ups (low-hanging, agreed 2026-09-29)

Principle: fusion is always the user's explicit decision. The framework never guesses heuristically when to fuse. What it offers is abstractions and guarantees that make fusing well possible. When fusion pays off, and how to profile it, belongs in the docs.

- [ ] Restrict pointers: pass component input/output pointers as `AdaptiveCpp_restrict_ptr` (ACPP_EXT_RESTRICT_PTR, generic JIT only). The framework owns the storage, so it knows the pointers don't alias. Open question: user slices that overlap inside one component.
- [ ] Single-pass cross-group patterns (decoupled lookback) on the `cs::group_waits_enabled` fused-group property. An atomic ticket gives each group its id in start order, so a group may wait on lower ids without co-residency; the property and `wait_group` are built, what is left is the library-level patterns (scan, compaction, running sums) on top. AdaptiveCpp's own scan does this: `include/hipSYCL/algorithms/scan/decoupled_lookback_scan.hpp` in the fork. It is a framework guarantee that users build single-pass scans, compaction and running sums on. Chosen over persistent threads with a grid barrier.
- [ ] JIT-constant stage records via `sycl::specialized` (ACPP_EXT_SPECIALIZED). Ranges and offsets become constants, so the per-work-item index division disappears. The records live in USM today and would need to be passed by value; each new shape set is a new cached kernel. Also try ACPP_ADAPTIVITY_LEVEL=2, which hardwires invariant kernel arguments.
- Not pursued: scoped parallelism. It is moot in the generic flow.

## Examples

- [ ] Sellar.
- [ ] LRPIM module + nonlocal gradient over it.
- [ ] MLS-MPM.
- [ ] FE heat conduction (a client demonstration, not only a test).
- [x] GEMVER→BiCGK→GEMVER chain (examples/gemver-chain)
- [x] Fusion benchmark (done, commit "examples: ns-residual fusion benchmark (unfused / fused per range_split / hand-written); remove softmax"; 15 rows check OK on CUDA; at 4096 squared: unfused 9.85 ms, fused 5.8 to 6.0 ms by strategy, handwritten 3.63 ms) (chosen 2026-10-11; replaces softmax; becomes a tutorial example): one compressible Navier–Stokes residual evaluation on a 2D structured grid as ONE fused group of work-group stages, one group per 16×16 tile. Compared against unfused CoSpan and one hand-written SYCL kernel.
  - Segment 1 (shared local memory): load the tile plus halo, apply the boundary condition, convert to primitives, compute gradients, reconstruct with a limiter, then the convective (HLLC) and viscous fluxes on the owned faces. One flux component is reused for x and y.
  - Segment 2 (partitioned): the convective and viscous divergence stages each wait (group waits) on the segment-1 tiles they read, and write the residual R as a plain tie. Diagnostic stages (min local timestep, max wave speed, residual sums of squares) reduce per tile, then both apply atomically and write per-tile partials.
  - global_barrier.
  - Segment 3 (partitioned, range_mode own): deterministic reductions of the partials (4 groups for the norms, 1 for the min timestep, 1 for the max wave speed).
  - Segment 4: an error stage waits on exactly those segment-3 groups and reports atomic minus deterministic for each diagnostic.
- [x] Delete examples/softmax once the replacement exists. Done in the ns-residual commit.

## User documentation (tutorial-style; before cycles and optimizers)

Topics learned while building the light-transport forcing case (2026-09-30). Each becomes a tutorial section with a worked example.

- [ ] The functional model: outputs are computed fresh and are never initialized; no global scratch; multi-step work is several components, producer to consumer.
- [ ] Accumulation patterns: group work with local memory inside one kernel; or a zeroing producer component plus the accumulators, joined with tie_outputs (atomic by default, allow_plain when writes are disjoint). Standard practice: an output that has to be initialized by computation (for example zeroed before accumulation) gets that initialization in its own first segment of the fused group, a `cs::global_barrier()` follows that segment, and every segment that writes into the output comes after. The framework does not check this; it needs work-group stages.
- [ ] Atomics are relaxed; use sycl::atomic_fence(acq_rel, device) where ordering matters (e.g. bottom-up tree builds).
- [ ] Namespaces, promotions and ties: subsystems get their own namespaces; Systems declare no I/O of their own; promote exposes one entry as a public alias (optional, only for what matters outside); ties are how inputs or outputs share a root; connect makes data links; names at one level are unique.
- [ ] ~~Registration order is execution order: add children in dataflow order.~~ Execution order is data dependencies plus `depends_on` (section 11).
- [ ] Value-driven sizes: a size computed by an earlier sibling in the same forward drives a later sibling's shape (count → scan → fill) through a `shape_hook` on the later sibling's input (`from_value`, `counts_from`, `offsets_from`); ~~a hook's sources must be produced by an earlier child~~ (the source-ordering check was removed in P5: a hook runs before the first child holding its target). ~~not inside Fused or Parallel systems~~ (ParallelSystem is gone).
- [ ] Run policy as a property of add_subsystem: eternal (the default), run_n, do_once, or transient (it exists for one forward of its parent: built at the start, deleted at the end). A variable lives while something that uses it exists. rearm(name) on the parent rebuilds a child.
- [ ] Rearm: the parent rebuilds a child by name; its wiring is replayed.
- [ ] Initialization as metadata: init_input on free inputs, ~~the highest level wins~~ the last declaration wins (a parent's setup runs after its children's), ignored when the input is connected; applied when the variable is materialized or stale; ~~Model::init_input~~ Model::update_input applies at the next run. An input declared without an initializer gets an implicit zero; an explicit initializer always beats the implicit one; outputs are never initialized; a reshape of a free input re-runs its initializer.
- [ ] Cycles: `connect` marks a cycle (a consumer registered before its producer); a cycle needs an explicitly declared initializer (an error at first materialization otherwise); the warning stays, at setup, until solvers exist.
- [ ] Packaging a CoSpan consumer with pixi: depend on cospanmdao as a host AND run dependency (SlabKit's shared library comes with it).

## Later, wanted

- [x] ~~Parallel Systems (beyond Components).~~ Superseded: ParallelSystem is removed; concurrency comes from data dependencies (r02 section 3).
- [ ] Serialization.
- [ ] A derivative-aware paragraph per forward feature.
- [ ] A test mocking system. Two event tests were removed in the component redesign (2026-10-11) because they held a kernel running by spinning on a host flag: (a) destroying a model waits for kernels still using its data; (b) two readers of one variable don't wait on each other and both land on its read list. Restore both on the mocking system.
- [ ] Test coverage analysis and reporting in CI (agreed with Jack 2026-10-11). Comes with the CI/CD step of the path to a public release.

## Housekeeping

- [x] CoSpan pixi manifest repinned to the naga toolchain (full previous manifest at dbebc3a:pixi.toml).
- [x] README.md rewrite (done with the 2026-10-03 rework docs).
- [x] Tests from the first code commit.
- [ ] Clean up tests/spike (tests/smoke is deleted, P8): standalone, unbuilt, and stale after D138 (they use init() hooks, initializers, EntryAccess/RootDriver). Fold anything still worth checking into the Catch2 suite, then remove them (Jack decides at cleanup time).
- [x] ~~Internal naming: the code still calls a variable's storage record 'root' or 'shadow'.~~ Superseded: roots and shadows are gone (instances and shared metadata, D144).

## After the 2026-10-01 checkpoint (D140 rework)

Found while porting LRPIM onto opaque chunked systems (C2b to C2e) and measuring it. Notes dated 2026-10-01.

- [x] Hold internal roots across forwards when the topology is unchanged. Why: removes the one release wait per chunked forward. Done: chunked internal variables are kept across chunks and forwards while the chunked system exists; D141 is superseded by D143.
- [x] D135's one stream wait per releasing component in plain systems. Why: pipeline stalls outside chunked systems. Superseded by D143: frees are ordered by events registered on the storage (SlabKit 572b581); no queue waits.
- [x] Registration-order diagnostics that name the offending registration. Done (D143): shape errors are reactive and name the needer, the dimension and what it waited on; a connection that forms a cycle warns at setup.
- [x] Audit of alias kinds and declaration forms for overlap. Done: D142 (FromSourceDim, cs::tied and alias::from_connection collapse into unspecified dims), and variables and ports under D143. Why: several forms now express nearly the same sharing; one should be canonical.
- [ ] Ragged ports in fused components. Why: ragged data is limited to plain Systems and components (fused components refuse ragged ports). ~~ParallelSystem and chunked systems~~: removed / not ported.
- [x] Device-side scan for extents known only on the device. Done (D144): `counts_from` and `offsets_from` build the column with a parallel device scan; only the total comes back to the host.
- [ ] SlabKit `from_host` without the `async_host` keep-alive task (a general fix). Why: host tasks fence the stream.
- [ ] Per-backend launch configuration (e.g. work-group size). Why: group bodies with many barriers are fast on CUDA but about 5x slower than a one-item-per-group body on the omp backend (LRPIM measurement).
- [ ] Sparse bundles over ragged ports (CSR, block CSR, SELL); memoryless functional ports; restrict_ptr (see Phase 1 follow-ups). Why: sparse data is what the ragged ports are used for, and each bundle would carry its own invariants.
- [x] Investigate: wave kernels JIT only in the translation unit that defines the stages. Why: a FusedSystem of kernels defined in another TU aborts with "Could not find function call specialization target". Done: fixed by the fork's cross-TU dynamic-function fix (fork f815ee1, toolchain 2026.10.06).
- [x] Housekeeping: tests/smoke is stale against the D139/D140 APIs (HostViewMut semantics changed in C2c). Why: it is unbuilt and no longer matches the API. Done: tests/smoke deleted in P8.
- [x] Housekeeping: a clean gemver timing rerun, not concurrent with other GPU work. Why: earlier timings may have been disturbed by other GPU jobs. Done: the fusion benchmarks in 2947f27 time each variant in its own process.

## Orchestration redesign (rough shape; specifics settled item by item)

CoSpan's differentiator is orchestration on top of AdaptiveCpp's JIT and fusion. This phase makes CoSpan model anything, fast and accurate, with good ergonomics, and everything after builds on the principles settled here.

- [x] ~~SlabKit queue: one queue, one mental model.~~ Superseded (D144): `sk::unordered_queue`, fully out of order, is a separate type next to `sk::queue`; CoSpan submits kernels to it. (~~In-order items and parallel task graphs as the backbone~~: not built.)
- [x] ~~SlabKit out-of-order lane:~~ Superseded by `sk::unordered_queue` (D144). Original note: async_host is fully asynchronous again (unordered except by the events passed in; async_host_in_order is the ordered form), and the lane also takes device (non-host) work. In-order items accept extra event dependencies, and the global wait covers the lane. Why: the primitive for the push hooks below.
- [ ] SlabKit: remove the async_host keep-alive task in from_host. Why: it puts a host callback in the stream only to keep a buffer alive.
- [x] usm_vector operations (assignment, move, copy, resize, from_host) accept dependent events. Done: SlabKit 572b581 (register_events and event-taking mutations).
- [x] ~~Push pattern: consumer-side work is registered on a root at setup~~ Superseded (D144): the producer pushes to the variable's other placements after its launch; nothing is registered at setup. 
- [x] Remove ParallelSystem. Done (D144): concurrency follows from data dependencies (events) and `depends_on`, not from a user-chosen flavor. ~~FusedSystem stays as the one special flavor~~: superseded, a fused group is built by `fuse_components` with segments, `cs::global_barrier()` and `cs::group_waits_enabled` (section 11).
- [x] Lifetime is scope: shared ownership of variables (a variable lives while some port that uses it exists); parents own their children, delete them when their run policy ends, and rebuild them by name (D143).
- [ ] Visibility for those trade-offs: a user-facing per-forward report of allocations, frees and waits per subsystem (from memory_report and the diag counters). Why: the modeler can only choose scopes from a profile.
- [ ] Design guidance to document: prefer shapes inherited from connections, keep connected components on one device, and treat repeatedly resizing value-driven shapes as a cost. Why: these avoid reallocation churn.
- [ ] Ergonomics pass over model building (declaration forms, aliases, task-graph construction). Why: the demo has to look as good to write as it runs.
- [ ] Demo target: meshless LBM fluid (LRPIM) with moving mesh boundaries at large node counts. Why: it shows the framework and the method together.

## Deferred discussions (2026-10-11)

- [ ] How the initializer and the late-holder copy register their write events as reads as well (variable.hpp, `initialize` ~1010 and `populate` ~1052), so that a later writer waits for them.
- [ ] The test-only counter `hook_resize_notes` on VarMeta, added for one shape-hook test.
- [ ] The spelling table of user-facing names in internal/impl-log-section11.md (Jack's review).
- [ ] Whether add_subsystem should be legal outside setup, as the other wiring calls now are.
- [ ] OpenMP:
  - gemver onepass gives a wrong q (and naive with it).
  - Whether the fork's OpenMP backend lets a task run before its dependencies. This is what may have broken the empty or counting join-task collapse. The test needs an effect the host reads back, so nothing can be optimized away.
