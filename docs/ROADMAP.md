# CoSpan roadmap

Messy by design. Checkboxes, notes, strike-throughs welcome. Each item points at the spec revision that will absorb it and the date of the project-knowledge notes it came from (2026-09-28 unless stated).

## Before any code (spec drafts still needed)

- [ ] Chunked systems — a system flavour with a scratch budget; nesting and sub-chunking are required; the chunk axis and the sub-chunk declaration are open. (absorbed by: r02)
- [ ] Cycles, solvers and implicit components — cycles by registration order; an implicit component is a self-connection with its own buffer; a balance helper. (absorbed by: r02)
- [ ] A "proven by" list per r01 section — the order-sensitive tests each contract needs. (absorbed by: r02)

## Build order (forward-complete first)

1. [ ] SlabKit on the fork — repin, six gap tests, spec/README refresh.
2. [ ] CoSpan skeleton, including chunked systems — forward passes for every non-cyclic pattern.
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

## Later, wanted

- [ ] Parallel Systems (beyond Components).
- [ ] Serialization.
- [ ] A derivative-aware paragraph per forward feature.

## Housekeeping

- [ ] CoSpan pixi manifest repinned to the naga toolchain (full previous manifest at dbebc3a:pixi.toml).
- [ ] README.md rewrite (it still describes the first-pass framework).
- [ ] Tests from the first code commit.
