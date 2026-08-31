# CospanMDAO Redesign — Spec Discussion Plan (Phase 0b)

**Approved 2026-08-31.** Companion to `00-pattern-ledger.md` (phase 0a,
locked). Each phase is a Jack + Research Lead discussion that closes with a
numbered spec document in this directory. No implementation until the full
spec is assembled (phase 11).

## Mechanics

- Discussion per phase: dictation and convergence, 0a-style.
- Every phase close checks three riders: **two-tier test** (abstraction
  tier AND possibility floor covered), **AD-shadow note** (what this
  phase's choices imply for phase 8), **modeling-lens sweep** (serves
  modular modeling, not just compute patterns).
- Open items forward-link to the phase that owns them. A genuinely new
  pattern amends the ledger loudly, never silently.
- Seed material: three canonical user stories written and maintained as
  *ideal user code*, evolving as phases lock — **Sellar** (baseline),
  **B/C mycelium composite** (parallel + shared I/O), **LBM operator
  construction** (batching, index-map I/O, do-once). They become the
  examples and documentation skeleton.

## Phases

| # | Phase | Closes as |
|---|---|---|
| 1 | **Components & I/O foundations** — I/O structs (metadata, dtype system with atomic variants, enumerability, lifetime attribute), the one-kernel contract, the three map abstractions + general-map floor, dynamic and value-driven sizing | `01-components-io.md` |
| 2 | **Memory substrate & the slabkit boundary** — slabkit's slimmed surface (pools, smart-pointer deletes, scratch's fate), the local aliasing struct, the allocation pass, ownership incl. non-durable lifetimes; **slabkit allocation support for multi-device queues** (`q.get_devices()` → context) | `02-memory-substrate.md` |
| 3 | **Topology semantics** — connection spectrum (identity/slice/index-map, index-map-valued I/O), promote/tie rules and blocking, resolution chains, DV/constraint/objective algebra | `03-topology.md` |
| 4 | **The System base contract** — virtual interface (declare/allocate/forward/adjoint surface), opacity law, extension rules incl. the declared host-locus property, observation/enumerability plumbing | `04-system-contract.md` |
| 5 | **Structural Systems** — ParallelSystem (queue assignment, shared-I/O rules, checks + overrides), FusedSystem (both modes, legality checks, dynamic-function splice), conditional, change-tracked execution | `05-structural-systems.md` |
| 6 | **Implicit & solvers** — tied lhs/rhs UX, solver attachment and configuration, convergence semantics, first solver library shape | `06-implicit-solvers.md` |
| 7 | **Temporal** — march, recurrence wiring, history windows; subcycling/events/hybrid-commit confirmed as composites via worked examples | `07-temporal.md` |
| 8 | **AD** — engine (map-class adjoints ∘ element-op differentiation), zero-declaration UX, FD/CS/analytical floor, rematerialization policy, gradient dtypes incl. fp64 | `08-ad.md` |
| 9 | **Batching** — batched shapes, batched connections, batched AD, ensemble execution | `09-batching.md` |
| 10 | **Helpers & observation** — reductions/scans/sorts, ragged typed wrappers, batched small solves; observation API; docs-from-declarations | `10-helpers-observation.md` |
| 11 | **Decisions, audit, assembly** — units/scaling, determinism, precision, memory budgeting ruled; full-spec consistency audit; consolidation + examples plan | `11-assembly.md` + unified spec |

Multi-device is not a phase: it arrives upstream (acpp multi-device queue
contribution, ruled — ledger §2.5); the sole spec touchpoint is phase 2's
slabkit item. The upstream contribution itself is a separate workstream,
scheduled outside this plan.
