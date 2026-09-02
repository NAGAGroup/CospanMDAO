# CospanMDAO Redesign — Spec Discussion Plan (Phase 0b)

**Approved 2026-08-31; reshaped for spec-as-code 2026-09-01** (phases carved by declaration surface, not topic; AD and batching are amendment phases). Companion to `00-pattern-ledger.md` (phase 0a,
locked). Each phase is a Jack + Research Lead discussion that closes with a
numbered spec document in this directory. No implementation until the full
spec is assembled (phase 10).

## Mechanics

- Discussion per phase: dictation and convergence, 0a-style.
- Every phase close checks three riders: **two-tier test** (abstraction
  tier AND possibility floor covered), **AD-shadow note** (what this
  phase's choices imply for phase 8), **modeling-lens sweep** (serves
  modular modeling, not just compute patterns).
- Open items forward-link to the phase that owns them. A genuinely new
  pattern amends the ledger loudly, never silently.
- `lifecycle.md` is the spine: every framework verb in firing order with
  the state after each; numbered documents declare the surfaces it walks.
- Spec documents stay ~500 lines or less; a phase that outgrows one file
  becomes a directory of chunk documents. Agents read short docs whole and
  filter long ones — the cap is what keeps the contract actually read.
- Seed material: three canonical user stories written and maintained as
  *ideal user code*, evolving as phases lock — **Sellar** (baseline),
  **B/C mycelium composite** (parallel + shared I/O), **LBM operator
  construction** (batching, index-map I/O, do-once). They become the
  examples and documentation skeleton.

## Phases

| # | Phase | Closes as |
|---|---|---|
| 1 | **Components & I/O foundations** — dtypes+wrappers, metadata+shapes+declaration API, kernel machinery (KernelBody, chained executor, Local scratch), Component, System execution slice + structural APIs | `01-components-io/` (01a–01e) — DONE |
| 2 | **Model, memory & finalize** — `Model` (setup driver, queue creation, run, final wait); the finalize pass as ONE walk: alias resolution order + allocation pass; slabkit's slimmed surface (pools, smart pointers); pool ownership flavor; multi-device-queue allocation | `02-model-memory/` |
| 3 | **Topology semantics & optimization marks** (DONE) — index-map connections and index-map-valued I/O, subarray composition, cross-level ties, the marks API (design vars / constraints / objectives with bounds and scaling slots) | `03-topology.md` |
| 4 | **System types** — ConditionalSystem, ChangeTrackedSystem, versions, observation plumbing (Systems never do work; custom operations are Components) | `04-system-types.md` — DONE |
| 5 | **Implicit & solvers** — ImplicitComponent, the Solver hierarchy, attachment API, convergence contract | `05-implicit-solvers.md` |
| 6 | **Temporal** — temporal System type(s), recurrence wiring, history windows; composites proven by samples | `06-temporal.md` |
| 7 | **AD** — engine document + AMENDMENTS to 01a (grad wrapper forms), 01d (adjoint verb), 01e (backward walk), 05 (solver adjoints); consistency audit inside the phase | `07-ad.md` + amendments |
| 8 | **Batching** — batched shapes/connections/AD as AMENDMENTS to 01b, 03, 07; ensemble execution System | `08-batching.md` + amendments |
| 9 | **Helpers & observation** — kernels roster (scans, sorts, ragged, batched small solves), observation API, docs-from-declarations | `09-helpers-observation.md` |
| 10 | **Decisions, audit, assembly** — units/scaling, determinism, precision, memory budgeting ruled; final consistency audit; consolidation + examples plan | `10-assembly.md` + unified spec |

Multi-device is not a phase: it arrives upstream (acpp multi-device queue
contribution, ruled — ledger §2.5); the sole spec touchpoint is phase 2's
slabkit item. The upstream contribution itself is a separate workstream,
scheduled outside this plan.
