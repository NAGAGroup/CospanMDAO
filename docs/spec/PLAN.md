# CospanMDAO Redesign — Spec Discussion Plan (Phase 0b)

**Approved 2026-08-31; reshaped for spec-as-code 2026-09-01; optimizers
added and implementation checkpoint set 2026-09-02.** Phases are carved by
declaration surface, not topic; AD and batching are amendment phases.
Companion to `00-pattern-ledger.md` (phase 0a, locked). Each phase is a
Jack + Research Lead discussion that closes with a numbered spec document
in this directory. **Every phase is first-release scope.**

## Mechanics

- Discussion per phase: dictation and convergence, 0a-style.
- Every phase close checks two riders: **two-tier test** (abstraction
  tier AND possibility floor covered) and **modeling-lens sweep** (serves
  modular modeling, not just compute patterns). Gradients/AD are NOT
  discussed before phase 7: an implemented base will inform that design
  better than foresight. The only AD-related constraints held earlier
  are the two structural ones in 01.A/01.C (wrapper `value_type` as the
  sole element spelling; KernelBody as a value-mode template) — cheap
  now, unrecoverable later.
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
  construction** (chunked scratch, index-map I/O, do-once). They become
  the examples and documentation skeleton.
- The spec prescribes only what is necessary. Opens are left where the
  answer is not yet known, and must be resolvable INSIDE what the spec
  allows; a resolution that surprises the spec's authors is a spec bug.
- **Spec follows implementation, destructively, in the same change**: a
  contract the implementation had to change gets its spec line edited in
  the same commit, or the spec stops being the authority.

## Implementation checkpoint — after phase 4

Phases 1–4 + `lifecycle.md` are a closed, implementable modeling framework
for acyclic models. **Implementation starts here, before phases 5–11**:
slabkit (02.A, with tests) → a spike on the load-bearing acpp facts
(dynamic-function call sequences with the cursor protocol, retarget on
both queues, custom operations under retarget) → cospan core in lifecycle
order (wrappers/metadata → KernelBody + generated device functions →
Component → System + ExecCtx → Model), with the 01.D sample components and
an acyclic multi-System model as end-to-end tests. The two structural
AD constraints above are enforced during implementation even though AD
is not built.
Resident architects (SlabKit, CospanMDAO) are briefed at this point, not
before. Phases 5–11 resume on the implemented base.

## Phases

| # | Phase | Closes as |
|---|---|---|
| 1 | **Components & I/O foundations** — dtypes+wrappers, metadata+shapes+declaration API, kernel machinery (KernelBody incl. custom-op bodies, chained executor, Local scratch), Subsystem verbs, Component, System + ExecCtx + structural APIs | `01-components-io/` (01a–01e) — DONE |
| 2 | **Model & memory** — `create_model`, the two retargetable queues, per-device substrates, the setup chain; slabkit's surface (arena/pool handles, malloc/make families, no Context, no sycl::context); aliasing as root+peers sets | `02-model-memory/` (02a in slabkit's repo, 02b, 02c) — DONE |
| 3 | **Topology semantics & optimization marks** — identity/slice/computed connections via mapping Systems, coverage, view composition, cross-level ties, the marks API | `03-topology.md` — DONE |
| 4 | **System types** — the extension contract by example (RunOnce/RunN/Conditional as custom System types via early return; Systems never do work; custom operations are Components); observation = path resolution + enumeration | `04-system-types.md` — DONE |
| — | **Implementation checkpoint** (above) | slabkit + cospan core |
| 5 | **Implicit & solvers** — ImplicitComponent, the Solver hierarchy (solvers are System types), attachment, convergence contract, the connection ledger that records cycles and owns the acyclic assumption's exception | `05-implicit-solvers.md` |
| 6 | **Temporal** — march (iterate N / until budget, no convergence semantics), recurrence wiring across iterations, history windows; subcycling/events/hybrid-commit proven as composites by samples | `06-temporal.md` |
| 7 | **AD** — engine document + AMENDMENTS to 01a (grad wrapper forms), 01d (adjoint verb), 01e (backward walk), 05 (solver adjoints); custom-op/host-work opt-outs (analytical, FD/CS); rematerialization; consistency audit inside the phase | `07-ad.md` + amendments |
| 8 | **Batching (chunked execution over bounded scratch)** — the LRPIM-class pattern: N independent instances whose per-instance intermediate scratch would cap problem size if materialized at once; run in chunks over reused transient scratch, results landing in the full-shaped output. A UX for this compute pattern; AMENDMENTS to 01b (shapes), transient lifetime, and 07 (AD through chunks) | `08-batching.md` + amendments |
| 9 | **Optimizers** — SYCL-native, from scratch, designed for thousands of DVs and constraints AS DECLARED (no lossy reduction assumed): primal-dual **interior point** (matrix-free KKT via AD products; the scale claim) and **SQP** (quasi-Newton, device QP; the medium regime). Shared device linear algebra (dense LDLᵀ/Cholesky, CSR, CG/MINRES/GMRES), line search / trust region, scaling (with phase 11), history + restart, convergence contract; KS aggregation as an OPTIONAL helper. Deferred to a later release: bound-constrained quasi-Newton, population methods. Validation: Hock–Schittkowski/CUTEst, then Sellar, aerostructural, airfoil | `09-optimizers.md` |
| 10 | **Helpers & observation** — kernels roster (scans, sorts, ragged, batched small solves), recorders/listing/N2 over `get_subsystem` + enumeration, docs-from-declarations | `10-helpers-observation.md` |
| 11 | **Decisions, audit, assembly** — units/scaling, determinism, precision, memory budgeting ruled; final consistency audit; consolidation + examples plan | `11-assembly.md` + unified spec |

Multi-device is not a phase: it is the two retargetable queues and the
per-device substrates of phase 2 (acpp `ACPP_EXT_CG_PROPERTY_RETARGET`);
the acpp multi-device-queue contribution is a separate workstream,
scheduled outside this plan.
