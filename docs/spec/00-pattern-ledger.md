# CospanMDAO Redesign — Phase 0a: The Pattern Ledger

**Status: LOCKED 2026-08-31** (Jack + Research Lead alignment sessions).
This document enumerates every pattern the redesigned CospanMDAO must serve,
separated into **primitives** (must be designed directly) and **composites**
(expressible from primitives; candidates for helper abstractions). It
contains **no solutions** — solutions belong to the phased spec discussions
that follow in this directory (`01-…`, `02-…`, …).

Why this exists: designing from computational patterns alone misses the
primitives that make *modeling* patterns safe and intuitive, and gaps found
late become corners. This ledger is the complete record so nothing is
silently forgotten.

---

## 1. The two-tier rule

Every pattern axis carries two obligations:

- **Abstraction tier** — common cases made easy (helper APIs, wrappers).
- **Possibility floor** — the general case guaranteed *expressible*,
  unaided by helpers.

A missing abstraction costs users convenience. A missing floor is a
designed-in corner. Both tiers are named for every axis below.

**Standing rider on every phase:** *transparent multi-device splitting*
(user programs single-device; framework splits kernels/data — Celerity-class,
possibly via contribution to AdaptiveCpp's multi-device queue) is a future
feature, NOT bare-minimum now — but every phase must ask: **"would this
decision block automatic splitting later?"** Connections, shape semantics,
and index-map-valued I/O are where that corner would get poured.

---

## 2. Primitives

### 2.1 Maps (intra-component)

A Component has inputs, outputs, and exactly ONE kernel. Intermediates are
their own components, composed in Systems.

- **Abstractions:** `bijective`, `injective`, `surjective` map helpers
  (reduction wrappers, masked ranges, etc.). Gather is surjective; scatter
  without collisions is injective; these three close the abstraction tier.
- **Possibility floor:** the **general map** — arbitrary mixed reads/writes
  in one kernel — guaranteed by (a) atomic dtypes available on every I/O
  type and (b) indexed access through the data wrappers. No helper; full
  possibility.
- **Adjoint mirror:** the floor exists on the backward pass too — atomic
  gradient accumulation available wherever the forward floor is used.
  Adjoints invert the access pattern (design symmetric from day one).
- **Attribute — shape source:** static, or **data-dependent extent** —
  an upstream output's *shape or value* may size downstream I/O.
- **Attribute — write-safety:** atomicity lives in the dtype
  (`atomic_f32`, `atomic_i32`, …); wrappers return atomic references so
  kernels cannot forget.

### 2.2 Structure (inter-component)

- **Abstractions:**
  - `sequence` — subsystems run depth-first in declaration order.
  - `parallel` — ParallelSystem; children on separate in-order queues,
    joined before forward returns; plus the shared-I/O rules (tie_outputs,
    atomic-dtype checks by default, user override).
  - `cycle` — solver-attached implicit solve; convergence semantics.
  - `march` — iterate N / until budget; NO convergence semantics.
  - `conditional` — predicate-gated execution.
  - `fuse` — FusedSystem; accepts components only; sequential mode (max
    range, masked indices) and parallel mode (flattened concatenated
    ranges); legality checks by default, user-overridable.
- **Possibility floor:** custom System subclasses. The extension point and
  the escape hatch are the same mechanism — a property of the
  modularization architecture, held deliberately.
- **Opacity law:** a subsystem is fully opaque to its owner; `forward`
  returning means *done* (solves solved, fusions fused, parallel children
  joined). Composition legality follows from opacity, not from a matrix.

### 2.3 Data

- **`connection`** — output→input aliasing; ONE primitive with a spectrum:
  identity → slice/stride (subarray) → general index map. The only way I/O
  aliases across the input/output divide.
- **`promote`** — single I/O promoted to the owning System's namespace.
- **`tie_inputs` / `tie_outputs`** — many same-kind I/O bound to one
  promoted alias. Same-kind ONLY (never input↔output). Tying/promoting
  **blocks** direct connection on members — only the alias connects; the
  reverse holds (connected I/O cannot be tied/promoted). The System owning
  the tie/promote holds a real I/O struct of its own, so resolution is one
  uniform rule: chained `get_subsystem`/`get_input` through the hierarchy,
  terminal is System or Component alike. No special aliasing logic.
- **Index-map-valued I/O** — index maps are *data produced by components*
  (neighbor searches, sort permutations, sparse operator structure) and
  consumed by gathers, scatters, and connections. Connectivity itself is
  computable by the model. (Current cospan supports this; recorded here so
  it is never forgotten.)
- **Non-durable I/O** — a lifetime attribute on declarations: durable vs
  scoped to a step or subgraph.
- **Host-locus boundary** — work that is not a device kernel (legacy
  code, files, callbacks). Floor provided by the System-base contract
  permitting host-side work in custom subclasses — this property must be
  DECLARED in the contract, not discovered.
- **Batching** — cross-cutting primitive: same model, N samples, batched
  execution (vmap-shaped). Touches shapes, connections, and AD
  simultaneously; without it, domain sizes are limited unnecessarily.

### 2.4 Semantics

- **DV / constraint / objective marks + propagation algebra:** the DV flag
  propagates upward through promotion and tying; marking a promoted alias
  works because the alias is the source; connecting an output to a DV'd
  input unmarks it. Constraint/objective semantics must be considered
  during I/O design, not bolted on.
- **AD:** full automatic differentiation is the DEFAULT — users write
  forward kernels and get exact gradients, always; no manual partials of
  any kind required. FD / complex-step / analytical remain as the
  possibility floor of the derivative axis (opt-outs for memory or choice,
  and the guarantee that nothing is underivable). Adjoint memory policy
  (store vs recompute — rematerialization) is an execution attribute.
- **Change-tracked execution** — run when inputs have changed, skip
  otherwise ("do once" for statically-placed nodes: LBVH/LRPIM/nonlocal
  operator construction). First run = always-changed, so this plausibly
  absorbs `initialization` (phase ruling). Distinct from `conditional` by
  who decides: framework-tracked dirtiness vs model predicate. Relaxing
  "never rerun" to "rerun on dirty" is how moving geometry arrives later
  with no new primitive.
- **Observation** — ONE non-invasive primitive behind recording,
  visualization (N2-style and live rendering), variable listing, debugging
  and profiling: read any I/O by path + **enumerate** the model
  (variables, shapes, connections) without prior knowledge of names.
  Design requirement now: I/O structs are enumerable. Docs-from-
  declarations builds on this.
- **`serialization`** — checkpoint/restore of model state (a cached
  operator/LBVH is exactly what restart wants).
- **Determinism policy** — bit-reproducible mode vs atomic
  nondeterminism; interacts with the general-map floor (which runs on
  atomics). A deliberate decision, recorded in §5.

### 2.5 Deployment

- **Abstraction (future):** transparent multi-device splitting, acceptance
  test "a few user lines, never a model rewrite." Rider in §1 applies NOW.
- **Possibility floor:** explicit device/queue control stays reachable.
- Heterogeneous single-binary portability (AdaptiveCpp generic SSCP) is
  table stakes beneath both.

### 2.6 ImplicitComponent (its own thing)

No aliasing — a single kernel cannot write its input, so tied lhs/rhs are
two real buffers. The UX speaks **cyclic connection** ("x_rhs = x_lhs") +
"a cycle needs a solver"; residual vocabulary never surfaces, though
residual math runs underneath. Rationale: OpenMDAO's residual-first UX
gatekeeps non-specialists; solver experimentation should not require
solver theory.

### 2.7 Temporal

Temporal components and temporal systems: `march` + recurrence connections
(state fed across iterations). History windows (last k steps), subcycling
(nested march), mid-march events (conditional-in-march), and
hybrid commit (sequence-after-cycle-in-march; gradient timing is
AD-phase material) are composites — see §3.

---

## 3. Dissolution table (composites — helper candidates, never primitives)

| Pattern | Composition |
|---|---|
| Stencil | surjective neighborhood read + bijective write |
| Scan, sort, compaction | multi-pass map compositions — shipped library abstractions; compaction also exercises data-dependent extent |
| Segmented reduction | surjective + index-map metadata |
| **Ragged shapes (CSR/SELL)** | flat data + index-map-valued I/O. RULED composite; the abstraction must be a **typed wrapper** carrying structure as metadata (connection checks, segmented helpers, adjoint layouts need to see it), never a naming convention |
| RNG streams | recurrence of a resource (input/output pair across invocations) |
| Recurrence, history windows | march + connection-across-iterations (window = k-deep) |
| Subcycling / multi-rate | nested march |
| Regime switching, mid-march events | conditional inside march |
| Hybrid commit (post_settle) | sequence after cycle inside march |
| Warm start | initialization from serialization/previous state |
| B/C submodel interface | parallel + tie_outputs + atomic checks (the general shared-I/O rules) |
| Shared parameter fan-out | tie/promote + DV propagation algebra |
| DOE / parameter sweeps | march over samples driving batching; orchestration is host code |
| NNs / DNNs | components + connections, full stop — the generalized-modeling thesis |
| Surrogate lifecycle | sweeps (data gen) + fit component + variant swap |
| Multi-fidelity | variant swap |
| **Model-variant swap** | RULED composite: models are declared **in code** — swapping subsystems behind fixed connections is ordinary programming; helpers optional |
| Nested/bilevel optimization | optimizers are components/systems; nesting is free |
| UQ / sensitivity products | AD forward mode as product surface |
| Standalone subsystem testing | opacity (any System runs alone) + observation |
| Recording, viz, docs, introspection | observation |
| Batched small dense solves (per-node operator fits) | batching + map-with-solve-body; named helper-library obligation |

---

## 4. Interaction flags (phase material; listed so they don't vanish)

- Non-durable I/O × AD rematerialization (transient data the backward pass
  needs is definitionally recompute-or-store).
- Non-durable I/O × slabkit slim-down (scoped lifetimes are what scratch
  allocators exist for; where scratch lives is now a sharper question).
- Change-tracked execution × serialization (cache the cached).
- Index-map build order (LBVH) × determinism policy.
- Batching × shapes × connections × AD (the reason batching is
  cross-cutting).
- General-map floor × determinism policy (the floor runs on atomics).
- DV/constraint/objective semantics × I/O struct design (forced
  consideration, per §2.4).

---

## 5. Deliberate-decisions side list

Decisions, not features. Kept visible so not-deciding never happens
silently: **units & scaling** (adopt/reject OpenMDAO-style units and driver
scaling) · **determinism policy** · **precision portability** (fp64-poor
consumer GPUs; Metal has none) · **memory budgeting** (pool sizing vs model
size, OOM strategy).

---

## 6. The bare-minimum core (what makes everything else possible)

**Components:** the three map abstractions · general-case completeness
(atomics floor, both directions) · dynamic I/O sizing · AD + fallback
partials · implicit solves · temporal components · atomic dtypes (explicit).

**I/O & execution topology:** connection/tie/promote semantics and blocking
rules · aliasing correctness · subarray connections · conditional execution
· DV/constraint/objective algebra · index-map-valued I/O · non-durable I/O
· enumerability of I/O structs (observation's one hard requirement).

**Systems:** the System base contract — arbitrary extension that integrates
without touching existing code (THE most important item), including the
declared host-locus property · ParallelSystem · FusedSystem · temporal
systems · solving implicit systems from cycles · change-tracked execution.

**Cross-cutting:** batching (seated deliberately in the phase plan).

Everything not in this section is composite, served by helper abstractions
built on it.
