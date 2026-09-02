# The pattern ledger — what modelers must be able to express

This document lists every compute pattern and modeling concept CospanMDAO
serves, separated into **primitives** (designed directly) and
**composites** (expressible from primitives; candidates for helpers). It
states needs, never solutions: how each need is met is the business of
the numbered specifications in this directory. It is kept current — read
cold, it describes what the framework promises today.

---

## 1. The two-tier rule

Every pattern axis carries two obligations:

- **Abstraction tier** — the common cases made easy (helpers, wrappers).
- **Possibility floor** — the general case guaranteed *expressible*,
  unaided by helpers.

A missing abstraction costs users convenience. A missing floor is a
designed-in corner. Both tiers are named for every axis below.

---

## 2. Primitives

### 2.1 Maps (intra-component)

A component is one unit of work with inputs and outputs; intermediates
are their own components, composed in systems.

- **Abstractions:** bijective, injective and surjective map helpers
  (reductions, gathers, scatters, region copies, masked ranges). These
  three classes close the abstraction tier.
- **Possibility floor:** the **general map** — arbitrary mixed reads and
  writes in one kernel — guaranteed by atomic element types available on
  every I/O and by indexed access through the data wrappers.
- **Work outside device kernels:** legacy code, file I/O, callbacks, and
  native libraries (BLAS, solvers) must be usable as ordinary components,
  ordered with everything else. Floor: a component body that receives
  the native handle and raw device pointers and does whatever it likes.
- **Shape source:** static, or **data-dependent extent** — an upstream
  output's *shape or value* may size downstream I/O.
- **Write-safety:** atomicity lives in the element type; a kernel cannot
  forget it.
- **Group-level parallelism:** work-group collectives and local memory
  are opt-in per kernel, with wrapper ergonomics, never required.
- **User types on the device:** a type with invariants, a non-trivial
  constructor or virtual functions must be constructible where it lives
  and usable from kernels through a pointer, even though the type itself
  is not device-copyable.
- **Minimal device payload:** what a kernel captures is the data it
  computes on and nothing the framework needs for bookkeeping; host-only
  concerns never ride into a launch.

### 2.2 Structure (inter-component)

- **Abstractions:**
  - `sequence` — subsystems run in declaration order.
  - `parallel` — independent subsystems run concurrently.
  - `cycle` — a feedback loop solved to convergence by an attached solver.
  - `march` — iterate N times or until a budget; no convergence semantics.
  - `fuse` — several components executed as one launch.
  - **Placement** — any subsystem may run on a device other than its
    owner's, with no change to the subsystems around it.
- **Possibility floor:** custom system types. The extension point and the
  escape hatch are the same mechanism: any orchestration a modeler needs
  — gating on a predicate, running once, running N times, rerunning only
  when inputs changed — is a system type, written without touching the
  components under it or the systems above it.
- **Opacity law:** a subsystem is fully opaque to its owner; returning
  from a forward means *done* (solves solved, fusions fused, parallel
  children joined). Composition legality follows from opacity, not from
  a matrix.
- **Inline composition:** parallelizing, fusing or placing a group of
  components must not require writing a new class.

### 2.3 Data

- **`connection`** — output→input aliasing; one primitive with a
  spectrum: identity → slice (a region of the source, zero-copy) →
  **assembly** (several producers' regions into one input, disjoint,
  with coverage checked by default) → general index map (computed
  gather). The only way I/O aliases across the input/output divide.
- **`promote`** — an I/O given a name in an owning system's namespace;
  always allowed, including for I/O that is already connected or
  aliased, so a higher level can always name what it needs.
- **`tie_inputs` / `tie_outputs`** — many same-kind I/O bound to one
  alias. Same kind only. A tie precedes any connection of its members;
  afterwards only the alias connects, and connecting through a member
  name is refused so that the tie is visible at every use site.
- **Index-map-valued I/O** — index maps are *data produced by components*
  (neighbour searches, sort permutations, sparse operator structure) and
  consumed by gathers, scatters and connections. Connectivity is
  computable by the model.
- **Non-durable I/O** — a lifetime attribute on declarations: durable, or
  scoped to a forward.
- **Chunked execution over bounded scratch** — N independent instances
  whose per-instance intermediate scratch would cap problem size if
  materialized at once (per-node dense solves, operator fits): run in
  chunks over reused transient scratch, results landing in the
  full-shaped output. A first-class pattern with its own ergonomics;
  touches shapes and transient lifetime.
- **Initialization** — free inputs and outputs receive initial values
  without a component existing to produce them; an owner can override
  initialization anywhere in its subtree by path.
- **Host access** — reading any I/O on the host by path; writing free
  inputs (driving design variables and parameters) and outputs
  (injecting external results), with host-side views that behave like
  the device wrappers.

### 2.4 Semantics

- **Design variables, constraints, objectives** — marks on I/O with
  bounds and scaling; marking any name of an aliased set marks the data;
  a design variable is a free input, an objective or constraint is an
  output.
- **Optimization at declared scale** — thousands of design variables and
  thousands of constraints *as the model declares them*: no aggregation
  of constraints to scalars and no parameter reduction is ever
  *required* to make optimization tractable. Aggregation remains
  available as a choice.
- **Derivatives** — exact gradients by automatic differentiation are the
  default; users write forward kernels only. Finite differences,
  complex step and analytical partials remain as the floor of the
  derivative axis (opt-outs for memory or choice, and the guarantee that
  nothing is underivable). Adjoint memory policy (store vs recompute) is
  an execution attribute.
- **Observation** — read any I/O by path and **enumerate** the model
  (subsystems, variables, shapes, connections) without prior knowledge
  of names. Recording, visualization, variable listing, debugging,
  profiling and generated documentation all build on this and on
  nothing else.
- **`serialization`** — checkpoint/restore of model state (a cached
  operator is exactly what restart wants).
- **Determinism policy** — bit-reproducible mode vs atomic
  nondeterminism; interacts with the general-map floor. A deliberate
  decision (§5).

### 2.5 Deployment

- **Heterogeneous single-binary portability** across vendors and the CPU
  is table stakes: the same model runs on a laptop CPU, a consumer GPU,
  and a datacenter node without source changes.
- **Multi-device** — subsystems placed on different devices in one model
  (§2.2 Placement), with data movement between them the framework's
  business, not the modeler's.
- **Possibility floor:** explicit device control stays reachable.

### 2.6 Implicit relations

A cycle is expressed as a **cyclic connection** ("x_rhs = x_lhs") plus "a
cycle needs a solver"; residual vocabulary never surfaces to the modeler,
though residual math runs underneath. Rationale: residual-first
interfaces gatekeep non-specialists; solver experimentation should not
require solver theory.

### 2.7 Temporal

Temporal components and systems: `march` + recurrence connections (state
fed across iterations). History windows (last k steps), subcycling
(nested march), mid-march events (gating inside a march) and hybrid
commit (sequence after cycle inside a march) are composites — §3.

---

## 3. Composites (expressible from primitives; helper candidates, never primitives)

| Pattern | Composition |
|---|---|
| Stencil | surjective neighbourhood read + bijective write |
| Scan, sort, compaction | multi-pass map compositions — shipped library helpers; compaction also exercises data-dependent extent |
| Segmented reduction | surjective + index-map metadata |
| **Ragged shapes (CSR/SELL)** | flat data + index-map-valued I/O; the abstraction must be a **typed wrapper** carrying structure as metadata (connection checks and segmented helpers need to see it), never a naming convention |
| Do once / run N times / run when changed / run if | custom system types over the structure floor (§2.2) |
| Regime switching, mid-march events | gating inside a march |
| RNG streams | recurrence of a resource (input/output pair across invocations) |
| Recurrence, history windows | march + connection across iterations (window = k-deep) |
| Subcycling / multi-rate | nested march |
| Hybrid commit | sequence after cycle inside a march |
| Warm start | initialization from serialization / previous state |
| Shared-I/O submodel interface | parallel + tie_outputs + atomic checks (the general shared-I/O rules) |
| Shared parameter fan-out | tie / promote |
| Cross-device data movement | placement + connection (copies inserted by the framework) |
| DOE / parameter sweeps | march over samples; orchestration is host code |
| Batched small dense solves (per-node operator fits) | chunked execution over bounded scratch + a map with a solve body; named helper obligation |
| NNs / DNNs | components + connections, full stop — the generalized-modeling thesis |
| Surrogate lifecycle | sweeps (data generation) + fit component + variant swap |
| Multi-fidelity | variant swap |
| **Model-variant swap** | models are declared **in code** — swapping subsystems behind fixed connections is ordinary programming; helpers optional |
| Nested / bilevel optimization | optimizers are systems; nesting is free |
| UQ / sensitivity products | forward-mode derivatives as a product surface |
| Standalone subsystem testing | opacity (any system runs alone) + observation |
| Recording, visualization, docs, introspection | observation |

---

## 4. Interactions (each must be designed together, not separately)

- Non-durable I/O × derivative rematerialization (transient data the
  backward pass needs is definitionally recompute-or-store).
- Non-durable I/O × transient scratch (scoped lifetimes are what scratch
  regions exist for).
- Do-once × serialization (cache the cached).
- Index-map build order × determinism policy.
- Chunked execution × shapes × transient lifetime × derivatives.
- General-map floor × determinism policy (the floor runs on atomics).
- Marks × I/O design (constraint/objective semantics considered with the
  I/O layer, never bolted on).
- Placement × ties (producers on two devices cannot share one output).

---

## 5. Deliberate decisions (decisions, not features)

Kept visible so that not-deciding never happens silently: **units &
scaling** · **determinism policy** · **precision portability** (fp64-poor
consumer GPUs; some backends have none) · **memory budgeting** (pre-
allocation vs model size; out-of-memory strategy).

---

## 6. The bare-minimum core (what makes everything else possible)

**Components:** the three map abstractions · general-case completeness
(atomics floor) · work outside device kernels · dynamic I/O sizing ·
derivatives with fallback partials · implicit solves · temporal
components · atomic element types (explicit).

**I/O & execution topology:** connection / tie / promote semantics ·
aliasing correctness · slice and assembly connections · marks · index-
map-valued I/O · non-durable I/O · initialization · host access ·
enumerability of I/O (observation's one hard requirement).

**Systems:** the system base contract — arbitrary extension that
integrates without touching existing code (THE most important item) ·
parallel · fused · placement · temporal systems · solving implicit
systems from cycles.

**Cross-cutting:** chunked execution over bounded scratch · optimization
at declared scale.

Everything not in this section is composite, served by helpers built on
it.

---

## 7. Principles (the yardsticks solutions are judged by)

Specifications cite these where a rule would otherwise look arbitrary.

- **P1 Legibility** — refuse an operation whose meaning would depend on
  knowledge not visible at the use site; what a name yields is what a
  reader at that site would expect.
- **P2 Adoption** — a subsystem written by someone else is placed,
  initialized, promoted, connected, wrapped and re-run without editing
  its source.
- **P3 Pay-for-use** — a model that does not use a capability pays
  nothing for it; no implicit synchronization, ever.
- **P4 Trust model** — the framework checks what it can prove; explicit
  overrides carry the rest, and the party holding the proof is the party
  entitled to override.
- **P5 Boundary** — nothing crosses the owner↔child boundary except the
  framework verbs; owners know nothing of children, children nothing of
  owners.
- **P6 Uniformity over detection** — every entry carries every
  mechanism's slot, used or not; no special-case logic anywhere.
- **P7 Users never classify their work** — no tags on kernels or
  components; the framework infers from signatures and declarations.
- **P8 Safe by default, explicit opt-out** — every check is on unless a
  user says otherwise, in code, at the site.
- **P9 RAII, never ordered destruction** — shared ownership wherever
  destruction order would otherwise matter.
