# Lifecycle — construct, set up, run, tear down

The spine the numbered documents hang from: every framework verb, in the
order it fires, with the state that exists after each. Binding contract.
Cast: Model (02.B) · Subsystem verbs (01.D §1) · System and ExecCtx (01.E)
· Component (01.D) · IOEntry sets (02.C) · slabkit memory (02.A).

**Assumption stated once:** the model is ACYCLIC through phase 4. Cycles are
a solver-phase construct (05); the connection ledger that records them is
where any ordering exception below will live.

---

## 1. Construct

```cpp
auto model = create_model<Top>(ModelConfig{...}, top_args...);
```

1. Model validates its config and creates its two queues: `q_in` —
   in-order, `AdaptiveCpp_retargetable`, `AdaptiveCpp_coarse_grained_events`;
   `q_ooo` — out-of-order, `AdaptiveCpp_retargetable`. Both bound to
   `default_device`; every submission may retarget.
2. Model constructs `Top(top_args...)`. A constructor is plain C++ state.
   Structural calls and declarations outside `setup()` throw.

**State:** two idle queues, one root object, nothing else. `get_*` throws
"not set up".

---

## 2. `model.setup()` — the setup chain

Every step is a Subsystem verb driven from the root; Systems recurse in
declared order (their `children_` vector IS the exec order); nobody asks
what kind a child is.

### 2.1 Declare — `root.run_setup()`

`System::run_setup` → `setup()`. Inside it:

- `add_subsystem<T>(name, [props], args...)` / `add_subsystem(name, props)`:
  construct the child, register the name, set its owner, **run the
  child's `run_setup` before returning** — so `connect("d1.y1", …)` later
  in this same `setup()` sees `d1` fully declared.
- `connect` / `promote` / `tie_*`: resolve paths in this namespace (all
  children declared), apply the set operation (02.C). A computed
  connection (03) creates or extends the input's mapping System and adds a
  mapping component; marks record on roots.
- Then this System's post-setup pass: each mapping System — framework
  `tie_outputs(allow_plain)`, alias → input, coverage check, insert
  before the consumer's child.

`Component::run_setup` → `setup()` → `declare_*` creates entries (roots of
singleton sets; flags by kind; DimRecords; lifetime; coverage). Zero
allocation; no device anywhere in this step.

**State:** final tree and exec order; final sets; marks. No storage.

### 2.2 Exec info — `bytes = root.get_exec_info(default_device)`

`System`: `ctx.device` = own `property::device` else `parent_device`; seed
`{device: 0}`; process **own alias entries first** (a tie alias root
counts its static bytes here); then each child in order, merging returned
maps (sum on key).

`Component`: device = `parent_device`; per own entry `e`:
- root → count static bytes on this device;
- peer, `root(e).device == device` → nothing (reads the root);
- peer, devices differ → count `e`'s bytes here, mark
  `owns_nonroot_allocation`, append `e` to `root(e).needs_memcpy`.

Every root is processed before any peer of it: alias roots before
children, producers before consumers (acyclic). Returns `{device: bytes}`.

### 2.3 Substrates

Model: for each `(device, bytes)`: `substrates[device] = { DeviceArena(bytes),
DevicePool iff configured for that device }`. Shared-backed handles (02.A).

### 2.4 Prepare — `root.prepare_exec(queues, substrates)`

`System`: complete `ctx` — `q_in`, `q_ooo`, `primary` and `lane` as its
owner handed down (plain owners pass their set unchanged; a Parallel owner
assigns `primary = q_ooo, lane = i`), `substrate = &substrates[device]`;
allocate own alias roots; recurse.

`Component`: allocate every own entry that is (root ∨
`owns_nonroot_allocation`) ∧ allocatable now — durable∧static from the
arena; durable∧dynamic with a known shape via the dynamic chain (pool →
OS); dynamic-shaped and transient stay unallocated. Nothing is pointed at
yet.

### 2.5 Compile — `root.compile(root_ctx)`

`Component`: device function(s) + id at the owner's fuse rank if any, else
native; individual config; custom-op bodies recorded. Returns
`CompiledInfo`. `System`: children in order; if it has a policy —
Fused: rank agreement, no custom ops, intra-fuse shape dependencies over
the final sets, parallel-mode producer-consumer unless allowed, barrier
interleaving, the sequence config; Parallel: cross-sibling shape
dependencies; ChangeTracked: record external-input roots. Checks live
here, not in the flavor's own `run_setup`, because a flavor's children
are wired by its owner after the flavor's setup returned.

### 2.6 Init — `root.run_init(root_ctx)`

`System`: children in order, then `init(q)`. `Component`: `init(q)`. `q` is
`q_in` retargeted to the subsystem's device. Post-order means an owner's
`init` has the last word by path. Initializers write roots — and,
for roots with `needs_memcpy` peers, copy into those peers. Unnecessary
inits (connected inputs) are harmless. An initializer on unallocated
storage throws.

### 2.7 Wire

`FromInput` DimRecords register their `dim_scratch` with their input's
root (the producing output); static input peers validate against their
roots (flat or subarray size). Dynamic pairs validate per forward.

### 2.8 Sync

Every member wrapper is pointed at its storage: own allocation if it has
one, else `root()->data()`; shape and stride (+ view) set. Unallocated
dynamic roots sync a null pointer and their declared extents. Model is
set up. **State is not yet meaningful** — it holds initialization; state is
defined after the first `run()`.

---

## 3. `model.forward()` / `model.run()`

`Model::forward()` → `root.forward()` → `root.run_compute_forward()`.

### 3.1 A System's turn (default loop, 01.E §3)

```
ctx.refresh_pending()                          // both queues' wait lists
for child in children:
    child.run_pre_compute(ctx)
    d = child.run_compute_forward()
    if d:  ctx.pending = { d->submit(ctx.target()) }   // this System submits
    else:  ctx.refresh_pending()               // child submitted itself
    child.run_post_compute(ctx)
return null
```

Every submission: `depends_on(pending)`, `AdaptiveCpp_retarget{ctx.device}`,
lane hint if any. In-order primary clears `pending` on first use;
out-of-order primary attaches it to each. **Parallel**: same loop on
`q_ooo` with per-child lanes and one pending snapshot. **Fused**:
pre-compute ALL → compute ALL (collect; null is an error) → pack, stamp
group slices, one staging, ONE submission with the sequence config →
post-compute ALL. **Conditional**: evaluate, run one branch's loop.
**ChangeTracked**: compare versions; run the loop and record, or return.

### 3.2 A Component's turn (01.D §5)

1. `run_pre_compute(ctx)`: automatic shape update (FromInput waits its
   scratch event, reads) → `pre_forward()` → dynamic validation →
   allocate-if-grown on `ctx.device` via the dynamic chain → wrapper sync.
2. `run_compute_forward()` → `compute_forward()` → description (kernel or
   custom op). The owner submits it.
3. `run_post_compute(ctx)`: D2H copies into registered `dim_scratch`es;
   memcpys root → each `needs_memcpy` peer; bump output roots' `version()`.
   All through `ctx`, after the owner's submission.

### 3.3 `run()`

`forward()`, then wait **both** queues.

**State:** every output root holds this forward's values; dynamic roots
sized to this forward; versions moved.

---

## 4. Between runs

`view_*` (D2H on `q_in`, waits its copy; writable output view writes back
on destruction and on `commit()`), `set_*` (H2D; version bump; copies to
cross-device peers), `memory_report()`, `visit`.

---

## 5. `~Model`

Wait both queues → destroy the root (entries release handles) → substrates
(allocations pin their slabs; the last handle frees) → queues.

---

## 6. Verification owed before these lines are normative

- The `sycl::context` a `Placement` for device D must carry so that
  kernels retargeted to D from `q_in`/`q_ooo` may access memory allocated
  for D (gates 2.3 for multi-platform device sets).
- `AdaptiveCpp_retarget` composes with `AdaptiveCpp_enqueue_custom_operation`
  (custom-op bodies on non-default devices).
- `queue::memcpy` device→device across devices: peer copy vs host-staged
  (cost note for `needs_memcpy`).
- `std::hash<sycl::device>` under acpp (`Substrates` keying).
