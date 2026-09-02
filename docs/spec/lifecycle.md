# Lifecycle — construct, set up, run

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
   `q_ooo` — out-of-order, `AdaptiveCpp_retargetable`. Both on
   `default_device`, default context; every submission may retarget.
2. Model constructs `Top(top_args...)`. A constructor is plain C++ state.
   Structural calls and declarations outside `setup()` throw.

**State:** two idle queues, one root object, nothing else.

---

## 2. `model.setup()` — the setup chain

`run_setup → get_exec_info → substrates → prepare_exec → compile → init`.
Every step is a Subsystem verb driven from the root; Systems recurse in
declared order (`children_` IS the exec order); nobody asks what kind a
child is.

### 2.1 Declare — `root.run_setup()`

`System::run_setup` → `setup()`. Inside it: `add_subsystem` constructs the
child, registers the name, sets its owner, **runs the child's `run_setup`
before returning**; `connect` / `promote` / `tie_*` resolve paths in this
namespace and apply the set operation (02.C) — computed connections (03)
create or extend the input's mapping System; marks record on roots. Then
this System's post-setup pass: each mapping System — framework
`tie_outputs(allow_plain)`, alias → input, coverage check, insert before
the consumer's child.

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
allocate own alias roots and point their wrappers; recurse.

`Component`: allocate every own entry that is (root ∨
`owns_nonroot_allocation`) ∧ allocatable now — durable∧static from the
arena; durable∧dynamic with a known shape via the dynamic chain (pool →
OS); dynamic-shaped and transient stay unallocated. Then point every
member wrapper at its storage: own allocation if it has one, else
`root()->data()` — guaranteed allocated by now (roots precede peers);
shape and stride (+ view) set; static input peers validate against their
roots here.

### 2.5 Compile — `root.compile()`

`Component`: device function(s) + id at the owner's fuse rank if any, else
native; individual config; custom-op bodies recorded. `System`: children
in order. Fused and Parallel legality is NOT checked here — it is checked
each forward on the descriptions the children return (cheap, host-side,
and it is where the information is).

### 2.6 Init — `root.run_init()`

`System`: children in order, then `init(q)`. `Component`: `init(q)`. `q` is
`q_in` retargeted to the subsystem's device. Post-order means an owner's
`init` has the last word by path. Initializers write roots — and, for
roots with `needs_memcpy` peers, copy into those peers. Unnecessary inits
(connected inputs) are harmless. An initializer on unallocated storage
throws.

**State:** storage for every static root, initialized; wrappers current;
kernels compiled. The model can run. Its state is not yet meaningful —
it holds initialization; state is defined after the first forward.

---

## 3. `model.forward()`

`Model::forward()` → `root.forward()` → `root.run_compute_forward()`.
Enqueue only. `model.wait()` waits both queues when the caller wants the
results on the host; a caller looping N forwards need not wait between
them — the model guarantees ordering internally.

### 3.1 A System's turn (default loop, 01.E §3)

```
ctx.refresh_pending()                          // both queues' wait lists
for child in children:
    child.run_pre_compute(ctx)
    d = child.run_compute_forward()
    if d:  ctx.pending = { d->submit(ctx.target()) }   // this System submits;
                                               //   routed on d->is_custom_op()
    else:  ctx.refresh_pending()               // child submitted itself (or skipped)
    child.run_post_compute(ctx)
return null
```

Every submission: `depends_on(pending)`, `AdaptiveCpp_retarget{ctx.device}`,
lane hint if any; kernel descriptions take the outer-kernel path, custom
ops take `AdaptiveCpp_enqueue_custom_operation` through the same handler.
In-order primary clears `pending` on first use; out-of-order primary
attaches it to each. **Parallel**: same loop on `q_ooo` with per-child
lanes and one pending snapshot. **Fused**: pre-compute ALL → compute ALL
(collect; a null or custom-op description is an error) → validate the
wave on the descriptions (rank agreement, scratch, producer-consumer in
parallel mode) → pack, stamp group slices, one staging, ONE submission
with the sequence config → post-compute ALL. **Custom System types**
(spec 04): early-return from `run_compute_forward`.

### 3.2 A Component's turn (01.D §5)

1. `run_pre_compute(ctx)`: for each own input driving a shape via
   `FromInput` — enqueue the D2H of that input into the entry's
   `dim_scratch`, then a custom operation (depending on that D2H event)
   that applies the resize host-side, reallocating via the OS route from
   inside the custom-op body if grown; then **wait on that custom op's
   event** — the one forced sync, so the launch geometry below is known.
   Then `pre_forward()` → dynamic validation → wrapper sync.
2. `run_compute_forward()` → `compute_forward()` → description (kernel or
   custom op). The owner submits it.
3. `run_post_compute(ctx)`: memcpys root → each `needs_memcpy` peer,
   through `ctx`, after the owner's submission.

**State:** every output root holds this forward's values; dynamic roots
sized to this forward.

---

## 4. Host access

`view_*` (D2H on `q_in`, waits its copy; writable output view writes back
on destruction and on `commit()`), `set_*` (H2D; copies to cross-device
peers), `memory_report()`, `get_subsystem(path)` chaining and
`get_inputs()/get_outputs()` enumeration (spec 04 §3).

---

## 5. End of life

`~Model` waits both queues. Everything else is RAII: allocations pin
their slabs (02.A), handles release in whatever order members go.
