# CoSpan specification — r03

**What this is.** The CoSpan design as built on `main` after the section 11 implementation (plan: `docs/plan-section11.md`,
log: `docs/impl-log-section11.md`). It describes what the code does and nothing else; where the code differs from the design
review it says so in the last section. Authority for the decisions: `docs/design-review-2026-10-03.md`, section 11 (it
supersedes sections 1 to 10 where they conflict). r02 is superseded and kept as history. Every API name is the name in
`include/naga/cospan/`.

Notation: `cs::` is `naga::cospan`, `sk::` is `naga::slabkit`. SlabKit owns the contracts of `sk::context`, `sk::queue`,
`sk::unordered_queue`, `sk::usm_vector` and the pool. SlabKit is pinned at `4f57086` (no collapse of event lists; pruning at assignment at 8 events; uninitialized allocations; no own-writes list).

---

## 1. Vocabulary and data

**Variable.** One piece of model data: everything joined by `connect`, `promote` or a tie. The variable *is* its shared
metadata object, `detail::VarMeta` (a ragged variable is `RaggedVar`; the typed layer is `Var<T>`). Holders share it by
`shared_ptr`; subsystems never do. It holds:

- dtype and rank;
- the shape **definition** (one `DimRecord` per dimension: specified, unspecified, or owned by a hook) and the **actual**
  shape (`ResolvedShape`);
- the initializer (the last declaration wins);
- the stale flag and the materialized flag;
- the cycle mark, the push-suppressed flag (tied outputs), and the tie marks (`tied_input`, `tied_output`, kept apart);
- weak pointers to the holders that share it and to the container of each placement.

It holds **no events and no memory** and keeps nothing alive.

**Holders and owners.** A holder (`detail::Instance`) is an input, an output, an alias (a promotion or a tie alias) or a host
view. Ports and aliases are owned by a component or a System by `shared_ptr`; a host view is owned by the user's scope. A
holder holds a `shared_ptr` to its variable, knows its device and storage kind (`DataKey{device, kind}`), and holds a
`shared_ptr` to the container of its own placement.

**Containers exist from declaration.** Declaring a holder takes the container of its placement (`detail::DeviceData<T>`, a
type-erased `DeviceDataBase`), creating it **empty and unbound** if the placement has none yet. The variable keeps a weak
pointer; entries are never removed (a key whose container is gone gets a new empty one when a holder asks again). An
unbound container has size 0, no events and no memory; `bind(Runtime&)` makes its `sk::usm_vector` at the first
materialization that reaches it (setup has no runtime). When a call merges two variables, the winner absorbs the container
entries the loser had and it lacked.

**Materialization** is a holder operation, called by the owner in its precompute (`Var<T>::materialize(Runtime&, trigger)`):

- *Variable not yet materialized — global materialization.* Every live container is bound and sized at once and the flag is
  set, so all allocation cost is up front and later submissions are cleanly ordered by events. The materializing holder
  initializes unless it is an output (its data is about to be written); a cycle variable runs its initializer whichever
  holder asks (section 2.5).
- *Variable already materialized.* The holder looks only at its own container. If it is empty it is sized and copied from the
  first live container with data (`populate`).
- A *stale* variable (section 4.4) is initialized in place, with no allocation.

**Lifetime is these pointers.** When the last holder on a placement goes, that container's memory is freed; the free waits on
the container's events. When the last holder anywhere goes, the variable goes. A rearmed subsystem runs its setup again and
creates fresh variables; a fresh variable owns its initialization.

**Shape changes.** A shape change resizes every container of the variable, the host container included.

**Promotions.** A promotion alias is a real holder. The promote call sets its device from the member's device.

---

## 2. Setup and wiring

### 2.1 Two passes

`Model::setup()` runs once (a second call, or a call after a failed first one, throws).

1. **Setup:** the root System's `setup()`, recursive through `add_subsystem`. Each `add_subsystem` builds the child and runs
   its `setup()` before returning.
2. **Finalize:** collects the marks by absolute path, as weak pointers to their holders. Marks have no scope rule: whatever
   is collected stays. A mark under a deleted child keeps its path and its pointer expires.

Setup never allocates, never submits, never compiles.

**Which calls are phase-bound.** `add_subsystem` and the marks (`add_design_var`, `add_constraint`, `add_objective`) are legal
inside `setup()` only. `connect`, `promote`, `tie_inputs`, `tie_outputs`, `init_input`, `shape_hook`, `depends_on`,
`override_run_policy` and `rearm` are legal at any time. Every rule is checked in its own call.

### 2.2 add_subsystem

```cpp
add_subsystem(cs::subsystem_desc<Axpb>("a", 2.0f, 1.0f));                                  // eternal, default device
add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<Setup>("s")); // run policy
add_subsystem(dev, cs::subsystem_desc<Axpb>("b", 1.0f, 0.0f));                             // device
add_subsystem(props, dev, cs::subsystem_desc<T>("t", args...));                            // both, properties first
```

`cs::subsystem_desc<T>(name, args...)` carries the name and decayed copies of the constructor arguments, so a slot can build
the child again. It carries no device; constructors never take one. `T` is a `Component` or a `System`.

| Property | Meaning |
|---|---|
| `property::run_n{n}` | the child takes its first `n` turns (`n >= 1`), then is deleted |
| `property::do_once{}` | `run_n{1}` |
| `property::transient{}` | the child exists for one forward of its parent: rebuilt at the start, deleted after its turn |

`run_n` and `do_once` exclude each other; no property means eternal. Each child lives in a **slot** of its parent (name,
factory, device, run policy). `get_subsystem(path)` returns null when the slot exists but its child is deleted, and throws
when the name was never declared.

### 2.3 Devices

Every holder takes AdaptiveCpp's default device when declared (`ModelConfig` has no default device). `set_device` places a
subtree: on a component, its holders; on a System, its own aliases and then each live child. `add_subsystem` builds the child,
runs its `setup()`, then applies `set_device` with the device it was given and then with the parent's own forced device if the
parent was placed. **The outermost call wins**, and `set_device` covers aliases too.

- `promote` copies the member's device at the call.
- `tie_inputs<DT, R>(members, name, shape = {}, device = {})`: the device (default device if none) is where the alias lives
  and where the tie's initializer runs. Members may be elsewhere; a device no member uses costs one more allocation.
- `tie_outputs<DT, R>(members, name, shape = {}, policy = require_atomic, device = {})`: every member must be on the tie's
  device, checked at the call; because `set_device` can later move one member, the owning System checks again at the start of
  every forward, before any submission, and the error names the member, both devices and the tie.
- `connect` takes no device.

### 2.4 Wiring and merges

| Call | Effect | Whose variable wins |
|---|---|---|
| `connect(src, dst)` | feeds `dst` from `src` | the source's |
| `promote<DT, R>(member, name)` | a peer alias joining the member's variable | the member's |
| `tie_inputs<DT, R>({members}, name, shape, device)` | a new variable (a target); the members merge into it | the alias's |
| `tie_outputs<DT, R>({members}, name, shape, policy, device)` | a new variable (a source); the members merge into it | the alias's |

Paths are dotted and relative to the calling System: `"a.y"` is output `y` of child `a`; a single segment is this System's
own alias. Each call checks its own rules before anything changes:

- Rank must match (promote and the ties also check `DT`).
- **Definitions merge:** unspecified absorbs specified; two different specified values are an error at the call (section 5).
- **Connect flags.** A connected, tied or hooked input can no longer be a connect target; a tied input cannot be connected and
  a connected one cannot be tied; a hooked input takes no connection (and a connected or tied input takes no hook).
- **Tie policy.** `TieOutputsPolicy::require_atomic` (default) needs atomic member dtypes; `allow_plain` says the writers are
  disjoint. Every tied-output variable is push-suppressed (section 3.5).
- **Ties of ties** are refused, on either side (the variable's `tied_input` / `tied_output` mark of that kind). Promoting a
  tie is allowed.
- **Merging two already-materialized variables** is not defined and throws at the call
  ("merging two materialized variables is not defined (yet)"). A rebuilt child rejoining a live variable is the one case where
  one side holds data: that side survives.

After the rules pass, the losing variable's live holders are re-pointed at the winner and the container entries merge.

### 2.5 Cycles

`connect` records a **cycle** on the variable when the consumer's child was registered before the producer's: the consumer
then reads the producer's previous value. `connect` keeps the target's initializer (it does not drop it). The first
materialization of a cycle variable runs the initializer even though the variable is connected, takes the shape from it, and
throws when there is none ("'<var>' closes a cycle ('<dst>' reads '<src>', which runs after it) and has no initializer").
That is checked at the first forward, not at setup, because an initializer may be set after setup. `update_input` accepts a
cycle variable and re-initializes it at the next forward. The warning to `std::cerr` (once per pair) stays.

### 2.6 Initial values

`init_input(path, init)` (on a System by path; on a component for its own input by name) declares the initial value of a free
input's variable; **the last declaration wins** (a parent's `setup()` runs after its children's). It is ignored when the input
ends up with a source; naming an output throws. `Model::get_input<DT, R>(path).update(init)` is the same declaration made
from outside (section 4).

| Factory | Meaning |
|---|---|
| `cs::init::fill(v)` | every element is `v` |
| `cs::init::zero()` | every element is 0 |
| `cs::init::from_host(vec)` | element `i` is `vec[i]` |
| `cs::init::custom<DT, R>(fn)` | `fn(unordered_queue&, device, deps, Initializer<DT, R>)` enqueues its work and returns its event |
| `cs::init::custom_host<DT, R>(fn)` | `fn(HostInit<DT, R>&)` fills a host buffer that is then copied to the device |

Each takes an optional `cs::init::Shape` (`{n, m}`, or `cs::init::ragged(n, d1)` / `ragged(n, counts)`) that supplies a free
input's unspecified dimensions.

### 2.7 Replay, rearm, run-policy override

Each System logs its own `connect`, `tie_*`, `promote`, `init_input` and `shape_hook` calls per child. Rebuilding a child (a
transient at the start of a forward, or `rearm`) runs its factory and `setup()`, applies the device, then the parent
**replays** the logged calls that name it. A logged call whose other side no longer exists is skipped (the rebuilt side then
rejoins a surviving variable, or stays fresh); a rebuilt consumer whose producer is gone and whose variable lives nowhere is a
user error, so promote what must stay durable.

- `rearm(name)` is a protected `System` verb, legal at any time. It deletes the child if live, builds it again, and gives it
  its full run policy.
- `override_run_policy(path, policy = {})` is public and legal at any time. The path may go through child Systems
  (`"b.c"`); the policy is a `property_list` (empty = eternal). It sets the policy on the subsystem's slot in its own parent
  and restarts the counter. A child that is gone and is not made transient comes back at once; a transient is rebuilt at the
  next precompute. It is **not** logged for replay: when this System rearms an ancestor of the target, the slot is built with
  its declared policy and the caller applies the override again.
- **Nested rearm** (A holds B, B rearms C, A wires into C) is a user error and is not handled. The escape hatch is
  `override_run_policy` (for example making `"b.c"` eternal); the author of B documents how B is used.

---

## 3. Execution and events

### 3.1 The forward

`Model::forward()` enqueues one forward of the root and returns: precompute, compute, postcompute, for components and Systems
alike. Each step is framework-owned and calls an empty-by-default user hook.

| Step | System | Component |
|---|---|---|
| precompute | rebuild transient children, then `pre_forward()` (where `rearm` and `override_run_policy` may act). **No child precompute runs here.** | materialize inputs, `pre_forward()`, materialize outputs |
| compute | the loop (3.2) | `compute_forward()` describes the launch; the component submits it and records its events |
| postcompute | `post_forward()` | push to the variable's other placements (3.5), then `post_forward()` |

### 3.2 The System loop

For each live child in declaration order: the child's precompute, compute and postcompute. A child whose counter is 0 is
skipped. After its turn, a `run_n` counter decrements (at 0 the child is deleted) and a transient child is deleted. **Deletion
issues no explicit wait**: the container frees wait on their own events, so a deletion stalls naturally until the child's
kernels are done. Add subsystems that can go out of scope as late as possible in `setup()`.

A `tie_outputs` alias takes a turn right after the postcompute of the last child (by first path segment) that holds a member
and will run, and before that child is deleted: that turn is the push (3.5). A tie whose member children are all skipped or
deleted does not push that forward.

A `shape_hook` step (section 5) runs in the loop immediately before the first child that holds its target.

### 3.3 The component steps

**Precompute**, in order: every distinct input variable is materialized; the user's `pre_forward()` runs (`shape_of` and
`set_shape` are legal here only); every distinct output variable is materialized (a dimension still unknown is the
shape-not-known error); each wrapper is pointed at its data.

**Compute.** `compute_forward()` returns a `std::unique_ptr<cs::LaunchDescription>`:
`create_launch_config(range, in, out, body)` (plain, group), `(range, wg, in, out, body)` (group, explicit work-group),
`(range, wg, in, out, scratch, body)` (group with `Scratch`), `(in, out, body)` (custom-op, host). A plain or group body
launches through the group launcher (section 6); a custom-op or host body submits itself on the unordered queue. The launch is
ordered by its dependencies only (3.4) and **its events are registered at submit** (`record_launch`): a read on every input
container and a write on every output container.

**Postcompute.** The pushes (3.5), then `post_forward()`.

### 3.4 Containers are the only source of event truth

Every container (`sk::usm_vector`) keeps a **read list** and a **write list**. Variables keep none.

- A reader waits on the write list (`read_deps()`). A writer waits on the read list (`write_deps()`). Writes never wait on
  other outside writes. There is one read list and one write list, nothing else.
- A kernel registers a read on each input container and a write on each output container.
- The container's own operations go into the same lists: a copy out is a read of the source; a copy in, a fill or the copy
  of a reallocation is a write of the destination; they wait on both of the container's lists. A copy between containers (`copy_from`) waits on
  the source's writes and on both of the destination's lists, then registers a read on the source and a write on the
  destination. A free waits on every event in both lists.
- **No collapse; pruning at assignment.** The lists are handed out whole. A registration (by us or the container) that makes a
  list longer than 8 events drops the events whose `command_execution_status` is complete. Reading a list never prunes it.
- **Allocations are uninitialized.** `resize(n)` and `reserve(n)` never fill and a size-only constructor does not fill; a call
  given a value does. A reallocation with elements to keep copies them over (a write, in the write list) and returns the
  copy's event as `std::optional<sycl::event>`; every other case returns an empty optional. Nothing in CoSpan may rely on
  zeroed memory: a variable with no initializer holds undefined values until a producer writes it.
- **The resize-event slot.** `VarMeta` keeps a list of pending resize events. Every container resize made for the variable
  (materialization sizing, `set_shape`, hooks, initializer shapes, the ragged index arrays) appends the event it returned.
  Reading the slot drops the completed events and returns the rest. Every launch adds the slot events of each variable it
  writes to its dependencies, together with the containers' `write_deps()`: bare components, every stage of a fused group,
  and both members of a tied output (they share the variable).
- **depends_on** `depends_on("b", "a")` (legal at any time; `a` added before `b`; a duplicate edge is ignored): `a`'s forward
  runs first, its events are collected and handed to `b`'s submissions for ONE forward; if `b` is a System every component
  inside adds them. Execution order is data dependencies and `depends_on` only: children are called in declaration order but
  kernels are ordered by events, so two writers of a tied output run concurrently, and a zeroing producer and an accumulator
  on one `tie_outputs` need `depends_on`.

### 3.5 Pushes

The host is just another placement. After a component's launch, postcompute copies each output variable to its other live
containers (including host views) with copy assignment, unless the variable is push-suppressed. A tied-output variable is
push-suppressed: its members only register their writes, and the owning System pushes through the alias in its loop turn.

### 3.6 Queues and waits

The Model owns one `sk::context`, one `sk::queue` (containers) and one `sk::unordered_queue` (kernels and host tasks, ordered
by the caller's events only). Nothing in the framework relies on in-order execution, and nothing waits on a queue to free.
`Model::wait()` is `wait_and_throw()` on both queues: optional, for "everything I submitted is done" and to surface
asynchronous errors. `~Model` does not wait; every container is freed after its registered events, so destroying a model with
work in flight is safe (a test holds a kernel open and checks the destructor blocks until it finishes).

---

## 4. Host access

### 4.1 Handles and shorthands

```cpp
auto in  = model.get_input<DType::f32, 1>("a.x");     // InputHandle<DT, R>:  view(), ragged_view() (R == 2), update(init)
auto out = model.get_output<DType::f32, 1>("b.y");    // OutputHandle<DT, R>: view(), ragged_view() (R == 2)
cs::HostView<DType::f32, 1> y = out.view();
y.sync();
```

`get_input` can also set an initializer (`update`); `get_output` never can. Naming an output in `get_input` (or an input in
`get_output`) throws `std::invalid_argument` naming the path and its kind. The shorthands `view_input`, `view_output`
(`HostView<DT, R>`), `view_ragged_input`, `view_ragged_output` (`HostRagged<DT, 2>`) and `update_input(path, init)` share the
code. A ragged view also works on a dense variable (its blocks are the rows). A handle is valid while its Model lives; a view
it made is not tied to the Model.

### 4.2 Views

A view is a read-only holder on the host, owned by the user. **It can be made at any time**: it materializes the variable like
any holder and throws only when the shape cannot be resolved yet ("its shape is not known yet"). Reading before anything is
computed shows whatever is in memory.

- While a view lives, **every forward pushes one copy to it** (each live view costs one copy per forward). An unwanted view
  should go out of scope.
- **`sync()`** waits on the host container's write list. Reading without syncing is the user's responsibility.
- **A shape change resizes the host container too, so a live view follows it**; spans taken before the change are invalid.
  After the next forward and a `sync()` the same view shows the new size and values.
- A view outlives the Model (it keeps the variable and the host container).
- `HostView`: `sync()`, `size()`, `data()`, `span()`, `shape()`, `operator()(i[, j[, k]])`. `HostRagged`: `sync()`, `size()`,
  `data()`, `span()`, `n()`, `block_offsets()`, `block(i)`.
- A zero-element variable behaves like any other size (present, size 0, a host copy of size 0).

### 4.3 Component views

`ComponentBase::view_input(const Input<DT, R>&)` (and the `RaggedInput` overload) is a protected method that makes a host view
of one of the component's **own inputs**, never its outputs. It is **legal inside `pre_forward()` only** and uses the runtime
that precompute received. Keep it as a member (an `std::optional`, made on the first forward) and `sync()` before reading. It
is how a component sizes its outputs from values.

### 4.4 update_input

`update_input(path, init)` sets the initializer (the last call wins) and marks the variable stale; it queues no work. A
variable that is not materialized is always stale. At the next forward the first holder to materialize it runs the initializer
on its own device and copies to the others; a materialized stale variable is initialized in place (a write: it waits on the
reads and is registered as a write). Several updates before a forward collapse to the last, and a variable nothing touches is
never initialized. A host view synced before the next forward still shows the old values. On an input whose variable has a
source it warns (`std::cerr`) and does nothing, unless the variable is part of a cycle. Naming an output throws.

### 4.5 wait

`wait()` is optional (3.6). A host-body exception surfaces from `wait()`, not from `forward()`.

---

## 5. Shapes

### 5.1 Definition and actual shape

The **definition** of a dimension is fixed once setup ends: **specified** (a value, never changes), **unspecified** (supplied
from outside) or **hook-owned** (one registered hook owns it). The **actual** shape changes at run time and only through
`VarMeta::set_by_producer` / `set_dims`, which check every change against the definition. Every dimension has exactly one
setter: a specified one the definition, a hook-owned one its hook, an unspecified one the producer (an output) or the
initializer (a free input). A shape that is needed and not known is an error at the forward; setup checks wiring rules only.

*Defining* a shape is: declaring a port, merging through the wiring calls, registering a hook, `fixed_at_run`. *Setting* it is:
an output's producer in `pre_forward`, a hook when it runs, a free input from its initializer.

### 5.2 Declarations

A declaration says per dimension only specified or `cs::unspecified`. There are no derived dimension specs.

```cpp
declare_input(x_, "x", cs::unspecified);              // rank 1, any length
declare_input(m_, "m", 3, cs::unspecified);           // rank 2: 3 rows, any columns
declare_output(y_, "y", 4);                           // rank 1, exactly 4
declare_input(x_, "x", cs::unspecified_shape<2>());   // array form
```

Ties declare a shape with `cs::shape(...)` (one entry per dimension; `cs::shape()` all unspecified). Promotions take none.

### 5.3 Ragged variables

Ragged is **rank 2 only**: a dense leading dimension N plus one per-index dimension:

```cpp
declare_input(r_, "r");                              // both dimensions unspecified
declare_input(r_, "r", 8);                           // N = 8, counts unspecified
declare_input(r_, "r", cs::counts({3, 0, 5}));       // host counts; N = 3
declare_input(r_, "r", cs::offsets({0, 3, 3, 8}));   // N + 1 offsets; N = 3
```

`declare_output` takes the same forms; a ragged tie uses `cs::shape()`, `cs::shape(n)`, `cs::shape(cs::counts(v))` or
`cs::shape(cs::offsets(v))`. Kernels read `cs::RaggedInput` / `cs::RaggedOutput` (`block(i)` gives a dense block).

### 5.4 set_shape, shape_of

Legal inside `pre_forward()` only. **Inputs are never set by a component**; the data's producer is the authority.

```cpp
sycl::range<1> shape_of(x_);                         // the shape of one of my ports now
set_shape(y_, n);                                    // one entry per dimension: a number or cs::as_is
set_shape(y_, cs::as_is, 12, cs::as_is);             // rank 3: leave dimensions 0 and 2 alone
cs::RaggedShape shape_of(r_);                        // n, total, counts (host or device column)
set_shape(r_out_, rshape);                           // from a RaggedShape
set_shape(r_out_, n, counts);                        // N (or cs::as_is) and host counts
set_shape(r_out_, n, cs::as_is);                     // N alone
```

An entry count that differs from the rank throws. A **number is allowed only on an unspecified dimension**: on a specified
dimension it throws at the call, even when it matches ("dimension N is declared as V; set_shape('x') takes a number only on an
unspecified dimension (pass cs::as_is)"), and on a hook-owned one it throws naming the hook ("... is owned by a shape hook; ...
cannot give it a number"). `cs::as_is` leaves a dimension alone; one still unknown at materialization is the shape-not-known
error. Nothing changes when a check throws. There is no "two sets in one forward must agree" rule.

**Tied outputs.** A tie call marks each member's output unable to set its shape (a component's `set_shape` on it throws and
points at the alias). The members' definitions merge at the tie; a dimension still unspecified must come from a hook on the
alias, otherwise materialization fails.

### 5.5 Shape hooks

```cpp
shape_hook(target_path, <one descriptor per dimension>);
```

| Descriptor | Dimension comes from |
|---|---|
| `cs::dim::as_is()` | left alone |
| `cs::dim::from_shape(path, d)` | dimension `d` of the shape of variable `path` |
| `cs::dim::from_value(path, index)` | the value of element `index` (flat) of the integer variable `path` |
| `cs::dim::custom(fn)` | `fn(cs::ShapeHookCtx&) -> std::size_t` (`shape_of(path)`, `value_of(path, index)`) |
| `cs::dim::fixed_at_run(v)` | a value fixed at the call: it makes the dimension specified (inside the hook it is the same as `as_is`) |
| `cs::ragged::counts_from(path)` | ragged target: per-index counts are the values of the integer variable `path` (rank 1 `[N]`, or column 0 of `[N, k]`) |
| `cs::ragged::offsets_from(path)` | ragged target: the N + 1 block offsets in the integer variable `path` |

A ragged target takes one ragged descriptor (or, for an output, `(dense descriptor, cs::dim::as_is())`: the hook owns N alone
and the producer sets the counts with `set_shape(out, cs::as_is, counts)`).

```cpp
shape_hook("c.x", cs::dim::from_shape("p.y", 0));                              // c.x as long as p.y
shape_hook("m.a", cs::dim::from_value("cfg.rows", 0), cs::dim::fixed_at_run(16));
shape_hook("r.x", cs::ragged::counts_from("counts.n"));
tie_outputs<DType::atomic_f32, 1>({"a.acc", "b.acc"}, "acc");
shape_hook("acc", cs::dim::from_value("cfg.bins", 0));                         // a hook on an output tie alias
```

- **Targets:** an input, an input tie alias, an output tie alias, or an ordinary untied output. For an untied output the hook
  runs before the producer, which passes `cs::as_is` for the hooked dimensions. A member of a `tie_outputs` is refused: hook
  the alias. Sources may be any variable, read only.
- **Ownership:** a hooked dimension must be unspecified at the call and has exactly one owning hook. On an input, a hook and a
  connection exclude each other.
- **One step per call:** each `shape_hook` call creates one templated step (`detail::ShapeHookStep<D...>`) that has only a
  forward and covers all its dimensions.
- **Run point:** every forward, in the System loop, immediately before the first child that holds its target, whatever the
  declaration order. **There is no source-ordering check at setup:** a hook works or fails when it runs. When a source no longer
  exists and the target's hooked dimensions are already known, the last shape stands; "... does not exist right now" is thrown
  only when the shape was never known.
- **The work:** the reads of one hook are batched into one host wait. `counts_from` / `offsets_from` build a device column on
  the source's device with the parallel exclusive scan (`device_scan.hpp`); only the total and a validity flag come back. A
  negative count, or offsets that do not start at 0 or that decrease, is an error.
- **Inside a fused group** stage precomputes run in order before the launch, so `from_shape` on an earlier stage's output can
  work and `from_value` cannot.
- A hook is replayed with the rest of the wiring when a child it names is rebuilt.

---

## 6. Fused groups

"Fused group" is a component built from several components that run as stages of ONE kernel launch. From outside it behaves
like a component and external wiring follows the normal rules. The framework provides tools and execution guarantees and
**inserts no barriers**: a stage that reads across work-items what an earlier stage wrote calls its own group barrier, and
across groups nothing is ordered unless a boundary says so.

### 6.1 One launch path

Every plain-body and group-body component launches through a `detail::GroupLauncher` (`group_launch.hpp`): a fused group owns
one for its stages, and a **bare component is a group of one stage** in one default segment. Custom-op and host components keep
their own submission and cannot be stages.

- **Stage function.** `stage_fn<KB>` is one `SYCL_EXTERNAL` function template per body type, with the same signature for every
  type: `(detail::StageItem<R>, const detail::StageGeometry*, const void* const*, int stage)`. It reads `geometry[k]` and
  `args[k]`, builds the item wrapper and calls the user's body. Its dynamic-function-definition id is a function-local static
  (`stage_definition_id<KB>()`) reached through the component's virtual `definition_id()`.
- **The kernel.** One generic kernel per rank calls the declared noinline placeholder `run_stages_r1/r2/r3(item, geometry*,
  args**, int)`, which the JIT replaces with an **indexed call sequence** of the stage functions
  (`define_as_indexed_call_sequence(&run_stages_rN, {ids}, index_param = 3)`, an AdaptiveCpp-fork extension: argument 3 is
  replaced by the position k of each definition).
- **Config cache.** The `dynamic_function_config` of a sequence lives in a process-wide, never-destroyed cache
  (`detail::group_config(rank, defs)`), keyed by (rank, the definitions' symbol names). A config must outlive every kernel that
  uses it, and a transient group can be deleted before its kernel is compiled.
- **Stage data.** Each component keeps a one-element device array of its argument struct (`StageArgs<KB>`), copied in from a
  host member every forward as a container write. The group's **pointer array** (one pointer per stage) is allocated, filled
  and waited on once, at the group's first precompute (devices are final then); it is redone only if the group is rebuilt. The
  group's **geometry array** is refilled every forward from a host member with event ordering: one `StageGeometry` per stage
  (`group_count`, `group_base`, `segment_groups`, `ticket_state`, `segment`, `segments`, `stages`, `local_offset`,
  `local_bytes`, `range_mode`), followed by the segment table (6.3).

### 6.2 The descriptor

```cpp
add_subsystem(cs::fuse_components(
    "f", cs::connects_desc({{"a.y", "b.x"}}),                              // optional; connects run forward in stage order
    cs::segment<cs::range_mode::whole>(cs::subsystem_desc<A>("a"),         // a and b share their groups
                                       cs::subsystem_desc<B>("b")),
    cs::global_barrier(),                                                  // a boundary item (6.4)
    cs::segment(cs::subsystem_desc<C>("c"))));                             // c has its own groups, in the same launch
```

`cs::fuse_components(name, [ConnectsDesc,] items...)` returns a descriptor for `add_subsystem`. An item is a bare component
descriptor (a one-stage segment), `cs::segment<opts...>(descs...)`, or a boundary item. `cs::system_from(name, [connects],
descs...)` builds a plain System of the same children instead. The result type is `FusedGroup<Items...>`.

- **Segment options** (at most one of each kind, `static_assert`ed): `cs::local_mem::partitioned` (default) /
  `cs::local_mem::shared`, and `cs::range_mode::own` (default) / `cs::range_mode::whole`.
- **Compile-time rules:** at least one stage; stages are components; no custom-op or host stage; all stages are plain or all
  are group; one kernel rank. Ragged ports are refused at the end of setup.
- **Connects inside a group** (`connects_desc({{src, dst}, ...})`) are the same `connect` with the same rules, scoped to the
  group, and mark the target variable internal. They are allowed in every range mode but must go **forward in stage order**: a
  connect from a stage's output to the input of the same or an earlier stage throws at setup:
  `"'<dst>' reads '<src>', which runs after it in fused group '<name>'"`.
- Ports of the children are reached by path (`"f.a.x"`). Ties inside a group follow the ordinary tie policy.

### 6.3 Segments, range, local memory

A **segment** is a run of consecutive stages that share groups. Its block is the largest group count of its stages; the launch
is the sum of the blocks. With no boundary, segments use the hardware group id and no atomics.

- **Range mode `own`:** a stage runs only for its first N groups of the block. **`whole`:** every stage runs for every group of
  the block (the `GroupItem` group range widens dimension 0 to the block).
- **Work-group shape.** Every forward, before the launch, every stage that asks for a shape must ask for the same one
  ("... ask for different work-group shapes ..."); a stage that asks for none takes the group's.
- **Local memory.** A stage requests bytes with `LaunchDescription::request_local_bytes(n)` (each request rounded up to 16)
  on top of its `Scratch`. `partitioned`: each stage gets its own slice, the segment's size is the sum; `shared`: every stage
  gets the same block, the size is the max. The launch's local block is the largest segment size, checked against the device's
  `local_mem_size` each forward ("... bytes of local memory ...", naming the stages). `GroupItem::local<T>(n)` is a typed bump
  allocator over the stage's slice after its `Scratch`; it returns null when the slice is exhausted.
- **Segment table.** After the `stages` geometry entries: the S block sizes, then the S boundary kinds (`none`,
  `ticket_order`, `global_barrier`), as `std::uint64_t`.

### 6.4 Boundaries and ticket mode

`cs::ticket_order()` and `cs::global_barrier()` are items **between two segments**, never first, never last, never next to
another boundary, and **only in groups whose stages are all group bodies** (a plain body has no work-group to cooperate with).
These are `static_assert`s of `FusedGroup` (traits `detail::boundaries_placed_ok<Items...>()`, `detail::has_boundary_v`,
`stage_checks<...>::all_group`).

If a group has any boundary, **the whole launch uses tickets**:

- At the start of each work-group the leader takes a ticket with a relaxed `fetch_add` on the ticket counter and broadcasts it
  to the group (`StageItem::group` carries it). The ticket, not the hardware id, gives the work-group's segment and its group
  within the segment through the segment table, in ticket order. Every earlier ticket is held by a group that has started, so
  a group waiting only on earlier groups cannot deadlock **and no co-residency is needed**.
- **Ticket state** is a device container owned by the group, `uint32` words: `[0]` the ticket counter, `[1, 1+S)` per-segment
  completed counters, then one completion flag per ticket (`1 + S + total groups` words). It is reset every forward by a
  container fill ordered after the previous launch; the launch is an outside writer of it and waits on the fill. Its pointer is
  in every geometry entry. A group without a boundary has a null pointer and allocates nothing.
- **Completion.** After the stage sequence the work-group does a group barrier and a device-scope release fence; the leader
  stores the group's flag and increments its segment's completed counter, both as release atomics.
- **`cs::global_barrier()`:** before its first stage, every work-group of the following segment has its leader spin (acquire)
  until every earlier segment's counter equals its block, then a group barrier and an acquire fence. **`cs::ticket_order()`**
  adds no wait.

### 6.5 The item wrapper

Every group body, fused or not, gets `cs::GroupItem<R>` and never raw SYCL group objects. Group-level indices are relative to
the stage's segment, from 0; local ids and ranges are physical.

| Member | Meaning |
|---|---|
| `get_local_id()`, `get_local_range()`, `get_local_linear_id()` | physical, as `nd_item` |
| `get_group_id()`, `get_group(d)`, `get_group_range()`, `get_group_range(d)`, `get_group_linear_id()` | group within the segment |
| `get_global_id()`, `get_global_range()`, `get_global_linear_id()` | within the segment's grid |
| `get_segment()`, `get_segment_count()`, `get_segment_group_count(s)` | the segment structure |
| `local<T>(n)` | a typed slice of the stage's local memory |
| `wait_group(s, g)`, `wait_earlier_segments()` | collective waits (below) |
| `get_group()`, `get_sub_group()` | the physical group, for barriers and reductions |

The waits are `const` and **collective** (every work-item calls them): one work-item spins with acquire loads, then a group
barrier and an acquire fence release the group. `wait_group(s, g)` waits for group `g` of segment `s`;
`wait_earlier_segments()` for every group of every earlier segment. They need ticket mode: without it the pointer is null and
the behavior is undefined, with no branches and no checks; waiting on your own segment or a later one never returns. A plain
body gets `sycl::id<R>`.

### 6.6 Events

The single launch waits on, and registers a read on: every stage's input containers, the argument arrays, the geometry array
and the pointer array. It registers a write on every stage's output containers and on the ticket state. Its dependencies are
those of one component (per stage: inputs' writes, outputs' reads and their variables' resize events, `depends_on` events) collected over every
stage, plus the geometry, pointer and ticket-state arrays. Internal connections work out naturally. The group's event is
recorded on every stage.

---

## 7. Memory and pool accounting

Each device gets one SlabKit pool, sized by the user up front:

```cpp
cs::ModelConfig cfg;                                   // the default-selector device
cfg.pool_bytes[dev]      = 512u << 20;                 // per device
cfg.default_pool_bytes   = 64u << 20;                  // for a device not listed
cfg.pool_node_size       = 1024;                       // the page granularity, in bytes
```

A pool serves any size as a contiguous run of nodes. Device-kind containers come from their device's pool; others are
allocated directly. When the pool is full or fragmented the allocation falls back to the OS, and the fallback is counted. A
device with no configured size has a pool of zero bytes: every allocation falls back and is counted.

`Model::memory_report()` is a snapshot between runs: per context device, `pool_bytes`, `pool_bytes_in_use` (= nodes
outstanding × node size), `os_allocations` (cumulative) and `os_bytes_in_use`.

**The rule:** `pool_bytes_in_use` is the sum of the live containers' capacities, each rounded up to whole nodes. Counted
containers include the launch arrays: each kernel component's argument array, the group's pointer array and geometry array,
and the ticket state in ticket mode. Each is far below one node, so each takes one: **a bare component costs three nodes, a
fused group of n stages costs n + 2** (plus one with a boundary), in addition to the variables. The tests assert these numbers
exactly (`support.hpp`: `pool_node_bytes`, `pool_bytes_of`, `solo_launch_bytes`). There is no arena; frees are ordered by the
events registered on the container.

---

## 8. Not supported now

- **Chunked systems, slices and leading ranges.** Not ported to this data model. The old implementation is kept as dead code
  in `include/naga/cospan/chunked_system.hpp` (not included by `cospan.hpp`, not built) and in git history (`c8553f6`);
  a redesign is next. A connect has no `Slice` argument.
- **Merging two materialized variables.** Throws at the call (2.4); deferred until a real use appears.
- **`add_subsystem` outside setup.** Setup-only for now (and the marks).
- Ties of ties, rank-3 ragged, a transformation alias, co-residency guarantees, persistent threads and vendor grid barriers.
- `ParallelSystem` and `FusedSystem` (concurrency comes from data dependencies; fusing is `fuse_components`).

---

## Departures from the design review

Section 11 is the design; where the code differs, this is where and why.

1. **Own-writes (11.3): tried, then removed.** SlabKit `e86ae9f` made an outside writer also wait on the container's own
   writes, because a resize filled new elements asynchronously and a kernel could run before the fill. The own-writes list is
   gone (SlabKit `4f57086`): allocations are uninitialized, nothing fills, and the copy of a reallocation goes through the
   variable's resize-event slot (a kernel that writes the variable waits on it). Reason: one read list and one write list is
   the design; the fill was the only thing that needed more.
2. **No collapse (11.3, "collapse on access"): retired in favor of status pruning at assignment at 8.** The design collapses a
   list with more than one event into one event with an empty `single_task`. As built, lists are returned whole and a
   registration that makes a list longer than 8 drops the completed events (SlabKit `4f57086`; `7655b84` used 64). Reason: with the collapse, the histogram test (a hook reading a value an earlier kernel wrote)
   read stale values in about half the runs on the OpenMP backend and the full suite showed zero shapes and heap corruption;
   a join task that was not empty failed the same way. Waiting on the kernel's event directly made it pass, so the symptom is a
   dependency that was not honored. **The cause is not known.**
3. **`view_input` in pre_forward only (11.4).** The design gives a component a protected method to view its own inputs; as
   built it throws outside `pre_forward()` because it needs the runtime that precompute received.
4. **Boundaries in plain groups (11.6).** The design is silent; as built a boundary is rejected at compile time in a group of
   plain bodies, because a plain body has no work-group (no leader, no group barrier, no collective wait). Boundary placement
   (not first, not last, not adjacent) is also a compile-time check; the design does not say when it fails.
5. **Config cache (11.6).** The design keeps one dynamic-function *definition* per component type (a function-local static).
   As built that holds, and the *configuration* of each distinct stage sequence is cached process-wide and never destroyed
   (keyed by rank and the definitions' names), because a configuration must outlive every kernel and a transient group can be
   deleted before its kernel is compiled.
6. **Ticket counters (11.6).** The design allows a fill or a forward-number comparison to reset the counters; as built it is a
   container fill before every launch, with the launch registered as an outside writer of the ticket state.
7. **Merging two materialized variables (11.2).** The design defers it; as built it throws at the call.
8. **Phase-bound calls (11.0).** `add_subsystem` and the marks are still setup-only; every other structural call is legal at
   any time. `add_subsystem` may follow the "anywhere" principle later.
9. **`override_run_policy` (11.2).** As built it also revives a gone child that is not made transient, and is not logged for
   replay (rearming an ancestor rebuilds the slot with its declared policy).
10. **Cycles (11.2).** `update_input` on a cycle variable after the first forward re-initializes it at the next forward; the
    replay of a connect carries the cycle mark and initializer. A declaration (`init_input`) on an already-materialized free
    input marks it stale; the replay of one does not.
11. **Tied-output pushes (11.3).** A tie whose member children are all skipped or deleted does not push that forward.
12. **Handles (11.4).** `get_input` / `get_output` return `InputHandle` / `OutputHandle` with `view()` (dense) and
    `ragged_view()` (rank 2), since a dense and a ragged view cannot share one return type. The `view_*` and `update_input`
    shorthands are kept.
13. **Host-component exceptions (11.4).** The design says the original exception comes out of `wait()`. As built `wait()` is
    `wait_and_throw` on both queues and the tests assert that it throws and that `forward()` does not; the exception's type and
    text are not asserted.
14. **Waits on a non-ticket launch (11.6).** As designed, undefined with no checks; no debug assert was added.
15. **Local-memory request (11.6).** Requests are rounded up to 16 bytes, and the check against the device limit names the
    stages that asked for local memory.
16. **Pool accounting.** The argument, pointer, geometry and ticket arrays are pool containers like any other, so the exact
    pool numbers include them (section 7).
17. **Open items (11.7).** The fork has the stage-index counter and the cross-translation-unit linking fix
    (`f815ee1e`); the empty-`single_task` question is moot (no collapse). The gemver `onepass` q bug on OpenMP is still open.
