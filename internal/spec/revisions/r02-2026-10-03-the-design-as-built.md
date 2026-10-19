# CoSpan specification — r02

**What this is.** The CoSpan design as built on branch `rework-2026-10-03`. It describes what the code implements and
nothing else. r01 (`r01-2026-09-28-the-design-as-agreed.md`) is superseded and kept as history. Authority for the
decisions behind this text: `internal/design-review-2026-10-03.md` (sections 1 to 10; where sections 6 and 9 conflict with 10,
section 10 wins) and D144 in `internal/phase1-decisions.md`. Every API name here is the name in `include/naga/cospan/`.

Notation: `cs::` is `naga::cospan`, `sk::` is `naga::slabkit` (SlabKit owns the contracts of `sk::context`, `sk::queue`,
`sk::unordered_queue`, `sk::usm_vector` and `sk::DevicePool`).

---

## 1. Vocabulary

**Variable.** One piece of model data: everything joined by `connect`, `promote` or a tie. A variable has a dtype, a rank,
a shape definition, an actual shape, an initializer, event lists, and per device a weak reference to that device's data.

**Instance.** Every input, output and alias is an instance, owned by exactly one component or System through a
`shared_ptr`. Nothing outside the owner holds an instance strongly. There is no global record: no roots, no shadows, no
ledgers. An instance holds:

- the variable's shared metadata;
- its own device and storage kind (a device is never on the metadata);
- a shared pointer to the data of its device;
- its connect flags (can it be a source, can it be a target);
- for a port, the wrapper its component's kernel reads (`cs::Input`, `cs::Output`, `cs::RaggedInput`, `cs::RaggedOutput`).

An instance has one of four roles: port (a component's input or output), promotion, tie alias (both owned by a System), or
host view.

**Shared metadata.** One object per variable, held by `shared_ptr` by every instance of the variable. It holds dtype, rank,
the shape definition, the actual shape, the initializer, the stale flag, the event lists, weak pointers to the instances
that share it (used by merges and to answer where the variable lives) and weak pointers to each device's data. A ragged
variable uses a second metadata type on the same idea.

**Device data.** The memory of a variable on one device: a SlabKit container. It lives while any instance on that device
holds it. The metadata's entry for it is weak.

**Lifetime by ownership.** The metadata lives while any instance holding it exists. A device's data lives while any
instance on that device holds it. "Initialized" means some device has live data; there is no flag for it. A rebuilt child
that is wired back in receives the surviving metadata, with its shape known and its data already initialized. A variable
that nothing holds is gone. Promote a variable (a promotion's alias instance holds the data) to keep it past its producer.

---

## 2. Setup

### 2.1 Two passes

`Model::setup()` runs once.

1. **Setup:** the root System's `setup()`, recursive through `add_subsystem`. Each `add_subsystem` builds the child and runs
   its `setup()` before it returns. Nothing is passed down: no context, no queue, no device.
2. **Finalize:** collects the marks (design variables, constraints, objectives) by absolute path, as weak pointers to the
   instances. Nothing else.

Setup never allocates, never submits, never compiles. Every structural call (`add_subsystem`, `connect`, `promote`,
`tie_inputs`, `tie_outputs`, `shape_hook`, `init_input`, `depends_on`, the marks) is legal inside `setup()` only, and every
rule is checked in its own call, at the call.

### 2.2 add_subsystem

`add_subsystem` is the factory. It takes optional properties, an optional device, and a descriptor:

```cpp
add_subsystem(cs::subsystem_desc<Axpb>("a", 2.0f, 1.0f));                                  // eternal, default device
add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<Setup>("s")); // properties
add_subsystem(dev, cs::subsystem_desc<Axpb>("b", 1.0f, 0.0f));                             // device
add_subsystem(props, dev, cs::subsystem_desc<T>("t", args...));                            // both: properties first
```

`cs::subsystem_desc<T>(name, args...)` carries the name and the constructor arguments (decayed copies, so a slot can build the
child again). It carries no device, and constructors never take one. `T` is a `Component` or a user-defined `System`.

Properties (`cs::property_list`, each at most once):

| Property | Meaning |
|---|---|
| `property::run_n{n}` | the child takes its first `n` turns (`n >= 1`), then is deleted |
| `property::do_once{}` | `run_n{1}` |
| `property::transient{}` | the child exists for one forward of its parent: built at the start, deleted after its turn |

`run_n` and `do_once` exclude each other, and `transient` excludes both. No property means eternal.

Each child lives in a **slot** in its parent, which keeps its name, its factory, the device it was given and its run policy.
The parent owns the child's lifecycle. A slot and its declarations outlive the child. `get_subsystem(path)` walks a dotted
path and returns a pointer: null when the slot exists but its child is deleted, and a throw when the name was never declared.

### 2.3 Devices

Every input, output and alias takes AdaptiveCpp's default device when declared. `set_device` places a subtree:

- on a component, it sets the device of each of its instances;
- on a System, it sets the device of each of its own aliases, then calls `set_device` on each live child, recursively.

`add_subsystem` builds the child, runs its `setup()`, then calls `set_device` with the device it was given, and then with the
parent's own forced device if the parent was placed. **The outer call wins**: a device given to `add_subsystem` overrides
every placement inside that subtree. An author's placements are recommendations; a user overrides them by passing a device
and keeps them by passing none.

- A promotion copies its member's device when made. An outer `set_device` overrides it later.
- `tie_outputs` requires all members on one device, checked in the call. This stays sound under overrides, since an override
  moves a whole subtree to one device.
- `tie_inputs` takes an optional device: where the tie's initialization runs (then copied to the other devices) and where its
  data stays while the tie's System exists.
- `connect` takes no device.

### 2.4 Wiring

| Call | Effect | Whose metadata wins |
|---|---|---|
| `connect(src, dst)` | feeds `dst` from `src` | the source's |
| `promote<DT, Rank>(member, name)` | a peer alias in this namespace joining the member's variable | the member's |
| `tie_inputs<DT, Rank>({members}, name, shape, device)` | a new variable; the members' variables merge into it | the alias's |
| `tie_outputs<DT, Rank>({members}, name, shape, policy)` | a new variable (a source); the members merge into it | the alias's |

Paths are dotted, relative to the calling System: `"a.y"` is output `y` of child `a`; a single segment is this System's own
alias. Ports of the children of a helper-built group are reached the same way (`"grp.a.x"`).

Each call checks its own rules before anything changes:

- **Rank** must match (`promote` and ties also check the dtype `DT`).
- **Shape definitions** merge (section 7.2): unspecified absorbs specified, and two specified values that disagree are an error.
- **Connect flags.** A connected input can no longer be a target, a tied input can no longer be a target, a hooked input can no
  longer be a target (see 7.5). A tied input cannot be connected, and a connected one cannot be tied. Connecting into a tie
  alias is connecting into any instance.
- **tie_outputs policy.** `TieOutputsPolicy::require_atomic` (the default) requires atomic member dtypes. `allow_plain` says the
  writers are disjoint. A parallel fused component adds no check of its own.
- **Cycle warning.** `connect` warns (to `std::cerr`, once per pair) when the source's child was added after the target's
  child: the consumer then reads the producer's previous value. Cycles are expressed by registration order.

After the rules pass, the call re-points the losing metadata's live instances at the winner and moves their weak entries.
Promotions take no shape. A merge never touches stale flags, events or data, except in one case: when exactly one side
already holds data (a rebuilt child rejoining live data), that side's metadata survives, carrying the merged definition.

### 2.5 Initial values

`init_input(path, init)` (on a component for its own input by name, on a System by path) declares the initial value of a free
input's variable. **The last declaration wins**: a parent's `setup()` runs after its children's, so a tie's, promotion's or
System's initializer replaces the members'. Tied members and promotions may carry their own initializers. An initializer on
an input that ends up with a source is ignored. Naming an output throws; outputs take no initializers.

Initializer factories (`init.hpp`):

| Factory | Meaning |
|---|---|
| `cs::init::fill(v)` | every element is `v` |
| `cs::init::zero()` | every element is 0 |
| `cs::init::from_host(vec)` | element `i` is `vec[i]` |
| `cs::init::custom<DT, R>(fn)` | `fn(unordered_queue&, device, deps, Initializer<DT, R>)` enqueues its own work and returns its event |
| `cs::init::custom_host<DT, R>(fn)` | `fn(HostInit<DT, R>&)` fills a host buffer that is then copied to the device |

Each form takes an optional `cs::init::Shape` (`{n, m}`, or `cs::init::ragged(n, d1)`) that supplies a free input's
unspecified dimensions. Every initializer returns the event of the work it enqueued; the variable records it as a write.

### 2.6 Marks and the eternal-subsystem rule

`add_design_var`, `add_constraint` and `add_objective` mark a variable by path. The last mark of a kind on a variable wins.
Finalize collects each mark by absolute path as `Mark{path, inst, rec}` (`inst` is a `weak_ptr`). **Marks live only on
subsystems that never go out of scope:** finalize walks the slots from the root to a mark's declaring subsystem and throws if
any slot on the way has a run policy (`run_n`, `do_once`, `transient`).

---

## 3. Execution

### 3.1 The forward

`Model::forward()` enqueues one forward of the root and returns. It does not wait. A forward is three framework-owned
steps, for components and Systems alike. Each step is not overridable and calls a user hook that is empty by default.

| Step | System | Component |
|---|---|---|
| precompute | rebuild transient children (allocates nothing), then `pre_forward()` | materialize, `pre_forward()`, materialize (4.2) |
| compute | call each live child's forward in declaration order, and nothing else | `compute_forward()` describes the launch, the component submits it |
| postcompute | push tied outputs to other devices, then `post_forward()` | push to the variable's other placements, then `post_forward()` |

### 3.2 The System loop and run policies

Compute calls each live child's precompute, compute and postcompute in declaration order. A child whose counter is 0 is
skipped. After a child's turn:

- a `run_n` counter decrements, and at 0 the child is deleted;
- a transient child is deleted.

**Deletion waits on that child's events only** (`Subsystem::events()`), never on the queue, then frees the child. Its memory
is free before later children allocate, while independent work already submitted keeps running. The child's variables
survive through any other instance that holds them (a connected consumer, an alias, a host view). Add subsystems that can go
out of scope as late as possible in `setup()`: deleting one makes its parent wait on that subsystem's events.

A transient child is rebuilt at the start of each forward, in the parent's precompute: the same build as `add_subsystem`
(factory, then `setup()`, then `set_device`) followed by the replay of the parent's wiring that names it.

A child's event getter: a component returns its launch event; a System returns its children's events plus its own push
events.

### 3.3 rearm

`rearm(name)` is a protected `System` verb, legal inside `pre_forward()` only. It deletes the child if it is live, builds it
again (factory, `setup()`, device, replay of the wiring that names it), and gives it its full run policy again. Use it for
triggers only the author knows, such as re-running a `do_once` child from the previous run's results:

```cpp
void pre_forward() override { if (changed_) rearm("setup_stage"); }
```

A rebuilt child rejoins the same variables. An init declared in `setup()` applies only when the variable is live nowhere, so a
rearm does not re-initialize a live variable. A rebuilt consumer whose producer is gone, with nothing holding the data, is a
user error: promote what must stay durable.

### 3.4 depends_on

`depends_on("b", "a")` makes child `b` wait on child `a` for one forward at a time. `a` must be added before `b`. The parent
runs `a`'s forward, collects its events, and hands them to `b`; if `b` is a System, every component inside adds them to its
submissions. A duplicate edge is ignored.

Execution order is data dependencies only. Children are *called* in declaration order, but kernels are ordered by events, so
two writers of a tied output run concurrently. A zeroing producer and an accumulator on one `tie_outputs` need `depends_on`.

### 3.5 Events

Each variable keeps its events in **two generations**: this forward's and the last forward's, each as writes and reads.

- A **writer** waits on the last generation's writers and readers, and never on another writer of its own forward. Tied
  outputs (atomic or disjoint) therefore run concurrently.
- A **reader** waits on this forward's writers if there are any, else on the last generation's. A reader that runs before any
  writer of its forward (a cycle) reads the previous value.
- The generation rolls over when a writer asks in a new forward (`Runtime::forward`, counted by the Model).

A component's submission depends on its inputs' write events, its outputs' write and read dependencies, the events of its own
device's containers, and any `depends_on` events. After submitting, it adds its event to each output's writes and each input's
reads, and registers it on the data it touched, so a later free or write waits for it.

### 3.6 Queues and waits

The Model owns one `sk::context`, one `sk::queue` (containers) and one `sk::unordered_queue` (kernels and host tasks, ordered
by the caller's events only). CoSpan never relies on in-order execution. There is no queue-wide wait anywhere in the
framework: repeated forwards stream with no artificial synchronization.

`Model::wait()` calls `wait_and_throw()` on both queues. It is optional: use it for "everything I submitted is done" and to
surface asynchronous errors from work nobody synced. `~Model` does not wait. When the model goes, every device copy is freed,
each after the events registered on it, so destroying a model with work in flight is safe.

---

## 4. Components

### 4.1 Kinds, by body type

A component derives from `cs::Component<Body>`. One kernel body per component. The body's call-operator signature fixes the
kind and the kernel rank at compile time; there is no run-time branching on rank.

| Kind | Body signature | Notes |
|---|---|---|
| plain | `void operator()(sycl::id<R>, Inputs, Outputs) const` | a range launch, no work-groups |
| group | `void operator()(cs::GroupItem<R>, Inputs, Outputs [, Scratch]) const` | work-groups, `Local` memory, barriers; `Scratch` is the fourth argument |
| custom-op | `void operator()(cs::CustomOp&, Inputs, Outputs) const` | native backend work through an interop handle |
| host | `void operator()(cs::Host, Inputs, Outputs) const` | runs as a host task over host-kind memory |

`Inputs` and `Outputs` are structs of wrappers (`cs::Input<DT, Rank>`, `cs::Output<DT, Rank>`, and the ragged wrappers).
Custom-op and host components cannot be fused; the group shape is defined by the component, never by a fuse.

### 4.2 The forward of a component

**Precompute**, in this order:

1. every distinct input variable is materialized (the enclosing Systems' shape hooks have already run);
2. the user's `pre_forward()` runs: `shape_of` and `set_shape` are legal here only (section 7);
3. every distinct output variable is materialized (a dimension that is still unknown is an error);
4. each member wrapper is pointed at its data.

Materialize is called on each variable's shared metadata and is a no-op when it is live and current. Otherwise it allocates on
every device that has an instance, then either runs the initializer (its event goes on the variable as a write) or copies
from a live device. A variable that is stale (see 6.2) is initialized in place.

**Compute.** `compute_forward()` returns a `std::unique_ptr<cs::LaunchDescription>`: the complete launch, configuration plus
the user's setup. The component submits it itself on the unordered queue and records its event on its variables. Build the
description with `create_launch_config(range, in, out, body)` (plain, group), `create_launch_config(range, wg, in, out, body)`
(group with an explicit work-group), `create_launch_config(range, wg, in, out, scratch, body)` (group with `Scratch`), or
`create_launch_config(in, out, body)` (custom-op, host).

**Postcompute.** The pushes to the variable's other placements, then `post_forward()`.

### 4.3 The JIT configuration

The JIT launch configuration is hidden from users. It is one per component body type, a function-local static built once at
the first submission (`jit_config<KB>()`), so it outlives the submission that the runtime reads on a worker thread. Nothing
is compiled in `setup()`.

### 4.4 Hooks

```cpp
class Scale : public cs::Component<ScaleBody> {
    cs::Input<DType::f32, 1>  x_;
    cs::Output<DType::f32, 1> y_;
public:
    void setup() override {                                // declare I/O: allocates nothing
        declare_input(x_, "x", cs::unspecified);
        declare_output(y_, "y", cs::unspecified);
    }
    void pre_forward() override { set_shape(y_, shape_of(x_)); }   // shapes: this hook only
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(x_.shape(), {x_}, {y_}, {});
    }
    void post_forward() override {}                         // after the pushes
};
```

`declare_input` and `declare_output` take one `DimSpec` per dimension: an integer (specified) or `cs::unspecified`. Array forms
take `cs::unspecified_shape<R>()`. Ragged declarations are in 7.3.

---

## 5. Fused components

A fused component is semantically a component: one kernel, one submission. Its stages are components whose bodies run as the
stages of one outer kernel, the wave.

- **Precompute** runs each stage's precompute.
- **Compute** collects each stage's launch (the stage's own submission is bypassed), checks that the stages' group shapes
  agree (only knowable here, since a component may choose its shape at run time), packs the stage records and byte streams,
  submits one launch, and gives its event to every stage. Each stage's variables then have their event lists exactly as if it
  had submitted alone. The packing buffers are rewritten each forward; that rewrite depends on the previous wave's event.
- **Postcompute** runs each stage's postcompute.

The call sequence (the JIT configuration) is built at submission and cached once per distinct sequence of stage components.

### 5.1 Strategies

`cs::fuse_strategy::seq`: the stages run one after the other inside each work-group. There are no framework barriers: a stage
that reads, across work-items, what an earlier stage wrote calls its own group barrier first. Across groups nothing is ordered.
Stage ranges may differ.

`cs::fuse_strategy::par`: the stages own disjoint slices of the work-groups and run unordered. A connect between them is
rejected in the call.

All stages are either all group components or all plain components, and share one kernel rank (both enforced at compile time).
Custom-op and host components cannot be stages. Ragged ports are refused at the end of setup. Local memory is summed across
stages for `seq` (each scratch stage gets its own slice) and maxed for `par`, and is checked against `local_mem_size` each
forward.

### 5.2 Helpers

```cpp
add_subsystem(dev, cs::fuse_components<cs::fuse_strategy::seq>("grp",
              cs::connects_desc({{"a.y", "b.x"}}),
              cs::subsystem_desc<A>("a"), cs::subsystem_desc<B>("b")));
add_subsystem(cs::system_from("grp", cs::connects_desc({{"a.y", "b.x"}}),     // a plain System of the same children
              cs::subsystem_desc<A>("a"), cs::subsystem_desc<B>("b")));
```

`fuse_components<S>(name, [connects_desc(...)], descs...)` and `system_from(name, [connects_desc(...)], descs...)` return a
descriptor, so the result goes straight into `add_subsystem([props], [device], ...)`. The device comes from that call, and
`set_device` applies to the whole fused component, so one device per fused component holds by construction.

`connects_desc({{src, dst}, ...})` is the same `connect`, with the same rules, scoped to the group. A connect inside a `seq`
fused component marks the intermediate variable internal.

The helpers are for the simple case: no per-child run policies, no promotions, no device splits. For those, write your own
`System`. Ties inside a fused component follow the ordinary `tie_outputs` policy and nothing else.

---

## 6. Host access

### 6.1 Views

A host view is an instance of the variable on a host placement. It holds the variable's metadata and its host memory (a SlabKit
container) by `shared_ptr`; the metadata holds the host memory weakly.

```cpp
cs::HostView<DType::f32, 1> y = model.view_output<DType::f32, 1>("b.y");   // after a forward
y.sync();                                                                   // before reading
float first = y[0];
```

- `Model::view_input`, `view_output` (dense, `HostView<DT, Rank>`) and `view_ragged_input`, `view_ragged_output`
  (`HostRagged<DT, Rank>`) return a view. A ragged view works on a dense variable too: its blocks are the rows.
- A view can be obtained **only after a run**; before the first forward the call throws. A view starts as a copy of a live
  placement and never initializes anything.
- While any view exists, the host copy is one of the variable's live placements: **every forward pushes one copy to it** (each
  live view costs one copy per forward). When the last view goes away, the host memory is freed. Keep a view to avoid
  reallocating it.
- The host memory is associated with the device that has the most instances of the variable.
- **`sync()`** waits on the variable's write events and on the host copy's own events. Reading without syncing, or while work is
  running, is the user's responsibility.
- **A view outlives the model.** It keeps the metadata and its host memory, and the container's handle keeps the queue alive.
- `HostView` has `sync()`, `size()`, `data()`, `span()`, `operator[]`, `shape()` and `operator()` with one index per rank.
  `HostRagged` has `sync()`, `size()`, `data()`, `span()`, `operator[]`, `n()`, `block_offsets()` and `block(i)`.

### 6.2 update_input

`Model::update_input(path, init)` is the only host write. It assigns an initializer to the input's variable and marks the
variable stale. It queues no work.

- A variable that is not materialized is always stale.
- At the next run, the first thing that touches the variable acts on it: not materialized, it materializes and initializes;
  materialized but stale, it initializes in place with no allocation. Either way the flag clears.
- Initializing a live variable in place is a write: it waits on the variable's read events and records its event as a write.
- Several updates before a run collapse to one (the last wins), and a variable nothing touches is never initialized.
- **A host view synced after `update_input` and before the next run still shows the old values.**
- On an input whose variable has a source, `update_input` warns and is never applied. Naming an output throws
  `std::invalid_argument`.

The `init_input` calls in `setup()` keep their meaning (section 2.5).

### 6.3 memory_report

`Model::memory_report()` is a snapshot between runs. `devices` has an entry for every device of the context, with
`pool_bytes`, `pool_bytes_in_use`, `os_allocations` (cumulative) and `os_bytes_in_use` (the bytes of those still live).

---

## 7. Shapes

### 7.1 Definition and actual shape

Each variable's metadata holds a shape **definition**, fixed once setup ends, and the **actual** shape, which changes at run
time. Per dimension the definition is one of:

- **specified:** a value, set at declaration, never changes;
- **unspecified:** no constraint, supplied from outside;
- **hook:** owned by exactly one registered shape hook.

A shape that is needed and not yet known is an error at the forward (shape errors are reactive; setup checks wiring rules only).

### 7.2 Declarations and merging

A component declaration says only specified or unspecified per dimension. It is the component's contract: specified forces
that dimension on anything joined to it; unspecified places no constraint. There are no derived dimension specs.

```cpp
declare_input(x_, "x", cs::unspecified);          // rank 1, any length
declare_input(m_, "m", 3, cs::unspecified);       // rank 2: 3 rows, any columns
declare_output(y_, "y", 4);                       // rank 1, exactly 4
```

`connect`, `tie_inputs`, `tie_outputs` and `promote` merge the definitions of all participants: unspecified absorbs specified,
and two specified values that disagree are an error at the call. A tie declares its own shape with `cs::shape(...)`: one
`DimSpec` per dimension, `cs::shape()` for every dimension unspecified. Promotions take no shape.

### 7.3 Ragged variables

Ragged is **rank 2 only**: a dense leading dimension N plus one per-index dimension. A ragged declaration is one of:

```cpp
declare_input(r_, "r");                          // both unspecified
declare_input(r_, "r", 8);                       // N = 8, counts unspecified
declare_input(r_, "r", cs::counts({3, 0, 5}));   // host counts; N = 3
declare_input(r_, "r", cs::offsets({0, 3, 3, 8}));  // the same as N + 1 offsets; N = 3
```

`declare_output` takes the same forms. A ragged tie uses `cs::shape()`, `cs::shape(n)`, `cs::shape(cs::counts(v))` or
`cs::shape(cs::offsets(v))`. Kernels read `cs::RaggedInput` and `cs::RaggedOutput` (`block(i)` gives a dense `cs::Block`);
a ragged wrapper also reads a dense port.

### 7.4 Run-time shape changes

Every run-time shape change is checked against the definition. A change may touch only the dimensions it is allowed to, a
specified dimension can never come out different, and two sets of one dimension in one forward must agree.

- **Inputs are never reshaped.** An input's shape comes from its definition, its initializer's shape, its connection's source,
  or a hook.
- **Outputs are reshaped only by their producing component's `pre_forward()`**, within the definition. An output decides the
  shape for everything downstream. Nothing outside a component may set an output's shape.

In `pre_forward()`:

```cpp
sycl::range<1> shape_of(x_)                             // the shape of one of my ports now
void set_shape(y_, sycl::range<1>{n})                   // an output's shape
cs::RaggedShape shape_of(r_)                            // ragged: n, total, counts (host or device)
void set_shape(r_out_, rshape)                          // from a RaggedShape
void set_shape(r_out_, n, counts)                       // from N and host counts
```

`shape_of` and `set_shape` are legal inside `pre_forward()` only; `set_shape` elsewhere throws "legal inside pre_forward() only".
An input's shape is known in `pre_forward()`; an output's only after an earlier forward or when its definition fixes it.

A component may read values to size its outputs on its own, for example through a host view kept as a member. There is no
framework helper for this.

**Tied outputs.** A component may not reshape an output that is tied (an error). The members' declared dimensions merge and are
checked in `tie_outputs`. Any dimension still unspecified on a `tie_outputs` alias must be set by a hook on that alias at the
System level: the one output-side hook.

### 7.5 Shape hooks

A hook is a System's contract for the shape of a target.

```cpp
shape_hook(target_path, <one descriptor per dimension>);
```

| Descriptor | Dimension comes from |
|---|---|
| `cs::dim::as_is()` | left alone |
| `cs::dim::from_shape(path, d)` | dimension `d` of the shape of variable `path` |
| `cs::dim::from_value(path, index)` | the value of element `index` (flat) of the integer variable `path` |
| `cs::dim::custom(fn)` | `fn(cs::ShapeHookCtx&) -> std::size_t` |
| `cs::dim::fixed_at_run(v)` | a value decided at setup: it just makes the dimension specified |
| `cs::ragged::counts_from(path)` | a ragged target: the per-index counts are the values of the integer variable `path` (rank 1 `[N]`, or column 0 of a rank 2 `[N, k]`) |
| `cs::ragged::offsets_from(path)` | a ragged target: the N + 1 block offsets in the integer variable `path` (rank 1) |

A ragged target takes one ragged descriptor for both dimensions. `cs::ShapeHookCtx` (given to `custom`) has `shape_of(path)` and
`value_of(path, index)`, reading by path in the namespace of the declaring System.

```cpp
void setup() override {
    add_subsystem(cs::subsystem_desc<Producer>("p"));
    add_subsystem(cs::subsystem_desc<Scale>("c"));
    shape_hook("c.x", cs::dim::from_shape("p.y", 0));              // c.x is as long as p.y
}
```

```cpp
// rank 2: rows from a value, columns fixed at setup
shape_hook("m.a", cs::dim::from_value("cfg.rows", 0), cs::dim::fixed_at_run(16));

// a ragged input sized by a counts variable
shape_hook("r.x", cs::ragged::counts_from("counts.n"));

// the one output-side hook: an unspecified dimension of a tie_outputs alias
tie_outputs<DType::atomic_f32, 1>({"a.acc", "b.acc"}, "acc");
shape_hook("acc", cs::dim::from_value("cfg.bins", 0));
```

**Rules.**

- **Targets** are an input, an input tie alias, or an output tie alias. Sources may be any variable, read only.
- **Hooked dimensions** are those that are not `as_is`. Each must be unspecified in the definition and owned by one hook only.
  A second hook on a dimension, or a hook on a specified dimension, is a setup error ("only one hook owns it").
- **Hooks and connections exclude each other.** A hook clears the input's can-be-target flag, so a later `connect` or tie into
  that input fails. A connected or tied input cannot take a hook. Tied members cannot take hooks; their tie alias can.
- **Run point.** A hook is a lightweight type-erased step, not a component. It runs on the main thread immediately before the
  first child that has an instance of the target. Every source's producer must be in an earlier slot; otherwise it is a setup
  error naming the hook, the source and the target (checked when the declaring System's setup ends). Free inputs and aliases
  are fine as sources anywhere. A hook declared after its target's child still runs before that child.
- **The work.** `from_value` reads of one hook are batched: each is a small copy to a host buffer, and the host waits once.
  `counts_from` and `offsets_from` build a device column (counts, offsets, total) on the source's device with a parallel
  exclusive scan; only the total and a validity flag come back to the host. A negative count, or offsets that do not start at
  0 or that decrease, is an error.
- **A gone source.** When a hook's named source no longer exists (its child was deleted) and the target's hooked dimensions are
  already known, the hook does nothing and the last shape stands. "... does not exist right now" is thrown only when the shape
  was never known.
- **Rebuilds.** A hook is replayed with the rest of the wiring when a child it names is rebuilt.

### 7.6 Removed from the shape model

Value-driven dimension specs (`FromInputDim`, `FromInput`, `FromInputValues`, `FromInputArray`), the driving-value accessor,
producer-side shape work and the shape-driving flag, tie shape rules evaluated per forward, tie shape constraints after a
connect, and rank 3 ragged.

---

## 8. Memory

Each device gets one SlabKit pool, sized by the user up front in `ModelConfig`:

```cpp
cs::ModelConfig cfg;                                   // the default-selector device
cfg.pool_bytes[dev]      = 512u << 20;                 // per device
cfg.default_pool_bytes   = 0;                          // for a device not listed
cfg.pool_node_size       = 1024;                       // the page granularity, in bytes
```

A pool serves any allocation size as a contiguous run of nodes. Device-kind data comes from the device's pool; host-kind and
other data is allocated directly. When the pool is full or too fragmented, the allocation falls back to the OS, and the
fallback is counted.

**A device without a pool is a zero-byte pool.** Every context device gets a pool, sized by its entry or by
`default_pool_bytes`, possibly 0. A zero-byte pool serves nothing, and every allocation falls back to the OS and is counted.
`memory_report()` (6.3) covers every device, so the fallback counters are always visible.

There is no arena. Frees are ordered by the events registered on the data; the framework never waits on a queue to free.

---

## 9. Not supported now

**Chunked systems, slices and leading ranges are not supported.** They were not ported to this data model. The old
implementation is kept as dead code in `include/naga/cospan/chunked_system.hpp` (it is not included by `cospan.hpp`, is not in
the build, and stops the compiler with `#error`) and in git history (`c8553f6`), pending a redesign. Its tests stay in the
repository but out of the tests CMake list. A connect has no `Slice` argument.

Also not present: `ParallelSystem` and `FusedSystem` (concurrency comes from data dependencies; fusing is a component built by
`fuse_components`), a transformation alias (a System-level alias that reinterprets rank), and a tie of a tie (open).
