# Migrating to the 2026-10-03 rework

From the API at `c8553f6` to the final API of the rework (branch `rework-2026-10-03`). Authority: `docs/design-review-2026-10-03.md`
(sections 1-10 cited below as "section N"; section 10 wins over 6 and 9). The design as built is
`docs/spec/revisions/r02-2026-10-03-the-design-as-built.md`. Everything here was checked against the headers.

## Adding subsystems

A child is added from a **descriptor**: the name and the constructor arguments, nothing else. Constructors never take a
device; the device is an `add_subsystem` argument (section 2, section 4).

```cpp
// before
add_subsystem<Axpb>("mid", cs::property_list{cs::property::device{dev}}, 2.0f, 1.0f);
// after
add_subsystem(dev, cs::subsystem_desc<Axpb>("mid", 2.0f, 1.0f));
add_subsystem(cs::subsystem_desc<Axpb>("a", 1.0f, 0.0f));                       // no device, no properties
add_subsystem(cs::property_list{cs::property::do_once{}}, cs::subsystem_desc<Setup>("s"));
add_subsystem(props, dev, cs::subsystem_desc<T>("t", args...));                  // both: properties first, then device
```

Properties are `run_n{n}`, `do_once{}` and `transient{}` only (`transient` cannot be combined with the other two).
`property::device`, `property::parallel` and `property::fused` are gone, and so is the helper form
`add_subsystem(name, props, child<T>(...)...)` with `child<T>`.

`FusedSystem` and `ParallelSystem` are gone. A fused group is a component, built by a helper that returns a descriptor:

```cpp
// before (illustrative: the helper form with a fused/parallel flavor property, children given as child<T>)
add_subsystem("grp", cs::property_list{cs::property::fused{...}}, cs::child<A>("a"), cs::child<B>("b"));
// after
add_subsystem(cs::fuse_components<cs::fuse_strategy::seq>("grp",                 // or ::par
              cs::connects_desc({{"a.y", "b.x"}}),
              cs::subsystem_desc<A>("a"), cs::subsystem_desc<B>("b")));
add_subsystem(cs::system_from("grp", cs::connects_desc({{"a.y", "b.x"}}),        // a plain System of the same children
              cs::subsystem_desc<A>("a"), cs::subsystem_desc<B>("b")));
```

The helpers are for the simple case (no per-child run policies, no promotions). Anything else is a user `System`.
Custom-op and host bodies cannot be fused; ragged ports are refused (section 4).

## Wiring

```cpp
// before                                         // after
connect("a.y", "b.x", cs::Slice{...});            connect("a.y", "b.x");           // slices are not ported
promote<DType::f32, 2>("a.in", "in");             promote<DType::f32, 2>("a.in", "in");  // kind-agnostic: a PEER alias
tie_inputs<DT, 1>({"a.x", "b.x"}, "x", shape);    tie_inputs<DT, 1>({"a.x", "b.x"}, "x", shape, /*device=*/dev);
tie_outputs<DT, 1>({...}, "t", shape, policy);    tie_outputs<DT, 1>({...}, "t", shape, policy);
```

`tie_inputs` takes an optional last argument, the device where the tie's initialization runs and its data stays.
Every wiring rule is checked in its own call, inside `setup()`.

- **Initial values:** `init_input(path, spec)` in `setup()` (component or System). The last declaration wins, with no
  conflict error: a parent's setup runs after its children's, so a tie's, promotion's or System's `init_input` replaces the
  members' own (tied members and promotions may carry initializers). Two members with inits and no tie-level init: the
  later merged member wins. It is ignored when the input
  has a source; naming an output throws. Run-time replacement is `Model::update_input` (below).
- **Marks:** `add_design_var`, `add_constraint`, `add_objective` take no region. A mark is `Mark{path, inst, rec}`.
- **Order:** execution order is **data dependencies only** (section 3). Add the real edge, or
  `depends_on("b", "a")` (`a` must be added before `b`): B's submissions then wait on A's events for one forward.
  Tied writers never wait on each other, so a zeroing producer and an accumulator on one `tie_outputs` need
  `depends_on`. Children are still *called* in declaration order, so a producer whose value is read **on the host** (the
  source of a `shape_hook`) must be declared before the child that has the hook's target (a setup error otherwise).
- **Re-running a child:** `rearm("name")` is legal inside `pre_forward()` only.

```cpp
void pre_forward() override { if (changed_) rearm("setup_stage"); }
```

## Components

`compute_forward()` and `create_launch_config(...)` are unchanged. What went:

- no `compile()` step and nothing compiled in setup; setup allocates and submits nothing (section 2);
- no `get_input` / `get_output`: use the declared members (`x_`, `y_`);
- `reshape(output, range)` is now `set_shape(output, range)`: **outputs only**, legal in `pre_forward()` only, checked against the
  variable's definition. An input's shape is its variable's and is never set; read it with `shape_of(port)`;
- a new `post_forward()` hook runs at the end of postcompute, after the pushes to other placements.

```cpp
void pre_forward() override { set_shape(y_, sycl::range<1>{n_}); }   // or: set_shape(y_, shape_of(x_));
void post_forward() override { /* after the pushes */ }
```

## The Model

```cpp
// before                                              // after
model.init_input("x", cs::init::fill(1.0f));           model.update_input("x", cs::init::fill(1.0f));
model.forward(); model.wait();                         model.forward();               // enqueues; wait() is optional
auto y = model.view_output<DType::f32,1>("y");         auto y = model.view_output<DType::f32,1>("y");   // after a forward
cs::ModelConfig cfg; cfg.default_device = d;           // gone: devices go to add_subsystem
                                                       y.sync();                      // before reading
```

- `forward()` does not wait. `wait()` is optional: it waits on both queues and surfaces asynchronous errors.
- `view_input` / `view_output` / `view_ragged_*` work **only after the first forward** (earlier they throw "only after a
  run"). A view never initializes anything. **Call `sync()` before reading.** Views outlive the model.
  Each live view costs one copy per forward. `HostView`: `sync() size() data() span() operator[] shape() operator()(i,j,k)`.
  `HostRagged`: `n() block_offsets() block(i)` (a span).
- `update_input(path, spec)` replaces `Model::init_input`. It marks the variable stale and queues no work; the next run
  applies it (in place if materialized). Several updates collapse to the last. On a connected input it prints a warning
  and is never applied; on an output it throws `std::invalid_argument`.
- `ModelConfig` has no `default_device`. Fields: `ctx`, `pool_bytes`, `default_pool_bytes`, `pool_node_size`, `async_handler`.
- `memory_report().devices[dev]` has `pool_bytes`, `pool_bytes_in_use`, `os_allocations`, `os_bytes_in_use`
  (no arena fields). Every context device has an entry: a device without a configured size is a zero-byte pool.
- `design_vars()`, `constraints()`, `objectives()` return `Mark{std::string path, std::weak_ptr<Instance> inst, MarkRecord rec}`;
  `rec` is `{bounds, equals, scaling}`. The old per-root `root` pointer is replaced by `inst.lock()->meta` (the shared variable).
- `root().entry_at(path)` is gone; `root().instance_at(path)` and `get_subsystem(path)` remain.

## Initializers

```cpp
// custom: enqueue your own work after `deps` and RETURN its event
cs::init::custom<DType::f32, 1>(cs::init::CustomFn<DType::f32, 1>{
    [](sk::unordered_queue& q, const sycl::device& dev, const std::vector<sycl::event>& deps,
       cs::Initializer<DType::f32, 1> out) {
        return q.parallel_for(dev, deps, sycl::range<1>{out.size()}, [=](sycl::id<1> i) { out[i] = 1.0f; });
    }});
// custom_host: a zeroed host buffer, copied to the device after the function returns
cs::init::custom_host<DType::f32, 1>(std::function<void(cs::HostInit<DType::f32, 1>&)>{
    [](cs::HostInit<DType::f32, 1>& h) { for (std::size_t k = 0; k < h.size(); ++k) h[k] = 1.0f; }});
```

`fill`, `zero`, `from_host` and `custom` all return the event of the work they enqueued. `HostViewMut` is gone.

## Behavior changes

- **Execution order is data dependencies only** (section 3); no queue-wide wait anywhere. Events are kept in two
  generations per variable.
- **Shape errors are reactive:** thrown at the first forward, not at setup (section 10). Setup checks wiring rules only.
- **No write epochs, no host read cache, no arena** (sections 3 and 5). Memory comes from the per-device pools, then the OS.
- **Lifetime:** a deleted child's data lives while any instance holds it (a connected consumer, an alias, a view).
  Instances are owned by their component or System; the shared variable metadata holds only weak pointers (section 1).
  A deleted subsystem's variables cannot be resolved by path ("... was deleted").
- "Initialized" means some device has live data; a stale flag drives re-initialization.
- Deleting a child waits on that child's events only. Add subsystems that can go out of scope as late as possible.
- `~Model` does not wait.
- Diagnostics keep `queue_waits` and `host_tasks` only.

## Removed

`ParallelSystem`, `FusedSystem`, `child<T>` and the helper `add_subsystem` form, `property::device`, `property::parallel`,
`property::fused`, `ChunkedSystem` (dead code, `#error` until redesigned), slices and leading ranges, `connect(.., Slice)`,
`Model::init_input`, `HostViewMut`, `ModelConfig::default_device`, arena report fields, `IOEntry::write_epoch`,
`registered_event_count`, `entry_at`, the host read cache and its counters, `get_input` / `get_output`, `compile()`,
the retire list, setup-time shape checks, `fused_system.hpp`, `wave_rules.hpp`, `alias_eval`, and the diagnostics counters
`frees`, `allocations`, `host_reads`, `host_read_hits`, `ragged_uploads`, `resizes`, `full_passes`, `resolve_shapes` and the chunk window.

## Corrections during implementation (design review, section 9)

- **Initializers:** the last declaration wins; no same-level error.
- **Ties and shapes:** superseded by section 10 (see Shapes below): a tie's shape is a definition merged at the call, not a constraint kept per forward. Promotions take no shape. A hook whose source is gone keeps the last shape.
- **Durability:** a rebuilt consumer whose producer is gone, with nothing holding its data, is a user error: promote what must stay durable.
- **Par fused components** add no tie check of their own: use `tie_outputs` with atomic members, or `allow_plain` for disjoint writers.
- **Pools:** every context device has a pool (0 bytes if unconfigured, then every allocation is an OS allocation, counted); `memory_report().devices` has an entry for every device.
- **Marks** (`add_design_var`, `add_constraint`, `add_objective`) are refused, at `Model::setup`, on a subsystem that can go out of scope: no `run_n`, `do_once` or `transient` slot on the path from the root.

## Shapes (section 10)

A declaration says only *specified* (an integer) or *unspecified* (`cs::unspecified`) per dimension. A shape is supplied from
outside: by a connection, an initializer, the producer's `pre_forward()`, or a System's `shape_hook`. Shape errors are thrown
at the forward; setup checks wiring rules and the merged definitions only.

**No derived dimension specs.** `FromInputDim`, `FromInput`, `FromInputValues`, `FromInputArray` and `alias::from_output_values`
are removed (and `alias::fixed`, `alias::dims` and the other alias shape kinds with them).

Where an output's length was derived from an input's dimension or value, declare it unspecified and let the component set it:

```cpp
declare_input(x_, "x", cs::unspecified);
declare_output(y_, "y", cs::unspecified);
void pre_forward() override { set_shape(y_, shape_of(x_)); }
```

Where an input's length came from another variable's value or shape, a System now declares a hook on that input (below).

**Reading and setting a shape** (in `pre_forward()` only): `shape_of(port)` gives the shape of one of the component's own ports
(`sycl::range<R>`; a `cs::RaggedShape` for a ragged port), and `set_shape(output, shape)` sets an output's shape. Inputs are
never reshaped, and a member of a `tie_outputs` cannot reshape its output. Outside `pre_forward()` both throw "legal inside
pre_forward() only".

**Tie shapes.** `tie_inputs` and `tie_outputs` take `cs::shape(...)`: one specified or unspecified value per dimension, `cs::shape()`
for all unspecified. A tie keeps no shape rule after a connect: the definitions merge at the call (unspecified absorbs
specified; two specified values that differ are a setup error).

**Shape hooks** replace every value-driven dimension. A System declares, in `setup()`:

```cpp
shape_hook("c.x", cs::dim::from_shape("p.y", 0));                         // dimension 0 of p.y
shape_hook("m.a", cs::dim::from_value("cfg.rows", 0), cs::dim::fixed_at_run(16));
shape_hook("d.x", cs::dim::custom([](cs::ShapeHookCtx& c) { return c.value_of("n", 0) * 2; }));
shape_hook("r.x", cs::ragged::counts_from("counts.n"));                   // ragged target: one descriptor for both dims
shape_hook("s.x", cs::ragged::offsets_from("offs.o"));
```

One descriptor per dimension (`cs::dim::as_is()` leaves a dimension alone). Targets: an input, an input tie alias, or an output
tie alias (the one output-side hook; it sets any dimension a `tie_outputs` leaves unspecified). Each hooked dimension must be
unspecified and owned by one hook. A hook and a connection exclude each other. The hook runs immediately before the first child
that has the target, and every source must be produced by an earlier child.

**Ragged is rank 2 only**, with the leading dimension N and one per-index dimension. Declare it with nothing, `n`,
`cs::counts(v)` or `cs::offsets(v)` (the last two also define N), and size it by hook with `counts_from` or `offsets_from`
(a parallel device scan; only the total comes to the host). Rank 3 ragged is removed.

Sizing an output from a value on the host is the component's own business (for example through a host view kept as a
member); there is no framework helper.
