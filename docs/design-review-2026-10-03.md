# Design review, started 2026-10-03

This is the working record of the design review that started 2026-10-03 (Jack and the research lead). The code at c8553f6 works, but it is being redesigned piece by piece for clarity. No new features (solvers, cycles, gradients, optimizers) come before this is done.

## 1. Vocabulary and data (settled)

- **Instances.** Every input, output and alias is an instance owned by exactly one component or system, held by that owner through a shared pointer. Nothing outside the owner holds an instance strongly, and there is no global record of anything: no roots, no shadows, no ledgers.
- **Shared metadata.** A variable's shared metadata is one object, held by shared pointer by every instance in its group. It contains dtype, rank, the shape definition, the initializer, and per device a weak pointer to that device's data. It also keeps weak pointers to the instances that share it (its own small local list, used only for merges). A ragged variable uses a different metadata type built on the same idea.
- **Local to an instance.** A shared pointer to its own device's data (created on demand at run time), its connect flags (can it be a source, can it be a target), and the wrapper its component's kernel reads.
- **Lifetime falls out of ownership.** The metadata lives while any instance holding it exists; a device's data lives while any instance on that device holds it. "Initialized" means some device has live data. There is no stored flag and no special case. A rebuilt child that is wired back in receives the surviving metadata, with its shape known and its data already initialized.
- **Merging lives in the three APIs.** connect: the source's metadata wins. tie: the alias's metadata wins. promote: the promoted member's metadata wins (a promotion is the same variable under a new name). Each call checks its own rules (rank equal; shape definitions compatible; for tie, the members' declared dimensions agree with each other and with the tie's shape rule), folds what the rules say must carry over, sets local flags (for example, tied inputs can no longer be targets), then walks the losing metadata's instance list, skips expired entries, re-points each live instance at the winner, and moves those weak pointers to the winner's list. The metadata and instances stay generic and passive; each feature's rules live in one place, the call that implements it.
- **Rank and shape compatibility.** Rank must match across connected instances, and shape definitions must be compatible (fixed dims agree; unspecified dims take the variable's; dims known only at run time are checked reactively when needed). Checked in connect/tie/promote at the call, not later.
- **Deferred.** Slices (to come back with the chunking redesign); a transformation alias (a System-level alias that reinterprets rank, for example flat to 3D, with its own metadata) for later; whether a tie of a tie is allowed (open).
- **Known drift in the current code (to fix in the rework).** A stale header comment in include/naga/cospan/metadata.hpp lines 8-13 says a peer's rank may differ (it can't; check_rank enforces it, but late, in constraint checks rather than in connect); variable-level state is split between detail::Variable and a hidden shadow entry; promotions and ties hold raw cross-subsystem pointers (mirror_of_, tie_alias_).

## 2. Setup flow (settled)

- **Two passes, called by the model.**
  - Setup: the top-level system's setup, recursive through add_subsystem (each add_subsystem builds the child and runs its setup before returning). Nothing is passed down: no context, no queue, no device.
  - Finalize (as in OpenMDAO's final setup pass): collects the marks (design variables, constraints, objectives) by absolute path as weak pointers to the instances. Nothing else.
- **Setup never allocates, never submits, never compiles.** Building a kernel's JIT launch configuration is a submission-time job, not a setup job. Systems never materialize anything; only components do.
- **Devices:** every input, output and alias instance takes AdaptiveCpp's default device when declared, and each instance has its own device field (devices are never on the shared metadata).
  - add_subsystem takes an optional device argument, next to the properties. add_subsystem is the factory: it builds the subsystem, runs its setup, then, if a device was given, calls set_device on it.
  - set_device on a component sets the device field of each of its input and output instances.
  - set_device on a System sets the device field of each of its own alias instances (promotions and ties), then calls set_device on each child, recursively, down to the components.
  - The OUTER call always wins: a device given to add_subsystem overrides every placement inside that subtree. A subsystem's author decides placements for their own use as a recommendation; a user of it overrides them by passing a device, or keeps them by passing none. Constructors and descriptors never take a device.
  - A promotion copies its member's device when it is made; an outer set_device can override it later like any instance.
  - tie_outputs requires all tied outputs on the same device, checked in the tie call. This stays sound under overrides, because an override always moves a whole subtree to one device.
  - tie_inputs may take an optional device: where the tie's initialization runs (then copied to the other devices) and where its data stays while the tie's System exists.
  - Connect takes no device.
- **Alias instances** (promotions and ties) are instances like any other: owned by their System, on their device, holding that device's data while the System exists. That is what keeps promoted and tied data durable; there is no special mechanism.
- **Every rule is checked in its own call, at the call:** rank and shape compatibility, the tie rules, the connect flags, the winning initializer, and the cycle warning.
  - Initializer: whose initializer wins, not whose value; value forms are conveniences that build a fill initializer. Highest level wins: a parent's setup always runs after its children's, so a later declaration from a higher level replaces the earlier one, and two declarations at the same level are an error, decided in the call.
  - Cycle warning: in connect. The System knows the order of the children it has added so far; if the source's child was added after the target's child, it warns there.
  - Since a tied instance can't be connected to or from, and a connected one can't be tied, connecting into a tie alias is just connecting into any instance.
- **Removed:** the resolve pass; the materialization-slot precomputation; device registration on variables at setup; the exec-info, prepare and compile passes; the exec context; the model's default device. Rebuilding a child is the same add_subsystem build (factory plus setup) followed by replaying the parent's wiring that names it; finalize does not re-run.
- **Memory:** no arena. Each device gets a SlabKit pool sized by the user up front (any allocation size, served as a contiguous run of nodes; node size is the page granularity), with OS allocation as the fallback when the pool is full or too fragmented.

## 3. Execution and submission (settled)

- **Forward = precompute, compute, postcompute,** for components and systems alike. Each step is a framework-owned function (not overridable) that calls a user hook (overridable, empty by default). Users can't replace what the framework does.
- **System precompute:** rebuild transient children (this allocates nothing). The user hook may rearm children by name, e.g. re-running a do-once child on a trigger only the user knows, based on the previous run's results.
- **System compute:** call each child's forward in declaration order, and nothing else. No materializing, no submitting. After a child's forward, if its run policy has ended (run_n reached 0, or a transient's one forward), the parent waits on that child's events only (not the queue), deletes it, then continues. Memory is freed before later subsystems allocate, while independent work already submitted keeps running.
- **Component precompute,** in this order: (1) resolve the derived shapes, calling the driving-value accessor for each driving input (it materializes that input if needed); (2) materialize the remaining inputs; (3) materialize the outputs. Materialize is still called unconditionally on each variable's shared metadata and is a no-op if already materialized. Otherwise it allocates on every device that has instances, then either runs the initializer (whose event goes on the variable) or copies from a live device.
- **Component compute:** the framework builds the JIT launch configuration (hidden from users; one per component type, a function-local static built once). The user supplies the launch setup. The component submits its own work. Two internal APIs: describe (returns the complete launch: configuration plus the user's setup) and submit (uses describe). A fused system calls describe on its children and submits the fused launch itself.
- **Queue:** SlabKit gains a separate fully out-of-order queue type (sk::unordered_queue) that CoSpan submits its kernels to; sk::queue keeps its current semantics and still serves containers. CoSpan no longer relies on in-order ordering. ParallelSystem is removed: concurrency comes from data dependencies, with no execution graph.
- **Events:** each variable's shared metadata keeps two event lists: the write events (the writes of the current forward) and the read events (the reads the next write must wait for).
  - A component's submission depends on its inputs' write events, its outputs' read events, the events of its own device's usm_vectors (new SlabKit getter), and any depends-on events.
  - After submitting, the component keeps its event, adds it to each output's write events and each input's read events, and passes it to the usm_vector assignment APIs for cross-device pushes.
  - Writers in the same forward never wait on each other, so tied outputs (atomic or disjoint) run concurrently.
  - The lists roll over per forward; exact rollover mechanics are settled at implementation.
  - No queue-wide waits: repeated runs stream with no artificial synchronization.
- **Tied outputs across devices:** the System that owns the tie alias does the push to other devices after all members have submitted, depending on all their writes.
- **depends_on:** a System may declare that child B depends on child A. The parent runs A's forward, collects A's events, and hands them to B. If B is a System, every component inside adds them to its submissions. They last one forward.
- **Event getter:** a component returns its event; a System returns its children's events plus its own push events. The same getter serves deletion and depends_on.
- **Removed:** System-side submission, the retire list, materialization slots, ParallelSystem, reliance on in-order queues.
- **User documentation note:** add subsystems that can go out of scope as late as possible in setup(), since deleting one makes its parent wait on that subsystem's events before continuing.
- **Next topics:** fused systems, then host access.

## 4. Fused components and descriptors (settled)

- **A fused "system" is semantically a component:** one kernel, one submission. It is a fused component and works like a component.
  - Precompute runs each stage's precompute.
  - Compute collects each stage's complete launch (the internal describe call; the stages' own submission is bypassed), checks that the stages' group shapes agree (only knowable here, since a component may choose its shape at run time), submits one launch, and gives the resulting event to every stage, so each stage's variables get their event lists exactly as if it had submitted alone.
  - Postcompute runs each stage's postcompute.
  - The fused call sequence (JIT configuration) is built at submission and cached once per distinct sequence of stage functions.
  - The packing buffers (one record per stage plus the stages' input wrappers, output wrappers and body data, flattened into device buffers the fused component owns and walked by the wave kernel with a cursor) are rewritten each forward; that rewrite depends on the previous wave's event.
  - Local memory is summed across stages for sequential and maxed for parallel, and checked against the device limit.
- **Strategies:** sequential (stages run in order inside each work-group; stage-to-stage reads across work-items need the stage's own group barrier) and parallel (stages own disjoint slices of work-groups and run unordered). Connects with the parallel strategy are rejected in the call. All stages of a fused component are either all work-group components or all plain components, enforced at compile time.
- **Component types,** each templated on kernel rank (no runtime branching on rank): plain (range launch, no work-groups), group (work-groups, local memory, barriers), custom-op (raw SYCL interop), host. Custom-op and host can't be fused, enforced by type. Group shape is defined by the component, never by the fuse.
- **Descriptors:** add_subsystem is the factory.
  - Every add_subsystem call takes optional properties (run policy, applying to the whole subsystem) plus a descriptor. The descriptor carries the name.
  - subsystem_desc<T>("name", args...) carries the name and constructor arguments only, with no device, and adds a single component or a user-defined System. add_subsystem([properties], [device], descriptor) takes the device. Constructors never take a device.
  - Helpers for the simple case only:
    - cs::fuse_components<strategy>(connects_desc(...), subsystem_desc<A>("a", args...), ...) builds a fused component and takes no device of its own; the device comes from the add_subsystem call that adds it, and one device per fused component holds by construction because set_device applies to the whole fused component.
    - The equivalent system helper (connects_desc plus descriptors) builds a plain System of those children in order.
    - Helpers accept no per-child run policies and no promotions: a user who needs those writes their own System.
  - Ports of children of a helper-built group are reached by path, e.g. "my_fused_comp.a.x".
  - Inheriting from FusedSystem is no longer the user-facing way to fuse; examples and tests should use the helper.
- **connects_desc** is the same connect, with the same rules, scoped to the group being built. Internal connections make an intermediate visible as internal, which later enables keeping it in local memory or registers.

## 5. Host access (settled)

- **A host view** is an instance of the variable on a host placement, like an input, output or alias. It holds the variable's shared metadata by shared pointer and its host memory (a SlabKit container) by shared pointer; the metadata holds the host memory weakly.
  - While any view exists, the host copy is one of the variable's live placements and is updated by the producer's push after each launch, like any device copy.
  - When the last view goes away, the host memory is freed. Users keep a view alive to avoid reallocating it. Each live view costs one copy to the host per forward (document this).
- **A host view can only be obtained after a run;** asking before the first run throws. A view of a live variable starts by copying from a live device, like any returning placement.
- **sync()** waits on the variable's write events and on the host copy's own container events, then returns. Reading without syncing, or while work is running, is the user's responsibility.
- **The host memory is associated with the device that has the most instances of the variable** (counted from the instance list, which also works before materialization). SlabKit routes the copy whatever the device.
- **A host view outlives the model:** it keeps the metadata and its host memory, and the container's handle keeps the queue alive. When the model goes, every device copy is freed, each waiting on its own registered events first, so destroying the model with work in flight is safe.
- **Model::wait is optional:** for "everything I submitted is done" and for surfacing asynchronous errors (wait_and_throw) from work nobody synced. Document that.
- **The only host write is Model::update_input** (an initializer applied at the variable's next run; see below). get_input and get_output, which handed out references into components, are removed.
- **Model::update_input(path, initializer)** (renamed from init_input; inputs only, since updating an output makes no sense) assigns an initializer to the input's variable and marks the variable STALE. It updates a value, not just an initial value. The declared init_input calls in setup() keep their name and meaning.
  - Each variable's shared metadata has one stale flag. A variable that is not materialized is always stale.
  - At the next run, the first thing that touches the variable acts on it:
    - Not materialized: it materializes and initializes.
    - Materialized but stale: it initializes in place, with no allocation.
    - Either way the flag clears.
    - The driving-value accessor follows the same rule: if stale, initialize, then redo the shape work.
  - Initializing a live variable in place is a write: it waits on the variable's read events (the previous forward's readers) and records its event as a write, like any writer.
  - On an input whose variable has a source (connected), update_input warns and is never applied.
  - update_input queues no work at the moment it is called. It only assigns the initializer and sets the stale flag; the initialization happens on demand in the precompute of the first component that touches the variable on the next run. So several updates before a run collapse to one (the last initializer wins), and a variable nothing touches is never initialized. A host view synced after update_input but before the next run still shows the old values (document this). An immediate-queue path is possible but deliberately not added.
- **Memory report** (between runs, per device): pool bytes reserved and in use, plus allocations and bytes that fell back to the OS, counted as they happen. No arena fields.
- **Removed:** the host cache keyed by write epoch, the "primary" placement, the queue-wide wait in host reads, references into components.

## 6. Value-driven shapes (settled)

- A dimension declared from a value (from_input value, per-index from_input values, tie alias shapes from values) marks the DRIVING variable's shared metadata as shape-driving.
- Whoever produces a shape-driving variable does the shape work right after its own launch, in its postcompute, as ordinary queued work ordered by events (no in-order reliance). For a tied output, the System owning the tie does it, as it does the device pushes. It is done once per variable, however many consumers it sizes.
  - Dense: copy just the driving element(s) to a small host copy kept on the driving variable's metadata.
  - Ragged: a device-side scan produces the offsets on the device (where the consumer's kernels use them), then only the total is copied to the host. This removes today's host-side scan and upload.
- The host still waits at the consumer's precompute: allocation sizes and launch ranges are host decisions. It waits only on that small copy's event, which was started right after the producer, so the wait is usually over already. No device-side sizing; possibly optimized later.
- A consumer gets a driving value through one accessor on its input instance. The result and its event are stored on the variable's shared metadata, so every consumer shares one computation. The accessor has two cases:
  - Not materialized (only possible for a free input; nothing upstream produces it): it materializes the variable, which runs its initializer, then runs the variable's own shape work (the copy, or the device scan plus the copy of the total).
  - Already materialized: the shape work was queued by the upstream producer in its postcompute (or done by an earlier accessor call after the initializer).
  - Either way it waits on the data's events and the shape work's event, then returns the result. Its form depends on the shape kind: a value, values, or a ragged total plus device offsets.
- Staleness is the variable's stale flag (see section 5): the accessor initializes, then redoes the shape work, when the variable is not materialized or is stale.
- A custom device initializer returns the event of the work it enqueues, so materialize can record it as the variable's write and the accessor can wait on it. A custom host initializer's copy to the device is framework-internal and recorded the same way.
- A consumer registered before the driver's producer is a cycle: first forward, the reactive shape error; later forwards, the previous forward's value.
- Interaction with chunked systems is left to the chunked redesign.

## 7. Deferred

- Chunked systems (Jack's redesign).
- Slices (with the chunking redesign).
- The transformation alias.
- Whether a tie of a tie is allowed.

## 8. Implementation phases (settled)

**Rules.** Every phase is source edits only, with no build or test until the final phase. Each phase covers one concern and gets its own complete plan before any edits. Work happens on a local branch with one wip commit per phase. The final phase builds, tests (default and cuda), iterates, squashes and pushes. Each old mechanism is deleted in the phase that replaces it. Base: CoSpan c8553f6, SlabKit 572b581.

1. **SlabKit:** a new sk::unordered_queue type, fully out of order (kernels and async_host, ordered only by the caller's events), separate from sk::queue (merging both into one queue type is a later change); plus usm_vector::events(). SlabKit keeps its arena.
2. **Data model:**
   - shared metadata (dense and ragged): dtype, rank, shape definition, initializer, stale flag, write and read events, a weak device-data entry per device, a weak instance list;
   - an instance base with its own device field and its own device-data share;
   - materialize.
   - Removes: the shadow entry, detail::Variable, root/peers, owner, lifetime remnants.
3. **Wiring:**
   - connect, tie and promote own their rules and the merge walk;
   - checks happen in the call;
   - the initializer winner is decided in the call;
   - the cycle warning is in connect.
   - Removes: resolve, and the late check_ties/check_rank.
4. **Setup and construction:**
   - two passes (setup, finalize marks);
   - add_subsystem([props], [device], subsystem_desc) as the factory, with recursive set_device where the outer call wins;
   - slots, factories, run policy;
   - get_subsystem returns a pointer;
   - the Model holds only a queue and per-device pools.
   - Removes: the exec info/prepare/compile passes, ExecCtx, the default device, CoSpan's arena use, materialize slots.
5. **Components and submission:**
   - plain, group, custom-op and host types, templated on rank;
   - describe/submit;
   - a per-type static JIT config;
   - dependency gathering and event recording.
   - Removes: System-side submission, the retire list.
6. **Forward and Systems:**
   - framework steps calling user hooks; the System loop;
   - transient rebuild in precompute;
   - deletion waits on the child's events;
   - depends_on and the event getter;
   - the tie System pushes to other devices.
   - Removes: ParallelSystem.
7. **Fused components and helpers:**
   - fuse_components<strategy>, the system helper, connects_desc;
   - packing-buffer events, group-shape checks at compute, strategy rules.
8. **Host side and shapes:**
   - host views with sync();
   - update_input and the stale flag;
   - the memory report (pool and OS bytes);
   - producer-side shape work and the driving-value accessor;
   - custom initializers return their event.
   - Removes: get_input/get_output, the write-epoch host cache, "primary".
9. **Chunked systems and slices: NOT ported.** include/naga/cospan/chunked_system.hpp stays in the tree, marked at the top as dead code awaiting the chunked redesign. It is excluded from cospan.hpp, the build, the tests and the docs. Its tests stay in the repo but out of the tests CMake list. Slices (connect with Slice, views, leading ranges) are not ported either; the old implementation stays readable in that header and in git history (c8553f6). The docs carry one note that this code exists and is unused pending the redesign, which is the next topic after this rework.
10. **Tests and examples** rewritten to the new API.
11. **Docs (last):** r01 rewrite, a D-entry, roadmap, README.
12. **Build and integrate:** push SlabKit, repin CoSpan, build, test, iterate, squash, push.
