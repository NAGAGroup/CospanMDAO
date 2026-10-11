# r04: Flat model metadata (living spec)

Status: **in discussion**. This revision is built phase by phase. Every phase stays open until the last one is settled; a later phase may revise an earlier one, and each revision is recorded in a change note in the section it changes. Nothing here is implemented.

## Purpose

Replace per-System ownership of structure with ONE model-wide metadata object that every subsystem reaches through a pointer handed down at setup. In it:

- names are fully resolved, and namespaces are a convention;
- the work of a run is a flat, ordered list of entries;
- wiring, policies and declarations are recorded once, in several representations.

The goals:
- build structure from any level, or procedurally from `main`, between `setup` and the first run;
- replay correctly across subsystem boundaries on rearm;
- keep the rules in one place;
- run without recursion;
- support model-level `get_subsystem`;
- simplify future visualization;
- allow ties of ties.

## Ideas raised so far (not decided)

- **Lifecycle.** Structure may change after `Model::setup` and before the first `Model::run`; after the first run it is fixed. Calling setup again is a reset (much like rebuilding the model).
- **The execution list.** It holds every leaf that does work: components, fused groups, shape hooks, wrappers for a System's `pre_forward` / `post_forward`, and depends_on handoffs. A System itself has only `setup` and never gets an entry.
- **Name maps.** The list is backed by a map from fully resolved name to list index, and by a separate map of Systems, so `get_subsystem` works on Systems too.
- **Subsystems don't own their inputs and outputs.** A declaration concatenates the declarer's resolved name with the input or output name and records it in a name → metadata map. Other indexes sit alongside it, such as subsystem → its inputs and outputs. Dots in a name are only a convention.
- **Connections, promotions and ties** are kept in several representations, for example per variable holder (for replay) and per declaring System.
- **Run policies** are objects the model checks every run for whether they have fired. Rearming a subsystem resets the policies of its whole subtree, fired or not, and probably reruns its setup. A subsystem that goes out of scope takes its inputs and outputs out of the maps.
- **Initialization is eager.** Every free-input initializer runs at the start of the first run, before anything else is submitted. Initializers whose shapes aren't known yet are deferred. A later initializer (update_input) joins the list and leaves it after it is applied.
- **Ties of ties** are wanted.
- **`Model::get_subsystem`** returns a namespace handle for inspection and wiring.

## Phases

### Phase 1: The model's lifecycle and the edit window
First pass, 2026-10-11. Inventory: 5, 30, 31.

Decisions (first pass):
- **States:**
  - *Created*: `create_model` has constructed the root.
  - *Editable*: after `model.setup()`. Structure may still change from `main`, through the root handle or `get_subsystem` handles.
  - *Frozen*: from the first run on, the structure is fixed.
  - *Reset*: `setup()` called again.
- **`setup()` stays a separate call.** It is not folded into `create_model`, so we can't regret removing it later.
- **Procedural building.** `create_model` with a plain System root. Everything is added from `main` between `setup()` and the first run.
- **`finalize()`.** The user may call it; otherwise it runs at the start of the first run.
- **Checks come in three tiers:**
  1. at the call (for example, the cycle warning at connect);
  2. at finalize;
  3. at run time (for example, shapes set by hooks, and shapes resolved lazily).

  A free input has a zero initializer by default.
- **Reverse operations.** Every operation that mutates the model metadata defines its reverse right alongside it: add_subsystem, declaring an input or output, connect, promote, tie, and so on.
  - The subsystem object that performed an operation owns its reverse, not the global metadata.
  - When a subsystem goes out of scope, the model calls its deconstruct, which plays its reverses. The object itself is not deleted.
  - A reverse removes what that subsystem owns. For example, removing a component's holder metadata releases its device data through the existing last-holder rule.
- **Outside declarations stay.** A declaration made outside a subsystem that reaches into it, such as a parent's connect, is not reversed when the subsystem goes out of scope. It stays as metadata, and replay uses it.
- **Rearm.** Rearm reruns setup on the same object, then replays the outside declarations that name it.
  - A rebuild reproduces the same structure by construction, so nothing checks it.
  - Users who do unusual things in hooks are on their own: the rules decide whether a replay succeeds.
- **Reset.** Reset plays the root's reverse operations, then calls `setup()`. Anything added procedurally from `main` is not kept.
- **Run policies are metadata entries.** They are replayed on rearm and can also be updated at run time.
- **Host views and update_input.** Whether a view can be made depends only on whether its variable can be materialized at that moment, not on the state. `update_input` may be called at any time; it only sets an initializer.

Open questions:
- **Run-policy override against a rearm.** Two options:
  - (a) record the override as a later declaration, so it replays after the declared policy;
  - (b) the override replaces the runtime policy and carries an owner, and a setter higher in the hierarchy wins.
- **Reverse operations in detail:** where exactly they are kept, and how and when the model calls deconstruct.
- **The full table of which calls are legal in which state.** First guess: structural calls only while Editable; rearm, policy changes, update_input, views, wait and inspection at run time.
- **Which checks belong to which tier,** item by item.

### Phase 2: Names and the metadata object
Fully resolved names; the maps (name → input/output metadata, subsystem → inputs/outputs, Systems, name → list index); subsystems not owning their inputs and outputs; how a declarer's resolved name is assigned; namespace views for Systems, `main` and `get_subsystem` handles. Inventory: 1, 3, 19, 32, 33.
- Decisions: none yet.
- Open questions: none yet.

### Phase 3: Declarations and construction
`add_subsystem`, factories, depth-first setup order; declaration sequence numbers; one declaration store with several indexes (by variable, by declarer, by subsystem touched); device placement and "the outer call wins", including edits after setup; helper-built systems (composed systems, fused groups). Inventory: 2, 4, 14, 34.
- Decisions: none yet.
- Open questions: none yet.

### Phase 4: Variables and wiring
Variable metadata and its representations; connect, promote and tie in the new store (which merge rules carry over); ties of ties and which level wins; shape definitions (fixed at the first run); aliases; marks; cycles from list position. Inventory: 12, 13, 18, 19, 20, 21.
- Decisions: none yet.
- Open questions: none yet.

### Phase 5: The execution list
The entry base class and its protocol (run, events, owner). What becomes an entry: components; fused groups (stages named, not entered); System pre/post wrappers; shape hooks; tied-output pushes; depends_on handoffs. How positions are computed and recomputed. What stays inside a component's own entry. Inventory: 6, 7, 8, 9, 10, 11, 29.
- Decisions: none yet.
- Open questions: none yet.

### Phase 6: Initialization and materialization
The initializer list at the start of the first run; deferral when a shape is unknown; update_input; when variables are sized and allocated; host views made before the first run. Inventory: 15, 16, 17, 27. Likely to be pulled forward when the discussion needs it.
- Decisions: none yet.
- Open questions: none yet.

### Phase 7: Run policies, rearm and deletion
Policy objects (when checked, what firing means, subtree reset); rearm of any path (rerun setup, replay outside declarations); override_run_policy (recorded, replayed or not); deletion (what leaves the maps, what stays); live data rejoining after a rebuild. Inventory: 22, 23, 24, 25, 17.
- Decisions: none yet.
- Open questions: none yet.

### Phase 8: Host access, inspection and teardown
Handles and the `get_subsystem` handle's API; views; update_input; memory_report; wait; diagnostics; destruction order. Inventory: 26, 27, 28.
- Decisions: none yet.
- Open questions: none yet.

### Phase 9: Fit check and migration
Check against chunked systems (a range run N times), a backward pass (the list in reverse) and solver groups over cycles. Then the final API surface, the implementation order, and the migration of tests, examples and docs. Inventory: 35, 36.
- Decisions: none yet.
- Open questions: none yet.

## Inventory (what the code does today, at 482fdd5)

Each item names a responsibility of the current code that the flat design must place somewhere.

1. **Subsystem tree:** a System's slots (name, factory, device, policy, counter), its child index, and each subsystem's name, path and owner.
2. **`add_subsystem`:** a descriptor (name and constructor arguments) becomes a stored factory. The child's setup runs before `add_subsystem` returns, so setup is depth-first.
3. **Names and paths:** a name is non-empty, contains no dot, and is unique in its namespace. `instance_at` resolves a path by walking down through children.
4. **Device placement:** `set_device` recurses and the outer call wins. `forced_device_` places children built later. Aliases follow, and so does a tie input's init device. The device is a field of each instance.
5. **Model setup:** runs once; a second call throws.
6. **System run:**
   - precompute rebuilds the transient children, then runs `pre_forward`;
   - the compute loop runs, per child: its hooks, the depends_on handoff, the child's own pre, compute and post steps, the tied-output pushes, and the run-policy counter and deletion;
   - postcompute runs `post_forward`.
7. **Component run:**
   - materialize the inputs, run the component's `pre_forward` (set_shape, view_input) and materialize the outputs;
   - point the wrappers, then submit, depending on container events and any depends_on events;
   - record the event, push to the variable's other placements, and run `post_forward`.
8. **Fused group:** a System subclass whose loop never runs. It runs the stages' precompute, makes one launch, pushes its own ties after the launch, then runs the stages' postcompute.
9. **Shape hooks:** one step per call, placed each run before the first child holding the target. There is no source-order check, and `from_value` waits on the host.
10. **Tied-output push:** after the last live member that will run. The member devices are checked every run.
11. **depends_on:** an edge between siblings only, with events handed over for one run. A System target passes them to every component inside it.
12. **Cycles:** found at connect by comparing positions among the calling System's own children only. Deeper paths compare only their first segment. A cycle warns, and the variable keeps its initializer.
13. **Wiring rules and merges:** connect, promote, tie_inputs and tie_outputs, applied per call to the variable metadata (wiring.hpp).
14. **Declaration records:** kept per System, keyed by the first path segment, with replay closures.
15. **Initial values:** the last declaration wins, which relies on a child's setup finishing before its parent's. A component's init_input is setup-only. A System's may be called at any time and marks the variable stale. A tie folds its members' initializers, and a cycle keeps its target's.
16. **Materialization is lazy:** the first holder that touches a variable sizes and initializes it.
17. **Merging two materialized variables** is undefined and an error. A rebuilt child that rejoins live data is the exception.
18. **No ties of ties:** refused through sticky tie flags.
19. **Aliases:** owned by the System that declares them. A promotion is a peer alias.
20. **Shape definitions:** fixed when setup ends.
21. **Marks:** setup-only, and the last mark on a variable wins. The model's finalize collects them by absolute path, holding the instances weakly.
22. **Run policies:** eternal, transient, run_n and do_once, each with a counter. A child is deleted after its last turn. A transient is rebuilt every precompute and deleted after its turn.
23. **rearm:** for the System's own children only. It deletes the child, rebuilds it (factory, setup, device), replays the declarations that name it and resets its counter. Nested rearm is unsupported.
24. **override_run_policy:** takes a path and is not recorded.
25. **Deletion:** instances drop, and data frees wait on their own events. Slots and declarations outlive the child. Wiring that names a deleted child throws.
26. **Model destruction order:** the root first, then the pools, the queues and the context.
27. **Host access:** get_input / get_output handles, views and update_input. A view joins the variable on a host placement and outlives the model.
28. **memory_report, wait, diagnostics,** and the cycle-warning set kept on the root.
29. **System hooks:** a System's `pre_forward` and `post_forward`, alongside the components' own.
30. **When calls are legal:** add_subsystem and marks only in setup. Wiring, init_input, shape_hook, depends_on, rearm and override_run_policy at any time.
31. **No procedural building:** a model needs a System subclass as its root.
32. **No model-level get_subsystem.**
33. **Protected APIs users can reach,** such as tie_inputs_of.
34. **Helper-built systems:** ComposedSystem and FusedGroup build in setup.
35. **Chunked systems:** dead code awaiting their redesign.
36. **Future work:** a backward pass and solvers over cycles.
