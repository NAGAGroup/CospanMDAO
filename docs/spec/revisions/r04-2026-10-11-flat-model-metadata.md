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

Change notes (from Phase 2):
- **Procedural building, revised.** Subsystems come from `main` only through `model.add_subsystem`, which adds to the root and only before `model.setup()`. Between `setup()` and the first run, `main` may connect, promote, tie, override run policies and call `update_input`. This replaces "everything is added from `main` between `setup()` and the first run".
- **Systems have no `pre_forward` / `post_forward`.** Rearm is `model.rearm_subsystem(path)`, called from `main` between runs.
- **Run-policy override, direction.** An override is judged by the policy itself, not by who sets it. For lifetimes, an override may not shorten the lifetime the owning system set. Future policies define their own override conditions. This replaces options (a) and (b) in the open questions above.

Change notes (from Phase 3):
- **Reverse operations are removed.** Scope changes are handled by dearm and rearm (Phase 3).
- **Rearm no longer reruns setup, and nothing is replayed.**
- **Reset** clears the metadata object and runs `setup()` again.

These replace the Phase 1 decisions on reverse operations, outside declarations kept for replay, and rearm rerunning setup.

Change notes (from Phase 4):
- **Host views** can be made only after finalize. No variable is materialized before then, so two materialized variables can never be merged.
- **Marks** may be declared after `setup()` and before finalize.
- **`finalize()` does real work:** anything that needs the whole structure in place. The first such job is registering the tied-output pushes (Phase 4).

Change notes (from Phase 7):
- **Run-policy overrides: the last caller wins.** No rule compares the old policy with the new one. A caller that breaks the owning system's expectations is responsible for the result. This replaces the "Run-policy override, direction" note from Phase 2.

### Phase 2: Names and the metadata object
First pass, 2026-10-11. Inventory: 1, 3, 19, 32, 33.

Decisions (first pass):
- **Names.**
  - A resolved name is the parent's resolved name, a ".", and the local name. The root is "" and adds no dot.
  - Dots are only for readability: names are never parsed. A local name may contain dots, for example a subsystem "b.d" or an input "x.y".
  - Every full name is unique across the whole model. A declaration whose full name already exists is refused at the call.
  - A lookup is a single map lookup by full name. Nothing walks the tree to resolve a name.
- **Representations in the metadata object** (first sketch):
  - full name → input/output (holder) metadata;
  - the execution list of leaves, and a separate list of systems;
  - subsystem name → a descriptor saying whether it is a system or a leaf, and its index in that list;
  - a tree of names: the root is "", each system sets its own children, and a map goes from a resolved subsystem name to its node.

  Subtree membership comes from the tree, never from a name prefix. No system holds a pointer to its parent. The model can walk the tree without recursion; this is mostly for visualization.

  A map from each subsystem to its inputs and outputs may not be needed: `get_subsystem(p).get_input(n)` resolves to `p + "." + n`.
- **Ownership.** Components keep owning their holder metadata, so the hot path is unchanged; the maps refer to it. Variable metadata, holders and device data keep today's design.
- **The execution list is built during setup.**
  - A leaf is appended the moment it is added. Depth-first setup guarantees the order.
  - Each system's leaves form a contiguous range that never moves.
- **Each execution item is a leaf plus its before- and after-operations.**
  - The before- and after-operations are special operations, such as shape hooks, tied-output pushes and depends_on handoffs.
  - A declaration that needs one attaches it to the relevant item at the call. For example, a tied push goes after whichever member comes last in list order, and depends on every member's events.
  - Special operations are called unconditionally, and each decides for itself whether to act. A tied push acts if any of its owners is in scope, depends only on the owners that are, and does nothing if none are.
- **Fused groups are the leaf.**
  - A plain component is a fused group with one stage.
  - Stages are not registered as leaves, and a group's internal metadata is opaque to the model.
- **Handles.**
  - `get_subsystem` returns a handle for any subsystem, system or leaf, found through the name → descriptor map.
  - A handle resolves a relative name as prefix + "." + name, and accepts only names below it.
- **Systems have only `setup()`.**
  - `pre_forward` and `post_forward` are removed from systems; components keep theirs.
  - Rearm is `model.rearm_subsystem(path)`, called from `main` between runs. Run policies may gain rearm conditions later.
- **Structural edits happen only during setup.** `add_subsystem` is only legal in a `setup()`, or as `model.add_subsystem` (on the root) before `model.setup()`.

Open questions:
- **An out-of-scope subsystem's holders:** do they leave the maps?
- **Reaching a fused group's stage through `get_subsystem`.** Three options:
  - refuse it;
  - return a wrapper around the group that maps the stage's names to the group's;
  - register stage records.
- **Reverse operations.** Two options:
  - each system owns its own and they are played recursively;
  - one log with a slice per system. A rearm appends new entries, which breaks the slice's contiguity.
- **Strings or interned ids** for the paths off the hot path.
- **Shape hooks:** setup-only or not.

Change notes (from Phase 3):
- **Out-of-scope holders.** Holders are not deleted. They release their device data and keep their variable metadata.
- **Reverse operations are removed** in favor of dearm and rearm.

These answer the open questions on out-of-scope holders and on reverse operations.

Change notes (from Phase 5):
- **Shape hooks may be declared from `main` before finalize.** All special operations are placed at finalize. This answers the open question on whether shape hooks are setup-only.

Change notes (from Phase 8, layout revision):
- **No flat execution list.** The run walks the tree of subsystems, using an explicit stack and no recursion. A system node holds its children in declaration order, and a leaf (a component or fused group) has no children.
- **The flat maps stay for lookups:** full names → holders, names → subsystem handles, and each system's connections. A lookup never walks the tree.
- **Each leaf gets a depth-first ordinal at setup.** Cycle detection and hook placement compare these ordinals, where they used to compare list positions.
- **The name → execution-index map** becomes name → tree node.

### Phase 3: Declarations and construction
First pass, 2026-10-11. Inventory: 2, 4, 14, 34.

Decisions (first pass):
- **`add_subsystem`** constructs the child from its descriptor and registers it before its `setup()` runs:
  - it gives the child its resolved name and the metadata pointer;
  - it adds the child's tree node and its name → descriptor entry;
  - it appends the child to the execution list if it is a leaf, or to the systems list if it is a system.

  It then runs the child's `setup()` immediately (depth-first). Declaration order is execution order.
- **No stored factories.**
- **Declaration order wins.** There are no sequence numbers. Declarations are never replayed, so their order plays out only once: during setup, and from `main` before the first run.
- **Dearm and rearm.** Every kind of subsystem has both.
  - **The walk.** When a run policy takes a subsystem out of scope, the model starts at its tree node and walks the subtree depth-first, parent before children, calling dearm. Rearm walks the same way.
  - **Dearm.** A leaf disables its execution entry and has its holders release their device data. A system has its aliases release their device data.
  - **Holders are not deleted.** They keep their variable metadata, so variable metadata lives until reset, and no separate variable map is needed.
  - **Rearm** re-enables the entries. Holders materialize again the next time they are touched, as on the first run. A variable whose data was all freed runs its initializer again: a free input re-enters the eager initializer list (Phase 6).
  - **Nothing is replayed.** The connections, the merged shape definitions and the initializers are already in the variable metadata. Hooked dimensions recompute, or reuse the last run's.
- **Devices.** `set_device` stays: it lets someone who uses a system override the devices its author chose. The outer call wins, and devices are fixed at finalize. As a result, the check that a tie's output members are all on one device runs at finalize, not every run.
- **Fused groups.** The group is the leaf. Its stages' inputs and outputs are registered by full name, as they are now.
- **Helper-built systems** (`ComposedSystem`, `FusedGroup`) still build through `setup()`, unchanged.

Open questions:
- Where depends_on goes when the dependent subsystem is a system (deferred to Phase 5).
- How `set_device` is applied in the flat design: walking the tree, or recorded per subsystem and resolved at finalize.

Change notes (from Phase 6):
- There is no separate eager initializer list. A rearmed free input whose data was freed is picked up by the start-of-run materialization pass (Phase 6). This replaces "re-enters the eager initializer list".

Change notes (from Phase 7):
- **Run policies live on execution items (leaves).** A policy set on a system is propagated to the leaves in its range (Phase 7). A leaf goes out of scope through its own policy, not through a dearm walk started from a system.
- **What releases a system's alias data is open again.** "A system has its aliases release their device data" is withdrawn. See the Phase 7 open questions.

### Phase 4: Variables and wiring
First pass, 2026-10-11. Inventory: 12, 13, 18, 19, 20, 21.

Decisions (first pass):
- **Carried over unchanged** (wiring.hpp, variable.hpp):
  - **The data model:** shared variable metadata, holders on it, and per-placement device data freed by the last holder.
  - **The merge rules,** checked at the call before anything changes: on a connect the source's variable wins; on a tie the alias's wins; on a promote the member's wins.
  - **Shape-definition merging:** unspecified absorbs specified, two specified values must agree, and a hooked dimension meets only an unspecified one.
  - **The connect checks:** an input has at most one source; a ragged source cannot feed a dense input; dtype and rank must match; a hooked or tied input cannot be connected to, and a tied output cannot be connected from.
  - **Aliases** (promotions, tie aliases) are holders owned by the system that declares them, registered by full name.
- **Resolution.** Every wiring call resolves names through the metadata object, relative to the caller's handle, with one lookup per name. Deep paths, such as `a.b.c.x`, are allowed.
- **No replay.** The merge exception for a rebuilt child rejoining live data is removed.
- **When wiring is legal:** in `setup()`, and from `main` between `setup()` and finalize. The rules are the same either way.
- **Marks** may be declared after `setup()` and before finalize.
- **Shape definitions** are fixed at finalize.
- **Two materialized variables can't merge.** Host views can be made only after finalize, so no variable is materialized before then.
- **Cycles** are found at the connect, by comparing positions in the execution list:
  - the consumer is the leaf that holds the input;
  - the producer is the leaf that produces the source variable, looking through promotions, and taking the last member for a tied output.

  This is exact across the whole model. The list is fixed once setup has run.
- **Ties of ties are allowed.**
  - **Inputs.** An inner tie's alias is a free input and can be a member of an outer `tie_inputs`. The existing merge rules decide the rest: the outer alias's variable wins, the definitions merge, and the initializers fold. The outer tie's own initializer wins, because it is declared later.
  - **Outputs.** An inner tie's alias can be a member of an outer `tie_outputs`. The atomic rule and the one-device rule apply to the outer tie as a whole.
- **Tied-output pushes are registered at finalize, not when the tie is declared.** By then every tie of ties has been merged, so each tied variable gets exactly one push, placed after its last member in list order. This is the first job that gives `finalize()` real work.

Open questions:
- How finalize finds the outermost tie of each tied variable, and attaches its single push.

### Phase 5: The execution list
First pass, 2026-10-11. Inventory: 6, 7, 8, 9, 10, 11, 29.

Decisions (first pass):
- **The run loop.** For each item in the execution list, run its before-operations, then the leaf if it is enabled (in scope), then its after-operations. The eager initializer list (Phase 6) comes before the loop; the run-policy checks (Phase 7) come after it. There is no recursion.
- **One `forward` per leaf.** Today's three steps (precompute, compute, postcompute) collapse into a single `forward`. The split existed only so that a System could interleave its children. A component's `forward` does, in order:
  1. materialize its inputs;
  2. call its own `pre_forward` (`set_shape`, `view_input`);
  3. materialize its outputs;
  4. point the wrappers;
  5. submit;
  6. record its events;
  7. push to the variable's other placements;
  8. call `post_forward`.

  A fused group runs these steps across its stages, around its single launch.

  Subsystems only ever have `forward`. `run` belongs to the Model alone, because it will grow into forward, backward and cycle solvers.
- **The rest of the leaf protocol:** `events()` (this run's events), `add_dependencies(events)` (extra events for the next submission), and `dearm()` / `rearm()` (Phase 3).
- **Special operations.** A special operation has its own run step and knows its owners. Every special operation is placed at finalize: a declaration records what it wants, and finalize decides where it runs.
- **Shape hooks.**
  - **Placement.** A hook is placed directly after the last producer of any of its sources: the paths named by `from_shape`, `from_value`, `counts_from` and `offsets_from`.
  - **Ordering error.** A hook must still come before the first holder of its target. If a source is produced after that holder, it is an error, like a cycle, and it is checked right away. The error stands even if something higher in the hierarchy might override the shape: the system was defined incorrectly.
  - **Fallback.** A `custom` hook has sources that can't be traced, so it falls back to just before the first holder of its target. A dimension that is `fixed_at_run` does nothing at run time, so a hook made only of those dimensions falls back the same way and submits nothing.
- **depends_on.**
  - It works over ranges: `depends_on(b, a)` collects the events of every leaf in a's range and hands them to every leaf in b's range. Either a or b may be a leaf or a system; a system means its contiguous range.
  - It is one after-operation, on the last item of a's range.
  - It may be declared from `main` before finalize.
- **Tied-output pushes** are placed at finalize (Phase 4).
- **What goes away:** each System's event aggregation (today's `forward_events_`), placing the hooks again on every run (`place_hooks`), and recursion in the run.

Open questions:
- How finalize computes "the last producer of a source" when a source is a promotion or a tie alias. (The rule matches cycle detection in Phase 4.)

Change notes (from Phase 8, layout revision):
- **The run loop is a tree walk.** For each node in declaration order:
  1. run its before-operations and its run policy's `before`;
  2. if the node is in scope: run it if it is a leaf, or descend into its children if it is a system;
  3. run its after-operations and its run policy's `after`.
- **Special operations attach to tree nodes,** and are still placed at finalize.

### Phase 6: Initialization and materialization
First pass, 2026-10-11. Inventory: 15, 16, 17, 27.

Decisions (first pass):
- **No initializer list.** At the start of a run, before anything else is submitted, a materialization pass walks the variables. Inputs are the one special case: they are also initialized.
  - **Shape known:** every variable whose shape is known has all its placements allocated up front, outputs included, so nothing is allocated in the middle of the run. A shape is known when the definition is fully specified, or when an initializer carries its own shape (for example `from_host` with a vector).
  - **Unconnected inputs** (free inputs, promoted or not, and tie input aliases) are also initialized in the same pass when they can be materialized. The ones that can't be are skipped.
  - **A variable whose shape isn't known yet,** for example a dimension owned by a shape hook, or an output whose producer sets its shape, is materialized the first time a holder touches it during the run. Today's lazy path survives for these alone.
- **The default zero initializer is just an initializer.** Declaring an input without one declares it with the zero initializer, so no special case exists anywhere.
- **The device that runs an initializer** comes from the merge rules:
  - a tied input runs on the tie's device;
  - a free input that isn't tied runs on the device its input was declared on. Promotions keep the same device.

  The result is then copied to the variable's other placements. The "busiest device" fallback in today's code (`variable.hpp`, `pick_init_device`) is removed.
- **`update_input`** marks the variable stale. The next run's pass initializes it again, in place, waiting on the previous run's readers. Several updates before a run collapse into the last one.
- **A rearmed free input whose data was freed** is picked up by the same pass, because it is no longer materialized.
- **Host views** can be made only after finalize. A view made between finalize and the first run may materialize and initialize its variable early, and that is fine.

Open questions:
- Whether the pass walks every variable on every run, or keeps a set of stale or unmaterialized variables to visit.

### Phase 7: Run policies, rearm and deletion
First pass, 2026-10-11. Inventory: 17, 22, 23, 24, 25.

Decisions (first pass):
- **A run policy is its own kind of object,** attached to an execution item (a leaf). It has three calls: `before`, `after` and `rearm`.
  - **When `before` runs:** after the item's before-operations and before the item itself.
  - **When `after` runs:** after the item's after-operations.
  - **Even when the item is disabled.** Both calls happen whether the item is enabled or not, and the policy decides what they do.
- **The current policies:**
  - **run_n and do_once** count turns and dearm the leaf in `after` when they run out. Their `rearm` resets the counter.
  - **transient** rearms the leaf in `before` and dearms it in `after`. Its own `rearm` does nothing.
- **There is no default policy.** A leaf with no policy just runs, and a policy can be cleared.
- **The last caller wins**, whether the policy comes from a declaration in setup, from `main`, or from an override. Nothing compares old and new. A caller that breaks the owning system's expectations is responsible for the result.
- **Setting a policy on a system** propagates it, at the call, to the leaves in the system's range. There are two modes:
  - override every leaf in the range;
  - or apply it only to the leaves that have no policy.
- **Nested policies are independent.** Each leaf's policy runs on its own, and an ancestor's scope is not consulted. For example, a transient leaf inside a system whose run_n leaves have run out keeps coming back every run, because its own policy says so.
- **`model.rearm_subsystem(path)`.** For every leaf in the subsystem's range, it rearms the leaf and calls its policy's `rearm`. On a subsystem that is already in scope, this resets its counters.
- **Policies are part of a system's contract.** A system's author documents the run policies it uses, so that users don't have to read its code. We don't try to handle every combination.
- **Consequences:**
  - **Data outlives its producer** while any holder still holds it. A do_once producer's output stays alive as long as some consumer holds that placement.
  - **The up-front allocation pass skips leaves that are out of scope.** A transient's variables are allocated when it is rearmed during the run, not at the start.
  - **Tied outputs and hooks** keep working while some of their owners are out of scope, because their operations check scope themselves.

Open questions:
- **What releases a system's alias data** (promotions and tie aliases)? It is not "when all of its leaves are out of scope": promotions exist precisely to keep data alive after the execution that produced it is no longer needed.
- **The API names** for clearing a policy, and for setting a policy on a system with each of the two modes.

Change notes (from Phase 8, layout revision):
- **Run policies sit on any node,** leaf or system, with `before` and `after` around that node. A node's own policy is called even when the node is out of scope, so a transient can rearm itself.
- **An out-of-scope system skips its whole subtree.** The walk does not descend into it, so the policies of its descendants are not called. This replaces "nested policies are independent".
- **Rearm walks the subtree** and rearms every subsystem in it, calling each policy's `rearm`.
- **A policy set on a system lives on that system's node.** Stamping it onto the leaves, and the two modes for doing so, are removed.

### Phase 8: Host access, inspection and teardown
First pass, 2026-10-11. Inventory: 26, 27, 28.

Decisions (first pass):
- **Handles.**
  - The model's own calls go through the root handle: `model.get_input`, `get_output`, the view calls, `update_input`, `list_inputs` and `list_outputs`.
  - `get_subsystem(path)` returns a handle for any subsystem. It offers the same inspection calls: `get_input<DT, R>(name)`, `get_output<DT, R>(name)`, views, `update_input`, `list_inputs` and `list_outputs`.
  - Inputs and outputs are resolved as `prefix + "." + name`.
- **Listing.** `list_inputs` and `list_outputs` are offered on every handle, so the per-subsystem map of inputs and outputs comes back. Its exact form waits for the metadata-layout proposal (open questions).
- **Wiring.** Only system handles accept wiring calls, and only before finalize: connect, promote, tie, marks, run policies, shape hooks and depends_on. A leaf handle (a component or a fused group) refuses them, because aliases belong to systems.
- **Views.**
  - Views can be made only after finalize.
  - **No holder has data:** a view on a variable whose holders hold no data (all of them out of scope) is refused, with an error saying so.
  - **Some holder has data:** the view joins the variable and copies from that holder, as it does today.
- **Out-of-scope subsystems.** `get_input` and `get_output` on an out-of-scope subsystem still work, because the holders exist even though their data has been released.
- **`update_input`** may be called at any time. On a connected input it keeps today's warning and has no effect.
- **Reset invalidates handles.** We don't guard against stale handles. Like iterators after a vector resize, any handle held across a reset is invalid, and that is documented. A handle holds a weak pointer to what it refers to, so a call on an invalid handle throws instead of being undefined behavior.
- **Views outlive the model.** A view keeps its variable's metadata and its host data, as it does today, and stays usable after a reset as a detached snapshot.
- **Teardown order:**
  1. the subsystem objects, and with them the holders and the device data (each free waits on its own events);
  2. the metadata object;
  3. the marks;
  4. the pools;
  5. the queues;
  6. the context.
- **Unchanged:** `memory_report()` and `wait()`. The cycle-warning set moves from the root system to the metadata object.

Open questions:
- **The metadata-object layout:** Jack plans to propose a change, which decides the form of the per-subsystem input and output maps.

### Phase 9: Dropped
Decided 2026-10-11. This phase is not part of this spec:
- **Chunked systems** stay as dead reference code until their own design discussion.
- **The backward pass and solvers** are far off.
- **Migration** belongs to implementation planning.

Inventory items 35 and 36 are out of scope here.

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
