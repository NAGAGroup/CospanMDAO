Implementation plan for r04, 2026-10-11 (overnight)

## Corrections to apply throughout (these override the phase text)
- Subsystems keep `name()` and `path()` accessors, both returning the full resolved name (the root's is ""), because error messages and launch descriptions use them. `owner_` / `owner()` are removed.
- Error types:
  - a call in the wrong lifecycle state (structural after finalize, `add_subsystem` on the Model after setup, a view before finalize, a handle used after a reset) throws `std::logic_error`;
  - a rule violation (duplicate name, merge or connect rule, hook ordering, tie checks) throws `std::invalid_argument`, using the existing "naga::cospan: <where>: <verb>: <reason>" format.
- Reset: `Model::Impl` holds the metadata (and through it the tree and the subsystem objects) in a `std::shared_ptr<detail::ModelMeta>`. Handles hold a `std::weak_ptr` to it. `setup()` called again replaces that shared_ptr with a fresh one, so old handles expire and throw on use. No generation counter.
- Operations (before/after lists) are `std::function<void(detail::Runtime&)>` with a kind tag, for diagnostics.
- Run policies take the existing `cs::property_list` (`property::run_n`, `do_once`, `transient`). `set_run_policy(path, property_list)` and `clear_run_policy(path)`. `override_run_policy` is removed.

## Phase 0: preparation
Read the spec. Grep include/naga/cospan for `SlotInfo`, `decls_`, `HookEntry`, `place_hooks`, `override_run_policy`, `run_precompute`, `run_compute`, `run_postcompute`, `find_instance`, `collect_instances`, `pick_init_device`, `model.root`, `run_policy_of`, `child_of`, `rearm(`. Record the hit counts at the top of r04-open-log.md as the starting checklist.

## Phase 1: the metadata core (new include/naga/cospan/model_meta.hpp; subsystem.hpp)
- `detail::ModelMeta` holds:
  - **The tree:** nodes owned by the meta. A node has:
    - its full name;
    - `std::unique_ptr<Subsystem>` (the node owns the object);
    - a system/leaf kind;
    - its children in declaration order;
    - a depth-first leaf ordinal (assigned at registration);
    - an `in_scope` flag (default true);
    - an optional run policy;
    - its before-operations and after-operations;
    - an optional `set_device` override;
    - the holders it owns.
  - **Maps:**
    - full name → node;
    - full name → holder (`std::shared_ptr<detail::Instance>` held weakly; the owner keeps the strong reference), for inputs, outputs, aliases and fused-group stage holders.
  - **Sets:** the cycle-warning set; the pending-materialization set of variables.
  - **Records:** the declared ties, hooks and depends_on, for finalize.
  - **The lifecycle state:** created / editable / finalized.
- **Names:** `child_name(parent, local)` returns `local` when the parent is "", else `parent + "." + local`. Names are never parsed. Registering a duplicate full name (a node or a holder) throws at the call. `resolve(prefix, rel)` does one map lookup.
- **subsystem.hpp:** the Subsystem interface is:
  - `setup()` (systems and leaves declare in it);
  - `forward(rt)` (leaves only);
  - `events()`;
  - `add_dependencies(events)`;
  - `name()` / `path()`;
  - a `meta_` pointer set at registration.

  Remove `run_precompute`, `run_compute`, `run_postcompute`, `find_instance`, `collect_instances`, `owner_` and the recursive `set_device`.

## Phase 2: systems (system.hpp)
- `add_subsystem([props], [device], desc)` does, in order:
  1. construct the child;
  2. compute its full name and refuse a duplicate;
  3. create its node and set the meta pointer and name;
  4. register it;
  5. run its `setup()` immediately;
  6. record any props as the node's run policy and any device as the node's device override.
- Delete: slots and factories, `child_ptrs_`, `decls_` and all replay, `hooks_` / `HookEntry` / `place_hooks`, `tie_outs_`, `dep_edges_`, `rearm`, `override_run_policy`, `apply_run_policy`, the walking `instance_at`, the recursive `set_device` and `forced_device_`, the three run steps, `pre_forward` and `post_forward`. A System has only `setup()` plus the declaration calls.
- **Declaration calls** (`connect`, `promote`, `tie_inputs`, `tie_outputs`, `init_input`, `shape_hook`, `depends_on`, `set_device`, `set_run_policy`, `clear_run_policy`, the marks) resolve names relative to the system's own name, check at the call, and record into ModelMeta. Each one requires the editable state: inside a `setup()`, or after `Model::setup()` and before finalize. `set_run_policy` / `clear_run_policy` are also allowed on the Model between runs.
- A plain `cs::System` root must work, so `create_model<cs::System>()` and then `model.add_subsystem(...)` can build a model procedurally.

## Phase 3: components and fused groups (component.hpp, fused_component.hpp, instance.hpp)
- **ComponentBase:** one `forward(rt)` does, in order:
  1. materialize inputs;
  2. the user `pre_forward`;
  3. materialize outputs;
  4. point the wrappers;
  5. submit through the solo launcher, or the custom-op / host path;
  6. record events;
  7. push to the other placements;
  8. the user `post_forward`.

  Components keep `pre_forward` / `post_forward`. Declaring an input or output registers its holder in the full-name map, as node name + "." + local, and in the node's holder list. Components still own their holders.
- **FusedBase** becomes a leaf (it no longer derives from System). Its `forward` does the same sequence across its stages around one launch, then pushes its own tied outputs. FusedGroup still builds its stages, then its ties, then its connects, inside its `setup()`. Stage holders are registered by full name (`group.stage.x`); stages are not tree nodes.
- **ComposedSystem** stays a System.
- **instance.hpp:** add an `absorbed` flag (a tie alias that became a member of another tie). Keep `tied_member`, `hooked`, `can_be_source` and `can_be_target`.

## Phase 4: wiring (wiring.hpp, variable.hpp flags)
- connect, promote and both ties resolve through ModelMeta. The merge rules are unchanged. Remove the rebuilt-child rejoin exception and the replay helpers.
- **Ties of ties:** remove the `tied_input` / `tied_output` refusals of a member that is itself a tie alias, and mark it absorbed. All other member checks stay.
- **Cycles:** replace `WiringSite` src/dst orders with leaf ordinals.
  - The consumer is the leaf holding the input.
  - The last producer is the highest-ordinal leaf holding an output holder of the source's variable (tie members included; promotions share the variable).
  - A cycle warns once per pair, via the meta's warning set. The same "last producer" function is reused by hook placement.

## Phase 5: materialization (variable.hpp, shape_hook.hpp)
- **Initializer device.** Remove `pick_init_device`'s busiest-device fallback:
  - a tied input runs on the tie's device;
  - a free input runs on the device its input was declared on, unchanged by promotions.
- **`materialize_pending(meta, rt)`** visits the pending set.
  - Shape known (the definition is fully specified, or the initializer carries its shape): allocate on every in-scope placement, outputs included, and initialize unconnected inputs.
  - Shape unknown: leave it, to be materialized lazily on first touch.
  - Then empty the set.
- **`update_input`** marks the variable stale and adds it to the pending set.
- **ShapeHookStep** becomes an operation payload; its old "before the first child" placement is gone.

## Phase 6: run policies (new run_policy.hpp)
- **The interface:** a `RunPolicy` with `before(meta, node)`, `after(meta, node)` and `rearm()`.
  - **run_n / do_once:** count turns; dearm the node's subtree in `after` when they run out; `rearm` resets the counter.
  - **transient:** rearms the subtree in `before`, dearms it in `after`; its own `rearm` is a no-op.
- No default policy; the last caller wins.
- **`dearm(node)`:** walk the subtree. Mark nodes out of scope; leaves' holders release their device data and keep their VarMeta; a system's own aliases release theirs too.
- **`rearm(node)`:** mark the subtree in scope, add its holders' variables to the pending set, and call each policy's `rearm`.

## Phase 7: the Model (model.hpp)
- **Construction and setup.** `create_model<Root>(...)` with the existing config overloads. `model.add_subsystem(...)` adds to the root, and only before `setup()`.
  - `setup()` builds the tree. Calling it again resets: a fresh ModelMeta, the root reconstructed from its stored constructor arguments, and the root's `setup()` run again. Marks are collected after setup.
- **`finalize()`** may be called by the user and runs only once. Otherwise the first `run()` calls it. In order, it:
  1. resolves devices with one tree walk (the outermost `set_device` wins) and applies them to the holders;
  2. places shape hooks:
     - after the last producer of its sources;
     - with no traceable source, before the target's first holder;
     - throws if a source is produced after the target's first holder;
  3. places one tied-output push per non-absorbed tied-output alias, as an after-operation on its last member leaf. It runs only for members in scope, and the one-device check happens here;
  4. places depends_on(b, a) as an after-operation on a's last leaf, which hands the events of a's leaves to every leaf under b;
  5. fixes the shape definitions;
  6. fills the pending set;
  7. sets the state to finalized.
- **`run()`:**
  1. finalize if needed;
  2. `materialize_pending`;
  3. an iterative tree walk with an explicit stack. Per node:
     - its before-operations, then its policy's `before`;
     - if in scope: a leaf runs `forward(rt)`, a system is descended into;
     - its after-operations, then its policy's `after`.

     An out-of-scope system's subtree is skipped.
- **Handles.** `get_subsystem(path)` returns a handle (weak_ptr to the meta, plus the path). A stage of a multi-stage fused group throws. Every handle offers:
  - `get_input<DT,R>(name)` and `get_output<DT,R>(name)`;
  - the view calls and `update_input`;
  - `list_inputs()` and `list_outputs()`, over its subtree.

  A system handle adds the declaration calls (until finalize), `set_run_policy` and `clear_run_policy`. The Model's own calls behave like the root handle's.
- **Also on the Model:** `rearm_subsystem(path)`, between runs.
- **Views** are allowed only after finalize; one is refused if no holder of the variable has data. `update_input` on a connected input keeps its warning.
- **Removed:** `model.root()` and the `Subsystem*` return. `memory_report` and `wait` stay.
- **Teardown order** (member order in Impl): the meta (subsystem objects, holders, device data), the marks, the pools, the queues, the context.

## Phase 8: tests (tests/*.cpp, tests/CMakeLists.txt)
- **Port** every built test to the new API:
  - handles in place of `Subsystem*`;
  - no `model.root()`;
  - `run_policy_of` / `child_of` replaced by observable checks (memory_report, values);
  - `override_run_policy` becomes `set_run_policy`;
  - a system's `pre_forward` rearm becomes `model.rearm_subsystem` between runs.
- **Delete** tests of behavior that no longer exists: system hooks, replay, rebuilt-child rejoin.
- **Add** tests (new file tests/test_r04.cpp) for:
  - the lifecycle: structural calls after finalize throw; `add_subsystem` on the Model after setup throws; `setup()` twice resets and old handles throw; a plain System root built from main;
  - names: dotted local names; duplicates refused; the root adds no dot;
  - ties of ties;
  - the cycle warning by ordinal;
  - hook placement and the hook ordering error;
  - a tied push with members out of scope;
  - depends_on;
  - an out-of-scope subtree skipping policies;
  - run_n / do_once / transient behavior and rearm;
  - the pending pass: shape known up front, shape unknown lazy, `update_input` collapsing;
  - the initializer device for tied and free inputs;
  - fused group stage names and `get_subsystem` on a stage throwing;
  - a view before finalize throwing;
  - `list_inputs` / `list_outputs`.

## Phase 9: examples, docs, audit
- Port examples/gemver-chain and examples/ns-residual (no system hooks; handles; `set_run_policy`).
- Port getting-started/main.cpp and the code in docs/site/getting-started.md (keep them identical), the README snippets, and docs/site/concepts/model.md (add `finalize`, the edit window, handles; keep the approved wording otherwise).
- Mark the r04 spec status "implemented on branch r04-impl (unreviewed)".
- **Audit:** grep the phase 0 names over include/, tests/ (built ones), examples/, getting-started/, docs/site/ and README.md. Nothing may remain except in chunked_system.hpp and the unbuilt tests. Check that no raw pointer into the meta outlives a reset.

## Phase 10: the one build and test
Build and run:
- the default suite;
- the cuda suite;
- gemver-chain and ns-residual `--check` on OpenMP and on CUDA at the usual sizes;
- `pixi run -e docs docs-build` (the baseline is 7 warnings);
- the getting-started directory (it uses the GitHub dependency, so point it at the branch only if needed, and log it).

Fix failures with targeted edits, re-run what failed, then run the suites once more in full. Commit as "r04: flat model metadata (tree walk), built and tested" and push the branch.
