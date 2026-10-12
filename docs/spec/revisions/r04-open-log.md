# r04 open log

What came up during the overnight implementation of r04 (branch `r04-impl`), what was chosen, and why. For Jack's morning review.

## Phase 0: starting checklist (hit counts on the branch point, 5b7d3b7)

Counted with ripgrep over `include/naga/cospan` (without `chunked_system.hpp`), the built tests (without `test_chunked.cpp`, `test_leading_range.cpp`, `test_replay.cpp`) and examples / getting-started / README / docs/site. Lines, not files.

| name | include | tests | examples + docs |
|---|---|---|---|
| SlotInfo | 6 | 0 | 0 |
| decls_ | 18 | 0 | 0 |
| HookEntry | 6 | 0 | 0 |
| place_hooks | 2 | 0 | 0 |
| override_run_policy | 9 | 12 | 0 |
| run_precompute | 7 | 0 | 0 |
| run_compute | 10 | 0 | 0 |
| run_postcompute | 7 | 0 | 0 |
| find_instance | 5 | 0 | 0 |
| collect_instances | 3 | 0 | 0 |
| pick_init_device | 1 | 0 | 0 |
| model\.root | 0 | 73 | 0 |
| run_policy_of | 1 | 0 | 0 |
| child_of | 1 | 0 | 0 |
| rearm\( | 4 | 5 | 0 |

The audit in phase 9 must bring all of these to zero outside `chunked_system.hpp` and the unbuilt tests.

## Decisions and open items

(Entries are appended below as they come up.)

1. **Initializer device: no fallbacks (rewritten after review).** A tie's device is the device passed to `tie_inputs` / `tie_outputs`, else the default device; the members' devices say nothing about it. A tied input initializes on the tie's device. A free input initializes on the device it was declared on: its variable gets `init_device` at the port's declaration, `set_device` moves it with the port (a port that is a member of a tie has the tie's variable and leaves it alone), and a promotion changes nothing. `VarMeta::init_home()` is `init_device`, else the default device (a variable with no free input is never initialized); the "first input instance" and "first instance" fallbacks are gone. `build_tie_alias` always sets the alias's `init_device`.
2. **FusedBase stays derived from System (it is a leaf node).** Came up: the plan says FusedBase is a leaf and no longer derives System, but a group needs the declaration calls (connect, tie_*, instance_at, name resolution) for its stages. Chose: FusedBase keeps `System` as a declaration helper, overrides `is_leaf()` to true, so its tree node is a leaf; its children are *stages* (`children_are_stages()` / `adopt_stage`): objects the group owns, registered in `ModelMeta::stage_names` and `node_of_sub` (answering to the group's node), not tree nodes. Stage ports register as `group.stage.x`. Why: reuses all the declaration logic with no second copy; the observable behaviour is the plan's.
3. **Group tied-output push.** Came up: the plan has FusedBase push its own tied outputs in its forward. Chose: finalize places the push for every tie_outputs the same way (after-operation on the last member leaf); a group's members all resolve to the group's leaf node, so it comes out right after the group's forward. FusedBase's `push_events_` and `check_tied_output_devices` are gone. Why: one mechanism.
4. **Ties of ties.** The member refusals are removed from wiring.hpp; System::tie_* mark the tie aliases already in the members' variables `absorbed` (read before the merge).
5. **Dearm / rearm.** Dearm = `inst->data.reset()` on the node's holders, keeping the VarMeta. Rearm = `refresh_container()`, plus `forget_materialization()` when the variable has no live data, plus `mark_pending`. CountedPolicy counts a turn in `after` only when the node is in scope; TransientPolicy rearms in `before` if out of scope and always dearms in `after`.
6. **Ordinals.** A leaf's ordinal is assigned at registration (depth-first, setup is depth-first); a system node carries the first/last ordinals of its subtree. Cycle detection uses the last producer's ordinal and the consumer's ordinal.
7. **Tests deleted in phase 8** (behavior that no longer exists in r04):
   - test_p4.cpp: "override_run_policy: applied from pre_forward"; "override_run_policy is not logged: rearming the parent applies the declared policy again"; "rearm is legal outside pre_forward"; "System precompute runs the user's pre_forward and never a child's precompute"; "replay skips a logged connection whose other side was deleted: the rebuilt child runs with a fresh variable" (replaced by "rearm_subsystem: a dearmed do_once producer runs again, between runs").
   - test_system.cpp: "lifetime: a do_once producer with a transient reader: the rebuilt reader's input has no source and its shape is not known"; "rearm: a rebuilt child rejoins the SAME variable and its consumer sees the new output"; "rearm: a rebuilt consumer of a deleted do_once producer still reads the producer's last value"; "placement: a rebuilt child keeps the outer device".
8. **Tests ported with changed meaning**: "deleted after its turn" is now "out of scope (dearmed)" via `ModelAccess::in_scope`; the view error for a dearmed variable is "no holder of this variable has data"; `update_input` on an out-of-scope input is ignored, not an error; a tie of a tie is allowed (test_p4); a structural call after finalize is a std::logic_error; a tie_outputs member moved by set_device is caught at finalize ("members live on one device"). `RunCounted` in test_system.cpp now owns a leaf `Tick` that logs the tag (a System has no forward).
9. **Shape hooks on a gone source, per dimension (rewritten after review).** A named source is gone when its instance expired or its variable was materialized and all of its data has been freed since. While the target's shape is known, a dimension whose source is gone is not set and keeps its last extent; the hook's other dimensions still follow their sources (Jack, 2026-10-12). A ragged hook has one source, so the whole hook is skipped. Hooks placed after an out-of-scope leaf still run.
10. **Open: pool accounting of dearmed components.** Dearm frees the holders' data but a component keeps its solo launcher arrays, so the exact pool byte counts in the transient tests (test_system.cpp, test_flavors.cpp) may differ from the old deleted-component numbers. Expectations are fixed from the real numbers in phase 10.
11. **Dearm frees data immediately (rewritten after review).** `dearm_subtree` releases every holder's container at once; a container another holder still holds lives on with that holder, and a free waits on the container's own events. The phase-10 failure ("a transient producer feeding a transient reader", test_system.cpp) was not a bug in dearm. Under r04 a transient is rearmed in its own `before` and dearmed in its own `after`, so between p's turn and t's turn nothing held p.y and t read a freshly allocated, never-written container. Before r04 every transient child of a System was rebuilt at the start of the System's forward, which is why the test passed then. That meaning is not coming back (Jack, 2026-10-12): data a later transient needs is kept by promoting it and connecting the promotion. The test was rewritten that way, and `release_dearmed` and a start-of-run rearm of transients were both removed; the model never special-cases a run policy.
12. **Initializer shapes are checked at the call (rewritten after review).** An initializer that carries a shape defines every dimension, and the standard merge rules apply (Jack, 2026-10-12): a fixed dimension must match and a hooked dimension is a conflict. `VarMeta::check_init_against_definition` runs at init_input (declare_init), update_input, shape_hook (against an initializer already declared) and finalize (against the final definition), so these errors no longer wait for the first run. `shape_resolvable` lost its hooked-dimension special case, since a hooked variable can no longer carry an initializer shape. test_init_shapes.cpp's mismatch tests now expect the throw from setup().
13. **Views need finalize; tests call `finalize()` first.** Tests that made views between setup and the first run now call `finalize()` after `setup()`. Tests deleted in phase 10: "a late init_input on a materialized free input re-initializes at the next forward" (test_shapes.cpp; `init_input` is illegal after finalize).
14. **Pool accounting.** A dearmed component keeps its solo launch arrays (a fused group its group arrays), so the transient pool-byte expectations in test_system.cpp and test_flavors.cpp now add `solo_launch_bytes(k)` / `group_launch_bytes(2)` for the components that have run. Not freed on dearm; could be a later optimization.
15. **Tests changed in phase 10:** "setup twice throws" is now "setup twice resets the model" (test_model.cpp); "a tie of a tie" in test_fused_groups.cpp is now allowed.
16. **getting-started.** Built from a temporary copy under build/ with `cospanmdao = { git = ..., branch = "r04-impl" }`; the committed getting-started/pixi.toml is unchanged (it points at the default branch, which does not have r04 yet). It prints `3 5 7 9`.

## Phase 10 results

- default suite: test cases 305 (294 passed, 11 skipped), assertions 3177 all passed.
- cuda suite: all tests passed (3243 assertions in 305 test cases).
- gemver-chain --n 2048 --check: ALL OK on OpenMP and on CUDA.
- ns-residual --nx 1024 --ny 1024 --limiter minmod --bc walls --range-split all --check: ALL OK on OpenMP and on CUDA (unfused, fused-flatten, fused-pad, fused-padded, handwritten).
- docs-build: build succeeded, 7 warnings (the baseline, all at api/index.md:5).
