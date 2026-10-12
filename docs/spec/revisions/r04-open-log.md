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

1. **Initializer device (`init_home()`).** Came up: the old `pick_init_device` took the trigger holder into account; with a flat model there is no trigger. Chose: the tie's `init_device` if set, else the device of the first input instance, else the first instance's device, else the default. `pick_init_container` uses only that. Why: deterministic and independent of run order.
2. **FusedBase stays derived from System (it is a leaf node).** Came up: the plan says FusedBase is a leaf and no longer derives System, but a group needs the declaration calls (connect, tie_*, instance_at, name resolution) for its stages. Chose: FusedBase keeps `System` as a declaration helper, overrides `is_leaf()` to true, so its tree node is a leaf; its children are *stages* (`children_are_stages()` / `adopt_stage`): objects the group owns, registered in `ModelMeta::stage_names` and `node_of_sub` (answering to the group's node), not tree nodes. Stage ports register as `group.stage.x`. Why: reuses all the declaration logic with no second copy; the observable behaviour is the plan's.
3. **Group tied-output push.** Came up: the plan has FusedBase push its own tied outputs in its forward. Chose: finalize places the push for every tie_outputs the same way (after-operation on the last member leaf); a group's members all resolve to the group's leaf node, so it comes out right after the group's forward. FusedBase's `push_events_` and `check_tied_output_devices` are gone. Why: one mechanism.
4. **Ties of ties.** The member refusals are removed from wiring.hpp; System::tie_* mark the tie aliases already in the members' variables `absorbed` (read before the merge).
5. **Dearm / rearm.** Dearm = `inst->data.reset()` on the node's holders, keeping the VarMeta. Rearm = `refresh_container()`, plus `forget_materialization()` when the variable has no live data, plus `mark_pending`. CountedPolicy counts a turn in `after` only when the node is in scope; TransientPolicy rearms in `before` if out of scope and always dearms in `after`.
6. **Ordinals.** A leaf's ordinal is assigned at registration (depth-first, setup is depth-first); a system node carries the first/last ordinals of its subtree. Cycle detection uses the last producer's ordinal and the consumer's ordinal.


## Stopped at

Phase 8 (tests) is partly done. Phases 0-7 are written (unbuilt). Phase 8 so far: the mechanical rename of removed Model API in the built tests (`model.root().X` -> `get_subsystem("").X`, `.root().instance_at(` -> `cs::detail::ModelAccess::instance(`, `get_subsystem(..) ==/->` -> `ModelAccess::subsystem`, `override_run_policy` -> `set_run_policy`), and the three 'deleted after its turn' checks in test_epochs, test_shapes and test_fused_groups, now `ModelAccess::node(..)->in_scope == false`.

Not done: test_system.cpp (about 77 sites: system hooks, `rearm(` in pre_forward, `run_policy_of`/`child_of`, replay and rebuilt-child tests to delete or port), test_p4.cpp (about 32), test_flavors.cpp (about 12; `get_subsystem("grp") == nullptr` style checks), and follow-ups such as `dynamic_cast` of get_subsystem results. Also not done: tests/test_r04.cpp and its CMake entry, phase 9 (examples, getting-started, README, docs, spec status, audit) and phase 10 (build, suites, --check runs, docs-build, push of the built result). Nothing has been compiled yet, so expect compile errors in phase 10.

7. **Tests deleted in phase 8** (behavior that no longer exists in r04):
   - test_p4.cpp: "override_run_policy: applied from pre_forward"; "override_run_policy is not logged: rearming the parent applies the declared policy again"; "rearm is legal outside pre_forward"; "System precompute runs the user''s pre_forward and never a child''s precompute"; "replay skips a logged connection whose other side was deleted: the rebuilt child runs with a fresh variable" (replaced by "rearm_subsystem: a dearmed do_once producer runs again, between runs").
   - test_system.cpp: "lifetime: a do_once producer with a transient reader: the rebuilt reader''s input has no source and its shape is not known"; "rearm: a rebuilt child rejoins the SAME variable and its consumer sees the new output"; "rearm: a rebuilt consumer of a deleted do_once producer still reads the producer''s last value"; "placement: a rebuilt child keeps the outer device".
8. **Tests ported with changed meaning**: "deleted after its turn" is now "out of scope (dearmed)" via `ModelAccess::in_scope`; the view error for a dearmed variable is "no holder of this variable has data"; `update_input` on an out-of-scope input is ignored, not an error; a tie of a tie is allowed (test_p4); a structural call after finalize is a std::logic_error; a tie_outputs member moved by set_device is caught at finalize ("members live on one device"). `RunCounted` in test_system.cpp now owns a leaf `Tick` that logs the tag (a System has no forward).
9. **Shape hooks on a gone source.** A hook placed after a leaf that is out of scope still runs (after-operations run for out-of-scope nodes). `ShapeHookStep::any_source_gone` now also counts a source whose variable is materialized but has no live data as gone, so the last shape stands.
10. **Open: pool accounting of dearmed components.** Dearm frees the holders'' data but a component keeps its solo launcher arrays, so the exact pool byte counts in the transient tests (test_system.cpp, test_flavors.cpp) may differ from the old deleted-component numbers. Expectations are fixed from the real numbers in phase 10.
