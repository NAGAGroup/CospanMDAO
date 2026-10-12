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
