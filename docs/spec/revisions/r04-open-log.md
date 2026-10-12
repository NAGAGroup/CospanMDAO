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
