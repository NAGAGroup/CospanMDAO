# Section 11 implementation log

Running log, newest entries at the bottom. Each entry: date/time (UTC), phase, what was done, commits (repo + SHA), test results, decisions taken, open questions for Jack.

## Spelling decisions (for Jack's review)

| Name | Where | Meaning |
|---|---|---|
| `define_as_indexed_call_sequence(call, definitions, index_param)` | AdaptiveCpp fork, `dynamic_function_config` | defines a call sequence in which definition k is called with the wrapper's own arguments, except that argument `index_param` is replaced by the integer constant k |
| `cs::fuse_components("name", cs::connects_desc({...}), cs::segment(...), cs::global_barrier(), cs::segment(...), ...)` | CoSpan, fused group helper | builds a fused group from segments and boundary primitives; a bare stage desc is a one-stage segment |
| `cs::ticket_order()` | CoSpan, fused group boundary primitive | tickets for the whole launch, and per-group completion recording |
| `cs::global_barrier()` | CoSpan, fused group boundary primitive | ticket order, plus waiting for every earlier group |
| `cs::segment<cs::local_mem::shared>(...)` | CoSpan, fused group segment | stages of the segment see the same local memory block (max of their requests) |
| `cs::segment<cs::local_mem::partitioned>(...)` | CoSpan, fused group segment (the default) | each stage gets its own local memory slice (sum of their requests) |
| `cs::range_mode::own` | CoSpan, fused group segment option (the default) | each stage sees only its own requested groups |
| `cs::range_mode::whole` | CoSpan, fused group segment option | each stage sees the whole range |
| `override_run_policy("b.c", policy)` | CoSpan, `System` method | sets a subsystem's run policy by path; legal anywhere |
| `view_input(input)` | CoSpan, protected `Component` method | makes a host view of one of the component's own inputs, using its precompute's queue |
| `model.get_input(path)` | CoSpan, `Model` | returns a handle with `.view()` and `.update(init)` |
| `model.get_output(path)` | CoSpan, `Model` | returns a handle with `.view()` |
| `set_shape(out, cs::as_is, 12, cs::as_is)` | CoSpan, component `pre_forward` | one entry per dimension: a number or `cs::as_is` |
| `segment()`, `num_segments()`, `segment_groups(s)`, `group_in_segment()` | CoSpan, `GroupItem` | the current segment, the number of segments, a segment's group count, the group's index within its segment (from 0) |
| `wait_group(s, g)`, `wait_earlier_segments()` | CoSpan, `GroupItem` | collective wait on a specific earlier group, or on all earlier groups |
| `local<T>(n)` | CoSpan, `GroupItem` | a typed slice of the group's local memory |

## Open questions for Jack

## Entries

### 2026-10-06 09:05 UTC, P0: documents
- **Done:** wrote `docs/plan-section11.md` (the implementation plan, verbatim from the lead) and this log, with the spelling table pre-filled from "Initial spelling decisions" in the plan.
- **Commits:** cospan, "docs: section 11 implementation plan and log" (SHA in the commit; see git log).
- **Tests:** none (docs only).
- **Decisions taken:** none.
- **Open questions for Jack:** none.

### 2026-10-06 09:15 UTC, P1a + P1b: fork change committed, toolchain release dispatched
- **Done:**
  - P1a, in `/home/jack/projects/acpp-fork`, branch `naga/develop`: the indexed call sequence and the cross-TU linking fix, plus tests and a doc entry in `doc/extensions.md`. The fork was not built locally.
  - P1b: dispatched `.github/workflows/build.yml` in `NAGAGroup/acpp-toolchain`, linux-64 publish monitored (monitor `m1`).
- **Commits:** acpp-fork `f815ee1efa0e887d18307e8d4169e320c702a1b6` ("jit: indexed call sequences; link dynamic-function definitions from other TUs").
- **Release:**
  - Date (UTC) 2026.10.06; staging had no `naga-acpp` build of that date on linux-64 (newest was 2026.10.01), so `build_number` = 0.
  - Run id 37441431821, https://github.com/NAGAGroup/acpp-toolchain/actions/runs/37441431821.
  - Dispatch: `gh workflow run build.yml -f fork_sha=f815ee1efa0e887d18307e8d4169e320c702a1b6 -f 'platforms=linux-64 linux-aarch64 osx-arm64 win-64 win-arm64' -f publish=true -f build_number=0`.
- **Test results:** none yet; the fork's lit tests (`dynamic_function_indexed.cpp`, `dynamic-function-cross-tu/`) are verified only through the published build and the CoSpan probe (P1c).
- **Decisions taken (fork):**
  - The index position is stored per entry in `fcall_specialized_config::function_call_map`, which is now a vector of `fcall_specialization_entry{function, replacements, index_param}`. It enters the configuration hash through each entry's `unique_hash` term, which `kernel_configuration::generate_id` already folds in, so `kernel_configuration.hpp` needed no edit.
  - Under `LinkOnlyNeeded`, a symbol is linked only if the module declares it, and a replacement named only in the config is not declared. `resolveExternalSymbols` therefore declares each missing replacement (with the type of the function it replaces) before resolving, and `jit.hpp` adds the replacement names to the imported symbols, as the plan says.
  - A replacement that is still only a declaration when specializations are applied is now reported as "Could not find function call specialization target". Before, it was a late backend failure.
  - The plain call sequence hashes one more value (index -1), so existing configurations get new cache keys. Their behavior is unchanged.
- **Open questions for Jack:** none.

### 2026-10-06 UTC, P2: SlabKit container events
- **Done:** `usm_vector` keeps a read list and a write list; they replace `last_` and the registered list. API: `read_deps()` (the write list, collapsed), `write_deps()` (the read list, collapsed), `register_read` / `register_write` with span overloads. `last_event`, `events`, `register_event(s)` and `registered_event_count` are gone. A list with more than one event is collapsed on access by an empty `single_task` on the container's queue.
- **Commits:** SlabKit `340b0244d5275d805c16525df802b4673afb8888` ("usm_vector: read/write event lists, collapse on access"), on main, pushed. CoSpan's slabkit pin is unchanged (P3 moves it).
- **Tests:** default suite: 169 test cases, 165 passed, 4 skipped (the OpenMP overlap skips), 1052443 assertions passed. CUDA suite three times: "All tests passed (1053886 assertions in 169 test cases)" each time.
- **Decisions taken:**
  - Both lists are emptied after a free, since the free waited on every event in them.
  - A free with no storage still waits on the lists.
  - An empty container's `reserve` leaves the lists alone.
  - Calls that submit nothing (fill of an empty container, a resize that does not grow, assign of an empty source) return the container's last write, or a completed event if there is none.
- **Open questions for Jack:**
  - An outside writer given `write_deps()` does not wait on the container's own earlier writes (writes never wait on writes), so a kernel that writes data() right after `fill()` may run before the fill unless the caller also passes `read_deps()`.
  - The collapse task runs on the container's slabkit queue, so it is ordered like any kernel task there, behind earlier kernel tasks of that stream.
