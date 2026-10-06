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

- The join task (collapse) breaks ordering in CoSpan on the OpenMP backend; we ship without collapse (SlabKit `7655b84`). Lists drop completed events past 64 entries. Root cause in AdaptiveCpp still to find.
- Section 11: outside writes never wait on writes. Added (SlabKit `e86ae9f`): an outside writer still waits on the container's own write operations (resize fill, reallocation copy, copy-in), which are not outside writes.
- add_subsystem is still setup-only; should it follow the anywhere principle?

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

### 2026-10-06 UTC, P1c: toolchain adopted, dynamic-function probe
- **Done:** SlabKit and CoSpan both pin naga-acpp 2026.10.06 (`naga-acpp-tools`, `c_compiler_version`, `cxx_compiler_version`) and are relocked; the channel index had no lag. CoSpan's slabkit `rev` is unchanged (dc51516; P3 moves it). The probe is `tests/test_acpp_dynfn.cpp` plus `tests/acpp_dynfn_defs.cpp` (the `SYCL_EXTERNAL` definitions live only there).
- **Commits:** SlabKit `67f0b715036cbbf3872fff78e5800b76fcb83692` ("build: naga-acpp 2026.10.06"); CoSpan `cdec366fe0b222631141de2b82168934138e0735` ("build: naga-acpp 2026.10.06; dynamic-function probe").
- **Probe cases, all green on default and cuda:** (a) define across TUs; (b) call sequence across TUs; (c) indexed sequence of three across TUs (`{1,12,23,0}`); (d) indexed sequence of one across TUs (`{7,0,0,0}`); (e) same-TU indexed control (`{1,12,23,0}`).
- **Tests:** SlabKit default `169 | 165 passed | 4 skipped`, cuda `All tests passed (1053886 assertions in 169 test cases)`. CoSpan default `182 | 178 passed | 4 skipped` (2008 assertions), cuda `All tests passed (2031 assertions in 182 test cases)`.
- **Toolchain run 37441431821:** published linux-64 and the others; `test-linux-aarch64` and `test-osx-arm64` passed. `test-linux-64` was still running when this was written.
- **Decisions taken / findings:**
  - **A `dynamic_function_config` must outlive every kernel that uses it, until the queue has been waited on.** The first probe versions kept the config as a local inside the launch lambda. It was destroyed on return, before the wait, and AdaptiveCpp's JIT thread read it afterwards: every case aborted (`std::bad_alloc`, once `std::length_error`) right after `kernel_cache: Cache MISS`, before any `jit: Specializing function call` line. That was a bug in the probe, not in the fork. With the config declared in the test scope all five cases pass. The fused-group design has to keep the config alive in the same way.
- **Open questions for Jack:** none.

### 2026-10-06 UTC, P3: containers own events; holders own containers; launch events registered at submit
- **Done:**
  - Events live in the SlabKit `usm_vector` lists, not in per-variable `EventLists`. A reader waits on `read_deps()` (the write list); a writer waits on `write_deps()` (the read list plus the container's own writes). Frees wait on the containers, so deletion needs no explicit waits.
  - Holders hold their own device's container. Declaring a holder creates the empty, unbound container; `bind()` makes the `usm_vector` at the first materialization that reaches it. The first materialization is global; later ones check only the asking holder's container and copy from the first live container with data (`populate`). `attach_missing`, `adopt_live_data`, `copy_index`, `give_data`, `has_data`, `make_data`, `metas_of` and `EventLists` are gone.
  - A component registers its launch on its containers at submit, in `record_launch`: a read on every input container and a write on every output container. Postcompute only pushes. (Before, the write was registered in postcompute, which left a window between submit and postcompute.)
  - `sync_view` waits on the host container's write list. Shape hooks, the device scan, the ragged column build and the fused component use the same lists.
  - Fix found on the way: `populate` returned early for a ragged variable with no elements, so a host view of an all-empty hooked ragged output had no index arrays and `block_offsets()` threw "its block offsets live on the device". It now sizes the index arrays.
- **Commits:** SlabKit `e86ae9f` ("usm_vector: outside writers wait on the container's own writes"; the message lost its apostrophe) and `7655b84` ("usm_vector: no collapse; event lists are handed out whole"); CoSpan `bf9e339` ("P3: containers own events; holders own containers; launch events registered at submit"), which pins SlabKit `7655b84`.
- **Tests:** SlabKit default `172 | 168 passed | 4 skipped` (1052532 assertions; one run in four had `test_host_work.cpp:295`, "a group mixes kernel and memory subtasks", fail with `9 == 29`: that test has no `usm_vector` in it and passed 20 of 20 alone), cuda `All tests passed (1053976 assertions in 172 test cases)`. CoSpan default `182 | 178 passed | 4 skipped` (2017 assertions), cuda `All tests passed (2040 assertions in 182 test cases)`. `WeightedHistogram*` 20 runs: 20 of 20.
- **Decisions taken / findings:**
  - **The collapse join task broke ordering in CoSpan on the OpenMP backend.** SlabKit collapsed a list of more than one event into one event with an empty `single_task` over it. With that, the CoSpan histogram test (a hook reading a value an earlier kernel wrote) read zero instead of 8 in about half the runs, and `ini.y` zeros, zero shapes in `test_shapes` and heap corruption (`malloc(): unaligned tcache chunk`) showed up in the full suite. A join whose task increments a counter in USM (so it is not empty) failed the same way (11 of 20). With no collapse the histogram passes 20 of 20 and the full suites are clean. A standalone SlabKit test (a spinning kernel on an unordered stream, its event registered on a container, a reader depending on the container's write list) passes with the old collapse too, so it did not reproduce the bug. **The cause is not known.** SlabKit now ships without collapse: `read_deps()` and `write_deps()` return the lists whole, and a registration that makes a list longer than 64 events drops the events that have completed.
  - Waiting on a kernel's own event right after submit also made the histogram pass, so the symptom is a dependency that was not honoured, not a wrong value computed.
  - The own-writes rule (SlabKit `e86ae9f`) is kept: it is valid on its own, a fresh allocation's zero-fill must come before an outside writer. It did not fix the flake.
  - Registering the launch event at submit (above) is kept: it closes a real window and does no harm.
- **Open questions for Jack:**
  - The join task (collapse) breaks ordering in CoSpan on the OpenMP backend; we ship without collapse. Lists drop completed events past 64 entries. Root cause in AdaptiveCpp still to find.
  - Section 11: outside writes never wait on writes. Added (SlabKit `e86ae9f`): an outside writer still waits on the container's own write operations (resize fill, reallocation copy, copy-in), which are not outside writes. (Also in the list at the top.)
  - Not run: the P3 tests for the destruction gate, two readers, a second holder on a new device, input-triggered materialization and the exact pool numbers (they follow in the next commit).

## P3: tests (destruction gate, materialization, exact pool numbers)

- Commit 702fba9. New file tests/test_events.cpp (in tests/CMakeLists.txt): destruction gate (a spinning kernel on a host flag; the model's destructor does not return until the flag opens), two readers not waiting on each other (checked through their events and the container's read list; skipped on OpenMP), a second holder on a new placement (a host view gets its own bound container, a copy), and the first materialization asked by an input (global: the other holder is bound and sized before it runs; it initializes).
- Every pool REVIEW marker is settled with exact numbers by the rule (capacity rounded up to whole 1024 B nodes, summed over live containers), in test_model.cpp, test_system.cpp and test_flavors.cpp. support.hpp gains `pool_node_bytes` and `pool_bytes_of`. No disagreement with the rule.
- Suites: default 186 cases (181 passed, 5 skipped), cuda all passed (2097 assertions in 186 cases).

## P4: setup, wiring and the System loop

Commit f2b72ae. Tests: tests/test_p4.cpp (new), plus edits in test_model.cpp and test_system.cpp. Decisions:

- **Wiring anywhere.** connect, promote, tie_inputs, tie_outputs and init_input lost the setup-only guard. add_subsystem, depends_on, shape_hook and the marks keep it (not in the P4 list). A merge where both variables are materialized throws at the call: "merging two materialized variables is not defined (yet)" (it carries the verb and the System). **Deferred:** merging two live variables.
- **override_run_policy(path, policy).** Public on System; the policy is a property_list (empty = eternal; transient, run_n, do_once). The path goes through child Systems; it sets requested and counter on the slot in the owning parent. Beyond the spec: a child that is gone and is not made transient is rebuilt at once (so an override from pre_forward can revive a transient that was deleted); a transient is rebuilt at the next precompute. Not logged for replay: rearming an ancestor rebuilds the slot with its declared policy.
- **rearm.** The pre_forward-only restriction is gone (decision under "don't restrict to phases"). Still protected. The in_pre_forward_ flag is removed.
- **Cycles.** connect marks the merged variable `cycle` (consumer registered before producer, as warn_cycle already decided) and keeps the target's initializer (and its init_device) unless the survivor is already materialized. The first materialization of a cycle variable runs the initializer whichever holder asks, takes the shape from it, and throws if there is none: "'<var>' closes a cycle ('<dst>' reads '<src>', which runs after it) and has no initializer". declare_init and update_input accept an initializer on a cycle variable. Beyond the spec: update_input on a cycle variable after the first forward re-initializes it at the next forward. The replay of a connect carries the cycle mark and initializer too. The warning stays.
- **Tie devices.** tie_inputs and tie_outputs take an optional device, default = the default device (tie_outputs gets it as a new LAST parameter, after policy). tie_inputs always has an init_device now (before: none, so the busiest device). tie_outputs: members must be on the tie's device at the call; the owning System re-checks at the start of every forward's loop, before any submission (the error names the member, both devices and the tie), by re-resolving the member paths it was declared with (rebuilt members are checked, deleted ones skipped). set_device on a System moves its aliases (already so) and the tie input's init_device.
- **No ties of ties.** VarMeta carries tied_input and tied_output; the tie call refuses a member whose variable has the mark of its kind (checked after the older "already tied" checks, so only aliases and their promotions reach it). merge_into ORs the marks, push_suppressed and cycle of the loser into the survivor. Promoting a tie works. Existing tests that tie promotions of plain members still pass (those are not ties).
- **Tied-output pushes.** tie_outputs sets push_suppressed on the variable. The System pushes through the alias right after the postcompute of the last child (by first path segment) that holds a member and will run, and before that child is deleted if its policy ended. The pushes are gone from System postcompute. Beyond the spec: a tie whose member children are all skipped or deleted does not push that forward.
- **System precompute / compute.** Already as specified in the code on main (precompute rebuilds transients then calls pre_forward and runs no child precompute; the loop runs precompute, compute, postcompute per child and deletes with no wait). Verified by a test of the order of pre_forwards; header comments updated.
- **Replay.** Already skipped a logged connect whose other side is gone (the rebuilt end rejoins a variable that still lives, else stays fresh). Test: a rearmed consumer whose do_once producer was deleted runs on a fresh (zero) variable. Only the connect case has a test; a promoted name or alias as the missing side is not tested.
- **Marks.** The scope check (check_mark_scope) is removed; marks are collected at setup and stay. The old error tests became "collected" tests. A mark under a transient keeps its path after the child is deleted (its instance pointer expires).
- **Merges.** Source wins on connect, alias on tie, member on promote, definitions merge; adopt_containers is unchanged. Test: a connect after setup keeps the host holder's container (identity) in the device variable.
- **Open.** Two materialized variables merging stays deferred. A declare_init on a materialized free input does not mark it stale (not in the spec). The cycle test covers a two-component cycle; nested-System cycles use the same detection as before (child positions in the calling System) and are not tested.
- Suites: default 208 cases (198 passed, 10 skipped, 2164 assertions); cuda all passed (2225 assertions in 208 cases); the two-device cases ran on cuda (7 cases, 43 assertions).

## P5: shapes and hooks

Commits 1cad7ab (code and converted tests) and the tests/log commit after it. Decisions:

- **set_shape(out, e0, e1, ...).** One entry per dimension, an integer or `cs::as_is`; an entry count that differs from the rank throws. `VarMeta::set_by_producer` refuses every number first (`refuse_number`), then sets; nothing changes when a check throws. A number on a specified dimension throws at the call, even when it matches ("dimension N is declared as V; set_shape('x') takes a number only on an unspecified dimension (pass cs::as_is)"); on a hook-owned one: "dimension N is owned by a shape hook; ... cannot give it a number". Ragged keeps the RaggedShape copy form and (n, counts); n may be `cs::as_is`; `(n, cs::as_is)` sets n alone. The old full-range form, `check_same_forward`, `dim_text`, `dim_equal`, `set_forward_` and `set_log_` are removed (no "two sets in one forward must agree").
- **Ownership rules.** `SetterId::Kind` gained `definition` (fixed dimensions are applied by the definition). `check_setter`: a producer or initializer setting a fixed dimension fails ("a specified dimension comes from the definition"); the hook rules are unchanged.
- **Tied members.** `Instance::tied_member`, set by `tie_outputs` and by the replay of a tie on a rebuilt member. `settable_output` throws the existing tied message when `tied_member` or the variable's `tied_output` is set (the second covers a member reached through a promotion).
- **Hook targets.** A hook may target an untied output; it runs before the first child holding it (the producer, which passes `as_is` for hooked dimensions). A tied member output is refused at the call: "... hook the tie alias instead ('t')".
- **Templated step.** One `detail::ShapeHookStep<D...>` per shape_hook call (descriptor types as template parameters, compile-time dispatch), stored in the System as a `HookEntry` behind `detail::HookBase` (only `forward(Runtime&)`). Behavior kept: from_value reads batched into one wait; counts_from / offsets_from use the device scan and bring back only the total.
- **Run point.** `compute_hook_slots` and the source-ordering check are removed. `place_hooks()` finds each hook's slot (the first child holding the target) at the start of the System's compute; the loop runs the hook when it reaches it. A gone source keeps the last shape; a shape never known throws the shape-not-known error. Tests that expected the setup error now check that setup succeeds.
- **Anywhere.** `shape_hook` and `depends_on` lost the setup-only guard. add_subsystem and the marks keep it.
- **declare_init / init_input** on an already-materialized free input marks it stale, like update_input (`System::init_input` passes restale; replay does not).
- **Beyond the spec.** A ragged hook marks both dimensions (n and counts) as hook-owned, so a declared N is checked against the source. `shape_hook` on a ragged target also accepts `(dense descriptor, cs::dim::as_is())`: the hook owns n alone and the producer sets the counts with `set_shape(out, cs::as_is, counts)`.
- **Test fixtures.** Producers whose output has a declared (or downstream-fixed) dimension now pass `as_is` (ShProd, VBins, Gen modes, InitAffine); Sum3's inputs are unspecified so a connected producer's number is legal. Old "declared as N but ... gave M" tests became "takes a number only on an unspecified dimension" tests. New tests (tests/test_shapes.cpp, [p5]): rank-3 {hook value, 7, 3}, number on a declared dimension, number on a hooked dimension, entry-count mismatch, hook on an untied output, hook on a tied member, shape_hook and depends_on after setup, a late init_input, ragged (as_is, counts) with a hooked n.
- **Open.** add_subsystem's guard (see the open questions). Docs other than this log (README, design review, plan) still show the old set_shape in places; not rewritten.
- Suites: default 217 cases (207 passed, 10 skipped, 2187 assertions); cuda all passed (2248 assertions in 217 cases).
