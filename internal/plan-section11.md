# Section 11 implementation plan

The authority is `internal/design-review-2026-10-03.md` section 11. Where code, older sections, spec r02, the old tests or this plan disagree with section 11, section 11 wins. Jack settled the following on 2026-10-06:
- Work on `main` and push as we go.
- The lead chooses the user-facing spellings, extending the existing design language. Jack reviews them later.
- The onepass OpenMP bug is out of scope; it waits for Jack.
- Custom-op and host components keep their own launch path.
- The new spec revision is r03.

## How this runs

- **Phases and hand-offs.** The phases below run in order. The lead fleshes out each phase before handing it over. If a phase arrives less detailed than this plan, work from this plan plus section 11, and record every decision taken in the log.
- **The log, `internal/impl-log-section11.md`.**
  - Add an entry at the end of every phase, and whenever stopping mid-phase.
  - Every new user-facing name goes in its spelling table.
  - Anything that needs Jack's opinion goes under "Open questions for Jack", and the work continues on the most conservative reading.
- **Phase gate.** Every phase ends with `pixi run test` (default, OpenMP) and `pixi run -e cuda timeout 600 cospan-tests`, both green. Then commit and push.
- **Tests.** Tests that encode superseded behavior are rewritten to section 11, with the reason given in the log. They are never weakened to pass. New behavior gets new tests.
- **Final gate.**
  - Default suite once, and the CUDA suite five times.
  - SlabKit's suites on both backends.
  - `gemver-chain --variant all --check` on CUDA and OpenMP. Onepass on OpenMP is excluded and noted.

## P0: documents

Write this plan and the log.

## P1: AdaptiveCpp fork, toolchain release, adoption

The fork is never built locally. Its changes are verified by publishing through acpp-toolchain to staging, installing the result, and testing. Publishing a broken build to staging is acceptable. Iterate until the probe passes.

**P1a. Fork changes** in `/home/jack/projects/acpp-fork`, branch `naga/develop`. Both changes go in one commit, which is pushed.

1. **Indexed call sequence.**
   - **API.** `sycl::AdaptiveCpp_jit::dynamic_function_config::define_as_indexed_call_sequence(call, definitions, index_param)`, in `include/hipSYCL/sycl/jit.hpp`. It mirrors the existing `define_as_call_sequence` overloads (typed and `dynamic_function_id`). It is void-return only.
   - **Behavior.** At JIT time, every definition k (counting from 0) is called with all of the wrapper's own arguments, except that argument `index_param` is replaced by the integer constant k.
   - **Recording.** The config entry records the index position. That position is part of the configuration hash (`include/hipSYCL/runtime/kernel_configuration.hpp`, function-call specializations around line 204), so two configs that differ only in it never share a cache entry.
   - **Applying it.** `glue/llvm-sscp/jit.hpp` (around lines 275-283) passes the position into `LLVMToBackendTranslator::specializeFunctionCalls` (`include/hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp`, `src/compiler/llvm-to-backend/LLVMToBackend.cpp` lines 610-687).
   - **Wrapper.** For an indexed sequence, the wrapper function is always built, even with a single definition. The substituted constant has the parameter's integer type. A non-integer or out-of-range `index_param` is an error, both at the host API and in the pass.
   - **Plain sequences.** Their behavior is unchanged.
2. **Cross-translation-unit linking fix.**
   - **The bug.** `runtime_linker` (`glue/llvm-sscp/jit.hpp` line 239; the import list is built at 377-380) resolves only the kernel image's imported symbols. Those are functions the kernel calls but does not define (`src/compiler/sscp/TargetSeparationPass.cpp` lines 427-438). A dynamic-function definition from another translation unit is named only in the config, so it is never linked. The lookup then fails with "Could not find function call specialization target" (`LLVMToBackend.cpp` lines 631 and 652).
   - **The fix.** Before `runtime_linker` is constructed, add every replacement name from the config's function-call specializations to the list of symbols to import. Existing resolution then finds them through the other translation unit's exported symbols (`SYCL_EXTERNAL` marks them for export).
3. **Tests.**
   - Next to `tests/compiler/sscp/dynamic_function.cpp`, add tests for the indexed sequence: 1, 2 and 3 definitions, each checking the index it received.
   - Add a two-translation-unit case following the structure of `tests/compiler/sscp/export-all/`: a `define` and a sequence whose definitions live in the other translation unit.

**P1b. Release** from `/home/jack/projects/acpp-toolchain`.

1. **Dispatch.** Run `.github/workflows/build.yml` (workflow_dispatch) with:
   - `fork_sha` = the full pushed fork SHA;
   - `platforms` = `linux-64 linux-aarch64 osx-arm64 win-64 win-arm64`;
   - `publish` = true;
   - `build_number` = the next number not yet used on `naga-labs-staging` for that UTC date. Bump it for every republish.

   The recipe needs no edit. The workflow passes the SHA in (`ACPP_NAGA_SHA`) and the date in (`ACPP_NAGA_DATE`, UTC `YYYY.MM.DD`). The version is that date.
2. **Monitor.** Set up a monitor that fires when the `publish-linux-64` job of that run finishes, whether it succeeds or fails. Do not wait for the other platforms.
3. **Failures.** If linux-64 build or publish fails, read the log, fix it in the fork or the toolchain, and republish with the next build number.

**P1c. Adoption and probe.**

1. **SlabKit** (`/home/jack/projects/naga-labs-research/ext/slabkit`).
   - In `pixi.toml`, set `naga-acpp-tools` to `==<date>` and `c_compiler_version` and `cxx_compiler_version` to `<date>`.
   - Relock. If the channel index still lags, wait and retry; never fall back to an older version.
   - Run both suites, then commit and push.
2. **CoSpan.** Make the same pin changes. Also set the slabkit `rev` to SlabKit's new main SHA. Relock.
3. **Probe test in CoSpan.** Add `tests/test_acpp_dynfn.cpp`, plus a second translation unit `tests/acpp_dynfn_defs.cpp` that holds `SYCL_EXTERNAL` definitions. It checks:
   - a `define` across translation units;
   - an indexed call sequence across translation units, where each definition writes its index into a distinct output slot.

   It must pass on default and CUDA. If it fails, go back to P1a and fix it.
4. **Exit.** The existing suite and the probe are green on both backends with the new toolchain.

## P2: SlabKit container events

All work is in SlabKit's `usm_vector`, together with its users.

- **Two event lists per container.** Each container keeps a read list and a write list, and these are the only event truth.
- **The container's own operations go into those lists.**
  - Copy-out is a read on the source.
  - Copy-in, fill, resize and `from_host` are writes on the destination.
  - The container's own write operations wait on both of its lists.
  - A copy between two containers waits on the source's writes and on both of the destination's lists. It then records a read on the source and a write on the destination.
- **Collapse on access.** Any access to a list holding more than one event collapses it into a single event, using an empty `single_task` that depends on them all, and stores that event back.
- **API.**
  - Accessors for "events to wait on before reading" and "events to wait on before writing".
  - Registration calls for an outside read and an outside write.
  - Frees, deallocation and `sync()` wait on both lists.
  - Replace or remove `last_event`, `events()` and `register_event(s)` as needed. Update every SlabKit caller and its tests.
- **Tests.**
  - Readers don't wait on each other.
  - A writer waits for in-flight readers.
  - Writes don't wait on writes.
  - A copy orders correctly against both containers.
  - Collapse keeps every covered event.
  - A free waits on everything.
- **Ship it.** Push SlabKit, then repin CoSpan's slabkit `rev` and relock.

## P3: CoSpan data model and events

Section 11.1 and the event parts of 11.3.

- **Variable and holders.**
  - The variable is the shared metadata, with no event lists.
  - Holders (input, output, alias, host view) hold a `shared_ptr` to the variable and a `shared_ptr` to their own device's container.
  - Declaring a holder creates the empty container for its device if there isn't one yet. The variable keeps weak pointers to the containers, and that set is never shrunk.
  - Merges absorb the loser's device entries.
- **Materialization.**
  - The materialized flag lives on the variable.
  - The first materialization is global: it allocates every device container. It initializes unless the materializing holder is an output.
  - After that, a holder checks only its own device. If that container is empty, it allocates it and queues a copy from the first live device.
  - Remove `adopt_live_data` and `attach_missing`, or fold them into this.
- **Submissions.**
  - A submission waits on its input containers' writes and its output containers' reads, plus any depends_on events.
  - It registers a read on its inputs and a write on its outputs.
  - The component keeps its last event for depends_on.
- **Postcompute.** It registers the write on every output container, then pushes to the output's other devices by copy assignment, unless the variable is push-suppressed.
- **Lifetime.** Deletion and lifetime need no explicit waits; frees wait on the containers.
- **Tests.** Rewrite the event-ordering tests to these rules. Add:
  - a destruction gate test: a kernel spins on a host flag, and destroying the model doesn't complete until the flag opens, cleanly;
  - exact pool-in-use numbers (capacity rounded up to whole nodes, summed over live containers on the device).

## P4: Setup, wiring and the System loop

Sections 11.2 and 11.3.

- **Calls legal anywhere.** Wiring calls (connect, promote, tie_*, declare_init) are no longer restricted to setup.
- **Run-policy override.** `override_run_policy(path, policy)` sets a subsystem's run policy by path and is legal anywhere.
- **Cycles.**
  - Connect records a cycle entry on the variable and keeps the target's initializer.
  - At first-forward materialization, a cycle variable runs its initializer. If it has none, that's an error naming the variable.
- **Tie devices.**
  - Both tie calls take a device, defaulting to the default device.
  - `tie_outputs` checks the members against that device at the call. The owning System checks again at run time.
  - `tie_inputs` uses its device for the alias and the initializer.
- **Tie marks.** Input and output marks are kept separately. A tie refuses a member whose variable already carries the tie mark of that kind.
- **Tied outputs.**
  - The tie sets push-suppressed on the variable.
  - Members only register their writes.
  - The System pushes through the alias, inside its compute loop, right after the last child holding a member. Remove the pushes from postcompute.
- **System precompute.**
  - It rebuilds transients, then calls the user's `pre_forward`.
  - It never runs children's precomputes.
- **System compute** loops over the children: each child's precompute, compute and postcompute. A child whose run policy has ended is deleted right after its postcompute.
- **Replay.** It skips a logged call whose other side no longer exists.
- **Marks.** Collected at setup; no scope rule.
- **Update-device.** Covers aliases; the outermost call wins.
- **Merges.**
  - Connect: the source wins.
  - Tie: the alias wins.
  - Promote: the member wins.
  - Definitions merge (unspecified absorbs specified; a conflict is an error at the call).

## P5: Shapes and hooks

Section 11.5.

- **set_shape.** `set_shape(out, d0, d1, ...)` takes one entry per dimension: a number or `cs::as_is`.
  - A number on a specified or hook-owned dimension is an error at the call.
  - Ragged shapes follow the same rule.
- **Tied members.** The tie call marks member output holders as unable to set their shape.
- **Hook targets.** `shape_hook` may target an input, an input tie alias, an output tie alias, or an untied output. The hook runs before the producer.
- **Hook implementation.** Each `shape_hook` call creates one templated pseudo-component with only a forward. It replaces the type-erased `HookStep`. It runs every forward right before the first child holding its target.
- **Removals.**
  - The source-ordering setup check.
  - The "two sets agree" rule.
- **Kept behavior.** A gone source keeps the last shape. `fixed_at_run` is still specified at the call.

## P6: Host access

Section 11.4.

- **Model.**
  - `get_input(path)` returns a handle with `view()` and `update(init)`.
  - `get_output(path)` returns a handle with `view()`.
  - `update_input` and `make_view` stay as shorthands.
- **Component view.** A component gets a protected `view_input(input)` that makes a host view of one of its own inputs, using its precompute's queue. There is no equivalent for outputs.
- **Views.**
  - A view can be made at any time. It fails only if the shape can't be resolved.
  - The host is just another device, pushed each forward.
  - `sync` waits on the host container's write list.

## P7: Fused groups

Section 11.6.

- **Rename.** Fused component becomes fused group.
- **Segments.** A segment is consecutive stages sharing groups.
  - **Range mode.** The default is that each stage sees only its own groups. The alternative is the whole range.
  - **Local memory.** Per segment, either `partitioned` (sum; each stage gets its own slice; the default) or `shared` (max; stages see the same block).
- **Boundary primitives between segments.**
  - `ticket_order`: tickets for the whole launch, and per-group completion recording.
  - `global_barrier`: ticket order, plus waiting for every earlier group.
  - With no primitive, plain hardware ids are used.
- **Item wrapper (GroupItem).** It exposes:
  - the current segment and the number of segments;
  - each segment's group count;
  - the group's index within its segment, starting from 0;
  - waiting on a specific earlier group, or on all earlier groups;
  - group barriers, reductions and local memory.

  Waits are collective. A null ticket state is UB, with no branches.
- **Launch path.**
  - One dynamic-function definition per component type, used through the P1 indexed call sequence.
  - A bare component is a group of one.
  - Each component keeps a 1-element argument `usm_vector`, copied from a host member each forward.
  - The fuser's pointer array is set once, at its first precompute.
  - The geometry array is filled each forward.
  - The work-group shape check against launch configs happens each forward.
  - The counters are reset each forward.
- **Events.** The launch is a read on every stage input, the argument containers and the geometry array, and a write on every stage output.
- **Connects** are allowed when they go forward in stage order; a backward connect is an error.
- **Removals.** `fuse_strategy` seq/par, the packing buffers and the per-launch static configs that this replaces.

## P8: Example and test cleanup

- **gemver-chain.**
  - Port it to fused groups.
  - Remove the barrier variant: `components.hpp`, `systems.hpp`, `main.cpp` and the README.
  - Parallel becomes segments with no boundary.
- **REVIEW markers.** Resolve every one in `tests/` and `examples/` to section 11. Put any that need Jack in the log.
- **Leftovers.** The smoke tests, and the stale comments.

## P9: Spec and close-out

- **Spec.** Write `internal/spec/revisions/r03-<date>-section-11-as-built.md`: the design as built, with exact API names. Point `internal/spec/README.md` at r03 as current.
- **Design review.** Update the status line in `internal/design-review-2026-10-03.md`.
- **Final gate**, as described above.
- **Final log entry.** Test counts, every spelling decision, and the open questions for Jack.

## Initial spelling decisions

- `define_as_indexed_call_sequence(call, definitions, index_param)`, in the AdaptiveCpp fork's `dynamic_function_config`.
- `cs::fuse_components("name", cs::connects_desc({...}), cs::segment(stage descs...), cs::global_barrier(), cs::segment(...), ...)`:
  - a bare stage desc is a one-stage segment;
  - `cs::ticket_order()` is the other boundary primitive.
- `cs::segment<cs::local_mem::shared>(...)` and `cs::segment<cs::local_mem::partitioned>(...)` (the default). A range-mode option: `cs::range_mode::own` (the default) or `cs::range_mode::whole`.
- `override_run_policy("b.c", policy)`, a System method.
- `view_input(input)`, a protected Component method.
- `model.get_input(path)` returns `.view()` and `.update(init)`; `model.get_output(path)` returns `.view()`.
- `set_shape(out, cs::as_is, 12, cs::as_is)`.
- GroupItem methods:
  - `segment()`, `num_segments()`, `segment_groups(s)`, `group_in_segment()`;
  - `wait_group(s, g)`, `wait_earlier_segments()`;
  - `local<T>(n)`.
