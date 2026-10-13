# r05 plan: components and drivers, placements by context, push ranking, SlabKit cross-context copies (2026-10-13)

Branches:
- CoSpan: r04-impl, from head 6a183aa.
- SlabKit: main, from a23b0c3.

One pass of edits, then one gate.

## A. SlabKit (ext/slabkit), first

1. **The `host_side` rule** (src/queue.cpp, `locate`, line 210). An operand is host-side when it is:
   - a pointer no context knows; or
   - `kind::host`.

   Shared memory is device-side (ruling 2026-10-13): it is accessed only through SYCL commands, because host access to shared memory is unsafe while any kernel runs on a device without concurrent managed access (CUDA on WSL2 and Windows), and that restriction covers the whole device.

   Delete the OMP special case: drop `|| is_host_backend(d)` and the now-unused `is_host_backend`, if nothing else uses it. OpenMP device memory is device-side like any other backend's.
2. **`copy_on`** (line 247):
   - (a) Both operands host-side: `std::memcpy` in a host task. Unchanged, legal across contexts.
   - (b) One device-side (device or shared kind) and one host-side: a single `h.memcpy` issued on the device side's device, only when the host side is a pointer no context knows, or USM whose context is `context_of(device-side dev)`. Otherwise stage it inside one host task:
     - device side to host side: device leg `memcpy` into a pageable `std::vector`, on the device's context via `legs` + `AdaptiveCpp_retarget{ctx, dev}`, `.wait()`; then `std::memcpy` into the host USM.
     - host side to device side: the reverse.
   - (c) Both device-side: unchanged.
   - The operand needs its context, so `locate` records the backend index or context. `strongest_answer` already returns the index.
3. **`set_on` and `fill_on`** follow the new `host_side`. OMP device memory now goes through SYCL memset or fill.
4. **Comments.** Update the comments at queue.cpp:192–206 and the header docs that describe the old rule.
5. **Tests.** Add them in tests/test_cross_device.cpp. Each skips when fewer than two backends are visible:
   - device ↔ host USM of another platform's context, both directions;
   - shared ↔ host across contexts;
   - host ↔ host across contexts (std::memcpy);
   - OMP device memory → CUDA device and back.

   Existing tests stay green.
6. **Gate.** Run `pixi run test` and `pixi run -e cuda test` in ext/slabkit. Commit as "copies: a SYCL command touches only its own context's USM; OpenMP is an ordinary backend" and push to origin main. Report the SHA.
7. **Repin.** In CoSpan pixi.toml, set `slabkit rev` to that SHA.

## B. CoSpan: spec and roadmap, before the code

- **Spec revision.** Write docs/spec/revisions/r05-2026-10-13-components-drivers-placements.md, containing every section C rule below as the agreed design, including:
  - the push ranking;
  - the SlabKit rule;
  - the WSL2/Windows caveat: CUDA there lacks concurrent managed access, so the host must not touch shared memory while any kernel runs.
- **Roadmap.** Add to docs/ROADMAP.md, under the path to public release: "C++20 audit: use concepts and constraints instead of static_assert throughout."

## C. CoSpan code

### C1. Hierarchy

**ComponentBase**
- It no longer derives from `Subsystem`. It has no name and is never a tree node.
- It keeps:
  - the declaration API (`declare_input`, `declare_output` and the ragged forms; `init_input`);
  - `shape_of` and `set_shape`;
  - `view_input`;
  - the forward steps (`precompute_steps`, `step_pre`, `step_submit`, `step_post`, `gather_deps`, `record_launch`);
  - the stage interface virtuals.
- Remove `ComponentBase::forward`, `run_setup`, `is_leaf`, and the `events`/`add_dependencies` overrides. They become plain members: `events()` returns `last_events_`, and `add_dependencies` appends.
- Add an internal `bind(detail::ModelMeta* m, std::string full_name, const Subsystem* driver)`, plus a `run_declare()` that sets `declaring_` and `SetupScope` and calls `declare()`.
- `describe()` returns `"'" + full_name + "'"`.
- `register_instance` registers holders with these arguments:
  - name: `ModelMeta::child_name(full_name, key)`;
  - node: `meta_->node_for(driver)`.

**Instance** (instance.hpp)
- Add `std::string owner_path`. `full_path()` becomes `owner_path.empty() ? name : owner_path + "." + name`.
- `declarer` stays a `const Subsystem*`, the node object:
  - for a component's holders, the driver, with `owner_path` = the component's full name;
  - for System aliases, the System, with `owner_path` = its path;
  - for host views, `nullptr` and an empty `owner_path`.
- The `make_input`, `make_output`, ragged and alias factories take `(owner_path, declarer)`.

**ModelMeta**
- Add `std::unordered_map<std::string, ComponentBase*> components`, keyed by full name and filled by `bind`. Forward-declare `naga::cospan::ComponentBase`.
- `name_taken` also checks `components`.

**New header include/naga/cospan/component_driver.hpp** (included by system.hpp, which includes component.hpp through it)
- `detail::ComponentDriver : Subsystem`:
  - `is_leaf` returns true;
  - it owns its component(s);
  - `events()` and `add_dependencies` forward to them.
- `detail::SoloDriver : ComponentDriver` holds one `std::unique_ptr<ComponentBase>`:
  - `run_setup()` binds the component (`meta_`, own path, `this`), then calls `run_declare()`;
  - `forward(rt)` runs `step_pre`, then `step_submit`, then `step_post`.

**system.hpp**
- `subsystem_desc` accepts `T` derived from `Subsystem` or from `ComponentBase`; use a concept.
- `detail::make_subsystem<T>` returns `std::make_unique<SoloDriver>(std::make_unique<T>(args...))` when `T` derives from `ComponentBase`.
- The Model's `add_subsystem` uses the same `make_subsystem`.
- Delete `children_are_stages`, `adopt_stage` and the stage branch of `add_child_unchecked`.

**fused_component.hpp**
- `detail::FusedBase` is renamed `detail::FusedDriver : ComponentDriver`. It owns its stages: `std::vector<std::unique_ptr<ComponentBase>>`, plus the raw pointers.
- Declaration scope: a private nested `class Scope final : public System` with `void setup() override {}`, and public using-declarations for `connect`, `instance_at`, `tie_inputs_of`, `tie_outputs_of` and `describe`. In its `run_setup()`, before anything else, `FusedDriver`:
  1. binds the Scope (`SubsystemAccess::bind(scope, path(), meta_)`);
  2. maps it to its own node (`meta_->adopt_sub(&scope, *meta_->node_for(this))`).
- Adding a stage (`add_stage(desc)`):
  1. Check `name_taken`.
  2. Refuse a device or a run policy; these are no longer possible through this path, since stages come from `fuse_components` items.
  3. Construct `T` from the descriptor arguments.
  4. Bind it with full name `path() + "." + desc.name` and driver `this`.
  5. Insert the name into `m.stage_names`.
  6. `run_declare()`.
- `FusedGroup::setup` keeps its order: stages, then ties, then connects with `check_forward` and `internal = true`. Ties and connects go through the Scope.
- `TieEntry` takes a `FusedDriver&`.
- `run_setup` then runs the ragged check.
- `forward` is unchanged in substance.
- `stage_checks` use `std::derived_from<T, Component<T>>`-style checks for plain Components only. Custom and host components are refused by type.
- Keep `get_subsystem` on a stage throwing.

**model.hpp**
- `SubsystemHandle` and `ModelAccess::subsystem` return drivers.
- Add `static ComponentBase* ModelAccess::component(Model&, std::string_view full)`, backed by `meta.components`.

### C2. Interface (component.hpp)

**Renames**
- `setup()` → `declare()`, components only; Systems keep `setup()`.
- `pre_forward()` → `prepare_forward()`. The flag `in_pre_forward_` becomes `in_prepare_forward_`. Update the messages ("legal inside prepare_forward() only").
- Delete `post_forward()` and its call in `step_post`. Verified: nothing overrides it.

**Static bodies**
- The static body is named `compute_forward` in all three types. `impl_traits` reads `&X::compute_forward`.
- `Component<X>`:
  - kernel bodies only: `sycl::id<R>`, or `GroupItem<R>` with optional Scratch;
  - user virtual `define_forward()` returns `create_launch_config(...)` (the old virtual `compute_forward`);
  - the internal `compute_forward_impl` becomes `describe_launch()`;
  - a host or custom body in a `Component` is a constraint failure.
- `CustomComponent<X>`:
  - static `compute_forward(CustomOp&, const Inputs&, const Outputs&)`;
  - no `define_forward`; `describe_launch()` builds `CustomOpLaunch` internally.
- `HostComponent<X>`:
  - static `compute_forward(const Inputs&, const Outputs&)`, with no Host tag since the type fixes the kind;
  - wrappers over host USM; `default_storage_kind` returns host;
  - `describe_launch()` builds `HostLaunch`. Internally `SynthImpl` kind 3 keeps the `Host` tag in its operator and calls `X::compute_forward(in, out)`.

**Checks**
- Each type checks its body with a constraint on its default constructor, a requires-clause naming a concept over `SynthBody<X>`. X is complete there. Do not use `static_assert` in changed code.
- `launch_descs_ok`'s custom and host branches can go: only `Component` calls `create_launch_config`.
- `ComponentBase` does not mention `compute_forward`.

### C3. Placements

**DataKey** (variable.hpp)
- Fields: `{sycl::device dev; sk::kind kind; sycl::context ctx;}`.
- Constructor `DataKey(dev, kind)` sets `ctx = dev.get_platform().khr_get_default_context()`.
- `operator==`: kinds equal, and for `kind::host` the contexts are equal; otherwise the devices are equal.
- `dev` on a host key is the device of the holder that created it. It is used for allocation and as the initializer's device preference.
- Update the constructions at shape_hook.hpp:212 and instance.hpp:134.

**Allocator** (runtime.hpp `allocator_for`): the host kind becomes `sk::allocator<T>{q, k, dev}`, so the memory is allocated in `context_of(dev)`.

**Views**
- host_view.hpp `make_view_instance`: `view->device = inst.device`. That is the holder's own device; for an alias, the alias's device.
- Delete `VarMeta::busiest_device()` and its comments.
- The view's host key reuses the context's host placement through `container_for`.

### C4. Push ranking

Add a new `VarMeta` member:

```
std::vector<sycl::event> push_from(const std::shared_ptr<DeviceDataBase>& src)
```

- Targets: every live, bound data other than `src`. Every copy uses `copy_from` (SlabKit makes each one legal).
- Group the targets by `key.ctx`. Then:
  1. Copy `src` into every target in `src`'s context. A host target there becomes `hostSrc`.
  2. For each other context C, in data order:
     - **C has a host placement H.** If `hostSrc` is set, copy `hostSrc` → H; otherwise copy `src` → H and set `hostSrc = H`. Then copy H → every device placement in C.
     - **C has no host placement.** Copy `src` → D, the first device placement in C, then D → C's other device placements.
- Return all copy events.

Use `push_from` in:
- `ComponentBase::step_post`, replacing the loop at 439–443;
- the tied-output push in model.hpp `place_tied_pushes`, appending the events to `host->extra_events`;
- `Var<T>::initialize`'s copy-to-others, which registers a read on each target as it does today.

`populate` is unchanged.

## D. Ports (same pass)

**Mechanical renames** in every built test (tests/CMakeLists.txt list), examples/gemver-chain/components.hpp, examples/ns-residual/components.hpp, getting-started/main.cpp, and the docs code in docs/site/getting-started.md (kept identical to main.cpp) and docs/site/concepts/*.md, plus the README snippets:
- `compute_impl` → `compute_forward`;
- the override `compute_forward()` → `define_forward()`;
- `pre_forward` → `prepare_forward`;
- a component's `setup()` → `declare()`, only in classes deriving from a component type;
- components whose body takes `CustomOp&` derive `CustomComponent<X>`;
- those taking `Host` derive `HostComponent<X>`, drop the Host parameter, and drop `define_forward`.

**Targeted fixes**
- test_component.cpp:848 uses `ModelAccess::component`.
- test_fused_groups.cpp:441,461 cast to `cs::detail::FusedDriver*`.

Leave the unbuilt chunked files untouched.

**Docs prose:** update the component concepts page for:
- the three types;
- `declare` / `prepare_forward` / `define_forward`;
- the static `compute_forward`.

**New tests (test_r04.cpp or a new test_r05.cpp added to CMake):**
- a lone component's input is `c.x`, and `get_subsystem("c")` is a leaf;
- an error inside a component names its full path;
- a fused stage is `f.s.x`, and `get_subsystem("f.s")` throws;
- `HostComponent` round trip: a GPU producer feeds a host component, which feeds a GPU consumer, and the values are right;
- two host holders on devices of the same context share one host placement;
- a view takes its holder's device;
- the `push_from` order, on a variable with placements in two contexts. Skip when one backend.

## E. Gate (once)

1. SlabKit: done in A.
2. CoSpan:
   - default suite;
   - cuda suite;
   - `gemver-chain --n 2048 --check` and `ns-residual --nx 1024 --ny 1024 --limiter minmod --bc walls --range-split all --check`, each on OpenMP and on CUDA;
   - `pixi run -e docs docs-build` (baseline 7 warnings);
   - getting-started built from a temporary copy under build/ pointing at the branch; it prints `3 5 7 9`.
3. Fix compile errors with targeted edits.
4. A test failure not explained by a rename: STOP and report it verbatim. Do not change expectations or the spec to pass.
5. When everything is green, commit on r04-impl as "r05: components and drivers, placements by context, push ranking" and push. Do not merge.
