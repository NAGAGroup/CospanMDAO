# r05: Components and drivers, placements by context, push ranking (living spec)

Status: **agreed design (2026-10-13), implemented on branch r04-impl (unreviewed)**. The plan this was built from is `internal/impl-plan-r05.md`. This revision sits on r04 (flat model metadata).

## Purpose

Separate what a component *is* (declarations, forward steps) from where it *sits in the tree* (a node with a name, a device and a run policy). Key data placements by the context that owns the memory, so one host placement serves every device of a context. Make pushes between placements explicit and ordered, and let SlabKit make every copy between any two operands legal.

## 1. Hierarchy

### ComponentBase
- It no longer derives from `Subsystem`. It has no name and is never a tree node.
- It keeps: the declaration API (`declare_input`, `declare_output`, the ragged forms, `init_input`), `shape_of` and `set_shape`, `view_input`, the forward steps (`precompute_steps`, `step_pre`, `step_submit`, `step_post`, `gather_deps`, `record_launch`) and the stage interface virtuals.
- `ComponentBase::forward`, `run_setup`, `is_leaf` and the `events`/`add_dependencies` overrides are gone. `events()` returns `last_events_`; `add_dependencies` appends.
- An internal `bind(ModelMeta*, full_name, const Subsystem* driver)` gives the component its full name and its driver; `run_declare()` sets `declaring_` and `SetupScope` and calls `declare()`.
- `describe()` returns `"'" + full_name + "'"`.
- Holders are registered with name `ModelMeta::child_name(full_name, key)` and node `meta_->node_for(driver)`.

### Instance
- `Instance::owner_path` is new. `full_path()` is `owner_path.empty() ? name : owner_path + "." + name`.
- `declarer` stays a `const Subsystem*` (the node object): a component's holders carry the driver, with `owner_path` the component's full name; System aliases carry the System with its path; host views carry `nullptr` and an empty path.
- The `make_input`, `make_output`, ragged and alias factories take `(owner_path, declarer)`.

### ModelMeta
- `components` maps a full name to its `ComponentBase*`, filled by `bind`. `name_taken` also checks it.

### Drivers (`component_driver.hpp`)
- `detail::ComponentDriver : Subsystem` is a leaf that owns its component(s); `events()` and `add_dependencies` forward to them.
- `detail::SoloDriver : ComponentDriver` holds one component. `run_setup()` binds it (meta, own path, `this`) and calls `run_declare()`; `forward(rt)` runs `step_pre`, `step_submit`, `step_post`.
- `subsystem_desc` accepts `T` derived from `Subsystem` or from `ComponentBase` (a concept). `make_subsystem<T>` wraps a component in a `SoloDriver`. The Model's `add_subsystem` uses the same function. `children_are_stages`, `adopt_stage` and the stage branch of `add_child_unchecked` are deleted.
- `detail::FusedBase` becomes `detail::FusedDriver : ComponentDriver` and owns its stages (`std::vector<std::unique_ptr<ComponentBase>>` plus raw pointers). Its declaration scope is a private nested `Scope final : System` (`setup()` empty; public `connect`, `instance_at`, `tie_inputs_of`, `tie_outputs_of`, `describe`). `run_setup()` first binds the Scope and maps it to the driver's own node. `add_stage` checks `name_taken`, refuses a device or run policy, constructs the stage, binds it with full name `path() + "." + name` and driver `this`, records the name in `stage_names`, and runs `run_declare()`. `FusedGroup::setup` keeps its order: stages, ties, connects (`check_forward`, `internal = true`), all through the Scope. A stage's inputs are `f.s.x`; `get_subsystem("f.s")` still throws.
- `stage_checks` accept plain `Component`s only; custom and host components are refused by type.
- `SubsystemHandle` and `ModelAccess::subsystem` return drivers. `ModelAccess::component(Model&, full_name)` returns the `ComponentBase*`.

A lone component `c` is a leaf: its input is `c.x` and `get_subsystem("c")` returns the leaf. An error raised inside a component names its full path.

## 2. Interface

### Renames
- A component's `setup()` is `declare()`. Systems keep `setup()`.
- `pre_forward()` is `prepare_forward()`; the flag is `in_prepare_forward_`; messages say "legal inside prepare_forward() only".
- `post_forward()` and its call in `step_post` are deleted (nothing overrides it).

### Three component types, one static body name
`impl_traits` reads `&X::compute_forward` in every type.
- `Component<X>`: kernel bodies only (`sycl::id<R>`, or `GroupItem<R>` with optional Scratch). The user virtual `define_forward()` returns `create_launch_config(...)`; the internal `describe_launch()` replaces `compute_forward_impl`. A host or custom body in a `Component` fails its constraint.
- `CustomComponent<X>`: static `compute_forward(CustomOp&, const Inputs&, const Outputs&)`; no `define_forward`; `describe_launch()` builds `CustomOpLaunch`.
- `HostComponent<X>`: static `compute_forward(const Inputs&, const Outputs&)` with no Host tag (the type fixes the kind); the wrappers are over host USM and `default_storage_kind` returns host; `describe_launch()` builds `HostLaunch`. Inside, `SynthImpl` kind 3 keeps the `Host` tag in its operator and calls `X::compute_forward(in, out)`.

### Checks
Each type checks its body with a constraint on its default constructor: a requires-clause naming a concept over `SynthBody<X>` (X is complete there). `static_assert` is not used in changed code. `launch_descs_ok`'s custom and host branches are gone (only `Component` calls `create_launch_config`). `ComponentBase` does not mention `compute_forward`.

## 3. Placements by context

- `DataKey` is `{sycl::device dev; sk::kind kind; sycl::context ctx;}`. `DataKey(dev, kind)` sets `ctx = dev.get_platform().khr_get_default_context()`. Two keys are equal when their kinds are equal and, for `kind::host`, their contexts are equal; otherwise when their devices are equal. On a host key `dev` is the device of the holder that created it: it is used for allocation and as the initializer's device preference.
- `allocator_for`: the host kind becomes `sk::allocator<T>{q, k, dev}`, so the memory is allocated in `context_of(dev)`.
- A host view takes its holder's device (an alias's own device for an alias). `VarMeta::busiest_device()` is deleted. The view's host key reuses the context's host placement through `container_for`.

## 4. Push ranking

`VarMeta::push_from(const std::shared_ptr<DeviceDataBase>& src)` returns `std::vector<sycl::event>`. Targets are every live, bound data other than `src`; every copy uses `copy_from` (SlabKit makes each one legal). Targets are grouped by `key.ctx`:
1. Copy `src` into every target in `src`'s context. A host target there becomes `hostSrc`.
2. For each other context C, in data order:
   - C has a host placement H: copy `hostSrc` → H if `hostSrc` is set, otherwise `src` → H and `hostSrc = H`; then H → every device placement in C.
   - C has no host placement: copy `src` → D, the first device placement in C, then D → C's other device placements.
3. All copy events are returned.

`push_from` is used in `ComponentBase::step_post` (replacing the loop over the other placements), in the tied-output push in `place_tied_pushes` (events appended to `host->extra_events`), and in `Var<T>::initialize`'s copy-to-others (a read registered on each target as before). `populate` is unchanged.

## 5. The SlabKit rule (cross-context copies)

SlabKit commit `6cc1d7e` (after `e3ec700`; "copies: host and shared memory copy on the host in any context; fills of shared go through SYCL"). CoSpan relies on it, so every copy `copy_from` issues is legal.
- An operand is **host-side** when it is a pointer no context knows, `kind::host` or `kind::shared`. Device memory, OpenMP device memory included, is **device-side** (there is no OpenMP special case).
- Both operands host-side (host USM, shared USM or plain memory, in any context): a `std::memcpy` in a host task. Never staged.
- One device-side and one host-side operand: a single `memcpy` issued on the device side's device, only when the host side is plain memory or USM of that device's context; otherwise one host task stages it through a pageable `std::vector` (device leg on the device's own context, host leg a `std::memcpy`).
- Both device-side: unchanged (a direct copy when a route exists, otherwise staged through a host intermediate).
- `set` and `fill` take the host path only for host-side, non-shared memory. **Fills of shared memory go through SYCL**, because a host loop migrates every page to the host and back.
- **WARNING (WSL2/Windows):** copies that touch shared memory from the host assume the device supports concurrent access to shared allocations. CUDA on Windows and WSL2 is the canonical case where that guarantee does not hold: there the host must not touch shared memory while any kernel runs on that device, and such a copy can fault. Know your platform, and either wait for the device first or use device memory.

## 6. Ports

Mechanical renames in tests, examples, getting-started and the docs code: `compute_impl` → `compute_forward`; the override `compute_forward()` → `define_forward()`; `pre_forward` → `prepare_forward`; a component's `setup()` → `declare()`; components with a `CustomOp&` body derive `CustomComponent<X>`; those with a `Host` parameter derive `HostComponent<X>`, drop the parameter and `define_forward`.

## Tests added

- a lone component's input is `c.x`, and `get_subsystem("c")` is a leaf;
- an error inside a component names its full path;
- a fused stage is `f.s.x`, and `get_subsystem("f.s")` throws;
- `HostComponent` round trip (GPU producer, host component, GPU consumer);
- two host holders on devices of one context share one host placement;
- a view takes its holder's device;
- the `push_from` order on a variable with placements in two contexts (skipped with one backend).
