# Phase 1 decisions

Decisions made while implementing Phase 1, for Jack's review. Newest last.

## D1 — CustomOp is for native backend work only
A custom operation hands a Component's device pointers to native backend work
on the backend's own stream (a vendor library call, a native kernel launch). Its
body runs when the work is submitted, not when it executes (AdaptiveCpp's
enqueue-custom-operation contract), so it may only enqueue asynchronous work on
the native queue. Host work is not a custom operation's job: that is what
SlabKit's async_host is for. The to_host / from_host helpers in r01's CustomOp
were leftovers and are removed. On the OpenMP backend there is no native queue;
the body still runs and receives correct device pointers.

## D2 — Library shape (spike result)
The spike (tests/spike) built and linked both variants in both environments; the difference showed at run time.

- Variant A (placeholder and outer kernel in a SHARED library, stage definitions instantiated in the consumer): FAILED on both backends. omp (OpenMP host device) and cuda (RTX 4090) each failed all five checks and aborted with the JIT error "Could not find function call specialization target spike::kb_fn<...>, was the function emitted to device code?", for Scale and for Reverse.
- Variant B (everything in one translation unit): PASSED all five checks (define/Scale, sequence/Scale+AddOne, parallel/Scale, parallel/AddOne, group/Reverse) on omp and on cuda.
- Placeholder default body: not needed and not tried. Variant A did not fail to link (the host side resolved spike::kb_entry_r1 with a declaration-only placeholder); its failure is the JIT error above, which the default body was not specified to address.

Choice: B: cospanmdao is header-only (INTERFACE library); placeholders, outer kernels and stages all compile in the consumer's translation units — because the JIT could only find the stage definitions when they were emitted in the same translation unit as the outer kernel, and not when the outer kernel came from a shared library.

## D3 — Host components: a fourth body kind, host work through async_host
Host work in a model is a Component whose body's first parameter is the tag
`cospan::Host`: `void operator()(Host, Inputs, Outputs) const`, with the same
Input/Output wrappers as every other body. It is submitted with the fork's
async_host through SlabKit's async_host_in_order: a host callable run by the
runtime that joins the kernel order as an ordinary work item. Plain async_host
is not ordered against memory tasks, so it could run before its inputs
finished copying in.

A host component has a device like any Component (its parent's, or its own if
placed). That device selects the SYCL context that owns its memory, as for any
USM allocation. Every input and output of a host component is HOST-kind USM on
that device: pinned host memory the body reads and writes directly, not
managed memory, so CUDA-under-WSL2's shared-memory limit does not apply. Data
reaches it and leaves it by the ordinary rule for a producer and a consumer on
different placements: root storage stays keyed by device, a host component's
roots are host-kind, and the existing per-device copies in post-compute move
data in and out. Custom operations stay separate and are for native backend
work only (D1).

## D4 — Stages and the outer kernel instantiate in the consumer's translation unit
Follows from D2. Everything CoSpan provides is a header; a Component's device
function and the outer kernel that calls it are both instantiated in the
translation unit that builds the model. The tested pattern is one translation
unit per model. Models split across several translation units are tested in
the test pass; the result is recorded here.


## D5 — Layering of the entry classes
r01 §3 names IOEntry and typed leaves but not where shared code sits.
`detail::IOEntry` is the type-erased base: identity, shape, marks (via the
root), set operations, placement, and the pure virtuals (update_shape, reshape,
validate_connection, allocate, point_wrapper, sync_other_devices).
`StorageEntry<T>` (T = C++ element type) implements everything that depends
only on the element type: shape updates, reshape, validate_connection,
allocation, sync between devices, primary-storage read/write. `TypedEntry<DT,
Rank>` adds the wrapper pointing (`point`, `make_initializer`).
`InputEntry`, `OutputEntry` and `AliasEntry` are the leaves. Reason: storage
depends on the element type only, so it is not instantiated per rank. r01 §3.

## D6 — AliasEntry is the System-level alias, not the view
r01 does not say what a slice peer is. `AliasEntry<DT,Rank>` is the
System-level alias (promotion or tie alias): no member wrapper, given flags, and
as a root it owns storage like any other. A slice is not a separate class: any
peer may carry an optional `SubarrayRef` (`view()`), and its wrapper geometry is
the view extents over the ROOT storage with the ROOT contiguous strides and the
offset folded into the pointer. r01 §3, §4.

## D7 — Substrates are declared in subsystem.hpp (amended in step 2a; was metadata.hpp)
`MemorySubstrate { optional<sk::DeviceArena> arena; optional<sk::DevicePool>
pool; }` and `Substrates = unordered_map<sycl::device, MemorySubstrate>` live in
subsystem.hpp (D19); metadata.hpp includes it because `IOEntry::allocate(queue,
Substrates&)` takes them. r01 §7.1 defines the substrate but not the header. The
System (later step) will own and fill the map; the smokes fill one by hand.

## D8 — MarkRecord sits with the geometry types
`MarkRecord { root, optional<SubarrayRef> region, Bounds, optional equals,
Scaling }` (plus Slice, Bounds, Scaling) is declared in metadata.hpp beside
SubarrayRef, not in a header of its own, because IOEntry stores marks (on the
root: design_var_, constraint_, objective_) and Model aliases the type. r01 §8.

## D9 — SubarrayRef and subarray<R>
r01 uses `SubarrayRef` and `subarray<R>` without defining them. Defined as
`struct SubarrayRef { int rank; array<size_t,3> offset; array<size_t,3>
extents; flat_size() }` (1-padded to three dimensions) and
`subarray<R>(array<size_t,R> offset, array<size_t,R> extents)` for R in 1..3,
called as `subarray<2>({r, c}, {rows, cols})`. `Slice { SubarrayRef src; }`.
Minimal choice. r01 §4, §8.1.

## D10 — Shape helpers
`from_source_shape<Rank>()` returns `std::array<FromSourceDim, Rank>`,
`from_input_shape<Rank>(name)` returns `std::array<FromInputDim, Rank>`,
`from_input_array(name)` returns the `FromInputArray` marker (expanded against
the member rank at declaration). DimSpec is a concept over integral,
FromSourceDim, FromInputDim and FromInput. Stored per dimension as
`detail::DimRecord`, which owns its input name as `std::string` because the
spec is built from a `string_view` that need not outlive the declaration. r01 §4.

## D11 — A slice view must have the root rank
r01 §8.1 gives the subarray "in the ROOT rank", while §4 and §3.3 let a peer
rank differ. Implemented: a peer with a view must itself have the root rank and
the view rank must equal it; `geometry()` throws std::invalid_argument
otherwise. A rank-changing slice is not supported in step 1. validate_connection
(inputs only) refuses a view that is not inside the root and a flat-size
mismatch (the view size for a slice). r01 §4, §8.1.

## D12 — update_shape and the resolved extent
`update_shape()` applies the DimRecords: fixed stays, FromSourceDim reads the
source (the view extent for a slice, else the root extent), and FromInputDim /
FromInput extents are taken as already resolved by the owning Component (a
separate step that stores `DimRecord::extent`; in the smoke test through
`EntryAccess::set_resolved_extent`). If the shape changed it calls `reshape`.
Unresolved dynamic extents start at 0. An unconnected input with a
FromSourceDim throws. r01 §4.

## D13 — Allocation route
`allocate` runs on roots only, once per registered storage device (primary
first). Route: a host-kind root takes the OS route on its device (D3); an
arena-eligible root (root, fully fixed, not transient, device kind) takes the
arena and NEVER falls back (an arena is a budget; a missing arena throws
std::logic_error); everything else takes the pool if the device has one, else
the OS route for device kind. Peers and host-kind roots count 0 static bytes.
r01 §4.4, D3.

## D14 — Host access to the primary storage and sync
`read_primary` returns a host copy of the WHOLE root from the primary device
(usm_vector::host_view for device kind; queue.wait() then get_access for host
kind). `write_primary` does `from_host` on the primary and then
`sync_other_devices`. `sync_other_devices` copy-assigns the primary
`usm_vector` to every other slot (routed, ordered copy; needs equal queues). A
`root.host_view()` for a peer on a non-primary device reads the primary
copy; r01 does not say which. r01 §4.2, §8.3.

## D15 — EntryAccess and HostViewAccess are test-only stand-ins
Until System and Component exist, `detail::EntryAccess` (merge, register_device,
set_device, set_storage_kind, set_transient, set_view, set_resolved_extent) and
`detail::HostViewAccess` are the sanctioned way to drive the friend-only
mutators from the smoke test. The leaf constructors are public for the same
reason. Both go away or narrow when System lands.

## D16 — HostView constructors take a non-const IOEntry&
r01 §8.3 declares them over `const detail::IOEntry&`. Reading the primary
storage goes through non-const usm_vector paths, so they take
`HostView(detail::IOEntry&, sk::queue&)`.

## D17 — HostViewMut writes back on commit and on destruction (superseded by D139: no mutable host views)
`HostViewMut` is move-only and snapshots the region. commit() and destruction write the buffer back unconditionally; there is no dirty tracking (from_host owns its copy of the bytes, so an unneeded write-back costs one copy). A moved-from view writes nothing. A failed write-back in the destructor cannot throw; it writes the message to stderr. r01 §8.3.

## D18 — `sk` alias
`namespace sk = ::naga::slabkit;` is declared in `naga::cospan` (metadata.hpp).
SlabKit does not provide the alias. Internal, not part of the public surface.

## D19 — subsystem.hpp is standalone
It holds `MemorySubstrate`/`Substrates` (D7), `SubmitTarget`, `ExecCtx`, `LaunchDescription`, and `Subsystem`. It forward-declares `KernelParts` and `System` only, so metadata.hpp (substrates) and kernel.hpp (`SubmitTarget`) can include it without a cycle. `LaunchDescription` is declared before its users; r01 §7.1 lists it after them.

## D20 — LaunchDescription gains `is_host()`
Pure virtual beside `is_custom_op()`; `kernel()` returns `nullptr` by default (custom-op and host descriptions). The owner routes host bodies on `is_host()`. r01 §7.1 names only `is_custom_op()`.

## D21 — `path()` is stored, and `run_system_init` is pure in Subsystem
`Subsystem` keeps `name_`, `path_` and `owner_`, written by the owner (System through friendship; `detail::SubsystemAccess::place` stands in until System exists); `path()` is not computed. `run_system_init()` is pure in `Subsystem` and a no-op in `ComponentBase` (a Component's System pass does nothing).

## D22 — Runner bodies live in ComponentBase
Friendship with `detail::IOEntry` is not inherited, so every runner body that touches no KB-typed state (`run_setup`, `get_exec_info`, `prepare_exec`, `run_init`, `run_pre_compute`, `run_post_compute`) is in `ComponentBase`. `Component<KB>` overrides only `compile()` and `run_compute_forward()`. It keeps no `CompiledKernelMeta`; the config pointer is its only compiled state. r01 §6.3 puts all runners in `Component<KB>`.

## D23 — A Component keeps the queue and substrates from prepare_exec
`ComponentBase` stores `sk::queue*` and `Substrates*` in `prepare_exec` and uses them to allocate roots and to read values on the host; it holds the whole map because root allocation takes `Substrates&` (`ExecCtx::substrate` is one device's substrate only). The device is recorded in `get_exec_info`.

## D24 — "Known shape" and allocation timing (empty-root rule superseded by D139)
A root is allocated when its shape is known, meaning `flat_size() > 0`. Roots with a known shape are allocated in `prepare_exec`; the rest (value-driven or derived shapes) at the first `run_pre_compute` that resolves them, after `pre_forward()`. A root left empty is never allocated. Allocation is idempotent per device (D13).

## D25 — Entry and wrapper additions
`IOEntry::has_storage()` (pure; the root has been allocated) guards `point_wrapper`, which throws on an unallocated root. `InputEntry::member()`/`OutputEntry::member()` return the wrapper a declaration bound. `friend struct CustomOp;` was added to `Input`, `Output` and `Initializer`; `friend struct detail::ScratchLayout;` to `Local`.

## D26 — CustomOp::device_ptr
Only `device_ptr` exists (D1). It returns `const value_type*` for an `Input` and `value_type*` for an `Output` or `Initializer` (the wrapper's `data_`). r01 §5.1 says `W::value_type*`, which loses the `Input` constness.

## D27 — KernelBodyC and launch_rank_v
`KernelBodyC` accepts exactly one of plain, group, custom-op and host (`HostBody` is a fourth kind, D3): a trivially copyable `KB`, `Inputs`, `Outputs` and `Scratch`. `kernel_rank_v<KB>` is 0 for custom-op and host bodies; `launch_rank_v<KB>` (never 0) types the `range` parameters of `create_launch_config`.

## D28 — GroupItem stores stage-local group ids
`GroupItem<R>` holds the physical `nd_item`, a stage-local `group_id_` and `group_range_` (from `StageRecord::group_offset` and `group_range`), instead of r01's `group_offset_`. `get_group()` returns the physical `sycl::group<R>` for barriers and reductions. Only `detail::ItemFixup` constructs one.

## D29 — Scratch layout by structured bindings
The Locals of a `Scratch` are enumerated by structured bindings over an aggregate (`field_count` by conversion probe): at most 6 members, all `Local`. `layout_scratch` aligns each Local to its element type, sums, and rounds the total to 16 bytes; `fixup_scratch` sets each pointer from the work-group's local base.

## D30 — Default work-groups and group counts
Default work-group: rank 1 `{128}`, rank 2 `{8,8}`, rank 3 `{4,4,4}` (`detail::default_wg`); a `wg` argument overrides it for group bodies. Physical launch: `global = {G*wg0, wg1, wg2}`, `local = wg`, groups laid along dim 0 only. A group body has `group_range[d] = ceil(range[d]/wg[d])` and `group_count` their product. A plain body has `group_count = ceil(flat_extent / wg_size)` and `group_range = {group_count, 1, 1}`, with the tail masked in `kb_fn`.

## D31 — Individual launches capture typed structs by value
`detail::KernelLaunch<KB>` (in component.hpp) captures one `Payload { StageRecord; Inputs; Outputs; { KB; Scratch } }` by value in the outer kernel's lambda. There is no staging allocation. `KernelParts.record` offsets are 0-relative to its spans (`input_off = output_off = body_off = 0`, `scratch_off` = the Scratch's offset in the body span); the owner re-bases them. A launch with `flat_extent == 0` submits nothing. Fused waves (step 3) stage the same bytes through device buffers.

## D32 — compile() builds the config once
`Component<KB>::compile()` makes one `dynamic_function_config` and calls `define(&kb_entry_rR, &kb_fn<KB>)` for the body's rank, naming both functions in source as the spike did. `create_launch_config` before `compile()` throws. Custom-op and host bodies throw `logic_error` "step 2b" from `compile()`.

## D33 — FromInput is honored on own inputs and outputs
Shape resolution (`run_pre_compute`) covers every own entry, inputs first, then outputs, with cycle detection. `FromInputDim` reads the sibling input's extent; `FromInput` reads the driving input's value with `take_snapshot` (integer dtypes only; a negative value throws), allocating the sibling root first if it is still unallocated. r01 §6.5 says "each own input", but its example (a) uses `FromInput` on an output.

## D34 — Initializers on an unallocated root (empty-root rule superseded by D139)
`fill_*`, `zero_*` and the initializer getters are legal in `init()` only. `fill_*`/`zero_*` write the root's PRIMARY storage, then sync the other devices. On a root that is unallocated and empty (`flat_size() == 0`, a shape still to be resolved) they are a no-op, since a fresh allocation is zero-filled; on an unallocated non-empty root they throw. r01 §6.7(a) `zero_output("bins")` conflicts with §11.2.6 "An initializer on unallocated storage throws."

## D35 — Declaration overloads
`declare_input`/`declare_output` accept: a variadic pack of `DimSpec` (count = Rank); a `std::array<S, Rank>` (what `from_source_shape`/`from_input_shape` return); and the `FromInputArray` marker, expanded against the member's Rank. `get_inputs()`/`get_outputs()` return `std::vector<std::reference_wrapper<const detail::IOEntry>>` in declaration order.

## D36 — create_launch_config overloads
`(range, in, out, body = {})` for plain and scratch-free group bodies; `(range, wg, in, out, body = {})` for scratch-free group bodies; `(range, wg, in, out, scratch, body = {})` for `HasScratch`; `(in, out, body = {})` for custom-op and host bodies, which `static_assert` "step 2b" when used. The scratch parameter is typed `detail::scratch_t<KB>` so a body without Scratch still declares the class.

## D37 — Post-compute sync and host-kind storage
`run_post_compute` refreshes the other-device copies of every own output ROOT that has storage on more than one device. A Component's I/O uses device storage, host storage for a host body (`default_storage_kind()`).

## D38 — TestDriver is test-only
`detail::TestDriver` (in tests/smoke/smoke_component.cpp) stands in for System: `run_setup`, `get_exec_info`, arena sized from the returned bytes plus 4096 per component and a 1 MiB pool, `prepare_exec`, `compile`, `run_init`; per forward `run_pre_compute`, `run_compute_forward`, `submit`, `run_post_compute`. Components are connected with `EntryAccess::merge(producer_out, consumer_in)`; producers are added before consumers.

## D39 — Root storage is keyed by Placement (amends D3: "root storage stays keyed by device")
`Placement {sycl::device device; sk::kind kind;}` (with `==`) is declared in metadata.hpp beside `IOKind`, in `naga::cospan`. A root's storage slots are keyed by Placement, not device: `register_device` -> `register_placement`, `storage_on(device)` -> `storage_on(placement)`, `storage_devices()` -> `storage_placements()` (primary first), `sync_other_devices` -> `sync_other_placements`, `raw_storage_on` and `Slot` likewise. Keying by device alone cannot hold both a device-kind and a host-kind copy of one root, which the host-kind rule (D3) requires when a GPU component feeds a host component on the same device. The primary slot is the root's own placement (the declarer's device plus the declarer's `default_storage_kind()`). `IOEntry::placement()` is added: this entry's (device, storage kind). A peer's wrapper points at `root->storage_on(peer.placement())`. Device-kind components behave as before.

## D40 — Registration in get_exec_info
Each entry gets `device_ = parent_device` and `storage_kind_ = default_storage_kind()`. A root sets its own placement as primary (`set_primary_placement`: moved to the front even if an input peer's owner registered first). An input peer registers its (device, default kind) on its root if absent. An output peer tied to a root must have a placement equal to the root's, else `logic_error` (a producer cannot write one storage from two placements).

## D41 — Host-kind slots cost no arena bytes
`arena_eligible()` still requires a device-kind root, so a host-primary root has `static_bytes() == 0`. A host-kind SECONDARY slot on a device-kind root adds nothing to the exec-info byte total; a device-kind secondary adds the root's `static_bytes()` once per device, as before. Routing (`StorageEntry::route(q, subs, placement)`): a non-device placement takes the OS route `allocator<T>{q, kind, placement.device}` (the device chooses the context; SlabKit already had `(queue, kind, device)`, so there is no SlabKit gap); a device placement takes its device's arena if the root is arena-eligible, else the pool if the device has one, else the OS. A device-kind secondary of a host-primary root is therefore pooled.

## D42 — sync_other_placements
`sync_other_placements` copy-assigns the primary `usm_vector` to every other slot: usm_vector's routed copy keeps the target's own allocator, needs the same queue, and depends on both slots' last events. Device<->host and cross-device movement is thus one mechanism, the ordinary copy rule. `run_post_compute` calls it for every own output root with more than one placement; `write_root` (initializers) calls it after writing the primary.

## D43 — Ordering of a host component in a chain
producer kernel -> device->host copy (memory task) -> `async_host_in_order` -> host->device copy (memory task) -> consumer kernel is ordered by the stream alone: memory tasks are ordered against kernel-order tasks in both directions, and `async_host_in_order` joins the kernel order. No events pass between Components. Smoke case d runs this chain on one device; it passed on omp and 5/5 runs on cuda.

## D44 — Custom-operation launch (D1)
`detail::CustomOpLaunch<KB>` captures Inputs, Outputs and the body by value. `submit(SubmitTarget&)` runs `h.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih) { CustomOp op{ih}; body(op, in, out); })` inside a command group, submitted with `t.group->submit(t.device, cgf)` under a group and `t.queue->submit(t.device, cgf)` otherwise. `is_custom_op()` is true. Custom-op I/O is device-kind USM.

## D45 — Host launch (D3)
`detail::HostLaunch<KB>` captures by value and runs `body(Host{}, in, out)`. `submit`: with no group `t.queue->async_host_in_order(t.device, f)`; under a group `t.group->async_host(t.device, f)`, because `sk::queue::group` has no in-order variant (ordering against kernel tasks is the group's). `is_host()` is true. `default_storage_kind()` is `kind::host` for a `HostBody` component, so its inputs and outputs are host-kind USM on its device.

## D46 — compile() and create_launch_config for the non-kernel bodies (amends D32, D36)
`compile()` builds nothing for a custom-op or host body (no `dynamic_function_config`, no `cfg_`). D36's step-2b `static_assert` and the throw are gone: `create_launch_config(in, out, body = {})` exists twice, constrained by `CustomOpBody<KB>` and by `HostBody<KB>`, returning `CustomOpLaunch<KB>` and `HostLaunch<KB>`.

## D47 — Custom-op smoke covers plumbing only
Case e records `device_ptr(in)` and `device_ptr(out)` into a `std::atomic<const void*>` pair captured by pointer in the body config, enqueues nothing, and checks the pointers equal the roots' (device, device) storage and that a following plain component reads the zero-filled output (w == 1). Native-stream work is not tested here: it needs a vendor library (the CUDA runtime headers and library), not present in the dev-cuda environment at this commit. The order check is weak by design: the recorder writes nothing, so it shows that the stream still runs the consumer after the custom-op node, not that a native write is visible to it.

## D48 — Smoke driver constraints
`TestDriver` runs each component's post-compute right after its submission, which is what moves the host copy between the GPU producer and the host component. `Placement` is used unqualified in tests through `using cs::Placement;`.

## D49 — Native CUDA custom-op test deferred (follows D47)
Attempted and reverted; nothing from the attempt is committed (pixi.toml, pixi.lock and the smoke files are back at 142a58c). What was tried: `cuda-cudart-dev = "*"` in `[workspace.dependencies]` and `[feature.cuda.dependencies]`; the solver picked cuda-cudart-dev 12.9.79 with no conflict against naga-acpp-runtime-cuda 2026.09.29, and `pixi install -e dev-cuda` succeeded. Smoke case f (a custom op taking `op.h.get_native_queue<sycl::backend::cuda>()` and calling `cudaMemcpyAsync` device-to-device, then a plain +1 component checked against a host reference) was gated in CMake by `find_package(CUDAToolkit QUIET)` and in the source by `COSPAN_SMOKE_NATIVE_CUDA` plus `defined(SYCL_EXT_HIPSYCL_BACKEND_CUDA)`.
What failed: `find_package(CUDAToolkit)` did not succeed in dev-cuda, so case f was never compiled. Build 1: no hint, "CUDAToolkit not found". Fix round: `-DCUDAToolkit_ROOT=<.pixi/envs/dev-cuda>`, same result. Observed layout: `cuda_runtime.h` exists at `<env>/targets/x86_64-linux/include/cuda_runtime.h` and is absent from `<env>/include/`; there is no nvcc in the environment. Both builds compiled and passed smoke_component 10/10 on cuda (RTX 4090) with case f printing "skipped: CUDAToolkit not found at configure time", so no result exists for the native-stream call itself.
Not reached, so still unknown: whether `SYCL_EXT_HIPSYCL_BACKEND_CUDA` is defined in the generic (SSCP) build, which decides whether `get_native_queue<sycl::backend::cuda>` is usable at all, and whether the cudart link works. Next attempt should start from the layout above (either give CMake the `targets/x86_64-linux` include dir and `libcudart` directly, bypassing `find_package(CUDAToolkit)`, or add a package that supplies nvcc) and check the macro first. D47 stands: custom-op native-stream work is untested.

## D50 — System is a Subsystem with setup()/init() hooks; structure is legal in setup() only
`System : Subsystem`. A user System overrides `setup()` (declares children, aliases, ties, marks) and optionally `init()` (path-addressed initializers). Every structural verb (`add_subsystem`, `connect`, `promote`, `tie_*`, `add_design_var`/`add_constraint`/`add_objective`) throws `std::logic_error` outside `setup()`; the initializer verbs throw outside `init()`. Reason: the exec-info, placement and transient passes assume the graph is frozen once `run_setup` returns. r01 §4.

## D51 — Error-message policy
Every System error is `naga::cospan: <System 'path' | the root System>: <verb>: <reason>`. Misuse of the API (wrong phase) is `std::logic_error`; bad wiring (unknown path, fan-in, kind or dtype mismatch, bad view) is `std::invalid_argument`. The reason names the offending path and, where useful, the fix ("tie before connecting, or connect from the alias"). The root's `describe()` reads "the root System" when its path is empty. Reason: a wiring error must be locatable from the message alone, without a debugger. r01 §4 (no wording given).

## D52 — Path resolution
Paths are dotted and descend through child Systems; the last segment names an alias of the owning System or an entry of a Component; a path may not descend past a Component; empty segments are refused. `find_child`/`find_alias`/`locate` are private; `entry_at(path)` is the public host-access hook and `get_subsystem(path)` returns the child. Reason: one resolver serves connect, ties, marks and initializers, so all error messages agree. r01 §10.3.

## D53 — declarer_ tracking
Every `IOEntry` records `declarer_` (a `const Subsystem*`), exposed as `declarer()`. `ComponentBase::register_entry` sets it for Component entries; System sets it on the alias entries it makes. Reason: transient marking walks `declarer() -> owner()` to find which System an entry lives in; without it no root was ever marked (found in the case c run of ddf2930). r01 §4.4.

## D54 — Transient roots: how and when they are marked
`mark_transient_roots()` runs at the END of the ROOT System's `run_setup` (`owner() == nullptr`), not per System, because peers declared later by an outer System decide whether a root escapes. A root declared inside a transient System's subtree, with no peer declared outside it, belongs to the INNERMOST qualifying transient System (`transient_by_`, `transient_owner()`). Such a root is not `arena_eligible()` and takes no arena bytes. The walk uses a `const_cast` on `owner()`, since marking is a setup-time mutation of a `const` accessor chain. r01 §4.4.

## D55 — release/revive storage verbs
`IOEntry` gains pure virtuals `release_storage()` and `revive_storage(sk::queue&, Substrates&)`, implemented in `StorageEntry` (release clears the slots and sets `released_`; revive clears the flag and calls `allocate`). `System::release()` drops the storage of the roots it owns and sets `released_` only if storage was actually dropped, so an empty transient System never reports released. `System::revive()` runs before the children's loop and `release()` after it. r01 §4.4.

## D56 — Revive route and re-init
`revive()` allocates through the pool route (transient roots are not arena-eligible, D41), recurses into released nested transient Systems, then re-runs BOTH init passes (Component, then System, children before owners) over the subtree, so a revived root has its initial values again. Nothing outside the subtree re-inits. Reason: a released root has lost its contents, so the init contract must hold on every forward. r01 §4.4.

## D57 — Host access to a released root throws
`HostView`/`HostViewMut` on a released root throw `throw_released()`: "'x' belongs to transient System 'path' and its storage is released; it exists only while that System runs". Reason: a silent empty view would read as valid data. r01 §4.4.

## D58 — Alias factory
`alias_factory::make_alias_entry(DType, name, IOKind, dims, can_source, can_target)` builds an alias entry from RUNTIME dtype and rank: a fold over the 13 `DType` values via `index_sequence`, then a rank 1-3 dispatch. Reason: `promote` and the ties know the dtype and rank only from a member entry at setup, not as template arguments. r01 §4.

## D59 — promote: alias kind and flags
`promote` makes a PEER alias with the kind of the member's ROOT and a snapshot of the root's `can_be_source_`/`can_be_target_` flags, copies the member's view, and sets its dims from the member. Reason: an alias must behave in connect exactly as the entry it stands for; the kind and flags the member itself carries can be stale after a tie. r01 §4, §8.1.

## D60 — connect checks
`connect(src, dst[, Slice])` checks four flag levels (src, src root, dst, dst root), dtype compatibility, and view composition per §8.2 (offsets add as `B.offset + A.offset`, `A` within `B`, view rank equals the root's rank). A view is applied to EVERY entry of dst's old set, and an identity connect from a viewed src passes that view on. A second connect into one input (fan-in) throws. r01 §8.1, §8.2.

## D61 — Inputs cannot be connect sources
A free input, and a tie alias root, cannot be a connect source; only an output root can. Consequence seen in smoke case a: a Slice has to read an output vector (`yy`), so `z` cannot be sliced for `obj`. Reason: an input has no producer, so "connect from it" would have to mean "share its root", which is what a tie already is. This restriction is not in r01 (gap; see the D50-D73 report).

## D62 — Tie aliases are new roots (the shape rule is superseded by D139 part 2)
`tie_inputs(members, alias)` requires each member to be a free input (target-not-source); `tie_outputs` requires source-not-target output roots. The alias is a NEW root whose set merges the members' sets (`IOEntry::merge(into, from)` needs two roots). Shape and dtype come from the first member (or that member's root) with a fixed shape; if none has one, `tie_inputs` throws (a `tie_outputs` of all value-driven members is allowed since the D138 amendment, r01 §4.3 SHAPE). Every member's `can_be_target_`/`can_be_source_` becomes false, so a later connect into a tied member is refused. r01 §4.3.

## D63 — A tie alias can be tied
A tie alias may itself be a member of another tie (case a: `tie_inputs({"mda.z", "obj.z"}, "z")` where `mda.z` is a tie alias of d1.z and d2.z); the member's whole root set is merged. Reason: it lets ties compose up the hierarchy, as promotions do. r01 §4.3.

## D64 — tie_outputs restrictions
Members must be output roots with no input-kind peers; dtypes must be atomic unless `allow_plain` is passed (`TieOutputsPolicy`). Reason: several producers writing one root is only defined for atomic accumulation; anything else is a race. r01 §4.3.

## D65 — get_exec_info ordering and per-child context
A System reports its own alias ROOTS first, then its children, then its own PEER aliases (the promotions). Each child gets its own `ExecCtx` (`child_ctx_`), and placed devices are held by the owner (`placed_`). Reason: reporting roots first avoids double-counting arena bytes for a promotion and lets a tie alias fix the device before its members register; peers last means their roots already exist. r01 §11.2.

## D66 — Alias extents are inert (superseded by D139 part 2: promotions mirror live, ties own the shape)
An alias copies its dims at setup and does not follow a dynamic root's resolved extent; refresh is deferred to the Model step. (Amended by D138: the dynamic alias root of a `tie_outputs` follows its members' resolved extents.) Reason: nothing at System level resizes a root yet. r01 §7.2 (gap).

## D67 — Typed access by path is Component-only
`get_input`/`get_output<DT,Rank>(path)` resolve Component entries only; an alias has no typed wrapper (r01 says aliases "point their wrappers" without a member API). `entry_at(path)` and the HostView path work for aliases. r01 §7.2 (gap).

## D68 — Marks are checked at setup
`add_design_var` needs a free input root that `can_be_target()` and is not dynamic ("connected inputs are computed, not designed"). `add_constraint`/`add_objective` need an output root; an objective needs flat size 1. Regions and sizes are checked at declaration, so the marked root needs a fixed shape. Marks ride on the root and the last mark of a kind wins (a later `add_constraint` on the same path replaces the first). r01 §8.4.

## D69 — property_list and the add_subsystem overloads
Properties are a SYCL-style `property_list{property::device{d}, property::transient{}, ...}`. `add_subsystem<T>(name, args...)` and `add_subsystem<T>(name, props, args...)` are kept apart by `leads_with_props_v`; a `property_list` first argument is required to select the second (a bare property is a constructor argument). Reason: constructor arguments and properties are both variadic, so a marker is the only unambiguous split. r01 §4.

## D70 — Helper-form plain System
`add_subsystem(name, props, child<T>(...)...)` creates a PLAIN System (no user class) with the given properties and populates it with the `child<T>` declarations; the non-template `add_subsystem(name, props = {})` makes an empty one to fill through paths. Reason: it lets a transient or device-placed group be declared in one line. r01 §4.

## D71 — Properties not yet supported
`property::parallel` and `property::fused{Mode, wg}` (with its own nested `Mode`) are declared but throw "step 3b" in `add_child`. `property::transient` on a non-System throws. `property::device` equal to the parent's device is a no-op placement, not an error. Reason: parallel and fused are the next step; a silent ignore would run a different schedule than declared. r01 §4.

## D72 — EntryAccess narrowed
Only `EntryAccess::set_transient` is removed, since System now owns transient marking (D54). The other members stay because `smoke.cpp` and `smoke_component.cpp` build entries by hand. r01 §4.2.

## D73 — Test-only RootDriver
`detail::RootDriver` in smoke_system does what a Model will: place the root, `run_setup`, `get_exec_info`, substrates (arena if the total is nonzero, with 4096*64 slack; pool 1<<20), `prepare_exec`, `compile`, `run_init`, `run_system_init`; `forward()` is `run_compute_forward()` + `q.wait()`. It is not public API. r01 §11.2.

## D74 — ParallelSystem children are Components, checked at compile time
`ParallelSystem::add_subsystem<T>` is constrained to `T` derived from `ComponentBase`. Its declarations hide every `add_subsystem` of System, so a System child, a nested Parallel and the untyped `add_subsystem(name, props)` (explicitly `= delete`d) have no viable overload. Reason: a System cannot discern its children at runtime (r01 §1 need 2.2 parallel, P5); the type system is the only place to refuse. r01 §7.5.

## D75 — Only property::device on a ParallelSystem child, checked at runtime
The property-list overload accepts a `property_list` and throws `invalid_argument` if it holds `parallel`, `transient` or `fused`. Reason: a `property_list` is a runtime value, so its content cannot be a compile-time constraint; the child's device is the one property that means something inside a group. r01 §7.5.

## D76 — The runner's order
`run_compute_forward`: `revive()`; pre-compute ALL children; compute ALL (collect the descriptions; a null description is a `logic_error`); the sibling-shape check; ONE `queue->submit_group` in which each description is submitted to its child's own `ExecCtx::target()`; post-compute ALL after the group returns; `release()`; return null. Reason: the group is one stream item, so pre/post work and the check (which reads only metadata) sit outside it. r01 §7.5, §11.3.1.

## D77 — ctx.group is set only during submit_group
An RAII `GroupScope` sets `group` on the Parallel's `ctx_` and on every `child_ctx_` entry for the extent of the `submit_group` lambda and resets it to null on any exit. Reason: a host body under a group must use `group.async_host` (D45) and nothing outside the group may see a dangling group. r01 §7.3.

## D78 — The sibling-shape check (subsumed by D99)
Subsumed: the sibling-shape check is removed from ParallelSystem. Its FromSourceDim case is now one instance of the no-connections rule (D99), which covers every input rooted at another stage's output, not only shape-dependent ones, and is shared with the parallel FusedSystem through `wave_rules.hpp` (D130). Reason: one rule for every unordered set of stages. r01 §7.5.

## D79 — The helper form builds a ParallelSystem; amends D71
`add_subsystem(name, props, child<T>(...)...)` with `property::parallel` builds a ParallelSystem; `fused` still throws "step 3b"; `parallel` and `fused` together throw `invalid_argument`. A TYPED child given `property::parallel` throws unless its type derives from ParallelSystem. Reason: the flavor selects a runner, which only the helper form can build without a user class. r01 §7.5.

## D80 — `detail::parallel_of` keeps the include order acyclic
system.hpp forward-declares ParallelSystem and names it through `detail::parallel_of<Children...>::type`, a dependent name, so the type only has to be complete when the helper form is instantiated. parallel_system.hpp includes system.hpp; use `naga/cospan/cospan.hpp` (or parallel_system.hpp) wherever the helper form is used. Reason: avoids a circular include without a separate translation unit. r01 §7.5.

## D81 — System access changes for flavors
`revive()`, `release()`, `describe()`, `ctx_`, `child_ctx_` and `children_` are `protected`; the rest of System's state stays private. `populate<S>` (private) builds the helper form's child through `add_child<S>` and raises the new System's declaring flag for the adds. Reason: a flavor's runner needs the transient contract and the per-child contexts; nothing else. r01 §7.2.

## D82 — Case d's compile-time check is a concept
`static_assert`s use `concept CanAdd<P, T> = requires(P& p) { p.template add_subsystem<T>(name); }`. Reason: a requires-expression with an invalid expression outside a template is ill-formed rather than false; a concept over dependent parameters yields false. The device-only-property, typed-parallel and flavor rules are runtime and are checked by the throws they raise. r01 §7.5.

## D83 — RootDriver moved to tests/smoke/driver.hpp
The driver is shared by smoke_system and smoke_parallel, and now exposes `info` (what the root's `get_exec_info` returned), `enqueue()` (a forward without a wait), `wait()` and `forward()`. Reason: the Parallel smoke enqueues three forwards back to back (§11.3: a loop of forwards need not wait). Test-only. r01 §11.2.

## D84 — An exception during a forward leaves the subtree unreleased
If a child's pre/compute (or the sibling check) throws after `revive()`, `release()` is not run. Reason: a forward that throws is an error state, not a supported flow; the RAII for that would hide the failure. Known consequence, not fixed here. r01 §4.4.

## D85 — Arena over-reservation not investigated in this step
The check of get_exec_info's byte total against the distinct arena-eligible roots (suspected from reading the code, not measured: a consumer declared before its producer may count a root's bytes twice, once from the input peer and once from the root's owner) is deferred to its own dispatch. Reason: user's scope for 3b-1. r01 §4.2, §11.2.
Measured since (D133, tests/test_model.cpp, `[model][memory]`): expected 1008 B (distinct roots), arena capacity 263152 B = 1008 + 262144 slack, in use 1008 B after setup and after a forward. Capacity matches, so no double counting was seen in that shape and the test is not `[!mayfail]`. The exhaustive check over every declaration order is still open.

## D86 — Kernel-shape virtuals on ComponentBase
`kernel_rank()`, `is_group_body()`, `has_scratch()`, `is_custom_op()`, `is_host()` and `definition_id()` are virtual on ComponentBase; `Component<KB>` overrides them from the body's traits. `definition_id()` throws for custom-op and host bodies. Reason: a FusedSystem holds type-erased children and must plan a wave (rank, scratch, barrier) without knowing the body types. r01 §5.4.

## D87 — A wave is one chained call sequence, with no barrier ids
`compile()` builds `define_as_call_sequence(wave_entry_id(rank), defs)` where `defs` is the children's definition ids in order and nothing else: the framework inserts no barriers, in either mode (`kb_barrier` and `wave_barrier_id` are deleted). The stage function pointer must be a literal at its call site (reflection), so ids come from `stage_definition_id<KB>()` and `wave_entry_id` in kernel.hpp. Reason: one launch runs the whole chain; ordering inside a work-group is the stage bodies' business (Jack, 2026-09-29). r01 §5.5.

## D88 — Wave geometry
Groups run along dim 0 (`wave_nd_range<R>`); dims >= rank are 1. A plain stage's group_count is ceil(flat_extent/wg_size), a group stage's is the product of ceil(range[d]/wg[d]); the launch uses the max over stages and each stage returns early for groups beyond its own count. Sequential mode has group_offset 0. Reason: stages of different sizes share one launch. r01 §5.6.

## D89 — Work-group choice
The FusedSystem's `wg` if given, else the stages' common wg (a mismatch throws, naming the stages), else the rank's default. Reason: every stage of a wave runs in the same nd_range. r01 §5.6.

## D90 — Local memory: per-stage slices in sequential mode, the max in parallel
Sequential mode gives each scratch stage its own slice of the local buffer: `StageRecord::local_off` is a running sum of the stages' scratch sizes, each rounded up to 16 bytes, and the local size is the total. Parallel mode gives every stage `local_off` 0 and the local size is the maximum, because the stages occupy disjoint groups. A stage no longer reuses another's local memory, so no barrier between stages is needed (D87). Reason: slices make the barrier-free chain safe. Replaces the earlier single shared buffer with a barrier between scratch stages. r01 §5.6.

## D91 — REMOVED: no chained-dependency check in a sequential wave
Removed (Jack, 2026-09-29). Sequential waves make no inter-stage access checks: a stage may read any earlier stage's output whatever the ranges or view shapes, and the stage ranges may differ. Reason: the check tried to prove metadata-level safety for accesses the stage bodies own; what remains framework-side is rank, wg and null-description agreement (D89) and the value-driven rule (D129). r01 §5.6.

## D92 — Packing
Records, inputs, outputs and body streams are packed into four 16-byte-aligned host buffers, uploaded to device USM each forward and passed to the launch. `scratch_off = body_off + record.scratch_off`. Reason: one launch has fixed arguments. r01 §5.7.

## D93 — Staging through member usm_vectors
The staging buffers are `std::optional<sk::usm_vector<T>>` members (`emplace` + `from_host`, which snapshots the source and is stream-ordered), allocated from the pool if the substrate has one, else the OS device allocator. Reason: usm_vector has no default constructor; back-to-back forwards stay valid because the snapshot is stream-ordered. Known risk: the vectors destruct after the RootDriver's pool if declared before it; not seen in the smoke. r01 §5.7.

## D94 — Custom-op and host children are refused in a wave
`compile()` throws naming the child. Reason: they have no device kernel to chain. r01 §5.4.

## D95 — Helper form for fused; amends D79
`add_subsystem(name, props, child<T>(...)...)` with `property::fused` builds a FusedSystem through `populate<S>` with a constructor tuple `{mode, wg}`; `parallel` and `fused` together still throw; a typed child given `property::fused` throws unless it derives from FusedSystem. Parallel mode throws "step 3b-3". `detail::fused_of` keeps the include order acyclic, as D80 does for parallel. FusedSystem's `add_subsystem` takes Components only, like ParallelSystem (D74). r01 §7.5.

## D96 — Test change in smoke_parallel
Its case d uses `property::fused{Mode::parallel}`, since plain `property::fused` now builds a FusedSystem. Test-only.

## D97 — Parallel mode: one slice of groups per stage
`group_offset` is the running sum of the previous stages' `group_count`, and the launch has the SUM of the group counts (sequential mode: offset 0, launch = the max). Each stage still returns early, as a whole group, outside its own slice. Reason: the stages are independent, so they occupy disjoint groups of one launch. r01 §5.4, §5.7.

## D98 — No barriers in either mode; amends the parallel-mode text
The framework inserts no barrier ids in any mode (D87). In parallel mode the stages own disjoint groups (D97), so the local buffer needs no reuse discipline either: every stage has `local_off` 0 and the size is the max (D90). A group body's own barrier is uniform because the slice check is per whole group. r01 §5.4.

## D99 — Replaced: no connections between the stages of a parallel wave
Every forward, in a parallel FusedSystem and in a ParallelSystem, a stage input whose root's aliasing set holds another stage's output throws `invalid_argument` naming the input and its stage and the output and its stage. There is NO override: the stages run unordered, so the pair is always a race. Replaces the producer-consumer check that `allow_producer_consumer()` could lift. Sequential mode has no such check (D91). Implemented by `detail::check_no_stage_connections` (D130). r01 §7.5.

## D100 — Replaced: allow_tied_outputs() is the opt-in for tied outputs
In a parallel wave, two stage outputs that share one root (`tie_outputs`) throw at forward, naming the tied outputs and `allow_tied_outputs()`; calling `allow_tied_outputs()` on the FusedSystem or ParallelSystem lifts only this check (the values are then the user's assertion, e.g. atomic accumulation). It never lifts D99. Sequential mode does not check ties. `allow_producer_consumer()` is deleted. r01 §7.5.

## D101 — The step 3b-3 throw is gone
The FusedSystem constructor takes either mode; the helper form passes `{mode, wg}` through unchanged (no change in system.hpp's logic, only its comments). smoke_parallel case d now expects `property::fused{Mode::parallel}` to build without throwing. Reason: parallel mode is implemented.

## D102 — Wave work-group in parallel mode
Chosen as in sequential mode (D89), one shape for the whole wave: a group stage's requested wg is the wave's and plain stages adopt it. A plain stage's slice is ceil(flat_extent/wg_size) groups, a group stage's the product of its per-dimension ceilings. r01 §5.6.

## D103 — Compile-time errors are the same in both modes
A custom-op child, a host child, a child on another device and children of different kernel ranks all throw `invalid_argument` from `compile()` (during `start()`), naming the child; smoke_fused case g checks the first, second and last. Reason: none has a stage function, or the wave has one rank and one device. r01 §5.4.

## D104 — The Model's state lives on the heap
`Model` is one `unique_ptr<Impl>`. After setup the Systems hold raw pointers to the queue and the substrates (ExecCtx, `subs_`), so those must not move when the Model does; moving the Model moves the pointer. Inside `Impl` the members are declared ctx, queue, substrates, root, the three mark vectors, cfg, so destruction is cfg, marks, root, substrates, queue, ctx (§9.4). A moved-from Model throws `logic_error` ("moved-from") from every verb. r01 §9.1, §9.4.

## D105 — default_device resolution
A set `default_device` is validated with `ctx.contains()` and a miss throws `invalid_argument` at construction. Unset: the runtime's default-selector device when the context holds it, else the context's first device (a device-list config need not contain the selector's pick). The queue is built on it: `queue{ctx, default_device, async_handler}`. r01 §9.1.

## D106 — Setup chain, as the driver did it
`setup()` places the root with an empty name and path (D136), then run_setup, get_exec_info(default device), substrates, prepare_exec, compile, run_init, run_system_init, marks. Per device in the exec-info map: an arena of `bytes + 4096*64` (the driver's alignment slack) iff bytes > 0, a pool of `pool_bytes[d]` or `default_pool_bytes` iff > 0 (node size `pool_node_size`); a device outside the context throws. The default device always gets a substrate entry. setup() runs once: a second call, or one after a failed first, throws `logic_error`; forward(), host access, retrieval, marks and memory_report before it has completed throw `logic_error`; wait() and root() do not. r01 §9.1, §11.

## D107 — memory_report: what is filled and what is not
`arena_bytes` is the arena's capacity (including the slack of D106), `arena_bytes_in_use` its `used()`; `pool_bytes` is slot_count x node_size, `pool_bytes_in_use` outstanding x node_size. `os_allocations` and `os_bytes_in_use` are always 0: Nothing in the runtime counts OS-route allocations yet, so the profile-then-size-a-pool loop of §9.3 cannot read OS use from this report; adding a counter is a later change. r01 §9.1.

## D108 — Mark collection order
The walk visits a System's own alias entries (inputs, then outputs), then its children in declaration order, recursing into child Systems and taking a Component's entries. Each root is read once (dedup by root pointer, first reached wins) and its design_var/constraint/objective is appended to the matching vector. The spans are cached at the end of setup(). r01 §8.4, §9.1.

## D109 — Host access goes through the entry (mutable host views, set_input, set_output superseded by D139)
view_input/set_input require an input entry and view_output/view_output_mut/set_output an output entry (`invalid_argument` otherwise, naming the path). Views are the private-constructor HostView/HostViewMut (Model is their friend); view_output_mut's writes land in host staging and are written back on commit() and on destruction. `set_*` check dtype and rank (`detail::check_entry`) and call `detail::write_back`, which writes the root's primary storage (through the entry's region if it has one) and refreshes the other placements. r01 §9.1, §9.2.

## D110 — ~Model and the moved Model
`~Model` and move-assignment call `queue.wait_and_throw()` inside try/catch (a stream error is printed to stderr, never thrown), then release the Impl so members go in the D104 order. r01 §9.4.
Superseded by D135.

## D111 — smoke_model notes
The Sellar Systems are copied from smoke_system (no shared header). Case d writes through `view_output_mut` on an OUTPUT root whose elements the producer does not overwrite (Hold writes y[0] of 4; the host owns y[1..3]): the dispatch said "a free input's root", but view_output_mut is the output view and a free input is written with set_input, so that literal case cannot exist. Case c's default_device-outside-context check needs two visible devices and is skipped with one. Observation: under `ACPP_VISIBILITY_MASK=cuda` the OpenMP host device was still enumerated, so case b also ran with that mask; with the mask removed the same result.

## D112 — The test package is its own pixi-build package
`cospanmdao-tests` is built from `tests/` by pixi-build-cmake (the root manifest's backend block) with host-dependencies `cospanmdao` and `catch2`, both `{ workspace = true }`; the pool form resolves in the sub-package, so no plain specs are needed. `tests/CMakeLists.txt` is standalone, like SlabKit's: cmake 3.24, `find_package(Threads/AdaptiveCpp/cospanmdao/Catch2 3)`, one executable `cospan-tests` (Catch2::Catch2WithMain + naga::cospan, `add_sycl_to_target`), installed to `bin`. The root CMakeLists does not `add_subdirectory(tests)`, and `tests/smoke` and `tests/spike` stay separate, standalone projects. `pixi run test` (task `cospan-tests`) runs it; the `cuda` environment holds both packages, so `pixi run -e cuda test` works with no layout change. r01 §11.

## D113 — Every test goes through a Model
The Catch2 tests build a small top System, make a Model (`create_model`, `setup`, `forward`, `wait`), and check with `view_*` and `set_*`. No RootDriver, no TestDriver, no EntryAccess. The only reach past the verbs is the Model's own retrieval (`root().entry_at(...)` for a storage kind or a storage pointer, `get_subsystem(...)` for a test component's public API). The smoke programs keep their drivers; they are not the suite. r01 §9, §11.

## D114 — Shared test helpers
`tests/support.hpp` (namespace `cst`) holds `default_config()` (a default `ModelConfig`), `close` / `all_close` (abs + rel tolerance, float and double), `to_vector` (a host view as a vector) and `wait_and_view` (forward, wait, read-only view of an output). Integer-valued results (histogram bins, prefix sums, 2x+1 of small integers) are compared exactly or with the tolerance; nothing depends on a device-specific rounding. r01 §11.

## D115 — Values that change between forwards are free inputs (set_input superseded by D139: init_input)
A value the test changes between two forwards (the histogram's `n_bins`) is a free input written with `set_input`, not a test-only producer with a mutable member. The kernel-fed inputs (histogram values and weights, the iota producers) stay components, so producer-to-consumer wiring is still covered. A value-driven output (`FromInput{"n_bins", 0}`) is driven from a host-written free input and grows and accumulates as it did behind the driver. r01 §8.

## D116 — Phase guards are checked through the component's public API
After `setup()`, the test reaches a test component with `dynamic_cast<InitDemo&>(model.get_subsystem(...))` and calls the public initializer, `reshape` and `declare_*` API outside its phase; each throws `logic_error` naming `init() only`, `pre_forward() only` or `setup() only`. The declare and reshape calls come from two small test hooks on the component (a declare_* or reshape needs a member wrapper). The Model is unharmed by the refused calls. r01 §6.2.

## D117 — Rank coverage by one templated body
The rank 1, 2 and 3 plain-body cases are one `AffineBody<R>` / `Affine<R>` / `TopAffine<R>` template (sizes 1000, 40x30, 7x9x11; two forwards on different input). `cs::Model` is spelled out instead of `auto` in the template, so `model.set_input<...>` needs no `template` keyword. r01 §11.

## D118 — One namespace per test file
`test_component.cpp` keeps `tc`, `test_system.cpp` uses `tsys`. All test files link into the one `cospan-tests` executable, so two files defining a component or body of the same name (an `Axpb`, a `Src`) in the same namespace would be an ODR clash the linker may not report. r01 §11.

## D119 — Transient release is observed through the System
The transient test reaches the transient System with `get_subsystem("tr")` (a `dynamic_cast` to `cs::System&`) and checks `transient()` and `released()` after `forward()` and `wait()`, and the roots' `released()` / `has_storage()` through `model.root().entry_at(...)`. A `view_output` of a released root must throw, and the message must name `top.tr` and "transient System". Revival is observed through init counters (`count_of`), not storage pointers. r01 §4 (transient subtrees), §7.4. (Paths are now relative to the root: see D136.)

## D120 — View composition is a Slice on a promoted Slice alias
"View composition" is tested by connecting a `Slice` on top of a promoted alias that already carries a `Slice`: the offsets add (2 + 1 = 3, elements 3..4), and the consumer's result is the check. It goes through `connect`, not through a hand-built view. r01 §8.2.

## D121 — KNOWN ISSUE: CUDA deferred-free deadlock without a pool
A no-pool Model on CUDA hangs in the first kernel submit after an OS-route free (a transient release, or a value-driven root growing): SlabKit's free runs `cuMemFree` in an `async_host` task while the next kernel's host callback blocks the stream on that task. Not a CoSpan bug. SlabKit e5e2ac1 puts deferred frees on their own queue (the design: nothing depends on a free), which as a side effect no longer triggers the fork bug, so the tags are removed and the WeightedHistogram and `[transient]` cases run in the full CUDA suite. The fork bug is still present in naga-acpp 2026.09.29 (a kernel depending on an unfinished async_host node gets a blocking host callback in the CUDA/HIP stream, which CoSpan host components followed by a device consumer still take); the fix is committed in the fork at b04d8349 (naga/develop, unbuilt, unpushed), see the doc. A pool in `ModelConfig` is still an option. Details and stacks: `docs/known-issues/cuda-deferred-free-deadlock.md`. r01 §4, §9.
Superseded by D135.

## D122 — Release without -g is the only working build for now
`RelWithDebInfo` (`-O2 -g`) and `-gline-tables-only` builds of CoSpan translation units crash clang 21.1.8 (exit 139) in the "Debug Variable Analysis" pass (`LiveDebugVariables::computeIntervals` -> `LexicalScopes::initialize`) on `naga::cospan::kb_entry_r1`, in both test TUs. Until that is fixed or worked around, build Release without `-g`; libraries still resolve exported symbols in gdb, header-only CoSpan frames stay unsymbolized. Not a CoSpan source problem. r01 §11.

## D123 — Flavor refusals are observed at the phase that throws them
The wiring rules of the two flavors are checked where the implementation checks them, and the tests follow: a connection between the stages of a parallel wave (Fused or Parallel, through a `FromSourceDim` input or a fixed-shape one), and a tied-output pair in a parallel wave without `allow_tied_outputs()`, set up cleanly and throw at `forward()` (the test then calls `wait()`); a non-device property on a Parallel child, two flavors on one System, a custom-op, host or mixed-rank Fused child, and a value-driven size whose driver aliases another stage's output (D129) throw at `setup()`; the local-memory limit (D128) throws at `forward()` on a device that reports one. Every case names the offending child or children (`top.par.a`, `top.f.bad`) in the message, and the test checks the names. r01 §7.5. (Paths are now relative to the root: see D136.)

## D124 — Fused against unfused is compared bitwise
Four element-wise stages as a plain System and as a sequential FusedSystem (777 non-round floats) are compared with `memcmp`, not a tolerance: a fused wave runs the same body on the same operands in the same order, so nothing may differ, and the test passed bitwise on omp and on CUDA. A host recomputation with `cst::all_close` keeps the check from being vacuous (two equally wrong outputs would not pass). If a backend ever differs by float rounding alone, relax this one case to rel 1e-6. r01 §7.5.

## D125 — The flavor tests and the OS free route
`test_flavors.cpp` (namespace `tfl`, tags `[flavors]` plus `[parallel]` or `[fused]`, `[errors]` on refusals) carries no `[os-free]` tag (D121) and every case except the local-memory-limit one runs on CUDA. Amended: the value-driven allowed cases (D129) now grow a root (`set_input` 8 then 13), which frees through the OS route when the substrate has no pool; that is acceptable since the free lane (D126) means no kernel depends on a free. The local-memory-limit case skips on a device with no usable limit (omp reports the size_t maximum). r01 §7.5, §11.
Superseded by D135.

## D126 — SlabKit repinned to the frees-lane commit
`pixi.toml` pins slabkit to `e5e2ac15f31a88bb4a565b2eb18933f5aba140e1` (was `5d6b7b5e6cb6563cfb285361ce05686518c18f04`): deferred frees run on their own lane, `begin_item` and `open_group` do not merge it, and `wait()` and the stream's wait list do, so no kernel depends on a free (D121). Lock refreshed with `pixi lock`. The `[os-free]` tag is removed from `test_component.cpp` and `test_system.cpp`, and the CUDA verification runs the whole suite with no filter, three times. The fork's root cause is unchanged and still pending a rebuild; see `docs/known-issues/cuda-deferred-free-deadlock.md`. r01 §4, §9.
Superseded by D135.

## D127 — StageRecord::local_off
`StageRecord` gets `uint32_t local_off`, between `scratch_bytes` and `rank`; `kb_fn` fixes a stage's scratch up at `local_base + local_off`. `submit_wave` fills it: sequential running sum with each slice rounded up to 16, parallel 0 (D90). Reason: per-stage slices without a barrier. r01 §5.6.

## D128 — The local-size limit is a forward-time error
`submit_wave` compares the wave's local bytes with `info::device::local_mem_size` and throws `invalid_argument` naming the scratch stages and their byte counts and the device's limit, instead of launching and failing in the backend. Sequential slices add up, so this is reachable where the old max-buffer was not. there is no special case for omp: it reports a `local_mem_size` so large that the check never fires; CUDA offers about 48 KB, and the test runs there. r01 §5.6.

## D129 — Value-driven sizes may not alias another stage's output (rule h)
`compile()` throws, in both modes, when a stage's output size is value-driven (`DimRecord::How::from_input`, its extent read from an input's VALUE) and the driver's aliasing set (root plus peers) holds another stage's output; the message names the stage, the driver and the other output and says "VALUE". This did not exist before this commit. `FromInputDim` and `FromSourceDim` through connections stay allowed. Reason: the size is read at launch, before the producing stage could have written the value. r01 §7.5.

## D130 — The wave rules live in one shared header, wave_rules.hpp
`include/naga/cospan/wave_rules.hpp` (namespace `naga::cospan::detail`) holds `check_no_stage_connections`, `check_tied_outputs`, `check_value_driven_sizes` and their helpers; FusedSystem and ParallelSystem both call them. Reason: one wording and one aliasing walk for both parallel runners. r01 §7.5.


## D131 — The Model tests live in test_model.cpp, namespace tmod
`tests/test_model.cpp` (namespace `tmod`, tags `[model]` plus `[sellar]`, `[marks]`, `[lifecycle]`, `[host]`, `[memory]`, `[pool]`, `[devices]`) goes through a `cospan::Model` for everything; the only reach past the Model's verbs is its retrieval (`root().entry_at`, D113). The Sellar Systems are templated on the dtype and run in f32 (abs and rel 1e-5) and f64 (1e-12) over five design points against a host reference in double, computed from the values the device saw. The default_device-outside-context check is a SECTION of the lifecycle case and the two-device case is a whole-case SKIP, both when the runtime shows too few devices (omp mask: both skip; cuda mask: both run, since the OpenMP host device still enumerates, D111). r01 §9, §11.

## D132 — set_input errors for dtype, rank and size name the entry, not the path (superseded by D139: set_input removed)
`check_entry` and `write_back` (host_view.hpp) build their messages from `IOEntry::name()`, which is the leaf name: a wrong dtype, rank or size on `d1.y2` says `'y2'`, on the top-level alias `z` says `'z'`. The unknown-path and wrong-kind errors do carry the full path (`entry_at` and `Model::entry_of`). The test asserts the leaf name (and the path for a top-level alias); the library is unchanged. Open for Jack: whether these two messages should carry the dotted path. Superseded by D136.

## D133 — Memory report measured: arena capacity is the distinct-root sum plus the setup slack, exactly
A model with roots gen.y f32[100], cvt.d f64[64], a tie alias bc f32[8] and comb.o f32[16] (1008 B), two connected inputs (peers) and a connected f64 input, one tie and one promote, reports arena capacity 263152 B = 1008 + 262144 (`kArenaSlack`, D106) and 1008 B in use after setup and after a forward, on omp and on CUDA. So neither a connected input peer (D85) nor the old member roots of a tie are counted twice in this shape. The test asserts that equality and carries no `[!mayfail]`. The pool case shows `pool_bytes` 1 MiB and `pool_bytes_in_use` 1024 B for 8 floats and 12288 B for 3000 floats (node size 1024). r01 §9.

## D134 — A CUDA run of the whole suite hung once in an existing flavors case
One of four CUDA runs of the full suite at this commit hit the 600 s timeout in `test_flavors.cpp:943` (`fused: a value-driven size driven from outside the fuse runs and follows the value, in both modes`, parallel section) before reaching the model tests; the other three passed (44 cases, 543 assertions). Not diagnosed here. That case grows a root and frees it through the free route (D125); whether the hang is related to `docs/known-issues/cuda-deferred-free-deadlock.md` is not established. No library change was made in this step.
A later CUDA run (run 1 of a repeat set) passed the whole suite (543 assertions in 44 test cases) and then the process dumped core (SIGSEGV, exit 245) after Catch2's summary, i.e. at teardown. Undiagnosed. Both symptoms (the hang and the exit crash) were seen on naga-acpp 2026.09.29 before the fork fix, and both may be the known fork bug (a value-driven root growing in a no-pool fused wave means an OS free plus SlabKit async_host copies). The next step is to move the lock to the fixed 2026.09.29 build and see what is left.

## D135 — Synchronous frees
SlabKit a484713 removes the frees lane: `deallocate` waits the events it is given, then frees on the calling thread (pool slot or sycl::free), and a usm_vector passes its own last task. CoSpan's kernels over a root are CoSpan's to wait for, so `StorageEntry::release_storage` (transient release) waits the stream before dropping its slots, and `reshape` waits the stream once before growing a stored root past its capacity. `~Model` and move-assignment do not wait, as a sycl::queue's destructor does not: work the user submitted through the Model's verbs (setup's initializers, set_input/set_output, a view_output_mut write-back, forward) is theirs to wait for with `Model::wait()` before the Model is destroyed or reassigned; a Model dropped with work in flight is undefined. The pin moves to a484713, and the lock stays on fork build 4, which carries the async_host fix. Accepted costs: a transient System's forward blocks until its subtree finishes, and a value-driven grow blocks once. Per-root events were considered and not taken: the primary lane is in order, so a subtree's last launch follows everything before it. r01 §4.4, §9.4.

## D136 — Paths are relative to the root System; host-access errors name the full dotted path.
The Model's root System has no name or path (as in OpenMDAO): a user addresses `d1.y2`, never `top.d1.y2`, and every path in an error message is relative to the root the same way. `Model::setup` places the root with an empty name and path; child paths join as the name alone under the root. Host-access errors (`set_input`/`set_output`, `view_*`, write-back, released or unstored roots) name the entry by `IOEntry::full_path()`: its declarer's path plus its name, or the name alone at the root (`'d1.y2'`, `'z'`). Supersedes D132, and the `top.` forms quoted in D119 and D123.

## D137 — RunNSystem; do-once is n = 1

RunNSystem(n) runs its subtree on the first n forwards and then returns early
(nullptr, nothing submitted). There is no RunOnceSystem: do-once is n = 1
(Jack, 2026-09-30). n == 0 throws std::invalid_argument at construction. Users
derive from it and override setup(). With property::transient it releases after
each run and revives on runs 2..n; after the n-th the early return reaches no
revive(), so the subtree stays released. Roots with a consumer outside persist.
Lives in include/naga/cospan/run_n_system.hpp, included by cospan.hpp.

## D138 — Initialization and transientness are root metadata; components allocate, init and release (Jack, 2026-09-30)

A. Functional model (r01 §6.0). Outputs are always computed fresh from inputs: no component may assume an initial output value, and outputs are not initializable. There is no global scratch; multi-step work is several components, producer to consumer. Accumulation is either group work with local memory inside one kernel, or a zeroing producer component followed by the accumulating components, all joined by tie_outputs, whose default stays require_atomic with the allow_plain opt-out. Atomic accesses are relaxed; kernels that need acquire/release ordering use sycl::atomic_fence(memory_order::acq_rel, memory_scope::device), which SYCL 2020 guarantees on every device.

B. Initialization is metadata on free input roots (a free input root is an input root with no source). Kinds: cs::init::fill(v), zero(), from_host(std::vector<T>), custom(fn(sk::queue&, Initializer<DT,R>)) and custom_host(fn(HostViewMut<DT,R>&)). Declared during setup() with ComponentBase::init_input(name, init) on the component's own input, or System::init_input(path, init) on a child input path or one of its promoted or tied input aliases. The highest level wins: the declaring subsystem's depth decides (root System = depth 0, smallest depth wins); two declarations at one depth on one root are an error at setup. An init on an input whose root ends up with a source is ignored, so a reusable component can carry a default for when it is left unconnected; an init on an output is an error at setup. A free input root with no init is zero-filled. The framework applies a root's init when its owner allocates that root from nothing (see the Amendment below: first run for durable roots, each (re)allocation for transient ones); a value-driven resize preserves existing content and does not re-init. REMOVED: the init(q, dev) hooks on Component and System, the init passes (run_init, run_system_init, so "the System has the last word" by ordering is gone), and every initializer helper, including all output forms (get_*_initializer, get_*_host_initializer, fill_*, zero_*, set_*_from_host); the HostInitializer alias goes with them.

C. Transientness is registration metadata. Only an owner declares a subsystem transient, by passing property::transient to add_subsystem; the subsystem may be a Component or a System (a ParallelSystem's children still take only property::device, D75). A registration is a component's entry, or a promoted or tied alias made by a System; it is transient if its registering subsystem or any ancestor of it is transient (highest level wins; an inner level cannot undo an outer transient). A root is transient if and only if EVERY registration in its alias set is transient: a promotion or tie made at a non-transient level, or any non-transient component touching the root, makes it durable. Transient roots are never arena-eligible (pool route, OS route without a pool). A mark on a transient root is an error at setup.

D. The component, always the leaf, does the lifecycle. In run_pre_compute, after shape resolution and pre_forward, it allocates every root it OWNS that has no storage and applies the init if it is a free input root (owner-based; see the Amendment below). In run_post_compute, after the other-placement sync, it releases the transient roots for which it is the last user: each root waits on the stream and is freed (synchronous frees, D135), with one stream wait for the component's batch. last_user is the last component, in depth-first registration order, that touches the root; children of a ParallelSystem or FusedSystem are ordered within their flavor, and their post_compute runs after the group or wave returns. A component that does not run in a forward (RunNSystem exhausted, a gate not taken) neither allocates nor releases; its roots wait. Host access to a released root (views and set_input) throws, naming the root and the transient subsystem whose registration made it transient. REMOVED: System revive(), release(), owned roots, nested revival, and the flavors' revive/release calls.

E. RunNSystem (r01 §10.1): an early return means its components neither allocate nor release, and after the n-th run its transient roots stay released. No mention of revive().

F. Setup chain (r01 §9.1, §11.2): declare, ending with the resolution pass that records, per root, transient, init and last_user (replacing transient marking) → exec info (arena bytes for durable static device roots only) → substrates → prepare (captures the queue and substrates; allocates nothing, see the Amendment) → compile → collect marks.

Supersedes the init-related parts of D34 (the init hooks, passes and helpers) and the revive/release parts of D55 and D56. Amends D57 (the released-root host-access message names the transient subsystem whose registration made the root transient, in D57's wording style), D119 (release is observed through memory_report and host-access errors, not a System flag) and D137 (RunNSystem has no revive; its exhausted state leaves transient roots released). D54's "the System owns transient marking" becomes "the root System's resolution pass records registration-derived transientness". Known cost: one host stream wait per releasing component per forward.

**Amendment (Jack, 2026-09-30; its host-write paragraph is superseded by D139).** Allocation and init have ONE path, the owning component's, for durable and transient roots alike. Each root gets an OWNER, recorded by the resolution pass next to last_user: the FIRST component, in depth-first registration order, that touches the root (children of a ParallelSystem or FusedSystem are ordered within their flavor, as for last_user). In run_pre_compute, after shape resolution and pre_forward, a component goes over the roots it OWNS, inputs and outputs: any owned root without storage is allocated (route unchanged: arena for static durable device-kind roots, else the pool, then the OS), and if it is a free input root its init from metadata is applied (zeros when none). A component never allocates or inits a root it does not own; non-owners only point their wrappers. A durable root is allocated once, on its owner's first run, and never released; a transient root is released after its last user and allocated again on its owner's next run. REMOVED: the setup-time allocation of roots in prepare_exec (it keeps queue/substrate capture and shape updates; arena sizing in get_exec_info is unchanged, since the arena is still created at setup), and apply_inits with its step in the setup chain (the chain is now run_setup + resolve, get_exec_info, substrates, prepare_exec, compile, marks). Host access: a host write to a root with no storage (Model::set_input, or a HostViewMut write-back or commit) allocates the root through the same allocation routine the owner uses (one function, called by both) and the written value is its initial value; when the owner later runs it finds storage and does nothing, so set_input beats init. Reading a root with no storage (view_*) throws, saying it has no data until its owner runs. RunNSystem, gating and D84 are unchanged. Consequence (superseded by D139 part 2): tie_outputs accepts value-driven (dynamic) members, the alias root following the members' resolved extents, which r01 §6.7's ZeroBins + Histogram sample requires; specified in r01 §4.3 (SHAPE) and §7.4.


## D139 — Initializer-only inputs, pending inits, and empty roots (Jack, 2026-10-01; part 1)

A. A host never writes data into the model. The ONLY way to give an input data from the host is an initializer: `Model::init_input(path, InitSpec)` (the factories of D138: fill, zero, from_host, custom, custom_host), callable any time after setup. REMOVED: `Model::set_input`, `Model::set_output`, `Model::view_output_mut`, the host-write-allocates path (`ensure_for_host_write`) and the host write API of `HostView`. `view_input` and `view_output` stay, read-only. `HostViewMut` survives only as the argument type of the function given to `cs::init::custom_host` (the framework builds it over the root it is initializing and commits it on return); the type has no public constructor and no Model entry point.

B. `Model::init_input(path, init)`: an OUTPUT path throws `std::invalid_argument`. Otherwise it sets the init of the path's ROOT. It wins over the Component's and the System's declarations (`resolve()` never overwrites it), and it marks the root PENDING. If the input is connected (its root is an output root, it has a source) the init is stored, one line is printed per call to `std::cerr`, `naga::cospan: warning: input '<path>' is connected to an output; your initializer may not run`, and the owner never applies it.

C. The owner's `run_pre_compute`, for each owned free input root: if the root is not PRESENT it is allocated and its init applied (zeros when none), as before; else if it is pending the init is applied to the existing storage. In both cases pending is cleared, so one `init_input` call initializes once. A transient root takes the current init at every allocation. A pending init waits for its owner: an owner that does not run (an exhausted RunNSystem, a gate not taken) leaves it pending, and the root's contents do not change.

D. Empty roots are valid. A root has an explicit PRESENT state, set when its owner's allocation path runs (`IOEntry::allocate_if_absent`) even when `flat_size() == 0`, and cleared by `release_storage`. A size-0 root is present with empty storage (`sk::usm_vector<T>(0, alloc)` allocates nothing and is valid; checked in SlabKit). `has_storage()` and the "no data until its owner runs" check use present, not "slots non-empty". A view of a present size-0 root is empty with the right shape; a view before the owner runs still throws. A launch with `flat_extent == 0` submits nothing (D31), so a consumer of an empty root runs as a no-op. An empty root that later grows (its driving value changes) is resized in place by `reshape`.

Supersedes: D109's mutable host views and `set_input`/`set_output`/`view_output_mut`; D17 (HostViewMut writes back on commit and destruction; no mutable views); D115 and D132 (`set_input` and its errors; a changed value is now an `init_input` call); the host-write amendment of D138 (a host write allocates the root and stands in for the init); D24's "a root left empty is never allocated" and D34's empty-root rule, replaced by D. Tests that existed only for the removed API were deleted (view_output_mut, set_input dtype/rank/size errors). One behavior change: after `init_input`, the new value is visible to `view_input` only after the owner has run.

## D139 part 2 — Ties own the shape; promotions mirror live; aliases are rank-templated (Jack, 2026-10-01)

A. Promotions mirror. `promote<DT, Rank>(path, name)`: DT must match the entry's value dtype and Rank its rank (setup error otherwise). The alias stays a peer and its shape is READ LIVE from the entry (`IOEntry::mirror_of_`; extents, flat size and geometry delegate), so a view through a promoted name shows the member's current shape, including after a value-driven size changes. REMOVED: `inert_dims` and the freezing of a promotion's extents. This fixes the parked radius-round bug (a promoted alias of a value-driven output kept a stale 0 shape in views).

B. Ties own the shape. Members are RANK-ONLY entries: `declare_input(w, "x", cs::tied)` / `declare_output(w, "y", cs::tied)` (dtype and rank from the wrapper; `Tied` marker and `DimRecord::How::from_root` in metadata.hpp). A rank-only entry's shape is its root's (its view's, if it has one), resolved in `update_shape`. The tie alias owns the shape through an `AliasShape` (metadata.hpp): `cs::alias::fixed(e0[, e1, e2])`, `from_connection()` (tie_inputs only), `from_output_shape(path)` (an upstream output of the alias's rank), `from_output_values(path)` (the extents are the VALUES of elements 0..Rank-1 of an upstream output of integer dtype, read with the same host snapshot as FromInput). Calls: `tie_inputs<DT, Rank>(members, name, AliasShape)` and `tie_outputs<DT, Rank>(members, name, AliasShape, policy)`.

C. Setup errors. A tie with fewer than two members ("a tie needs two or more members; use promote for one"); DT or Rank not matching a member; a member that declares a shape (named); `tie_outputs` with from_connection ("an output alias defines the shape of its connections"); from_connection on an alias nothing is connected to; a from_output_* path that is not an output that can be a source; a from_output_* source whose producing component (the last component writing the output's root) is not registered before the tie's first member, checked in `System::resolve()`; and, after resolve, a rank-only entry that is in no tie's set (directly or through promotions). A tie alias carries a shape, so it cannot itself be a tie member: pass members up by promotion. FusedSystem and ParallelSystem extend `check_value_driven_sizes` to alias specs: a tied entry whose alias reads an output of another stage of the same wave is refused.

D. Semantics. A tie_outputs alias is always the root. A tie_inputs alias is the root until something connects into it; then it joins that output's root like any connected input (the producer owns and allocates it), the members take that output's shape, and a fixed or from_output_* AliasShape is checked against it on the first run (`invalid_argument` on a mismatch). The root's owner resolves an alias root's shape in `run_pre_compute` BEFORE allocation (a from_output_* source has already run), every run, and resizes the root if it changed. Views of an alias before its root's owner has run throw, as for any root. Model init on a connected tie_inputs alias follows the part-1 rule (stored, warned, skipped).

E. REMOVED: `System::shape_source` (the "first fixed member" rule), inert alias dims, the untemplated promote/tie_inputs/tie_outputs, and the D138 follow-up that propagated a tied output member's resolved extents to the alias root (StorageEntry::reshape).

F. Supersedes: D62's shape rule (a tie alias takes its shape from a fixed member); D66 (alias extents are inert); the D138 amendment's consequence that tie_outputs accepts value-driven members and the alias root follows them (the ZeroBins + Histogram sample now reads the bin count from an upstream output, `alias::from_output_values`). Where tests needed a component both tied and untied they got a parameter or a tied-only variant: gemver-chain's `Gemv` and `GemvTPartial` take a `tied_b` flag (BiCGK's B tie); `ZeroCounters` and `GridBarrier` outputs are always tied (BarrierTop ties them with `alias::fixed(1)`); test components `Axpb`, `Src` (flag or tag constructor), `InitAffine` (flag), and always-tied `Adder`, `ZeroAcc`, the Sellar `D1`/`D2`/`Obj` and `Comb`. Nested ties (a tie of a tie alias) became ties of promotions.

## D140 — Ragged dimensions and chunked systems (approved by Jack, 2026-10-01)

1. Ragged dimensions, available to any component. Dim 0 (leading) is always a single value, set by any existing rule (fixed, FromInput value, FromSourceDim, FromInputDim). Dims ≥ 1 accept any existing rule plus per-index (array-valued) rules: a host array given at construction (cs::per_index); the values of an input that is an array (cs::FromInputValues, rank 1 [N] or a column of rank 2 [N,k]); inheriting dim d of another input's shape where that dim is per-index; wholesale inheritance from a connection. Outputs never take a shape from a connection; they take it from an input of the same component (which may itself come from an upstream connection). A port is ragged when any dim ≥ 1 is per-index; per-index and fixed dims mix freely (e.g. [N, c_i, 3], [N, Q_i, c_i]). Every per-index array has length = dim 0. Layout: blocks end to end; block i size = Π_{d≥1} extent_d(i); block offsets by an exclusive scan computed on the host (per-index arrays are snapshotted like FromInput) and uploaded. Kernels use RaggedInput/RaggedOutput (block(i) → a small dense view), which also read dense ports; binding a ragged port to a dense Input/Output is an error. No separate code path for uniform shapes.

2. Chunked systems. A chunked system chunks along the leading dimension of every port, which is never named. Every input and output of the chunked system, and every port of every child, has leading length = the chunk length; a mismatch is an error. Every dim ≥ 1 must be resolvable before the chunked system runs, from things crossing its boundary; a shape driven by a value computed inside the chunk is an error. Pricing per index = Σ over internal ports of sizeof(dtype)·block size, from the declared shape rules only (pre_forward is not run for pricing; don't reshape manually in pre_forward inside a chunked system); inputs from outside already exist and are not priced. A scan of the per-index cost against the budget (constructor argument, bytes) cuts contiguous chunks; one index over budget is an error. There is no separate pool: for each chunk the framework re-points the boundary connections — inputs from outside to views of that chunk's leading range (blocks for ragged data), outputs leaving to views into a full-size root at the chunked system's level, allocated once before the first chunk — and runs the subsystem once. Internal ports follow through the dynamic shape rules and are allocated by their owners and released by their last users; preferably they are allocated once at their largest chunk size, kept across chunks and released after the last chunk. Children never know which chunk they are in. Global work (e.g. an LBVH, gathering neighbour coordinates) runs outside.
   Part 2 (chunked systems) is implemented (B2, B3).

3. Future, not in D140: memoryless functional ports (chunk-shaped inputs computed on the fly by a functor), and sparse matrix bundles (CSR, block CSR, SELL) over ragged ports.
