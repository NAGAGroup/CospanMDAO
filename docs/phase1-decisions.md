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

## D17 — HostViewMut writes back on commit and on destruction
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

## D24 — "Known shape" and allocation timing
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

## D34 — Initializers on an unallocated root
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

## D62 — Tie aliases are new roots
`tie_inputs(members, alias)` requires each member to be a free input (target-not-source); `tie_outputs` requires source-not-target output roots. The alias is a NEW root whose set merges the members' sets (`IOEntry::merge(into, from)` needs two roots). Shape and dtype come from the first member (or that member's root) with a fixed shape; if none has one, the tie throws. Every member's `can_be_target_`/`can_be_source_` becomes false, so a later connect into a tied member is refused. r01 §4.3.

## D63 — A tie alias can be tied
A tie alias may itself be a member of another tie (case a: `tie_inputs({"mda.z", "obj.z"}, "z")` where `mda.z` is a tie alias of d1.z and d2.z); the member's whole root set is merged. Reason: it lets ties compose up the hierarchy, as promotions do. r01 §4.3.

## D64 — tie_outputs restrictions
Members must be output roots with no input-kind peers; dtypes must be atomic unless `allow_plain` is passed (`TieOutputsPolicy`). Reason: several producers writing one root is only defined for atomic accumulation; anything else is a race. r01 §4.3.

## D65 — get_exec_info ordering and per-child context
A System reports its own alias ROOTS first, then its children, then its own PEER aliases (the promotions). Each child gets its own `ExecCtx` (`child_ctx_`), and placed devices are held by the owner (`placed_`). Reason: reporting roots first avoids double-counting arena bytes for a promotion and lets a tie alias fix the device before its members register; peers last means their roots already exist. r01 §11.2.

## D66 — Alias extents are inert
An alias copies its dims at setup and does not follow a dynamic root's resolved extent; refresh is deferred to the Model step. Reason: nothing at System level resizes a root yet. r01 §7.2 (gap).

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
