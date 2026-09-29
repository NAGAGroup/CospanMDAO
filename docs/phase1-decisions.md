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

## D7 — Substrates are declared in metadata.hpp
`MemorySubstrate { optional<sk::DeviceArena> arena; optional<sk::DevicePool>
pool; }` and `Substrates = unordered_map<sycl::device, MemorySubstrate>` are in
metadata.hpp because `IOEntry::allocate(queue, Substrates&)` takes them. r01
§7.1 defines the substrate but not the header. The System (later step) will own
and fill the map; the smoke test fills one by hand.

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
