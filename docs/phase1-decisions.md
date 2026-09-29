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
runtime, placed on no device, that joins the kernel order as an ordinary work
item. Plain async_host is not ordered against memory tasks, so it could run
before its inputs finished copying in.

The only real constraint is memory: device memory is not host-readable, so a
host component's entries live in HOST-kind USM (pinned host memory, addressable
by the host and by devices; not managed memory, so CUDA-under-WSL2's
shared-memory limit does not apply). Root storage is keyed by placement — a
device, or the host — instead of by device alone. A host component's input
whose root lives on a device gets a host-placement copy, refreshed by the
producer's post-compute like any other-device copy; its outputs are host-kind
roots that device consumers read directly or receive copies of by the same
rule. Custom operations stay separate and are for native backend work only
(D1).
