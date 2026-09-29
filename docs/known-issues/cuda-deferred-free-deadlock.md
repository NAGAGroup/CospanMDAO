# CUDA: deadlock after a deferred OS-route free (no pool)

Status: OPEN. The fix is pending Jack's decision (see "Where a fix belongs").
Below CoSpan: the tests are correct and are tagged `[os-free]` so the CUDA runs can skip them.

## Symptom

A `Model` built with no pool for the device (the default: `ModelConfig::default_pool_bytes = 0`,
so every dynamic allocation takes the OS route) hangs on CUDA in the first kernel submit that
follows an OS-route free. Two things cause such a free:

- a transient release: `System::release()` clears the roots of a `property::transient`
  subtree after its run, and the next kernel submitted after that hangs
  (`tests/test_system.cpp`, "transient: release after forward, ...", tag `[transient]`);
- a value-driven root growing: `set_input` changes `n_bins` (8 -> 13), the root is reallocated,
  the old storage is freed, and the following kernel submit hangs
  (`tests/test_component.cpp`, "WeightedHistogram: ...", tag `[histogram]`, second SECTION).

The hang is deterministic on CUDA and never appears on OpenMP. It disappears when the
Model gets a device pool (for example `default_pool_bytes = 64 MiB`): `[transient]` then passes.
No error is printed; the process waits forever.

## The three threads

Stacks were taken by running `cospan-tests "[histogram]"` and `cospan-tests "[transient]"`
under gdb with `ACPP_VISIBILITY_MASK=cuda` and interrupting after 30 s (attach was refused,
`ptrace_scope` = 1). Release build, no `-g`, so libcuda frames are unsymbolized. Frames are
trimmed; `...` is the pixi env's `lib/` directory.

**1. Main thread**: in the next kernel submit, blocked on a libcuda rwlock.
`[histogram]` (kernel `tc::HistBody`) blocks in `cuEventCreate`:

```
#1  pthread_rwlock_wrlock () from /lib/x86_64-linux-gnu/libc.so.6
#2  ?? () from .../libcuda.so.1.1
#4  cuEventCreate () from .../libcuda.so.1.1
#6  cudaEventCreate () from .../libcudart.so.12
#7  hipsycl::rt::cuda_event_factory::create(CUevent_st*&) () from .../librt-backend-cuda.so
#8  hipsycl::rt::cuda_queue::insert_event() ()
#9  hipsycl::rt::inorder_executor::submit_inline(...) () from .../libacpp-rt.so
#14 hipsycl::rt::dag_manager::flush() ()
#16 hipsycl::sycl::handler::create_task(...) ()
#22 naga::cospan::detail::KernelLaunch<tc::HistBody>::submit(naga::cospan::SubmitTarget&) const ()
#23 naga::cospan::System::run_compute_forward() ()
```

`[transient]` (kernel `tsys::AxpbBody`, the sink's, after the transient release) blocks in
`cuLaunchKernel`:

```
#1  pthread_rwlock_rdlock () from /lib/x86_64-linux-gnu/libc.so.6
#5  cuLaunchKernel () from .../libcuda.so.1.1
#6  hipsycl::rt::(anonymous namespace)::launch_kernel_from_module(...) () from .../librt-backend-cuda.so
#7  hipsycl::rt::cuda_queue::submit_sscp_kernel_from_code_object(...) ()
#10 hipsycl::rt::cuda_queue::submit_kernel(...) ()
#14 hipsycl::rt::inorder_executor::submit_directly(...) () from .../libacpp-rt.so
#18 hipsycl::rt::dag_manager::flush() ()
#26 naga::cospan::detail::KernelLaunch<tsys::AxpbBody>::submit(naga::cospan::SubmitTarget&) const ()
#27 naga::cospan::System::run_compute_forward() ()
```

**2. An `async_host` worker**: running SlabKit's deferred free, inside `cuMemFree`:

```
#5  cuMemFree_v2 () from .../libcuda.so.1.1
#7  cudaFree () from .../libcudart.so.12
#8  hipsycl::rt::cuda_allocator::raw_free(void*) () from .../librt-backend-cuda.so
#9  std::_Function_handler<void (), naga::slabkit::detail::guarded_host_fn<naga::slabkit::detail::AllocRoute::raw_deallocate(void*, unsigned long, std::vector<hipsycl::sycl::event, ...> const&)::$_0> >::_M_invoke(std::_Any_data const&) () from .../libnaga-slabkit.so.0
#10 hipsycl::rt::(anonymous namespace)::host_operation_dispatcher::dispatch_async_host(...) () from .../libacpp-rt.so
#12 std::_Function_handler<void (), hipsycl::rt::async_host_executor::submit_directly(...)::{lambda()#1}>::_M_invoke(...) ()
#13 hipsycl::rt::worker_thread::work() ()
```

In `[transient]` there are two such workers, both in `cuMemFree_v2`: one waits in
`pthread_rwlock_wrlock` inside libcuda, the other is running inside libcuda beneath `cuMemFree_v2`.

**3. A CUDA callback thread**: a host function enqueued in the stream, waiting on a DAG node:

```
#1  std::__atomic_futex_unsigned_base::_M_futex_wait_until (...) at futex.cc:122
#2  hipsycl::rt::async_host_event::wait() () from .../libacpp-rt.so
#3  hipsycl::rt::dag_node::wait() const ()
#4  hipsycl::rt::(anonymous namespace)::host_synchronization_callback(void*) () from .../librt-backend-cuda.so
#5  ?? () from .../libcuda.so.1.1
```

The remaining threads are idle AdaptiveCpp workers in `pthread_cond_wait` and libcuda `poll()`
helper threads.

## Reading

This is what the stacks are consistent with; the exact DAG node the callback waits on was not
traced.

1. SlabKit's `AllocRoute::raw_deallocate` posts the free as one `async_host` task
   (`q.async_host(deps, ...)`). On the OS route the task calls `sycl::free`, which reaches
   `cuda_allocator::raw_free` -> `cudaFree` -> `cuMemFree_v2`.
2. The next kernel depends on that host node. The fork's CUDA backend cannot order a stream
   operation after an `async_host` node with a CUDA event, so it enqueues a host callback in the
   stream (`host_synchronization_callback`) that blocks until the node finishes.
3. `cuMemFree` waits for outstanding work on the device's streams, and that includes the blocked
   callback. The free waits for the stream, the stream waits for the callback, the callback waits
   for the free's node. Nothing can finish.
4. The main thread, still inside the kernel submit, needs libcuda locks that the in-flight
   `cuMemFree` holds (`cuEventCreate` / `cuLaunchKernel` on a rwlock).

## Why a pool avoids it, and why the bare repro did not hang

With a pool, `raw_deallocate` calls `pool_impl->release(p, nodes)`: it returns nodes to the pool
and makes no CUDA call, so the host node never waits for the stream and the cycle cannot form.

A standalone SlabKit repro (a `usm_vector<float>` freed at scope exit, then a kernel on the same
allocator and queue) ran 10 times without hanging. The reading above needs the free's host
node to be running while the next kernel is submitted; that is a timing race, and the bare repro
did not hit it. Inside CoSpan the submit follows the release immediately and hits it every time.

## Workaround

Give the device a pool in `ModelConfig`, so that dynamic roots take the pool route:

```cpp
cs::ModelConfig cfg;
cfg.default_pool_bytes = 64u << 20;   // or pool_bytes for the specific device
auto model = cs::create_model<Top>(cfg);
```

`tests/support.hpp`'s `default_config()` still returns `ModelConfig{}` (no pool), on purpose,
so the affected tests keep exercising the OS route.

## Test tag

Every test case that frees on the OS route carries `[os-free]`: the WeightedHistogram case in
`test_component.cpp` and the `[transient]` case in `test_system.cpp`. To run the suite on CUDA
without them: `pixi run -e cuda test "~[os-free]"`. On OpenMP they pass and are part of the
normal full run. Do not run `[os-free]` on CUDA until the issue is fixed; it hangs.

## Where a fix belongs

Pending Jack's decision. Not a CoSpan bug. The options are:

- SlabKit's free path: do not call `cuMemFree` from an `async_host` task (for example, a
  stream-ordered free, or a synchronous free that takes no dependency on the stream), or
- the fork's `async_host` / CUDA integration: do not block the stream in a host callback on a
  node that may itself call into the CUDA driver.

Reproduce with the two tests above on a machine with an NVIDIA GPU; a hang means the issue is
still there.
