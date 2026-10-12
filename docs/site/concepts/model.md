# Model

A **model** runs a system. It holds the root system, and the memory and work queues every component in it uses.

```cpp
cs::Model model = cs::create_model<Chain>();
model.setup();

model.update_input("first.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4}));
model.run();

auto y = model.view_output<DType::f32, 1>("second.y");
y.sync();
```

Every path given to a model is a path from its root system.

## Creating a model

`cs::create_model<Root>(args...)` builds a model with a system of type `Root` at its root. The arguments are passed to the root system's constructor:

```cpp
cs::Model model = cs::create_model<Pipeline>(n, R);
```

A model can be configured by passing a `cs::model_config_desc` first:

```cpp
cs::Model model = cs::create_model<Pipeline>(
    cs::model_config_desc(
        cs::pool_desc(gpu, 1u << 30, cs::node_size_desc(4096)),
        cs::async_handler_desc(handler)),
    n, R);
```

- **`cs::pool_desc(target, bytes[, cs::node_size_desc(n)])`** pre-allocates a pool of memory on each device of `target`: one `sycl::device`, a list of devices, or `cs::all_devices`. The pool is a dynamic allocator that the model's allocations on that device draw from. It is a performance optimization, useful for example when shapes change often enough that allocating from the system each time would cost too much. A device has at most one pool, and a device without one allocates from the system.
- **`cs::node_size_desc`** sets the pool's page size. The default is 1024 bytes.
- **`cs::async_handler_desc`** receives asynchronous errors.

The descriptors can be given in any order. Every one is optional.

## setup

`setup()` builds the whole tree: every system's and every component's `setup()`, and every connection, promotion and tie. Mistakes in the structure are reported here. Nothing is allocated or run. Calling `setup()` again resets the model and builds it from scratch; handles taken before the reset throw.

A model whose root is a plain `cs::System` can be built from `main`: `cs::create_model<cs::System>()`, then `model.add_subsystem(...)` before `setup()`.

## The edit window and finalize

Between `setup()` and `finalize()` the structure can still be edited from `main`: connect, promote, tie, `init_input`, shape hooks, `depends_on`, `set_device` and the optimization marks, through the model's root handle or the handle of any system. `finalize()` ends the window. It fixes devices, places shape hooks and tied-output pushes, and places `depends_on`. You can call it yourself; otherwise the first `run()` does. After it, structural calls throw `std::logic_error`, and host views become available.

## Handles

`model.get_subsystem(path)` returns a handle to a subsystem: a system or a component. A handle offers `get_input<DT, Rank>(name)`, `get_output<DT, Rank>(name)`, the view calls, `update_input`, `list_inputs()` and `list_outputs()`; a system's handle also takes the declaration calls, `set_run_policy` and `clear_run_policy`. The model's own calls are the root handle's. A stage of a fused group is not a subsystem of its own, so asking for one throws.

## run

`run()` finalizes the model if needed, then runs it: every component in scope runs, in an order the data allows. It returns once the work is submitted, without waiting for it to finish. It can be called as many times as needed; each run sees any inputs updated since the last.

`wait()` blocks until everything submitted so far has finished, and rethrows any asynchronous error.

## Feeding and reading

- **`update_input(path, initializer)`** gives a free input new values from the host. They are applied on the next run.
- **`view_output` and `view_input`** give a view of a value on the host, readable after `sync()`. They are available after finalize.
- **`rearm_subsystem(path)`** puts a finished subsystem (and everything under it) back in scope between runs.

More: [Inspecting your models](inspecting.md).
