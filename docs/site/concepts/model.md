# Model

A **model** takes a system you've built and turns it into something you can run.

```cpp
cs::Model model = cs::create_model<Chain>();
model.setup();

model.update_input("first.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4}));
model.run();

auto y = model.view_output<DType::f32, 1>("second.y");
y.sync();
```

## Creating a model

`cs::create_model<Sys>(args...)` makes a model from a system of type `Sys`. The arguments go to the system's constructor:

```cpp
cs::Model model = cs::create_model<Pipeline>(n, R);
```

A model can also be built procedurally, from a generic system: `cs::create_model<cs::System>()`, then `model.add_subsystem(...)` before `setup()`, as in [Getting started](../getting-started.md).

## setup

`setup()` builds and configures your system so it can be run. Calling it again resets the model.

## finalize

After `setup()`, there is one more step before a model can run: `finalize()`. `run()` calls it for you, or you can call it yourself.

Between `setup()` and `finalize()`, you can still edit your model, except to add subsystems. Connections, promotions and ties are examples. Other kinds of edit are covered elsewhere in this documentation, and each one says whether it can be made here.

## Handles

`model.get_subsystem(path)` returns a handle to a subsystem.

## run and wait

`run()` runs your model asynchronously. `wait()` waits until it is done. `run()` can be called again without waiting: each run is queued behind the last. Calling `wait()` after several `run()` calls waits for all of them to finish.

## Setting inputs and inspecting your model

- **`update_input(path, initializer)`** sets a free input, from an initializer, for the next `run()`. Setting a connected input is a no-op, not an error.
- **`view_output` and `view_input`** let you inspect a model's outputs and inputs. They are available after `finalize()`. A view's `sync()` waits for anything still writing to it.

More: [Inspecting your models](inspecting.md).
