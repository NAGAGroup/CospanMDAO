# Components

A **component** is the unit of work in CoSpan. It is a leaf of the model: it does one kind of work, written as a single kernel.

A component talks to the rest of the model only through the inputs and outputs it declares. It never knows who produces its inputs or who reads its outputs: wiring components together is the job of [systems](systems.md).

## Reading a component

This is the component from [Getting started](../getting-started.md):

```cpp
class Axpb : public cs::Component<Axpb> {
public:
    // Typed inputs and outputs, grouped in two nested structs.
    struct Inputs  { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };

    // The kernel. It is static and sees only its inputs and outputs.
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i] + 1.0f; }

    // The component's inputs and outputs.
    Inputs  inputs;
    Outputs outputs;

    // Runs once: declare each input and output, with a name and a shape (or leave the shape unspecified).
    void setup() override {
        declare_input(inputs.x, "x", cs::unspecified);
        declare_output(outputs.y, "y", cs::unspecified);
    }

    // Runs before every forward.
    void pre_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }

    // Describes the launch: the range to run over.
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};
```

- **`Inputs` and `Outputs`** describe the component's inputs and outputs: their types and ranks.
- **`compute_impl`** is the kernel. It is static: it has no access to the component object, only to the inputs and outputs it is given.
- **`inputs` and `outputs`** hold the component's inputs and outputs.
- **`setup()`** runs once, when the model is set up. It declares each input and output with a name, and with a shape or `cs::unspecified`. Declaring allocates nothing.
- **`pre_forward()`** runs before every forward. It is optional. It is typically where an output takes its shape, often from an input as here, but it is the component's own hook to use as it needs.
- **`compute_forward()`** describes the launch of this forward. Its arguments are descriptors: `cs::range_desc` (the range), and optionally `cs::wg_desc` (the work-group shape) and `cs::scratch_desc` (local memory).
- **`post_forward()`**, not shown, runs after every forward and is also optional. We have not needed it ourselves yet. If you find yourself reaching for it, check that it is really necessary.

## The four kinds of kernel

The first parameter of `compute_impl` decides what kind of kernel a component has:

```cpp
static void compute_impl(sycl::id<R> i,       const Inputs& in, const Outputs& out);  // plain
static void compute_impl(cs::GroupItem<R> it, const Inputs& in, const Outputs& out);  // group
static void compute_impl(cs::CustomOp& op,    const Inputs& in, const Outputs& out);  // custom operation
static void compute_impl(cs::Host,            const Inputs& in, const Outputs& out);  // host
```

- **Plain.** One work-item per index of the range.
- **Group.** Work-items cooperate in work-groups, with local memory, group barriers and group algorithms.
- **Custom operation.** CoSpan's form of AdaptiveCpp's [custom operation extension](https://adaptivecpp.github.io/AdaptiveCpp/enqueue-custom-operation/): native backend work, enqueued on the backend queue.
- **Host.** The kernel runs on the host, as an asynchronous task. Like every other kind it is ordered by the framework: it runs only once its inputs are written and while nothing else writes its outputs. Submitting it does not block.

## The functional model

- **Outputs are computed fresh on every forward, and are never initialized.** A component writes every element of its outputs it is responsible for.
- **A component keeps no data of its own between forwards.** Everything it reads comes in through its inputs.
- **Multi-step work is several components,** each one's outputs feeding the next one's inputs, not one component with hidden state.

## What a component does not do

- **It does not synchronize.** Ordering is the framework's job.
- **It does not allocate.** The memory behind its inputs and outputs is managed for it.
- **It does not know its neighbours.** Wiring belongs to systems.
- **It does not decide where it runs.** Placement belongs to systems.
