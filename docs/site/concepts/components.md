# Components

A **component** is a unit of work in CoSpan. It has inputs and outputs, much like the parameters and results of a free function, and it computes its outputs from its inputs.

## Reading a component

This is the component from [Getting started](../getting-started.md):

```cpp
class Axpb : public cs::Component<Axpb> {
public:
    struct Inputs  { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };

    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", cs::unspecified);
        declare_output(outputs.y, "y", cs::unspecified);
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};
```

- **`Inputs` and `Outputs`** describe the inputs and outputs: their types and ranks.
- **`compute_forward`** implements the work. Here, each index of the range computes one element of `y` from one element of `x`.
- **`inputs` and `outputs`** hold the inputs and outputs.
- **`declare()`** runs once, when the model is set up. It declares each input and output with a name, and with a shape or `cs::unspecified`.
- **`prepare_forward()`** runs before every forward, and is optional. Here it sets the shape of `y` from the shape of `x`. The shape of `x` comes from outside the component: see [Setting shapes](todo/setting-shapes.md).
- **`define_forward()`** describes the forward's launch. Here it gives a range, the shape of `x`, much like the range of a SYCL `parallel_for`.

## A map from inputs to outputs

A component is very nearly a pure function. On every forward, its outputs are computed fresh from its inputs; they are never initialized.

## Other component types

There are two other component types:

- **[`cs::CustomComponent<X>`](todo/custom-components.md)**: native backend work, through AdaptiveCpp's [custom operation extension](https://adaptivecpp.github.io/AdaptiveCpp/enqueue-custom-operation/).
- **[`cs::HostComponent<X>`](todo/host-components.md)**: a body that runs on the host, over host memory.
