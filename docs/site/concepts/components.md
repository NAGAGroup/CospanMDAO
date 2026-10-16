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


## Two kinds of kernel component

The first parameter of `compute_forward` fixes the kind of a kernel component.

- **Index-range** (`sycl::id<R>` first): the kernel runs once per index of the range. The launch description is
  `create_launch_config(range_desc(r))`. There is no work-group shape, no group functions and no local memory:
  `wg_desc` and `local_desc` are rejected at compile time.
- **Group-range** (`cs::GroupItem<R>` first): the range counts work-items and is split into work-groups of the shape given
  by `wg_desc`. The work-group shape is **required**, the way an `nd_range` always states its local range: a group-range
  component without a `wg_desc` does not compile. The range **must be a multiple of the work-group shape in every
  dimension**: you round it up in the component and guard the tail in the body; CoSpan does not round for you, and
  `create_launch_config` throws `std::invalid_argument` naming the component path, the range and the shape when it is not a
  multiple. An empty range is an empty stage.

```cpp
std::unique_ptr<cs::LaunchDescription> define_forward() override {
    const std::size_t n = (inputs.x.shape()[0] + 63) / 64 * 64;       // whole work-groups
    return create_launch_config(cs::range_desc(sycl::range<1>{n}), cs::wg_desc(sycl::range<1>{64}));
}
// in the body: if (it.get_global_id(0) >= inputs.x.shape()[0]) return;   (after any group barrier)
```

### Local memory

A group-range body that needs work-group local memory takes a `cs::LocalMem` as its fourth parameter, one
`cs::LocalArray<DType, Rank>` per array, and the component gives the shapes with `local_desc`:

```cpp
using Mem = cs::LocalMem<cs::LocalArray<DType::f32, 1>, cs::LocalArray<DType::i32, 2>>;
static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out, Mem lm) {
    auto [tile, counts] = lm.get(it);          // views: tile[l], counts[sycl::id<2>{i, j}]
    ...
}
// define_forward: create_launch_config(range_desc(r), wg_desc(wg), local_desc(sycl::range<1>{64}, sycl::range<2>{4, 3}))
```

Each array starts at an offset aligned to its element type. The number and ranks of the `local_desc` shapes must match the
arrays (a mismatch does not compile). In a fused component, `segment<local_mem::shared>` gives every stage of the segment
the same block and `partitioned` gives each stage its own slice; segment local-memory options, `global_barrier` and group
waits apply to group-range components only.
