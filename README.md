# CospanMDAO

**GPU-native modeling for multidisciplinary design.** Compose systems of components, connect them by shared variables,
and run them on the GPU (or any SYCL device) through open standards: SYCL via
[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp). C++20, header-only, namespace `naga::cospan` (`cs::`).

CospanMDAO is part of the **NAGA ecosystem** ([NAGAGroup](https://github.com/NAGAGroup)): ecosystem code shares the
`naga::` namespace and the `include/naga/` root.

## Lineage

CospanMDAO is a GPU-native successor to the ideas in [OpenMDAO](https://openmdao.org/): components declaring typed I/O,
systems composing them with promotions and connections. The name says both halves: *cospans* are category theory's
formalism for composing open systems along shared boundaries, and *MDAO* credits the multidisciplinary-design lineage.

## Core concepts

- **Component.** A leaf that does work. It declares typed inputs and outputs (`cs::Input`, `cs::Output`), and its kernel
  body is one function whose signature fixes its kind: plain, group (work-groups, local memory), custom-op or host.
- **System.** A node that orders work. It adds children with `add_subsystem`, wires them with `connect`, `promote`,
  `tie_inputs` and `tie_outputs`, and has no I/O of its own beyond those aliases.
- **Variable.** Everything joined by wiring is one variable. Its metadata is shared by all of its ports, and its memory lives
  while some port on that device holds it. A connection costs nothing at run time.
- **Shapes.** A component says per dimension only "specified" or `cs::unspecified`. A shape comes from a connection, an
  initializer, a producer's `pre_forward()` (`set_shape`), or a System's `shape_hook`.
- **Run.** `Model::run()` enqueues one run and returns. Kernels are ordered by events, never by queue order, so
  independent work overlaps. Run policy (`run_n`, `do_once`, `transient`) is set on a subsystem; a finished
  subsystem goes out of scope and releases its data, and `Model::rearm_subsystem` brings it back.
- **Host access.** `Model::update_input` is the only write; `view_output` gives a host view of a result after a run, and
  `sync()` makes it current.
- **Memory.** One [SlabKit](https://github.com/NAGAGroup/SlabKit) pool per device, sized by you, with a counted OS fallback.

## A small example

Two copies of a component that doubles its input, connected in a chain:

```cpp
#include <naga/cospan/cospan.hpp>

namespace cs = naga::cospan;
using cs::DType;

class Double : public cs::Component<Double> {
public:
    struct Inputs  { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };
    // the kernel: one work-item per element (its first parameter makes this a plain, rank-1 body)
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i]; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {                                  // declare I/O; allocates nothing
        declare_input(inputs.x, "x", cs::unspecified);
        declare_output(outputs.y, "y", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }   // the output follows the input
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};

class Chain : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Double>("a"));
        add_subsystem(cs::subsystem_desc<Double>("b"));
        connect("a.y", "b.x");                               // b reads what a wrote
    }
};

int main() {
    cs::Model model = cs::create_model<Chain>(
        cs::model_config_desc{cs::pool_desc{cs::all_devices, 64u << 20}});  // a 64 MiB pool on every device
    model.setup();                                           // wiring and shape rules are checked here

    model.update_input("a.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4}));  // supplies a.x and its shape
    model.run();                                         // enqueues and returns; the update is applied here

    auto y = model.view_output<DType::f32, 1>("b.y");        // views are available once the model is finalized (the first run does it)
    y.sync();                                                // wait for the data before reading
    // y.shape()[0] == 4, y[0] == 4.0f, y[3] == 16.0f
}
```

A System can also size an input from another variable. This makes `c.x` as long as `p.y`:

```cpp
add_subsystem(cs::subsystem_desc<Producer>("p"));
add_subsystem(cs::subsystem_desc<Double>("c"));
shape_hook("c.x", cs::dim::from_shape("p.y", 0));
```

### Fused groups

Components with plain or work-group bodies can run as the stages of ONE kernel launch. A *segment* is a run of consecutive
stages that share work-groups; a bare component is a one-stage segment. Segment options pick the range
(`range_mode::own`, the default: a stage runs for its own groups; `whole`: every stage runs for every group of the
segment) and the local memory (`local_mem::partitioned`, the default: each stage its own slice; `shared`: one block):

```cpp
add_subsystem(cs::fuse_components(
    "f", cs::connects_desc({{"a.y", "b.x"}}),                    // connects run forward in stage order
    cs::segment<cs::range_mode::whole>(cs::subsystem_desc<StageA>("a"),   // a and b share their groups
                                       cs::subsystem_desc<StageB>("b")),
    cs::segment(cs::subsystem_desc<StageC>("c"))));            // c has its own groups, in the same launch
```

`cs::global_barrier()` between two segments of work-group stages makes every group of the later segment wait for every group
of the earlier ones. The fused-group property `cs::group_waits_enabled`
(`cs::fuse_components<cs::group_waits_enabled>(...)`) lets stages wait for earlier groups themselves
(`GroupItem::wait_group`, `wait_earlier_segments`) without every group being resident; a barrier implies it. See `examples/gemver-chain` for fused groups in a real chain.

## Using CospanMDAO

CospanMDAO is a [pixi](https://pixi.sh) **source package**: dependency management lives in pixi, not in CMake. pixi is the
one host dependency; it builds the dependency chain (SlabKit, cereal, the AdaptiveCpp toolchain) from pinned sources, the
same way on every machine.

**From a pixi project (recommended):**

```toml
# pixi.toml
[workspace]
preview = ["pixi-build"]

[dependencies]
cospanmdao = { git = "https://github.com/NAGAGroup/CospanMDAO.git", tag = "v0.1.0" }
```

then in CMake: `find_package(cospanmdao CONFIG REQUIRED)` and link `naga::cospan`. A host dependency is enough: cospanmdao
run-exports SlabKit, so its shared library comes along at run time.

**Support policy:** the pixi pipeline is the only officially supported way to build and consume CospanMDAO. Other embeddings
may work, but issues specific to them will be closed.

## Developing CospanMDAO

```sh
pixi run test                       # build and run the test suite (OpenMP host backend)
pixi run -e cuda test               # the same on CUDA
pixi run gemver-chain               # the GEMVER chain example (see examples/gemver-chain/README.md)
pixi run -e cuda gemver-chain       # on CUDA
pixi run -e dev <command>           # the dev environment: the full build closure, nothing built
```

The `dev` and `dev-cuda` environments carry the package's full dependency closure without building the package.
`pixi run -e git setup-git` points git at the project hooks.

## Documentation

- **Specification:** [`docs/spec/revisions/r03-2026-10-06-section-11-as-built.md`](docs/spec/revisions/r03-2026-10-06-section-11-as-built.md)
  (r02 and r01 are superseded and kept as history).
- **Migrating from the API before the 2026-10-03 rework:** [`docs/rework-migration-2026-10-03.md`](docs/rework-migration-2026-10-03.md).
- **Decisions:** [`docs/phase1-decisions.md`](docs/phase1-decisions.md) (D144 is the rework) and
  [`docs/design-review-2026-10-03.md`](docs/design-review-2026-10-03.md).
- **Roadmap:** [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Status

v0.1.0. The API is not yet stable. Chunked systems, slices and leading ranges are not supported at present: they are being
redesigned (the old code is kept as dead code in `include/naga/cospan/chunked_system.hpp`).

## License

Apache-2.0. Copyright 2026 Jack Myers.
