# CoSpan

[![CI](https://github.com/NAGALabsOrg/CospanMDAO/actions/workflows/ci.yml/badge.svg)](https://github.com/NAGALabsOrg/CospanMDAO/actions/workflows/ci.yml)
[![version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FNAGALabsOrg%2FCospanMDAO%2Fmain%2Fcospanmdao%2Fpixi.toml&query=%24.package.version&label=version)](cospanmdao/pixi.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**Heterogeneous computing by composition.**

Writing fast code for GPUs and other accelerators usually means writing two programs. One is the numerics. The other is the orchestration around them: allocating memory on each device, moving data, launching kernels, and wiring up the dependencies that keep everything in order. The orchestration is most of the code and most of the bugs, and it is why the clean design a project starts with rarely survives performance tuning.

CoSpan lets you write only the first program. You build your application as a model: components, each a small, self-contained piece of numerics, whose inputs and outputs you connect into shared variables. A CoSpan model is a fully declared program. Everything about how it will execute is known to the runtime before it runs, and that changes what a framework can do for you.

Because the whole model is known, the orchestration problems that have a right answer get the right answer automatically, and doing them by hand would not do better. Because the model is one consistent representation, harder optimizations can be written once, inside CoSpan, and apply to every model, without you thinking about them. And where the best choice depends on your problem, CoSpan gives you a simple way to make it yourself instead of guessing for you.

None of this would be possible without [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp). Its single-pass compiler keeps your kernels as portable code that is compiled, and specialized, at run time for the device that is actually present. That is what lets CoSpan be an ordinary C++ library rather than a compiler of its own, and assemble the kernels a model needs while the program runs. It is also why CoSpan runs on whatever hardware AdaptiveCpp supports: its device backends are loaded at run time, so support for new hardware arrives as a new backend, without rebuilding AdaptiveCpp, CoSpan, or your application. In AdaptiveCpp's own peer-reviewed benchmarks, its JIT-specialized code outran the vendors' compilers on their own hardware.

That raises a question most frameworks never have to ask. When you could hand-write a native, vendor-specific kernel instead, should you? You still can, and for vendor-specific hardware such as tensor cores you will want to. But a hand-tuned kernel is tuned for one runtime state, while CoSpan's kernels keep being specialized for the state your program is actually in. With CoSpan, it is a question worth asking.

The result is code you can read, reuse and grow: each piece written once, small and testable, and composed into larger models without being rewritten. In our early experiments, chains of ordinary building blocks composed this way have outrun vendor libraries on the same hardware.

## Features

- **Modeling in the OpenMDAO tradition.** Models are built from components and systems, with named, typed inputs and outputs, as in [OpenMDAO](https://openmdao.org/), the NASA framework for multidisciplinary design, analysis and optimization.
- **Fused kernels.** Several components can run as a single kernel. Work that would otherwise be many small kernel launches, each waiting on the last, runs as one, while each component stays a small, focused unit of computation. Components that use work-groups and local memory can be fused too.
- **Component variants.** Kernel components hold device numerics, written per element or per work-group. Host components bring host-side work into the model, ordered with everything else. Custom components give you full control, down to raw SYCL or vendor-native code.
- **Promotions.** A system can expose inputs and outputs from deep inside it under its own names. Large nested models stay readable, and a system can present a clean interface to whatever uses it.
- **Ties.** Inputs can be tied together, and so can outputs, making them one variable. Components developed separately stay composable, even when they have to work on the same data.
- **Heterogeneous models.** One model can span several devices at once, from different vendors and of different kinds, with its parts placed where they run best.

## A taste

Two components, connected:

```cpp
#include <naga/cospan/cospan.hpp>

namespace cs = naga::cospan;
using cs::DType;

class Double : public cs::Component<Double> {
public:
    struct Inputs  { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i]; }

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

class Chain : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Double>("a"));
        add_subsystem(cs::subsystem_desc<Double>("b"));
        connect("a.y", "b.x");
    }
};

int main() {
    cs::Model model = cs::create_model<Chain>();
    model.setup();
    model.update_input("a.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4}));
    model.run();

    auto y = model.view_output<DType::f32, 1>("b.y");
    y.sync();   // 4 8 12 16
}
```

## Platforms

| | CPU | NVIDIA | AMD | Intel GPU | Apple GPU |
|---|:---:|:---:|:---:|:---:|:---:|
| **Linux x86-64** | ![](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FNAGALabsOrg%2FCospanMDAO%2Fbadges%2Flinux-64-cpu.json) | ![](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FNAGALabsOrg%2FCospanMDAO%2Fbadges%2Flinux-64-nvidia.json) | ![](https://img.shields.io/badge/-untested-lightgrey) | ![](https://img.shields.io/badge/-untested-lightgrey) | — |
| **Linux ARM64** | ![](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FNAGALabsOrg%2FCospanMDAO%2Fbadges%2Flinux-aarch64-cpu.json) | ![](https://img.shields.io/badge/-untested-lightgrey) | — | — | — |
| **macOS ARM64** | ![](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FNAGALabsOrg%2FCospanMDAO%2Fbadges%2Fosx-arm64-cpu.json) | — | — | — | ![](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FNAGALabsOrg%2FCospanMDAO%2Fbadges%2Fosx-arm64-apple.json) |
| **Windows x86-64** | ![](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FNAGALabsOrg%2FCospanMDAO%2Fbadges%2Fwin-64-cpu.json) | ![](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FNAGALabsOrg%2FCospanMDAO%2Fbadges%2Fwin-64-nvidia.json) | ![](https://img.shields.io/badge/-untested-lightgrey) | ![](https://img.shields.io/badge/-untested-lightgrey) | — |
| **Windows ARM64** | ![](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FNAGALabsOrg%2FCospanMDAO%2Fbadges%2Fwin-arm64-cpu.json) | — | — | — | — |

*Untested* means a runtime is available but we have no hardware to test it on.

## Status

CoSpan is early, and its API is still changing. Derivatives, solvers for coupled systems, and optimizers, the rest of what modeling frameworks like OpenMDAO provide, are not here yet. They are where CoSpan is going next.

## Getting started

CoSpan is a [pixi](https://pixi.sh) package:

```toml
cospanmdao = { git = "https://github.com/NAGALabsOrg/CospanMDAO.git", branch = "main", subdirectory = "cospanmdao" }
```

The [getting-started guide](docs/site/getting-started.md) walks through a complete project, and the [documentation](docs/site/) covers the concepts in depth.

## About

CoSpan is developed by [NAGA Labs](https://github.com/NAGALabsOrg), alongside [SlabKit](https://github.com/NAGALabsOrg/SlabKit), the device-memory library it is built on. The name comes from category theory, where *cospans* describe how open systems compose along their shared boundaries.

## License

Apache-2.0. Copyright 2026 NAGA Labs LLC.
