# Getting started

**CoSpan** (CospanMDAO) is a C++20 framework for heterogeneous numerical applications, built on [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)'s SYCL.

You build an application out of small, composable parts called **components**. You compose them into **systems**, and connections say how data flows between them. From that description, CoSpan runs the whole application on whatever hardware you have, so you write the numerics and not the plumbing.

This page builds the smallest complete CoSpan application: one component that computes `y = a·x + b`, run once, with its result printed. Along the way it shows the three files every CoSpan project has: a pixi manifest, a CMake file and the source.

## What you need

[pixi](https://pixi.sh). That's all: pixi brings the compiler, CMake and CoSpan itself.

## 1. The pixi manifest

`pixi.toml` describes the project and everything it needs.

- **The project is itself a package** (`[package]`), built by pixi's CMake backend.
- **CoSpan** is a host dependency, fetched from its git repository.
- **The build variants** select NAGA Labs' AdaptiveCpp compiler toolchain. Every CoSpan project needs this block as it is.
- **The default environment** installs the project's own package, and one task runs it.
- **Every device runtime** is included, for every platform it is published for, so the project runs on whatever hardware is present. Remove the runtimes for hardware you don''t have: AdaptiveCpp then stops trying to load them, and stops warning that they are missing.

```toml
[workspace]
name      = "getting-started"
version   = "0.1.0"
channels  = ["https://prefix.dev/jackm97/naga-labs-staging", "conda-forge"]
platforms = [
  { platform = "linux-64", archspec = "x86_64_v3" },
  "linux-aarch64",
  { platform = "osx-arm64", macos = "26.0" },
  { platform = "win-64", archspec = "x86_64_v3" },
  "win-arm64",
]
preview   = ["pixi-build"]

[workspace.build-variants]
c_compiler           = ["naga-acpp-clang"]
cxx_compiler         = ["naga-acpp-clangxx"]
c_compiler_version   = ["2026.10.21"]
cxx_compiler_version = ["2026.10.21"]
naga-acpp-llvm       = ["23"]
cmake                = ["4.4.*"]

[package]
name    = "getting-started"
version = "0.1.0"

[package.build.backend]
name     = "pixi-build-cmake"
version  = "*"
channels = ["https://prefix.dev/pixi-build-backends"]

[package.host-dependencies]
cospanmdao = { git = "https://github.com/NAGAGroup/CospanMDAO.git", subdirectory = "cospanmdao" }

[dependencies]
getting-started = { path = "." }

# Every device runtime, on every platform it is published for: the project runs on whatever is present. Remove a
# runtime for hardware you don''t have and AdaptiveCpp stops trying to load it (and stops warning that it is missing).
# macOS has none to add: Metal is part of the base runtime. intel-opencl-rt is Intel''s CPU OpenCL driver, so the
# OpenCL backend always has a device on x86-64.
[target.linux-64.dependencies]
naga-acpp-runtime-cuda       = "*"
naga-acpp-runtime-level-zero = "*"
naga-acpp-runtime-ocl        = "*"
naga-acpp-runtime-rocm       = "*"
intel-opencl-rt              = "*"

[target.linux-aarch64.dependencies]
naga-acpp-runtime-cuda = "*"
naga-acpp-runtime-ocl  = "*"

[target.win-64.dependencies]
naga-acpp-runtime-cuda       = "*"
naga-acpp-runtime-level-zero = "*"
naga-acpp-runtime-ocl        = "*"
naga-acpp-runtime-rocm       = "*"
intel-opencl-rt              = "*"

[target.win-arm64.dependencies]
naga-acpp-runtime-ocl = "*"

[tasks]
getting-started = "getting-started"
```

The toolchain targets x86-64-v3 (Haswell, Zen and newer) on linux-64 and win-64 and macOS 26 on osx-arm64, and pixi needs that archspec and macOS version stated to solve.

## 2. The CMake file

CMake only describes the build. Dependencies are pixi's job, so nothing is fetched here: find the two packages, build one executable, and let AdaptiveCpp compile its kernels.

```cmake
cmake_minimum_required(VERSION 3.24)
project(getting-started LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(AdaptiveCpp REQUIRED)
find_package(cospanmdao REQUIRED)

add_executable(getting-started main.cpp)
target_link_libraries(getting-started PRIVATE naga::cospan)
add_sycl_to_target(TARGET getting-started SOURCES main.cpp)

install(TARGETS getting-started)
```

## 3. The source

A complete model in one file:

- **The component** is one class. Its nested `Inputs` and `Outputs` are its inputs and outputs, the static `compute_forward` is the kernel (what one work-item computes; a first parameter of `sycl::id<R>` makes it an *index-range* component), and the public members `inputs` and `outputs` hold them. It declares them in `declare()`. In `prepare_forward()` it gives its output the shape of its input. In `define_forward()` it launches over the input's range. The other kind, the group-range component, works in work-groups and must also state their shape; see [Components](concepts/components.md).
- **`main`** builds the model from a plain root, adds the component to it, gives it an input, runs it once and reads the output back on the host.

```cpp
#include <iostream>
#include <vector>

#include <naga/cospan/cospan.hpp>

namespace cs = naga::cospan;
using cs::DType;

// The component: y = a * x + b (here a = 2, b = 1), one work-item per element.
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

int main() {
    cs::Model model = cs::create_model<cs::System>();
    model.add_subsystem(cs::subsystem_desc<Axpb>("axpb"));
    model.setup();

    model.update_input("axpb.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4}));
    model.run();

    auto y = model.view_output<DType::f32, 1>("axpb.y");
    y.sync();
    for (std::size_t i = 0; i < y.shape()[0]; ++i) std::cout << y[i] << ' ';
    std::cout << '\n';
}
```

## 4. Run it

```shell
pixi run getting-started
```

The first run takes a while, because pixi builds CoSpan's dependencies and the project. Then:

```text
3 5 7 9
```

## Where next

- **[Concepts](concepts/index.md):** the mental model behind components, systems and variables.
- **[Academy → Main series](academy/main-series/index.md):** building a real project step by step.
