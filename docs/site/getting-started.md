# Getting started

**CoSpan** (CospanMDAO) is a C++20 framework for building GPU-native numerical applications out of small, composable parts.

You write **components**: each one declares typed inputs and outputs and runs one SYCL kernel. You put components together in **systems**, connecting outputs to inputs. CoSpan takes care of the rest: memory on each device, the order kernels must run in, overlapping independent work, and moving data where it needs to be. You never write synchronization yourself. The same code runs on a GPU, or on the CPU when there isn't one, through SYCL and [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp).

This page builds the smallest complete CoSpan application: one component that computes `y = a·x + b`, run once, with its result printed. Along the way it shows the three files every CoSpan project has: a pixi manifest, a CMake file and the source.

## What you need

[pixi](https://pixi.sh), on Linux (x86-64). Nothing else: pixi brings the compiler, CMake and CoSpan itself.

## 1. The pixi manifest

`pixi.toml` describes the project and everything it needs.

- **The project is itself a package** (`[package]`), built by pixi's CMake backend.
- **CoSpan** is a host dependency, fetched from its git repository.
- **The build variants** select NAGA Labs' AdaptiveCpp compiler toolchain. Every CoSpan project needs this block as it is.
- **The default environment** installs the project's own package, and one task runs it.

```toml
[workspace]
name      = "getting-started"
version   = "0.1.0"
channels  = ["https://prefix.dev/jackm97/naga-labs-staging"]
platforms = ["linux-64"]
preview   = ["pixi-build"]

[workspace.build-variants]
c_compiler           = ["naga-acpp-clang"]
cxx_compiler         = ["naga-acpp-clangxx"]
c_compiler_version   = ["2026.10.06"]
cxx_compiler_version = ["2026.10.06"]
naga-acpp-llvm       = ["21"]
cmake                = ["3.31.*"]

[package]
name    = "getting-started"
version = "0.1.0"

[package.build.backend]
name     = "pixi-build-cmake"
version  = "*"
channels = ["https://prefix.dev/pixi-build-backends"]

[package.host-dependencies]
cospanmdao = { git = "https://github.com/NAGAGroup/CospanMDAO.git" }

[dependencies]
getting-started = { path = "." }

[tasks]
getting-started = "getting-started"
```

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

- **The component** is one class: its nested `Inputs` and `Outputs` are its ports, the static `compute_impl` is the kernel (what one work-item computes), and the public members `inputs` and `outputs` hold the ports. It declares them in `setup()`, says its output has the shape of its input, and launches `compute_impl` over the input's range.
- **The system** is the model's root. Here it holds just the one component.
- **`main`** builds the model, gives it an input, runs it once and reads the output back on the host.

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
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.x, "x", cs::unspecified);
        declare_output(outputs.y, "y", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};

// The model's root: a system holding the one component.
class App : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("axpb")); }
};

int main() {
    cs::Model model = cs::create_model<App>();
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
