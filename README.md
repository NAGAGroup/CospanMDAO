# CospanMDAO

**GPU-native differentiable modeling and optimization.** Compose systems of
components, connect them by zero-copy pointer aliasing, settle coupled
systems to equilibrium, and differentiate through the fixed point — all on
the GPU, all through open standards (SYCL via
[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)).

CospanMDAO is part of the **NAGA ecosystem**
([NAGAGroup](https://github.com/NAGAGroup)) — ecosystem code shares the
`naga::` namespace umbrella and `include/naga/` root; this framework is
`naga::cospan`.

## Lineage

CospanMDAO is a GPU-native successor to the ideas in
[OpenMDAO](https://openmdao.org/): components declaring typed I/O, systems
composing them with promotions and connections, coupled models driven to
consistency, and gradients propagated through the whole structure. The name
says both halves out loud — *cospans* are category theory's formalism for
composing open systems along shared boundaries, and the *MDAO* suffix
credits the multidisciplinary-design lineage the ML world usually ignores.

Where OpenMDAO orchestrates Python components on CPUs, CospanMDAO is C++20
on device: connections are pointer aliasing (no copies, ever), execution is
layered and concurrent, and adjoints are hand-written SYCL kernels flowing
through the same aliased storage.

## Key ideas

- **Component / System composition.** Components declare inputs, outputs,
  and state; systems compose components, promote shared inputs, and wire
  connections. Everything resolves to fully-qualified names in one registry.
- **Connection = aliasing.** A connection makes the target's data and grad
  pointers alias the source. Composition costs nothing at runtime.
- **Cycles are first-class.** Marking a component as an *entry point* makes
  connections into it previous-sweep reads (a register cut), turning cyclic
  models into layered DAGs — Gauss-Seidel sweeps expressed structurally.
- **Equilibrium as an execution mode.** `EquilibriumExecutor` iterates the
  forward model to a fixed point, then solves the adjoint fixed point
  (implicit function theorem — memory cost independent of iteration count;
  no tape, no BPTT).
- **Hand-written adjoints.** Every component implements `backward()` as a
  SYCL kernel; gradients accumulate atomically through the aliased graph.
  Stop-gradient is simply not accumulating.
- **Optimizers included.** Adam (with objective-reduction stopping and
  step gating) and Langevin dynamics, driving any input marked as a design
  variable.
- **Memory via [SlabKit](https://github.com/NAGAGroup/SlabKit).** All device
  memory flows through SlabKit's USM pools and views.

## Example: the Sellar problem

[`examples/sellar`](examples/sellar/sellar.cpp) solves the canonical MDAO
test problem — two mutually-coupled disciplines optimized through their
fixed point:

```
[sellar] finished: 1000 iterations, objective 3.178551
[sellar]   x  = -0.0000   (expect ~0)
[sellar]   z1 = +1.9773   (expect ~1.978)
[sellar]   z2 = -0.0049   (expect ~0)
[sellar]   y1 = +3.1551   (expect ~3.16, constraint-active)
[sellar]   y2 = +3.7486   (expect ~3.755)
[sellar]   f  = +3.1786   (known optimum ~3.183)
```

The coupling cycle is closed with a single `connect()` plus an entry-point
mark; the optimizer differentiates through the settled system.

## Using CospanMDAO

CospanMDAO is a [pixi](https://pixi.sh) **source package** — dependency
management lives in pixi, not in CMake. pixi is the one host dependency; it
builds the entire dependency chain (SlabKit, cereal, the AdaptiveCpp
toolchain) hermetically from pinned sources, the same way on every machine.
If you haven't met pixi-build yet: your project declares this repo as a
dependency, and `pixi install` does the rest — no submodules, no
FetchContent, no system SYCL install.

**From a pixi project (recommended):**

```toml
# pixi.toml
[workspace]
preview = ["pixi-build"]

[dependencies]
cospanmdao = { git = "https://github.com/NAGAGroup/CospanMDAO.git", tag = "v0.1.0" }
```

then in CMake: `find_package(cospanmdao CONFIG REQUIRED)` and link
`naga::cospan`.

**From a non-pixi project:** use pixi once to produce a fully-installed,
relocatable environment, then consume it like any installed CMake package:

```sh
pixi exec pixi-pack --environment pack --create-executable
./environment.sh        # self-extracting: creates ./env, no pixi/conda needed
```

Add the unpacked env to `CMAKE_PREFIX_PATH` and use the same
`find_package(cospanmdao CONFIG)` / `naga::cospan` — the packages ship proper CMake config
files.

**Support policy:** the pixi pipeline is the only officially supported way
to build and consume CospanMDAO. Conda's hermetic builds are the point:
what pixi produces is compatible with your project without you fighting
toolchain flags. Traditional embeddings (FetchContent, CPM, add_subdirectory
of a raw clone) may happen to work, but issues specific to them will be
closed.

## Developing CospanMDAO

```sh
pixi run configure dev
pixi run build dev
pixi run test
ACPP_VISIBILITY_MASK=omp ./build/dev/examples/sellar   # CPU
```

A `dev` environment (`pixi run -e dev ...`) provides the package's full
dependency closure for ad-hoc work inside the build-time environment.

## Status

v0.1.0 — extracted from a private research monorepo where the framework has
been developed and exercised since early 2026. The API is not yet stable.
Applications built on it so far include Mycelium (a private neural-field
architecture research project) and meshless discretization research.

Roadmap, roughly in order:

- Test suite rebuild (the research-era suite predates several core API
  revisions and did not make the extraction cut)
- Spatial/nonlocal compute toolboxes (LBVH, meshless operators) as
  companion libraries
- CI

## License

Apache-2.0. Copyright 2026 Jack Myers.
