# CospanMDAO

**GPU-native differentiable modeling and optimization.** Compose systems of
components, connect them by zero-copy pointer aliasing, settle coupled
systems to equilibrium, and differentiate through the fixed point — all on
the GPU, all through open standards (SYCL via
[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)).

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

## Building

Requirements are managed by [pixi](https://pixi.sh) — the only host
dependency. Toolchain: AdaptiveCpp (SSCP / `generic` target), CMake + Ninja,
C++20.

```sh
pixi run configure dev
pixi run build dev
ACPP_VISIBILITY_MASK=omp ./build/dev/examples/sellar   # CPU
./build/dev/examples/sellar                            # default device (GPU if visible)
```

Dependencies ([SlabKit](https://github.com/NAGAGroup/SlabKit), cereal) are
**pixi-build source packages** — `pixi install` fetches and builds them from
source automatically. No submodules, no vendoring. CospanMDAO is itself a
source package: depend on it with
`cospanmdao = { git = "https://github.com/NAGAGroup/CospanMDAO.git" }`.

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
- Namespace/include-path rename (`naga` → `cospanmdao`); the current root
  include path is inherited from the research tree
- CI

## License

Apache-2.0. Copyright 2026 Jack Myers.
