# ns-residual: one compressible Navier–Stokes residual evaluation, fused

One evaluation of `R = −div F_conv + div F_visc` per cell of a uniform 2D grid (f32, conserved variables `(ρ, ρu, ρv, E)`, ideal
gas γ = 1.4, constant viscosity μ = 0.01, Pr = 0.72, dx = dy = 1, no time stepping), plus three diagnostics:

- the minimum local timestep (CFL form 0.5, with the viscous term),
- the maximum wave speed,
- the sum of `R²` per variable (four values).

Every diagnostic is computed **twice**: atomically (`fetch_min`, `fetch_max`, `fetch_add` on `atomic_f32` outputs, one global
value) and deterministically (reductions of per-tile partials in a fixed order). An error stage writes `atomic − deterministic`.

Three variants compute exactly this, from the same initial field (uniform flow plus a smooth perturbation of fixed 32-cell
wavelength, generated on the device):

| variant | what it is |
|---|---|
| `unfused` | every stage its own CoSpan component and kernel, intermediates in global memory (padded arrays for the tile) |
| `fused` | **one** CoSpan launch (`cs::fuse_components`), five segments, one work-group of 16×16 per tile |
| `handwritten` | one monolithic SYCL kernel outside CoSpan: its own queue and USM buffers, an atomic ticket counter, the same schedule |

```
ns-residual [--variant all|unfused|fused|handwritten] [--nx 1024] [--ny 1024] [--iters 20] [--warmup 3] [--check]
            [--limiter minmod|vanleer] [--bc walls|periodic] [--range-split flatten|pad|padded|all]
```

`--range-split` picks the `cs::range_split` strategy of the fused variant (`all` runs the three, reported as `fused-flatten`,
`fused-pad`, `fused-padded`). The design is the same for every strategy; the stages are rank 2, so all three apply. Had a design
been rank 3, `padded` would be skipped, not forced to rank 2. `pixi run ns-residual ...` runs it (`-e cuda` for the GPU).

## Conventions (fixed by the host reference, `reference.hpp`, written first)

Arrays are row-major `(y, x)`. A tile of 16×16 cells **owns the west and south faces of its cells**; tiles on the east and north
domain edges also own the domain-boundary faces. The halo is 2 cells (a face needs its two cells and their MUSCL slopes).
Boundaries: either periodic in x with **no-slip adiabatic walls in y** (`--bc walls`, ghost cells mirror ρ and E and negate both
momenta), or fully periodic (`--bc periodic`). Reconstruction: MUSCL on primitives with a minmod or van Leer limiter. Convective
flux: HLLC (Davis speeds). Viscous flux: normal derivative from the face's two cells, tangential derivative from the mean of the
two cell-centred gradients, heat flux `k ∇T`. `physics.hpp` holds the numerics shared by the host (double) and the device (float).

## The fused layout, segment by segment

`cs::fuse_components<cs::group_waits_enabled, Split>("ns", cs::connects_desc(...), cs::ties_desc(...), items...)`. Every stage is a
rank-2 work-group stage, work-group 16×16, one work-group per tile.

| # | segment | stages | features used |
|---|---|---|---|
| 0 | **init**, one group | `InitDiagnostics`: min dt ← +inf, max wave speed ← −inf, four sums ← 0 | a segment of its own; **no barrier after it**; outputs are never initialized, so the atomics need this |
| 1 | **tile pipeline**, `segment<local_mem::shared>` | `LoadTile` (20×20 tile with halo into local memory), `ApplyBoundary<Bc>` (swappable component: walls or periodic; nothing on interior tiles), `ToPrimitive`, `Gradients` (u, v, T), `Reconstruct<Lim>` (MUSCL, swappable limiter), `ConvFlux<Dir>` ×2 (HLLC, **one** component template instanced for x, `Dir` 0, and y, `Dir` 1: the direction is a template parameter, not a runtime flag), `ViscFlux<Dir>` ×2 (the same, x and y) | `local_mem::shared` (the tile stays in local memory from stage to stage); `cs::LocalMem` / `local_desc` / `lm.get(it)` (structured bindings); swappable component types; one component type, two instances; the flux stages write the owned faces to global outputs |
| 2 | **divergence and diagnostics**, `segment<local_mem::partitioned>` | `ConvDivergence<true>` (writes R), `ViscDivergence` (adds into R), `LocalTimestep<true>`, `WaveSpeed<true>`, `ResidualSquares<true>` | `wait_group` **inside the body**, on exactly the tile-pipeline groups read (own tile, east and north neighbour, as `sycl::id<2>`); `cs::ties_desc` with a plain dtype and `allow_plain` for R, stage order sequencing the two writes; group reductions (`cs::reduce_over_group`) then `fetch_min` / `fetch_max` / `fetch_add` on atomic outputs and a per-tile partial; the three diagnostics each `wait_group` on the init group right before their atomic. Only stages that read other groups' data wait |
| – | `cs::global_barrier()` | | every group of the next segment waits for every group of the earlier segments |
| 3 | **deterministic reductions**, `segment<partitioned, range_mode::own>` | `DetNorm` (4 groups, one per variable), `DetMin` (1 group), `DetMax` (1 group) | `range_mode::own`: the one-group stages skip the other groups in any dimension; reductions of the per-tile partials in a fixed order |
| 4 | **error**, one group | `ErrorStage<true>`: six values, each atomic − deterministic | `wait_group` on exactly the segment-3 groups it compares (groups 0–3) |

Every component is `class X : public cs::Component<X>` with a static `compute_forward`; nothing is passed to a kernel except through
its ports. The grid (nx, ny, tx, ty) is one i32 input `geo` of extent 4 on the components that need it; the system ties those
inputs into one alias and the driver sets it with `update_input`. The fused and unfused versions of a stage are different
component types (`ConvDivergence<true>` waits on the tile-pipeline groups, `ConvDivergence<false>` does not), as are the two directions.

The unfused variant has the same stages as separate kernels (`PadBoundary` merges load and boundary; `ToPrimitiveG`,
`GradientsG`, `ReconstructG`, `ConvFluxG<Dir>`, `ViscFluxG<Dir>` work on padded global arrays) and a small `InitDiagnostics` kernel of its own.
The handwritten kernel takes a ticket per work-group (so ticket order is segment order and every wait targets a lower ticket),
initializes the atomics in the group holding ticket 0 (the diagnostics wait on its flag), waits on the same neighbour tickets, has
its own barrier (wait until every earlier group is done) before the deterministic reductions, and ends with the error group. Its
groups are laid along the last dimension.

`ConvFlux`, `ViscFlux`, `ApplyBoundary`, `Reconstruct` and the stage bodies call small functions in `tile.hpp`; the hand-written
kernel calls the same functions, so the three variants run one arithmetic and differ in scheduling and memory traffic only.
Those functions take the physical `sycl::group<3>` (`item.get_group().to_sycl()`) because the hand-written kernel is raw SYCL.

## Checks (`--check`)

Against the host double-precision reference, for every variant and strategy: R relative L2 ≤ 1e-4; the deterministic diagnostics
within 1e-4 relative; atomic min and max **equal** to the deterministic ones; fully periodic uniform flow gives R **exactly 0**.
The error stage's values and the gap between the atomic and deterministic norms are **reported only**. See the results below.

## Results

Measured by `examples/run_benchmarks.nu --only ns-residual` (`pixi run benchmarks` runs both benchmarks), copied from `build/benchmarks/results.md`.

GPU: NVIDIA GeForce RTX 4090 (24564 MiB), driver KMD 610.74, CUDA UMD 13.3 (nvidia-smi 610.53, WSL2). 2026-10-11. `ACPP_ADAPTIVITY_LEVEL` unset.

## ns-residual, one process per variant and strategy, walls BC, minmod

GB/s = nx * ny * 32 B / time, the nominal traffic: 4 state fields read + 4 R fields written per cell, f32, that is nx*ny*8*4 B, not measured.

| n, nx = ny | variant | ms/eval median | min–max | GB/s nominal | R rel L2 vs reference | atomic-vs-deterministic norm gap | check |
|---:|---|---:|---|---:|---:|---:|---|
| 1024 | unfused | 3.260 | 3.131–3.561 | 10.29 | 9.02e-06 | 3.88e-05 | OK |
| 1024 | fused-flatten | 1.532 | 1.423–1.597 | 21.90 | 9.02e-06 | 3.85e-05 | OK |
| 1024 | fused-pad | 1.551 | 1.531–1.639 | 21.64 | 9.02e-06 | 3.86e-05 | OK |
| 1024 | fused-padded | 1.528 | 1.512–1.596 | 21.96 | 9.02e-06 | 3.84e-05 | OK |
| 1024 | handwritten | 0.268 | 0.267–0.283 | 125.2 | 9.02e-06 | 3.85e-05 | OK |
| 2048 | unfused | 3.263 | 3.117–3.716 | 41.14 | 1.02e-05 | 1.43e-04 | OK |
| 2048 | fused-flatten | 1.842 | 1.765–1.881 | 72.86 | 1.02e-05 | 1.44e-04 | OK |
| 2048 | fused-pad | 1.848 | 1.835–1.893 | 72.61 | 1.02e-05 | 1.44e-04 | OK |
| 2048 | fused-padded | 1.934 | 1.876–1.965 | 69.40 | 1.02e-05 | 1.44e-04 | OK |
| 2048 | handwritten | 0.945 | 0.941–0.946 | 142.0 | 1.02e-05 | 1.44e-04 | OK |
| 4096 | unfused | 9.685 | 9.617–9.726 | 55.43 | 1.11e-05 | 6.61e-04 | OK |
| 4096 | fused-flatten | 5.810 | 5.789–5.939 | 92.40 | 1.11e-05 | 6.61e-04 | OK |
| 4096 | fused-pad | 6.062 | 5.715–6.145 | 88.57 | 1.11e-05 | 6.61e-04 | OK |
| 4096 | fused-padded | 5.914 | 5.795–6.047 | 90.78 | 1.11e-05 | 6.61e-04 | OK |
| 4096 | handwritten | 3.550 | 3.547–3.742 | 151.2 | 1.11e-05 | 6.61e-04 | OK |

- n = 1024, unfused: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 1024, fused-flatten: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 1024, fused-pad: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 1024, fused-padded: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 1024, handwritten: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 2048, unfused: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 2048, fused-flatten: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 2048, fused-pad: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 2048, fused-padded: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 2048, handwritten: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 4096, unfused: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 4096, fused-flatten: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 4096, fused-pad: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 4096, fused-padded: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0
- n = 4096, handwritten: warm-up runs needed: 1, converged; timed runs that still printed the JIT warning: 0

Notes (facts, not analysis):

- The fused variant is 1.6 to 5.8 times the handwritten kernel's time (1024: 5.7 to 5.8x, 2048: 1.95 to 2.05x, 4096: 1.6 to 1.7x). Why is NOT investigated yet.
- The three `cs::range_split` strategies are within about 5 percent of each other at every size (1024: 1.53 to 1.55 ms); no strategy is consistently fastest.
- The unfused time barely changes between 1024 and 2048 (3.26 and 3.26 ms), while the fused and handwritten times grow; the cause is not investigated.
- The atomic-vs-deterministic norm gap grows with the number of tiles (4096 tiles at 1024, 65536 at 4096): the atomic sum is a float running sum of one `fetch_add` per tile. It is reported, not checked; the deterministic sums are within 1e-6 of the double reference.
- Atomic min and max equal the deterministic ones in every run, and uniform periodic flow gives R exactly 0 (`--check`, every variant and strategy, OpenMP and CUDA).
- GB/s is the nominal traffic, not measured: the fluxes and partials the kernels also move are not counted.

