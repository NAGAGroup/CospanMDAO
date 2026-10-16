# gemver-chain: GEMVER -> BiCGK -> GEMVER

A memory-bound demo of what CoSpan's fusion flavors buy, with a cuBLAS reference.
Everything is row-major, `alpha = beta = 1/sqrt(n)`.

```
GEMVER(A, u1, v1, u2, v2, y, z):  B = A + u1 v1^T + u2 v2^T
                                  x = beta * B^T y + z
                                  w = alpha * B x
G1     A = A0 (hash), u1..z hash vectors          -> B1, x1, w1
BiCGK  on B1:  q = B1 p (p = x1),  s = B1^T r (r = w1)
G2     A = B1, same u1..v2, y = q, z = s          -> B2, x2, w2
```

The inputs are a deterministic integer hash of (seed, index) in [-1, 1]
(`common/chain_data.hpp`, plain C++, shared by both programs, with a host
double-precision reference and a relative-L2 helper). `n` must be a multiple of 256
(the work-group size) and of `R` (rows per strip). `n` and `R` are the constructor arguments of the root system only: the
components read their sizes from the shapes of their ports, and the systems give the free inputs and the partial-sum outputs
their shapes with shape hooks (`systems.hpp`).

## Components (`components.hpp`)

Every kernel is rank 1, work-group 256, a group-range body, grid-stride. Each component states its work-group shape and rounds its range up to whole work-groups; the body guards the tail. Matrices are
indexed only by `sycl::id<2>{i, j}`.

| component | work | notes |
|---|---|---|
| `Rank2Update` | `(n/R)*n` items, `t -> (c = t/n, j = t%n)` | `B[cR+k][j] = A + u1 v1 + u2 v2`, `k < R` |
| `GemvTPartial` | same mapping | `P[c][j] = sum_k B[cR+k][j] y[cR+k]`; a work-item reads the B it wrote in `Rank2Update` |
| `ColReduce<Z>` | `n` items | `x[j] = (Z ? z[j] : 0) + scale * sum_c P[c][j]`; `Z = false` has no `z` input |
| `Gemv` | a group per row, grid-stride over rows | `reduce_over_group`; works under the generic JIT |
| `BicgkOnePass`, `RowReduce` | a group per (strip, 256-column block); `n` items | q and s from one pass over `B`, and the row reducer of its q |
| `Rank2UpdateLocal`, `GemvTPartialLocal`, `BicgkTileLocal`, `Rank2UpdateFromLocal` | one group per 32 x 256 tile of `B` | the `crafted` components: the tile lives in local memory between stages (below) |

## Variants (`systems.hpp`)

`Gemver` and `Bicgk` are reusable Systems with promoted ports; the root of every
variant promotes the same names, so the driver is variant-agnostic.

* **unfused**: every component its own kernel: 4 kernels per GEMVER, 3 for BiCGK (11 launches).
* **parallel**: as unfused, but BiCGK's `gemv_q` and `gtp_s` are a **`system_from` group** (still an ordinary System, not a fused group): two launches, independent stages, concurrency comes from data dependencies alone. (11 launches.)
* **fused**: GEMVER = `fuse_components` of ONE segment holding `r2u` and `gtp` with `range_mode::whole`, then `cred`, then `gemv`;
  BiCGK = `fuse_components` of `gemv_q` and `gtp_s`, each its own segment with no boundary (own slices of groups, one launch), then `cred_s`. (8 launches.)
* **onepass**: GEMVER as in fused, BiCGK in ONE pass over `B`: a group-range body over (strip, 256-column
  block) that accumulates `Ps[c][j] = sum_k r[i] B[i][j]` and, per row, `reduce_over_group(B[i][j] p[j])`
  into `Pq[i][block]`, then `RowReduce` (q) and the scale-1 `ColReduce` (s). It reads `B` once for BiCGK
  instead of twice, but sums in a different order, so it is not bitwise equal to the others
  (relative L2 about 3e-6 against unfused). `--variant all` runs it on every backend.
* **naive**: the SAME components, fused with global barriers into **three launches for the whole chain** (the
  barrier items make every group of the next segment wait for every group of the earlier ones, in ticket order, so no
  co-residency is needed):
  `G1 = fuse_components(segment<whole>(r2u, gtp), global_barrier(), cred, global_barrier(), gemv)`,
  `BiCGK = fuse_components(onepass, global_barrier(), segment(rowred, cred_s))`, `G2` as `G1`.
* **crafted**: components written to be fused, three launches (see below). Passes `--check` on omp as well.

### What `crafted` does differently

The non-tile components each read `B` from global memory: the rank-2 update writes it, the partial column sums read it back,
and BiCGK reads it again (and G2's rank-2 update reads B1 a third time). `crafted` gives every work-group a **tile of
32 rows by 256 columns of `B` (32 KB of local memory)** and keeps it there between the stages of one segment
(`local_mem::shared`: every stage of the segment sees the same block). The tile is declared as a `cs::LocalMem`, sized with `cs::local_desc`, and read with `auto [tile] = lm.get(it)`:

* **G1 launch**: `Rank2UpdateLocal` computes its tile of `B1`, writes it to global `B1` (an output) and leaves it in local
  memory; `GemvTPartialLocal` (after a group barrier) reads the tile from local memory for the partial column sums, so
  `B1` is not read back. Then `global_barrier`, `cred`, `global_barrier`, `gemv` (the existing components).
* **Middle launch** (BiCGK and G2's rank-2 update share ONE read of `B1`): `BicgkTileLocal` loads the `B1` tile into local
  memory and produces the `q` and `s` partials from it, as `BicgkOnePass` does; `Rank2UpdateFromLocal` (after a group
  barrier) writes `B2 = tile + u1b v1b^T + u2b v2b^T` from local memory. G2's update does not depend on `q` or `s`, which is
  why it can share the read. Then `global_barrier`, then `rowred` and `cred_s`.
* **G2 launch**: `gtp` reads `B2` from global memory (it needs `y = q`, which is not known when `B2` is made, so it cannot
  come from local memory), then `global_barrier`, `cred`, `global_barrier`, `gemv`.

The tile stages process exactly one tile per work-group (one group per tile, no grid-stride loop): stages run one after
the other inside a work-group, so a stage that looped over several tiles would leave only the last one in local memory.
The reducers use `n / 32` strips (not `--R`). All new components expose the same promoted outputs, so the driver and
`--check` are unchanged. Every launch of a fused group reserves the largest local block of its segments, so `gemv` and
`cred` groups of the same launch also hold 32 KB each.

**The limit.** `B` stays in local memory only within a segment, between two stages of the same work-group. Keeping it there
across the `x -> w` dependency of GEMVER (`gemv` needs the whole `x`, which needs every strip's partials) would need all the
work-groups to be resident at once (co-resident groups, persistent threads); that is deferred, so `gemv` and `gtp` in G2
read `B` from global memory again.

### A fused group with segments

```cpp
// GEMVER's first two stages: r2u and gtp share their (c, j) mapping, so they sit in ONE segment where every stage
// runs for every group of the block (range_mode::whole); connects inside the group run forward in stage order.
add_subsystem(cs::fuse_components(
    "fw", cs::connects_desc({{"r2u.B", "gtp.B"}}),
    cs::segment<cs::range_mode::whole>(cs::subsystem_desc<Rank2Update>("r2u", R),
                                       cs::subsystem_desc<GemvTPartial>("gtp"))));

// BiCGK's independent stages: a bare descriptor is a one-stage segment, so each gets its own groups of one launch.
add_subsystem(cs::fuse_components("fw", cs::subsystem_desc<Gemv>("gemv_q"),
                                        cs::subsystem_desc<GemvTPartial>("gtp_s")));
```

### What can and cannot fuse

Sequential fusion inside a work-group is only safe when a stage reads what the *same
work-item* (or, with its own group barrier, the same group) wrote. That holds for
`r2u -> gtp`: same `(c, j)` mapping, same work-item, so they fuse. It holds for
nothing else in GEMVER:

* `cred` needs *every* strip's partial `P[:, j]`, and `gemv` needs the *whole* `x`
  (`w = alpha B x` reads all of `x`). `x` must be complete before `B x`: a global dependency,
  which a kernel boundary provides and a work-group cannot.
* BiCGK's `gemv_q` and `gtp_s` are independent, so they fuse as two segments (disjoint
  groups); `cred_s` again needs all of `P`.

A global dependency like these can be put inside one fused group with `cs::global_barrier()` between two segments, the
only boundary item (or, in a group built with the `cs::group_waits_enabled` property, with `GroupItem::wait_group` in
the stages).
The `naive` and `crafted` variants use them: the global dependencies of the whole chain become barriers inside three launches.

## cuBLAS reference (`cublas/`)

A standalone CUDA project, built in the `cublas` pixi environment (conda-forge CUDA
13.3, nvcc 13.3.73, gxx 14.4.0, libcublas 13.6.0.2). Row-major `B` is column-major `M = B^T`:
`sger` for the rank-1 updates, `sgemv` OP_N / OP_T for the products.

**Caveat, a real BLAS cost:** `sger` is in place, so `B = A + ...` needs a
device-to-device copy of `A` into `B` first (one extra full read and write of the matrix,
twice per chain), and each `sger` re-reads and re-writes `B`. A fused kernel does the rank-2
update in one pass. Also each BLAS call and copy is a separate launch.

## Results

Measured on 2026-10-06 on an NVIDIA GeForce RTX 4090 (128 SMs, 24 GiB; driver 610.74, CUDA 13.3 as reported by nvidia-smi;
cuBLAS from conda-forge CUDA 13.3, libcublas 13.6), AdaptiveCpp with its default `ACPP_ADAPTIVITY_LEVEL` and the same
compile flags for every CoSpan variant. Protocol (`examples/run_benchmarks.nu`, `pixi run benchmarks`): **each variant in its
own process**, JIT warm-up runs until AdaptiveCpp stops printing its "new binaries being JIT-compiled" warning (one run was
enough everywhere, the kernel cache being warm), then **5 timed runs: the median, with the min-max spread**; each run times 20
chains after 3 warm-up chains, `--R 64`. GB/s uses the **nominal** traffic of 8 n^2 * 4 B per chain (not measured). The
relative L2 error is the worst of x1 w1 q s x2 w2 (and B2 for n <= 4096) against the host double-precision reference, from
`--check`. `naive` and `crafted` are not bitwise equal to the others (other summation orders).

| n | variant | ms / chain | min-max | GB/s (nominal) | worst rel L2 |
|---:|---|---:|---|---:|---:|
| 2048 | naive | 1.169 | 1.133-1.183 | 114.9 | 1.12e-7 |
| 2048 | crafted | 1.132 | 1.122-1.205 | 118.6 | 1.24e-7 |
| 2048 | cuBLAS | 0.155 | 0.148-0.166 | 866.3 | 1.26e-7 |
| 4096 | naive | 1.339 | 1.328-1.441 | 400.9 | 1.38e-7 |
| 4096 | crafted | 1.238 | 1.166-1.546 | 433.5 | 1.7e-7 |
| 4096 | cuBLAS | 0.789 | 0.787-0.812 | 680.1 | 1.24e-7 |
| 8192 | naive | 2.230 | 2.229-2.231 | 962.9 | 1.69e-7 |
| 8192 | crafted | 2.229 | 2.222-2.734 | 963.4 | 2.4e-7 |
| 8192 | cuBLAS | 5.102 | 5.089-5.178 | 420.9 | 1.85e-7 |
| 16384 | naive | 8.531 | 8.469-8.589 | 1007.0 | 2.38e-7 |
| 16384 | crafted | 8.359 | 8.296-10.25 | 1027.7 | 3.4e-7 |
| 16384 | cuBLAS | 20.68 | 20.67-20.75 | 415.4 | 2.33e-7 |

The other variants, the same way (one process each):

| n | variant | ms / chain | min-max | GB/s (nominal) |
|---:|---|---:|---|---:|
| 2048 | unfused | 1.925 | 1.861-2.007 | 69.7 |
| 2048 | parallel | 1.970 | 1.899-2.146 | 68.1 |
| 2048 | fused | 1.582 | 1.560-1.699 | 84.8 |
| 2048 | onepass | 1.707 | 1.690-1.733 | 78.6 |
| 4096 | unfused | 1.948 | 1.895-2.121 | 275.6 |
| 4096 | parallel | 1.882 | 1.863-2.267 | 285.2 |
| 4096 | fused | 1.661 | 1.637-1.812 | 323.2 |
| 4096 | onepass | 1.742 | 1.689-1.935 | 308.1 |
| 8192 | unfused | 2.994 | 2.976-3.036 | 717.2 |
| 8192 | parallel | 2.984 | 2.979-3.108 | 719.7 |
| 8192 | fused | 2.447 | 2.436-2.464 | 877.5 |
| 8192 | onepass | 2.368 | 2.354-2.421 | 906.9 |
| 16384 | unfused | 11.41 | 11.33-11.81 | 753.1 |
| 16384 | parallel | 11.43 | 11.27-12.08 | 751.4 |
| 16384 | fused | 9.150 | 8.987-9.372 | 938.8 |
| 16384 | onepass | 8.655 | 8.543-9.040 | 992.5 |

Notes, honestly:

* **At n = 16384 (a 1 GiB matrix) the three-launch variants and onepass are all near 8.5 ms (about 1000 GB/s of nominal
  traffic, against a 1008 GB/s card); `crafted` is 2% ahead of `naive` and 3%
  ahead of `onepass`.** The local-memory tiles save `B` traffic that the nominal figure does not count, so they show up
  only as a small difference here; I did not profile why they do not show up as more.
* **cuBLAS at n = 2048 runs out of the L2 cache**: the matrix is 16 MB (2048 * 2048 * 4 B) and the 4090 has
  72 MB of L2, so its 0.155 ms is not a memory-bound figure, and at n = 4096 (64 MB) it is still mostly cached. At n = 16384
  the data comes from DRAM and cuBLAS takes 20.7 ms (it copies `A` into `B` before each in-place `sger`, see above). The CoSpan
  variants at n <= 4096 take about 1.2 ms whatever n is, which is not bandwidth; launch and barrier costs are the suspect, not profiled.
* **Running all the variants in one process (`--variant all`) slows the fused ones down and makes them bimodal; the cause
  was not investigated.** At n = 16384 `crafted` and `naive` measured 13-14 ms (min-max 12.97-16.19 ms) as the last variants
  of an `all` run against 8.9-9.0 ms alone, and `fused` and `parallel` spread 10.1-14.5 ms and 16.2-20.4 ms. That is why the
  tables above use one process per variant. `--variant all` is kept for `--check` and for a quick look.
* A `--check` run is the correctness record: every variant, one process, relative L2 against the host reference below 1e-4
  at n = 2048, 4096, 8192 and 16384 (the table above lists the worst value per variant).
* On omp, `crafted` passes `--check` at n = 2048 and 4096 (worst relative L2 about 1.7e-7); `onepass` and `naive` pass it too.

## Running

```
pixi run -e cuda gemver-chain --check --n 2048            # every variant vs the host reference
pixi run -e cuda gemver-chain --n 16384 --variant all
pixi run -e cuda gemver-chain --n 2048 --dump /tmp/gc       # raw f32 of x1 w1 q s x2 w2 per variant

pixi run -e cublas build-cublas
pixi run -e cublas gemver-chain-cublas --check --n 2048
pixi run -e cublas gemver-chain-cublas --n 16384 --compare /tmp/gc   # rel L2 against the CoSpan dumps
```

Flags: `--n --R --iters --warmup --variant all|unfused|parallel|fused|onepass|naive|crafted --check --dump`.
`--check` compares every output with the host double reference (relative L2 <= 1e-4; B2 too
when n <= 4096) and the variants with each other (bitwise for unfused, parallel and fused; within the tolerance for onepass, naive and crafted). To pick the GPU on a machine that also
has the omp device, set `ACPP_VISIBILITY_MASK=cuda` (the default selector already picks it here).

## Open questions — to explore further

**These results need deeper investigation before they are treated as conclusions.** The numbers
above are single-machine, few-run measurements, several effects are unexplained, and nothing here
was profiled. The onepass row is provisional: it was measured in two full-size runs only
(11.12 and 10.95 ms, two values per cell above), outside the run that produced the other rows.

a. **Fused is bimodal across runs** (15.05, 15.08, 14.91, 11.82, 11.66 ms) while the other variants
   are stable. Hypothesis to check: AdaptiveCpp's runtime adaptivity keeps re-optimizing kernels
   across application runs (`ACPP_ADAPTIVITY_LEVEL`, the on-disk kernel cache), so it is not a
   one-time JIT. Try fixed adaptivity levels and repeated runs.
b. **Compiler flags.** Check whether AdaptiveCpp enables fast-math or similar by default, and how
   that affects accuracy and timing for every variant and for the cuBLAS comparison.
c. **Small n.** At n = 2048 the old fused variant was much slower than unfused (about 1.7 ms vs
   0.25 ms). The packing buffers suspected then are gone; whether the new launch path (argument and
   geometry arrays, one kernel) still loses at small n is not measured.
d. **parallel brings no gain over unfused.** Both products still read the matrix, and nothing was
   profiled to see whether the two kernels actually overlap.
