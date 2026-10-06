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
(the work-group size) and of `R` (rows per strip).

## Components (`components.hpp`)

Every kernel is rank 1, work-group 256, a group body, grid-stride. Matrices are
indexed only by `sycl::id<2>{i, j}`.

| component | work | notes |
|---|---|---|
| `Rank2Update` | `(n/R)*n` items, `t -> (c = t/n, j = t%n)` | `B[cR+k][j] = A + u1 v1 + u2 v2`, `k < R` |
| `GemvTPartial` | same mapping | `P[c][j] = sum_k B[cR+k][j] y[cR+k]`; a work-item reads the B it wrote in `Rank2Update` |
| `ColReduce<Z>` | `n` items | `x[j] = (Z ? z[j] : 0) + scale * sum_c P[c][j]`; `Z = false` has no `z` input |
| `Gemv` | a group per row, grid-stride over rows | `reduce_over_group`; works under the generic JIT |
| `BicgkOnePass`, `RowReduce` | a group per (strip, 256-column block); `n` items | q and s from one pass over `B`, and the row reducer of its q |

## Variants (`systems.hpp`)

`Gemver` and `Bicgk` are reusable Systems with promoted ports; the root of every
variant promotes the same names, so the driver is variant-agnostic.

* **unfused**: every component its own kernel: 4 kernels per GEMVER, 3 for BiCGK (11 launches).
* **parallel**: as unfused, but BiCGK's `gemv_q` and `gtp_s` are a **`system_from` group** (still a plain System, not a fused group): two launches, independent stages, concurrency comes from data dependencies alone. (11 launches.)
* **fused**: GEMVER = `fuse_components` of ONE segment holding `r2u` and `gtp` with `range_mode::whole`, then `cred`, then `gemv`;
  BiCGK = `fuse_components` of `gemv_q` and `gtp_s`, each its own segment with no boundary (own slices of groups, one launch), then `cred_s`. (8 launches.)
* **onepass**: GEMVER as in fused, BiCGK in ONE pass over `B`: a group body over (strip, 256-column
  block) that accumulates `Ps[c][j] = sum_k r[i] B[i][j]` and, per row, `reduce_over_group(B[i][j] p[j])`
  into `Pq[i][block]`, then `RowReduce` (q) and the scale-1 `ColReduce` (s). It reads `B` once for BiCGK
  instead of twice, but sums in a different order, so it is not bitwise equal to the others
  (relative L2 about 3e-6 against unfused). **Skipped by `--variant all` on omp**, where its `q` is
  wrong (see the open questions); an explicit `--variant onepass` still runs it there.

### A fused group with segments

```cpp
// GEMVER's first two stages: r2u and gtp share their (c, j) mapping, so they sit in ONE segment where every stage
// runs for every group of the block (range_mode::whole); connects inside the group run forward in stage order.
add_subsystem(cs::fuse_components(
    "fw", cs::connects_desc({{"r2u.B", "gtp.B"}}),
    cs::segment<cs::range_mode::whole>(cs::subsystem_desc<Rank2Update>("r2u", n, R),
                                       cs::subsystem_desc<GemvTPartial>("gtp", n, R))));

// BiCGK's independent stages: a bare descriptor is a one-stage segment, so each gets its own groups of one launch.
add_subsystem(cs::fuse_components("fw", cs::subsystem_desc<Gemv>("gemv_q", n, 1.0f, true),
                                        cs::subsystem_desc<GemvTPartial>("gtp_s", n, R, true)));
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

A global dependency like these can be put inside one fused group with a boundary item
(`cs::global_barrier()` between two segments, or `cs::ticket_order()` plus `GroupItem::wait_group`).
This example does not use them: the chain keeps its global dependencies at kernel boundaries.

## cuBLAS reference (`cublas/`)

A standalone CUDA project, built in the `cublas` pixi environment (conda-forge CUDA
13.3, nvcc 13.3.73, gxx 14.4.0, libcublas 13.6.0.2). Row-major `B` is column-major `M = B^T`:
`sger` for the rank-1 updates, `sgemv` OP_N / OP_T for the products.

**Caveat, a real BLAS cost:** `sger` is in place, so `B = A + ...` needs a
device-to-device copy of `A` into `B` first (one extra full read and write of the matrix,
twice per chain), and each `sger` re-reads and re-writes `B`. A fused kernel does the rank-2
update in one pass. Also each BLAS call and copy is a separate launch.

## Results

**These numbers were measured before the fused-group rework (P7) and were NOT re-measured for P8**: the fused launch path
has changed since. RTX 4090 (128 SMs), n = 16384, R = 64, 20 iterations after 3 warm-up, one wait after the batch
(CoSpan) / CUDA events (cuBLAS). Second run of each program (the JIT cache is warm).
GB/s uses a **nominal** 8 * n^2 * 4 B = 8.59 GB per chain, not measured traffic
(the unfused chain really moves about 11 matrix passes).

| variant | ms / chain | speedup vs unfused | GB/s (nominal) |
|---|---:|---:|---:|
| unfused | 15.99 | 1.00x | 537 |
| parallel | 16.19 | 0.99x | 531 |
| fused | 11.66 | 1.37x | 737 |
| onepass (provisional) | 11.12 / 10.95 | 1.44x / 1.47x | 772 / 784 |
| cuBLAS | 21.07 | 0.76x | 408 |

Notes, honestly:

* **Noise in `fused`.** Across five full-size runs it measured 15.05, 15.08, 14.91, 11.82 and 11.66 ms
  (1.06x to 1.37x); the other variants moved by under 2%. I did not find why. Read fused as
  "1.06x to 1.37x", not 1.37x.
* `parallel` is a wash: the two kernels are already large enough to fill the GPU.
* **Small n is different.** At n = 2048 (`--check` runs, 5 iterations) unfused took about 0.25 ms,
  parallel 0.32 to 0.56 ms and fused about 1.7 ms: fusion lost badly when the work was tiny. The suspect was
  the per-forward host-side overhead of the old fused wave (packing and uploading its stage records, now
  gone); not re-measured.
* cuBLAS is the slowest, largely because of the copies and the in-place `sger` passes above.
* The unfused, parallel and fused variants are bitwise equal at n = 2048 and n = 16384, and agree with
  cuBLAS to a relative L2 of 3e-7 (x1, s, x2) to 2.7e-6 (w1, q, w2). Against the host double
  reference the errors are the same size (see `--check`).

## Running

```
pixi run -e cuda gemver-chain --check --n 2048            # every variant vs the host reference
pixi run -e cuda gemver-chain --n 16384 --variant all
pixi run -e cuda gemver-chain --n 2048 --dump /tmp/gc       # raw f32 of x1 w1 q s x2 w2 per variant

pixi run -e cublas build-cublas
pixi run -e cublas gemver-chain-cublas --check --n 2048
pixi run -e cublas gemver-chain-cublas --n 16384 --compare /tmp/gc   # rel L2 against the CoSpan dumps
```

Flags: `--n --R --iters --warmup --variant all|unfused|parallel|fused|onepass --check --dump`.
`--check` compares every output with the host double reference (relative L2 <= 1e-4; B2 too
when n <= 4096) and the variants with each other (bitwise). To pick the GPU on a machine that also
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
d. **onepass gives a wrong `q` on the omp backend** (`q[0]` right, the rest garbage). `s` and every
   other variant are correct. Cause unknown: possibly the backend, possibly `reduce_over_group`
   inside the per-row loop of a group body. Not confirmed.
e. **parallel brings no gain over unfused.** Both products still read the matrix, and nothing was
   profiled to see whether the two kernels actually overlap.
