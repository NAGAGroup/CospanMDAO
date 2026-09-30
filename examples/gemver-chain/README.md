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
| `GridBarrier` | `G` groups | atomic counter that is never reset (crossing k waits for `k*G`), release/acquire fences, a spin limit and a `timeout` flag |

Each component takes an optional `resident_items`: when set, the launch range is exactly that.

## Variants (`systems.hpp`)

`Gemver` and `Bicgk` are reusable Systems with promoted ports; the root of every
variant promotes the same names, so the driver is variant-agnostic.

* **unfused**: every component its own kernel: 4 kernels per GEMVER, 3 for BiCGK (11 launches).
* **parallel**: as unfused, but BiCGK's `gemv_q` and `gtp_s` are a `ParallelSystem`.
* **fused**: GEMVER = `FusedSystem(sequential, wg 256) [r2u, gtp]`, then `cred`, then `gemv`;
  BiCGK = `FusedSystem(parallel) [gemv_q, gtp_s]`, then `cred_s`. (8 launches.)
* **barrier**: the whole chain as ONE sequential `FusedSystem` (one launch), 7 software grid
  barriers between the dependent stages, every stage on `G = SMs * blocks-per-cu` resident groups.
  A group that waits `--spin-limit` polls (default 2^24) gives up and sets a timeout flag; the
  driver reports "grid not co-resident at G=..." and fails the run.

### What can and cannot fuse

Sequential fusion inside a work-group is only safe when a stage reads what the *same
work-item* (or, with its own group barrier, the same group) wrote. That holds for
`r2u -> gtp`: same `(c, j)` mapping, same work-item, so they fuse. It holds for
nothing else in GEMVER:

* `cred` needs *every* strip's partial `P[:, j]`, and `gemv` needs the *whole* `x`
  (`w = alpha B x` reads all of `x`). `x` must be complete before `B x`: a global dependency,
  which a kernel boundary provides and a work-group cannot.
* BiCGK's `gemv_q` and `gtp_s` are independent, so they fuse in parallel mode (disjoint
  groups); `cred_s` again needs all of `P`.

The barrier variant forces those global dependencies into one kernel with a software
grid barrier. That is only correct when all groups are co-resident, which the
framework does not guarantee: the sweep below shows where it stops holding on this GPU.

## cuBLAS reference (`cublas/`)

A standalone CUDA project, built in the `cublas` pixi environment (conda-forge CUDA
13.3, nvcc 13.3.73, gxx 14.4.0, libcublas 13.6.0.2). Row-major `B` is column-major `M = B^T`:
`sger` for the rank-1 updates, `sgemv` OP_N / OP_T for the products.

**Caveat, a real BLAS cost:** `sger` is in place, so `B = A + ...` needs a
device-to-device copy of `A` into `B` first (one extra full read and write of the matrix,
twice per chain), and each `sger` re-reads and re-writes `B`. A fused kernel does the rank-2
update in one pass. Also each BLAS call and copy is a separate launch.

## Results

RTX 4090 (128 SMs), n = 16384, R = 64, 20 iterations after 3 warm-up, one wait after the batch
(CoSpan) / CUDA events (cuBLAS). Second run of each program (the JIT cache is warm).
GB/s uses a **nominal** 8 * n^2 * 4 B = 8.59 GB per chain, not measured traffic
(the unfused chain really moves about 11 matrix passes).

| variant | ms / chain | speedup vs unfused | GB/s (nominal) |
|---|---:|---:|---:|
| unfused | 15.99 | 1.00x | 537 |
| parallel | 16.19 | 0.99x | 531 |
| fused | 11.66 | 1.37x | 737 |
| barrier (G = 256) | 13.13 | 1.22x | 654 |
| cuBLAS | 21.07 | 0.76x | 408 |

Notes, honestly:

* **Noise in `fused`.** Across five full-size runs it measured 15.05, 15.08, 14.91, 11.82 and 11.66 ms
  (1.06x to 1.37x); the other variants moved by under 2%. I did not find why. Read fused as
  "1.06x to 1.37x", not 1.37x.
* `parallel` is a wash: the two kernels are already large enough to fill the GPU.
* `barrier` is steady (13.1 ms) and beats unfused by 1.22x, but it does not beat the plain
  fused variant at its best. Why it trails the best fused run is not measured
  (the 7 barriers and the smaller resident grid are the candidates).
* **Small n is different.** At n = 2048 (`--check` runs, 5 iterations) unfused took about 0.25 ms,
  parallel 0.32 to 0.56 ms, barrier 0.49 to 0.59 ms and fused about 1.7 ms: fusion loses badly
  when the work is tiny. Per-forward host-side overhead of the fused wave (staging its
  records and byte streams) is the suspect; I did not profile it.
* cuBLAS is the slowest, largely because of the copies and the in-place `sger` passes above.
* All four CoSpan variants are bitwise equal at n = 2048 and n = 16384, and agree with
  cuBLAS to a relative L2 of 3e-7 (x1, s, x2) to 2.7e-6 (w1, q, w2). Against the host double
  reference the errors are the same size (see `--check`).

### Barrier sweep (`--variant barrier --n 16384`)

| blocks per SM | G | ms / chain | result |
|---:|---:|---:|---|
| 1 | 128 | 14.91 | ok |
| 2 | 256 | 13.24 | ok |
| 3 | 384 | 13.11 | ok |
| 4 | 512 | 13.22 | ok |
| 5 | 640 | - | timeout: grid not co-resident (run took about 70 s to give up) |
| 6 | 768 | - | timeout: grid not co-resident (about 74 s) |

The chain kernel is co-resident up to 4 blocks per SM here and no more; the limit is
whatever the JIT'd kernel's register use allows, so it can change with the compiler.
More than one block per SM is a win (14.9 to 13.2 ms), more than two adds nothing.
On the omp backend the barrier variant is skipped: its groups do not run concurrently.

## Running

```
pixi run -e cuda gemver-chain --check --n 2048            # every variant vs the host reference
pixi run -e cuda gemver-chain --n 16384 --variant all
pixi run -e cuda gemver-chain --variant barrier --blocks-per-cu 3
pixi run -e cuda gemver-chain --n 2048 --dump /tmp/gc       # raw f32 of x1 w1 q s x2 w2 per variant

pixi run -e cublas build-cublas
pixi run -e cublas gemver-chain-cublas --check --n 2048
pixi run -e cublas gemver-chain-cublas --n 16384 --compare /tmp/gc   # rel L2 against the CoSpan dumps
```

Flags: `--n --R --iters --warmup --variant all|unfused|parallel|fused|barrier --blocks-per-cu --spin-limit --check --dump`.
`--check` compares every output with the host double reference (relative L2 <= 1e-4; B2 too
when n <= 4096) and the variants with each other (bitwise). To pick the GPU on a machine that also
has the omp device, set `ACPP_VISIBILITY_MASK=cuda` (the default selector already picks it here).
