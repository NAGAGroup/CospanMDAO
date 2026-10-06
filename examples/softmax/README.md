# softmax: row-wise softmax, fused in local memory

```
y[i][j] = exp(x[i][j] - max_i) / sum_j exp(x[i][j] - max_i)        x, y: n rows x m columns of f32
```

A memory-bound kernel with a row-wise dependency (the max, then the sum, then the division): the case where a fused
group keeps a whole row in **local memory** and reads `x` and writes `y` once. Work-group size 256, one work-group per row.
The input is generated on the device from a deterministic integer hash (`input_value(k)` in `main.cpp`, values in [-4, 4)),
and the host reference recomputes the same floats in double precision.

## The three variants

* **unfused** (`components.hpp`): three global-memory components, three launches, five passes over the matrix.
  `RowMax` reads `x` and writes `max[n]`; `ExpSum` reads `x` and `max`, writes `e = exp(x - max)` (`n x m`) and `sum[n]`;
  `Normalize` reads `e` and `sum` and writes `y`.
* **fused**: ONE `fuse_components` launch with a single `segment<local_mem::shared>` of three new stages written for local
  memory: `LoadMax` loads the row into local memory and reduces its max; `ExpSumLocal` exponentiates the row in local
  memory and reduces the sum; `NormalizeWrite` divides the row by the sum and writes `y`. The row (`m` floats) and its two
  scalars share one local block, the same for every stage of a `shared` segment; the stages group-barrier before they
  read what other work-items of the group wrote (the scalars). Two passes over the matrix: the ideal traffic. The row must
  fit in 48 KB (`m <= 12286`). The fused launch has exactly one work-group per row, because the stages of a fused group
  run one after the other inside a work-group, so a stage that looped over several rows would leave only the last one in
  local memory. (`LoadMax` and `ExpSumLocal` also write their scalar to a `[n]` output, 1/m of the matrix traffic, so that
  every stage is an ordinary component with a port.)
* **handwritten** (`main.cpp`): one plain SYCL kernel outside CoSpan, on its own queue and its own USM buffers, doing the
  same algorithm with the same work-group size and local memory, on the same device (a grid-stride loop over rows,
  launched, like the fused stages, with one group per row).

The driver also times a **device-to-device copy of the same bytes** (`q.memcpy`, and a float4 copy kernel; the faster of
the two medians is the practical peak) and reports every variant as a percentage of it.

## Results

Measured on 2026-10-06 on an NVIDIA GeForce RTX 4090 (24 GiB; driver 610.74, CUDA 13.3 as reported by nvidia-smi),
AdaptiveCpp with its default `ACPP_ADAPTIVITY_LEVEL`. Protocol (`examples/run_benchmarks.nu`, `pixi run benchmarks`): **each
variant in its own process**, JIT warm-up runs until AdaptiveCpp stops printing its "new binaries being JIT-compiled" warning
(one run was enough everywhere, the kernel cache being warm), then **5 timed runs: the median, with the min-max spread**; each
run times 20 softmaxes after 3 warm-up ones. `n * m * 4 B` = 1 GiB for every `m`. GB/s uses the **ideal** traffic of
2 n m 4 B = 2.147 GB (read `x`, write `y`), not measured traffic. "% of copy" is against the copy peak (2.382, 2.388 and
2.387 ms for `m` = 1024, 4096, 8192). Max absolute error against the host double reference (tolerance 1e-5) is in the last
column.

| m | n | variant | ms | min-max | GB/s (ideal) | % of copy | time / handwritten | max abs error |
|---:|---:|---|---:|---|---:|---:|---:|---:|
| 1024 | 262144 | unfused | 5.945 | 5.905-5.976 | 361.2 | 40.1% | 2.473 | 2.066e-9 |
| 1024 | 262144 | fused | 2.562 | 2.549-2.587 | 838.1 | 93.0% | 1.066 | 2.066e-9 |
| 1024 | 262144 | handwritten | 2.404 | 2.377-2.417 | 893.2 | 99.1% | 1.000 | 2.066e-9 |
| 4096 | 65536 | unfused | 5.991 | 5.970-6.015 | 358.5 | 39.9% | 2.464 | 4.868e-10 |
| 4096 | 65536 | fused | 2.567 | 2.534-2.602 | 836.5 | 93.0% | 1.056 | 4.868e-10 |
| 4096 | 65536 | handwritten | 2.431 | 2.409-2.434 | 883.4 | 98.2% | 1.000 | 4.868e-10 |
| 8192 | 32768 | unfused | 6.023 | 6.007-6.056 | 356.5 | 39.6% | 2.475 | 2.415e-10 |
| 8192 | 32768 | fused | 2.547 | 2.534-2.555 | 843.1 | 93.7% | 1.047 | 2.415e-10 |
| 8192 | 32768 | handwritten | 2.434 | 2.420-2.447 | 882.4 | 98.1% | 1.000 | 2.415e-10 |

The fused / handwritten time ratio is 1.066, 1.056 and 1.047 for `m` = 1024, 4096 and 8192: CoSpan's fused launch is
within 5-7% of the hand-written kernel, which itself is within 2% of a plain copy. The unfused chain takes about 2.5x the
time of the handwritten kernel (five passes against two). The extra 5-7% of the fused launch was not profiled; two
candidates are the group launcher's dispatch through the JIT-ed stage sequence and the two `[n]` scalar outputs.

## Running

```
pixi run -e cuda softmax --n 65536 --m 4096 --check          # every variant vs the host reference (max abs error <= 1e-5)
pixi run -e cuda softmax --n 65536 --m 4096 --variant fused
pixi run benchmarks                                           # the whole protocol, both benchmarks
```

Flags: `--n --m --iters --warmup --variant all|unfused|fused|handwritten --check --tol`.
