# Research: generating CoSpan's backward kernels from the JIT IR (2026-10-16)

**Decision (Jack, 2026-10-16):** start from Enzyme, built against our LLVM 23 and run inside the AdaptiveCpp JIT on stage-2 IR, then extend it for CoSpan's needs. Don't write our own forward-mode-plus-transpose pass first. Much of the parallel-backward work (atomic accumulation, parallel systems) was already solved by hand in the pre-CoSpan mycelium work.

## Where it would hook in
At `JIT_OPTIMIZATIONS` the dynamic-function call sequence is concrete, so a fused driver's stage functions are direct calls in one inlined, optimized SPMD kernel. Differentiating there gives, from IR rather than source, either a backward kernel per stage function (fused in reverse stage order by the same dynamic-function machinery) or the backward of the whole fused kernel at once. Either way, nobody writes or fuses `compute_backward` by hand. AD after optimization matters: in the Enzyme papers it is ~4.2x faster on CPU, and on GPU reverse mode without prior optimization was intractable (LULESH 2979x overhead vs 2.4x optimized).

## The two approaches considered
| | Enzyme (LLVM plugin) | Own pass: forward mode, then transpose |
|---|---|---|
| What | Mature reverse/forward AD on LLVM IR; type, activity analysis, caching | JVP pass, then reverse via linearize/unzip/transpose (JAX/Dex; Radul et al., *You Only Linearize Once*, POPL 2023) |
| GPU kernels | Yes (SC'21): barriers, shared memory, atomics, register pressure | We'd build it |
| Integration | Build against LLVM 23, load in the JIT pipeline, declare `__acpp_sscp_*` builtins (barriers, group functions) to it | All in the fork |
| Risk | LLVM-version lag; SYCL builtins unknown until taught | More work |

## Why reverse mode of SPMD kernels is hard (and known)
- Races flip: a benign read race forward becomes a write race backward (adjoint accumulation). Fallback is atomics; cheaper when thread-local or turned into a group reduction.
- Barriers mirror: a race-free forward stays race-free with barriers at mirrored points.
- Group functions have dual adjoints: reduce_over_group <-> broadcast.
- The tape is the cost: cache per work-item (memory) or recompute (time); Enzyme recomputes from shared memory to save registers.
- CoSpan-specific: cross-group ordering (`wait_earlier_segments`, global barriers) runs in reverse segment order backward.

## Complex numbers
Forward mode with a complex tangent is always defined; non-holomorphic functions need the Wirtinger pair. Derivatives are generated code, not complex step, so forward kernels stay free to use complex types.

## First experiment (when there's time)
1. Get Enzyme building against our LLVM 23 and loadable in the stage-2 JIT pipeline.
2. Differentiate an index-range component (no barriers, no local memory), triggered via a dynamic-function slot kind ("the derivative of stage F"); validate against central differences and complex step on gemver components.
3. Then one group-range component (e.g. ColReduce), teaching Enzyme our barrier and group-function builtins.

## Open questions
- Packaging Enzyme in the toolchain against LLVM 23.
- How much JIT knowledge (uniformity, work-group shape) can feed Enzyme's analyses.
- System level: OpenMDAO-style linear solves and propagation stay on the host, with component VJPs/JVPs as the per-component pieces.

## References
- Moses et al., Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via Enzyme, SC'21. https://doi.org/10.1145/3458817.3476165
- Radul et al., You Only Linearize Once: Tangents Transpose to Gradients, POPL 2023. https://doi.org/10.1145/3571236
- Enzyme: https://enzyme.mit.edu/
