# CoSpan roadmap

Messy by design. Checkboxes, notes, strike-throughs welcome. Each item points at the spec revision that will absorb it and the date of the project-knowledge notes it came from (2026-09-28 unless stated).

## Before any code (spec drafts still needed)

- [ ] Chunked systems — a system flavour with a scratch budget; nesting and sub-chunking are required; the chunk axis and the sub-chunk declaration are open. (absorbed by: r02)
- [ ] Cycles, solvers and implicit components — cycles by registration order; an implicit component is a self-connection with its own buffer; a balance helper. (absorbed by: r02)
- [ ] A "proven by" list per r01 section — the order-sensitive tests each contract needs. (absorbed by: r02)

## Build order (forward-complete first)

1. [ ] SlabKit on the fork — repin, six gap tests, spec/README refresh.
2. [ ] CoSpan skeleton, including chunked systems — forward passes for every non-cyclic pattern.
3. [ ] Forward time stepping (MLS-MPM).
4. [ ] Cycles, solvers, implicit components — Sellar converges.
5. [ ] Implicit time stepping (FE heat conduction; steady state matches the solver).
6. [ ] AD partials with a compute_partials override; total derivatives redesigned.
7. [ ] SLSQP, then IPOPT; constraints and objectives.

## Examples

- [ ] Sellar.
- [ ] LRPIM module + nonlocal gradient over it.
- [ ] MLS-MPM.
- [ ] FE heat conduction (a client demonstration, not only a test).

## Later, wanted

- [ ] Parallel Systems (beyond Components).
- [ ] Serialization.
- [ ] A derivative-aware paragraph per forward feature.

## Housekeeping

- [ ] CoSpan pixi manifest repinned to the naga toolchain (full previous manifest at dbebc3a:pixi.toml).
- [ ] README.md rewrite (it still describes the first-pass framework).
- [ ] Tests from the first code commit.
