# Spec 02 — Model, memory & finalize

Phase-2 chunks:

- **02.A slabkit surface** — lives in slabkit's repo, which owns it:
  `slabkit/docs/spec/02a-slabkit-surface.md` (arena/pool memory objects,
  malloc and make families, one handle type, no Context).
- **02.B Model** — construction, memory objects, queue creation, the setup
  driver, `run`, final wait. (pending)
- **02.C The finalize pass** — alias resolution order + accumulation +
  allocation as ONE walk; ownership flavor. (pending)
