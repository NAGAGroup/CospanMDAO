# Spec 02 — Model & memory

Phase-2 chunks:

- **02.A slabkit surface** — lives in slabkit's repo, which owns it:
  `slabkit/docs/spec/02a-slabkit-surface.md` (arena/pool memory objects,
  malloc and make families, one handle type, no Context).
- **02.B Model** — `create_model`, the two queues, per-device substrates,
  the setup chain, `forward`/`run`, host views and setters, teardown.
- **02.C Aliasing** — root+peers sets, construction-time flags, the
  structural operations as set operations, storage rules.
