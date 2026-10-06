# Section 11 implementation log

Running log, newest entries at the bottom. Each entry: date/time (UTC), phase, what was done, commits (repo + SHA), test results, decisions taken, open questions for Jack.

## Spelling decisions (for Jack's review)

| Name | Where | Meaning |
|---|---|---|
| `define_as_indexed_call_sequence(call, definitions, index_param)` | AdaptiveCpp fork, `dynamic_function_config` | defines a call sequence in which definition k is called with the wrapper's own arguments, except that argument `index_param` is replaced by the integer constant k |
| `cs::fuse_components("name", cs::connects_desc({...}), cs::segment(...), cs::global_barrier(), cs::segment(...), ...)` | CoSpan, fused group helper | builds a fused group from segments and boundary primitives; a bare stage desc is a one-stage segment |
| `cs::ticket_order()` | CoSpan, fused group boundary primitive | tickets for the whole launch, and per-group completion recording |
| `cs::global_barrier()` | CoSpan, fused group boundary primitive | ticket order, plus waiting for every earlier group |
| `cs::segment<cs::local_mem::shared>(...)` | CoSpan, fused group segment | stages of the segment see the same local memory block (max of their requests) |
| `cs::segment<cs::local_mem::partitioned>(...)` | CoSpan, fused group segment (the default) | each stage gets its own local memory slice (sum of their requests) |
| `cs::range_mode::own` | CoSpan, fused group segment option (the default) | each stage sees only its own requested groups |
| `cs::range_mode::whole` | CoSpan, fused group segment option | each stage sees the whole range |
| `override_run_policy("b.c", policy)` | CoSpan, `System` method | sets a subsystem's run policy by path; legal anywhere |
| `view_input(input)` | CoSpan, protected `Component` method | makes a host view of one of the component's own inputs, using its precompute's queue |
| `model.get_input(path)` | CoSpan, `Model` | returns a handle with `.view()` and `.update(init)` |
| `model.get_output(path)` | CoSpan, `Model` | returns a handle with `.view()` |
| `set_shape(out, cs::as_is, 12, cs::as_is)` | CoSpan, component `pre_forward` | one entry per dimension: a number or `cs::as_is` |
| `segment()`, `num_segments()`, `segment_groups(s)`, `group_in_segment()` | CoSpan, `GroupItem` | the current segment, the number of segments, a segment's group count, the group's index within its segment (from 0) |
| `wait_group(s, g)`, `wait_earlier_segments()` | CoSpan, `GroupItem` | collective wait on a specific earlier group, or on all earlier groups |
| `local<T>(n)` | CoSpan, `GroupItem` | a typed slice of the group's local memory |

## Open questions for Jack

## Entries

### 2026-10-06 09:05 UTC, P0: documents
- **Done:** wrote `docs/plan-section11.md` (the implementation plan, verbatim from the lead) and this log, with the spelling table pre-filled from "Initial spelling decisions" in the plan.
- **Commits:** cospan, "docs: section 11 implementation plan and log" (SHA in the commit; see git log).
- **Tests:** none (docs only).
- **Decisions taken:** none.
- **Open questions for Jack:** none.
