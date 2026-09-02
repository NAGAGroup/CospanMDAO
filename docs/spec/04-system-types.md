# Spec 04 — System types: gating flavors, versions, observation plumbing

Phase 4. Binding contract; declarations only. The base System, Parallel and
Fused: 01.E. Systems never do work — every flavor here alters ORCHESTRATION
(what runs, when), never compute. Work is a Component's, including custom
operations on the same queue (01.C §1, custom-op bodies).

## 1. Versions — the one addition to IOEntry

```cpp
namespace naga::cospan::detail {
class IOEntry {   // ...continued (01.B, 02.C)
public:
    /// ROOTS ONLY (peers report their root's). Monotonic. Bumped by: the
    /// owning Component's run_post_compute (every output root, at
    /// enqueue); Model::set_input/set_output; a host view's writeback;
    /// init writes. The framework's change signal — ChangeTrackedSystem
    /// and recorders read it; nothing else is tracked.
    uint64_t version() const;
};
}
```

## 2. ConditionalSystem — the model decides

```cpp
namespace naga::cospan {

/// Predicate-gated execution. Two forms, because WHERE the decision
/// lives is the design:
///   Predicate::host(fn)       — a host callable evaluated each forward;
///                               the model's own logic (counters,
///                               schedules, flags). No device sync.
///   Predicate::from_input(p)  — a scalar integer/bool INPUT of this
///                               System, read through the same
///                               producer-push dim_scratch mechanism as
///                               FromInput (01.C §7): an event wait,
///                               never a drain. Regime switching on
///                               computed state.
struct Predicate {
    static Predicate host(std::function<bool()> fn);
    static Predicate from_input(std::string_view input_path);
};

/// Two branches, both Systems the user fills via add_subsystem.
/// run_compute_forward: evaluate; run exactly one branch (else may be
/// empty); return null. THE UNTAKEN BRANCH'S OUTPUTS HOLD their previous
/// values — not zeroed, not invalidated; dynamic shapes inside it do not
/// update. Documented, not configurable. Property form:
/// property::conditional{pred}.
class ConditionalSystem : public System {
public:
    explicit ConditionalSystem(Predicate p);
    System& then_branch();
    System& else_branch();
};

} // namespace naga::cospan
```

## 3. ChangeTrackedSystem — the framework decides

```cpp
namespace naga::cospan {

/// Run when any EXTERNAL input has changed since the last run; skip
/// otherwise. Dirtiness is local:
///   * at compile this System records the roots of its external inputs
///     (inputs of its subtree whose roots are produced outside it);
///   * run_compute_forward: if any recorded root's version() differs from
///     the value seen at the last run — or there was no last run — run
///     the children (the default loop) and record; else return null
///     without touching a queue.
/// FIRST RUN IS ALWAYS DIRTY, so this absorbs initialization / do-once: a
/// subtree with NO external inputs runs exactly once (LBVH, LRPIM,
/// nonlocal-operator construction at static placement). Cascading skip
/// is free: a skipped subtree bumps no versions, so tracked consumers
/// downstream skip too. Moving geometry needs no new primitive: the
/// geometry root's version moves -> dirty -> rebuild.
/// Property form: property::change_tracked{}.
class ChangeTrackedSystem : public System {
public:
    bool ran_last_forward() const;   ///< observation
};

} // namespace naga::cospan
```

## 4. Observation plumbing

Everything observation-shaped — recording, N2-style rendering, variable
listing, profiling, generated documentation — is built from exactly three
things and nothing else:

- **`visit`** (System/Model): pre-order traversal delivering every
  Subsystem with its `name()`/`path()` and, for each, its entries via
  `get_inputs()`/`get_outputs()` (01.B enumerability guarantee; alias
  entries included).
- **path-resolved access**: `get_input`/`get_output`, `view_*`, `set_*`
  (02.B).
- **`version()`**: the change signal; a recorder sampling by `visit`
  records only roots whose version moved.

The observation API proper (recorders, listing, docs-from-declarations)
is spec 09; this section fixes what it may rely on.

## 5. Sample

```cpp
class Flagship : public System {
    void setup() override {
        auto& build = add_subsystem("operators", property_list{property::change_tracked{}});
        build.add_subsystem<LBVHBuild>("lbvh");         // no external inputs -> runs once
        build.add_subsystem<NonlocalOps>("ops");

        auto& step = add_subsystem("regime",
            property_list{property::conditional{Predicate::from_input("regime_flag")}});
        static_cast<ConditionalSystem&>(step).then_branch().add_subsystem<LaminarStep>("lam");
        static_cast<ConditionalSystem&>(step).else_branch().add_subsystem<TurbulentStep>("turb");

        add_subsystem<Gemv>("legacy_solve");             // a custom-op Component (01.D §7c):
                                                         //   work outside SYCL, on the same queue
    }
};
```
