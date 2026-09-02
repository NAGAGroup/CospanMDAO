# Spec 04 — System types: the extension contract, by example

Phase 4. Binding contract; declarations only. The base System, Parallel and
Fused: 01.E. Systems never do work — a System type alters ORCHESTRATION
(what runs, when, on which queue), never compute. Work is a Component's,
including custom operations on the same queue (01.C §1).

## 1. The principle

Wondering where some unique behaviour goes? It is a custom System type.
The runners are virtual precisely so that a System subclass can override
the orchestration while every Component under it stays untouched and
every owner above it stays ignorant. **No framework primitive exists for
gating, counting, or change detection**: those are ordinary System
subclasses — provided in the library where common, written by users
where not — and their entire mechanism is an early return from
`run_compute_forward`.

```cpp
namespace naga::cospan {

/// Run the subtree exactly once, ever. Construction-time operators
/// (LBVH, LRPIM, nonlocal-operator assembly at static placement).
class RunOnceSystem : public System {
    bool done_ = false;
protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        if (done_) return nullptr;            // nothing submitted; owner refreshes pending
        done_ = true;
        return System::run_compute_forward(); // the default loop
    }
};

/// Run the subtree the first N forwards, then never.
class RunNSystem : public System {
    size_t remaining_;
public:
    explicit RunNSystem(size_t n) : remaining_{n} {}
protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        if (remaining_ == 0) return nullptr;
        --remaining_;
        return System::run_compute_forward();
    }
};

/// Predicate-gated on COMPUTED state: the System names one of its own
/// (promoted) inputs; each forward it waits on its ctx's pending
/// events, reads that input on the host, and decides. The wait is the
/// honest cost of a host-side decision on device data, paid only by
/// Systems that make one. The untaken case submits nothing; outputs of
/// the subtree simply hold their previous values.
class ConditionalSystem : public System {
    std::string flag_path_;
public:
    explicit ConditionalSystem(std::string flag_input_path);
protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        ctx_.refresh_pending();
        sycl::event::wait(ctx_.pending);                       // all prior work done
        if (!read_scalar<bool>(flag_path_)) return nullptr;    // D2H of one scalar
        return System::run_compute_forward();
    }
};

/// Host-decided gating: the model's own logic (schedules, counters).
class HostConditionalSystem : public System {
    std::function<bool()> pred_;
public:
    explicit HostConditionalSystem(std::function<bool()> p) : pred_{std::move(p)} {}
protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        return pred_() ? System::run_compute_forward() : nullptr;
    }
};

} // namespace naga::cospan
```

These are added like any realized type: `add_subsystem<RunOnceSystem>("operators")`,
`add_subsystem<ConditionalSystem>("regime", "regime_flag")`. They compose by
nesting.

## 2. What the contract guarantees a custom System type

- **Runners are virtual, hooks are for users.** A System type overrides
  `run_compute_forward` (and, when it must, `prepare_exec`, `compile`,
  `run_init`); its users override `setup`/`init`/`pre_forward` as with any
  System.
- **Returning `nullptr` means "I submitted (or chose not to)".** The owner
  refreshes its pending dependencies from the queues (01.E §3) and
  carries on. Returning a description means "submit this for me" —
  a System type may also do that (a fused wave is exactly that).
- **The queues, device and substrate are in `ctx_`**, complete after
  `prepare_exec`. A System type that needs a different primary queue for
  its children (ParallelSystem) sets it in its `prepare_exec` override
  before recursing.
- **Ordering never needs cooperation from children or owners.** Pending
  dependencies are derived from the queues; a System type that submits
  on its own inherits correctness by construction.
- **Nothing above or below can tell.** The opacity contract (01.E) is
  what makes a custom System type composable anywhere in a tree.

## 3. Observation

Observation is path resolution and enumeration, nothing more:
`model.get_input("subs1.subs2.x")` resolves as
`get_subsystem("subs1") → get_subsystem("subs2") → get_input("x")`, with
`get_subsystem("subs1.subs2")` as the chaining convenience; every
Subsystem exposes `get_inputs()` / `get_outputs()` (01.B enumerability
guarantee) and a System exposes its children. Recorders, listings and
renderers are built over these calls in spec 09.
