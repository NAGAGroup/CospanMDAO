# Migration notes, 2026-10-02 (for nlmflbm-light-transport)

Two decisions land together: D141 (run policy and lifetime by counting) and D142 (shape authority). Both change what existing code means, so read the lists below against your components.

## 1. RunNSystem is gone: use `run_n` / `do_once`

Before:

```cpp
class Operators : public cs::RunNSystem {
public:
    Operators() : RunNSystem(1) {}
    void setup() override { /* ... */ }
};
add_subsystem<Operators>("operators");
```

After: make `Operators` a plain `cs::System` and give the property to whoever adds it.

```cpp
class Operators : public cs::System { void setup() override { /* ... */ } };
add_subsystem<Operators>("operators", cs::property_list{cs::property::do_once{}});    // n = 1
add_subsystem<Operators>("operators", cs::property_list{cs::property::run_n{3}});      // first 3 forwards
```

- `run_n{0}` throws `std::invalid_argument` ("naga::cospan: run_n needs n >= 1") when the `property_list` is constructed.
- `run_n` together with `do_once` is an error.
- A child asking for more runs than its parent allows is clamped to the parent's, with a warning on `std::cerr`: `'<child>' asks to run <C> time(s) but '<parent>' allows <P>; clamped to <result>`. Under a transient parent a child becomes transient ("transient (every run, revived)").
- `include/naga/cospan/run_n_system.hpp` no longer exists.
- `rearm()` on a subsystem resets its run counters and sends every root it owns back out of scope; the next forward runs it again and re-initializes its variables.

## 2. What `transient` means now

Transient is the run policy "run every forward, revive my variables each time". A variable lives by a COUNT of its users (partners), not by "the last component that touches it":

- A root comes into scope when it is first needed (revival). Its count is the sum of what its partners will still run (an eternal partner, `-1`, makes it eternal; a transient partner answers 1).
- Each partner decrements after every run it takes part in; at 0 the root is freed (the roots freed by one turn are freed behind ONE queue wait).
- So an output read by an eternal consumer outside a transient subtree stays alive, as before. But an intermediate shared by a transient and an eternal component is now eternal, and a variable shared with a `run_n` subtree is freed after the last run unless something eternal still uses it.
- **Marks no longer conflict with transient**: a design variable, constraint or objective (and any root promoted at the model level) is simply eternal. The old "marked but transient" setup error is gone.
- **Host access** to a freed root now throws "'<root>' was released after its last use (by '<path>')" (the old message named the transient subsystem).
- **A finite producer with a transient reader fails on the second run.** If a `do_once`/`run_n` writer's root is revived by a later run of a reader that outlives it, and nothing that will run writes it, the forward throws, naming the root, the reviver and the producer. Fix: promote the root (eternal) or give the reader the same policy as its writer.
- **Chunked systems** no longer force their internal roots transient: they are revived once per forward, kept at the capacity of the largest chunk, and, if eternal, kept across forwards. Memory after a forward now includes one chunk's worth of each internal root; tests that measured "gone after the forward" should measure the capacity instead.
- Arena: a static root takes the arena only when it is marked or every partner is eternal. A static root with a finite or transient partner now takes the pool.

## 3. `transient` + `run_n` / `do_once` is an error

`add_subsystem('x'): transient cannot be combined with run_n or do_once; a transient subsystem runs every forward`. A transient System that contains a `run_n` child is allowed: the child is clamped to transient with the warning above.

## 4. `cs::tied`, `FromSourceDim`, `from_source_shape`, `alias::from_connection` -> unspecified dims

Whatever supplies a root's value also supplies its shape (a connection or slice, a tie, or an initializer). A dim you do not specify is `cs::unspecified`; a dim you do specify is a constraint, never silently overridden. Mechanical rewrites:

| before | after |
|---|---|
| `cs::FromSourceDim{d}` | `cs::unspecified` |
| `cs::from_source_shape<R>()` | `cs::unspecified_shape<R>()` |
| `declare_input(w, "x", cs::tied)` (wrapper of rank R) | `declare_input(w, "x", cs::unspecified_shape<R>())` |
| `tie_inputs<..>({...}, "t", cs::alias::from_connection())` | `tie_inputs<..>({...}, "t")` |

Things to check:

- A `FromSourceDim{d}` whose `d` differs from its position (a transposing read of the source) has no equivalent; declare the dim you mean.
- Tie members may now declare dims. Unspecified dims come from the tie; specified dims are checked against it, and a conflict throws `invalid_argument` naming both sides ("shape mismatch: 'a.y' dimension 0 is fixed = 5 but its shape is supplied by 't' (a tie) ...").
- The **rank must match** the supplier's. A connection used as a reshape between ranks now throws.
- **Setup errors**: an untied OUTPUT with an unspecified dim ("... is not a member of a tie_outputs ..."), and a FREE INPUT with an unspecified dim and no initializer that gives a shape. An unconnected `FromSourceDim` input used to fail at the first forward; it now fails at `setup()`.
- A previously rank-only member that nothing tied no longer gets the "rank-only but not tied" error; it gets one of the two errors above.
- **Chunked systems**: a FIXED dim 0 inside a `ChunkedSystem` that differs from N used to be overridden silently; it is now an error at the full pass ("dimension 0 of '<entry>' is fixed = K but chunked system '<path>' supplies the full length N"). Declare dim 0 `cs::unspecified`, or equal to N.

## 5. Initializers carry shapes

A free input with unspecified dims takes its shape from its initializer:

```cpp
init_input("x", cs::init::fill(1.0f, {n, m}));
init_input("x", cs::init::zero({n}));
init_input("x", cs::init::from_host(data));              // rank 1: its size
init_input("x", cs::init::from_host(data, {n, m}));      // rank > 1
init_input("x", cs::init::custom<DType::f32, 2>(fn, {n, m}));
init_input("x", cs::init::custom_host<DType::f32, 2>(fn, {n, m}));
```

The shape-less forms (`fill(v)`, `zero()`, `custom(fn)`, `custom_host(fn)`) stay valid when every dim of the input is specified. The initializer's rank must equal the input's; a fixed dim must equal the initializer's.

## 6. `tie_inputs`' shape argument is optional

`tie_inputs<DT, R>({members...}, "name")` leaves the tie unspecified: it takes its shape from the output connected into it, else from an `init_input` on the tie (with a shape), else the free-input setup error. `tie_outputs` still requires a shape (`fixed`, `dims`, `from_output_shape`, `from_output_values`, `from_input_shape`).
