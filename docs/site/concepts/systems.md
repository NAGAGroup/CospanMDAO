# Systems

A **system** puts components together. It holds **subsystems**, which can be components or other systems, and wires them together.

## A small system

```cpp
class Chain : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Axpb>("first"));
        add_subsystem(cs::subsystem_desc<Axpb>("second"));
        connect("first.y", "second.x");
    }
};
```

- **`setup()`** is where a system builds itself: it adds its subsystems and wires them together. It runs once.
- **`add_subsystem`** adds a subsystem from a descriptor: its type, its name, and any arguments for its constructor, as in `cs::subsystem_desc<T>("name", args...)`.
- **`connect`** feeds an input from an output: here, `second` reads what `first` writes.

## Something more complex

Consider the following system:

```cpp
class Pipeline : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Source>("source"));
        add_subsystem(cs::subsystem_desc<Filter>("filter"));
        add_subsystem(cs::subsystem_desc<Report>("report"));

        connect("source.y", "filter.smooth.x");

        promote<DType::f64, 3>("filter.smooth.y", "smoothed");
        promote<DType::f64, 3>("report.stats.x", "stats");
        connect("smoothed", "stats");
    }
};
```

Let's walk through it.

### Adding subsystems

The three `add_subsystem` calls we've seen before.

### A nested connect

This `connect` looks a little different.

- **`source.y`** reads as before: the output `y` of a component, `source`.
- **`filter.smooth.x`** has one level more: the input `x` of a subsystem `smooth`, inside `filter`. So `filter` is a system.

### Promote

`promote` is new. It gives `filter.smooth.y` the name `smoothed`, and `report.stats.x` the name `stats`, in `Pipeline`.

### Connecting promotions

- Promoted names connect like any others.
- This `connect` also tells us that `smoothed` is an output and `stats` an input.

## We lied: why you need to document your system

- **Lie 1: `source` is a component.** It could just as well be a system that promoted something inside it as `y`, the way `Pipeline` promotes `smoothed` and `stats`.
- **Lie 2: `filter` is a system.** A component may declare an input named `smooth.x`. In practice, `filter` is a system, but the code alone doesn't settle it. The same goes for `report`.
- **Lie 3: `filter` holds `smooth`, and `report` holds `stats`.** They do, but they may hold more that this setup never mentions.

### Why these lies matter

A system's setup doesn't tell you everything about what it holds. Documentation is how anyone using your system, whether you, other maintainers or another project, knows what it does and how it works.

## In a system's setup

The calls above are the core of a system's setup. **`tie_inputs`** and **`tie_outputs`** work like `promote`, on several at once: `tie_inputs` promotes several inputs to one promotion, and `tie_outputs` several outputs to one promotion. More: [Connect, promote and tie](connect-promote-tie.md).

A system can do more in its setup, such as setting shapes and initial values, devices, and when its subsystems run. More: [Shapes and initialization](shapes-and-initialization.md), [Devices and placement](devices-and-placement.md), [Lifetime](lifetime.md).

## Helpers

Helpers build common subsystems without writing a class. For example, `cs::system_from` builds a system from a list of subsystems and their connections. More: [Helpers](todo/helpers.md).
