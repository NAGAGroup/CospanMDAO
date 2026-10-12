# Systems

A **system** puts components together. It holds **subsystems**, which can be components or other systems, and it decides how they relate. A system does no work of its own: it has no kernel and declares no inputs or outputs of its own. Everything is computed by components. How those components relate is decided by systems.

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
- **`add_subsystem`** adds a subsystem from a descriptor: its type, its name, and any arguments for its constructor, as in `cs::subsystem_desc<T>("name", args...)`. The subsystem is built and set up right away.
- **`connect`** feeds an input from an output: here, `second` reads what `first` writes.

## Systems of systems

A system can hold other systems, and `cs::system_from` builds a system from a list of subsystems and their connections, without writing a class for it:

```cpp
class Pipeline : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Source>("source"));           // output: data

        add_subsystem(cs::system_from(
            "process", cs::connects_desc({{"filter.y", "post.scale.x"}}),
            cs::subsystem_desc<Filter>("filter"),                       // input: x, output: y
            cs::system_from(
                "post", cs::connects_desc({{"scale.y", "clamp.x"}}),
                cs::subsystem_desc<Scale>("scale"),                     // input: x, output: y
                cs::subsystem_desc<Clamp>("clamp"))));                  // input: x, output: y

        connect("source.data", "process.filter.x");
        promote<DType::f32, 1>("process.post.clamp.y", "result");
    }
};
```

`process` is a system holding a component (`filter`) and another system (`post`), which itself holds two components.

## Names and paths

Every subsystem has a name, unique within the system that holds it. An input or output is reached by its **path**: the names from the current system down to it, joined by dots. Inside `process`, the scale's input is `post.scale.x`. Inside `Pipeline`, the same input is `process.post.scale.x`.

Each system is its own namespace. Two systems can each have a subsystem named `scale` without conflict.

## Wiring

- **`connect`** feeds an input from an output.
- **`promote`** promotes an input or output into the current system's namespace. Above, `Pipeline` promotes the clamp's output to `result`, so it can be reached as `result` instead of `process.post.clamp.y`. Promotion is often used to make a value easy to reach when inspecting a model, or to define a system's "public interface".
- **`tie_inputs` and `tie_outputs`** make several inputs, or several outputs, share one value under one name.

More: [Connect, promote and tie](connect-promote-tie.md).

## What else a system decides

- **Shapes and values.** Shape hooks can be used to size inputs and outputs, and input initializers to give free inputs their starting values. More: [Shapes and initialization](shapes-and-initialization.md).
- **Order.** Subsystems are added producers first. A consumer added before its producer is read as a cycle, and a warning is printed. Which subsystems can run at the same time follows from the data. More: [Execution and synchronization](execution-and-synchronization.md).
- **When and where subsystems run.** Run policies, rearming a subsystem, and which device a subsystem runs on. More: [Advanced](advanced.md).

## No forward of its own

A system does no work and has no `pre_forward()` or `post_forward()`: it only groups subsystems and declares wiring. Work belongs to components. To bring a finished subsystem back into play, call `model.rearm_subsystem(path)` from `main` between runs.
