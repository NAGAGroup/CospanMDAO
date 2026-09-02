# Spec 01.E — System execution slice and structural APIs

Part of the phase-1 spec (`docs/spec/01-components-io/`). Binding contract;
declarations only. Component: 01.D. Kernel machinery: 01.C. This document
covers what a System DOES to its children each forward and the APIs a
System's `setup()` uses to assemble them. Alias resolution order, index-map
connections, optimization marks, and cross-level tie semantics are owned by
later specs (forward-linked where they arise).

## 1. System — the execution contract

```cpp
namespace naga::cospan {

/// THE OPACITY CONTRACT, execution side: children are opaque. When a
/// child's forward-participation returns, its effects are ORDERED on q
/// ahead of every later sibling — the owner cannot tell (and never asks)
/// whether a plain launch, a fused wave, a parallel fan-out, or a solver
/// iteration happened inside. Completion is the queue's business;
/// Model::run() owns the final wait.
class System {
public:
    virtual ~System();

    /// USER: add children, wire I/O (§4 APIs). Once, at setup.
    virtual void setup() = 0;

    /// FRAMEWORK: the forward walk. Default implementation: children in
    /// declared order, depth-first —
    ///     for (child : children_) visit(child, q);
    /// where visit() dispatches: Component -> execute() below;
    /// System -> child.forward(q). Overridable by System subclasses —
    /// the extensibility contract: new orchestration = a new System
    /// subclass overriding forward(), composed of the same protected
    /// helpers, with no base changes.
    virtual void forward(sycl::queue& q);

protected:
    // ---- the friend-helper layer ----------------------------------------
    /// ComponentBase befriends System ONLY (01.D); subclasses reach the
    /// framework surface through these. Thin, static, complete:
    static void run_pre_forward(ComponentBase& c, sycl::queue& q);
    static KernelLaunchConfig compute_forward(ComponentBase& c);
    static void post_launch(ComponentBase& c, sycl::queue& q);
    static void compile_kernel(ComponentBase& c, int launch_rank);
    static const CompiledKernelMeta& kernel_meta(const ComponentBase& c);

    /// The individual-launch path — the canonical order's step 3 (01.D
    /// §4), executed for one component:
    ///   1. pad cfg.global to work-group multiples of the launch
    ///      geometry (cfg.wg if has_wg, else the rank default);
    ///   2. stamp the degenerate slice {0, all_groups} into cfg.record;
    ///   3. stage device memory (ONE transient allocation: records[1] +
    ///      inputs + outputs + bodies streams), one H2D copy;
    ///   4. submit the rank-matched outer kernel (01.C §5) with the
    ///      component's static individual_cfg and a local_accessor of
    ///      cfg.scratch_bytes (zero when unused).
    static sycl::event execute(ComponentBase& c, sycl::queue& q);
    // { run_pre_forward(c, q);
    //   auto cfg = compute_forward(c);
    //   auto ev  = detail::launch_individual(cfg, q);
    //   post_launch(c, q);
    //   return ev; }

    std::vector<child_ref> children_;   // Components and Systems, declared order
};

} // namespace naga::cospan
```

## 2. FusedSystem

```cpp
namespace naga::cospan {

/// Children: COMPONENTS ONLY (structural rule). Mode fixed at
/// construction. Rank and wg shape are SYSTEM-LEVEL parameters here —
/// component wg intent is ignored inside a fuse (01.C §6).
class FusedSystem final : public System {
public:
    enum class Mode : uint8_t { sequential, parallel };

    FusedSystem(Mode mode, int rank /* = 1 */,
                sycl::range<3> wg /* = rank default */);

    /// setup(), beyond the base: the STATIC assembly, once —
    ///   * compile every child at the fuse rank (compile_kernel; id-only
    ///     bodies lazily instantiate here; an nd_item child whose native
    ///     rank differs from the fuse rank is a HARD ERROR);
    ///   * intra-fuse shape-dependency check: any determiner-or-
    ///     connection chain from one child's shape to another child's
    ///     output is a HARD ERROR naming the chain (01.C §7);
    ///   * parallel mode ONLY: producer-consumer check from I/O
    ///     metadata — a child consuming another child's output in the
    ///     same wave is an ERROR by default, user-overridable
    ///     (allow_producer_consumer(); the tie_outputs trust model);
    ///   * sequential mode ONLY: interleave kb_barrier between
    ///     consecutive scratch-declaring stages (01.C §5). Sequential
    ///     fusion's semantics: same-index producer->consumer inside one
    ///     work-item's call chain works; cross-index dependence does
    ///     not — documented, not checked;
    ///   * build THE sequence config, once:
    ///     define_as_call_sequence(kb_entry_r{rank}, ordered fn ids).
    void setup() override;

    /// forward(): one wave, one launch —
    ///   1. run_pre_forward for every child, declared order;
    ///   2. compute_forward for every child; collect configs;
    ///   3. PACK: concatenate inputs/outputs/bodies streams and
    ///      StageRecords in EXECUTION ORDER (= sequence order = the
    ///      cursor's order — the four-way identity, 01.C §5); stamp
    ///      slices: sequential -> {0, own group count} per stage,
    ///      launch extent = max; parallel -> prefix-sum group offsets,
    ///      launch extent = Σ; scratch = max over stages;
    ///   4. ONE transient device staging, ONE H2D, ONE submit via the
    ///      setup-built sequence config;
    ///   5. post_launch for every child.
    void forward(sycl::queue& q) override;
};

} // namespace naga::cospan
```

## 3. ParallelSystem

```cpp
namespace naga::cospan {

/// Children: any (Components or Systems), one LANE each — a private
/// in-order queue per child, same device and context as the parent's.
/// The user asserts lane independence by using this type (data
/// independence is the user's responsibility; the ONE thing the
/// framework checks is cross-sibling shape dependencies — hard error,
/// 01.C §7, because a shape read is provably a data dependency).
class ParallelSystem final : public System {
public:
    /// forward():
    ///   1. FENCE-IN: each lane's first work must not begin before
    ///      prior siblings' work on q — the parent enqueues an event
    ///      join from q into every lane;
    ///   2. dispatch: child i executes on lane_q[i] (Component ->
    ///      execute(); System -> forward(lane_q[i]));
    ///   3. FENCE-OUT: q's subsequent work must not begin before every
    ///      lane completes — event join from every lane back into q.
    ///      The join mechanism is implementation-defined (event
    ///      dependencies preferred; a host wait is a conforming v1);
    ///      the CONTRACT is only the ordering, stated in those two
    ///      fences. The opacity contract holds: to the owner this was
    ///      one child step on q.
    void forward(sycl::queue& q) override;

private:
    std::vector<sycl::queue> lanes_;   // created at setup; in-order
};

} // namespace naga::cospan
```

**Queues:** created at the Model and flow DOWN through `forward(q)` —
Systems receive and route them; ParallelSystem manufactures its lanes at
setup; Components never orchestrate a queue (their framework surface
receives `q` only for event-waits and the post_launch enqueues, both
framework-driven).

## 4. Structural APIs

All structural calls are setup-time only; a structural call outside
`setup()` throws.

```cpp
namespace naga::cospan {

class System {   // ...continued
public:
    // ---- children --------------------------------------------------------

    /// Emplace a child (Component or System), owned by this System,
    /// named in this System's namespace. Returns the typed reference:
    ///     auto& hist = add_subsystem<WeightedHistogram>("hist", 0.f, 1.f);
    template <class T, class... Args>
        requires(std::derived_from<T, ComponentBase> ||
                 std::derived_from<T, System>)
    T& add_subsystem(std::string name, Args&&... ctor_args);

    /// Pre-built child transfer (ownership moves in).
    template <class T>
    T& add_subsystem(std::string name, std::unique_ptr<T> child);

    // ---- connections -----------------------------------------------------

    /// THE ONLY WAY TO ALIAS I/O BETWEEN SIBLINGS. Wires src (an OUTPUT
    /// path) to dst (an INPUT path). Paths are dot-separated and resolve
    /// in THIS System's namespace: child paths ("solver.mda.y1") and
    /// this-System promoted/tied names both legal. RULES:
    ///   * output -> input only, always;
    ///   * an input has AT MOST ONE source — a second connection to the
    ///     same input is an error at the call;
    ///   * dtype compatibility on value_dtype_of (01.A): the atomic
    ///     modifier never blocks a connection;
    ///   * size compatibility is FLAT-SIZE equality, checked at the
    ///     01.B validation point (setup-checked when both static;
    ///     re-checked each forward when either side dynamic);
    ///   * ALIASING (mechanics: 02.C): a connection merges the target's
    ///     set into the source's; connecting through ANY name of a set
    ///     is the same connection. Promotion is always allowed; ties
    ///     must precede connections ("tie before connecting, or connect
    ///     from the alias").
    void connect(std::string_view src_output, std::string_view dst_input);

    /// Non-identity forms take a ConnectionMap (spec 03): Slice (a view,
    /// zero-copy), SliceInto / IndexMap / IndexMapInto (computed, via a
    /// hidden mapping System).
    void connect(std::string_view src_output, std::string_view dst_input,
                 ConnectionMap map);

    // ---- promotion -------------------------------------------------------

    /// Lift ONE child I/O into this System's namespace under
    /// promoted_name. Kind-agnostic and ALWAYS allowed — including on
    /// already-connected or already-aliased entries (a higher level may
    /// always want a name, e.g. probing an adopted System). Creates an
    /// ALIAS ENTRY on this System — a real detail::IOEntry that joins
    /// the promoted entry's set as a PEER with a snapshot of the root's
    /// flags (02.C), enumerated under get_inputs()/get_outputs() by
    /// those flags, retrievable by get_input()/get_output(), and
    /// connectable AT THE PARENT LEVEL under the promoted name. No data
    /// exists on the System: peers own nothing.
    void promote(std::string_view child_path, std::string promoted_name);

    // ---- ties ------------------------------------------------------------

    /// Alias N child INPUTS to ONE promoted input in this namespace.
    /// All members receive the SAME data: one upstream connection to
    /// promoted_name feeds every member (e.g. every discipline consuming
    /// the same design vector). Members must agree on value_dtype; flat
    /// sizes validated per 01.B. NEVER inputs with outputs — there is
    /// no mixed tie, by design. Members must be unconnected (02.C);
    /// after the tie only the alias is a legal connection target.
    void tie_inputs(std::initializer_list<std::string_view> child_input_paths,
                    std::string promoted_name);

    /// Alias N child OUTPUTS to ONE promoted output: the members WRITE
    /// TO THE SAME STORAGE. Collision discipline is the whole question,
    /// and the default answer is the type system's: every tied member
    /// SHALL have an atomic dtype (01.A), checked at this call — the
    /// B/C accumulation pattern with no forced copies and no adder
    /// component. The override is the trust model verbatim: the user
    /// asserting the writes cannot conflict (e.g. disjoint index sets)
    /// takes responsibility with allow_plain. Members must have no
    /// downstream connections yet (02.C); after the tie only the alias
    /// is a legal connection source.
    enum class TieOutputsPolicy : uint8_t { require_atomic, allow_plain };
    void tie_outputs(std::initializer_list<std::string_view> child_output_paths,
                     std::string promoted_name,
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic);
};

/// Rectangular subarray descriptor (rank-typed helper -> stored form):
///     subarray<2>({row0, col0}, {rows, cols})
struct SubarrayRef {
    sycl::range<3> offset;   // 0-padded
    sycl::range<3> extents;  // 1-padded
    uint8_t        rank;
};
template <int R>
SubarrayRef subarray(sycl::range<R> offset, sycl::range<R> extents);

} // namespace naga::cospan
```

## 5. Sample — Sellar-shaped wiring

```cpp
class SellarMDA : public System {
public:
    void setup() override {
        auto& d1 = add_subsystem<Discipline1>("d1");
        auto& d2 = add_subsystem<Discipline2>("d2");

        connect("d1.y1", "d2.y1_in");            // plain connection
        connect("d2.y2", "d1.y2_in");            //   (the cycle: solvers spec)

        tie_inputs({"d1.z", "d2.z"}, "z");       // shared design vector:
                                                 //   one entry, one upstream
        promote("d1.y1", "y1");                  // visible to the parent
        promote("d2.y2", "y2");
    }
};

class Top : public System {
    void setup() override {
        auto& mda  = add_subsystem<SellarMDA>("mda");
        auto& obj  = add_subsystem<Objective>("obj");

        connect("mda.y1", "obj.y1");             // promoted names connect
        connect("mda.y2", "obj.y2_head",      // like any output —
                Slice{subarray<1>({0}, {1})});   //   slices included
    }
};
```

Load-bearing from earlier chunks: alias entries are ordinary `IOEntry`s,
so the enumerability guarantee and `get_input<DT,R>("mda.y1")` hierarchy
retrieval need no special cases (01.B's RTTI anchor); peers own nothing,
so allocation is roots-only (02.C); `value_dtype_of` compatibility lets
an atomic-tied output connect onward to a plain consumer input without
ceremony (01.A).

Deferred, explicitly: subarray-of-subarray composition,
DV/constraint/objective marks on promoted entries (spec 03).
