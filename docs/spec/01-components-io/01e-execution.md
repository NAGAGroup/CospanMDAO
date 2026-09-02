# Spec 01.E — System: execution context, orchestration, structural APIs

Part of the phase-1 spec (`docs/spec/01-components-io/`). Binding contract;
declarations only. Subsystem verbs and Component: 01.D. Aliasing mechanics:
02.C. Topology semantics: 03. Flavors beyond Parallel/Fused: 04. The
end-to-end lifecycle: `lifecycle.md`.

## 1. ExecCtx — System-owned execution state

```cpp
namespace naga::cospan {

/// Every System owns exactly one. Populated across the setup chain
/// (device in get_exec_info; queues, lane and substrate in prepare_exec)
/// and never handed to anyone: children receive a reference only to USE
/// it — to allocate, to look up the device, to submit copies in
/// post-compute. Nothing in the tree ever reads another System's ctx.
struct ExecCtx {
    sycl::device            device;      ///< this System's choice (own or inherited)
    sycl::queue*            primary;     ///< where THIS System submits: q_in by
                                         ///  default; q_ooo when a Parallel owner
                                         ///  assigned this System a lane
    sycl::queue*            q_in;        ///< both queues are always present
    sycl::queue*            q_ooo;
    std::optional<size_t>   lane;        ///< hint for q_ooo submissions
    MemorySubstrate*        substrate;   ///< this device's arena (+ pool)
    std::vector<sycl::event> pending;    ///< dependencies for the next submission

    /// SubmitTarget view: what a LaunchDescription::submit consumes —
    /// primary, pending (with retarget=device and lane applied), then
    /// pending is cleared on an in-order primary and kept on an
    /// out-of-order one.
    SubmitTarget target();
    /// Re-derive pending from both queues (ACPP_EXT_QUEUE_WAIT_LIST:
    /// ≤1 event each for in-order; coarse-grained events elide backend
    /// event creation). Called at the top of run_compute_forward and
    /// after any child that submitted on its own.
    void refresh_pending();
};

struct Queues { sycl::queue* q_in; sycl::queue* q_ooo; };
struct MemorySubstrate { slabkit::DeviceArena arena; std::optional<slabkit::DevicePool> pool; };
using  Substrates = std::unordered_map<sycl::device, MemorySubstrate>;

/// What a submission consumes (01.D LaunchDescription::submit): the
/// queue, the dependency snapshot, and the CG properties to attach.
struct SubmitTarget { sycl::queue& q; std::span<const sycl::event> pending;
                      sycl::device retarget; std::optional<size_t> lane; };
/// What a fused owner reads off a kernel description (01.D): the facts it
/// validates and the parts it packs.
struct KernelParts { int rank; bool group_body; bool has_scratch;
                     const StageRecord& record; /* input/output/body streams */ };

} // namespace naga::cospan
```

## 2. System — concrete, orchestration only

```cpp
namespace naga::cospan {

/// A System never does work: it orders, submits and gates the work its
/// children describe. It has no I/O of its own beyond alias entries
/// (02.C). `System` is a complete type: a System with no overrides is a
/// plain sequential container (the property-list add_subsystem overload
/// creates exactly that).
class System : public Subsystem {
public:
    // ---- HOOKS ------------------------------------------------------------
    virtual void setup() {}                     ///< structure & wiring (§4)
    virtual void init(sycl::queue& q) {}        ///< runs AFTER children's; override by path
    virtual void pre_forward() {}

    // ---- Model's entries ------------------------------------------------------
    /// Model knows its root is a System and has no ctx of its own, so the
    /// root supplies its own: init_tree() == run_init(ctx_);
    /// forward() == run_compute_forward() (no pre/post-compute exists for
    /// the root).
    void init_tree();
    void forward();

    // ---- initializers by path (legal inside init() only) ----------------------
    template <DType DT, int Rank> Initializer<DT, Rank>     get_input_initializer (std::string_view path);
    template <DType DT, int Rank> Initializer<DT, Rank>     get_output_initializer(std::string_view path);
    template <DType DT, int Rank> HostInitializer<DT, Rank> get_input_host_initializer (std::string_view path, sycl::queue&);
    template <DType DT, int Rank> HostInitializer<DT, Rank> get_output_host_initializer(std::string_view path, sycl::queue&);
    template <class T> void fill_input (std::string_view path, T v);
    template <class T> void fill_output(std::string_view path, T v);

    // ---- retrieval & enumeration (hierarchy-aware; 01.B guarantee) ------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input (std::string_view path);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view path);
    auto get_inputs()  const;   ///< this System's alias entries
    auto get_outputs() const;
    Subsystem& get_subsystem(std::string_view path);          ///< chains through children

    // ---- runners (Subsystem verbs) — the DEFAULT orchestration ----------------
protected:
    /// setup(); children's run_setup fire inside add_subsystem (§4).
    void run_setup() override;
    /// ctx.device = own property::device or parent_device. Process OWN
    /// alias entries first (§ 01.D §4 logic — a tie alias root counts its
    /// bytes here), then children in order; merge maps (sum on key).
    std::unordered_map<sycl::device, size_t>
        get_exec_info(const sycl::device& parent_device) override;
    /// Complete ctx (primary = q_in, no lane, substrate = substrates[device]);
    /// allocate own alias roots and point their wrappers; then children
    /// with the SAME queue set.
    void prepare_exec(const Queues&, Substrates&) override;
    /// children.compile() in order.
    void compile() override;
    /// children.run_init in order, then init(q retargeted to ctx.device).
    void run_init(const ExecCtx&) override;
    void run_pre_compute(ExecCtx&) override {}     ///< Systems do no work
    void run_post_compute(ExecCtx&) override {}
    /// THE DEFAULT LOOP (§3).
    std::unique_ptr<LaunchDescription> run_compute_forward() override;

    ExecCtx ctx_;
    std::vector<std::unique_ptr<Subsystem>> children_;   ///< declared order = exec order
    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>> aliases_;
};

} // namespace naga::cospan
```

## 3. The default `run_compute_forward` — and how ordering works

```
ctx_.refresh_pending();                       // deps on everything both queues hold
for (child : children_) {
    child->run_pre_compute(ctx_);
    auto d = child->run_compute_forward();
    if (d) ctx_.pending = { d->submit(ctx_.target()) };   // WE submit, routed on
                                                         //   d->is_custom_op(); our event is next's dep
    else   ctx_.refresh_pending();            // child submitted on its own; re-read queues
    child->run_post_compute(ctx_);            // copies AFTER the work they depend on
}
return nullptr;                               // "I submitted"
```

- **Ordering is derived, never passed.** Which queue a System submits to
  is static (`primary`, fixed in `prepare_exec` by what its owner handed
  down). Cross-queue dependencies come from the queues themselves
  (`get_wait_list`), refreshed at the top of the loop and after any child
  that submitted for itself. In-order primaries consume `pending` with
  their first submission; out-of-order primaries attach it to each.
- **Devices are per submission.** Every submission carries
  `AdaptiveCpp_retarget{ctx.device}` (the queues are `retargetable`); an
  in-order queue keeps its ordering across devices (acpp
  `ACPP_EXT_CG_PROPERTY_RETARGET`). Nobody creates queues but the Model.
- **Waiting is the caller's choice**: `Model::wait()` waits both queues;
  a loop of forwards need not wait between them.

## 4. Structural APIs — all inside `setup()`, else throw

```cpp
namespace naga::cospan {

/// Options are SYCL-style properties: unordered, self-naming, type-checked.
namespace property {
    struct device        { sycl::device dev; };            // placement; inherited when absent
    struct parallel      {};                               // flavor
    struct fused         { FusedSystem::Mode mode; int rank = 1; std::optional<sycl::range<3>> wg; };
}
// Gating, counting and change detection are custom System TYPES (spec
// 04), added with the templated overload — not properties.
class property_list { /* SYCL idiom: constructed from any subset; has_property<T>(), get_property<T>() */ };

class System {   // ...continued
public:
    /// A REALIZED System or Component type, optionally with properties
    /// (device).
    template <class T, class... Args> requires std::derived_from<T, Subsystem>
    T& add_subsystem(std::string name, Args&&... ctor_args);
    template <class T, class... Args> requires std::derived_from<T, Subsystem>
    T& add_subsystem(std::string name, const property_list& props, Args&&... ctor_args);
    /// A BASE System configured by properties: device and/or ONE flavor
    /// (parallel | fused — both on one call is an error; nest to combine). Returns the System to add
    /// children to inline — no class derivation needed.
    System& add_subsystem(std::string name, const property_list& props = {});
    // add_subsystem constructs the child, registers the name, and runs the
    // child's run_setup BEFORE returning, so later structural calls in this
    // same setup() see it fully declared.

    void connect(std::string_view src, std::string_view dst);                 // identity
    void connect(std::string_view src, std::string_view dst, ConnectionMap);  // spec 03
    void promote(std::string_view child_path, std::string promoted_name);
    void tie_inputs (std::initializer_list<std::string_view> members, std::string name);
    void tie_outputs(std::initializer_list<std::string_view> members, std::string name,
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic);
    void add_design_var(...);  void add_constraint(...);  void add_objective(...);   // spec 03
};

} // namespace naga::cospan
```

Rules (mechanics in 02.C; semantics in 03): output → input only; an input
has at most one source, enforced at the ROOT's flag after the first
connection; connecting through any name of a set is the same connection;
`promote` is kind-agnostic and always allowed; ties must precede
connections ("tie before connecting, or connect from the alias");
`tie_outputs` requires atomic member dtypes unless `allow_plain`.

## 5. ParallelSystem and FusedSystem — the two policy flavors

```cpp
namespace naga::cospan {

/// Children are lanes on q_ooo. prepare_exec hands child i the queue set
/// with primary = q_ooo, lane = i. run_compute_forward: refresh pending
/// (this captures everything q_in holds); for each child: pre-compute →
/// compute → submit on q_ooo with lane i and the SAME pending snapshot →
/// post-compute. Return null; the owner's refresh then picks up q_ooo's
/// wait list. Data independence of children is the USER's assertion;
/// the framework checks only cross-sibling shape dependencies (hard
/// error, checked each forward on the children's descriptions).
class ParallelSystem : public System { /* runner overrides as described */ };

/// COMPONENTS ONLY — enforced by the type system at the add, since a
/// System cannot discern children at runtime:
///   template <class T, class... A> requires std::derived_from<T, ComponentBase>
///   T& add_subsystem(std::string, A&&...);
/// (no property-list overload). Rank and wg shape are the fuse's;
/// children compile at the fuse rank (id-only bodies lazily); the
/// sequence config is built once at compile.
/// run_compute_forward: pre-compute ALL children → compute ALL (collect
/// descriptions) → VALIDATE on the descriptions, every forward, host-
/// side and cheap: none null, none custom-op, rank agreement for group
/// bodies, scratch handling, parallel mode: no producer-consumer pairs
/// unless allowed → pack streams + records, stamp group slices, one
/// staging, ONE submission with the sequence config → post-compute ALL.
/// The order is forced: post-compute copies depend on the wave.
class FusedSystem : public System {
public:
    enum class Mode : uint8_t { sequential, parallel };
    FusedSystem(Mode, int rank = 1, std::optional<sycl::range<3>> wg = {});
    void allow_producer_consumer();   // parallel-mode override (trust model)
};

} // namespace naga::cospan
```

## 6. Sample — Sellar with inline Systems

```cpp
class SellarMDA : public System {
public:
    void setup() override {
        auto& d1 = add_subsystem<Discipline1>("d1");
        auto& d2 = add_subsystem<Discipline2>("d2");
        connect("d1.y1", "d2.y1_in");
        connect("d2.y2", "d1.y2_in");            // the cycle: solvers (spec 05)
        tie_inputs({"d1.z", "d2.z"}, "z");
        promote("d1.y1", "y1");  promote("d2.y2", "y2");
    }
    void init(sycl::queue&) override { fill_input("z", 1.0); }   // last word over d1/d2 inits
};

class Top : public System {
    void setup() override {
        auto& mda = add_subsystem<SellarMDA>("mda", property_list{property::device{gpu0}});
        auto& obj = add_subsystem<Objective>("obj");
        connect("mda.y1", "obj.y1");
        connect("mda.y2", "obj.y2_head", Slice{subarray<1>({0}, {1})});

        auto& par = add_subsystem("post", property_list{property::parallel{}});
        par.add_subsystem<Report>("r1");  par.add_subsystem<Report>("r2");
    }
};
```
