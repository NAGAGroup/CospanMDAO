# Spec 01.D — Subsystem and Component

Part of the CospanMDAO specification. Binding contract;
declarations only. Wrappers: 01.A. Metadata/shapes: 01.B. Kernel
machinery: 01.C. System: 01.E. The end-to-end lifecycle: `lifecycle.md`.

## 1. Subsystem — the ONE contract between an owner and what it owns

A System owns Subsystems and never learns which kind they are. Everything
a System calls on a child is declared here; everything a Component does
in response is the private body of its own runner overrides.

```cpp
namespace naga::cospan {

/// Framework verbs (runners). Default implementations forward to the
/// user hooks; framework-provided types (Component, System and its
/// flavors) override runners as they need; users override HOOKS. Runners
/// are virtual so authors of new System types can override them — that
/// is the extensibility contract — but a typical model never does.
/// Why: ledger §2.2 possibility floor · P5.
class Subsystem {
public:
    virtual ~Subsystem();
    std::string_view name() const;   ///< in the owner's namespace
    std::string      path() const;   ///< dotted from the root
    const System*    owner() const;  ///< null at the root

    // ---- setup chain (Model drives; lifecycle.md) ------------------------
    /// Declare structure and I/O. No device, no allocation.
    virtual void run_setup() = 0;
    /// Decide/inherit device, account static bytes. Returns
    /// device -> bytes for this subtree (merged upward). §4 for the
    /// per-entry logic.
    virtual std::unordered_map<sycl::device, size_t>
        get_exec_info(const sycl::device& parent_device) = 0;
    /// Allocate what can be allocated now and point every wrapper at its
    /// storage (own allocation, else root()->data() — roots are allocated
    /// before their peers); Systems complete their ExecCtx.
    virtual void prepare_exec(const Queues& queues, Substrates& substrates) = 0;
    /// Compile bodies (Components); recurse (Systems). Flavor legality is
    /// checked per forward on the descriptions, not here.
    virtual void compile() = 0;
    /// Initialize: children before self (Systems), so an owner's init
    /// has the last word by path.
    virtual void run_init(const ExecCtx& owner_ctx) = 0;

    // ---- forward (per forward; §5) --------------------------------------
    virtual void run_pre_compute(ExecCtx& owner_ctx) = 0;
    /// Components: return a description of their work; they never
    /// submit it. Systems: submit their subtree's work themselves and
    /// return null — which tells the owner "queues were used; refresh
    /// pending". Why: ledger §2.2 opacity · P5 (orchestration is a
    /// System's; a Component only works).
    virtual std::unique_ptr<LaunchDescription> run_compute_forward() = 0;
    virtual void run_post_compute(ExecCtx& owner_ctx) = 0;
};

/// What a Component's compute turn yields. The owner ROUTES on
/// is_custom_op(): kernels take the outer-kernel path (01.C §5), custom
/// operations take AdaptiveCpp_enqueue_custom_operation through the same
/// handler with the same dependencies/retarget/lane. A kernel
/// description additionally exposes rank, group-body and scratch facts
/// and its streams/record, for fused owners to validate and pack.
class LaunchDescription {
public:
    virtual ~LaunchDescription();
    virtual bool is_custom_op() const = 0;
    virtual sycl::event submit(SubmitTarget& t) const = 0;
    virtual const KernelParts* kernel() const { return nullptr; }  // null for custom ops
};

} // namespace naga::cospan
```

## 2. ComponentBase — the user surface

```cpp
namespace naga::cospan {

class ComponentBase : public Subsystem {
public:
    ComponentBase(const ComponentBase&) = delete;
    ComponentBase& operator=(const ComponentBase&) = delete;

    // ---- HOOKS (user-implemented) ----------------------------------------
    /// Declare I/O. Runs once, during the declare pass. ZERO allocation,
    /// no device, no queue.
    virtual void setup() = 0;
    /// Initialize own I/O. Runs once, after allocation and compilation,
    /// children before owners. Not limited beyond what `q` permits:
    /// write with an initializer inside your own kernel, fill, copy —
    /// anything. Initializing a connected input is harmless: the
    /// producer overwrites it at the first forward. Default no-op.
    /// Why: ledger §2.3 initialization · P2 (an adopted subsystem's
    /// inits must not need editing when its inputs get connected).
    virtual void init(sycl::queue& q) {}
    /// Per-forward manual shape control of a subset of entries; runs
    /// inside the automatic pre-compute after automatic derivation.
    /// Default no-op.
    virtual void pre_forward() {}

    // ---- declaration API (semantics: 01.B; PUBLIC, OpenMDAO-style) ------
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_input(Input<DT, Rank>& m, std::string name, Specs... specs,
                       Lifetime life = Lifetime::durable,
                       Coverage cov = Coverage::required);
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_output(Output<DT, Rank>& m, std::string name, Specs... specs,
                        Lifetime life = Lifetime::durable);
    template <DType DT, int Rank>
    void declare_input(Input<DT, Rank>& m, std::string name, from_input_array_t,
                       Lifetime life = Lifetime::durable,
                       Coverage cov = Coverage::required);
    template <DType DT, int Rank>
    void declare_output(Output<DT, Rank>& m, std::string name, from_input_array_t,
                        Lifetime life = Lifetime::durable);

    /// pre_forward-slot-only resize; same setter as automatic derivation.
    template <DType DT, int Rank> void reshape(Input<DT, Rank>& m, sycl::range<Rank>);
    template <DType DT, int Rank> void reshape(Output<DT, Rank>& m, sycl::range<Rank>);

    // ---- retrieval & enumeration ------------------------------------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input(std::string_view name);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view name);
    auto get_inputs() const;    ///< range of const detail::IOEntry& (01.B guarantee;
                                ///  why: ledger §2.4 observation)
    auto get_outputs() const;

    // ---- initializers (legal inside init() only; else throw) --------------
    /// Device initializer: the non-const-reference twin of Input/Output —
    /// usable directly inside a kernel the user writes on `q`.
    /// Why: ledger §2.3 initialization (not limited beyond the queue).
    template <DType DT, int Rank> Initializer<DT, Rank> get_input_initializer(std::string_view name);
    template <DType DT, int Rank> Initializer<DT, Rank> get_output_initializer(std::string_view name);
    /// Host initializer: writable host view (multi-dim accessors, shape);
    /// writes back to the device on destruction and on commit().
    template <DType DT, int Rank> HostInitializer<DT, Rank> get_input_host_initializer(std::string_view name, sycl::queue& q);
    template <DType DT, int Rank> HostInitializer<DT, Rank> get_output_host_initializer(std::string_view name, sycl::queue& q);
    /// Common-case helpers, by name.
    template <class T> void fill_input (std::string_view name, T value);
    template <class T> void fill_output(std::string_view name, T value);
    void zero_input (std::string_view name);
    void zero_output(std::string_view name);
    template <class T> void set_input_from_host (std::string_view name, std::span<const T>);
    template <class T> void set_output_from_host(std::string_view name, std::span<const T>);

protected:
    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>> by_name_;
    std::unordered_map<const void*, detail::IOEntry*>                 by_member_;
    // Runner overrides (Subsystem verbs) are implemented by Component<KB>;
    // their bodies are private to it.
};

} // namespace naga::cospan
```

## 3. Component&lt;KB&gt; — the typed layer

```cpp
namespace naga::cospan {

/// One KernelBody type per component: the one-kernel rule, by type.
template <KernelBodyC KB>
class Component : public ComponentBase {
public:
    /// USER, every forward: assemble Inputs/Outputs (and body config, and
    /// Scratch shapes) from the already-synced members; state the launch
    /// parameters. Returns the description the owner will submit.
    virtual std::unique_ptr<LaunchDescription> compute_forward() = 0;

protected:
    /// Base-provided staging. Kernel bodies: range[, wg], Inputs, Outputs
    /// [, Scratch], body. Custom-op bodies (01.C §1): Inputs, Outputs,
    /// body — no range.
    std::unique_ptr<LaunchDescription> create_launch_config(
        sycl::range<kernel_rank_v<KB>> range,
        typename KB::Inputs in, typename KB::Outputs out, KB body = {});
    std::unique_ptr<LaunchDescription> create_launch_config(
        sycl::range<kernel_rank_v<KB>> range, sycl::range<kernel_rank_v<KB>> wg,
        typename KB::Inputs in, typename KB::Outputs out, KB body = {})
        requires GroupBody<KB>;
    std::unique_ptr<LaunchDescription> create_launch_config(
        sycl::range<kernel_rank_v<KB>> range, sycl::range<kernel_rank_v<KB>> wg,
        typename KB::Inputs in, typename KB::Outputs out,
        typename KB::Scratch scratch, KB body = {})
        requires(GroupBody<KB> && HasScratch<KB>);
    std::unique_ptr<LaunchDescription> create_launch_config(
        typename KB::Inputs in, typename KB::Outputs out, KB body = {})
        requires CustomOpBody<KB>;

private:
    // ---- runner overrides (the private bodies of the Subsystem verbs) ---
    void run_setup() override;                       // -> setup()
    std::unordered_map<sycl::device, size_t>
        get_exec_info(const sycl::device& parent) override;   // §4
    void prepare_exec(const Queues&, Substrates&) override;   // §4
    void compile() override;
    void run_init(const ExecCtx& c) override;        // -> init(c.queue_for_init())
    void run_pre_compute(ExecCtx&) override;         // §5
    std::unique_ptr<LaunchDescription> run_compute_forward() override; // -> compute_forward()
    void run_post_compute(ExecCtx&) override;        // §5

    CompiledKernelMeta meta_;   // static after compile (01.C)
};

} // namespace naga::cospan
```

## 4. A Component in the setup chain

- **`run_setup`**: `setup()`. Entries exist; sets are formed by the owner's
  structural calls as they happen.
- **`get_exec_info(parent_device)`** — a Component has no device choice;
  it records `parent_device`, then for each own entry `e`:
  - `e` is root: register this device as its primary; count its static
    bytes here.
  - `e` is an input peer: if `root(e)` already has storage registered on
    this device → nothing; else register this device on `root(e)` and
    count the root's static bytes here (the root owns one copy per
    device its set spans — 02.C §2).
  - `e` is an output peer (only a tied output is one): `root(e)`'s device
    must equal this device, else ERROR naming the tie — producers cannot
    write one storage from two devices.
  (Every root is processed before any peer of it in an acyclic model —
  Systems process their alias entries before their children, and
  producers precede consumers. Cycles arrive with solvers; their
  connection ledger owns the exception.)
  - Return `{device: bytes}`.
- **`prepare_exec`**: for every own ROOT allocatable now, one allocation
  per registered device from that device's substrate (durable∧static →
  arena; durable∧dynamic with known shape → dynamic chain); then point
  every member wrapper at `root()->storage_on(this device)` and validate
  static input peers against their roots. An unconnected input with a
  `FromSourceDim` dimension errors here.
- **`compile`**: the device function at the owner's fuse rank if any,
  else native; the individual config; custom-op bodies recorded.
- **`run_init`**: `init(q)` with the owner's queue retargeted to this
  device.

## 5. A Component's forward turn

Called by its owner in this order; a Component never submits its compute
work and never sees a queue except through the owner's ctx.

1. **`run_pre_compute(ctx)`**: value-driven shapes first — for each own
   input carrying a `FromInput` spec, enqueue the D2H of that input into
   the entry's `dim_scratch`, then a custom operation (depending on the
   D2H event) that applies the resize on the host, reallocating through
   the substrate's (device, context) route from inside the custom-op
   body if grown; **wait on that
   custom op's event** — the one forced sync, so this turn's launch
   geometry is known (why: ledger §2.1 shape source · P3 — only
   value-driven entries pay it). Then shape-derived dims, `pre_forward()` hook
   (per-entry override, last write wins), dynamic connection validation,
   allocate-if-grown for the rest via the dynamic chain on `ctx.device`,
   wrapper sync.
2. **`run_compute_forward()`**: `compute_forward()` — returns the
   description. The owner submits it (01.E §3).
3. **`run_post_compute(ctx)`**: through `ctx`, after the owner's
   submission: for each output root with storage on other devices,
   memcpy primary → each. Safe to submit here because the ordering the
   work depends on already exists.

## 6. Provided kernels — v1

```cpp
namespace naga::cospan::kernels {
template <DType DT, class Op = sycl::plus<cpp_type_t<DT>>>
struct GroupReduce {      // fast; reduce_over_group + leader atomic; accumulates ONTO out
    struct Inputs  { Input<DT, 1> in; };
    struct Outputs { Output<to_atomic_v<DT>, 1> out; };
    void operator()(sycl::nd_item<1>, Inputs, Outputs) const;
};
template <DType DT, class Op = sycl::plus<cpp_type_t<DT>>>
struct OrderedReduce {    // deterministic; one agent, ascending order; fuses anywhere
    struct Inputs  { Input<DT, 1> in; };
    struct Outputs { Output<DT, 1> out; };
    void operator()(sycl::id<1>, Inputs, Outputs) const;
};
// RegionCopy, GatherInto, Scatter: spec 03.
}
```

## 7. End-to-end components

```cpp
// (a) Dynamic + value-driven + atomic; id-only body.
struct HistBody {
    struct Inputs  { Input<DType::f32,1> values; Input<DType::f32,1> weights;
                     Input<DType::u32,1> n_bins; };
    struct Outputs { Output<DType::atomic_f32,1> bins; };
    float lo, hi;
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        auto b = static_cast<size_t>((in.values[i] - lo) / (hi - lo)
                                     * static_cast<float>(in.n_bins[0]));
        out.bins[b] += in.weights[i];
    }
};
class WeightedHistogram : public Component<HistBody> {
    Input<DType::f32,1> values_, weights_;  Input<DType::u32,1> n_bins_;
    Output<DType::atomic_f32,1> bins_;  float lo_, hi_;
public:
    WeightedHistogram(float lo, float hi) : lo_{lo}, hi_{hi} {}
    void setup() override {
        declare_input (values_, "values",  FromSourceDim{0});
        declare_input (weights_,"weights", from_input_shape<1>("values"));
        declare_input (n_bins_, "n_bins",  1);
        declare_output(bins_,   "bins",    FromInput{"n_bins", 0});
    }
    void init(sycl::queue&) override { zero_output("bins"); }   // accumulate-onto needs a zero
    std::unique_ptr<LaunchDescription> compute_forward() override {
        return create_launch_config(values_.shape(),
            {values_, weights_, n_bins_}, {bins_}, {lo_, hi_});
    }
};

// (b) Group + wg override + Local scratch; nd_item body.
struct TransposeBody {
    struct Inputs  { Input<DType::f32,2> in; };
    struct Outputs { Output<DType::f32,2> out; };
    struct Scratch { Local<DType::f32,2> tile; };
    void operator()(sycl::nd_item<2> it, Inputs in, Outputs out, Scratch s) const {
        const auto l = it.get_local_id(), g = it.get_global_id();
        const bool live = g[0] < in.in.shape()[0] && g[1] < in.in.shape()[1];
        if (live) s.tile[{l[0], l[1]}] = in.in[{g[0], g[1]}];   // guard loads
        sycl::group_barrier(it.get_group());                    // never participation
        const auto gt = sycl::id<2>{ it.get_group(1) * it.get_local_range(1) + l[0],
                                     it.get_group(0) * it.get_local_range(0) + l[1] };
        if (gt[0] < out.out.shape()[0] && gt[1] < out.out.shape()[1])
            out.out[gt] = s.tile[{l[1], l[0]}];
    }
};
class TiledTranspose : public Component<TransposeBody> {
    Input<DType::f32,2> in_;  Output<DType::f32,2> out_;  static constexpr size_t WG = 16;
public:
    void setup() override {
        declare_input (in_,  "in",  from_source_shape<2>());
        declare_output(out_, "out", FromInputDim{"in",1}, FromInputDim{"in",0});
    }
    std::unique_ptr<LaunchDescription> compute_forward() override {
        TransposeBody::Scratch s;  s.tile.reshape({WG, WG + 1});
        return create_launch_config(in_.shape(), {WG, WG}, {in_}, {out_}, s, {});
    }
};

// (c) A custom operation: hand device pointers to a native library.
struct GemvBody {
    struct Inputs  { Input<DType::f32,2> A; Input<DType::f32,1> x; };
    struct Outputs { Output<DType::f32,1> y; };
    void operator()(CustomOp& op, Inputs in, Outputs out) const {
        native_gemv(op.h /* interop handle: native stream */,
                    op.device_ptr(in.A), op.device_ptr(in.x), op.device_ptr(out.y),
                    in.A.shape()[0], in.A.shape()[1]);
    }
};
class Gemv : public Component<GemvBody> {
    Input<DType::f32,2> A_;  Input<DType::f32,1> x_;  Output<DType::f32,1> y_;
public:
    void setup() override {
        declare_input (A_, "A", from_source_shape<2>());
        declare_input (x_, "x", FromInputDim{"A", 1});
        declare_output(y_, "y", FromInputDim{"A", 0});
    }
    std::unique_ptr<LaunchDescription> compute_forward() override
        { return create_launch_config({A_, x_}, {y_}); }
};
```

None of these contains a guard for machinery, a queue, a submission, an
allocation, or a pointer the user manages. `setup()`, `compute_forward()`,
optionally `init()` — and physics.
