# Spec 01.D — Component

Part of the phase-1 spec (`docs/spec/01-components-io/`). Binding contract;
declarations only. Wrappers: 01.A. Metadata/shapes: 01.B. Kernel
machinery: 01.C. System-side execution: 01.E.

## 1. ComponentBase

```cpp
namespace naga::cospan {

class ComponentBase {
public:
    virtual ~ComponentBase();
    ComponentBase(const ComponentBase&) = delete;      // entries hold
    ComponentBase& operator=(const ComponentBase&) = delete; // member addrs

    // ---- lifecycle -------------------------------------------------------

    /// USER: declare I/O. Once, during Model::setup(), depth-first in
    /// declared order. ZERO allocation. Kernel compilation is NOT done
    /// here — the System calls compile_kernel() after setup() returns.
    virtual void setup() = 0;

    // ---- declaration API: PUBLIC, OpenMDAO-style (semantics: 01.B) ------

    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_input(Input<DT, Rank>& m, std::string name,
                       Specs... specs, Lifetime life = Lifetime::durable,
                       Coverage cov = Coverage::required);   // spec 03 §2.4
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_output(Output<DT, Rank>& m, std::string name,
                        Specs... specs, Lifetime life = Lifetime::durable);
    template <DType DT, int Rank>   // whole-shape value-driven (01.B)
    void declare_input(Input<DT, Rank>& m, std::string name,
                       from_input_array_t, Lifetime life = Lifetime::durable);
    template <DType DT, int Rank>
    void declare_output(Output<DT, Rank>& m, std::string name,
                        from_input_array_t, Lifetime life = Lifetime::durable);

    /// The resize escape hatch — legal ONLY inside the pre_forward slot
    /// (else: throw). Same setter as automatic derivation; a manual call
    /// runs after the automatic pass and wins by ordering, PER ENTRY.
    template <DType DT, int Rank>
    void reshape(Input<DT, Rank>& m, sycl::range<Rank> shape);
    template <DType DT, int Rank>
    void reshape(Output<DT, Rank>& m, sycl::range<Rank> shape);

    // ---- retrieval & enumeration: PUBLIC, OpenMDAO-style ----------------

    /// Typed retrieval; dynamic_casts the IOEntry (01.B RTTI anchor).
    /// Throws on name miss or type mismatch. Hierarchy paths
    /// ("sub.comp.x") are Systems' business (spec 04); here names are
    /// local.
    template <DType DT, int Rank> Input<DT, Rank>&  get_input(std::string_view name);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view name);

    /// THE enumerability guarantee (01.B), split by kind:
    /// const iteration over this component's entries, uniform through
    /// the hierarchy (System alias entries included, spec 04).
    auto get_inputs() const;    ///< range of const detail::IOEntry&
    auto get_outputs() const;   ///< range of const detail::IOEntry&

protected:
    /// The ONLY overridable pre-forward piece; default no-op. Runs
    /// INSIDE run_pre_forward (below), after automatic shape derivation:
    /// reshape() a SUBSET of entries and the rest keep their
    /// auto-derived shapes — per-entry last-write-wins.
    virtual void pre_forward() {}

private:
    // ---- framework surface: System-only ---------------------------------
    friend class System;    // Parallel/Fused access via System-provided
                            // protected helpers (01.E); friendship is
                            // deliberately NOT inherited.

    /// THE AUTOMATIC PRE-FORWARD DRIVER — non-virtual; the System calls
    /// it each forward. The user hook is a slot inside it and cannot
    /// displace the machinery:
    void run_pre_forward(sycl::queue& q);
    // {
    //     update_shapes(q);        // DimRecords applied to ALL dynamic
    //                              //   entries; FromInput dims event-wait
    //                              //   their dim_scratch, then read it
    //     pre_forward();           // USER SLOT (default no-op)
    //     validate_connections();  // inputs vs upstream, flat-size;
    //                              //   skipped when both sides static
    //     allocate_grown();        // dynamic chain iff extents grew past
    //                              //   capacity; no-op for aliased entries
    //     sync_wrappers();         // {data, shape, stride} -> members
    // }

    /// For each output feeding a FromInput determiner (links fixed at
    /// finalize): enqueue the D2H copy into the consumer's dim_scratch.
    /// System calls it after the launch (individual) or wave (fused).
    void post_launch(sycl::queue& q);

    /// Kernel compilation (typed layer implements): System-called at
    /// setup for the native rank, and lazily for other ranks a fuse
    /// demands. User-invisible.
    virtual void compile_kernel(int launch_rank) = 0;
    virtual const CompiledKernelMeta& kernel_meta() const = 0;

    /// Untyped entry access for the walk (validation, packing, wiring).
    detail::IOEntry&       io(std::string_view name);
    const detail::IOEntry& io(std::string_view name) const;

    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>> by_name_;
    std::unordered_map<const void*, detail::IOEntry*>                 by_member_;
};

} // namespace naga::cospan
```

## 2. Compiled metadata and the launch config

```cpp
namespace naga::cospan {

/// Static after compile_kernel — the unchanging compiled metadata the
/// Component holds and the System queries. NO runtime config cache
/// exists anywhere: an individual config never changes, and a
/// FusedSystem builds its sequence config once at ITS setup, because
/// fuse membership is static.
struct CompiledKernelMeta {
    int  native_rank;                 // deduced from the body signature
    bool group_body;                  // nd_item form?
    bool has_scratch;                 // Scratch declared?
    /// Per launch rank; native eagerly, others LAZILY on first fuse
    /// demand (SSCP lowers only invoked codepaths):
    sycl::AdaptiveCpp_jit::dynamic_function_id fn_id[3];
    /// The component's own static config for INDIVIDUAL launches:
    /// define(kb_entry_r{native}_id, fn_id[native]), built once.
    sycl::AdaptiveCpp_jit::dynamic_function_config individual_cfg;
    uint32_t inputs_bytes, outputs_bytes, body_bytes, scratch_meta_bytes;
};

/// What compute_forward returns — host-staged; the System executes it.
struct KernelLaunchConfig {
    int               rank;
    sycl::range<3>    global;         // LOGICAL; System pads to groups
    sycl::range<3>    wg;             // meaningful iff has_wg
    bool              has_wg;
    StageRecord       record;         // slice fields blank; System stamps
    detail::byte_staging inputs, outputs, body, scratch_meta;
    size_t            scratch_bytes;  // local-memory requirement
    const CompiledKernelMeta* meta;
};

} // namespace naga::cospan
```

## 3. Component&lt;KB&gt; — the typed layer

```cpp
namespace naga::cospan {

/// One KernelBody type per component: the one-kernel rule, enforced by
/// the type system.
template <KernelBodyC KB>
class Component : public ComponentBase {
public:
    /// USER, every forward — the whole kernel-side story in one place:
    /// assemble Inputs/Outputs (and body config, and Scratch shapes)
    /// from the already-synced members; state the launch parameters.
    /// The System calls this after run_pre_forward.
    virtual KernelLaunchConfig compute_forward() = 0;

protected:
    /// Base-provided staging: copies the argument structs into host
    /// staging, fills the StageRecord template (range, flat_extent,
    /// stream offsets), lays out Scratch (aligned Local offsets ->
    /// total scratch_bytes), attaches kernel_meta().
    KernelLaunchConfig create_launch_config(
        sycl::range<kernel_rank_v<KB>> range,
        typename KB::Inputs in, typename KB::Outputs out, KB body = {});
    KernelLaunchConfig create_launch_config(          // nd_item bodies
        sycl::range<kernel_rank_v<KB>> range,
        sycl::range<kernel_rank_v<KB>> wg,
        typename KB::Inputs in, typename KB::Outputs out, KB body = {})
        requires GroupBody<KB>;
    KernelLaunchConfig create_launch_config(          // + Scratch
        sycl::range<kernel_rank_v<KB>> range,
        sycl::range<kernel_rank_v<KB>> wg,
        typename KB::Inputs in, typename KB::Outputs out,
        typename KB::Scratch scratch, KB body = {})
        requires(GroupBody<KB> && HasScratch<KB>);

private:
    /// System-only (ComponentBase friend surface): instantiates the
    /// device function for KB at the requested rank (01.C §4), registers
    /// its id; native rank additionally builds individual_cfg.
    void compile_kernel(int launch_rank) override;
    const CompiledKernelMeta& kernel_meta() const override;

    CompiledKernelMeta meta_;
};

} // namespace naga::cospan
```

## 4. The canonical per-forward order — who calls what

The owning System's walk, per child Component, in exactly this order
(01.E invokes it and never rearranges it):

1. `run_pre_forward(q)` — the automatic driver: shape update →
   `pre_forward()` user slot → validation → allocation → wrapper sync.
2. `compute_forward()` — the user assembles argument structs and launch
   parameters; returns the KernelLaunchConfig.
3. Execute the config — individual: pad, stamp the degenerate slice
   `{0, all_groups}`, upload the streams (one transient allocation),
   submit via `individual_cfg` with the config's geometry. Fused: the
   wave packs all children's streams and records in execution order,
   stamps real group slices, uploads once, submits via the fuse's
   setup-built sequence config (01.E).
4. `post_launch(q)` — the dim_scratch D2H enqueues.

Steps in `run_pre_forward` are free for fully-static components;
`compute_forward` is a struct-assembly call. A user never invokes any of
these; a user only ever writes `setup()`, `compute_forward()`, and
optionally `pre_forward()`.

## 5. Provided kernels — v1 roster

```cpp
namespace naga::cospan::kernels {

/// Fast whole-array reduction: reduce_over_group + ONE leader atomic
/// combine per group. Output is the atomic variant BY MEMBER TYPE (01.A
/// discipline dogfooded). ACCUMULATES ONTO the existing value (zeroing
/// is a trivial upstream component, or a future knob). Not
/// bit-reproducible. Tails internal: loads guarded, identity
/// participation (01.C masking law).
template <DType DT, class Op = sycl::plus<cpp_type_t<DT>>>
struct GroupReduce {
    struct Inputs  { Input<DT, 1> in; };
    struct Outputs { Output<to_atomic_v<DT>, 1> out; };   // extent {1}
    void operator()(sycl::nd_item<1> it, Inputs, Outputs) const;
};

/// Deterministic whole-array reduction: id-only, launch range {1}, one
/// agent combining in ascending index order. Bit-reproducible BY
/// CONSTRUCTION; fuses anywhere (id-only => rank-exempt); slower for
/// large arrays, and the docs say so plainly.
template <DType DT, class Op = sycl::plus<cpp_type_t<DT>>>
struct OrderedReduce {
    struct Inputs  { Input<DT, 1> in; };
    struct Outputs { Output<DT, 1> out; };
    void operator()(sycl::id<1>, Inputs, Outputs) const;
};

/// v1 ships exactly these two. Scatters are already expressible (atomic
/// outputs + indexed writes — the general-case floor working as
/// designed); scans and sorts arrive as library additions when a model
/// needs them, requiring no spec change.

} // namespace naga::cospan::kernels
```

## 6. End-to-end components

```cpp
// (a) Dynamic + value-driven + atomic general-case; id-only body.
struct HistBody {
    struct Inputs  { Input<DType::f32,1> values; Input<DType::f32,1> weights;
                     Input<DType::u32,1> n_bins; };
    struct Outputs { Output<DType::atomic_f32,1> bins; };
    float lo, hi;                                   // POD config

    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        auto b = static_cast<size_t>((in.values[i] - lo) / (hi - lo)
                                     * static_cast<float>(in.n_bins[0]));
        out.bins[b] += in.weights[i];               // atomic_ref by type
    }
};

class WeightedHistogram : public Component<HistBody> {
    Input<DType::f32,1> values_, weights_;  Input<DType::u32,1> n_bins_;
    Output<DType::atomic_f32,1> bins_;
    float lo_, hi_;
public:
    WeightedHistogram(float lo, float hi) : lo_{lo}, hi_{hi} {}
    void setup() override {
        declare_input (values_, "values",  FromSourceDim{0});   // upstream-sized
        declare_input (weights_,"weights", from_input_shape<1>("values"));
        declare_input (n_bins_, "n_bins",  1);                  // scalar
        declare_output(bins_,   "bins",    FromInput{"n_bins", 0}); // value-driven
    }
    KernelLaunchConfig compute_forward() override {
        return create_launch_config(values_.shape(),
            {values_, weights_, n_bins_}, {bins_}, {lo_, hi_});
    }
    // no pre_forward override: every shape is declaratively derived
};

// (b) Group + wg override + Local scratch; nd_item body.
struct TransposeBody {
    struct Inputs  { Input<DType::f32,2> in; };
    struct Outputs { Output<DType::f32,2> out; };
    struct Scratch { Local<DType::f32,2> tile; };

    void operator()(sycl::nd_item<2> it, Inputs in, Outputs out,
                    Scratch s) const {
        const auto l = it.get_local_id();
        const auto g = it.get_global_id();
        const bool live = g[0] < in.in.shape()[0] && g[1] < in.in.shape()[1];
        if (live) s.tile[{l[0], l[1]}] = in.in[{g[0], g[1]}];  // guard loads
        sycl::group_barrier(it.get_group());              // not participation
        const auto gt = sycl::id<2>{
            it.get_group(1) * it.get_local_range(1) + l[0],
            it.get_group(0) * it.get_local_range(0) + l[1]};
        if (gt[0] < out.out.shape()[0] && gt[1] < out.out.shape()[1])
            out.out[gt] = s.tile[{l[1], l[0]}];
    }
};

class TiledTranspose : public Component<TransposeBody> {
    Input<DType::f32,2> in_;  Output<DType::f32,2> out_;
    static constexpr size_t WG = 16;
public:
    void setup() override {
        declare_input (in_,  "in",  from_source_shape<2>());
        declare_output(out_, "out", FromInputDim{"in",1}, FromInputDim{"in",0});
    }
    KernelLaunchConfig compute_forward() override {
        TransposeBody::Scratch s;
        s.tile.reshape({WG, WG + 1});                   // bank-skewed
        return create_launch_config(in_.shape(), {WG, WG},
            {in_}, {out_}, s, {});
    }
};
```

Neither component contains a guard, a launch, an allocation, an event, a
registration, or a byte of device-memory management. `setup()`,
`compute_forward()`, physics — nothing else.
