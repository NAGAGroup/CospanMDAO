# Spec 02.B — Model

Part of the CospanMDAO specification. Binding contract; declarations only. Memory objects:
slabkit 02.A. Aliasing: 02.C. Subsystem verbs: 01.D. System: 01.E. The
full end-to-end walk: `lifecycle.md`.

```cpp
namespace naga::cospan {

/// The Model owns exactly three kinds of thing: the two queues, the
/// per-device memory substrates, and the root System. It drives the
/// setup chain and the forward; it never touches a Subsystem except
/// through the Subsystem verbs and System::forward().
struct ModelConfig {
    /// Per-device dynamic-pool sizes; a device not listed gets
    /// `default_pool_bytes`. 0 = no pool = every dynamic allocation takes
    /// the OS route. Pre-allocation is a profiling-discovered
    /// optimization; memory_report() is how it is discovered.
    std::unordered_map<sycl::device, size_t> pool_bytes;
    size_t default_pool_bytes = 0;
    size_t pool_node_size     = 1024;
    /// The default device (root System's device unless it overrides;
    /// what every System inherits by default). Defaults to the runtime's
    /// default selector.
    std::optional<sycl::device> default_device;
};

/// THE factory. Constructs the root System with `args...`, the Model
/// around it, and returns the Model by value (prvalue; Model is movable,
/// so this is also merely ordinary).
template <class Sys, class... Args> requires std::derived_from<Sys, System>
Model create_model(ModelConfig cfg, Args&&... args);

class Model {
public:
    Model(Model&&) noexcept;  Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;

    // ---- lifecycle -------------------------------------------------------------
    /// THE SETUP CHAIN. Runs once. Every step is a Subsystem verb driven
    /// from the root; the Model adds only the two things nobody else
    /// owns (queues, substrates):
    ///   0. queues: q_in (in-order, AdaptiveCpp_retargetable,
    ///      AdaptiveCpp_coarse_grained_events) and q_ooo (out-of-order,
    ///      retargetable), both bound to default_device — created at
    ///      construction, so they exist for everything below.
    ///   1. root.run_setup()                       — declare (01.D/01.E)
    ///   2. bytes = root.get_exec_info(default_device)   — device -> bytes
    ///   3. substrates[d] = { DeviceArena(bytes[d]), pool iff configured }
    ///   4. root.prepare_exec(queues, substrates) — allocate, point wrappers; ExecCtx complete
    ///   5. root.compile()                        — bodies
    ///   6. root.init_tree()                      — children before owners
    /// After setup the model's STATE is not yet meaningful: it holds
    /// initialization. State is defined after the first forward.
    void setup();

    /// root.forward(): enqueue one forward. Returns immediately. Loop it
    /// freely — the model guarantees ordering internally.
    void forward();
    /// Wait both queues. The caller's choice, never implicit.
    void wait();

    // ---- host access (between runs) ---------------------------------------------
    /// Views: wrapper ergonomics on the host — shape(), operator[] with
    /// id<Rank> and flat, size(). Constructing a view performs the D2H
    /// copy of the root on q_in — ordered after both queues' wait lists,
    /// like every framework submission — and waits for it (the honest
    /// cost, paid only where asked). Read-only variants cost no writeback.
    template <DType DT, int Rank> HostView<DT, Rank>    view_input (std::string_view path) const;
    template <DType DT, int Rank> HostView<DT, Rank>    view_output(std::string_view path) const;   // read-only
    /// Writable output view: writes land in host staging; written back
    /// to the device on destruction AND on commit() (both; the
    /// destructor is the protection, commit() the mid-scope flush).
    template <DType DT, int Rank> HostViewMut<DT, Rank> view_output_mut(std::string_view path);
    /// Setters: H2D into the root's primary storage on q_in, then copied
    /// to the root's other-device storage like any root write. Inputs:
    /// meaningful for free inputs (a connected one is overwritten at the
    /// next forward — harmless, like init).
    template <DType DT, int Rank> void set_input (std::string_view path, std::span<const cpp_type_t<DT>>);
    template <DType DT, int Rank> void set_output(std::string_view path, std::span<const cpp_type_t<DT>>);

    // ---- retrieval & observation ------------------------------------------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input (std::string_view path);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view path);
    System& root();
    Subsystem& get_subsystem(std::string_view path);           // root.get_subsystem

    struct MemoryReport {
        struct PerDevice { size_t arena_bytes, pool_bytes, pool_bytes_in_use,
                                  os_allocations, os_bytes_in_use, transient_peak_bytes; };
        std::unordered_map<sycl::device, PerDevice> devices;
    };
    MemoryReport memory_report() const;

    // ---- marks (spec 03) --------------------------------------------------------
    std::span<const MarkRecord> design_vars() const;
    std::span<const MarkRecord> constraints() const;
    std::span<const MarkRecord> objectives()  const;

    /// Waits both queues. Everything else is RAII.
    ~Model();

private:
    sycl::queue q_in_, q_ooo_;
    Substrates  substrates_;
    std::unique_ptr<System> root_;
    ModelConfig cfg_;
};

} // namespace naga::cospan
```

## Sample — the whole user-side lifecycle

```cpp
auto model = create_model<Top>(ModelConfig{});          // default device, no pools
model.setup();                                          // declare → exec-info → substrates
                                                        //   → prepare → compile → init
model.set_input<DType::f64, 1>("mda.z", {{1.0, 2.0}});  // drive a free input
model.forward();  model.wait();                         // enqueue; wait when results are wanted

auto y1 = model.view_output<DType::f64, 1>("mda.y1");   // D2H, read
auto r  = model.memory_report();                        // r.devices[gpu0].os_bytes_in_use > 0 ?
                                                        //   -> rerun with pool_bytes set to it
```
