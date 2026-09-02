# Spec 02.B — Model

Phase-2 chunk B. Binding contract; declarations only. Memory objects:
slabkit 02.A. Aliasing and the finalize walks: 02.C.

```cpp
namespace naga::cospan {

// -----------------------------------------------------------------------------
// 1. Model — the top of the tree, and the only owner of memory and queues
// -----------------------------------------------------------------------------
/// ONE queue, ONE device. Multi-device is the queue's business (acpp's
/// multi-device queue, ruled in phase 0): a Model built on such a queue
/// changes nothing here — slabkit's queue-target overloads allocate in
/// that queue's context. Cospan never enumerates devices.
struct ModelOptions {
    size_t dynamic_pool_bytes = 0;    ///< THE knob. 0 = no pool: every
                                      ///  dynamic allocation goes to the OS
                                      ///  route. Pre-allocation is a
                                      ///  profiling-discovered optimization
                                      ///  (memory_report() is how).
    size_t pool_node_size     = 1024;
};

class Model {
public:
    /// The queue SHALL be in-order: the execution contract (01.E) is
    /// stated in terms of in-order enqueue. An out-of-order queue is
    /// rejected at construction.
    explicit Model(sycl::queue q, ModelOptions opts = {});
    Model(const Model&) = delete;  Model& operator=(const Model&) = delete;

    // ---- the tree -----------------------------------------------------------
    /// The root System, emplaced. Exactly one; a second call throws.
    template <class S, class... Args>
        requires std::derived_from<S, System>
    S& set_root(Args&&... ctor_args);
    System& root();

    // ---- lifecycle ------------------------------------------------------------
    /// THE SETUP DRIVER. Runs once; a second call throws. In order:
    ///   1. DECLARE — root.setup() recursively, depth-first in declared
    ///      order. Every add_subsystem / declare_* / connect / promote /
    ///      tie happens inside, and each structural call leaves the
    ///      aliasing sets consistent as it returns (02.C). ZERO
    ///      allocation.
    ///   2. FINALIZE — the byte FOLD (02.C §5.1): depth-first, each
    ///      System reports the durable∧static bytes of ITS OWN root
    ///      entries, each Component its own; Model sums. No registry.
    ///   3. MEMORY — construct the DeviceArena at exactly the folded
    ///      size; construct the DevicePool iff dynamic_pool_bytes > 0.
    ///   4. ALLOCATE — the same walk again: every durable∧static root
    ///      allocates from the arena; other roots stay unallocated until
    ///      first need (01.B). Peers own nothing.
    ///   5. WIRE — FromInput DimRecords register with their input's root
    ///      (02.C §5.4); static input peers validate against their roots
    ///      once (dynamic pairs re-validate each forward).
    ///   6. COMPILE — compile_kernel on every Component at its native
    ///      rank; FusedSystem::setup assembly (01.E).
    ///   7. SYNC — every wrapper reads its root's {data, shape, stride}.
    void setup();

    /// Enqueue one forward: root.forward(q). Returns immediately (the
    /// queue is in-order; effects are ordered). For callers composing
    /// loops (temporal, solvers — later phases) without host round-trips.
    void forward();

    /// forward() + q.wait(). The final wait belongs here and nowhere
    /// else (01.E: "Model::run() owns the final wait").
    void run();

    sycl::queue& queue();

    // ---- access ---------------------------------------------------------------
    /// Path-resolved retrieval from the root ("mda.d1.y1"; promoted names
    /// legal at every level) — the uniform chained lookup of 01.B.
    template <DType DT, int Rank> Input<DT, Rank>&  get_input(std::string_view path);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view path);

    /// The profiling surface that makes the knob discoverable.
    struct MemoryReport {
        size_t arena_bytes;          // durable∧static total (exact)
        size_t pool_bytes;           // 0 when no pool
        size_t pool_bytes_in_use;
        size_t os_allocations;       // count of dynamic-chain OS-route hits
        size_t os_bytes_in_use;      // what a pool of this size would absorb
        size_t transient_peak_bytes; // launch streams / arenas, peak per forward
    };
    MemoryReport memory_report() const;

private:
    // ---- the allocation target handed down the tree ---------------------------
    /// IOEntry::allocate (01.B/02.C) takes exactly this. Route selection is
    /// lifetime-driven, never a user choice:
    ///   durable ∧ static  -> arena (step 4, once)
    ///   anything else     -> the dynamic chain, at first need and on growth:
    ///       pool ? slabkit::try_malloc_device<std::byte>(n, *pool) : nullopt
    ///         -> slabkit::malloc_device<std::byte>(n, q)
    ///         -> AllocationError{OutOfMemory} propagates: no fallback
    ///            around a broken invariant; exit with the cause named.
    struct MemoryTargets {
        sycl::queue&             q;
        slabkit::DeviceArena&    arena;
        slabkit::DevicePool*     pool;    // null when no pool
    };
    MemoryTargets targets();

    sycl::queue                              q_;
    std::unique_ptr<System>                  root_;
    std::optional<slabkit::DeviceArena>      arena_;   // exists after step 3
    std::optional<slabkit::DevicePool>       pool_;    // iff knob > 0
    friend class System;  friend class detail::IOEntry;
};

} // namespace naga::cospan
```

## Sample — the whole user-side lifecycle

```cpp
sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
Model model{q};                              // no pool: the safe default
auto& top = model.set_root<Top>();           // Top::setup wires the tree (01.E sample)
model.setup();                               // declare → fold → memory → allocate → wire → compile → sync

model.get_input<DType::f64, 1>("mda.z");     // host-side access by path
model.run();                                 // forward + wait

auto r = model.memory_report();              // r.os_bytes_in_use > 0 ?
                                             //   -> rerun with {.dynamic_pool_bytes = r.os_bytes_in_use}
```
