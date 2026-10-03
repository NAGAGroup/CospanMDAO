#pragma once

// =============================================================================
// naga/cospan/model.hpp — the Model
// =============================================================================
//
// The Model owns one sk::context, one sk::queue (containers), one sk::unordered_queue (kernels and host tasks, ordered
// by events only), one pool per device (sized by the user up front), the root System, and the marks. It drives the
// two setup passes of docs/design-review-2026-10-03.md, section 2, and never touches a Subsystem except through the
// Subsystem verbs:
//   setup:    the root's run_setup, recursive through add_subsystem.
//   finalize: collects the marks (design variables, constraints, objectives) by absolute path, as weak pointers to
//             their instances. Nothing else.
// Setup never allocates, never submits and never compiles. There is no default device (section 2): every instance
// takes AdaptiveCpp's default when it is declared, and a device given to add_subsystem overrides a subtree.
//
// The forward (section 3), host views and update_input (section 5) and the memory report are all here.
//
// STATE LIVES ON THE HEAP. The Model is one pointer to an Impl; moving a Model moves the pointer. Inside Impl the
// members are declared so that destruction (the reverse of declaration) runs the root System first: its instances go,
// and each device's data, each waiting on its own registered events, is freed while the queues, the pools and the
// context are still alive. Then the marks, the config, the Runtime (the pools), the queues, and the context.
//
// LAYERING. ~Model does not wait; call wait() when "everything I submitted is done" matters.

#include <concepts>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/diagnostics.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/system.hpp"
#include "naga/cospan/wiring.hpp"

namespace naga::cospan {

struct ModelConfig {
    /// The platform context every device of the model lives in.
    sk::context ctx;

    /// The runtime's default-selector device only.
    ModelConfig() : ctx(sycl::device{sycl::default_selector_v}) {}
    explicit ModelConfig(sk::context c) : ctx(std::move(c)) {}
    /// A context over exactly these devices.
    explicit ModelConfig(std::span<const sycl::device> devices)
        : ctx(std::vector<sycl::device>(devices.begin(), devices.end())) {}

    /// Per-device pool sizes in bytes; a device not listed gets `default_pool_bytes`. 0 = no pool = every allocation
    /// takes the OS route. `pool_node_size` is the page granularity (section 2).
    std::unordered_map<sycl::device, std::size_t> pool_bytes;
    std::size_t                                   default_pool_bytes = 0;
    std::size_t                                   pool_node_size     = 1024;

    /// Handed to both queues at construction; reached by wait().
    std::optional<sycl::async_handler> async_handler;
};

/// One mark, by absolute path, collected at the end of setup. The instance is held weakly: a mark on a deleted
/// subsystem's instance expires.
struct Mark {
    std::string                     path;
    std::weak_ptr<detail::Instance> inst;
    MarkRecord                      rec;
};

class Model;

/// THE factory: constructs the root System with `args...` and the Model around it.
template <class Sys, class... Args>
    requires std::derived_from<Sys, System>
Model create_model(ModelConfig cfg, Args&&... args);

class Model {
public:
    /// Use create_model. Throws std::invalid_argument for a null root.
    Model(ModelConfig cfg, std::unique_ptr<System> root) : s_{std::make_unique<Impl>(std::move(cfg), std::move(root))} {}

    Model(Model&&) noexcept = default;
    Model& operator=(Model&& o) noexcept {
        if (this != &o) {
            finish();
            s_ = std::move(o.s_);
        }
        return *this;
    }
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    /// Does not wait; never throws.
    ~Model() { finish(); }

    // ---- lifecycle -------------------------------------------------------------
    /// SETUP, then FINALIZE (section 2). Runs once: a second call throws, and so does a call after a failed first
    /// one. Nothing is allocated, submitted or compiled.
    void setup() {
        Impl& s = impl("setup");
        if (s.setup_started)
            throw std::logic_error("naga::cospan: Model::setup() runs once; it has already run (or failed part-way)");
        s.setup_started = true;

        System& root  = *s.root;
        root.name_    = "";   // the root: no name, no path, no owner
        root.path_    = "";
        root.owner_   = nullptr;
        root.run_setup();       // pass 1: declare, recursively
        collect_marks(root, s);  // pass 2: finalize
        s.ready = true;
    }

    /// ENQUEUES one forward and returns: precompute, compute and postcompute of the root, in that order. It does not
    /// wait: forwards stream with no artificial synchronization, so call wait() or sync a host view when you need the
    /// results (section 5).
    void forward() {
        Impl& s = ready("forward");
        ++s.rt.forward;
        s.root->run_precompute(s.rt);
        s.root->run_compute(s.rt);
        s.root->run_postcompute(s.rt);
    }

    /// queue.wait_and_throw() on both queues. OPTIONAL: for "everything I submitted is done" and to surface
    /// asynchronous errors (wait_and_throw) from work nobody synced (section 5). Repeated runs stream without it.
    void wait() {
        diag::count_queue_waits();
        Impl& s = impl("wait");
        s.queue.wait_and_throw();
        s.uqueue.wait_and_throw();
    }

    // ---- host access (section 5), between and after runs ------------------------------------
    // A host view is an instance of the variable on a host placement. While it lives, every forward pushes one copy to it
    // (each live view costs one copy per forward); obtain one only after a run; call sync() before reading it. See
    // host_view.hpp.
    template <DType DT, int Rank>
    HostView<DT, Rank> view_input(std::string_view path) {
        return HostView<DT, Rank>{make_view("view_input", path, IOKind::input, DT, Rank, /*dense=*/true)};
    }
    template <DType DT, int Rank>
    HostView<DT, Rank> view_output(std::string_view path) {
        return HostView<DT, Rank>{make_view("view_output", path, IOKind::output, DT, Rank, /*dense=*/true)};
    }
    /// Views of a ragged variable (also of a dense one: its blocks are the rows).
    template <DType DT, int Rank>
    HostRagged<DT, Rank> view_ragged_input(std::string_view path) {
        return HostRagged<DT, Rank>{make_view("view_ragged_input", path, IOKind::input, DT, Rank, /*dense=*/false)};
    }
    template <DType DT, int Rank>
    HostRagged<DT, Rank> view_ragged_output(std::string_view path) {
        return HostRagged<DT, Rank>{make_view("view_ragged_output", path, IOKind::output, DT, Rank, /*dense=*/false)};
    }

    /// The only host write (section 5): assigns an initializer to the input's variable and marks it STALE. It queues NO
    /// work now. At the next run, the first thing that touches the variable acts on it: not materialized, it
    /// materializes and initializes; materialized, it initializes IN PLACE (waiting on the previous run's readers, and
    /// recorded as a write). Several updates before a run collapse to the last one, and a variable nothing touches is never
    /// initialized. A host view synced before the next run still shows the OLD values. If the input has a source (it is
    /// connected to an output) a warning is printed and the update is never applied. Naming an output throws
    /// std::invalid_argument.
    void update_input(std::string_view path, InitSpec init) {
        Impl&                                   s    = ready("update_input");
        const std::shared_ptr<detail::Instance> inst = s.root->instance_at(path);
        if (inst->kind != IOKind::input)
            throw std::invalid_argument("naga::cospan: Model::update_input: '" + std::string{path} +
                                        "' is an output; outputs are never initialized");
        detail::VarMeta& v = *inst->meta;
        if (detail::first_of_kind(v, IOKind::output)) {
            std::cerr << "naga::cospan: warning: input '" << path << "' is connected to an output; update_input is never applied\n";
            return;
        }
        v.init       = std::move(init);
        v.init_depth = 0;  // outranks every declaration made in setup
        v.init_by    = "Model::update_input";
        v.mark_stale();
    }

    // ---- retrieval & observation ------------------------------------------------
    System&    root() { return *impl("root").root; }
    Subsystem* get_subsystem(std::string_view path) { return ready("get_subsystem").root->get_subsystem(path); }

    struct MemoryReport {
        /// A snapshot at the time of the call, between runs.
        struct PerDevice {
            std::size_t pool_bytes = 0, pool_bytes_in_use = 0;    ///< slots x node size
            std::size_t os_allocations = 0, os_bytes_in_use = 0;  ///< allocations the pool sent to the OS (cumulative) and the bytes of those still live; tracked for devices with a pool only
        };
        std::unordered_map<sycl::device, PerDevice> devices;
    };
    MemoryReport memory_report() const {
        Impl&        s = ready("memory_report");
        MemoryReport r;
        for (const auto& [d, pool] : s.rt.pools) {
            MemoryReport::PerDevice p;
            p.pool_bytes        = pool.slot_count() * pool.node_size();
            p.pool_bytes_in_use = pool.outstanding() * pool.node_size();
            p.os_allocations    = pool.fallback_allocations();
            p.os_bytes_in_use   = pool.fallback_bytes_in_use();
            r.devices.emplace(d, p);
        }
        return r;
    }

    // ---- marks, collected at the end of setup() -----------------------------------
    std::span<const Mark> design_vars() const { return ready("design_vars").design_vars; }
    std::span<const Mark> constraints() const { return ready("constraints").constraints; }
    std::span<const Mark> objectives() const { return ready("objectives").objectives; }

private:
    struct Impl {
        Impl(ModelConfig cfg_in, std::unique_ptr<System> r)
            : ctx(cfg_in.ctx)
            , queue{ctx, cfg_in.async_handler.value_or(sycl::async_handler{})}
            , uqueue{ctx, cfg_in.async_handler.value_or(sycl::async_handler{})}
            , rt{queue, uqueue, build_pools(ctx, queue, cfg_in)}
            , root{std::move(r)}
            , cfg{std::move(cfg_in)} {
            if (!root) throw std::invalid_argument("naga::cospan: Model needs a root System");
        }

        /// One pool per context device whose size (its own, else the default) is above 0. Built in place: a pool is
        /// not copied.
        static std::unordered_map<sycl::device, sk::DevicePool> build_pools(const sk::context& ctx, const sk::queue& q,
                                                                           const ModelConfig& c) {
            std::unordered_map<sycl::device, sk::DevicePool> pools;
            for (const sycl::device& d : ctx.devices()) {
                const auto        it    = c.pool_bytes.find(d);
                const std::size_t bytes = it != c.pool_bytes.end() ? it->second : c.default_pool_bytes;
                if (bytes > 0) pools.try_emplace(d, q, d, sk::kind::device, bytes, c.pool_node_size);
            }
            return pools;
        }

        // Declaration order = construction order; destruction is the reverse: setup_started, ready, cfg, marks, root
        // (its instances and their data first), the Runtime (pools), the queues, the context.
        sk::context             ctx;
        sk::queue               queue;
        sk::unordered_queue     uqueue;
        detail::Runtime         rt;  ///< holds the pools by value
        std::unique_ptr<System> root;
        std::vector<Mark>       design_vars, constraints, objectives;
        ModelConfig             cfg;
        bool                    setup_started = false;
        bool                    ready         = false;
    };

    /// The state, or a throw for a moved-from Model.
    Impl& impl(const char* verb) const {
        if (!s_) throw std::logic_error(std::string{"naga::cospan: Model::"} + verb + " on a moved-from Model");
        return *s_;
    }
    /// The state once setup() has completed; else a throw.
    Impl& ready(const char* verb) const {
        Impl& s = impl(verb);
        if (!s.ready) throw std::logic_error(std::string{"naga::cospan: Model::"} + verb + " before setup() has completed");
        return s;
    }

    /// A host view instance of the variable at `path`: checked, joined to the variable on a host placement on the device
    /// with the most instances (section 5), and attached (a copy of a live placement; nothing is initialized).
    std::shared_ptr<detail::Instance> make_view(const char* verb, std::string_view path, IOKind want, DType dt, int rank, bool dense) {
        Impl& s = ready(verb);
        if (s.rt.forward == 0)
            throw std::logic_error(std::string{"naga::cospan: Model::"} + verb +
                                   ": a host view can be obtained only after a run (call forward() first)");
        const std::shared_ptr<detail::Instance> inst = s.root->instance_at(path);
        if (inst->kind != want)
            throw std::invalid_argument(std::string{"naga::cospan: Model::"} + verb + ": '" + std::string{path} + "' is " +
                                        (want == IOKind::input ? "an output" : "an input"));
        if (!dtype_compatible(inst->meta->dtype, dt) || inst->meta->rank != rank)
            throw std::invalid_argument(std::string{"naga::cospan: Model::"} + verb + ": '" + std::string{path} +
                                        "' was asked for with the wrong dtype or rank");
        if (dense && inst->meta->shape.known && inst->meta->shape.ragged())
            throw std::invalid_argument("naga::cospan: '" + std::string{path} +
                                        "' is ragged and has no rectangular host view; use view_ragged_input / view_ragged_output");
        const std::shared_ptr<detail::VarMeta> meta = inst->meta;
        const sycl::device                     dev  = meta->busiest_device();  // counted before the view joins
        std::shared_ptr<detail::Instance> view = detail::make_alias_instance(dt, rank, "<host view>", nullptr, inst->kind, detail::Role::host_view,
                                                                              meta->definition, false, false, meta);
        view->device       = dev;
        view->storage_kind = sk::kind::host;
        meta->attach_missing(s.rt);
        return view;
    }

    /// Releases the Impl in the order the header names. Does not wait.
    void finish() noexcept {
        if (!s_) return;
        s_.reset();
    }

    /// FINALIZE: depth-first in declaration order, a System's own marks and then its live children. Each declared
    /// mark becomes a Mark by absolute path (the declaring System's path, a dot, the declared path; at the root, the
    /// declared path).
    static void collect_marks(const System& sys, Impl& s) {
        for (const System::MarkDecl& m : sys.marks()) {
            Mark out{sys.path().empty() ? m.path : sys.path() + "." + m.path, m.inst, m.rec};
            switch (m.kind) {
                case System::MarkDecl::Kind::design_var: s.design_vars.push_back(std::move(out)); break;
                case System::MarkDecl::Kind::constraint: s.constraints.push_back(std::move(out)); break;
                case System::MarkDecl::Kind::objective: s.objectives.push_back(std::move(out)); break;
            }
        }
        for (const Subsystem* c : sys.children())
            if (const auto* sub = dynamic_cast<const System*>(c)) collect_marks(*sub, s);
    }

    std::unique_ptr<Impl> s_;
};

template <class Sys, class... Args>
    requires std::derived_from<Sys, System>
Model create_model(ModelConfig cfg, Args&&... args) {
    return Model{std::move(cfg), std::make_unique<Sys>(std::forward<Args>(args)...)};
}

}  // namespace naga::cospan
