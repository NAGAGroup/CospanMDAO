#pragma once

// =============================================================================
// naga/cospan/model.hpp — the Model (r01 §9, §11)
// =============================================================================
//
// The Model owns exactly four kinds of thing: one sk::context, one sk::queue
// (the in-order stream), the per-device memory substrates ({arena, optional
// pool}) and the root System. It drives the setup chain and the forward, and
// never touches a Subsystem except through the Subsystem verbs. The chain is
// what the smoke tests' RootDriver proved; this is its production form.
//
// STATE LIVES ON THE HEAP. After setup the Systems hold raw pointers to the
// queue and to the substrates (ExecCtx), so those objects must not move when
// the Model does. The Model is therefore one pointer to an Impl that owns them;
// moving a Model moves the pointer. Inside Impl the members are declared in
// the order that makes their destruction the one §9.4 names: cfg, the mark
// records, root, substrates, queue, ctx.
//
// LAYERING (§9.4). Each layer orders what it submits; ~Model does not wait
//: call wait() before a Model is destroyed or reassigned.

#include <concepts>
#include <iostream>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/component.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/system.hpp"

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

    /// Per-device dynamic-pool sizes; a device not listed gets
    /// `default_pool_bytes`. 0 = no pool = every dynamic allocation takes the
    /// OS route.
    std::unordered_map<sycl::device, std::size_t> pool_bytes;
    std::size_t                                   default_pool_bytes = 0;
    std::size_t                                   pool_node_size     = 1024;

    /// The default device (the root System's device, what every System
    /// inherits). Set: validated with ctx.contains(), else the Model throws.
    /// Unset: the runtime's default-selector device if the context holds it,
    /// else the context's first device.
    std::optional<sycl::device> default_device;

    /// Handed to the queue at construction; reached by wait().
    std::optional<sycl::async_handler> async_handler;
};

class Model;

/// THE factory: constructs the root System with `args...` and the Model
/// around it.
template <class Sys, class... Args>
    requires std::derived_from<Sys, System>
Model create_model(ModelConfig cfg, Args&&... args);

class Model {
public:
    using MarkRecord = cospan::MarkRecord;

    /// Use create_model. Throws std::invalid_argument for a null root or a
    /// default_device outside the context.
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
    /// THE SETUP CHAIN (§9.1). Runs once: a second call throws, and so does a
    /// call after a failed first one. After setup the model holds
    /// initialization; its state is defined after the first forward.
    void setup() {
        Impl& s = impl("setup");
        if (s.setup_started)
            throw std::logic_error("naga::cospan: Model::setup() runs once; it has already run (or failed part-way)");
        s.setup_started = true;

        System&             root = *s.root;
        const sycl::device& dev  = s.queue.get_device();  // the default device (step 0)

        detail::SubsystemAccess::place(root, "", "", nullptr);
        root.run_setup();                                            // 1. declare
        const auto info = root.get_exec_info(dev);                   // 2. device -> bytes
        for (const auto& [d, bytes] : info) {                        // 3. substrates
            if (!s.ctx.contains(d))
                throw std::invalid_argument("naga::cospan: a subsystem is placed on a device that is not in the "
                                            "model's context");
            MemorySubstrate m;
            // Slack for the arena's per-allocation alignment (as the driver did).
            if (bytes > 0) m.arena.emplace(s.queue, d, sk::kind::device, bytes + kArenaSlack);
            const std::size_t pool = s.pool_bytes_for(d);
            if (pool > 0) m.pool.emplace(s.queue, d, sk::kind::device, pool, s.cfg.pool_node_size);
            s.substrates.emplace(d, std::move(m));
        }
        s.substrates.try_emplace(dev);  // the default device always has a substrate entry
        root.prepare_exec(s.queue, s.substrates);                    // 4. capture queue/substrates, shapes (no allocation)
        root.compile();                                              // 5. bodies
        std::unordered_set<const detail::IOEntry*> seen;             // 6. marks (§8.4)
        collect_marks(root, s, seen);
        s.ready = true;
    }

    /// The root's run_pre_compute, run_compute_forward, run_post_compute: enqueue one forward and return.
    void forward() {
        System&  root = *ready("forward").root;
        ExecCtx& ctx  = detail::SystemAccess::root_ctx(root);
        root.run_pre_compute(ctx);
        (void)root.run_compute_forward();
        root.run_post_compute(ctx);
    }

    /// queue.wait_and_throw(): the caller's choice, never implicit.
    void wait() {
        diag::count_queue_waits();
        Impl& s = impl("wait");
        s.queue.wait_and_throw();
        if (s.root) s.root->clear_retired();  // every launch is done: nothing retired is still read
    }

    // ---- host access (between runs) ---------------------------------------------
    template <DType DT, int Rank>
    HostView<DT, Rank> view_input(std::string_view path) const {
        Impl& s = ready("view_input");
        return HostView<DT, Rank>{entry_of(s, path, IOKind::input, "view_input"), s.queue};
    }
    template <DType DT, int Rank>
    HostView<DT, Rank> view_output(std::string_view path) const {
        Impl& s = ready("view_output");
        return HostView<DT, Rank>{entry_of(s, path, IOKind::output, "view_output"), s.queue};
    }
    /// host copies of a ragged entry (also of a dense one: offsets are the row starts).
    template <DType DT, int Rank>
    HostRagged<DT, Rank> view_ragged_input(std::string_view path) const {
        Impl& s = ready("view_ragged_input");
        return detail::HostViewAccess::ragged<DT, Rank>(entry_of(s, path, IOKind::input, "view_ragged_input"), s.queue);
    }
    template <DType DT, int Rank>
    HostRagged<DT, Rank> view_ragged_output(std::string_view path) const {
        Impl& s = ready("view_ragged_output");
        return detail::HostViewAccess::ragged<DT, Rank>(entry_of(s, path, IOKind::output, "view_ragged_output"), s.queue);
    }
    // ---- initial values: the only way to give an input data from the host -----------
    /// Set the initial value of the root of the input at `path`, any time after setup(). It wins
    /// over the Component's and the Systems' declarations, and the root's OWNER applies it at its
    /// next run (once per call; a root not yet allocated gets it at its allocation). An output
    /// path throws std::invalid_argument. If the input is connected to an output (it has a
    /// source) the init is stored, a warning is printed to std::cerr, and it is never applied.
    void init_input(std::string_view path, InitSpec init) {
        Impl&            s = ready("init_input");
        detail::IOEntry& e = s.root->entry_at_from_outside(path);
        if (e.kind() != IOKind::input)
            throw std::invalid_argument("naga::cospan: Model::init_input: '" + std::string{path} +
                                        "' is an output; outputs are never initialized");
        e.set_model_init(std::move(init));
        if (e.root()->kind() == IOKind::output)
            std::cerr << "naga::cospan: warning: input '" << path
                      << "' is connected to an output; your initializer may not run\n";
    }

    // ---- retrieval & observation ------------------------------------------------
    template <DType DT, int Rank>
    Input<DT, Rank>& get_input(std::string_view path) {
        System& root = *ready("get_input").root;
        (void)root.entry_at_from_outside(path);  // the opacity check, as an OUTSIDE caller
        return root.template get_input<DT, Rank>(path);
    }
    template <DType DT, int Rank>
    Output<DT, Rank>& get_output(std::string_view path) {
        System& root = *ready("get_output").root;
        (void)root.entry_at_from_outside(path);  // the opacity check, as an OUTSIDE caller
        return root.template get_output<DT, Rank>(path);
    }
    System&    root() { return *impl("root").root; }
    Subsystem* get_subsystem(std::string_view path) { return ready("get_subsystem").root->get_subsystem(path, /*outside=*/true); }

    struct MemoryReport {
        /// A snapshot at the time of the call, between runs.
        struct PerDevice {
            std::size_t arena_bytes = 0, arena_bytes_in_use = 0;  ///< capacity (with the alignment slack) / handed out
            std::size_t pool_bytes = 0, pool_bytes_in_use = 0;    ///< slots x node size
            std::size_t os_allocations = 0, os_bytes_in_use = 0;  ///< not tracked anywhere yet: always 0
        };
        std::unordered_map<sycl::device, PerDevice> devices;
    };
    MemoryReport memory_report() const {
        Impl&        s = ready("memory_report");
        MemoryReport r;
        for (const auto& [d, m] : s.substrates) {
            MemoryReport::PerDevice p;
            if (m.arena) {
                p.arena_bytes        = m.arena->capacity();
                p.arena_bytes_in_use = m.arena->used();
            }
            if (m.pool) {
                p.pool_bytes        = m.pool->slot_count() * m.pool->node_size();
                p.pool_bytes_in_use = m.pool->outstanding() * m.pool->node_size();
            }
            r.devices.emplace(d, p);
        }
        return r;
    }

    // ---- marks (§8.4), collected at the end of setup() ------------------------------
    std::span<const MarkRecord> design_vars() const { return ready("design_vars").design_vars; }
    std::span<const MarkRecord> constraints() const { return ready("constraints").constraints; }
    std::span<const MarkRecord> objectives() const { return ready("objectives").objectives; }

private:
    static constexpr std::size_t kArenaSlack = 4096 * 64;

    struct Impl {
        Impl(ModelConfig cfg_in, std::unique_ptr<System> r)
            : ctx(cfg_in.ctx)
            , queue{ctx, resolve_default(cfg_in), cfg_in.async_handler.value_or(sycl::async_handler{})}
            , root{std::move(r)}
            , cfg{std::move(cfg_in)} {
            if (!root) throw std::invalid_argument("naga::cospan: Model needs a root System");
        }

        std::size_t pool_bytes_for(const sycl::device& d) const {
            const auto it = cfg.pool_bytes.find(d);
            return it != cfg.pool_bytes.end() ? it->second : cfg.default_pool_bytes;
        }

        static sycl::device resolve_default(const ModelConfig& c) {
            if (c.default_device) {
                if (!c.ctx.contains(*c.default_device))
                    throw std::invalid_argument("naga::cospan: ModelConfig::default_device is not a device of the "
                                                "config's context");
                return *c.default_device;
            }
            try {
                const sycl::device sel{sycl::default_selector_v};
                if (c.ctx.contains(sel)) return sel;
            } catch (const sycl::exception&) {  // no default-selector device: fall through to the first
            }
            return c.ctx.devices().front();
        }

        // Declaration order = construction order; destruction is the reverse:
        // cfg, marks, root, substrates, queue, ctx (§9.4).
        sk::context             ctx;
        sk::queue               queue;
        Substrates              substrates;
        std::unique_ptr<System> root;
        std::vector<MarkRecord> design_vars, constraints, objectives;
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
        if (!s.ready)
            throw std::logic_error(std::string{"naga::cospan: Model::"} + verb + " before setup() has completed");
        return s;
    }

    static detail::IOEntry& entry_of(Impl& s, std::string_view path, IOKind want, const char* verb) {
        detail::IOEntry& e = s.root->entry_at_from_outside(path);
        if (e.kind() != want)
            throw std::invalid_argument(std::string{"naga::cospan: Model::"} + verb + ": '" + std::string{path} +
                                        "' is " + (want == IOKind::input ? "an output" : "an input"));
        return e;
    }

    /// Releases the Impl in the §9.4 order. Does not wait: the user waits before destroying or reassigning a Model.
    void finish() noexcept {
        if (!s_) return;
        s_.reset();
    }

    /// Depth-first in declaration order: a System's own alias entries, then its
    /// children (Components' entries; Systems recursively). A mark rides on its
    /// ROOT entry, so each root is read once, whichever entry reached it first.
    static void collect_marks(const System& sys, Impl& s, std::unordered_set<const detail::IOEntry*>& seen) {
        const auto note = [&](const detail::IOEntry& e) {
            const detail::IOEntry* r = e.root();
            if (!seen.insert(r).second) return;
            if (const auto& m = r->design_var()) s.design_vars.push_back(*m);
            if (const auto& m = r->constraint()) s.constraints.push_back(*m);
            if (const auto& m = r->objective()) s.objectives.push_back(*m);
        };
        for (const auto& e : sys.get_inputs()) note(e.get());
        for (const auto& e : sys.get_outputs()) note(e.get());
        for (const Subsystem* c : sys.children()) {
            if (const auto* sub = dynamic_cast<const System*>(c)) {
                collect_marks(*sub, s, seen);
            } else if (const auto* comp = dynamic_cast<const ComponentBase*>(c)) {
                for (const auto& e : comp->get_inputs()) note(e.get());
                for (const auto& e : comp->get_outputs()) note(e.get());
            }
        }
    }

    std::unique_ptr<Impl> s_;
};

template <class Sys, class... Args>
    requires std::derived_from<Sys, System>
Model create_model(ModelConfig cfg, Args&&... args) {
    return Model{std::move(cfg), std::make_unique<Sys>(std::forward<Args>(args)...)};
}

}  // namespace naga::cospan