#pragma once

// =============================================================================
// naga/cospan/model.hpp — the Model
// =============================================================================
//
// The Model owns one sk::context (EVERY available root device: sk::context{}), one sk::queue (containers), one sk::unordered_queue (kernels and host tasks, ordered
// by events only), one pool per context device (sized by the pool_desc that covers it, else 0 bytes), the root System, and the marks. It drives the
// two setup passes of docs/design-review-2026-10-03.md, section 2, and never touches a Subsystem except through the
// Subsystem verbs:
//   setup:    the root's run_setup, recursive through add_subsystem.
//   finalize: collects the marks (design variables, constraints, objectives) by absolute path, as weak pointers to
//             their instances. Nothing else.
// Setup never allocates, never submits and never compiles. The Model is configured by descriptors
// (model_config.hpp). There is no device gating: any context device may be given to add_subsystem, whether or not a pool_desc
// names it. There is no default device (section 2): every instance takes AdaptiveCpp's default when it is declared, and a device given to add_subsystem overrides a subtree.
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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/diagnostics.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/model_config.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/system.hpp"
#include "naga/cospan/wiring.hpp"

namespace naga::cospan {

/// One mark, by absolute path, collected at the end of setup. The instance is held weakly: a mark on a deleted
/// subsystem's instance expires.
struct Mark {
    std::string                     path;
    std::weak_ptr<detail::Instance> inst;
    MarkRecord                      rec;
};

class Model;

namespace detail {
/// The only host write (section 5), by instance: assigns the initializer to the input's variable and marks it STALE; on an
/// input that has a source it warns and does nothing (unless the variable is part of a cycle). Shared by
/// Model::update_input and InputHandle::update.
inline void update_input_instance(const std::string& path, const Instance& inst, InitSpec init) {
    detail::VarMeta& v = *inst.meta;
    if (detail::first_of_kind(v, IOKind::output) && !v.cycle) {  // a cycle variable keeps an initializer: update_input counts
        std::cerr << "naga::cospan: warning: input '" << path << "' is connected to an output; update_input is never applied\n";
        return;
    }
    v.init = std::move(init);  // later than every declaration made in setup, so it replaces them
    v.mark_stale();
}
}  // namespace detail

/// Model::get_input<DT, Rank>(path): the input at `path`. view() makes a host view (a dense variable), ragged_view() a
/// HostRagged one (a ragged variable, Rank 2; also fine on a dense one: its blocks are the rows). update(init) is
/// update_input. A handle is valid while its Model lives (a view it made is not tied to the Model).
template <DType DT, int Rank>
class InputHandle {
public:
    /// A host view of the variable, made now: it materializes the variable like any holder and throws only when its shape
    /// cannot be resolved yet. Call sync() before reading.
    HostView<DT, Rank> view() const {
        return detail::HostViewAccess::wrap<HostView<DT, Rank>>(detail::make_view_instance(*rt_, *inst_, DT, Rank, /*dense=*/true));
    }
    HostRagged<DT, Rank> ragged_view() const
        requires(Rank == 2)
    {
        return detail::HostViewAccess::wrap<HostRagged<DT, Rank>>(detail::make_view_instance(*rt_, *inst_, DT, Rank, /*dense=*/false));
    }
    /// Model::update_input on this input.
    void update(InitSpec init) const { detail::update_input_instance(path_, *inst_, std::move(init)); }

private:
    friend class Model;
    InputHandle(detail::Runtime* rt, std::shared_ptr<detail::Instance> inst, std::string path)
        : rt_{rt}, inst_{std::move(inst)}, path_{std::move(path)} {}
    detail::Runtime*                  rt_;
    std::shared_ptr<detail::Instance> inst_;
    std::string                       path_;
};

/// Model::get_output<DT, Rank>(path): the output at `path`; it reads only (view / ragged_view, as InputHandle).
template <DType DT, int Rank>
class OutputHandle {
public:
    HostView<DT, Rank> view() const {
        return detail::HostViewAccess::wrap<HostView<DT, Rank>>(detail::make_view_instance(*rt_, *inst_, DT, Rank, /*dense=*/true));
    }
    HostRagged<DT, Rank> ragged_view() const
        requires(Rank == 2)
    {
        return detail::HostViewAccess::wrap<HostRagged<DT, Rank>>(detail::make_view_instance(*rt_, *inst_, DT, Rank, /*dense=*/false));
    }

private:
    friend class Model;
    OutputHandle(detail::Runtime* rt, std::shared_ptr<detail::Instance> inst) : rt_{rt}, inst_{std::move(inst)} {}
    detail::Runtime*                  rt_;
    std::shared_ptr<detail::Instance> inst_;
};

namespace detail {
/// Whether the first of `Args` decays to model_config_desc (false for no arguments).
template <class... Args>
inline constexpr bool first_is_config_v = false;
template <class A, class... Rest>
inline constexpr bool first_is_config_v<A, Rest...> = std::is_same_v<std::remove_cvref_t<A>, model_config_desc>;
}  // namespace detail

/// THE factory: constructs the root System with `args...` and the Model around it. The second form takes a
/// model_config_desc first (selected when the first argument's decayed type is model_config_desc).
template <class Sys, class... Args>
    requires(std::derived_from<Sys, System> && !detail::first_is_config_v<Args...>)
Model create_model(Args&&... args);

template <class Sys, class Cfg, class... Args>
    requires(std::derived_from<Sys, System> && std::is_same_v<std::remove_cvref_t<Cfg>, model_config_desc>)
Model create_model(Cfg&& cfg, Args&&... args);

class Model {
public:
    /// Use create_model. Throws std::invalid_argument for a null root.
    Model(model_config_desc cfg, std::unique_ptr<System> root)
        : s_{std::make_unique<Impl>(std::move(cfg.data_), std::move(root))} {}

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

    // ---- host access (section 5), at any time after setup ------------------------------------
    /// The handles (design review 11.4): get_input reads (view) and writes initializers (update); get_output reads only.
    /// Same template parameters and path rules as the view_* / update_input shorthands below. Naming an output in
    /// get_input, or an input in get_output, throws std::invalid_argument naming the path and its kind.
    template <DType DT, int Rank>
    InputHandle<DT, Rank> get_input(std::string_view path) {
        Impl&                                   s    = ready("get_input");
        const std::shared_ptr<detail::Instance> inst = checked_instance("get_input", path, IOKind::input, DT, Rank);
        return InputHandle<DT, Rank>{&s.rt, inst, std::string{path}};
    }
    template <DType DT, int Rank>
    OutputHandle<DT, Rank> get_output(std::string_view path) {
        Impl&                                   s    = ready("get_output");
        const std::shared_ptr<detail::Instance> inst = checked_instance("get_output", path, IOKind::output, DT, Rank);
        return OutputHandle<DT, Rank>{&s.rt, inst};
    }

    // A host view is an instance of the variable on a host placement. While it lives, every forward pushes one copy to it
    // (each live view costs one copy per forward); a view can be made at any time (it materializes the variable like any
    // holder and fails only when the shape cannot be resolved yet); call sync() before reading it. See host_view.hpp.
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
        detail::update_input_instance(std::string{path}, *inst, std::move(init));
    }

    // ---- retrieval & observation ------------------------------------------------
    System&    root() { return *impl("root").root; }
    Subsystem* get_subsystem(std::string_view path) { return ready("get_subsystem").root->get_subsystem(path); }

    struct MemoryReport {
        /// A snapshot at the time of the call, between runs.
        struct PerDevice {
            std::size_t pool_bytes = 0, pool_bytes_in_use = 0;    ///< slots x node size
            std::size_t os_allocations = 0, os_bytes_in_use = 0;  ///< allocations the pool sent to the OS (cumulative) and the bytes of those still live
        };
        /// The devices whose pool has bytes, or that recorded any allocation (pool or OS route), and no others: a device
        /// without a configured size is a pool of 0 bytes, whose allocations all count as OS allocations, so it appears once
        /// it has had one.
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
            if (p.pool_bytes > 0 || p.os_allocations > 0) r.devices.emplace(d, p);
        }
        return r;
    }

    // ---- marks, collected at the end of setup() -----------------------------------
    std::span<const Mark> design_vars() const { return ready("design_vars").design_vars; }
    std::span<const Mark> constraints() const { return ready("constraints").constraints; }
    std::span<const Mark> objectives() const { return ready("objectives").objectives; }

private:
    struct Impl {
        Impl(detail::ResolvedConfig cfg_in, std::unique_ptr<System> r)
            : ctx{}  // every available root device
            , queue{make_queue(ctx, cfg_in)}
            , uqueue{make_uqueue(ctx, cfg_in)}
            , rt{queue, uqueue, build_pools(ctx, queue, cfg_in)}
            , root{std::move(r)}
            , cfg{std::move(cfg_in)} {
            if (!root) throw std::invalid_argument("naga::cospan: Model needs a root System");
        }

        // The async handler, when given, goes to both queues; otherwise the no-handler constructors.
        static sk::queue make_queue(const sk::context& c, const detail::ResolvedConfig& d) {
            if (d.async_handler) return sk::queue{c, *d.async_handler};
            return sk::queue{c};
        }
        static sk::unordered_queue make_uqueue(const sk::context& c, const detail::ResolvedConfig& d) {
            if (d.async_handler) return sk::unordered_queue{c, *d.async_handler};
            return sk::unordered_queue{c};
        }

        static std::string name_of(const sycl::device& d) { return d.get_info<sycl::info::device::name>(); }

        /// The (bytes, node size) each context device gets: from the pool_desc that covers it, else 0 bytes and 1024.
        /// Throws std::invalid_argument, naming the device, for a device covered twice (by two pool_descs or twice in one
        /// list) or cs::all_devices combined with another pool_desc.
        static std::vector<std::pair<std::size_t, std::size_t>> resolve_pools(const sk::context& ctx, const detail::ResolvedConfig& c) {
            const auto devices = ctx.devices();
            std::vector<std::pair<std::size_t, std::size_t>> out(devices.size(), {std::size_t{0}, std::size_t{1024}});
            std::vector<bool>                                covered(devices.size(), false);
            for (const detail::PoolEntry& e : c.pools) {
                if (e.all && c.pools.size() > 1) {
                    // name a device the other pool_desc covers
                    const detail::PoolEntry* other = nullptr;
                    for (const detail::PoolEntry& o : c.pools)
                        if (&o != &e) {
                            other = &o;
                            break;
                        }
                    const sycl::device d = (other != nullptr && !other->all && !other->devices.empty()) ? other->devices.front() : devices.front();
                    throw std::invalid_argument("naga::cospan: cs::all_devices cannot be combined with another pool_desc: device '" + name_of(d) +
                                                "' would be covered twice");
                }
                const auto cover = [&](std::size_t i) {
                    if (covered[i])
                        throw std::invalid_argument("naga::cospan: device '" + name_of(devices[i]) +
                                                    "' is covered by more than one pool_desc (or twice in one list)");
                    covered[i] = true;
                    out[i]     = {e.bytes, e.node_size};
                };
                if (e.all) {
                    for (std::size_t i = 0; i < devices.size(); ++i) cover(i);
                } else {
                    for (const sycl::device& d : e.devices)
                        for (std::size_t i = 0; i < devices.size(); ++i)
                            if (devices[i] == d) cover(i);
                }
            }
            return out;
        }

        /// One pool per context device, sized by the pool_desc that covers it, else 0 bytes (a zero-byte pool serves
        /// nothing and every allocation falls back and is counted). Built in place: a pool is not copied.
        static std::unordered_map<sycl::device, sk::DevicePool> build_pools(const sk::context& ctx, const sk::queue& q,
                                                                           const detail::ResolvedConfig& c) {
            const auto                                       sizes = resolve_pools(ctx, c);
            std::unordered_map<sycl::device, sk::DevicePool> pools;
            for (std::size_t i = 0; i < ctx.devices().size(); ++i)
                pools.try_emplace(ctx.devices()[i], q, ctx.devices()[i], sk::kind::device, sizes[i].first, sizes[i].second);
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
        detail::ResolvedConfig cfg;
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
    /// with the most instances (section 5), and materialized like any holder (detail::make_view_instance).
    std::shared_ptr<detail::Instance> make_view(const char* verb, std::string_view path, IOKind want, DType dt, int rank, bool dense) {
        Impl&                                   s    = ready(verb);
        const std::shared_ptr<detail::Instance> inst = checked_instance(verb, path, want, dt, rank);
        return detail::make_view_instance(s.rt, *inst, dt, rank, dense);
    }

    /// The instance at `path`, of kind `want`, of this dtype and rank; else a throw naming the path (and its kind).
    std::shared_ptr<detail::Instance> checked_instance(const char* verb, std::string_view path, IOKind want, DType dt, int rank) const {
        Impl&                                   s    = ready(verb);
        const std::shared_ptr<detail::Instance> inst = s.root->instance_at(path);
        if (inst->kind != want)
            throw std::invalid_argument(std::string{"naga::cospan: Model::"} + verb + ": '" + std::string{path} + "' is " +
                                        (want == IOKind::input ? "an output" : "an input"));
        if (!dtype_compatible(inst->meta->dtype, dt) || inst->meta->rank != rank)
            throw std::invalid_argument(std::string{"naga::cospan: Model::"} + verb + ": '" + std::string{path} +
                                        "' was asked for with the wrong dtype or rank");
        return inst;
    }

    /// Releases the Impl in the order the header names. Does not wait.
    void finish() noexcept {
        if (!s_) return;
        s_.reset();
    }

    /// FINALIZE: depth-first in declaration order, a System's own marks and then its live children. Each declared
    /// mark becomes a Mark by absolute path (the declaring System's path, a dot, the declared path; at the root, the
    /// declared path). There is no scope rule (design review, 11.2): whatever is collected at setup stays, even under a
    /// subsystem that later goes out of scope (the mark's instance pointer is weak and just expires with it).
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
    requires(std::derived_from<Sys, System> && !detail::first_is_config_v<Args...>)
Model create_model(Args&&... args) {
    return Model{model_config_desc{}, std::make_unique<Sys>(std::forward<Args>(args)...)};
}

template <class Sys, class Cfg, class... Args>
    requires(std::derived_from<Sys, System> && std::is_same_v<std::remove_cvref_t<Cfg>, model_config_desc>)
Model create_model(Cfg&& cfg, Args&&... args) {
    return Model{std::forward<Cfg>(cfg), std::make_unique<Sys>(std::forward<Args>(args)...)};
}

}  // namespace naga::cospan
