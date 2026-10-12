#pragma once

// =============================================================================
// naga/cospan/model.hpp — the Model (r04)
// =============================================================================
//
// The Model owns one sk::context (EVERY available root device: sk::context{}), one sk::queue (containers), one
// sk::unordered_queue (kernels and host tasks, ordered by events only), one pool per context device (sized by the
// pool_desc that covers it, else 0 bytes), and ONE ModelMeta (model_meta.hpp): the tree of subsystems, the flat name maps,
// the records, the marks. It never touches a Subsystem object except through the Subsystem verbs.
//
// LIFECYCLE (docs/spec/revisions/r04-2026-10-11-flat-model-metadata.md)
//   created    create_model<Root>(...), then model.add_subsystem(...) (adds to the root; before setup() only).
//   editable   after setup() (the root's setup(), depth-first): from main, through get_subsystem handles, wiring, init_input,
//              shape hooks, depends_on, set_device, marks. Views are not allowed yet.
//   finalized  finalize() (the user's, or the first run's): devices, shape-hook placement, tied-output pushes, depends_on.
//              Structural calls throw std::logic_error from now on. run / wait / views / update_input / rearm_subsystem /
//              set_run_policy / clear_run_policy / get_* / list_* / memory_report are legal.
//   A second setup() RESETS the model: a fresh ModelMeta, the root rebuilt, the children added before setup() added again,
//   the root's setup() run again. Handles of the old structure throw std::logic_error.
//
// THE RUN is a tree walk with an explicit stack (no recursion): per node its before-operations and policy `before`, then a
// leaf in scope runs forward(rt) / a system in scope is descended into, then its after-operations and policy `after`.
//
// STATE LIVES ON THE HEAP. The Model is one pointer to an Impl; moving a Model moves the pointer. Inside Impl the members
// are declared so that destruction (the reverse of declaration) runs the ModelMeta first (the subsystem objects, their
// holders, and each device's data, each waiting on its own registered events), then the config, the Runtime (the pools),
// the queues, and the context.
//
// LAYERING. ~Model does not wait; call wait() when "everything I submitted is done" matters.

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
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
#include "naga/cospan/model_meta.hpp"
#include "naga/cospan/run_policy.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/system.hpp"
#include "naga/cospan/wiring.hpp"

namespace naga::cospan {

class Model;

namespace detail {

/// The only host write (section 5), by instance: assigns the initializer to the input's variable and marks it STALE (and
/// pending, so the next run's first pass acts on it); on an input that has a source it warns and does nothing (unless the
/// variable is part of a cycle).
inline void update_input_instance(ModelMeta& m, const std::string& path, const Instance& inst, InitSpec init) {
    VarMeta& v = *inst.meta;
    if (first_of_kind(v, IOKind::output) && !v.cycle) {  // a cycle variable keeps an initializer: update_input counts
        std::cerr << "naga::cospan: warning: input '" << path << "' is connected to an output; update_input is never applied\n";
        return;
    }
    v.init = std::move(init);  // later than every declaration made in setup, so it replaces them
    v.mark_stale();
    m.mark_pending(inst.meta);
}

/// A host view of the variable `inst` belongs to. Views are allowed only after finalize, and only while some holder of the
/// variable is in scope (a variable all of whose holders were dearmed has no data to show).
inline std::shared_ptr<Instance> open_view(ModelMeta& m, Runtime& rt, const Instance& inst, DType dt, int rank, bool dense,
                                           const char* verb) {
    if (m.state != Lifecycle::finalized)
        throw std::logic_error(std::string{"naga::cospan: "} + verb + ": '" + inst.full_path() +
                               "': a host view can be made only after finalize (call finalize(), or run once)");
    bool any_in_scope = false;
    for (const std::shared_ptr<Instance>& i : inst.meta->live_instances())
        if (!i->dearmed) any_in_scope = true;
    if (!any_in_scope)
        throw std::logic_error(std::string{"naga::cospan: "} + verb + ": '" + inst.full_path() +
                               "': no holder of this variable has data (its subsystems are out of scope)");
    return make_view_instance(rt, inst, dt, rank, dense);
}

/// The holder `full` of kind `want`, of this dtype and rank; else a throw naming the path.
inline std::shared_ptr<Instance> checked_holder(ModelMeta& m, const std::string& verb, const std::string& full, IOKind want,
                                                DType dt, int rank) {
    const std::shared_ptr<Instance> inst = m.find_holder(full);
    if (!inst) throw std::invalid_argument("naga::cospan: " + verb + ": no input, output or alias '" + full + "'");
    if (inst->kind != want)
        throw std::invalid_argument("naga::cospan: " + verb + ": '" + full + "' is " + (want == IOKind::input ? "an output" : "an input"));
    if (!dtype_compatible(inst->meta->dtype, dt) || inst->meta->rank != rank)
        throw std::invalid_argument("naga::cospan: " + verb + ": '" + full + "' was asked for with the wrong dtype or rank");
    return inst;
}

/// The handle's metadata, or a throw when the model it came from was reset or destroyed.
inline std::shared_ptr<ModelMeta> lock_meta(const std::weak_ptr<ModelMeta>& w, const char* verb) {
    std::shared_ptr<ModelMeta> m = w.lock();
    if (!m)
        throw std::logic_error(std::string{"naga::cospan: "} + verb +
                               ": this handle is stale (its model was reset by a second setup(), or destroyed)");
    return m;
}
}  // namespace detail

/// Model::get_input<DT, Rank>(path) / SubsystemHandle::get_input: the input at the path. view() makes a host view (a dense
/// variable), ragged_view() a HostRagged one (a ragged variable, Rank 2; also fine on a dense one: its blocks are the rows).
/// update(init) is update_input. A handle is valid while its Model lives and is not reset (a view it made is not tied to
/// the Model); a stale handle throws std::logic_error.
template <DType DT, int Rank>
class InputHandle {
public:
    /// A host view of the variable, made now (after finalize only): it materializes the variable like any holder and throws
    /// only when its shape cannot be resolved yet. Call sync() before reading.
    HostView<DT, Rank> view() const {
        const auto m = detail::lock_meta(meta_, "InputHandle::view");
        return detail::HostViewAccess::wrap<HostView<DT, Rank>>(detail::open_view(*m, *rt_, *inst_, DT, Rank, /*dense=*/true, "view"));
    }
    HostRagged<DT, Rank> ragged_view() const
        requires(Rank == 2)
    {
        const auto m = detail::lock_meta(meta_, "InputHandle::ragged_view");
        return detail::HostViewAccess::wrap<HostRagged<DT, Rank>>(
            detail::open_view(*m, *rt_, *inst_, DT, Rank, /*dense=*/false, "ragged_view"));
    }
    /// Model::update_input on this input.
    void update(InitSpec init) const {
        const auto m = detail::lock_meta(meta_, "InputHandle::update");
        detail::update_input_instance(*m, path_, *inst_, std::move(init));
    }

private:
    friend class SubsystemHandle;
    InputHandle(std::weak_ptr<detail::ModelMeta> meta, detail::Runtime* rt, std::shared_ptr<detail::Instance> inst, std::string path)
        : meta_{std::move(meta)}, rt_{rt}, inst_{std::move(inst)}, path_{std::move(path)} {}
    std::weak_ptr<detail::ModelMeta>  meta_;
    detail::Runtime*                  rt_;
    std::shared_ptr<detail::Instance> inst_;
    std::string                       path_;
};

/// The output at a path; it reads only (view / ragged_view, as InputHandle).
template <DType DT, int Rank>
class OutputHandle {
public:
    HostView<DT, Rank> view() const {
        const auto m = detail::lock_meta(meta_, "OutputHandle::view");
        return detail::HostViewAccess::wrap<HostView<DT, Rank>>(detail::open_view(*m, *rt_, *inst_, DT, Rank, /*dense=*/true, "view"));
    }
    HostRagged<DT, Rank> ragged_view() const
        requires(Rank == 2)
    {
        const auto m = detail::lock_meta(meta_, "OutputHandle::ragged_view");
        return detail::HostViewAccess::wrap<HostRagged<DT, Rank>>(
            detail::open_view(*m, *rt_, *inst_, DT, Rank, /*dense=*/false, "ragged_view"));
    }

private:
    friend class SubsystemHandle;
    OutputHandle(std::weak_ptr<detail::ModelMeta> meta, detail::Runtime* rt, std::shared_ptr<detail::Instance> inst)
        : meta_{std::move(meta)}, rt_{rt}, inst_{std::move(inst)} {}
    std::weak_ptr<detail::ModelMeta>  meta_;
    detail::Runtime*                  rt_;
    std::shared_ptr<detail::Instance> inst_;
};

/// A subsystem seen from main (Model::get_subsystem(path); the root is the path ""). It holds the model's metadata weakly
/// and a path: nothing in it points into the metadata, so it throws std::logic_error once its model was reset.
///   Every handle: get_input / get_output (by name relative to the subsystem), the view calls (after finalize), update_input,
///                 list_inputs / list_outputs (over the subtree), set_device.
///   System handles add the declaration calls (connect, promote, tie_inputs, tie_outputs, init_input, shape_hook, depends_on,
///   the marks: until finalize) and set_run_policy / clear_run_policy. A leaf handle refuses them (std::invalid_argument).
class SubsystemHandle {
public:
    /// The full resolved name ("" for the root).
    const std::string& path() const noexcept { return path_; }
    bool               is_leaf() const {
        const auto m = detail::lock_meta(meta_, "is_leaf");
        return node(*m).kind == detail::NodeKind::leaf;
    }

    // ---- inspection ----------------------------------------------------------------
    template <DType DT, int Rank>
    InputHandle<DT, Rank> get_input(std::string_view name) const {
        const auto        m    = detail::lock_meta(meta_, "get_input");
        const std::string full = detail::ModelMeta::child_name(path_, name);
        return InputHandle<DT, Rank>{meta_, rt_, detail::checked_holder(*m, "get_input", full, IOKind::input, DT, Rank), full};
    }
    template <DType DT, int Rank>
    OutputHandle<DT, Rank> get_output(std::string_view name) const {
        const auto        m    = detail::lock_meta(meta_, "get_output");
        const std::string full = detail::ModelMeta::child_name(path_, name);
        return OutputHandle<DT, Rank>{meta_, rt_, detail::checked_holder(*m, "get_output", full, IOKind::output, DT, Rank)};
    }

    // A host view is an instance of the variable on a host placement. While it lives, every run pushes one copy to it (each
    // live view costs one copy per run); a view can be made after finalize (it materializes the variable like any holder and
    // fails only when the shape cannot be resolved yet); call sync() before reading it. See host_view.hpp.
    template <DType DT, int Rank>
    HostView<DT, Rank> view_input(std::string_view name) const {
        return view_of<HostView<DT, Rank>, DT, Rank>("view_input", name, IOKind::input, true);
    }
    template <DType DT, int Rank>
    HostView<DT, Rank> view_output(std::string_view name) const {
        return view_of<HostView<DT, Rank>, DT, Rank>("view_output", name, IOKind::output, true);
    }
    /// Views of a ragged variable (also of a dense one: its blocks are the rows).
    template <DType DT, int Rank>
    HostRagged<DT, Rank> view_ragged_input(std::string_view name) const {
        return view_of<HostRagged<DT, Rank>, DT, Rank>("view_ragged_input", name, IOKind::input, false);
    }
    template <DType DT, int Rank>
    HostRagged<DT, Rank> view_ragged_output(std::string_view name) const {
        return view_of<HostRagged<DT, Rank>, DT, Rank>("view_ragged_output", name, IOKind::output, false);
    }

    /// The only host write (section 5): assigns an initializer to the input's variable and marks it STALE. It queues NO
    /// work now. At the next run, the start-of-run pass (or the first holder that touches the variable) acts on it: not
    /// materialized, it materializes and initializes; materialized, it initializes IN PLACE (waiting on the previous run's
    /// readers, and recorded as a write). Several updates before a run collapse to the last one, and a variable nothing
    /// touches is never initialized. A host view synced before the next run still shows the OLD values. If the input has a
    /// source (it is connected to an output) a warning is printed and the update is never applied. Naming an output throws
    /// std::invalid_argument.
    void update_input(std::string_view name, InitSpec init) const {
        const auto        m    = detail::lock_meta(meta_, "update_input");
        const std::string full = detail::ModelMeta::child_name(path_, name);
        const auto        inst = m->find_holder(full);
        if (!inst) throw std::invalid_argument("naga::cospan: update_input: no input, output or alias '" + full + "'");
        if (inst->kind != IOKind::input)
            throw std::invalid_argument("naga::cospan: Model::update_input: '" + full + "' is an output; outputs are never initialized");
        detail::update_input_instance(*m, full, *inst, std::move(init));
    }

    /// The full names of the inputs (outputs) held under this subsystem: its ports and aliases and those of everything below
    /// it, in declaration order.
    std::vector<std::string> list_inputs() const { return list(IOKind::input); }
    std::vector<std::string> list_outputs() const { return list(IOKind::output); }

    // ---- placement (any handle; legal until finalize) ---------------------------------
    /// Places this subsystem's holders and everything under it on `d`. The OUTER call wins (resolved at finalize).
    void set_device(const sycl::device& d) const {
        const auto m = detail::lock_meta(meta_, "set_device");
        m->require_declaring("set_device", describe());
        detail::Node& n = node(*m);
        n.device_override = d;
        detail::apply_device_subtree(n, d);
    }

    // ---- system handles: declaration calls (until finalize) ---------------------------
    void connect(std::string_view src, std::string_view dst) const {
        const auto m = detail::lock_meta(meta_, "connect");
        sys(*m, "connect").connect(src, dst);
    }
    template <DType DT, int Rank>
    void promote(std::string_view child_path, std::string promoted_name) const {
        const auto m = detail::lock_meta(meta_, "promote");
        sys(*m, "promote").template promote<DT, Rank>(child_path, std::move(promoted_name));
    }
    template <DType DT, int Rank>
    void tie_inputs(std::initializer_list<std::string_view> members, std::string name, ShapeSpec shape = {},
                    std::optional<sycl::device> device = {}) const {
        const auto m = detail::lock_meta(meta_, "tie_inputs");
        sys(*m, "tie_inputs").template tie_inputs<DT, Rank>(members, std::move(name), std::move(shape), std::move(device));
    }
    template <DType DT, int Rank>
    void tie_outputs(std::initializer_list<std::string_view> members, std::string name, ShapeSpec shape = {},
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic, std::optional<sycl::device> device = {}) const {
        const auto m = detail::lock_meta(meta_, "tie_outputs");
        sys(*m, "tie_outputs").template tie_outputs<DT, Rank>(members, std::move(name), std::move(shape), policy, std::move(device));
    }
    template <HookDescriptor... D>
    void shape_hook(std::string_view target, D... descriptors) const {
        const auto m = detail::lock_meta(meta_, "shape_hook");
        sys(*m, "shape_hook").shape_hook(target, std::move(descriptors)...);
    }
    void init_input(std::string_view target, InitSpec init) const {
        const auto m = detail::lock_meta(meta_, "init_input");
        sys(*m, "init_input").init_input(target, std::move(init));
    }
    void depends_on(std::string_view b, std::string_view a) const {
        const auto m = detail::lock_meta(meta_, "depends_on");
        sys(*m, "depends_on").depends_on(b, a);
    }
    /// The subsystem at `rel` (relative to this System) goes to `d`.
    void set_device(std::string_view rel, const sycl::device& d) const {
        const auto m = detail::lock_meta(meta_, "set_device");
        sys(*m, "set_device").set_device(rel, d);
    }
    void add_design_var(std::string_view p, Bounds b = {}, Scaling s = {}) const {
        const auto m = detail::lock_meta(meta_, "add_design_var");
        sys(*m, "add_design_var").add_design_var(p, std::move(b), std::move(s));
    }
    void add_constraint(std::string_view p, Bounds b = {}, std::optional<std::vector<double>> equals = {}, Scaling s = {}) const {
        const auto m = detail::lock_meta(meta_, "add_constraint");
        sys(*m, "add_constraint").add_constraint(p, std::move(b), std::move(equals), std::move(s));
    }
    void add_objective(std::string_view p, Scaling s = {}) const {
        const auto m = detail::lock_meta(meta_, "add_objective");
        sys(*m, "add_objective").add_objective(p, std::move(s));
    }

    // ---- run policies (system handles; legal at any time after the model's setup) ---------
    /// The run policy of the node at `rel` (relative to this System): see System::set_run_policy.
    void set_run_policy(std::string_view rel, const property_list& policy = {}) const {
        const auto m = detail::lock_meta(meta_, "set_run_policy");
        sys(*m, "set_run_policy").set_run_policy(rel, policy);
    }
    void clear_run_policy(std::string_view rel) const {
        const auto m = detail::lock_meta(meta_, "clear_run_policy");
        sys(*m, "clear_run_policy").clear_run_policy(rel);
    }

private:
    friend class Model;
    SubsystemHandle(std::weak_ptr<detail::ModelMeta> meta, detail::Runtime* rt, std::string path)
        : meta_{std::move(meta)}, rt_{rt}, path_{std::move(path)} {}

    std::string describe() const { return path_.empty() ? std::string{"the root System"} : "subsystem '" + path_ + "'"; }

    detail::Node& node(detail::ModelMeta& m) const {
        detail::Node* n = m.find_node(path_);
        if (n == nullptr) throw std::logic_error("naga::cospan: " + describe() + " no longer exists in its model");
        return *n;
    }
    /// The System of a system node (a leaf has none: aliases belong to systems).
    System& sys(detail::ModelMeta& m, const char* verb) const {
        detail::Node& n = node(m);
        if (n.kind != detail::NodeKind::system || n.sys == nullptr)
            throw std::invalid_argument(std::string{"naga::cospan: "} + describe() + ": " + verb +
                                        ": a leaf (a component or a fused group) takes no declaration calls: they belong to systems");
        return *n.sys;
    }

    template <class View, DType DT, int Rank>
    View view_of(const char* verb, std::string_view name, IOKind want, bool dense) const {
        const auto        m    = detail::lock_meta(meta_, verb);
        const std::string full = detail::ModelMeta::child_name(path_, name);
        const auto        inst = detail::checked_holder(*m, verb, full, want, DT, Rank);
        return detail::HostViewAccess::wrap<View>(detail::open_view(*m, *rt_, *inst, DT, Rank, dense, verb));
    }

    std::vector<std::string> list(IOKind kind) const {
        const auto               m = detail::lock_meta(meta_, "list");
        std::vector<std::string> out;
        std::vector<detail::Node*> stack{&node(*m)};
        while (!stack.empty()) {
            detail::Node* n = stack.back();
            stack.pop_back();
            for (const std::weak_ptr<detail::Instance>& w : n->holders) {
                const std::shared_ptr<detail::Instance> inst = w.lock();
                if (inst && inst->kind == kind) out.push_back(inst->full_path());
            }
            for (auto it = n->children.rbegin(); it != n->children.rend(); ++it) stack.push_back(*it);  // declaration order
        }
        return out;
    }

    std::weak_ptr<detail::ModelMeta> meta_;
    detail::Runtime*                 rt_;
    std::string                      path_;
};

namespace detail {
/// Whether the first of `Args` decays to model_config_desc (false for no arguments).
template <class... Args>
inline constexpr bool first_is_config_v = false;
template <class A, class... Rest>
inline constexpr bool first_is_config_v<A, Rest...> = std::is_same_v<std::remove_cvref_t<A>, model_config_desc>;
}  // namespace detail

/// THE factory: the root System of type Sys, built from `args...` (kept, so a second setup() can build it again), and the
/// Model around it. The second form takes a model_config_desc first (selected when the first argument's decayed type is
/// model_config_desc). `create_model<cs::System>()` is a plain root: build it with model.add_subsystem before setup().
template <class Sys, class... Args>
    requires(std::derived_from<Sys, System> && !detail::first_is_config_v<Args...>)
Model create_model(Args&&... args);

template <class Sys, class Cfg, class... Args>
    requires(std::derived_from<Sys, System> && std::is_same_v<std::remove_cvref_t<Cfg>, model_config_desc>)
Model create_model(Cfg&& cfg, Args&&... args);

class Model {
public:
    using RootFactory = std::function<std::unique_ptr<System>()>;

    /// Use create_model. Throws std::invalid_argument for a missing root.
    Model(model_config_desc cfg, RootFactory make_root) : s_{std::make_unique<Impl>(std::move(cfg.data_), std::move(make_root))} {}

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

    // ---- building from main (created state) ------------------------------------------
    /// Adds a child to the ROOT System, and runs its setup() before this returns. Legal before setup() only (a
    /// std::logic_error after). The children added here come before the ones the root's own setup() adds, and a second
    /// setup() adds them again.
    template <class T, class... A>
    void add_subsystem(SubsystemDesc<T, A...> desc) {
        add_to_root(property_list{}, std::nullopt, std::move(desc));
    }
    template <class T, class... A>
    void add_subsystem(const property_list& props, SubsystemDesc<T, A...> desc) {
        add_to_root(props, std::nullopt, std::move(desc));
    }
    template <class T, class... A>
    void add_subsystem(const sycl::device& dev, SubsystemDesc<T, A...> desc) {
        add_to_root(property_list{}, dev, std::move(desc));
    }
    template <class T, class... A>
    void add_subsystem(const property_list& props, const sycl::device& dev, SubsystemDesc<T, A...> desc) {
        add_to_root(props, dev, std::move(desc));
    }

    // ---- lifecycle -------------------------------------------------------------
    /// SETUP: the root's setup(), depth-first (the state becomes editable). Nothing is allocated, submitted or compiled.
    /// A second call RESETS the model (see the header): handles taken before it throw std::logic_error.
    void setup() {
        Impl& s = impl("setup");
        if (s.setup_done) s.build();  // reset: a fresh ModelMeta, a fresh root, the pre-setup additions again
        s.setup_done = true;
        try {
            detail::Node& root = *s.meta->root;
            root.sub->run_setup();
        } catch (...) {
            s.setup_done = false;  // part-way: not usable; setup() again starts over
            throw;
        }
        s.meta->state = detail::Lifecycle::editable;
        s.rt.forward  = 0;
    }

    /// FINALIZE: places everything that needs the whole structure (see finalize_impl). Optional: the first run() does it.
    /// Runs once: a second call throws std::logic_error.
    void finalize() {
        Impl& s = ready("finalize");
        if (s.meta->state == detail::Lifecycle::finalized)
            throw std::logic_error("naga::cospan: Model::finalize() runs once; the model is already finalized");
        finalize_impl(s);
    }

    /// ENQUEUES one run and returns: finalize if needed, the start-of-run pass over the pending variables, then the walk of the
    /// tree. It does not wait: runs stream with no artificial synchronization, so call wait() or sync a host view when you
    /// need the results (section 5).
    void run() {
        Impl& s = ready("run");
        if (s.meta->state != detail::Lifecycle::finalized) finalize_impl(s);
        ++s.rt.forward;
        detail::materialize_pending(*s.meta, s.rt);
        walk(*s.meta, s.rt);
    }

    /// queue.wait_and_throw() on both queues. OPTIONAL: for "everything I submitted is done" and to surface
    /// asynchronous errors (wait_and_throw) from work nobody synced (section 5). Repeated runs stream without it.
    void wait() {
        diag::count_queue_waits();
        Impl& s = impl("wait");
        s.queue.wait_and_throw();
        s.uqueue.wait_and_throw();
    }

    // ---- handles and host access (after setup) -------------------------------------------
    /// The handle of the subsystem (a system or a leaf) at the full resolved name `path` ("" is the root). A stage of a fused
    /// group is not a subsystem of the tree: it throws std::invalid_argument (its ports are reached by full name through
    /// the group's handle, or the root's).
    SubsystemHandle get_subsystem(std::string_view path) {
        Impl& s = ready("get_subsystem");
        if (s.meta->find_node(path) == nullptr) {
            if (s.meta->stage_names.contains(std::string{path}))
                throw std::invalid_argument("naga::cospan: Model::get_subsystem: '" + std::string{path} +
                                            "' is a stage of a fused group, not a subsystem of the tree: reach its ports through the group");
            throw std::invalid_argument("naga::cospan: Model::get_subsystem: no subsystem '" + std::string{path} + "'");
        }
        return SubsystemHandle{s.meta, &s.rt, std::string{path}};
    }

    /// The model's own calls are the root handle's.
    template <DType DT, int Rank>
    InputHandle<DT, Rank> get_input(std::string_view path) {
        return root_handle("get_input").template get_input<DT, Rank>(path);
    }
    template <DType DT, int Rank>
    OutputHandle<DT, Rank> get_output(std::string_view path) {
        return root_handle("get_output").template get_output<DT, Rank>(path);
    }
    template <DType DT, int Rank>
    HostView<DT, Rank> view_input(std::string_view path) {
        return root_handle("view_input").template view_input<DT, Rank>(path);
    }
    template <DType DT, int Rank>
    HostView<DT, Rank> view_output(std::string_view path) {
        return root_handle("view_output").template view_output<DT, Rank>(path);
    }
    template <DType DT, int Rank>
    HostRagged<DT, Rank> view_ragged_input(std::string_view path) {
        return root_handle("view_ragged_input").template view_ragged_input<DT, Rank>(path);
    }
    template <DType DT, int Rank>
    HostRagged<DT, Rank> view_ragged_output(std::string_view path) {
        return root_handle("view_ragged_output").template view_ragged_output<DT, Rank>(path);
    }
    /// See SubsystemHandle::update_input.
    void update_input(std::string_view path, InitSpec init) { root_handle("update_input").update_input(path, std::move(init)); }

    std::vector<std::string> list_inputs() { return root_handle("list_inputs").list_inputs(); }
    std::vector<std::string> list_outputs() { return root_handle("list_outputs").list_outputs(); }

    /// Run policies, from main between runs (root-relative paths): see System::set_run_policy.
    void set_run_policy(std::string_view path, const property_list& policy = {}) {
        root_handle("set_run_policy").set_run_policy(path, policy);
    }
    void clear_run_policy(std::string_view path) { root_handle("clear_run_policy").clear_run_policy(path); }

    /// Puts the subsystem at `path` (and everything under it) back in scope, counters reset, between runs: for every leaf in
    /// its range the leaf is rearmed and its policy's rearm called. On a subsystem already in scope it resets the counters.
    void rearm_subsystem(std::string_view path) {
        Impl& s = ready("rearm_subsystem");
        if (s.meta->state != detail::Lifecycle::finalized)
            throw std::logic_error("naga::cospan: Model::rearm_subsystem: legal after finalize (between runs) only");
        detail::Node* n = s.meta->find_node(path);
        if (n == nullptr) throw std::invalid_argument("naga::cospan: Model::rearm_subsystem: no subsystem '" + std::string{path} + "'");
        detail::rearm_subtree(*s.meta, *n);
    }

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

    // ---- marks, by absolute path, collected as they are declared ------------------------
    std::span<const Mark> design_vars() const { return ready("design_vars").meta->design_vars; }
    std::span<const Mark> constraints() const { return ready("constraints").meta->constraints; }
    std::span<const Mark> objectives() const { return ready("objectives").meta->objectives; }

private:
    struct Impl {
        Impl(detail::ResolvedConfig cfg_in, RootFactory mk)
            : ctx{}  // every available root device
            , queue{make_queue(ctx, cfg_in)}
            , uqueue{make_uqueue(ctx, cfg_in)}
            , rt{queue, uqueue, build_pools(ctx, queue, cfg_in)}
            , cfg{std::move(cfg_in)}
            , make_root{std::move(mk)} {
            if (!make_root) throw std::invalid_argument("naga::cospan: Model needs a root System");
            build();
        }

        /// A fresh ModelMeta with a fresh root, and the children added before setup() added again (a reset starts here).
        /// The old meta goes first: its data is freed while the pools are alive.
        void build() {
            meta.reset();
            auto fresh = std::make_shared<detail::ModelMeta>();
            auto r     = make_root();
            if (!r) throw std::invalid_argument("naga::cospan: Model needs a root System");
            System* sys = r.get();
            fresh->register_node(nullptr, "", std::move(r), detail::NodeKind::system, sys, "the model");
            meta       = std::move(fresh);
            setup_done = false;
            for (const std::function<void(System&)>& add : adds) add(*sys);
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

        // Declaration order = construction order; destruction is the reverse: the ModelMeta first (the subsystem objects,
        // their holders and the marks, and each device's data), then adds and the factory, the config, the Runtime (pools), the
        // queues, the context.
        sk::context         ctx;
        sk::queue           queue;
        sk::unordered_queue uqueue;
        detail::Runtime     rt;  ///< holds the pools by value
        detail::ResolvedConfig cfg;
        RootFactory         make_root;
        std::vector<std::function<void(System&)>> adds;  ///< model.add_subsystem calls, in order, for a reset
        bool                setup_done = false;
        std::shared_ptr<detail::ModelMeta> meta;  ///< LAST: destroyed first
    };

    /// The state, or a throw for a moved-from Model.
    Impl& impl(const char* verb) const {
        if (!s_) throw std::logic_error(std::string{"naga::cospan: Model::"} + verb + " on a moved-from Model");
        return *s_;
    }
    /// The state once setup() has completed; else a throw.
    Impl& ready(const char* verb) const {
        Impl& s = impl(verb);
        if (!s.setup_done || s.meta->state == detail::Lifecycle::created)
            throw std::logic_error(std::string{"naga::cospan: Model::"} + verb + " before setup() has completed");
        return s;
    }
    SubsystemHandle root_handle(const char* verb) const {
        Impl& s = ready(verb);
        return SubsystemHandle{s.meta, &s.rt, std::string{}};
    }

    /// The root-level add_subsystem: built and registered now (its setup() runs), and remembered for a reset.
    template <class T, class... A>
    void add_to_root(const property_list& props, std::optional<sycl::device> dev, SubsystemDesc<T, A...> desc) {
        Impl& s = impl("add_subsystem");
        if (s.setup_done)
            throw std::logic_error("naga::cospan: Model::add_subsystem is legal before setup() only (use setup() of a System, or a handle's declarations)");
        const std::function<void(System&)> add = [props, dev, desc](System& root) mutable {
            auto copy = desc;  // each run of the call (and each reset) builds from fresh copies of the arguments
            add_child(root, copy.name, props, dev, detail::make_subsystem<T>(copy.args));
        };
        add(*s.meta->root->sys);
        s.adds.push_back(add);
    }
    static void add_child(System& root, std::string name, const property_list& props, std::optional<sycl::device> dev,
                          std::unique_ptr<Subsystem> child) {
        root.add_child_unchecked(std::move(name), props, std::move(dev), std::move(child));
    }

    /// Releases the Impl in the order the header names. Does not wait.
    void finish() noexcept {
        if (!s_) return;
        s_.reset();
    }

    // ---- finalize ----------------------------------------------------------------------
    /// In order: (1) devices, one tree walk, the outermost set_device wins; (2) shape-hook placement; (3) one tied-output push
    /// per non-absorbed tie_outputs alias, with the one-device check; (4) depends_on; (5) the shape definitions are fixed (the
    /// state change is what fixes them: structural calls throw from now on); (6) every variable goes into the pending set; (7)
    /// state = finalized. A throw leaves the state editable with the placed operations removed again, so finalize can be
    /// called again after the fault is repaired.
    static void finalize_impl(Impl& s) {
        detail::ModelMeta& m = *s.meta;
        for (auto& [name, n] : m.nodes) {  // a retry starts clean
            n->before.clear();
            n->after.clear();
        }
        try {
            place_devices(m);
            place_shape_hooks(m);
            place_tied_pushes(m);
            place_depends(m);
        } catch (...) {
            for (auto& [name, n] : m.nodes) {
                n->before.clear();
                n->after.clear();
            }
            throw;
        }
        fill_pending(m);
        m.state = detail::Lifecycle::finalized;
    }

    /// (1) One walk from the root; a node with a device override places its whole subtree and the walk does not go inside it,
    /// so the outermost override wins.
    static void place_devices(detail::ModelMeta& m) {
        std::vector<detail::Node*> stack{m.root};
        while (!stack.empty()) {
            detail::Node* n = stack.back();
            stack.pop_back();
            if (n->device_override) {
                detail::apply_device_subtree(*n, *n->device_override);
                continue;
            }
            for (detail::Node* c : n->children) stack.push_back(c);
        }
    }

    /// (2) A hook runs directly after the last producer of the sources it names; with no traceable source, just before the
    /// first holder of its target. A source produced at or after the first holder of the target is an error.
    static void place_shape_hooks(detail::ModelMeta& m) {
        for (detail::HookRecord& h : m.hooks) {
            const std::shared_ptr<detail::Instance> target = h.target_inst.lock();
            if (!target) continue;
            detail::Node* first = m.first_holder(*target->meta);
            detail::Node* last  = nullptr;
            std::string   last_src;
            for (std::size_t i = 0; i < h.source_insts.size(); ++i) {
                const std::shared_ptr<detail::Instance> src = h.source_insts[i].lock();
                if (!src) continue;
                detail::Node* p = m.last_producer(*src->meta);
                if (p != nullptr && (last == nullptr || p->ordinal > last->ordinal)) {
                    last     = p;
                    last_src = h.sources[i];
                }
            }
            detail::HookBase* step = h.step.get();
            detail::Operation op{"shape_hook", [step](detail::Runtime& rt) { step->forward(rt); }};
            if (last != nullptr) {
                if (first != nullptr && last->ordinal >= first->ordinal)
                    throw std::invalid_argument("naga::cospan: finalize: the shape hook on '" + h.target + "': its source '" + last_src +
                                                "' is produced by '" + last->name + "', which does not run before '" + first->name +
                                                "', the first holder of the target: the hook cannot set the shape in time");
                last->after.push_back(std::move(op));
            } else if (first != nullptr) {
                first->before.push_back(std::move(op));
            }
        }
    }

    /// (3) One push per tied-output chain: through the outermost alias, after the last leaf that writes a member, for the
    /// members in scope. The members (and the alias) must be on one device.
    static void place_tied_pushes(detail::ModelMeta& m) {
        for (const detail::TieOutRecord& t : m.tie_outs) {
            const std::shared_ptr<detail::Instance> alias = t.alias;
            if (!alias || alias->absorbed) continue;
            std::vector<detail::Node*> leaves;
            for (const std::shared_ptr<detail::Instance>& i : alias->meta->live_instances()) {
                if (i->kind != IOKind::output || i->role != detail::Role::port) continue;
                if (!(i->device == alias->device))
                    throw std::invalid_argument("naga::cospan: finalize: tie_outputs '" + alias->full_path() + "': member '" + i->full_path() +
                                                "' is on device '" + i->device.get_info<sycl::info::device::name>() +
                                                "' but the tie is on '" + alias->device.get_info<sycl::info::device::name>() +
                                                "': a tie's members live on one device (set_device the owner, or move the whole tie)");
                detail::Node* n = m.node_of(*i);
                if (n != nullptr && n->kind == detail::NodeKind::leaf && std::find(leaves.begin(), leaves.end(), n) == leaves.end())
                    leaves.push_back(n);
            }
            if (leaves.empty()) continue;
            detail::Node* host = *std::max_element(leaves.begin(), leaves.end(),
                                                   [](const detail::Node* a, const detail::Node* b) { return a->ordinal < b->ordinal; });
            std::weak_ptr<detail::Instance> wa = alias;
            host->after.push_back(detail::Operation{"tied_push", [wa, leaves, host](detail::Runtime&) {
                bool any = false;
                for (const detail::Node* l : leaves) any = any || detail::ModelMeta::effective_scope(*l);
                if (!any) return;  // no member runs this turn: nothing was written
                const std::shared_ptr<detail::Instance> a = wa.lock();
                if (!a || !a->data || !a->data->bound()) return;
                for (const auto& entry : a->meta->data) {
                    const std::shared_ptr<detail::DeviceDataBase> dst = entry.second.lock();
                    if (!dst || dst == a->data || !dst->bound()) continue;  // a holder not materialized yet fills itself
                    host->extra_events.push_back(dst->copy_from(*a->data, std::span<const sycl::event>{}));
                }
            }});
        }
    }

    /// The leaf nodes under `n`, in ordinal order.
    static std::vector<detail::Node*> leaves_under(detail::Node& n) {
        std::vector<detail::Node*> out, stack{&n};
        while (!stack.empty()) {
            detail::Node* x = stack.back();
            stack.pop_back();
            if (x->kind == detail::NodeKind::leaf) out.push_back(x);
            for (detail::Node* c : x->children) stack.push_back(c);
        }
        std::sort(out.begin(), out.end(), [](const detail::Node* a, const detail::Node* b) { return a->ordinal < b->ordinal; });
        return out;
    }

    /// (4) depends_on(b, a): after a's last leaf, the events of every leaf of a go to every leaf of b that runs.
    static void place_depends(detail::ModelMeta& m) {
        for (const detail::DependsRecord& e : m.depends) {
            detail::Node* na = m.find_node(e.a);
            detail::Node* nb = m.find_node(e.b);
            if (na == nullptr || nb == nullptr) continue;
            const std::vector<detail::Node*> from = leaves_under(*na);
            const std::vector<detail::Node*> to   = leaves_under(*nb);
            if (from.empty() || to.empty()) continue;
            detail::Node* host = from.back();
            host->after.push_back(detail::Operation{"depends_on", [from, to](detail::Runtime&) {
                std::vector<sycl::event> evs;
                for (const detail::Node* l : from) {
                    const std::vector<sycl::event> own = l->sub->events();
                    evs.insert(evs.end(), own.begin(), own.end());
                    evs.insert(evs.end(), l->extra_events.begin(), l->extra_events.end());
                }
                if (evs.empty()) return;
                for (detail::Node* l : to)
                    if (detail::ModelMeta::effective_scope(*l)) l->sub->add_dependencies(std::span<const sycl::event>{evs});
            }});
        }
    }

    /// (6) Every variable that has a holder, in tree order.
    static void fill_pending(detail::ModelMeta& m) {
        std::vector<detail::Node*> stack{m.root};
        while (!stack.empty()) {
            detail::Node* n = stack.back();
            stack.pop_back();
            for (const std::weak_ptr<detail::Instance>& w : n->holders)
                if (const std::shared_ptr<detail::Instance> inst = w.lock()) m.mark_pending(inst->meta);
            for (auto it = n->children.rbegin(); it != n->children.rend(); ++it) stack.push_back(*it);
        }
    }

    // ---- the run -------------------------------------------------------------------------
    /// The iterative tree walk (explicit stack). Per node: its before-operations, its policy's before; then a leaf in scope
    /// runs forward(rt) and a system in scope is descended into (an out-of-scope system's subtree is skipped); then its
    /// after-operations and its policy's after.
    static void walk(detail::ModelMeta& m, detail::Runtime& rt) {
        struct Frame {
            detail::Node* n;
            std::size_t   next;
        };
        std::vector<Frame> stack;
        // Returns whether the node's children are to be walked.
        const auto enter = [&](detail::Node* n) -> bool {
            n->extra_events.clear();
            for (const detail::Operation& op : n->before) op.fn(rt);
            if (n->policy) n->policy->before(m, *n);
            if (n->kind == detail::NodeKind::leaf) {
                if (n->in_scope) n->sub->forward(rt);
                return false;
            }
            return n->in_scope && !n->children.empty();
        };
        const auto leave = [&](detail::Node* n) {
            for (const detail::Operation& op : n->after) op.fn(rt);
            if (n->policy) n->policy->after(m, *n);
        };
        const auto visit = [&](detail::Node* n) {
            if (enter(n)) stack.push_back(Frame{n, 0});
            else leave(n);
        };
        visit(m.root);
        while (!stack.empty()) {
            Frame& f = stack.back();
            if (f.next < f.n->children.size()) {
                detail::Node* c = f.n->children[f.next++];
                visit(c);  // may push: f is not used again
            } else {
                detail::Node* n = f.n;
                stack.pop_back();
                leave(n);
            }
        }
    }

    std::unique_ptr<Impl> s_;
};

template <class Sys, class... Args>
    requires(std::derived_from<Sys, System> && !detail::first_is_config_v<Args...>)
Model create_model(Args&&... args) {
    return create_model<Sys>(model_config_desc{}, std::forward<Args>(args)...);
}

template <class Sys, class Cfg, class... Args>
    requires(std::derived_from<Sys, System> && std::is_same_v<std::remove_cvref_t<Cfg>, model_config_desc>)
Model create_model(Cfg&& cfg, Args&&... args) {
    // The arguments are copied once and kept: a second setup() builds the root again from them.
    auto kept = std::make_shared<std::tuple<std::decay_t<Args>...>>(std::forward<Args>(args)...);
    return Model{std::forward<Cfg>(cfg), [kept]() -> std::unique_ptr<System> {
                     return std::apply([](const auto&... a) -> std::unique_ptr<System> { return std::make_unique<Sys>(a...); }, *kept);
                 }};
}

}  // namespace naga::cospan
