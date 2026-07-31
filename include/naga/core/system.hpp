#pragma once

// =============================================================================
// naga/core/system.hpp — System: composite composable
// =============================================================================
//
// System is a Composable that owns child composables (subsystems) and delegates
// execution order to Context's topological sort during finalize().
//
// Systems are transparent in the execution flow graph — they have no I/O
// surface and are not nodes in the topological sort. Their children that
// ARE components (non-empty I/O metadata) appear as nodes.
//
// Construction API (called during declare):
//   add_subsystem(ctx, name, shared_ptr<Composable>, register_flow=true)
//   connect(ctx, source_local, target_local)
//   promote_input<Dim>(ctx, local_name, range, entries)
//   promote_output<Dim>(ctx, local_name, range, entry)
//   assign_objective(ctx, local_name)
//
// Lifecycle:
//   declare() — pure virtual from Composable. Each concrete System subclass
//               must override declare() to call add_subsystem(), connect(),
//               and assign_objective() as needed.
//               (Child name-assignment, declaration, and registration happens
//                in add_subsystem.)
//
//   Systems are not executable — they only declare(). Their child Components
//   are what the context executor drives (forward/backward/init/reset). Systems
//   do not implement or inherit those methods.

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "naga/core/composable.hpp"
#include "naga/core/context.hpp"

namespace naga::core {

class System : public Composable {
public:
    System() = default;
    ~System() override = default;

    // Construction API (called during declare)
    void add_subsystem(Context& ctx,
                       std::string child_name,
                       std::shared_ptr<Composable> component,
                       bool register_flow = true);
    void connect(Context& ctx, std::string source_local,
                 std::string target_local);
    void disconnect(Context& ctx, std::string source_local,
                    std::string target_local);
    template <int Dim>
    void promote_input(Context& ctx, std::string local_name,
                       const sycl::range<Dim>& range,
                       std::vector<std::string> entries);
    template <int Dim>
    void promote_output(Context& ctx, std::string local_name,
                        const sycl::range<Dim>& range,
                        std::string entry);
    void assign_objective(Context& ctx, std::string local_name);
    struct Subsystem {
        std::string name;
        std::shared_ptr<Composable> component;
        bool register_flow;
    };

    void assign_design_var(Context& ctx, std::string local_name);

    void share_outputs(Context& ctx, std::string shared_local_name,
                       const sycl::range<1>& range,
                       std::vector<std::string> member_local_names);

    const std::vector<Subsystem>& get_subsystems() const { return subsystems_; }

    std::shared_ptr<Composable> get_subsystem(const std::string& name) const {
        for (const auto& sub : subsystems_) {
            if (sub.name == name) return sub.component;
        }
        return nullptr;
    }

    /// Typed child access (S7): dynamic-casts the named child. Throws on a
    /// missing child or type mismatch — callers asking by type mean it.
    template <class T>
    std::shared_ptr<T> get_subsystem(const std::string& child_name) const {
        auto base = get_subsystem(child_name);
        if (!base)
            throw std::invalid_argument("System::get_subsystem(): no child '" +
                                        child_name + "' in '" + name() + "'");
        auto typed = std::dynamic_pointer_cast<T>(base);
        if (!typed)
            throw std::invalid_argument("System::get_subsystem(): child '" +
                                        child_name + "' in '" + name() +
                                        "' is not the requested type");
        return typed;
    }

    /// Typed child access by type alone — the unique child of type T.
    /// Throws if none or more than one child matches.
    template <class T>
    std::shared_ptr<T> get_subsystem() const {
        std::shared_ptr<T> found;
        for (const auto& sub : subsystems_) {
            if (auto typed = std::dynamic_pointer_cast<T>(sub.component)) {
                if (found)
                    throw std::logic_error(
                        "System::get_subsystem<T>(): ambiguous — multiple "
                        "children match in '" + name() + "'");
                found = std::move(typed);
            }
        }
        if (!found)
            throw std::invalid_argument(
                "System::get_subsystem<T>(): no child of the requested type "
                "in '" + name() + "'");
        return found;
    }

    // =========================================================================
    // Unified I/O access (S7) — resolve any FQN relative to this system
    // through the ordinary Context resolver (promotion-aware: promoted names,
    // members under a promotion, anything that resolves today). Deliberately
    // uncached full lookups — system-level access is not a hot loop.
    // =========================================================================

    /// Qualify a relative FQN with this system's name (root systems pass
    /// through unchanged).
    std::string qualify(std::string_view rel_fqn) const {
        return name().empty() ? std::string(rel_fqn)
                              : name() + "/" + std::string(rel_fqn);
    }

    template <int Dim>
    Input<Dim> get_input(Context& ctx, std::string_view rel_fqn) const {
        return ctx.get_input<Dim>(qualify(rel_fqn));
    }
    template <DType DT, int Dim>
    auto get_input(Context& ctx, std::string_view rel_fqn) const {
        return ctx.get_input<DT, Dim>(qualify(rel_fqn));
    }
    template <int Dim>
    Output<Dim> get_output(Context& ctx, std::string_view rel_fqn) const {
        return ctx.get_output<Dim>(qualify(rel_fqn));
    }
    template <DType DT, int Dim>
    auto get_output(Context& ctx, std::string_view rel_fqn) const {
        return ctx.get_output<DT, Dim>(qualify(rel_fqn));
    }
    template <int Dim>
    Output<Dim> get_passthrough(Context& ctx, std::string_view rel_fqn) const {
        return ctx.get_passthrough<Dim>(qualify(rel_fqn));
    }
    template <int Dim>
    StateVar<Dim> get_state_var(Context& ctx, std::string_view rel_fqn) const {
        return ctx.get_state_var<Dim>(qualify(rel_fqn));
    }
    template <int Dim>
    InternalBuffer<Dim> get_buffer_var(Context& ctx,
                                       std::string_view rel_fqn) const {
        return ctx.get_buffer_var<Dim>(qualify(rel_fqn));
    }
    template <DType DT, int Dim>
    IOInitializer<Dim, DT> get_initializer(Context& ctx,
                                           std::string_view rel_fqn) const {
        return ctx.get_initializer<DT, Dim>(qualify(rel_fqn));
    }
    template <int Dim>
    IOInitializer<Dim> get_initializer(Context& ctx,
                                       std::string_view rel_fqn) const {
        return ctx.get_initializer<Dim>(qualify(rel_fqn));
    }

    // post_settle — the discrete half of the hybrid system (core extension,
    // 2026-07-16). Called by the EquilibriumExecutor exactly
    // once per settle, AFTER the settle loop exits (converged or budget) and
    // — in training — AFTER the adjoint solve (gradients belong to the
    // pre-commit equilibrium). Systems override to mutate the conditions
    // held constant through the settle (e.g. Mosaic commits the scratch
    // buffer to the committed buffer that output vision perceives).
    // Default: recurse into child Systems so overrides are reached anywhere
    // in the tree.
    virtual void post_settle(Context& ctx) {
        for (const auto& sub : subsystems_) {
            if (auto sys = std::dynamic_pointer_cast<System>(sub.component))
                sys->post_settle(ctx);
        }
    }

protected:
    void add_entry_point(Context& ctx, std::string child_name,
                         std::shared_ptr<Composable> component);

    std::vector<Subsystem> subsystems_;
};

template <int Dim>
inline void System::promote_input(Context& ctx, std::string local_name,
                                   const sycl::range<Dim>& range,
                                   std::vector<std::string> entries) {
    std::string fqn = name().empty() ? local_name : name() + "/" + local_name;
    ctx.declare_promotion<Dim>(fqn, range);
    // If any target entry is a DV, mark the passthrough as DV.
    // Subsequent connect calls will unmark the targets.
    bool any_dv = false;
    for (const auto& entry_path : entries) {
        std::string full_path = name().empty() ? entry_path : name() + "/" + entry_path;
        auto it = ctx.entry_map().find(full_path);
        if (it != ctx.entry_map().end() && ctx.entries_vec()[it->second].is_design_variable) {
            any_dv = true;
            break;
        }
    }
    if (any_dv) {
        ctx.assign_design_var(fqn);
    }
    for (const auto& entry_path : entries) {
        std::string full_path = name().empty() ? entry_path : name() + "/" + entry_path;
        ctx.connect(fqn, full_path);
    }
}

template <int Dim>
inline void System::promote_output(Context& ctx, std::string local_name,
                                    const sycl::range<Dim>& range,
                                    std::string entry) {
    std::string fqn = name().empty() ? local_name : name() + "/" + local_name;
    ctx.declare_promotion<Dim>(fqn, range);
    std::string full_path = name().empty() ? entry : name() + "/" + entry;
    ctx.connect(full_path, fqn);  // child output → passthrough var (passthrough is target)
}

inline void System::share_outputs(Context& ctx, std::string shared_local_name,
                                   const sycl::range<1>& range,
                                   std::vector<std::string> member_local_names) {
    // range is accepted for API symmetry with promote_output; owner shape is
    // derived from the first member inside Context::share_outputs().
    (void)range;
    auto fqn = [&](const std::string& local) {
        return name().empty() ? local : name() + "/" + local;
    };
    std::string owner_fqn = fqn(shared_local_name);
    std::vector<std::string> member_fqns;
    member_fqns.reserve(member_local_names.size());
    for (const auto& m : member_local_names)
        member_fqns.push_back(fqn(m));
    ctx.share_outputs(owner_fqn, member_fqns);
}

}  // namespace naga::core
