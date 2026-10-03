#pragma once

// =============================================================================
// naga/cospan/parallel_system.hpp — ParallelSystem (r01 §7.5, §7.3, §11.3.1)
// =============================================================================
//
// CHILDREN ARE COMPONENTS ONLY, enforced by the type system at the add: a
// System cannot discern its children at runtime (r01 §1 need 2.2 parallel · P5).
// Nested Parallel is not a concept.
//
// One forward turn: pre-compute ALL children; compute ALL (collect
// the descriptions); check the siblings (below); ONE queue.submit_group in which
// every child's description is submitted with its own device; post-compute ALL
// after the group returns. The group is ONE item of the stream: its
// children are unordered among themselves and ordered both ways against
// everything outside (§7.3).
//
// THE SIBLING RULES (every forward, from the metadata; wave_rules.hpp, shared
// with the parallel mode of FusedSystem):
//   - NO CONNECTIONS between siblings: an input whose root is an output of
//     another sibling throws, naming both. There is no override.
//   - NO TIED OUTPUTS: two or more siblings whose outputs share one root
//     (tie_outputs) throw, naming them, unless allow_tied_outputs() was called.
// Anything else about the siblings' data independence is the USER's assertion (P4).
//
// Include order: this header includes system.hpp, whose helper form
// (property::parallel) needs ParallelSystem complete when it is instantiated,
// so include naga/cospan/cospan.hpp (or this header) wherever the helper form
// is used.

#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "naga/cospan/system.hpp"
#include "naga/cospan/wave_rules.hpp"

namespace naga::cospan {

class ParallelSystem : public System {
public:
    ~ParallelSystem() override = default;

    /// Assert that siblings whose outputs share one root (tie_outputs) do not
    /// race (atomic writers, say); the framework then skips that check. The
    /// no-connections rule has no override.
    void allow_tied_outputs() { allow_tied_ = true; }

    // ---- adding children: Components only, at compile time ---------------------------------
    // These declarations HIDE every add_subsystem of System, so a System child
    // (or the helper form, or the empty form) has no viable overload here.

    template <class T, class... Args>
        requires(std::derived_from<T, ComponentBase> && !detail::leads_with_props_v<Args...>)
    void add_subsystem(std::string name, Args&&... ctor_args) {
        System::add_subsystem<T>(std::move(name), std::forward<Args>(ctor_args)...);
    }

    /// property::device only: each child may sit on its own device (the group
    /// is one item of the stream whichever devices its members target). A
    /// property_list is a runtime value, so any other property throws.
    template <class T, class... Args>
        requires std::derived_from<T, ComponentBase>
    void add_subsystem(std::string name, const property_list& props, Args&&... ctor_args) {
        if (props.has_property<property::parallel>() || props.has_property<property::transient>() ||
            props.has_property<property::fused>() || props.has_property<property::run_n>() ||
            props.has_property<property::do_once>())
            throw std::invalid_argument("naga::cospan: " + describe() + ": add_subsystem('" + name +
                                        "'): a ParallelSystem child takes property::device only");
        System::add_subsystem<T>(std::move(name), props, std::forward<Args>(ctor_args)...);
    }

    /// A ParallelSystem has no untyped children to add.
    void add_subsystem(std::string, const property_list& = {}) = delete;
    /// A flavor keeps its children: no run policies, no deletion, no rearm.
    bool slots_deletable() const noexcept override { return false; }

    /// ragged ports are not supported in a ParallelSystem yet (a setup error).
    void compile() override {
        System::compile();  // the children compile first
        detail::check_no_ragged_ports(describe(), children());
    }

    // ---- the runner (r01 §7.5, §11.3.1) ---------------------------------------------------------
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        const std::size_t n = children_.size();
        for (std::size_t i = 0; i < n; ++i) {
            materialize_slot(i);
            children_[i]->run_pre_compute(child_ctx_[i]);
        }

        std::vector<std::unique_ptr<LaunchDescription>> ds;
        ds.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto d = children_[i]->run_compute_forward();
            if (!d)
                throw std::logic_error("naga::cospan: " + describe() + ": forward: child '" +
                                       children_[i]->path() +
                                       "' returned no launch description; a ParallelSystem child must be a "
                                       "Component that describes its work");
            ds.push_back(std::move(d));
        }

        detail::check_no_stage_connections(describe(), "ParallelSystem", children());
        if (!allow_tied_) detail::check_tied_outputs(describe(), "ParallelSystem", children());

        std::vector<std::vector<sycl::event>> events(n);  // each child's own launch events
        ctx_.queue->submit_group([&](sk::queue::group& g) {
            GroupScope scope{*this, &g};
            for (std::size_t i = 0; i < n; ++i) {
                SubmitTarget t = child_ctx_[i].target();  // the child's device, and the group
                events[i]      = ds[i]->submit(t);
            }
        });

        // Every child's launch events go on that child's ports (r01 §4.4), before any post_compute pushes.
        for (std::size_t i = 0; i < n; ++i) children_[i]->register_launch_events(events[i]);
        for (std::size_t i = 0; i < n; ++i) children_[i]->run_post_compute(child_ctx_[i]);
        return nullptr;  // "I submitted"
    }

private:
    /// ctx.group is set for the extent of the submit_group and null otherwise
    /// (§7.5), on this System's ctx and on each child's.
    struct GroupScope {
        GroupScope(ParallelSystem& s, sk::queue::group* g) : s_{s} { set(g); }
        ~GroupScope() { set(nullptr); }
        GroupScope(const GroupScope&)            = delete;
        GroupScope& operator=(const GroupScope&) = delete;

        void set(sk::queue::group* g) {
            s_.ctx_.group = g;
            for (ExecCtx& c : s_.child_ctx_) c.group = g;
        }
        ParallelSystem& s_;
    };

    bool allow_tied_ = false;  // siblings may share an output root
};

}  // namespace naga::cospan