#pragma once

// =============================================================================
// naga/cospan/parallel_system.hpp — ParallelSystem (r01 §7.5, §7.3, §11.3.1)
// =============================================================================
//
// CHILDREN ARE COMPONENTS ONLY, enforced by the type system at the add: a
// System cannot discern its children at runtime (r01 §1 need 2.2 parallel · P5).
// Nested Parallel is not a concept.
//
// One forward turn: revive(); pre-compute ALL children; compute ALL (collect
// the descriptions); check cross-sibling shape dependencies; ONE
// queue.submit_group in which every child's description is submitted with its
// own device; post-compute ALL after the group returns; release(). The group
// is ONE item of the stream: its children are unordered among themselves and
// ordered both ways against everything outside (§7.3). Data independence of
// the children is the USER's assertion (P4); the framework checks only the
// shape dependency it can see on the metadata.
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

namespace naga::cospan {

class ParallelSystem : public System {
public:
    ~ParallelSystem() override = default;

    // ---- adding children: Components only, at compile time ---------------------------------
    // These declarations HIDE every add_subsystem of System, so a System child
    // (or the helper form, or the empty form) has no viable overload here.

    template <class T, class... Args>
        requires(std::derived_from<T, ComponentBase> && !detail::leads_with_props_v<Args...>)
    T& add_subsystem(std::string name, Args&&... ctor_args) {
        return System::add_subsystem<T>(std::move(name), std::forward<Args>(ctor_args)...);
    }

    /// property::device only: each child may sit on its own device (the group
    /// is one item of the stream whichever devices its members target). A
    /// property_list is a runtime value, so any other property throws.
    template <class T, class... Args>
        requires std::derived_from<T, ComponentBase>
    T& add_subsystem(std::string name, const property_list& props, Args&&... ctor_args) {
        if (props.has_property<property::parallel>() || props.has_property<property::transient>() ||
            props.has_property<property::fused>())
            throw std::invalid_argument("naga::cospan: " + describe() + ": add_subsystem('" + name +
                                        "'): a ParallelSystem child takes property::device only");
        return System::add_subsystem<T>(std::move(name), props, std::forward<Args>(ctor_args)...);
    }

    /// A ParallelSystem has no untyped children to add.
    System& add_subsystem(std::string, const property_list& = {}) = delete;

    // ---- the runner (r01 §7.5, §11.3.1) ---------------------------------------------------------
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        revive();
        const std::size_t n = children_.size();
        for (std::size_t i = 0; i < n; ++i) children_[i]->run_pre_compute(child_ctx_[i]);

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

        check_sibling_shape_dependencies();

        ctx_.queue->submit_group([&](sk::queue::group& g) {
            GroupScope scope{*this, &g};
            for (std::size_t i = 0; i < n; ++i) {
                SubmitTarget t = child_ctx_[i].target();  // the child's device, and the group
                ds[i]->submit(t);
            }
        });

        for (std::size_t i = 0; i < n; ++i) children_[i]->run_post_compute(child_ctx_[i]);
        release();
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

    /// The one thing the framework checks (§7.5): a child input whose ROOT is a
    /// SIBLING's output and whose shape is derived from its source
    /// (FromSourceDim). Siblings run unordered, so the source's extent is not
    /// known when the consumer launches. A hard error, every forward.
    void check_sibling_shape_dependencies() const {
        for (const Subsystem* c : children()) {
            // Children are Components by construction (the add_subsystem constraint).
            const auto* comp = dynamic_cast<const ComponentBase*>(c);
            if (comp == nullptr) continue;
            for (const auto& ref : comp->get_inputs()) {
                const detail::IOEntry& e = ref.get();
                const detail::IOEntry* r = e.root();
                if (r == &e || r->kind() != IOKind::output) continue;
                const Subsystem* producer = r->declarer();
                if (producer == c || !is_child(producer)) continue;
                for (int d = 0; d < e.rank(); ++d)
                    if (e.dim(d).how == detail::DimRecord::How::from_source_dim)
                        throw std::invalid_argument(
                            "naga::cospan: " + describe() + ": forward: input '" + std::string{e.name()} + "' of '" +
                            c->path() + "' takes dimension " + std::to_string(d) + " of its shape from '" +
                            std::string{r->name()} + "', an output of its sibling '" + producer->path() +
                            "' (FromSourceDim); siblings of a ParallelSystem run unordered, so that extent is "
                            "not known when the consumer launches (connect them through a sequential System, "
                            "or give the input a fixed shape)");
            }
        }
    }

    bool is_child(const Subsystem* s) const {
        for (const Subsystem* c : children())
            if (c == s) return true;
        return false;
    }
};

}  // namespace naga::cospan
