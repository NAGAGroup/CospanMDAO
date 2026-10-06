#pragma once

// =============================================================================
// naga/cospan/fused_component.hpp — fused groups and the descriptor helpers
// =============================================================================
//
// A FUSED GROUP is semantically a component: one kernel, one submission (docs/design-review-2026-10-03.md, 11.6). Its
// stages are components whose bodies run as the stages of ONE kernel through the group launcher (group_launch.hpp):
//   precompute   runs each stage's precompute, then builds the launcher's pointer array once;
//   compute      the launcher describes every stage, checks the work-group shapes, uploads the stages' arguments and the
//                geometry, and submits ONE launch whose event every stage and array registers;
//   postcompute  runs each stage's postcompute.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/component.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/group_launch.hpp"
#include "naga/cospan/kernel.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/system.hpp"
#include "naga/cospan/variable.hpp"
#include "naga/cospan/wiring.hpp"

namespace naga::cospan {

enum class fuse_strategy : uint8_t { seq, par };

/// The connections of a helper-built group, as (source path, target path) pairs in the group's namespace:
/// connects_desc({{"a.y", "b.x"}, {"b.z", "c.w"}}). The same connect, with the same rules, scoped to the group.
struct ConnectsDesc {
    std::vector<std::pair<std::string, std::string>> pairs;
};

inline ConnectsDesc connects_desc(std::initializer_list<std::pair<std::string_view, std::string_view>> pairs) {
    ConnectsDesc c;
    for (const auto& [a, b] : pairs) c.pairs.emplace_back(std::string{a}, std::string{b});
    return c;
}

namespace detail {

/// A fused group's machinery: not the user path, a base for FusedComponent.
class FusedBase : public System {
public:
    explicit FusedBase(fuse_strategy s) : strategy_{s} {}
    ~FusedBase() override = default;

    /// Setup, then the setup-end check: no ragged ports.
    void run_setup() override {
        System::run_setup();
        check_no_ragged_ports(stages());
    }

    void run_precompute(Runtime& rt) override {
        const std::vector<ComponentBase*> st = stages();
        for (ComponentBase* s : st) {
            s->precompute_steps(rt);
            s->stage_args_ensure(rt);
        }
        launcher_.prepare(rt, st);  // once, at the first precompute
    }

    /// One kernel, one submission.
    void run_compute(Runtime& rt) override {
        const std::vector<ComponentBase*> st = stages();
        if (st.empty()) throw std::invalid_argument("naga::cospan: " + describe() + ": a fused group has no stages");
        (void)launcher_.launch(rt, st, segments(st.size()), describe());
    }

    void run_postcompute(Runtime& rt) override {
        for (ComponentBase* s : stages()) s->run_postcompute(rt);
    }

    std::vector<sycl::event> events() const override { return launcher_.events(); }

    /// depends_on: every stage's submission (all of it is one launch) adds them for one forward.
    void add_dependencies(std::span<const sycl::event> deps) override {
        for (ComponentBase* s : stages()) s->add_dependencies(deps);
    }

private:
    /// seq: one segment (every stage starts at group 0, each runs for its own groups); par: every stage its own segment.
    std::vector<SegmentSpec> segments(std::size_t n) const {
        std::vector<SegmentSpec> out;
        if (strategy_ == fuse_strategy::seq) {
            out.push_back(SegmentSpec{0, n, LocalMem::partitioned, RangeMode::own});
        } else {
            for (std::size_t i = 0; i < n; ++i) out.push_back(SegmentSpec{i, 1, LocalMem::partitioned, RangeMode::own});
        }
        return out;
    }

    /// The live children, as components: a fused group's children are components only.
    std::vector<ComponentBase*> stages() const {
        std::vector<ComponentBase*> out;
        for (Subsystem* c : child_ptrs_) {
            if (c == nullptr) continue;
            auto* s = dynamic_cast<ComponentBase*>(c);
            if (s == nullptr)
                throw std::logic_error("naga::cospan: " + describe() + ": child '" + c->path() + "' is not a Component");
            out.push_back(s);
        }
        return out;
    }

    /// A child with any ragged port (RaggedInput / RaggedOutput, or a per-index dimension) is a setup error.
    void check_no_ragged_ports(const std::vector<ComponentBase*>& st) const {
        for (const ComponentBase* s : st)
            if (const std::shared_ptr<Instance> e = s->first_ragged_port())
                throw std::invalid_argument("naga::cospan: " + describe() + ": '" + e->name + "' of '" + s->path() +
                                            "' is a ragged port: ragged ports are not supported in fused components yet");
    }

    fuse_strategy strategy_;
    GroupLauncher launcher_;
};

/// The component type of a SubsystemDesc.
template <class D>
struct desc_component;
template <class T, class... A>
struct desc_component<SubsystemDesc<T, A...>> {
    using type = T;
};
template <class D>
using desc_component_t = typename desc_component<std::remove_cvref_t<D>>::type;
template <class D>
using desc_body_t = typename desc_component_t<D>::body_type;

/// Whether the first of `Ds` is a ConnectsDesc (false for none).
template <class... Ds>
inline constexpr bool leads_with_connects_v = false;
template <class D, class... Ds>
inline constexpr bool leads_with_connects_v<D, Ds...> = std::is_same_v<std::remove_cvref_t<D>, ConnectsDesc>;

/// A plain System of the given children in order, with the given connections.
template <class... Descs>
class ComposedSystem : public System {
public:
    ComposedSystem(ConnectsDesc c, Descs... d) : connects_{std::move(c)}, descs_{std::move(d)...} {}

    void setup() override {
        std::apply([this](auto&... d) { (this->add_subsystem(d), ...); }, descs_);
        for (const auto& [a, b] : connects_.pairs) connect(a, b);
    }

private:
    ConnectsDesc        connects_;
    std::tuple<Descs...> descs_;
};

}  // namespace detail

/// One kernel, one submission: the stages `Descs` (descriptors of components), added in order, run as one wave.
/// Connections (seq only) are made after the stages are added and their variables are marked `internal`. Tied outputs of
/// its stages follow the ordinary tie policy (section 9).
template <fuse_strategy S, class... Descs>
class FusedComponent final : public detail::FusedBase {
    static_assert(sizeof...(Descs) >= 1, "naga::cospan: a fused component needs at least one stage");
    static_assert((std::derived_from<detail::desc_component_t<Descs>, ComponentBase> && ...),
                  "naga::cospan: the stages of a fused component are components (subsystem_desc<T> with T a Component)");
    static_assert(((!CustomOpBody<detail::desc_body_t<Descs>> && !HostBody<detail::desc_body_t<Descs>>) && ...),
                  "naga::cospan: custom-op and host bodies cannot be fused");
    static_assert((GroupBody<detail::desc_body_t<Descs>> && ...) || (!GroupBody<detail::desc_body_t<Descs>> && ...),
                  "naga::cospan: a fused component's stages are all work-group components or all plain");
    static_assert(((kernel_rank_v<detail::desc_body_t<Descs>> == kernel_rank_v<detail::desc_body_t<std::tuple_element_t<0, std::tuple<Descs...>>>>) && ...),
                  "naga::cospan: the stages of a fused component share one kernel rank");

public:
    FusedComponent(ConnectsDesc c, Descs... d) : detail::FusedBase{S}, connects_{std::move(c)}, descs_{std::move(d)...} {}

    void setup() override {
        std::apply([this](auto&... d) { (this->add_subsystem(d), ...); }, descs_);
        for (const auto& [a, b] : connects_.pairs) {
            if constexpr (S == fuse_strategy::par) {
                detail::wiring_fail(detail::WiringSite{describe()}, "connect",
                                    "a par fused component's stages run unordered: connect is rejected ('" + a + "' -> '" + b + "')");
            } else {
                connect(a, b);
                instance_at(b)->meta->internal = true;  // connected inside one fused component
            }
        }
    }

private:
    ConnectsDesc        connects_;
    std::tuple<Descs...> descs_;
};

// ---- helpers: the simple case only -------------------------------------------------

/// A fused component of the given stages (in order) with the given connections, as a descriptor for add_subsystem.
template <fuse_strategy S, class... Descs>
auto fuse_components(std::string name, ConnectsDesc c, Descs... d) {
    return subsystem_desc<FusedComponent<S, Descs...>>(std::move(name), std::move(c), std::move(d)...);
}
/// The same with no connections.
template <fuse_strategy S, class... Descs>
    requires(sizeof...(Descs) >= 1 && !detail::leads_with_connects_v<Descs...>)
auto fuse_components(std::string name, Descs... d) {
    return subsystem_desc<FusedComponent<S, Descs...>>(std::move(name), ConnectsDesc{}, std::move(d)...);
}

/// A plain System of the given children (in order) with the given connections, as a descriptor for add_subsystem.
template <class... Descs>
auto system_from(std::string name, ConnectsDesc c, Descs... d) {
    return subsystem_desc<detail::ComposedSystem<Descs...>>(std::move(name), std::move(c), std::move(d)...);
}
template <class... Descs>
    requires(sizeof...(Descs) >= 1 && !detail::leads_with_connects_v<Descs...>)
auto system_from(std::string name, Descs... d) {
    return subsystem_desc<detail::ComposedSystem<Descs...>>(std::move(name), ConnectsDesc{}, std::move(d)...);
}

}  // namespace naga::cospan
