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

/// A fused group's machinery: not the user path, a base for FusedGroup. `segments` says which consecutive stages share
/// groups and how (range mode, local memory); they cover the stages in order.
class FusedBase : public System {
public:
    explicit FusedBase(std::vector<SegmentSpec> segments) : segments_{std::move(segments)} {}
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
        (void)launcher_.launch(rt, st, segments_, describe());
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

    std::vector<SegmentSpec> segments_;
    GroupLauncher            launcher_;
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

/// A segment descriptor: consecutive stages that share groups, with the segment's options.
template <LocalMem LM, RangeMode RM, class... Descs>
struct SegmentDesc {
    static constexpr LocalMem    local = LM;
    static constexpr RangeMode   range = RM;
    static constexpr std::size_t count = sizeof...(Descs);
    std::tuple<Descs...>         descs;
};
template <class T>
struct is_segment_desc : std::false_type {};
template <LocalMem LM, RangeMode RM, class... Ds>
struct is_segment_desc<SegmentDesc<LM, RM, Ds...>> : std::true_type {};

template <class O>
inline constexpr bool is_local_option_v = std::is_same_v<O, local_mem::partitioned> || std::is_same_v<O, local_mem::shared>;
template <class O>
inline constexpr bool is_range_option_v = std::is_same_v<O, range_mode::own> || std::is_same_v<O, range_mode::whole>;
template <class T, class... Opts>
inline constexpr std::size_t option_count_v = (std::size_t{std::is_same_v<T, Opts>} + ... + std::size_t{0});

/// The stage descriptors an item contributes: a bare descriptor is one stage; a segment holds its own.
template <class I>
struct item_stages {
    using type = std::tuple<I>;
};
template <LocalMem LM, RangeMode RM, class... Ds>
struct item_stages<SegmentDesc<LM, RM, Ds...>> {
    using type = std::tuple<Ds...>;
};
template <class... Items>
using all_stage_descs_t = decltype(std::tuple_cat(std::declval<typename item_stages<std::remove_cvref_t<Items>>::type>()...));

/// What a fused group requires of its stages, checked at compile time.
template <class T>
struct stage_checks;
template <class... Ds>
struct stage_checks<std::tuple<Ds...>> {
    static constexpr bool nonempty   = sizeof...(Ds) >= 1;
    static constexpr bool components = (std::derived_from<desc_component_t<Ds>, ComponentBase> && ...);
    static constexpr bool fusible    = ((!CustomOpBody<desc_body_t<Ds>> && !HostBody<desc_body_t<Ds>>) && ...);
    static constexpr bool one_kind   = (GroupBody<desc_body_t<Ds>> && ...) || (!GroupBody<desc_body_t<Ds>> && ...);
    static constexpr bool one_rank   = ((kernel_rank_v<desc_body_t<Ds>> == kernel_rank_v<desc_body_t<std::tuple_element_t<0, std::tuple<Ds...>>>>) && ...);
};

template <class Item>
void add_segment_spec(std::vector<SegmentSpec>& out, std::size_t& at) {
    if constexpr (is_segment_desc<Item>::value) {
        out.push_back(SegmentSpec{at, Item::count, Item::local, Item::range});
        at += Item::count;
    } else {
        out.push_back(SegmentSpec{at, 1, LocalMem::partitioned, RangeMode::own});  // a bare stage: a one-stage segment
        at += 1;
    }
}
/// The segments of a group's items, in order.
template <class... Items>
std::vector<SegmentSpec> segment_specs() {
    std::vector<SegmentSpec> out;
    std::size_t              at = 0;
    (add_segment_spec<std::remove_cvref_t<Items>>(out, at), ...);
    return out;
}

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

/// One kernel, one submission: the stages of `Items` (a bare stage descriptor is a one-stage segment; cs::segment groups
/// consecutive stages that share groups), added in order, run as ONE launch. Connections are made after the stages are
/// added and their variables are marked `internal`; they are allowed in either range mode. Tied outputs of its stages
/// follow the ordinary tie policy (section 9).
template <class... Items>
class FusedGroup final : public detail::FusedBase {
    using Checks = detail::stage_checks<detail::all_stage_descs_t<Items...>>;
    static_assert(Checks::nonempty, "naga::cospan: a fused group needs at least one stage");
    static_assert(Checks::components,
                  "naga::cospan: the stages of a fused group are components (subsystem_desc<T> with T a Component)");
    static_assert(Checks::fusible, "naga::cospan: custom-op and host bodies cannot be fused");
    static_assert(Checks::one_kind, "naga::cospan: a fused group's stages are all work-group components or all plain");
    static_assert(Checks::one_rank, "naga::cospan: the stages of a fused group share one kernel rank");

public:
    FusedGroup(ConnectsDesc c, Items... items)
        : detail::FusedBase{detail::segment_specs<Items...>()}, connects_{std::move(c)}, items_{std::move(items)...} {}

    void setup() override {
        std::apply([this](auto&... it) { (this->add_item(it), ...); }, items_);
        for (const auto& [a, b] : connects_.pairs) {
            connect(a, b);
            instance_at(b)->meta->internal = true;  // connected inside one fused group
        }
    }

private:
    template <class Item>
    void add_item(Item& it) {
        if constexpr (detail::is_segment_desc<std::remove_cvref_t<Item>>::value)
            std::apply([this](auto&... d) { (this->add_subsystem(d), ...); }, it.descs);
        else
            this->add_subsystem(it);
    }

    ConnectsDesc         connects_;
    std::tuple<Items...> items_;
};

// ---- helpers: the simple case only -------------------------------------------------

/// A segment of consecutive stages that share groups: cs::segment(descs...) takes the defaults (local_mem::partitioned,
/// range_mode::own); cs::segment<cs::local_mem::shared, cs::range_mode::whole>(descs...) names options (at most one of
/// each kind).
template <class... Opts, class... Descs>
auto segment(Descs... d) {
    static_assert(sizeof...(Descs) >= 1, "naga::cospan: a segment needs at least one stage");
    static_assert(((detail::is_local_option_v<Opts> || detail::is_range_option_v<Opts>) && ...),
                  "naga::cospan: segment options are cs::local_mem::partitioned / shared and cs::range_mode::own / whole");
    static_assert(detail::option_count_v<local_mem::partitioned, Opts...> + detail::option_count_v<local_mem::shared, Opts...> <= 1,
                  "naga::cospan: a segment takes at most one local-memory option");
    static_assert(detail::option_count_v<range_mode::own, Opts...> + detail::option_count_v<range_mode::whole, Opts...> <= 1,
                  "naga::cospan: a segment takes at most one range-mode option");
    constexpr detail::LocalMem  lm = detail::option_count_v<local_mem::shared, Opts...> ? detail::LocalMem::shared : detail::LocalMem::partitioned;
    constexpr detail::RangeMode rm = detail::option_count_v<range_mode::whole, Opts...> ? detail::RangeMode::whole : detail::RangeMode::own;
    return detail::SegmentDesc<lm, rm, Descs...>{std::tuple<Descs...>{std::move(d)...}};
}

/// A fused group of the given items (in order) with the given connections, as a descriptor for add_subsystem.
template <class... Items>
auto fuse_components(std::string name, ConnectsDesc c, Items... items) {
    return subsystem_desc<FusedGroup<Items...>>(std::move(name), std::move(c), std::move(items)...);
}
/// The same with no connections.
template <class... Items>
    requires(sizeof...(Items) >= 1 && !detail::leads_with_connects_v<Items...>)
auto fuse_components(std::string name, Items... items) {
    return subsystem_desc<FusedGroup<Items...>>(std::move(name), ConnectsDesc{}, std::move(items)...);
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
