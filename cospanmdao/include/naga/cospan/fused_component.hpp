#pragma once

// =============================================================================
// naga/cospan/fused_component.hpp — fused groups and the descriptor helpers
// =============================================================================
//
// A FUSED GROUP is semantically a component: one kernel, one submission (internal/design-review-2026-10-03.md, 11.6). Its
// stages are components whose bodies run as the stages of ONE kernel through the group launcher (group_launch.hpp):
//   precompute   runs each stage's precompute, then builds the launcher's pointer array once;
//   compute      the launcher describes every stage, checks the work-group shapes, uploads the stages' arguments and the
//                geometry, and submits ONE launch whose event every stage and array registers;
//   postcompute  runs each stage's postcompute.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <functional>
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
class FusedDriver;
}  // namespace detail

/// One tie declared in a group: cs::tie_outputs<DT, R>(...) or cs::tie_inputs<DT, R>(...), applied to the group in setup().
using TieEntry = std::function<void(detail::FusedDriver&)>;

/// The ties of a helper-built group, in order: ties_desc({cs::tie_outputs<DType::f32, 1>({"a.y", "b.y"}, "y"), ...}). Each
/// entry takes the parameters of System::tie_outputs / System::tie_inputs (member paths relative to the group: stage name,
/// then port), minus the device, which is the group's own. The alias lives in the group, at `<group>.<name>`.
struct TiesDesc {
    std::vector<TieEntry> entries;
};

inline TiesDesc ties_desc(std::initializer_list<TieEntry> entries) { return TiesDesc{std::vector<TieEntry>{entries}}; }

/// A fused-group property, given as a template option of fuse_components: cs::fuse_components<cs::group_waits_enabled>(...).
/// With it the stages may wait on work-groups of EARLIER segments (GroupItem::wait_group, wait_earlier_segments) without
/// any co-residency of groups; a cs::global_barrier() in the group implies it. The group's stages must all be group-range
/// stages. Without it (and without a global_barrier) the groups carry no wait state at all.
struct group_waits_enabled {};

/// How a fused group's work-groups are laid on the physical (always rank-3) launch; a fuse_components<...> option next to
/// cs::group_waits_enabled, at most one per group. The stage rank R only exists in the wrappers: a stage's range and work-group
/// shape live in the TRAILING physical dimensions (rank 1: {1, 1, N}, rank 2: {1, Y, X}). Raw SYCL ids (sycl::nd_item<3>,
/// sycl::group<3>) are PHYSICAL and mean nothing to a stage; a stage uses the rank-R ids of GroupItem and cs::Group.
///   flatten_and_split (the default)  every group along the LAST physical dimension; the rank-R group id is rebuilt row-major
///                                    over the segment's range, last dimension fastest.
///   pad_and_split                    the segments are stacked along the last dimension (its extent is the sum of the
///                                    segments' extents there); the other dimensions are the maximum over all segments.
///                                    Group ids are the hardware's, minus the segment's offset along the last dimension.
///   padded                           physical dimension 0 is the segment index (work-group extent 1); the other dimensions
///                                    are the maximum over all segments. Stages must be rank 2 or lower.
namespace range_split {
struct flatten_and_split {};
struct pad_and_split {};
struct padded {};
}  // namespace range_split

namespace detail {

/// Whether a range_split strategy supports stages of rank `Rank`: padded needs rank 2 or lower (its first physical dimension
/// is the segment index). FusedGroup static_asserts it; the trait is for code that wants to ask first.
template <Split Sp, int Rank>
inline constexpr bool split_supports_rank_v = Sp != Split::padded || Rank <= 2;

/// A fused group's machinery: not the user path, a base for FusedGroup. `segments` says which consecutive stages share
/// groups and how (range mode, local memory); they cover the stages in order. `group_waits`: the group has the
/// group_waits_enabled property or contains a global_barrier, so its launch carries the wait state.
///
/// A fused group is a LEAF of the model's tree (r05: a ComponentDriver): one forward, one launch. It OWNS its stages, which are
/// components (ComponentBase), not tree nodes: their ports are registered as `<group>.<stage>.<port>` and answer to the group's
/// node. Its own declarations (ties, connections) go through a declaration SCOPE, a private System bound to the group's path
/// and mapped to the group's node, so the System rules apply unchanged.
class FusedDriver : public ComponentDriver {
public:
    FusedDriver(std::vector<SegmentSpec> segments, bool group_waits, Split split)
        : segments_{std::move(segments)}, group_waits_{group_waits}, split_{split} {}
    ~FusedDriver() override = default;

    /// The group's declaration hook: its stages, ties and connections (implemented by FusedGroup).
    virtual void setup() = 0;

    /// Binds the declaration scope to this group, maps it to this group's node, then runs setup() and the setup-end check: no
    /// ragged ports.
    void run_setup() override {
        SubsystemAccess::bind(scope_, path(), meta_);
        meta_->adopt_sub(&scope_, *meta_->node_for(this));
        {
            ModelMeta::SetupScope setup_scope{meta_};
            setup();
        }
        check_no_ragged_ports(parts_);
    }

    /// ONE forward for the group: every stage's pre steps, the single launch, every stage's post steps. The group's own tied
    /// outputs are pushed by the after-operation finalize placed on this leaf (the same as for any leaf).
    void forward(Runtime& rt) override {
        if (parts_.empty()) throw std::invalid_argument("naga::cospan: " + describe() + ": a fused group has no stages");
        for (ComponentBase* s : parts_) {
            s->precompute_steps(rt);
            s->stage_args_ensure(rt);
        }
        launcher_.prepare(rt, parts_);  // once, at the first forward
        (void)launcher_.launch(rt, parts_, segments_, describe(), group_waits_, split_);
        for (ComponentBase* s : parts_) s->step_post(rt);
    }

    /// The launch's events.
    std::vector<sycl::event> events() const override { return launcher_.events(); }

    /// The group's own ties (cs::ties_desc): the same logic as System::tie_inputs / tie_outputs, on the group, on the default
    /// device that the group's set_device moves with the rest. Called by the TieEntry objects in setup().
    template <DType DT, int Rank>
    void group_tie_inputs(const std::vector<std::string>& members, std::string name, ShapeSpec shape) {
        const std::vector<std::string_view> v(members.begin(), members.end());
        scope_.template tie_inputs_of<DT, Rank>(std::span<const std::string_view>{v}, std::move(name), std::move(shape));
    }
    template <DType DT, int Rank>
    void group_tie_outputs(const std::vector<std::string>& members, std::string name, ShapeSpec shape, TieOutputsPolicy policy) {
        const std::vector<std::string_view> v(members.begin(), members.end());
        scope_.template tie_outputs_of<DT, Rank>(std::span<const std::string_view>{v}, std::move(name), std::move(shape), policy);
    }

    /// Whether the group has allocated its wait state (group_waits_enabled or a global_barrier, after a forward).
    bool has_wait_state() const { return launcher_.has_wait_state(); }

protected:
    /// "System 'path'": the prefix of the group's errors (the scope's, as before).
    std::string describe() const { return scope_.describe(); }

    /// connect / instance_at in the group's namespace.
    void connect(std::string_view src, std::string_view dst) { scope_.connect(src, dst); }
    std::shared_ptr<Instance> instance_at(std::string_view rel) { return scope_.instance_at(rel); }

    /// Adds a stage: a component the group owns (not a tree node). Its full name is `<group>.<name>`; it takes no device and no
    /// run policy (the group has them). Its declare() runs before this returns.
    template <class T, class... A>
        requires std::derived_from<T, ComponentBase>
    void add_stage(SubsystemDesc<T, A...>& desc) {
        ModelMeta& m = *meta_;
        m.require_in_setup("add_subsystem", describe());
        ModelMeta::check_local_name(desc.name, describe(), "subsystem");
        const std::string full = ModelMeta::child_name(path(), desc.name);
        if (m.name_taken(full))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + full + "' is already a subsystem or alias name in this model");
        std::unique_ptr<ComponentBase> stage =
            std::apply([](auto&... a) -> std::unique_ptr<ComponentBase> { return std::make_unique<T>(a...); }, desc.args);
        ComponentBase* raw = stage.get();
        ComponentAccess::bind(*raw, meta_, full, this);
        m.stage_names.insert(full);
        stage_objs_.push_back(std::move(stage));
        parts_.push_back(raw);
        ComponentAccess::run_declare(*raw);
    }

private:
    /// The declaration scope: a System that exists to carry the group's connect, tie and resolution calls. Bound to the group's
    /// path and mapped to the group's node when the group is set up.
    class Scope final : public System {
    public:
        void setup() override {}
        using System::connect;
        using System::describe;
        using System::instance_at;
        using System::tie_inputs_of;
        using System::tie_outputs_of;
    };

    /// A child with any ragged port (RaggedInput / RaggedOutput, or a per-index dimension) is a setup error.
    void check_no_ragged_ports(const std::vector<ComponentBase*>& st) const {
        for (const ComponentBase* s : st)
            if (const std::shared_ptr<Instance> e = s->first_ragged_port())
                throw std::invalid_argument("naga::cospan: " + describe() + ": '" + e->name + "' of '" + s->path() +
                                            "' is a ragged port: ragged ports are not supported in fused components yet");
    }

    std::vector<SegmentSpec> segments_;
    bool                     group_waits_ = false;
    Split                    split_       = Split::flatten;
    GroupLauncher            launcher_;
    Scope                                   scope_;
    std::vector<std::unique_ptr<ComponentBase>> stage_objs_;  ///< the group owns its stages (parts_ holds the same objects raw)
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

/// Whether the first of `Ds` is a TiesDesc (false for none).
template <class... Ds>
inline constexpr bool leads_with_ties_v = false;
template <class D, class... Ds>
inline constexpr bool leads_with_ties_v<D, Ds...> = std::is_same_v<std::remove_cvref_t<D>, TiesDesc>;

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

/// A boundary item between two segments of a group (cs::global_barrier(), the only one).
template <Boundary K>
struct BoundaryItem {
    static constexpr Boundary kind = K;
};
template <class T>
struct is_boundary : std::false_type {};
template <Boundary K>
struct is_boundary<BoundaryItem<K>> : std::true_type {};
template <class... Items>
inline constexpr bool has_boundary_v = (is_boundary<std::remove_cvref_t<Items>>::value || ...);

/// A boundary goes BETWEEN two segments: not first, not last, not next to another boundary.
template <class... Items>
constexpr bool boundaries_placed_ok() {
    constexpr std::size_t N = sizeof...(Items);
    if constexpr (N == 0) {
        return true;
    } else {
        constexpr bool b[N] = {is_boundary<std::remove_cvref_t<Items>>::value...};
        if (b[0] || b[N - 1]) return false;
        for (std::size_t i = 0; i + 1 < N; ++i)
            if (b[i] && b[i + 1]) return false;
        return true;
    }
}

/// The stage descriptors an item contributes: a bare descriptor is one stage; a segment holds its own; a boundary none.
template <class I>
struct item_stages {
    using type = std::tuple<I>;
};
template <Boundary K>
struct item_stages<BoundaryItem<K>> {
    using type = std::tuple<>;
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
    // kernel Components only: a CustomComponent or HostComponent is refused by type
    static constexpr bool components = (std::derived_from<desc_component_t<Ds>, Component<desc_component_t<Ds>>> && ...);
    static constexpr bool fusible    = ((!CustomOpBody<desc_body_t<Ds>> && !HostBody<desc_body_t<Ds>>) && ...);
    static constexpr bool one_kind   = (GroupRangeBody<desc_body_t<Ds>> && ...) || (!GroupRangeBody<desc_body_t<Ds>> && ...);
    static constexpr bool all_group  = (GroupRangeBody<desc_body_t<Ds>> && ...);
    static constexpr int  rank       = kernel_rank_v<desc_body_t<std::tuple_element_t<0, std::tuple<Ds...>>>>;
    static constexpr bool one_rank   = ((kernel_rank_v<desc_body_t<Ds>> == kernel_rank_v<desc_body_t<std::tuple_element_t<0, std::tuple<Ds...>>>>) && ...);
};

/// `pending` is the boundary item just before this one, if any: it goes on the next segment.
template <class Item>
void add_segment_spec(std::vector<SegmentSpec>& out, std::size_t& at, Boundary& pending) {
    if constexpr (is_boundary<Item>::value) {
        pending = Item::kind;
    } else if constexpr (is_segment_desc<Item>::value) {
        out.push_back(SegmentSpec{at, Item::count, Item::local, Item::range, pending});
        at += Item::count;
        pending = Boundary::none;
    } else {
        out.push_back(SegmentSpec{at, 1, LocalMem::partitioned, RangeMode::own, pending});  // a bare stage: a one-stage segment
        at += 1;
        pending = Boundary::none;
    }
}
/// The segments of a group's items, in order.
template <class... Items>
std::vector<SegmentSpec> segment_specs() {
    std::vector<SegmentSpec> out;
    std::size_t              at      = 0;
    Boundary                 pending = Boundary::none;
    (add_segment_spec<std::remove_cvref_t<Items>>(out, at, pending), ...);
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
template <bool Waits, detail::Split Sp, class... Items>
class FusedGroup final : public detail::FusedDriver {
    using Checks = detail::stage_checks<detail::all_stage_descs_t<Items...>>;
    static_assert(Checks::nonempty, "naga::cospan: a fused group needs at least one stage");
    static_assert(Checks::components,
                  "naga::cospan: the stages of a fused group are components (subsystem_desc<T> with T a Component)");
    static_assert(Checks::fusible, "naga::cospan: custom-op and host bodies cannot be fused");
    static_assert(Checks::one_kind, "naga::cospan: a fused group's stages are all group-range components or all index-range components");
    static_assert(Checks::one_rank, "naga::cospan: the stages of a fused group share one kernel rank");
    static_assert(detail::boundaries_placed_ok<Items...>(),
                  "naga::cospan: a cs::global_barrier() goes between two segments: not first, not last, and not next to "
                  "another barrier");
    static_assert(!detail::has_boundary_v<Items...> || Checks::all_group,
                  "naga::cospan: cs::global_barrier() is for groups of group-range stages: an index-range group has no "
                  "work-group to cooperate with");
    static_assert(detail::split_supports_rank_v<Sp, Checks::rank>,
                  "naga::cospan: cs::range_split::padded needs stages of rank 2 or lower (its first physical dimension is the "
                  "segment index); use flatten_and_split or pad_and_split for rank-3 stages");
    static_assert(!Waits || Checks::all_group,
                  "naga::cospan: cs::group_waits_enabled is for groups of group-range stages: an index-range group has no "
                  "work-group to wait with");

public:
    FusedGroup(ConnectsDesc c, TiesDesc t, Items... items)
        : detail::FusedDriver{detail::segment_specs<Items...>(), Waits || detail::has_boundary_v<Items...>, Sp}, connects_{std::move(c)}, ties_{std::move(t)}, items_{std::move(items)...} {}

    void setup() override {
        std::apply([this](auto&... it) { (this->add_item(it), ...); }, items_);
        // The ties come first: a tie refuses members that are already connected, and a connect may name a tie alias. Only the
        // System tie rules apply (no ordering or pattern checks among tied stages: that is the designer's, by stage order,
        // segments, global_barrier and group waits).
        for (const TieEntry& e : ties_.entries) e(*this);
        for (const auto& [a, b] : connects_.pairs) {
            check_forward(a, b);
            connect(a, b);
            instance_at(b)->meta->internal = true;  // connected inside one fused group
        }
    }

private:
    template <class Item>
    void add_item(Item& it) {
        if constexpr (detail::is_boundary<std::remove_cvref_t<Item>>::value) {
            // a boundary adds no child
        } else if constexpr (detail::is_segment_desc<std::remove_cvref_t<Item>>::value) {
            std::apply(
                [this](auto&... d) {
                    (this->add_stage(d), ...);
                },
                it.descs);
        } else {
            this->add_stage(it);
        }
    }
    template <class D>
    void add_stage(D& d) {
        names_.push_back(d.name);  // the stage order, for the connect check
        this->detail::FusedDriver::add_stage(d);
    }

    /// A connect inside the group runs forward in item order: the source stage comes before the target stage. A connect from
    /// a stage's output to the input of the same or an earlier stage would read what has not been written yet.
    void check_forward(const std::string& src, const std::string& dst) const {
        const auto stage_of = [this](const std::string& p) -> std::optional<std::size_t> {
            const std::string head = p.substr(0, p.find('.'));
            for (std::size_t i = 0; i < names_.size(); ++i)
                if (names_[i] == head) return i;
            return std::nullopt;
        };
        const auto s = stage_of(src);
        const auto d = stage_of(dst);
        if (s && d && *s >= *d)
            detail::wiring_fail(detail::WiringSite{describe()}, "connect",
                                "'" + dst + "' reads '" + src + "', which runs after it in fused group '" + path() + "'");
    }

    std::vector<std::string> names_;

    ConnectsDesc         connects_;
    TiesDesc             ties_;
    std::tuple<Items...> items_;
};

// ---- helpers: the simple case only -------------------------------------------------

/// A segment of consecutive stages that share groups: cs::segment(descs...) takes the defaults (local_mem::partitioned,
/// range_mode::own); cs::segment<cs::local_mem::shared, cs::range_mode::whole>(descs...) names options (at most one of
/// each kind). The local_mem options are for segments of group-range stages only.
template <class... Opts, class... Descs>
auto segment(Descs... d) {
    static_assert(sizeof...(Descs) >= 1, "naga::cospan: a segment needs at least one stage");
    static_assert(((detail::is_local_option_v<Opts> || detail::is_range_option_v<Opts>) && ...),
                  "naga::cospan: segment options are cs::local_mem::partitioned / shared and cs::range_mode::own / whole");
    static_assert(detail::option_count_v<local_mem::partitioned, Opts...> + detail::option_count_v<local_mem::shared, Opts...> <= 1,
                  "naga::cospan: a segment takes at most one local-memory option");
    static_assert(detail::option_count_v<range_mode::own, Opts...> + detail::option_count_v<range_mode::whole, Opts...> <= 1,
                  "naga::cospan: a segment takes at most one range-mode option");
    static_assert(detail::option_count_v<local_mem::partitioned, Opts...> + detail::option_count_v<local_mem::shared, Opts...> == 0 ||
                      (GroupRangeBody<detail::desc_body_t<Descs>> && ...),
                  "naga::cospan: the cs::local_mem options are for segments of group-range stages: an index-range stage has no "
                  "local memory");
    constexpr detail::LocalMem  lm = detail::option_count_v<local_mem::shared, Opts...> ? detail::LocalMem::shared : detail::LocalMem::partitioned;
    constexpr detail::RangeMode rm = detail::option_count_v<range_mode::whole, Opts...> ? detail::RangeMode::whole : detail::RangeMode::own;
    return detail::SegmentDesc<lm, rm, Descs...>{std::tuple<Descs...>{std::move(d)...}};
}

/// The one boundary item for fuse_components, between two segments of a group of work-group stages: every group of the
/// segment after it waits for every group of the earlier segments before its first stage. Each work-group waits only on
/// EARLIER groups, so no co-residency is needed. It implies the group_waits_enabled property. Any other ordering pattern
/// is the user's logic inside their stages (GroupItem::wait_group / wait_earlier_segments).
inline detail::BoundaryItem<detail::Boundary::global_barrier> global_barrier() { return {}; }

namespace detail {
template <class O>
inline constexpr bool is_fuse_option_v = std::is_same_v<O, group_waits_enabled> || std::is_same_v<O, range_split::flatten_and_split> ||
                                         std::is_same_v<O, range_split::pad_and_split> || std::is_same_v<O, range_split::padded>;

template <class... Opts>
inline constexpr bool split_options_ok_v = (option_count_v<range_split::flatten_and_split, Opts...> + option_count_v<range_split::pad_and_split, Opts...> +
                                            option_count_v<range_split::padded, Opts...>) <= 1;

/// The strategy named among the options (flatten_and_split when none).
template <class... Opts>
constexpr Split split_of() {
    if constexpr (option_count_v<range_split::pad_and_split, Opts...> > 0)
        return Split::pad;
    else if constexpr (option_count_v<range_split::padded, Opts...> > 0)
        return Split::padded;
    else
        return Split::flatten;
}
}  // namespace detail

/// A tie_outputs declared in a fused group (cs::ties_desc): the parameters of System::tie_outputs, minus the device.
template <DType DT, int Rank>
TieEntry tie_outputs(std::initializer_list<std::string_view> members, std::string name, ShapeSpec shape = {},
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic) {
    std::vector<std::string> ms;
    for (const std::string_view m : members) ms.emplace_back(m);
    return [ms = std::move(ms), name = std::move(name), shape = std::move(shape), policy](detail::FusedDriver& g) {
        g.template group_tie_outputs<DT, Rank>(ms, name, shape, policy);
    };
}
/// A tie_inputs declared in a fused group (cs::ties_desc): the parameters of System::tie_inputs, minus the device.
template <DType DT, int Rank>
TieEntry tie_inputs(std::initializer_list<std::string_view> members, std::string name, ShapeSpec shape = {}) {
    std::vector<std::string> ms;
    for (const std::string_view m : members) ms.emplace_back(m);
    return [ms = std::move(ms), name = std::move(name), shape = std::move(shape)](detail::FusedDriver& g) {
        g.template group_tie_inputs<DT, Rank>(ms, name, shape);
    };
}

/// A fused group of the given items (in order), as a descriptor for add_subsystem:
/// cs::fuse_components[<options>](name, [connects_desc(...)], [ties_desc(...)], items...). Options are template arguments:
/// cs::fuse_components<cs::group_waits_enabled>(...). Both descriptions are optional; connects comes first.
template <class... Opts, class... Items>
    requires(!detail::leads_with_ties_v<Items...>)
auto fuse_components(std::string name, ConnectsDesc c, Items... items) {
    static_assert(((detail::is_fuse_option_v<Opts>) && ...),
                  "naga::cospan: the fuse_components options are cs::group_waits_enabled and cs::range_split::*");
    static_assert(detail::split_options_ok_v<Opts...>, "naga::cospan: a fused group takes at most one cs::range_split option");
    constexpr bool         waits = detail::option_count_v<group_waits_enabled, Opts...> > 0;
    constexpr detail::Split sp    = detail::split_of<Opts...>();
    return subsystem_desc<FusedGroup<waits, sp, Items...>>(std::move(name), std::move(c), TiesDesc{}, std::move(items)...);
}
/// Connections and ties.
template <class... Opts, class... Items>
auto fuse_components(std::string name, ConnectsDesc c, TiesDesc t, Items... items) {
    static_assert(((detail::is_fuse_option_v<Opts>) && ...),
                  "naga::cospan: the fuse_components options are cs::group_waits_enabled and cs::range_split::*");
    static_assert(detail::split_options_ok_v<Opts...>, "naga::cospan: a fused group takes at most one cs::range_split option");
    constexpr bool         waits = detail::option_count_v<group_waits_enabled, Opts...> > 0;
    constexpr detail::Split sp    = detail::split_of<Opts...>();
    return subsystem_desc<FusedGroup<waits, sp, Items...>>(std::move(name), std::move(c), std::move(t), std::move(items)...);
}
/// Ties, no connections.
template <class... Opts, class... Items>
    requires(!detail::leads_with_connects_v<Items...>)
auto fuse_components(std::string name, TiesDesc t, Items... items) {
    static_assert(((detail::is_fuse_option_v<Opts>) && ...),
                  "naga::cospan: the fuse_components options are cs::group_waits_enabled and cs::range_split::*");
    static_assert(detail::split_options_ok_v<Opts...>, "naga::cospan: a fused group takes at most one cs::range_split option");
    constexpr bool         waits = detail::option_count_v<group_waits_enabled, Opts...> > 0;
    constexpr detail::Split sp    = detail::split_of<Opts...>();
    return subsystem_desc<FusedGroup<waits, sp, Items...>>(std::move(name), ConnectsDesc{}, std::move(t), std::move(items)...);
}
/// Neither.
template <class... Opts, class... Items>
    requires(sizeof...(Items) >= 1 && !detail::leads_with_connects_v<Items...> && !detail::leads_with_ties_v<Items...>)
auto fuse_components(std::string name, Items... items) {
    static_assert(((detail::is_fuse_option_v<Opts>) && ...),
                  "naga::cospan: the fuse_components options are cs::group_waits_enabled and cs::range_split::*");
    static_assert(detail::split_options_ok_v<Opts...>, "naga::cospan: a fused group takes at most one cs::range_split option");
    constexpr bool         waits = detail::option_count_v<group_waits_enabled, Opts...> > 0;
    constexpr detail::Split sp    = detail::split_of<Opts...>();
    return subsystem_desc<FusedGroup<waits, sp, Items...>>(std::move(name), ConnectsDesc{}, TiesDesc{}, std::move(items)...);
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
