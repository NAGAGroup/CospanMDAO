#pragma once

// =============================================================================
// naga/cospan/system.hpp — System: structure, wiring, ordering (r01 §4, §7, §8)
// =============================================================================
//
// A System never does work: it orders, submits and gates the work its
// children describe. It has no I/O of its own beyond ALIAS entries (§4):
// promotions (peers) and tie aliases (new roots). It is a complete type: a
// System with no overrides is a plain sequential container.
//
// Everything structural (add_subsystem, connect, promote, tie_*, marks) is
// legal inside setup() only. Aliasing is complete the moment each call
// returns; storage follows in the setup chain (§11).

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
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
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/component.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

// -----------------------------------------------------------------------------
// Properties (SYCL idiom): unordered, self-naming, type-checked
// -----------------------------------------------------------------------------

namespace property {
struct device { sycl::device dev; };  ///< placement; inherited when absent
struct parallel {};                   ///< flavor: the helper form builds a ParallelSystem (§7.5)
struct transient {};                  ///< given by an OWNER at registration (Component or System): a transient registration (§4.4)
struct fused {                        ///< flavor: the helper form builds a FusedSystem (§7.5)
    enum class Mode : uint8_t { sequential, parallel };
    Mode                          mode = Mode::sequential;
    std::optional<sycl::range<3>> wg;
};
}  // namespace property

namespace detail {
template <class P>
inline constexpr bool is_property_v =
    std::is_same_v<std::remove_cvref_t<P>, property::device> ||
    std::is_same_v<std::remove_cvref_t<P>, property::parallel> ||
    std::is_same_v<std::remove_cvref_t<P>, property::transient> ||
    std::is_same_v<std::remove_cvref_t<P>, property::fused>;
}  // namespace detail

/// Constructed from any subset of the properties, each at most once.
class property_list {
public:
    property_list() = default;

    template <class... Ps>
        requires(sizeof...(Ps) >= 1 && (detail::is_property_v<Ps> && ...))
    property_list(Ps... ps) {  // NOLINT: implicit by design, SYCL idiom
        (set(std::move(ps)), ...);
    }

    template <class P>
    bool has_property() const {
        return std::get<std::optional<P>>(props_).has_value();
    }
    template <class P>
    const P& get_property() const {
        const auto& o = std::get<std::optional<P>>(props_);
        if (!o) throw std::invalid_argument("naga::cospan: the property_list does not hold that property");
        return *o;
    }

private:
    template <class P>
    void set(P p) {
        auto& o = std::get<std::optional<P>>(props_);
        if (o) throw std::invalid_argument("naga::cospan: a property was given twice");
        o = std::move(p);
    }

    std::tuple<std::optional<property::device>, std::optional<property::parallel>,
               std::optional<property::transient>, std::optional<property::fused>>
        props_;
};

enum class TieOutputsPolicy : uint8_t { require_atomic, allow_plain };

class ParallelSystem;  // parallel_system.hpp (r01 §7.5)
class FusedSystem;     // fused_system.hpp (r01 §7.5)

namespace detail {

/// child<T>(name, args...): a Component to be added by the helper form.
template <class T, class... Args>
struct ChildDecl {
    std::string                    name;
    std::tuple<std::decay_t<Args>...> args;
};

template <class>
inline constexpr bool is_child_decl_v = false;
template <class T, class... A>
inline constexpr bool is_child_decl_v<ChildDecl<T, A...>> = true;

/// Whether the first constructor argument is a property_list (selects the
/// property overload of add_subsystem).
template <class... Args>
inline constexpr bool leads_with_props_v = false;
template <class A, class... R>
inline constexpr bool leads_with_props_v<A, R...> = std::is_same_v<std::remove_cvref_t<A>, property_list>;

/// The type the helper form builds for property::parallel. It is a member of
/// a class template only so that the name is DEPENDENT: parallel_system.hpp
/// includes this header, so ParallelSystem is complete only at the point the
/// helper form is instantiated (cospan.hpp includes both).
template <class...>
struct parallel_of {
    using type = ParallelSystem;
};

/// The same dependent-name device for property::fused (fused_system.hpp).
template <class...>
struct fused_of {
    using type = FusedSystem;
};

}  // namespace detail

// -----------------------------------------------------------------------------
// System
// -----------------------------------------------------------------------------

class System : public Subsystem {
public:
    ~System() override = default;

    // ---- HOOKS ---------------------------------------------------------------
    /// Structure and wiring (legal here only).
    virtual void setup() {}
    // There is no init hook: initial values are declared with init_input (§4.4, D138).

    // ---- adding children (legal inside setup() only) ---------------------------
    /// A REALIZED System or Component type. The child is constructed, named,
    /// placed, and its run_setup has run BEFORE this returns.
    template <class T, class... Args>
        requires(std::derived_from<T, Subsystem> && !detail::leads_with_props_v<Args...>)
    T& add_subsystem(std::string name, Args&&... ctor_args) {
        return add_child<T>(std::move(name), property_list{}, std::forward<Args>(ctor_args)...);
    }
    template <class T, class... Args>
        requires std::derived_from<T, Subsystem>
    T& add_subsystem(std::string name, const property_list& props, Args&&... ctor_args) {
        return add_child<T>(std::move(name), props, std::forward<Args>(ctor_args)...);
    }

    /// A Component to be added through the helper form: type, name, ctor args.
    template <class T, class... Args>
        requires std::derived_from<T, ComponentBase>
    static auto child(std::string name, Args&&... ctor_args) {
        return detail::ChildDecl<T, Args...>{
            std::move(name), std::tuple<std::decay_t<Args>...>{std::forward<Args>(ctor_args)...}};
    }

    /// The HELPER FORM: a System configured by properties (device, transient
    /// and ONE flavor), populated with the given Components in order. With
    /// property::parallel it is a ParallelSystem (§7.5); a plain System
    /// otherwise. With property::fused it is a FusedSystem (§7.5) carrying the
    /// property's mode (sequential or parallel) and work-group.
    template <class... Children>
        requires(sizeof...(Children) >= 1 &&
                 (detail::is_child_decl_v<std::remove_cvref_t<Children>> && ...))
    System& add_subsystem(std::string name, const property_list& props, Children&&... children) {
        if (props.has_property<property::parallel>() && props.has_property<property::fused>())
            throw std::invalid_argument("naga::cospan: " + describe() + ": add_subsystem('" + name +
                                        "'): parallel and fused are two flavors; give one");
        if (props.has_property<property::parallel>())
            return populate<typename detail::parallel_of<Children...>::type>(
                std::move(name), props, std::tuple<>{}, std::forward<Children>(children)...);
        if (props.has_property<property::fused>()) {
            const property::fused& f = props.get_property<property::fused>();
            return populate<typename detail::fused_of<Children...>::type>(
                std::move(name), props,
                std::tuple<property::fused::Mode, std::optional<sycl::range<3>>>{f.mode, f.wg},
                std::forward<Children>(children)...);
        }
        return populate<System>(std::move(name), props, std::tuple<>{}, std::forward<Children>(children)...);
    }

    /// An empty System; add children with the templated overload inside its setup().
    System& add_subsystem(std::string name, const property_list& props = {}) {
        return add_child<System>(std::move(name), props);
    }

    // ---- structure: connect / promote / tie (legal inside setup() only) --------
    /// Identity: dst joins the root's set as a peer (§8.1).
    void connect(std::string_view src, std::string_view dst) { connect_impl(src, dst, std::nullopt); }
    /// dst sees a rectangular region of src's root (§8.1); composes with a
    /// view `src` carries itself (§8.2).
    void connect(std::string_view src, std::string_view dst, Slice view) {
        connect_impl(src, dst, view.src);
    }

    /// Kind-agnostic and always allowed: a PEER alias in this namespace with a
    /// snapshot of its root's flags; copies the promoted entry's view. DT must match the
    /// entry's value dtype and Rank its rank (a setup error otherwise). The alias MIRRORS the
    /// entry: its shape is read live from it (D139 part 2).
    template <DType DT, int Rank>
    void promote(std::string_view child_path, std::string promoted_name) {
        static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
        require_declaring("promote");
        check_new_name(promoted_name, "alias");
        detail::IOEntry& e = entry_at(child_path);
        check_typed<DT, Rank>(e, "promote", child_path);
        detail::IOEntry& r = *e.root();
        auto a = detail::make_alias_entry(DT, promoted_name, r.kind(), promoted_dims(e), r.can_be_source(),
                                          r.can_be_target());
        a->view_      = e.view();
        a->declarer_  = this;
        a->mirror_of_ = &e;
        detail::IOEntry::merge(r, *a);
        register_alias(std::move(a));
    }

    /// A NEW ROOT (target, not source); every member's set merges into it. The members are
    /// rank-only (cs::tied) entries of dtype DT and rank Rank; the alias OWNS the shape: its
    /// `shape` (cs::alias::fixed / from_connection / from_output_shape / from_output_values) is
    /// the root's shape, or, once an output is connected into the alias, that output's (D139 part 2).
    template <DType DT, int Rank>
    void tie_inputs(std::initializer_list<std::string_view> members, std::string name, AliasShape shape) {
        static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
        require_declaring("tie_inputs");
        check_new_name(name, "alias");
        const auto ms = resolve_members(members, "tie_inputs");
        for (const auto& [path, m] : ms) {
            const detail::IOEntry& r = *m->root();
            if (r.kind() != IOKind::input)
                fail("tie_inputs", "'" + std::string{path} + "' is not a free input: its root '" +
                                       std::string{r.name()} + "' is an output root (tie before connecting, or connect from the alias)");
            if (!r.can_be_target())
                fail("tie_inputs", "'" + std::string{path} + "' is already connected (root '" +
                                       std::string{r.name()} + "'): tie before connecting, or connect from the alias");
            if (!m->can_be_target())
                fail("tie_inputs", "'" + std::string{path} + "' is already tied: connect through its tie alias");
        }
        check_tie_members<DT, Rank>(ms, "tie_inputs");
        detail::IOEntry* upstream = check_alias_shape<Rank>(shape, "tie_inputs", /*is_input=*/true);
        auto a = detail::make_alias_entry(DT, name, IOKind::input, alias_dims<Rank>(shape),
                                          /*can_source=*/false, /*can_target=*/true);
        finish_tie(*a, std::move(shape), upstream, ms);
        for (const auto& [path, m] : ms) {  // THE EXCEPTION (§4.3): visible at every use site
            (void)path;
            m->can_be_target_ = false;
        }
        register_alias(std::move(a));
    }

    /// A NEW ROOT (source, not target); atomic member dtypes unless allow_plain. The members are
    /// rank-only (cs::tied) outputs of dtype DT and rank Rank; the alias owns the shape (an output
    /// alias defines the shape of its connections, so alias::from_connection is an error).
    template <DType DT, int Rank>
    void tie_outputs(std::initializer_list<std::string_view> members, std::string name, AliasShape shape,
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic) {
        static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
        require_declaring("tie_outputs");
        check_new_name(name, "alias");
        const auto ms = resolve_members(members, "tie_outputs");
        for (const auto& [path, m] : ms) {
            const detail::IOEntry& r = *m->root();
            if (r.kind() != IOKind::output)
                fail("tie_outputs", "'" + std::string{path} + "' is not an output (its root '" +
                                        std::string{r.name()} + "' is an input root)");
            if (!r.can_be_source() || !m->can_be_source())
                fail("tie_outputs", "'" + std::string{path} + "' is already tied: connect from its tie alias");
            for (const detail::IOEntry* p : r.peers())
                if (p->kind() == IOKind::input)
                    fail("tie_outputs", "'" + std::string{path} + "' is already connected downstream to '" +
                                            std::string{p->name()} + "': tie before connecting, or connect from the alias");
            if (policy == TieOutputsPolicy::require_atomic && !is_atomic(m->dtype()))
                fail("tie_outputs", "'" + std::string{path} +
                                        "' is not atomic (require_atomic; pass allow_plain to assert the "
                                        "producers do not race)");
        }
        check_tie_members<DT, Rank>(ms, "tie_outputs");
        detail::IOEntry* upstream = check_alias_shape<Rank>(shape, "tie_outputs", /*is_input=*/false);
        auto a = detail::make_alias_entry(DT, name, IOKind::output, alias_dims<Rank>(shape),
                                          /*can_source=*/true, /*can_target=*/false);
        finish_tie(*a, std::move(shape), upstream, ms);
        for (const auto& [path, m] : ms) {
            (void)path;
            m->can_be_source_ = false;
        }
        register_alias(std::move(a));
    }

    // ---- optimization marks (§8.4): they ride on the ROOT; the last of a kind wins
    /// Root must be a free INPUT root and fixed in shape.
    void add_design_var(std::string_view path, Bounds b = {}, Scaling s = {},
                        std::optional<SubarrayRef> region = {}) {
        require_declaring("add_design_var");
        detail::IOEntry& e = entry_at(path);
        detail::IOEntry& r = *e.root();
        if (r.kind() != IOKind::input || !r.can_be_target())
            fail("add_design_var", e.kind() == IOKind::input
                                       ? "connected inputs are computed, not designed ('" + std::string{path} +
                                             "' is fed by '" + std::string{r.name()} + "')"
                                       : "'" + std::string{path} + "' is an output; a design variable is a free input");
        if (r.is_dynamic())
            fail("add_design_var", "'" + std::string{path} + "' is dynamically shaped; a dynamic design variable is ill-formed");
        const std::size_t n = check_region(r, region, "add_design_var", path);
        check_sizes(n, b, std::nullopt, s, "add_design_var", path);
        r.design_var_ = MarkRecord{&r, region, std::move(b), std::nullopt, std::move(s)};
    }

    /// Root must be an OUTPUT root; `equals` and bounds are exclusive.
    void add_constraint(std::string_view path, Bounds b = {},
                        std::optional<std::vector<double>> equals = {}, Scaling s = {},
                        std::optional<SubarrayRef> region = {}) {
        require_declaring("add_constraint");
        detail::IOEntry& e = entry_at(path);
        detail::IOEntry& r = *e.root();
        if (r.kind() != IOKind::output)
            fail("add_constraint", "the root of '" + std::string{path} + "' ('" + std::string{r.name()} +
                                       "') is not an output root");
        if (equals && (b.lower || b.upper))
            fail("add_constraint", "'equals' and bounds are exclusive");
        const std::size_t n = check_region(r, region, "add_constraint", path);
        check_sizes(n, b, equals, s, "add_constraint", path);
        r.constraint_ = MarkRecord{&r, region, std::move(b), std::move(equals), std::move(s)};
    }

    /// Root must be an OUTPUT root; the marked region must have flat size 1.
    void add_objective(std::string_view path, Scaling s = {}, std::optional<SubarrayRef> region = {}) {
        require_declaring("add_objective");
        detail::IOEntry& e = entry_at(path);
        detail::IOEntry& r = *e.root();
        if (r.kind() != IOKind::output)
            fail("add_objective", "the root of '" + std::string{path} + "' ('" + std::string{r.name()} +
                                      "') is not an output root");
        const std::size_t n = check_region(r, region, "add_objective", path);
        if (n != 1)
            fail("add_objective", "the marked region of '" + std::string{path} + "' has flat size " +
                                      std::to_string(n) + "; an objective is one value (use region)");
        check_sizes(n, Bounds{}, std::nullopt, s, "add_objective", path);
        r.objective_ = MarkRecord{&r, region, Bounds{}, std::nullopt, std::move(s)};
    }

    // ---- retrieval & enumeration (hierarchy-aware) ----------------------------------
    using EntryRefs = std::vector<std::reference_wrapper<const detail::IOEntry>>;

    /// Typed member wrappers of a Component's declared entry, by path from
    /// this namespace. Alias entries carry no member wrapper.
    template <DType DT, int Rank>
    Input<DT, Rank>& get_input(std::string_view path) {
        return component_of(path).template get_input<DT, Rank>(leaf_of(path));
    }
    template <DType DT, int Rank>
    Output<DT, Rank>& get_output(std::string_view path) {
        return component_of(path).template get_output<DT, Rank>(leaf_of(path));
    }

    /// This System's alias entries, by the kind they snapshotted.
    EntryRefs get_inputs() const { return aliases_of(IOKind::input); }
    EntryRefs get_outputs() const { return aliases_of(IOKind::output); }

    /// OPACITY (D140 amendment): a chunked system's only interface is its own promotions. A path that descends
    /// THROUGH an opaque system into its children is refused, unless the opaque system is the CALLER resolving its own
    /// subtree. Overridden by ChunkedSystem.
    virtual bool is_opaque() const noexcept { return false; }

    /// Chains through children by dotted path. `outside`: the caller is a Model (an OUTSIDE caller of this root, even
    /// when this root is opaque itself).
    Subsystem& get_subsystem(std::string_view path, bool outside = false) {
        const auto seg = split_path(path);
        System*    s   = this;
        Subsystem* c   = nullptr;
        for (std::size_t i = 0; i < seg.size(); ++i) {
            check_opaque_descent(*s, path, outside);
            c = s->find_child(seg[i]);
            if (c == nullptr)
                throw std::out_of_range("naga::cospan: " + s->describe() + " has no subsystem '" +
                                        std::string{seg[i]} + "' (resolving '" + std::string{path} + "')");
            if (i + 1 < seg.size()) {
                s = dynamic_cast<System*>(c);
                if (s == nullptr)
                    throw std::invalid_argument("naga::cospan: '" + std::string{seg[i]} +
                                                "' is a Component; '" + std::string{path} + "' descends into it");
            }
        }
        return *c;
    }

    /// Declaration order = execution order; the Model's walks use this and nothing else.
    std::span<Subsystem* const> children() const noexcept { return child_ptrs_; }

    /// The entry a path names (Component I/O or alias), for the Model's host
    /// access and for tests; never for wiring.
    detail::IOEntry& entry_at(std::string_view path) { return entry_at_impl(path, false); }
    /// The MODEL-facing resolver: the caller is outside this system even when it is itself a chunked system, so a Model
    /// whose root is chunked sees its promotions only. (Everything else is entry_at.)
    detail::IOEntry& entry_at_from_outside(std::string_view path) { return entry_at_impl(path, true); }

    // ---- initial values (legal inside setup() only; r01 §4.4, D138) -------------------------
    /// Declare the initial value of a free input root: `path` is a child input path, or one of
    /// this System's promoted or tied input aliases. The declaration is at THIS System's depth:
    /// it wins over any declaration below it and loses to any above it; two at one depth on one
    /// root are an error at resolve. It is ignored when the input's root ends up with a source,
    /// and naming an output throws.
    void init_input(std::string_view target, InitSpec init) {
        require_declaring("init_input");
        detail::IOEntry& e = entry_at(target);
        if (e.kind() != IOKind::input)
            fail("init_input", "'" + std::string{target} + "' is an output; outputs are never initialized");
        e.init_decls_.push_back(detail::IOEntry::InitDecl{std::move(init), detail::depth_of(*this), path()});
    }

    // ---- Subsystem verbs: the DEFAULT orchestration (r01 §7.2, §11.2) ------------------------
    /// setup(); children's run_setup fired inside add_subsystem. The ROOT
    /// (no owner) then runs resolve(): only now is every registration a later
    /// structural call could add known (§4.4).
    void run_setup() override {
        {
            detail::FlagScope scope{declaring_};
            setup();
        }
        if (owner() == nullptr) resolve();
    }

    /// ctx.device = the placed device or `parent_device` (the owner passes the
    /// child's placed device as its parent_device). OWN alias ROOTS first
    /// (a tie alias root counts its bytes here and fixes the device its
    /// members' Components are checked against), then children in order, then
    /// own PEER aliases (promotions: they register this System's device on a
    /// root the children have already placed). Maps merge (sum on key).
    std::unordered_map<sycl::device, std::size_t> get_exec_info(const sycl::device& parent_device) override {
        ctx_.device = parent_device;
        std::unordered_map<sycl::device, std::size_t> total;
        total[parent_device] += 0;
        for (detail::IOEntry* a : alias_order_)
            if (a->is_root()) {
                place_alias(*a, parent_device);
                a->set_primary_placement(a->placement());
                total[parent_device] += a->static_bytes();
            }
        child_ctx_.clear();
        for (std::size_t i = 0; i < children_.size(); ++i) {
            ExecCtx c = ctx_;
            c.device  = placed_[i].value_or(parent_device);
            child_ctx_.push_back(c);
            for (const auto& [d, bytes] : children_[i]->get_exec_info(c.device)) total[d] += bytes;
        }
        for (detail::IOEntry* a : alias_order_)
            if (!a->is_root()) {
                place_alias(*a, parent_device);
                detail::IOEntry& root = *a->root();
                const Placement  own  = a->placement();
                if (!registered(root, own)) {
                    root.register_placement(own);
                    if (own.kind == sk::kind::device) total[parent_device] += root.static_bytes();
                }
            }
        return total;
    }

    /// Complete ctx (the Model's queue; group null; this device's substrate),
    /// allocate own alias roots, children, point alias wrappers.
    void prepare_exec(sk::queue& queue, Substrates& substrates) override {
        queue_ = &queue;
        subs_  = &substrates;
        ctx_.queue     = &queue;
        ctx_.group     = nullptr;
        ctx_.substrate = &substrates[ctx_.device];
        for (ExecCtx& c : child_ctx_) {
            c.queue     = &queue;
            c.group     = nullptr;
            c.substrate = &substrates[c.device];
        }
        // No allocation here: the owning Component allocates each root in its pre_compute (§4.4).
        for (detail::IOEntry* a : alias_order_)
            if (a->is_root()) a->bind_substrates(substrates);
        for (auto& c : children_) c->prepare_exec(queue, substrates);
        for (detail::IOEntry* a : alias_order_)
            if (a->has_storage()) a->point_wrapper();
    }

    void compile() override {
        for (auto& c : children_) c->compile();
    }

    void run_pre_compute(ExecCtx&) override {}   ///< Systems do no work
    void run_post_compute(ExecCtx&) override {}

    /// THE DEFAULT LOOP (§7.3): per child pre_compute -> compute_forward ->
    /// submit (through this System's queue, to the child's device) ->
    /// post_compute (the Components allocate and release their own roots there, §4.4).
    /// "I submitted": null.
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        for (std::size_t i = 0; i < children_.size(); ++i) {
            Subsystem& c  = *children_[i];
            ExecCtx&   cc = child_ctx_[i];
            c.run_pre_compute(cc);
            if (auto d = c.run_compute_forward()) {
                SubmitTarget t = cc.target();
                d->submit(t);
            }
            c.run_post_compute(cc);
        }
        return nullptr;
    }

    /// The ROOT System's end-of-setup pass (§4.4, §11.2.1; run_setup runs it when this System has
    /// no owner). Per root it records: `transient` (EVERY registration in the alias set is
    /// transient: the registering subsystem or any ancestor was added with property::transient),
    /// the winning initial value (smallest declaring depth; two at one depth throw; an init on a
    /// root with a source is ignored), and `last_user` (the last Component, in depth-first
    /// registration order, that touches the root). A mark on a transient root throws.
    void resolve() {
        classify_chunks();  // D140 B2: ChunkedSystems classify their roots and re-root boundary ones first
        std::vector<detail::IOEntry*> roots;
        collect_roots(roots);

        std::vector<ComponentBase*> comps;
        collect_components(comps);
        for (detail::IOEntry* r : roots) {
            r->last_user_ = nullptr;
            r->owner_     = nullptr;
        }
        for (ComponentBase* c : comps)
            for (detail::IOEntry* e : c->order_) {
                detail::IOEntry* r = e->root();
                r->last_user_      = c;
                if (r->owner_ == nullptr) r->owner_ = c;  // the FIRST toucher owns the root
            }
        for (detail::IOEntry* r : roots)
            if (r->chunk_rerooted_) r->owner_ = nullptr;  // D140 B2: its ChunkedSystem allocates it, not a Component

        for (detail::IOEntry* r : roots) {
            bool all = registration_transient(*r);
            for (const detail::IOEntry* p : r->peers_) all = all && registration_transient(*p);
            const bool chunk_internal = !r->chunk_internal_by_.empty();  // D140 B2: forced transient, per chunk
            if (chunk_internal) all = true;
            if (r->chunk_rerooted_) all = false;                        // consumed outside: never transient
            r->transient_    = all;
            r->released_     = all;  // a transient root has no storage until its first user allocates it
            r->transient_by_ = all ? (chunk_internal ? r->chunk_internal_by_ : topmost_transient(*r)) : std::string{};
            if (all && (r->design_var_ || r->constraint_ || r->objective_))
                throw std::invalid_argument("naga::cospan: '" + r->full_path() +
                                            "' is marked (design variable, constraint or objective) but is "
                                            "transient (subsystem '" + r->transient_by_ +
                                            "'); a marked root must outlive a forward");
            resolve_init(*r);
        }
        check_ties(comps);  // D139 part 2: rank-only entries are tied; alias shape sources are ordered
    }

protected:
    /// D140 B2: called on every System of the tree (children first) at the start of resolve(); a
    /// ChunkedSystem classifies its roots here.
    virtual void chunk_classify() {}

private:
    friend class ChunkedSystem;

    void classify_chunks() {
        for (Subsystem* c : child_ptrs_)
            if (auto* s = dynamic_cast<System*>(c)) s->classify_chunks();
        chunk_classify();
    }

    static bool registration_transient(const detail::IOEntry& e) {
        for (const Subsystem* s = e.declarer(); s != nullptr; s = s->owner())
            if (s->transient()) return true;
        return false;
    }

    /// The path of the topmost transient subsystem on the root's own registration chain.
    static std::string topmost_transient(const detail::IOEntry& r) {
        std::string by;
        for (const Subsystem* s = r.declarer(); s != nullptr; s = s->owner())
            if (s->transient()) by = s->path();
        return by;
    }

    /// The winning init of a root: the declaration of the smallest depth among its alias set's.
    static void resolve_init(detail::IOEntry& r) {
        if (r.model_init_) return;  // Model::init_input (D139) wins and is never overwritten
        r.init_spec_.reset();
        if (r.kind() == IOKind::output) return;  // the root has a source: the inputs' inits are ignored
        const detail::IOEntry::InitDecl* best  = nullptr;
        const detail::IOEntry::InitDecl* clash = nullptr;
        const auto consider = [&](const detail::IOEntry& e) {
            for (const auto& d : e.init_decls_) {
                if (best == nullptr || d.depth < best->depth) {
                    best  = &d;
                    clash = nullptr;
                } else if (d.depth == best->depth) {
                    clash = &d;
                }
            }
        };
        consider(r);
        for (const detail::IOEntry* p : r.peers_) consider(*p);
        if (clash != nullptr)
            throw std::invalid_argument("naga::cospan: two initial values are declared at depth " +
                                        std::to_string(best->depth) + " for '" + r.full_path() + "' (by '" +
                                        best->by + "' and '" + clash->by +
                                        "'); the highest level wins, and one level may declare only one");
        if (best != nullptr) r.init_spec_ = best->spec;
    }

    void collect_roots(std::vector<detail::IOEntry*>& out) {
        for (detail::IOEntry* a : alias_order_)
            if (a->is_root()) out.push_back(a);
        for (Subsystem* c : child_ptrs_) {
            if (auto* s = dynamic_cast<System*>(c)) {
                s->collect_roots(out);
            } else if (auto* comp = dynamic_cast<ComponentBase*>(c)) {
                for (detail::IOEntry* e : comp->order_)
                    if (e->is_root()) out.push_back(e);
            }
        }
    }

    /// Depth-first in registration order: the order of "last user".
    void collect_components(std::vector<ComponentBase*>& out) {
        for (Subsystem* c : child_ptrs_) {
            if (auto* s = dynamic_cast<System*>(c))
                s->collect_components(out);
            else if (auto* comp = dynamic_cast<ComponentBase*>(c))
                out.push_back(comp);
        }
    }

    // ---- placement ----------------------------------------------------------------------------
    static bool registered(const detail::IOEntry& root, const Placement& p) {
        for (const Placement& e : root.storage_placements())
            if (e == p) return true;
        return false;
    }
    /// An alias lives on this System's device, in device-kind storage.
    static void place_alias(detail::IOEntry& a, const sycl::device& dev) {
        a.device_       = dev;
        a.storage_kind_ = sk::kind::device;
    }

    // ---- helpers ---------------------------------------
    sk::queue& queue_ref() const {
        if (queue_ == nullptr)
            throw std::logic_error("naga::cospan: " + describe() + " has no queue yet (prepare_exec)");
        return *queue_;
    }
    detail::IOEntry& entry_of_kind(std::string_view path, IOKind k) {
        detail::IOEntry& e = entry_at(path);
        if (e.kind() != k)
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{path} + "' is an " +
                                        (k == IOKind::input ? "output" : "input"));
        return e;
    }
private:
    detail::IOEntry& entry_at_impl(std::string_view path, bool outside) {
        const Located l = locate(path, outside);
        detail::IOEntry* e = l.comp != nullptr ? l.comp->lookup_entry(l.leaf) : l.sys->find_alias(l.leaf);
        if (e == nullptr)
            throw std::out_of_range("naga::cospan: " + l.sys->describe() + " has no " +
                                    (l.comp != nullptr ? "entry '" : "alias '") + std::string{l.leaf} +
                                    "' (resolving '" + std::string{path} + "')");
        return *e;
    }

    /// About to look up a child of `s` while resolving `path` from this System: refuse it when `s` is an opaque
    /// system other than the caller (or the caller itself, for an outside caller).
    void check_opaque_descent(const System& s, std::string_view path, bool outside) const {
        if (!s.is_opaque() || (&s == this && !outside)) return;
        const std::string full = this->path().empty() || outside ? std::string{path} : this->path() + "." + std::string{path};
        throw std::invalid_argument("naga::cospan: '" + full + "' lives inside chunked system '" + s.Subsystem::path() +
                                    "'; only its promotions are visible outside it; promote it to use it here");
    }

    /// What a path resolves to: the System reached and either one of its
    /// Components' (comp != null) or its own alias (comp == null) named `leaf`.
    struct Located {
        System*        sys;
        ComponentBase* comp;
        std::string_view leaf;
    };

    static std::vector<std::string_view> split_path(std::string_view p) {
        std::vector<std::string_view> out;
        std::size_t                   i = 0;
        while (true) {
            const std::size_t j = p.find('.', i);
            out.push_back(p.substr(i, j == std::string_view::npos ? std::string_view::npos : j - i));
            if (j == std::string_view::npos) break;
            i = j + 1;
        }
        for (const std::string_view s : out)
            if (s.empty())
                throw std::invalid_argument("naga::cospan: the path '" + std::string{p} + "' has an empty segment");
        return out;
    }

    /// Dotted, descending children from THIS namespace; the last segment is an
    /// alias of the System reached or a Component's declared entry (§10.3).
    Located locate(std::string_view path, bool outside = false) {
        const auto seg = split_path(path);
        System*    s   = this;
        for (std::size_t i = 0; i + 1 < seg.size(); ++i) {
            check_opaque_descent(*s, path, outside);
            Subsystem* c = s->find_child(seg[i]);
            if (c == nullptr)
                throw std::out_of_range("naga::cospan: " + s->describe() + " has no subsystem '" +
                                        std::string{seg[i]} + "' (resolving '" + std::string{path} + "')");
            if (auto* sub = dynamic_cast<System*>(c)) {
                s = sub;
                continue;
            }
            auto* comp = dynamic_cast<ComponentBase*>(c);
            if (comp == nullptr || i + 2 != seg.size())
                throw std::invalid_argument("naga::cospan: '" + std::string{seg[i]} +
                                            "' is a Component; '" + std::string{path} + "' descends past it");
            return Located{s, comp, seg.back()};
        }
        return Located{s, nullptr, seg.back()};
    }

    ComponentBase& component_of(std::string_view path) {
        const Located l = locate(path);
        if (l.comp == nullptr)
            throw std::invalid_argument("naga::cospan: '" + std::string{path} + "' is an alias entry of " +
                                        l.sys->describe() + "; aliases have no member wrapper");
        return *l.comp;
    }
    static std::string_view leaf_of(std::string_view path) {
        const std::size_t j = path.rfind('.');
        return j == std::string_view::npos ? path : path.substr(j + 1);
    }

    Subsystem* find_child(std::string_view name) const {
        const auto it = child_index_.find(std::string{name});
        return it == child_index_.end() ? nullptr : child_ptrs_[it->second];
    }
    detail::IOEntry* find_alias(std::string_view name) const {
        const auto it = aliases_.find(std::string{name});
        return it == aliases_.end() ? nullptr : it->second.get();
    }

protected:
    /// "the root System" or "System 'path'": the prefix of every System error.
    std::string describe() const {
        return path().empty() ? std::string{"the root System"} : "System '" + path() + "'";
    }

private:
    EntryRefs aliases_of(IOKind k) const {
        EntryRefs out;
        for (const detail::IOEntry* a : alias_order_)
            if (a->kind() == k) out.emplace_back(*a);
        return out;
    }

    // ---- guards & names --------------------------------------------------------------
    void require_declaring(const char* what) const {
        if (!declaring_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside setup() only (" +
                                   describe() + ")");
    }

    [[noreturn]] void fail(const char* what, const std::string& why) const {
        throw std::invalid_argument(std::string{"naga::cospan: "} + describe() + ": " + what + ": " + why);
    }

    void check_new_name(const std::string& name, const char* what) const {
        if (name.empty() || name.find('.') != std::string::npos)
            throw std::invalid_argument("naga::cospan: " + describe() + ": the " + what + " name '" + name +
                                        "' must be non-empty and contain no '.'");
        if (child_index_.contains(name) || aliases_.contains(name))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + name +
                                        "' is already a subsystem or alias name here");
    }

    // ---- adding children ---------------------------------------------------------------
    template <class T, class... Args>
    T& add_child(std::string name, const property_list& props, Args&&... ctor_args) {
        require_declaring("add_subsystem");
        check_new_name(name, "subsystem");
        // property::fused is a flavor like parallel: the helper form builds a
        // FusedSystem; a typed child names its own type (derive from FusedSystem).
        if (props.has_property<property::fused>() && !std::is_base_of_v<FusedSystem, T>)
            throw std::invalid_argument("naga::cospan: " + describe() + ": add_subsystem('" + name +
                                        "'): property::fused builds a FusedSystem in the helper form "
                                        "(add_subsystem(name, props, child<T>(...)...)); a typed child names "
                                        "its own type, so derive it from FusedSystem");
        // property::parallel selects a flavor, which only the helper form can
        // build; a typed child names its own type (derive from ParallelSystem).
        if (props.has_property<property::parallel>() && !std::is_base_of_v<ParallelSystem, T>)
            throw std::invalid_argument("naga::cospan: " + describe() + ": add_subsystem('" + name +
                                        "'): property::parallel builds a ParallelSystem in the helper form "
                                        "(add_subsystem(name, props, child<T>(...)...)); a typed child names "
                                        "its own type, so derive it from ParallelSystem");
        auto owned = std::make_unique<T>(std::forward<Args>(ctor_args)...);
        T&   ref   = *owned;
        attach(std::move(owned), std::move(name), props);
        ref.run_setup();  // fully declared before the caller's next structural call
        return ref;
    }

    /// The helper form's body: build a child System of type S (System, or
    /// ParallelSystem for property::parallel) and add the given Components to
    /// it in order. add_child<S> has already run S::setup(), which left the
    /// child's declaring flag off, so it is raised here for the adds.
    template <class S, class Ctor, class... Children>
    System& populate(std::string name, const property_list& props, Ctor ctor, Children&&... children) {
        // `ctor` is a tuple of S's constructor arguments (empty for System and ParallelSystem).
        S& sys = std::apply(
            [&](auto&&... a) -> S& {
                return this->add_child<S>(std::move(name), props, std::forward<decltype(a)>(a)...);
            },
            std::move(ctor));
        detail::FlagScope scope{sys.declaring_};
        (sys.add_decl(std::forward<Children>(children)), ...);
        return sys;
    }

    template <class T, class... A>
    void add_decl(detail::ChildDecl<T, A...> d) {
        std::apply([&](auto&&... a) { this->add_child<T>(std::move(d.name), property_list{}, std::move(a)...); },
                   std::move(d.args));
    }

    /// Name, path, owner, placed device, transient: the owner writes them
    /// (Subsystem's friend).
    void attach(std::unique_ptr<Subsystem> c, std::string name, const property_list& props) {
        const bool tr = props.has_property<property::transient>();
        Subsystem& b = *c;
        b.path_      = path().empty() ? name : path() + "." + name;
        b.name_      = name;
        b.owner_     = this;
        b.transient_ = tr;
        placed_.push_back(props.has_property<property::device>()
                              ? std::optional<sycl::device>{props.get_property<property::device>().dev}
                              : std::nullopt);
        child_index_.emplace(std::move(name), child_ptrs_.size());
        child_ptrs_.push_back(c.get());
        children_.push_back(std::move(c));
    }

    void register_alias(std::unique_ptr<detail::IOEntry> a) {
        detail::IOEntry* p = a.get();
        alias_order_.push_back(p);
        aliases_.emplace(std::string{p->name()}, std::move(a));
    }

    // ---- alias shape (D139 part 2) ---------------------------------------------------------
    using Members = std::vector<std::pair<std::string_view, detail::IOEntry*>>;

    /// A promotion's dims. Its shape is read live from the entry (it mirrors it), so the dims are
    /// only classification: fixed and from_root kept (rank-only stays rank-only), derived ones
    /// become inert markers.
    static std::vector<detail::DimRecord> promoted_dims(const detail::IOEntry& e) {
        std::vector<detail::DimRecord> out;
        for (int d = 0; d < e.rank(); ++d) {
            detail::DimRecord r = e.dim(d);
            if (r.how != detail::DimRecord::How::fixed && r.how != detail::DimRecord::How::from_root) {
                r.how         = detail::DimRecord::How::from_input;
                r.input.clear();
                r.dim         = 0;
                r.value_index = 0;
            }
            out.push_back(std::move(r));
        }
        return out;
    }

    /// A tie alias's dims: fixed extents for alias::fixed, placeholders (resolved by the root's
    /// owner, extent 0 until then) for the others.
    template <int Rank>
    static std::vector<detail::DimRecord> alias_dims(const AliasShape& s) {
        std::vector<detail::DimRecord> out(static_cast<std::size_t>(Rank));
        for (int d = 0; d < Rank; ++d) {
            detail::DimRecord& r = out[static_cast<std::size_t>(d)];
            if (s.kind == AliasShape::Kind::fixed) {
                r.how    = detail::DimRecord::How::fixed;
                r.extent = s.extents[static_cast<std::size_t>(d)];
            } else if (s.kind == AliasShape::Kind::dims &&
                       s.dim_specs[static_cast<std::size_t>(d)].how == AliasDim::How::fixed) {
                r.how    = detail::DimRecord::How::fixed;
                r.extent = s.dim_specs[static_cast<std::size_t>(d)].extent;
            } else {
                r.how = detail::DimRecord::How::from_input;
                if (s.kind == AliasShape::Kind::dims) {  // D140: statically per-index dims (resolved by the owner)
                    const AliasDim& sp = s.dim_specs[static_cast<std::size_t>(d)];
                    if (sp.how == AliasDim::How::per_index_host || sp.how == AliasDim::How::output_values) r.per = true;
                }
            }
        }
        return out;
    }

    template <DType DT, int Rank>
    void check_typed(const detail::IOEntry& e, const char* what, std::string_view path) const {
        if (!dtype_compatible(e.dtype(), DT))
            fail(what, "'" + std::string{path} + "' has a different dtype than the alias (the alias is declared <DType " +
                           std::to_string(static_cast<int>(DT)) + ", " + std::to_string(Rank) + ">)");
        if (e.rank() != Rank)
            fail(what, "'" + std::string{path} + "' has rank " + std::to_string(e.rank()) + " but the alias is declared rank " +
                           std::to_string(Rank));
    }

    /// Every member is of dtype DT and rank Rank, and rank-only (declared cs::tied).
    template <DType DT, int Rank>
    void check_tie_members(const Members& ms, const char* what) const {
        for (const auto& [path, m] : ms) {
            check_typed<DT, Rank>(*m, what, path);
            if (!m->is_rank_only())
                fail(what, "'" + std::string{path} + "' declares a shape; a tie member is rank-only (declare it with cs::tied) "
                           "because the tie alias owns the shape");
        }
    }

    /// Validates an AliasShape against the alias's Rank; returns the upstream output a
    /// from_output_* shape reads (null for the others).
    template <int Rank>
    detail::IOEntry* check_alias_shape(AliasShape& s, const char* what, bool is_input) {
        switch (s.kind) {
            case AliasShape::Kind::dims: {
                if (s.rank != Rank || static_cast<int>(s.dim_specs.size()) != Rank)
                    fail(what, "alias::dims gives " + std::to_string(s.dim_specs.size()) + " specs but the alias is rank " +
                                   std::to_string(Rank));
                for (int d = 0; d < Rank; ++d) {
                    AliasDim&         sp  = s.dim_specs[static_cast<std::size_t>(d)];
                    const std::string dim = "alias::dims, dimension " + std::to_string(d) + ": ";
                    if (sp.how == AliasDim::How::fixed) continue;
                    if (sp.how == AliasDim::How::per_index_host) {
                        if (d == 0) fail(what, dim + "a per-index spec is for dimensions >= 1 (dimension 0 is the number of indices)");
                        const AliasDim& d0 = s.dim_specs[0];
                        if (d0.how == AliasDim::How::fixed && sp.values.size() != d0.extent)
                            fail(what, dim + "per_index has " + std::to_string(sp.values.size()) +
                                           " extents but dimension 0 is " + std::to_string(d0.extent));
                        continue;
                    }
                    if (sp.how == AliasDim::How::output_values && d == 0)
                        fail(what, dim + "output_values gives per-index extents, which are for dimensions >= 1");
                    detail::IOEntry& up = entry_at(sp.path);
                    if (sp.input) {
                        if (up.kind() != IOKind::input)
                            fail(what, dim + "'" + sp.path + "' is not an input (input_value / input_values / input_dim name an "
                                           "input of the declaring system: a promoted input or a tie_inputs alias)");
                    } else if (up.kind() != IOKind::output || !up.can_be_source() || !up.root()->can_be_source()) {
                        fail(what, dim + "'" + sp.path + "' is not an output that can be a source");
                    }
                    if (sp.how != AliasDim::How::output_dim && !detail::is_integer_dtype(up.dtype()))
                        fail(what, dim + "'" + sp.path + "' is not of integer dtype");
                    if (sp.how == AliasDim::How::output_values && up.rank() > 2)
                        fail(what, dim + "'" + sp.path + "' drives per-index extents by value and must be rank 1 [N] or rank 2 [N,k]");
                    if (sp.how == AliasDim::How::output_dim && static_cast<int>(sp.index) >= up.rank())
                        fail(what, dim + "'" + sp.path + "' has rank " + std::to_string(up.rank()) + ", no dimension " +
                                       std::to_string(sp.index));
                    sp.src = &up;
                }
                return nullptr;
            }
            case AliasShape::Kind::fixed:
                if (s.rank != Rank)
                    fail(what, "alias::fixed gives " + std::to_string(s.rank) + " extents but the alias is rank " +
                                   std::to_string(Rank));
                return nullptr;
            case AliasShape::Kind::from_connection:
                if (!is_input)
                    fail(what, "an output alias defines the shape of its connections; alias::from_connection is for tie_inputs");
                return nullptr;
            case AliasShape::Kind::from_input_shape: {
                detail::IOEntry& up = entry_at(s.path);
                if (up.kind() != IOKind::input)
                    fail(what, "alias::from_input_shape('" + s.path + "') names an entry that is not an input");
                if (up.rank() != Rank)
                    fail(what, "alias::from_input_shape('" + s.path + "') names an input of rank " +
                                   std::to_string(up.rank()) + " but the alias is rank " + std::to_string(Rank));
                return &up;
            }
            case AliasShape::Kind::from_output_shape:
            case AliasShape::Kind::from_output_values: {
                detail::IOEntry& up = entry_at(s.path);
                if (up.kind() != IOKind::output || !up.can_be_source() || !up.root()->can_be_source())
                    fail(what, "the alias shape names '" + s.path + "', which is not an output that can be a source");
                if (s.kind == AliasShape::Kind::from_output_shape && up.rank() != Rank)
                    fail(what, "alias::from_output_shape('" + s.path + "') names an output of rank " +
                                   std::to_string(up.rank()) + " but the alias is rank " + std::to_string(Rank));
                if (s.kind == AliasShape::Kind::from_output_values && !detail::is_integer_dtype(up.dtype()))
                    fail(what, "alias::from_output_values('" + s.path + "') names an output that is not of integer dtype");
                return &up;
            }
        }
        return nullptr;
    }

    /// Records the tie on the alias, merges the members' sets into it, and marks every entry of
    /// the merged set as tied (resolve() requires each rank-only entry to be in a tie's set).
    void finish_tie(detail::IOEntry& a, AliasShape shape, detail::IOEntry* upstream, const Members& ms) {
        a.declarer_    = this;
        a.alias_shape_ = std::move(shape);
        a.alias_src_   = upstream;
        for (const auto& [path, m] : ms) {
            (void)path;
            a.tie_members_.push_back(m);
            if (m->root() != &a) detail::IOEntry::merge(a, *m->root());
        }
        for (detail::IOEntry* p : a.peers_) p->tie_alias_ = &a;
    }

    Members resolve_members(std::initializer_list<std::string_view> members, const char* what) {
        if (members.size() < 2) fail(what, "a tie needs two or more members; use promote for one");
        Members out;
        for (const std::string_view m : members) {
            detail::IOEntry* e = &entry_at(m);
            for (const auto& [p, o] : out)
                if (o == e) fail(what, "'" + std::string{m} + "' is listed twice (as '" + std::string{p} + "')");
            out.emplace_back(m, e);
        }
        return out;
    }

    // ---- ties, checked once at the end of setup (D139 part 2) ------------------------------
    void collect_aliases(std::vector<detail::IOEntry*>& out) {
        for (detail::IOEntry* a : alias_order_) out.push_back(a);
        for (Subsystem* c : child_ptrs_)
            if (auto* s = dynamic_cast<System*>(c)) s->collect_aliases(out);
    }

    /// Every rank-only entry is in a tie's set; a from_connection alias is connected; a
    /// from_output_* alias reads an output whose producer runs BEFORE the tie's first member.
    void check_ties(const std::vector<ComponentBase*>& comps) {
        for (ComponentBase* c : comps)
            for (detail::IOEntry* e : c->order_) {
                if (!e->is_rank_only()) continue;
                const detail::IOEntry* r    = e->root();
                bool                   tied = r->tie_alias_ != nullptr;
                for (const detail::IOEntry* p : r->peers_) tied = tied || p->tie_alias_ != nullptr;
                if (!tied)
                    throw std::invalid_argument("naga::cospan: '" + e->full_path() +
                                                "' is rank-only (cs::tied) but is not a member of a tie "
                                                "(tie_inputs / tie_outputs, directly or through promotions): "
                                                "only a tie alias can give it a shape");
            }

        std::unordered_map<const Subsystem*, std::size_t> index;
        for (std::size_t i = 0; i < comps.size(); ++i) index.emplace(comps[i], i);
        std::vector<detail::IOEntry*> aliases;
        collect_aliases(aliases);
        for (detail::IOEntry* a : aliases) {
            if (!a->alias_shape_) continue;
            const AliasShape& s = *a->alias_shape_;
            if (s.kind == AliasShape::Kind::from_connection && a->is_root())
                throw std::invalid_argument("naga::cospan: tie alias '" + a->full_path() +
                                            "' takes its shape from a connection (alias::from_connection) but nothing is "
                                            "connected to it");
            if (s.kind != AliasShape::Kind::from_output_shape && s.kind != AliasShape::Kind::from_output_values &&
                s.kind != AliasShape::Kind::dims && s.kind != AliasShape::Kind::from_input_shape)
                continue;

            // The tie's owner: the first member to run (members seen through their promotions).
            std::size_t      first = comps.size();
            std::string      first_path;
            for (const detail::IOEntry* m : a->tie_members_) {
                while (m->mirror_of_ != nullptr) m = m->mirror_of_;
                const auto it = index.find(m->declarer());
                if (it != index.end() && it->second < first) {
                    first      = it->second;
                    first_path = m->declarer()->path();
                }
            }
            // The producer of EACH upstream output the shape reads: the LAST component writing its root.
            for (const auto& [src_path, up] : a->alias_sources()) {
                const detail::IOEntry* root     = up->root();
                bool                   found    = false;
                std::size_t            producer = 0;
                std::string            producer_path;
                const auto             consider = [&](const detail::IOEntry* x) {
                    if (x->kind() != IOKind::output) return;
                    const auto it = index.find(x->declarer());
                    if (it == index.end()) return;
                    if (!found || it->second > producer) {
                        producer      = it->second;
                        producer_path = x->declarer()->path();
                    }
                    found = true;
                };
                consider(root);
                for (const detail::IOEntry* p : root->peers_) consider(p);
                if (!found && up->kind() == IOKind::input) {
                    // A FREE input root (no producer): it is allocated, and its init applied, by its owner (the
                    // first Component to touch it); that owner must run no later than the alias's owner, which
                    // reads it on the host. A root re-rooted by a ChunkedSystem exists before the first chunk.
                    const auto oit = root->owner_ != nullptr ? index.find(root->owner_) : index.end();
                    if (root->owner_ != nullptr && !root->chunk_rerooted_ && oit != index.end() && oit->second > first)
                        throw std::invalid_argument("naga::cospan: tie alias '" + a->full_path() + "' takes its shape from the free input '" +
                                                    std::string{src_path} + "', whose owner '" + root->owner_->path() +
                                                    "' is registered after the tie's first member ('" + first_path +
                                                    "'); the value is read before the members run (register the input's owner earlier)");
                    continue;
                }
                if (!found)
                    throw std::invalid_argument("naga::cospan: tie alias '" + a->full_path() + "' takes its shape from '" +
                                                std::string{src_path} + "', which no component produces");
                if (producer >= first)
                    throw std::invalid_argument("naga::cospan: tie alias '" + a->full_path() + "' takes its shape from '" +
                                                std::string{src_path} + "', produced by '" + producer_path +
                                                "', which is not registered before the tie's first member ('" + first_path +
                                                "'); the shape is read before the members run");
            }
        }
    }

    // ---- connect ---------------------------------------------------------------------------
    void connect_impl(std::string_view src, std::string_view dst, std::optional<SubarrayRef> slice) {
        require_declaring("connect");
        detail::IOEntry& s  = entry_at(src);
        detail::IOEntry& d  = entry_at(dst);
        detail::IOEntry& sr = *s.root();
        detail::IOEntry& dr = *d.root();
        const std::string what = "connect('" + std::string{src} + "', '" + std::string{dst} + "')";
        if (!s.can_be_source())
            fail(what.c_str(), "'" + std::string{src} + "' cannot be a source (an input, or a tied output: connect from its tie alias)");
        if (!sr.can_be_source())
            fail(what.c_str(), "the root of '" + std::string{src} + "' ('" + std::string{sr.name()} + "') cannot be a source");
        if (!d.can_be_target())
            fail(what.c_str(), "'" + std::string{dst} + "' cannot be a target (an output, or a tied input: connect through its tie alias)");
        if (!dr.can_be_target())
            fail(what.c_str(), "'" + std::string{dst} + "' already has a source (its root '" + std::string{dr.name()} +
                                   "' is not an input root); an input has at most one");
        if (!dtype_compatible(s.dtype(), d.dtype()) || !dtype_compatible(sr.dtype(), dr.dtype()))
            fail(what.c_str(), "the value dtypes differ");

        // D140: a Slice cannot select a region of a ragged root (no rectangular geometry).
        if (slice && (sr.declared_ragged() || sr.is_ragged_port() || s.is_ragged_port() || d.is_ragged_port() ||
                      dr.is_ragged_port()))
            fail(what.c_str(), "'" + std::string{sr.name()} + "' is ragged; a Slice view onto a ragged root is not supported");

        // View: what `src` sees (B), composed with the Slice (A).
        std::optional<SubarrayRef> view = s.view();
        if (slice) {
            const SubarrayRef& a = *slice;
            if (a.rank != sr.rank())
                fail(what.c_str(), "the Slice is rank " + std::to_string(a.rank) + " but the root '" +
                                       std::string{sr.name()} + "' is rank " + std::to_string(sr.rank()) +
                                       " (views are in the root's rank)");
            for (int k = 0; k < 3; ++k)
                if (a.extents[static_cast<std::size_t>(k)] == 0) fail(what.c_str(), "the Slice is empty");
            if (view) {
                for (std::size_t k = 0; k < 3; ++k)
                    if (a.offset[k] + a.extents[k] > view->extents[k])
                        fail(what.c_str(), "the Slice is not contained in the view '" + std::string{src} +
                                               "' carries (view extents " + dims_text(view->extents) + ", slice ends at " +
                                               std::to_string(a.offset[k] + a.extents[k]) + " on dim " + std::to_string(k) + ")");
                for (std::size_t k = 0; k < 3; ++k) view->offset[k] += a.offset[k];
                view->extents = a.extents;
            } else {
                if (!sr.is_dynamic()) {
                    const sycl::range<3> e = sr.extents();
                    for (std::size_t k = 0; k < 3; ++k)
                        if (a.offset[k] + a.extents[k] > e[k])
                            fail(what.c_str(), "the Slice is not inside '" + std::string{sr.name()} + "' on dim " + std::to_string(k));
                }
                view = a;
            }
        }

        // Everything in dst's set shares dst's data: they all see the region.
        std::vector<detail::IOEntry*> dset{&dr};
        for (detail::IOEntry* p : dr.peers_) dset.push_back(p);
        if (view)
            for (const detail::IOEntry* e : dset)
                if (e->rank() != sr.rank())
                    fail(what.c_str(), "'" + std::string{e->name()} + "' (rank " + std::to_string(e->rank()) +
                                           ") would see a subarray of '" + std::string{sr.name()} +
                                           "' and must have the root's rank (" + std::to_string(sr.rank()) + ")");
        if (view)
            for (detail::IOEntry* e : dset) e->view_ = view;
        detail::IOEntry::merge(sr, dr);
    }

    static std::string dims_text(const std::array<std::size_t, 3>& a) {
        return std::to_string(a[0]) + "x" + std::to_string(a[1]) + "x" + std::to_string(a[2]);
    }

    // ---- marks ---------------------------------------------------------------------------------
    /// The flat size a mark covers: the region's, or the root's; validates the region.
    std::size_t check_region(const detail::IOEntry& r, const std::optional<SubarrayRef>& region,
                             const char* what, std::string_view path) const {
        if (!region) return r.flat_size();
        if (region->rank != r.rank())
            fail(what, "the region of '" + std::string{path} + "' is rank " + std::to_string(region->rank) +
                           " but the root is rank " + std::to_string(r.rank()));
        if (!r.is_dynamic()) {
            const sycl::range<3> e = r.extents();
            for (std::size_t k = 0; k < 3; ++k)
                if (region->offset[k] + region->extents[k] > e[k])
                    fail(what, "the region of '" + std::string{path} + "' is not inside the root '" +
                                   std::string{r.name()} + "'");
        }
        return region->flat_size();
    }

    /// Bounds, equals and scaling are per-element or scalar (size 1 broadcasts).
    void check_sizes(std::size_t n, const Bounds& b, const std::optional<std::vector<double>>& eq,
                     const Scaling& s, const char* what, std::string_view path) const {
        auto ok = [&](const std::vector<double>& v) { return v.size() == 1 || v.size() == n; };
        auto bad = [&](const char* which) {
            fail(what, std::string{"'"} + std::string{path} + "': " + which + " must have 1 or " +
                           std::to_string(n) + " values (the marked region's flat size)");
        };
        if (b.lower && !ok(*b.lower)) bad("lower");
        if (b.upper && !ok(*b.upper)) bad("upper");
        if (eq && !ok(*eq)) bad("equals");
        if (!ok(s.ref)) bad("scaling ref");
        if (!ok(s.ref0)) bad("scaling ref0");
    }

    // ---- state ---------------------------------------------------------------------------------
    bool declaring_ = false;

protected:
    // What a flavor's runner needs (ParallelSystem): this System's ctx, one ctx
    // per child (its placed device's substrate), and the children in order.
    ExecCtx                                                            ctx_;
    std::vector<ExecCtx>                                               child_ctx_;  ///< per child: its device's substrate
    std::vector<std::unique_ptr<Subsystem>>                            children_;   ///< declared order = exec order

private:
    std::vector<Subsystem*>                                            child_ptrs_;
    std::unordered_map<std::string, std::size_t>                       child_index_;
    std::vector<std::optional<sycl::device>>                           placed_;     ///< property::device per child
    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>>  aliases_;
    std::vector<detail::IOEntry*>                                      alias_order_;
    sk::queue*                                                         queue_ = nullptr;
    Substrates*                                                        subs_  = nullptr;
};

}  // namespace naga::cospan
