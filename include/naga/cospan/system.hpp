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

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/alias_eval.hpp"
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
struct run_n { std::size_t n; };       ///< run policy: the child runs on the first n forwards of its owner, then is skipped (n >= 1)
struct do_once {};                    ///< run policy: run_n{1}
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
    std::is_same_v<std::remove_cvref_t<P>, property::run_n> ||
    std::is_same_v<std::remove_cvref_t<P>, property::do_once> ||
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
        if (has_property<property::run_n>() && get_property<property::run_n>().n == 0)
            throw std::invalid_argument("naga::cospan: run_n needs n >= 1");
        if (has_property<property::run_n>() && has_property<property::do_once>())
            throw std::invalid_argument("naga::cospan: run_n and do_once are two run policies; give one");
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
               std::optional<property::transient>, std::optional<property::fused>,
               std::optional<property::run_n>, std::optional<property::do_once>>
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

namespace detail {
struct SystemAccess;
}

class System : public Subsystem {
public:
    ~System() override = default;

    // ---- HOOKS ---------------------------------------------------------------
    /// Structure and wiring (legal here only).
    virtual void setup() {}
    // There is no init hook: initial values are declared with init_input (§4.4).

    // ---- adding children (legal inside setup() only) ---------------------------
    /// A REALIZED System or Component type. The child is constructed, named,
    /// placed, and its run_setup has run BEFORE this returns.
    template <class T, class... Args>
        requires(std::derived_from<T, Subsystem> && !detail::leads_with_props_v<Args...>)
    void add_subsystem(std::string name, Args&&... ctor_args) {
        add_child<T>(std::move(name), property_list{}, std::forward<Args>(ctor_args)...);
    }
    template <class T, class... Args>
        requires std::derived_from<T, Subsystem>
    void add_subsystem(std::string name, const property_list& props, Args&&... ctor_args) {
        add_child<T>(std::move(name), props, std::forward<Args>(ctor_args)...);
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
    void add_subsystem(std::string name, const property_list& props, Children&&... children) {
        if (props.has_property<property::parallel>() && props.has_property<property::fused>())
            throw std::invalid_argument("naga::cospan: " + describe() + ": add_subsystem('" + name +
                                        "'): parallel and fused are two flavors; give one");
        if (props.has_property<property::parallel>()) {
            populate<typename detail::parallel_of<Children...>::type>(
                std::move(name), props, std::tuple<>{}, std::forward<Children>(children)...);
            return;
        }
        if (props.has_property<property::fused>()) {
            const property::fused& f = props.get_property<property::fused>();
            populate<typename detail::fused_of<Children...>::type>(
                std::move(name), props,
                std::tuple<property::fused::Mode, std::optional<sycl::range<3>>>{f.mode, f.wg},
                std::forward<Children>(children)...);
            return;
        }
        populate<System>(std::move(name), props, std::tuple<>{}, std::forward<Children>(children)...);
    }

    /// An empty System; add children with the templated overload inside its setup().
    void add_subsystem(std::string name, const property_list& props = {}) {
        add_child<System>(std::move(name), props);
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
    /// entry: its shape is read live from it.
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
        record({first_seg(child_path)}, [cp = std::string{child_path}, nm = std::move(promoted_name)](System& s, std::size_t i) {
            s.replay_promote(cp, nm, i);
        });
    }

    /// A NEW ROOT (target, not source); every member's set merges into it. The members are
    /// entries of dtype DT and rank Rank; their unspecified dims come from the tie and their specified dims are constraints on it (§3.6). The alias OWNS the shape: its
    /// `shape` (fixed / dims / from_output_shape / from_output_values / from_input_shape), or, when left unspecified,
    /// the output connected into the alias, else the tie's initializer.
    template <DType DT, int Rank>
    void tie_inputs(std::initializer_list<std::string_view> members, std::string name,
                    AliasShape shape = detail::unspecified_alias_shape()) {
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
        record_tie(members, name, /*is_input=*/true);
    }

    /// A NEW ROOT (source, not target); atomic member dtypes unless allow_plain. The members are
    /// outputs of dtype DT and rank Rank (unspecified dims come from the tie, specified ones are constraints); the alias owns the shape (an output
    /// alias defines the shape of its connections, so it cannot take its shape from one).
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
        record_tie(members, name, /*is_input=*/false);
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

    /// OPACITY: a chunked system's only interface is its own promotions. A path that descends
    /// THROUGH an opaque system into its children is refused, unless the opaque system is the CALLER resolving its own
    /// subtree. Overridden by ChunkedSystem.
    virtual bool is_opaque() const noexcept { return false; }

    /// Chains through children by dotted path. `outside`: the caller is a Model (an OUTSIDE caller of this root, even
    /// when this root is opaque itself).
    /// Null if a slot on the path exists but is empty (a deleted child); throws if a name was never declared.
    Subsystem* get_subsystem(std::string_view path, bool outside = false) {
        const auto seg = split_path(path);
        System*    s   = this;
        Subsystem* c   = nullptr;
        for (std::size_t i = 0; i < seg.size(); ++i) {
            check_opaque_descent(*s, path, outside);
            if (!s->child_index_.contains(std::string{seg[i]}))
                throw std::out_of_range("naga::cospan: " + s->describe() + " has no subsystem '" +
                                        std::string{seg[i]} + "' (resolving '" + std::string{path} + "')");
            c = s->find_child(seg[i]);
            if (c == nullptr) return nullptr;  // declared, but its slot is empty (the child was deleted)
            if (i + 1 < seg.size()) {
                s = dynamic_cast<System*>(c);
                if (s == nullptr)
                    throw std::invalid_argument("naga::cospan: '" + std::string{seg[i]} +
                                                "' is a Component; '" + std::string{path} + "' descends into it");
            }
        }
        return c;
    }

    /// Declaration order = execution order; the Model's walks use this and nothing else.
    std::span<Subsystem* const> children() const noexcept { return child_ptrs_; }

    /// The entry a path names (Component I/O or alias), for the Model's host
    /// access and for tests; never for wiring.
    detail::IOEntry& entry_at(std::string_view path) { return entry_at_impl(path, false); }
    /// The MODEL-facing resolver: the caller is outside this system even when it is itself a chunked system, so a Model
    /// whose root is chunked sees its promotions only. (Everything else is entry_at.)
    detail::IOEntry& entry_at_from_outside(std::string_view path) { return entry_at_impl(path, true); }

    // ---- initial values (legal inside setup() only; r01 §4.4) -------------------------
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
        const detail::InitDecl decl{init, detail::depth_of(*this), path()};
        e.init_decls_.push_back(decl);
        record({first_seg(target)}, [t = std::string{target}, decl](System& s, std::size_t) {
            s.entry_at(t).init_decls_.push_back(decl);
        });
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
            if (a->local_shape()) {
                place_alias(*a, parent_device);
                a->root()->set_primary_placement(a->placement());
                total[parent_device] += a->root()->static_bytes();
            }
        child_ctx_.clear();
        for (std::size_t i = 0; i < children_.size(); ++i) {
            ExecCtx c = ctx_;
            c.device  = placed_[i].value_or(parent_device);
            child_ctx_.push_back(c);
            if (children_[i])
                for (const auto& [d, bytes] : children_[i]->get_exec_info(c.device)) total[d] += bytes;
        }
        for (detail::IOEntry* a : alias_order_)
            if (!a->local_shape()) {
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
        ev_.queue_     = &queue;
        ctx_.queue     = &queue;
        ctx_.group     = nullptr;
        ctx_.substrate = &substrates[ctx_.device];
        for (ExecCtx& c : child_ctx_) {
            c.queue     = &queue;
            c.group     = nullptr;
            c.substrate = &substrates[c.device];
        }
        // No allocation here: a variable is materialized on need (§4.5). A shape-driving read may need one:
        ev_.ensure_ = [this](detail::IOEntry& r) { r.variable().materialize(queue_ref(), nullptr, !chunk_replay_); };
        for (detail::IOEntry* a : alias_order_)
            if (a->local_shape()) a->bind_substrates(substrates);
        for (auto& c : children_)
            if (c) c->prepare_exec(queue, substrates);
        for (detail::IOEntry* a : alias_order_)
            if (a->has_storage()) a->point_wrapper();
    }

    void compile() override {
        for (auto& c : children_)
            if (c) c->compile();
    }

    void run_pre_compute(ExecCtx&) override {}   ///< Systems do no work
    void run_post_compute(ExecCtx&) override {}

    /// THE DEFAULT LOOP (§7.3): per child pre_compute -> compute_forward -> submit (through this System's queue, to
    /// the child's device) -> post_compute. A TRANSIENT child exists for exactly one forward of its parent: every missing
    /// one is built at the START of the loop (before any child's turn, so a transient reader's ports exist while its
    /// transient producer runs) and every one is deleted at the END (after every child's turn). A deleted child (empty
    /// slot) is skipped. After a run_n / do_once child's turn its counter moves and, at 0, the parent deletes it right
    /// after its post_compute. "I submitted": null.
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        if (owner() == nullptr) prune_retired();
        build_missing_transients();
        for (std::size_t i = 0; i < children_.size(); ++i) {
            materialize_slot(i);  // the tie aliases whose first user is child i
            if (!children_[i]) continue;  // deleted: no pre, compute or post
            Subsystem& c  = *children_[i];
            ExecCtx&   cc = child_ctx_[i];
            c.run_pre_compute(cc);
            if (auto d = c.run_compute_forward()) {
                SubmitTarget t = cc.target();
                // The launch's events go on the storage of every port the child touched, before its
                // post_compute pushes to the other placements (r01 §4.4). Inside a ChunkedSystem this
                // loop runs once per chunk, so each chunk's launch is registered.
                c.register_launch_events(d->submit(t));
            }
            c.run_post_compute(cc);
            if (slots_[i].requested == -2) {
                slots_[i].ran = true;  // a transient is deleted at the end of the loop, not after its turn
            } else if (defer_finish_) {
                // Inside a ChunkedSystem's chunk loop the counter moves once per forward, after the last chunk.
                if (std::find(pending_.begin(), pending_.end(), i) == pending_.end()) pending_.push_back(i);
            } else {
                finish_turn(i);
            }
        }
        for (std::size_t i = 0; i < children_.size(); ++i)  // the END of the forward: every transient child goes
            if (slots_[i].requested == -2 && children_[i]) {
                if (!defer_finish_)
                    finish_turn(i);
                else if (std::find(pending_.begin(), pending_.end(), i) == pending_.end())
                    pending_.push_back(i);
            }
        return nullptr;
    }

    /// Build every transient child that is missing (deleted at the end of the previous forward). `deep`: the child
    /// Systems below do the same (a ChunkedSystem does this before it classifies, so nothing is built inside its
    /// chunk loop).
    void build_missing_transients(bool deep = false) {
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].requested == -2 && !children_[i]) rebuild_child(i);
            if (deep)
                if (auto* s = dynamic_cast<System*>(children_[i].get())) s->build_missing_transients(true);
        }
    }

    /// Drop the retired entries whose launches have completed (no waits).
    void prune_retired() {
        std::erase_if(retired_, [](const Subsystem::InflightKeep& k) {
            for (const sycl::event& e : k.first)
                if (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) return false;
            return true;
        });
    }
    /// After a wait: every launch is done.
    void clear_retired() { retired_.clear(); }
    void collect_inflight(std::vector<Subsystem::InflightKeep>& out) const override {
        for (const auto& c : children_)
            if (c) c->collect_inflight(out);
    }

    /// This System reads `v` in a tie alias's shape rule: it is a USER of the variable and holds its bundle, as a port does,
    /// for as long as the System exists (Variable::materialize hands it the bundle).
    void hold_source(detail::Variable& v) {
        auto h = std::make_shared<detail::UserHold>();
        source_holds_.push_back(h);
        v.users.push_back(h);
    }
    std::vector<std::shared_ptr<detail::UserHold>> source_holds_;

    /// A turn of slot i ended: the counter moves (an eternal child never; run_n / do_once down by one; a transient
    /// counts as finished) and the child is deleted at 0 or after a transient's turn.
    void finish_turn(std::size_t i) {
        SlotInfo& si = slots_[i];
        si.ran       = true;
        if (si.requested == -1) return;
        if (si.requested == -2) {
            delete_child(i);
            return;
        }
        if (--si.counter <= 0) {
            si.counter = 0;
            delete_child(i);
        }
    }

    /// A ChunkedSystem, around its chunk loop: counter moves (and so deletions) of this System's children wait.
    void defer_finishes(bool on) {
        defer_finish_ = on;
        if (!on) pending_.clear();
    }
    /// ... and after the last chunk: the turns that ended are applied, once each.
    void flush_finishes() {
        defer_finish_ = false;
        const std::vector<std::size_t> todo = std::move(pending_);
        pending_.clear();
        for (const std::size_t i : todo) finish_turn(i);
    }

    /// REARM a child by name (r01 §7.4): delete it if present, build a fresh one from its factory and run its
    /// setup, replay this System's structural declarations that name it (its ports rejoin the SAME variables and
    /// aliases), run get_exec_info / prepare_exec / compile for it only, and reset its counter to the policy asked
    /// for. Throws for a name never declared; a ParallelSystem / FusedSystem keeps its children.
    void rearm(std::string_view child) {
        const auto it = child_index_.find(std::string{child});
        if (it == child_index_.end())
            throw std::out_of_range("naga::cospan: " + describe() + " has no subsystem '" + std::string{child} + "'");
        if (!slots_deletable())
            throw std::logic_error("naga::cospan: " + describe() + ": this flavor keeps its children; rearm is for plain "
                                   "and chunked Systems");
        const std::size_t i = it->second;
        rebuild_child(i);
        slots_[i].counter = slots_[i].requested;
        slots_[i].ran     = false;
    }
    /// False for the flavors that refuse run policies on their children (ParallelSystem, FusedSystem).
    virtual bool slots_deletable() const noexcept { return true; }

    /// Delete slot i's child (if present), build a fresh one from the factory and run its setup, replay this
    /// System's structural declarations that name it (so its ports join the EXISTING variables and aliases), then
    /// get_exec_info / prepare_exec / compile for that child only. The arena is sized once at setup: a rebuilt
    /// child's allocations take the pool route.
    void rebuild_child(std::size_t i) {
        delete_child(i);
        build_child(i, /*rebuilt=*/true);
        notify_changed();
        for (const Decl& d : decls_)
            if (std::find(d.refs.begin(), d.refs.end(), slots_[i].name) != d.refs.end()) d.replay(*this, i);
        refresh_inits(*children_[i]);
        // setup's end-of-setup pass (the root's resolve) never ran for this subtree: classify its chunked systems and
        // order its tie aliases' materialization now, after the replay has joined its ports to the outside variables
        if (auto* sub = dynamic_cast<System*>(children_[i].get())) {
            sub->classify_chunks();
            sub->build_materialize_slots();
        }
        if (queue_ != nullptr) {  // after setup completed
            children_[i]->get_exec_info(child_ctx_[i].device);
            children_[i]->prepare_exec(*queue_, *subs_);
            children_[i]->compile();
        }
    }

    /// The ROOT System's end-of-setup pass (§4.4, §11.2.1; run_setup runs it when this System has
    /// no owner). Per variable it resolves the winning initial value (smallest declaring depth; two at one depth
    /// throw; an init on a variable with a source is ignored), then checks the ties and orders the tie
    /// aliases' materialization.
    void resolve() {
        classify_chunks();  // ChunkedSystems classify their roots and re-root boundary ones first
        std::vector<detail::IOEntry*> roots;
        collect_roots(roots);
        std::vector<ComponentBase*> comps;
        collect_components(comps);
        for (detail::IOEntry* r : roots) resolve_init(*r);
        check_ties(comps);  // entries with unspecified dims are tied or supplied; alias shape sources are ordered
        build_materialize_slots();
    }

protected:
    /// called on every System of the tree (children first) at the start of resolve(); a
    /// ChunkedSystem classifies its roots here.
    virtual void chunk_classify() {}
    /// A descendant child was deleted or rebuilt (see notify_changed).
    virtual void subtree_changed() {}

    /// Right before child i, every alias in slot i is brought into scope. A flavor's runner calls
    /// this before each child's pre_compute, on the thread that drives the loop.
    void materialize_slot(std::size_t i) {
        if (i >= materialize_before_.size()) return;
        for (detail::IOEntry* a : materialize_before_[i]) materialize(*a);
    }

private:
    friend class ChunkedSystem;
    friend struct detail::SystemAccess;

    std::vector<std::vector<detail::IOEntry*>> materialize_before_;  ///< size children + 1: aliases touched before child i
    detail::AliasEvaluator                     ev_;                  ///< evaluates the shape rules of the tie roots this System owns
    bool                                       chunk_replay_ = false;  ///< set by a replaying ChunkedSystem on itself and its descendant Systems: no shape resolution

    /// A TIE alias's shape is evaluated, and its variable materialized, right before the first child that uses it
    /// (r01 §4.5: a System's tie alias evaluation is one of the call sites of materialize). Not for a tie that is an
    /// OUTSIDE boundary root of a chunked system (that system resolves and allocates it), and not for a
    /// promotion: its variable is materialized by whoever uses it. Not while a ChunkedSystem replays its cached
    /// topology (the shapes are applied from the cache).
    void materialize(detail::IOEntry& a) {
        detail::IOEntry& root = *a.root();
        if (!a.local_shape() || !a.alias_shape_ || a.declarer_ != this || root.chunk_outside_tie_) return;
        if (!chunk_replay_ && !(a.alias_shape_->kind == AliasShape::Kind::fixed && !a.dim0_override_))
            ev_.apply_alias_root(a);
        root.variable().materialize(queue_ref(), nullptr, !chunk_replay_);
        if (!chunk_replay_) root.upload_ragged(queue_ref());
    }

    /// The winning init of every variable a rebuilt subtree's ports joined, resolved again.
    void refresh_inits(Subsystem& c) {
        std::vector<detail::IOEntry*> roots;
        const std::function<void(Subsystem&)> walk = [&](Subsystem& s) {
            if (auto* comp = dynamic_cast<ComponentBase*>(&s)) {
                for (detail::IOEntry* e : comp->order_) roots.push_back(e->root());
            } else if (auto* sys = dynamic_cast<System*>(&s)) {
                for (detail::IOEntry* a : sys->alias_order_) roots.push_back(a->root());
                for (Subsystem* k : sys->child_ptrs_)
                    if (k != nullptr) walk(*k);
            }
        };
        walk(c);
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
        for (detail::IOEntry* r : roots) resolve_init(*r);
    }

    void classify_chunks() {
        for (Subsystem* c : child_ptrs_)
            if (auto* s = dynamic_cast<System*>(c)) s->classify_chunks();
        chunk_classify();
    }

    /// Per System: the aliases touched right before child i (slot i): the first child whose subtree
    /// holds a Component partner of the alias's root; slot 0 when there is none. A re-rooted alias is its
    /// ChunkedSystem's own business (allocate_rerooted).
    void build_materialize_slots() {
        materialize_before_.assign(children_.size() + 1, {});
        for (detail::IOEntry* a : alias_order_) {
            if (!a->local_shape() || !a->alias_shape_ || a->root()->chunk_rerooted_) continue;  // ties only
            std::size_t slot  = 0;
            bool        found = false;
            for (const detail::IOEntry* p : a->root()->peers()) {
                if (dynamic_cast<const ComponentBase*>(p->declarer()) == nullptr) continue;
                const Subsystem* x = p->declarer();
                while (x != nullptr && x->owner() != this) x = x->owner();
                if (x == nullptr) continue;
                const std::size_t i = child_index_.at(std::string{x->name()});
                if (!found || i < slot) {
                    slot  = i;
                    found = true;
                }
            }
            materialize_before_[slot].push_back(a);
        }
        for (Subsystem* c : child_ptrs_)
            if (auto* s = dynamic_cast<System*>(c)) s->build_materialize_slots();
    }

    /// The winning init of a root: the declaration of the smallest depth among its alias set's.
    static void resolve_init(detail::IOEntry& r) {
        if (r.model_init_) return;  // Model::init_input wins and is never overwritten
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
        for (const detail::IOEntry* p : r.peers()) consider(*p);
        if (clash != nullptr)
            throw std::invalid_argument("naga::cospan: two initial values are declared at depth " +
                                        std::to_string(best->depth) + " for '" + r.full_path() + "' (by '" +
                                        best->by + "' and '" + clash->by +
                                        "'); the highest level wins, and one level may declare only one");
        if (best != nullptr) r.init_spec_ = best->spec;
    }

    void collect_roots(std::vector<detail::IOEntry*>& out) {
        for (detail::IOEntry* a : alias_order_)
            if (a->local_shape()) out.push_back(a->root());
        for (Subsystem* c : child_ptrs_) {
            if (auto* s = dynamic_cast<System*>(c)) {
                s->collect_roots(out);
            } else if (auto* comp = dynamic_cast<ComponentBase*>(c)) {
                for (detail::IOEntry* e : comp->order_)
                    if (e->local_shape()) out.push_back(e->root());
            }
        }
    }

    /// Depth-first in registration order: the order that decides a root''s owner (its first toucher).
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
            if (c == nullptr) {
                if (s->child_index_.contains(std::string{seg[i]}))
                    throw std::logic_error("naga::cospan: " + s->describe() + ": subsystem '" + std::string{seg[i]} +
                                           "' is deleted (resolving '" + std::string{path} + "'); rearm it first");
                throw std::out_of_range("naga::cospan: " + s->describe() + " has no subsystem '" +
                                        std::string{seg[i]} + "' (resolving '" + std::string{path} + "')");
            }
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
    void add_child(std::string name, const property_list& props, Args&&... ctor_args) {
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
        // The FACTORY builds a fresh child from DECAYED COPIES of the constructor arguments (as ChildDecl does):
        // the slot uses it for the first build and for every rebuild (a transient child, rearm).
        auto make = [args = std::tuple<std::decay_t<Args>...>{std::forward<Args>(ctor_args)...}]() mutable
            -> std::unique_ptr<Subsystem> {
            return std::apply([](auto&... a) -> std::unique_ptr<Subsystem> { return std::make_unique<T>(a...); }, args);
        };
        declare_child(std::move(name), props, std::move(make), nullptr);
    }

    /// The helper form's body: a slot whose factory builds a System of type S (System, or ParallelSystem /
    /// FusedSystem for the flavor properties), and whose after-setup step adds the given Components to it in
    /// order (so a rebuild repeats the population). S::setup() has run before it, which left the child's declaring
    /// flag off, so it is raised here for the adds.
    template <class S, class Ctor, class... Children>
    void populate(std::string name, const property_list& props, Ctor ctor, Children&&... children) {
        // `ctor` is a tuple of S's constructor arguments (empty for System and ParallelSystem).
        auto make = [ctor = std::move(ctor)]() mutable -> std::unique_ptr<Subsystem> {
            return std::apply([](auto&... a) -> std::unique_ptr<Subsystem> { return std::make_unique<S>(a...); }, ctor);
        };
        auto after = [decls = std::tuple<std::decay_t<Children>...>{std::forward<Children>(children)...}](System& sys) mutable {
            detail::FlagScope scope{sys.declaring_};
            std::apply([&](auto&... d) { (sys.add_decl(d), ...); }, decls);
        };
        check_new_name(name, "subsystem");
        require_declaring("add_subsystem");
        declare_child(std::move(name), props, std::move(make), std::move(after));
    }

    template <class T, class... A>
    void add_decl(detail::ChildDecl<T, A...> d) {
        std::apply([&](auto&&... a) { this->add_child<T>(std::move(d.name), property_list{}, std::move(a)...); },
                   std::move(d.args));
    }

    /// One SLOT per child (r01 §7.2): the child itself (null once deleted), a factory for a fresh one, the run
    /// policy asked for, the live counter and whether it has taken a turn. The parent owns the lifecycle.
    /// Every child on the path (all segments but the last) is present, i.e. the path can be resolved right now.
    bool path_alive(std::string_view p) const {
        const System* s = this;
        std::size_t   i = 0;
        while (true) {
            const std::size_t j = p.find('.', i);
            if (j == std::string_view::npos) return true;
            const Subsystem* c = s->find_child(p.substr(i, j - i));
            if (c == nullptr) return false;
            s = dynamic_cast<const System*>(c);
            if (s == nullptr) return true;  // a Component: the rest names its port
            i = j + 1;
        }
    }
    static std::string first_seg(std::string_view p) { return std::string{p.substr(0, p.find('.'))}; }
    void record(std::vector<std::string> refs, std::function<void(System&, std::size_t)> replay) {
        decls_.push_back(Decl{std::move(refs), std::move(replay)});
    }
    void record_tie(std::initializer_list<std::string_view> members, const std::string& name, bool is_input) {
        std::vector<std::string> ms, refs;
        for (const std::string_view m : members) {
            ms.emplace_back(m);
            refs.push_back(first_seg(m));
        }
        record(std::move(refs), [ms = std::move(ms), name, is_input](System& s, std::size_t i) { s.replay_tie(ms, name, is_input, i); });
    }
    static bool under(const detail::IOEntry& e, const Subsystem* c) {
        for (const Subsystem* s = e.declarer(); s != nullptr; s = s->owner())
            if (s == c) return true;
        return false;
    }
    /// REPLAY (rearm): the declaration's new-child side joins the EXISTING variable (no validation: it passed when
    /// first made). The surviving variable is the one the other ports already share.
    void replay_connect(const std::string& src, const std::string& dst, const std::optional<SubarrayRef>& slice,
                        std::size_t i) {
        // The other side may be a deleted child (a do_once producer, a transient reader): there is nothing to join.
        // A live consumer whose producer is gone keeps the producer's NAME on its variable, so materializing it before
        // the producer exists is the reactive error, naming the missing producer.
        const bool src_alive = path_alive(src), dst_alive = path_alive(dst);
        if (!src_alive || !dst_alive) {
            if (!src_alive && dst_alive) {
                detail::IOEntry&  d  = entry_at(dst);
                detail::Variable& vd = d.variable();
                // Another consumer of the same producer still holds the variable the producer fed: the rebuilt port
                // rejoins THAT one (it is the same variable), and it keeps the producer's name.
                std::vector<ComponentBase*> comps;
                collect_components(comps);
                detail::Variable* keep = nullptr;
                for (ComponentBase* comp : comps) {
                    for (detail::IOEntry* e : comp->order_) {
                        detail::Variable& v = e->variable();
                        if (&v != &vd && v.source_path == src) {
                            keep = &v;
                            break;
                        }
                    }
                    if (keep != nullptr) break;
                }
                if (keep != nullptr) {
                    if (slice)
                        for (detail::IOEntry* p : vd.ports) p->view_ = slice;
                    keep->absorb(vd);
                } else {
                    vd.source_path = std::string{src};
                }
            }
            return;
        }
        detail::IOEntry& s = entry_at(src);
        detail::IOEntry& d = entry_at(dst);
        detail::Variable& vs = s.variable();
        detail::Variable& vd = d.variable();
        if (&vs == &vd) return;
        std::optional<SubarrayRef> view = s.view();
        if (slice) {
            if (view) {
                for (std::size_t k = 0; k < 3; ++k) view->offset[k] += slice->offset[k];
                view->extents = slice->extents;
            } else {
                view = slice;
            }
        }
        const Subsystem* c     = children_[i].get();
        const bool       s_new = under(s, c);
        const bool       d_new = under(d, c);
        detail::Variable& keep  = (s_new && !d_new) ? vd : vs;
        detail::Variable& other = (&keep == &vs) ? vd : vs;
        if (view && &other == &vd)
            for (detail::IOEntry* p : other.ports) p->view_ = view;
        keep.absorb(other);
        if (s.kind() == IOKind::output) {
            keep.source_port = &s;
            keep.source_path = s.full_path();
        }
    }
    void replay_promote(const std::string& child_path, const std::string& name, std::size_t) {
        detail::IOEntry* a = find_alias(name);
        detail::IOEntry& e = entry_at(child_path);
        a->mirror_of_ = &e;
        a->view_      = e.view();
        a->variable().absorb(e.variable());
    }
    void replay_tie(const std::vector<std::string>& members, const std::string& name, bool is_input, std::size_t i) {
        detail::IOEntry* a = find_alias(name);
        std::erase_if(a->tie_members_, [](const AliasSrc& ms) { return ms.live_port() != ms.port; });
        for (const std::string& m : members) {
            if (first_seg(m) != slots_[i].name) continue;
            detail::IOEntry& e = entry_at(m);
            a->variable().absorb(e.variable());
            (is_input ? e.can_be_target_ : e.can_be_source_) = false;
            a->tie_members_.push_back(AliasSrc{e.var_, &e, std::nullopt});
        }
        for (detail::IOEntry* p : a->variable().ports)
            if (p != a) p->tie_alias_ = a;
    }

    struct SlotInfo {
        std::string                                 name;
        std::function<std::unique_ptr<Subsystem>()> make;         ///< a fresh child
        std::function<void(System&)>                after_setup;  ///< the helper form's population (else empty)
        long                                        requested = -1;  ///< -1 eternal, -2 transient, n > 0 turns
        long                                        counter   = -1;  ///< live: -1 eternal / transient, n > 0 turns left
        bool                                        ran       = false;  ///< took a turn in an earlier forward
    };

    void declare_child(std::string name, const property_list& props, std::function<std::unique_ptr<Subsystem>()> make,
                       std::function<void(System&)> after_setup) {
        const bool tr = props.has_property<property::transient>();
        if (tr && (props.has_property<property::run_n>() || props.has_property<property::do_once>()))
            throw std::invalid_argument("naga::cospan: " + describe() + ": add_subsystem('" + name +
                                        "'): transient cannot be combined with run_n or do_once; a transient subsystem "
                                        "runs every forward");
        const bool has_n = props.has_property<property::run_n>();
        const bool once  = props.has_property<property::do_once>();
        SlotInfo   si;
        si.name        = name;
        si.make        = std::move(make);
        si.after_setup = std::move(after_setup);
        si.requested   = has_n ? static_cast<long>(props.get_property<property::run_n>().n) : once ? 1 : tr ? -2 : -1;
        si.counter     = si.requested;
        placed_.push_back(props.has_property<property::device>()
                              ? std::optional<sycl::device>{props.get_property<property::device>().dev}
                              : std::nullopt);
        const std::size_t i = slots_.size();
        child_index_.emplace(std::move(name), i);
        slots_.push_back(std::move(si));
        children_.emplace_back();
        child_ptrs_.push_back(nullptr);
        build_child(i, /*rebuilt=*/false);
    }

    /// Build slot i's child from its factory: named, placed in this namespace, its run_setup run BEFORE this returns.
    void build_child(std::size_t i, bool rebuilt) {
        SlotInfo& si = slots_[i];
        auto      c  = si.make();
        Subsystem& b = *c;
        b.path_          = path().empty() ? si.name : path() + "." + si.name;
        b.name_          = si.name;
        b.owner_         = this;
        b.requested_run_ = si.requested;
        b.rebuilt_       = rebuilt;
        children_[i]     = std::move(c);
        child_ptrs_[i]   = &b;
        b.run_setup();  // fully declared before the caller's next structural call
        if (si.after_setup) si.after_setup(static_cast<System&>(b));
    }

    /// Delete slot i's child: its ports leave their variables; a variable lives on while another port holds a bundle.
    void delete_child(std::size_t i) {
        if (children_[i]) {
            // launches of the child still in flight keep what they read alive on the root's retire list
            std::vector<Subsystem::InflightKeep> keep;
            children_[i]->collect_inflight(keep);
            System& r = root_system();
            for (auto& k : keep) r.retired_.push_back(std::move(k));
            children_[i].reset();
            r.prune_retired();
            notify_changed();
        }
        child_ptrs_[i] = nullptr;
    }
    /// A child under this System (or under one of its owners) was deleted or rebuilt: a ChunkedSystem above must
    /// classify again (it keeps raw pointers to the children's ports).
    void notify_changed() {
        for (System* s = this; s != nullptr; s = const_cast<System*>(s->owner())) s->subtree_changed();
    }
    System& root_system() {
        System* s = this;
        while (s->owner() != nullptr) s = const_cast<System*>(s->owner());
        return *s;
    }
    std::vector<Subsystem::InflightKeep> retired_;  ///< (events, object): held until the events complete (root only)

    void register_alias(std::unique_ptr<detail::IOEntry> a) {
        detail::IOEntry* p = a.get();
        alias_order_.push_back(p);
        aliases_.emplace(std::string{p->name()}, std::move(a));
    }

    // ---- alias shape ---------------------------------------------------------
    using Members = std::vector<std::pair<std::string_view, detail::IOEntry*>>;

    /// A promotion's dims. Its shape is read live from the entry (it mirrors it), so the dims are
    /// only classification: fixed and from_root kept (unspecified stays unspecified), derived ones
    /// become inert markers.
    static std::vector<detail::DimRecord> promoted_dims(const detail::IOEntry& e) {
        std::vector<detail::DimRecord> out;
        for (int d = 0; d < e.rank(); ++d) {
            detail::DimRecord r = e.dim(d);
            if (r.how != detail::DimRecord::How::fixed && r.how != detail::DimRecord::How::unspecified) {
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
                if (s.kind == AliasShape::Kind::dims) {  // statically per-index dims (resolved by the owner)
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

    /// Every member is of dtype DT and rank Rank. A member may declare any dims: the unspecified ones come from the
    /// tie, the specified ones are constraints on it.
    template <DType DT, int Rank>
    void check_tie_members(const Members& ms, const char* what) const {
        for (const auto& [path, m] : ms) check_typed<DT, Rank>(*m, what, path);
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
                    hold_source(*up.var_);
                    sp.src = AliasSrc{up.var_, &up,
                                              (sp.input && up.view()) ? std::optional<std::array<std::size_t, 3>>{up.view()->extents}
                                                                      : std::nullopt};
                }
                return nullptr;
            }
            case AliasShape::Kind::fixed:
                if (s.rank != Rank)
                    fail(what, "alias::fixed gives " + std::to_string(s.rank) + " extents but the alias is rank " +
                                   std::to_string(Rank));
                return nullptr;
            case AliasShape::Kind::unspecified:
                if (!is_input)
                    fail(what, "an output alias needs a shape (alias::fixed, dims, from_output_shape, from_output_values or from_input_shape): "
                               "an output with unspecified dims is an error");
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
    /// the merged set as tied (an entry whose dims the tie supplies must be in a tie's set).
    void finish_tie(detail::IOEntry& a, AliasShape shape, detail::IOEntry* upstream, const Members& ms) {
        a.declarer_    = this;
        a.alias_shape_ = std::move(shape);
        if (upstream != nullptr) hold_source(*upstream->var_);
        if (upstream != nullptr)
            a.alias_src_ = AliasSrc{upstream->var_, upstream,
                                            (a.alias_shape_->kind == AliasShape::Kind::from_input_shape && upstream->view())
                                                ? std::optional<std::array<std::size_t, 3>>{upstream->view()->extents}
                                                : std::nullopt};
        for (const auto& [path, m] : ms) {
            (void)path;
            a.tie_members_.push_back(AliasSrc{m->var_, m, std::nullopt});
            detail::IOEntry::merge(a, *m);  // same Variable: no-op
        }
        for (detail::IOEntry* p : a.variable().ports)
            if (p != &a) p->tie_alias_ = &a;
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

    // ---- ties, checked once at the end of setup ------------------------------
    void collect_aliases(std::vector<detail::IOEntry*>& out) {
        for (detail::IOEntry* a : alias_order_) out.push_back(a);
        for (Subsystem* c : child_ptrs_)
            if (auto* s = dynamic_cast<System*>(c)) s->collect_aliases(out);
    }

    /// An untied output with an unspecified dim, a free input with one and no initializer shape, and an
    /// unspecified tie_inputs root nothing supplies are setup errors; a
    /// from_output_* alias reads an output some component produces (when it runs is reactive, not checked here).
    void check_ties(const std::vector<ComponentBase*>& comps) {
        for (ComponentBase* c : comps)
            for (detail::IOEntry* e : c->order_) {
                if (!e->local_shape() || !e->has_unspecified()) continue;
                if (e->root()->chunk_rerooted_) continue;  // a chunked boundary: its chunked system supplies the shape
                if (e->kind() == IOKind::output)
                    throw std::invalid_argument("naga::cospan: output '" + e->full_path() +
                                                "' has an unspecified dimension but is not a member of a tie_outputs: nothing "
                                                "supplies its shape (declare the dimension, or tie it)");
                const auto& is = e->init_spec();
                if (!is || !is->shape())
                    throw std::invalid_argument("naga::cospan: free input '" + e->full_path() +
                                                "' has an unspecified dimension and no initializer that gives it a shape "
                                                "(connect it, tie it, or declare init_input with a shape: fill(v, shape), "
                                                "zero(shape), from_host(data) for rank 1 or from_host(data, shape), custom(fn, shape))");
                if (is->shape()->rank != e->rank())
                    throw std::invalid_argument("naga::cospan: shape mismatch: free input '" + e->full_path() + "' has rank " +
                                                std::to_string(e->rank()) + " but its initializer gives a rank-" +
                                                std::to_string(is->shape()->rank) + " shape");
                for (int d = 0; d < e->rank(); ++d) {
                    const detail::DimRecord& r = e->dim(d);
                    const bool init_per = d >= 1 && is->shape()->per[static_cast<std::size_t>(d)];
                    if (r.how == detail::DimRecord::How::fixed && (init_per || r.extent != is->shape()->extents[static_cast<std::size_t>(d)]))
                        throw std::invalid_argument("naga::cospan: shape mismatch: dimension " + std::to_string(d) + " of '" +
                                                    e->full_path() + "' is fixed = " + std::to_string(r.extent) +
                                                    " but its initializer gives " +
                                                    (init_per ? std::string{"a per-index dimension"}
                                                              : std::to_string(is->shape()->extents[static_cast<std::size_t>(d)])));
                }
            }

        std::unordered_map<const Subsystem*, std::size_t> index;
        for (std::size_t i = 0; i < comps.size(); ++i) index.emplace(comps[i], i);
        std::vector<detail::IOEntry*> aliases;
        collect_aliases(aliases);
        for (detail::IOEntry* a : aliases) {
            if (!a->alias_shape_) continue;
            const AliasShape& s = *a->alias_shape_;
            if (s.kind == AliasShape::Kind::unspecified && a->local_shape() && !(a->init_spec() && a->init_spec()->shape()))
                throw std::invalid_argument("naga::cospan: tie alias '" + a->full_path() +
                                            "' has an unspecified shape and nothing supplies it: connect an output to it, or "
                                            "declare init_input on it with a shape");
            if (s.kind != AliasShape::Kind::from_output_shape && s.kind != AliasShape::Kind::from_output_values &&
                s.kind != AliasShape::Kind::dims && s.kind != AliasShape::Kind::from_input_shape)
                continue;

            // Every upstream output the shape reads must have a component that produces it. WHEN that producer runs is
            // not checked here: shapes are reactive (r01 4.5), and a tie that materializes before its producer has run
            // fails at the forward with the shape-not-known error (reactive_shape.hpp).
            for (const auto& [src_path, up] : a->alias_sources()) {
                const detail::IOEntry* root  = up->root();
                bool                   found = false;
                for (const detail::IOEntry* p : root->peers())
                    if (p->kind() == IOKind::output && index.contains(p->declarer())) found = true;
                if (!found && up->kind() == IOKind::input) continue;  // a free input: materialized on need, whoever touches it first
                if (!found)
                    throw std::invalid_argument("naga::cospan: tie alias '" + a->full_path() + "' takes its shape from '" +
                                                std::string{src_path} + "', which no component produces");
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

        // a Slice cannot select a region of a ragged root (no rectangular geometry).
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
        // (the shadow is no participant: only the ports carry a view)
        const std::vector<detail::IOEntry*> dset(dr.peers().begin(), dr.peers().end());
        if (view)
            for (const detail::IOEntry* e : dset)
                if (e->rank() != sr.rank())
                    fail(what.c_str(), "'" + std::string{e->name()} + "' (rank " + std::to_string(e->rank()) +
                                           ") would see a subarray of '" + std::string{sr.name()} +
                                           "' and must have the root's rank (" + std::to_string(sr.rank()) + ")");
        if (view)
            for (detail::IOEntry* e : dset) e->view_ = view;
        const bool producer_is_src = s.kind() == IOKind::output;
        detail::IOEntry::merge(sr, dr);
        record({first_seg(src), first_seg(dst)}, [a = std::string{src}, b = std::string{dst}, slice](System& sys, std::size_t i) {
            sys.replay_connect(a, b, slice, i);
        });
        (void)producer_is_src;
        if (s.kind() == IOKind::output) {  // the producer, structurally: it stays after the producer's child is deleted
            detail::Variable& v = sr.variable();
            if (v.source_port == nullptr) {
                v.source_port = &s;
                v.source_path = s.full_path();
            }
        }
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
    std::vector<Subsystem*>                                            child_ptrs_;  ///< null where the slot's child is deleted
    std::vector<SlotInfo>                                              slots_;
    /// This System's own structural declarations that name a child, recorded as data when first made: a rebuilt
    /// child's ports are attached to the EXISTING variables and aliases by replaying them (connect, promote, tie_*,
    /// init_input; marks stay on the variables and are not replayed).
    struct Decl {
        std::vector<std::string>                     refs;    ///< the first path segment of every entry it names
        std::function<void(System&, std::size_t)>    replay;  ///< (this System, the rebuilt slot)
    };
    std::vector<Decl>                                                  decls_;
    bool                                                               defer_finish_ = false;  ///< inside a chunk loop: counter moves wait
    std::vector<std::size_t>                                           pending_;               ///< slots whose turn ended while deferred
    std::unordered_map<std::string, std::size_t>                       child_index_;
    std::vector<std::optional<sycl::device>>                           placed_;     ///< property::device per child
    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>>  aliases_;
    std::vector<detail::IOEntry*>                                      alias_order_;
    sk::queue*                                                         queue_ = nullptr;
    Substrates*                                                        subs_  = nullptr;
};

inline bool Subsystem::runs_eternally() const noexcept {
    for (const Subsystem* s = this; s != nullptr; s = s->owner_)
        if (s->requested_run_ != -1 || s->rebuilt_) return false;
    return true;
}

namespace detail {
/// INTERNAL: the root System's own ExecCtx (its device, the Model's queue, that device's
/// substrate), which Model::forward passes to the root's pre/post hooks.
struct SystemAccess {
    static ExecCtx& root_ctx(System& s) { return s.ctx_; }
};
}  // namespace detail

}  // namespace naga::cospan