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
struct parallel {};                   ///< flavor (step 3b: throws when used)
struct transient {};                  ///< the subtree's memory is released after each run (§4.4); Systems only
struct fused {                        ///< flavor (step 3b: throws when used)
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
    /// Runs in the System pass of init_tree: children before owners.
    virtual void init(sk::queue& q, const sycl::device& dev) {
        (void)q;
        (void)dev;
    }

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

    /// The HELPER FORM: a plain System configured by properties (device,
    /// transient), populated with the given Components in order. Flavors
    /// (parallel, fused) are step 3b and throw.
    template <class... Children>
        requires(sizeof...(Children) >= 1 &&
                 (detail::is_child_decl_v<std::remove_cvref_t<Children>> && ...))
    System& add_subsystem(std::string name, const property_list& props, Children&&... children) {
        System& sys = add_child<System>(std::move(name), props);
        detail::FlagScope scope{sys.declaring_};
        (sys.add_decl(std::forward<Children>(children)), ...);
        return sys;
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
    /// snapshot of its root's flags; copies the promoted entry's view.
    void promote(std::string_view child_path, std::string promoted_name) {
        require_declaring("promote");
        check_new_name(promoted_name, "alias");
        detail::IOEntry& e = entry_at(child_path);
        detail::IOEntry& r = *e.root();
        auto a = detail::make_alias_entry(e.dtype(), promoted_name, r.kind(), inert_dims(e),
                                          r.can_be_source(), r.can_be_target());
        a->view_     = e.view();
        a->declarer_ = this;
        detail::IOEntry::merge(r, *a);
        register_alias(std::move(a));
    }

    /// A NEW ROOT (target, not source); every member's set merges into it.
    void tie_inputs(std::initializer_list<std::string_view> members, std::string name) {
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
        check_same_dtype(ms, "tie_inputs");
        const detail::IOEntry& shape = shape_source(ms, "tie_inputs");
        auto a = detail::make_alias_entry(shape.dtype(), name, IOKind::input, shape_dims(shape),
                                          /*can_source=*/false, /*can_target=*/true);
        a->declarer_ = this;
        detail::IOEntry& aref = *a;
        for (const auto& [path, m] : ms) {
            (void)path;
            if (m->root() != &aref) detail::IOEntry::merge(aref, *m->root());
        }
        for (const auto& [path, m] : ms) {  // THE EXCEPTION (§4.3): visible at every use site
            (void)path;
            m->can_be_target_ = false;
        }
        register_alias(std::move(a));
    }

    /// A NEW ROOT (source, not target); atomic member dtypes unless allow_plain.
    void tie_outputs(std::initializer_list<std::string_view> members, std::string name,
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic) {
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
        check_same_dtype(ms, "tie_outputs");
        const detail::IOEntry& shape = shape_source(ms, "tie_outputs");
        auto a = detail::make_alias_entry(shape.dtype(), name, IOKind::output, shape_dims(shape),
                                          /*can_source=*/true, /*can_target=*/false);
        a->declarer_ = this;
        detail::IOEntry& aref = *a;
        for (const auto& [path, m] : ms) {
            (void)path;
            if (m->root() != &aref) detail::IOEntry::merge(aref, *m->root());
        }
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

    /// Chains through children by dotted path.
    Subsystem& get_subsystem(std::string_view path) {
        const auto seg = split_path(path);
        System*    s   = this;
        Subsystem* c   = nullptr;
        for (std::size_t i = 0; i < seg.size(); ++i) {
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
    detail::IOEntry& entry_at(std::string_view path) {
        const Located l = locate(path);
        detail::IOEntry* e = l.comp != nullptr ? l.comp->lookup_entry(l.leaf) : l.sys->find_alias(l.leaf);
        if (e == nullptr)
            throw std::out_of_range("naga::cospan: " + l.sys->describe() + " has no " +
                                    (l.comp != nullptr ? "entry '" : "alias '") + std::string{l.leaf} +
                                    "' (resolving '" + std::string{path} + "')");
        return *e;
    }

    // ---- initializers by path (legal inside init() only; else throw) -------------------------
    template <DType DT, int Rank>
    Initializer<DT, Rank> get_input_initializer(std::string_view path) {
        return typed_initializer<DT, Rank>(path, IOKind::input, "get_input_initializer");
    }
    template <DType DT, int Rank>
    Initializer<DT, Rank> get_output_initializer(std::string_view path) {
        return typed_initializer<DT, Rank>(path, IOKind::output, "get_output_initializer");
    }
    template <DType DT, int Rank>
    HostInitializer<DT, Rank> get_input_host_initializer(std::string_view path) {
        require_init("get_input_host_initializer");
        return HostInitializer<DT, Rank>{entry_of_kind(path, IOKind::input), queue_ref()};
    }
    template <DType DT, int Rank>
    HostInitializer<DT, Rank> get_output_host_initializer(std::string_view path) {
        require_init("get_output_host_initializer");
        return HostInitializer<DT, Rank>{entry_of_kind(path, IOKind::output), queue_ref()};
    }
    template <class T>
    void fill_input(std::string_view path, T value) {
        fill_entry(entry_of_kind(path, IOKind::input), value, "fill_input");
    }
    template <class T>
    void fill_output(std::string_view path, T value) {
        fill_entry(entry_of_kind(path, IOKind::output), value, "fill_output");
    }
    void zero_input(std::string_view path) { zero_entry(entry_of_kind(path, IOKind::input), "zero_input"); }
    void zero_output(std::string_view path) { zero_entry(entry_of_kind(path, IOKind::output), "zero_output"); }
    template <class T>
    void set_input_from_host(std::string_view path, std::span<const T> values) {
        require_init("set_input_from_host");
        detail::write_back<T>(entry_of_kind(path, IOKind::input), queue_ref(), values);
    }
    template <class T>
    void set_output_from_host(std::string_view path, std::span<const T> values) {
        require_init("set_output_from_host");
        detail::write_back<T>(entry_of_kind(path, IOKind::output), queue_ref(), values);
    }

    // ---- Subsystem verbs: the DEFAULT orchestration (r01 §7.2, §11.2) ------------------------
    /// setup(); children's run_setup fired inside add_subsystem. The ROOT
    /// (no owner) then settles the transient subtrees: only now is every
    /// peer a later structural call could add known.
    void run_setup() override {
        {
            detail::FlagScope scope{declaring_};
            setup();
        }
        if (owner() == nullptr) mark_transient_roots();
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
        for (detail::IOEntry* a : alias_order_)
            if (a->is_root() && !a->has_storage() && a->flat_size() > 0) a->allocate(queue, substrates);
        for (auto& c : children_) c->prepare_exec(queue, substrates);
        for (detail::IOEntry* a : alias_order_) a->point_wrapper();
    }

    void compile() override {
        for (auto& c : children_) c->compile();
    }

    /// Component pass of init_tree: forwards to the children in order.
    void run_init(const ExecCtx& owner_ctx) override {
        (void)owner_ctx;
        for (std::size_t i = 0; i < children_.size(); ++i) children_[i]->run_init(child_ctx_[i]);
    }

    /// System pass: child Systems first (a Component's is a no-op), then this
    /// System's own init() — the last word over its children's.
    void run_system_init() override {
        for (auto& c : children_) c->run_system_init();
        detail::FlagScope scope{in_init_};
        init(*ctx_.queue, ctx_.device);
    }

    void run_pre_compute(ExecCtx&) override {}   ///< Systems do no work
    void run_post_compute(ExecCtx&) override {}

    /// THE DEFAULT LOOP (§7.3): per child pre_compute -> compute_forward ->
    /// submit (through this System's queue, to the child's device) ->
    /// post_compute; revive() first, release() last. "I submitted": null.
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        revive();
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
        release();
        return nullptr;
    }

    /// Whether this transient System's roots are currently released.
    bool transient() const noexcept { return transient_; }
    bool released() const noexcept { return released_; }

private:
    // ---- transient subtrees (§4.4) -------------------------------------------------------------
    /// The ROOT's job, at the end of its setup: every root declared in a
    /// transient System's subtree with NO peer declared outside that subtree
    /// belongs to the INNERMOST transient System for which that holds. It
    /// takes no arena bytes (arena_eligible reads the flag) and is released
    /// after that System's run.
    void mark_transient_roots() {
        std::vector<detail::IOEntry*> roots;
        collect_roots(roots);
        for (detail::IOEntry* r : roots) {
            for (const Subsystem* s = r->declarer(); s != nullptr; s = s->owner()) {
                // owner() is const; the tree is this root System's own, so it may be mutated here.
                auto* x = const_cast<System*>(dynamic_cast<const System*>(s));
                if (x == nullptr || !x->transient_) continue;
                bool outside = false;
                for (const detail::IOEntry* p : r->peers_)
                    if (!inside(p->declarer(), x)) outside = true;
                if (outside) continue;
                r->transient_    = true;
                r->transient_by_ = x->path();
                x->owned_roots_.push_back(r);
                break;
            }
        }
    }

    static bool inside(const Subsystem* d, const System* x) {
        for (; d != nullptr; d = d->owner())
            if (d == x) return true;
        return false;
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

    /// Re-allocate released roots (pool route, current extents) and re-run the
    /// subtree's init, both passes, when released. A no-op otherwise.
    void revive() {
        if (!transient_ || !released_) return;
        revive_storage();
        for (std::size_t i = 0; i < children_.size(); ++i) children_[i]->run_init(child_ctx_[i]);
        run_system_init();
    }

    /// Drop the storage of every root this System owns. SlabKit defers the
    /// free, so work already submitted keeps its pointers.
    void release() {
        if (!transient_) return;
        for (detail::IOEntry* r : owned_roots_)
            if (r->has_storage()) {
                r->release_storage();
                released_ = true;
            }
    }

    /// This System's roots, and those of every released nested transient
    /// System (their storage is needed by the init this revive re-runs).
    void revive_storage() {
        for (detail::IOEntry* r : owned_roots_) r->revive_storage(*queue_, *subs_);
        released_ = false;
        for (Subsystem* c : child_ptrs_)
            if (auto* s = dynamic_cast<System*>(c)) s->revive_nested();
    }
    void revive_nested() {
        if (transient_ && released_) {
            revive_storage();
            return;
        }
        for (Subsystem* c : child_ptrs_)
            if (auto* s = dynamic_cast<System*>(c)) s->revive_nested();
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

    // ---- initializer helpers (the Component's, by path) ---------------------------------------
    void require_init(const char* what) const {
        if (!in_init_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside init() only (" +
                                   describe() + ")");
    }
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
    template <DType DT, int Rank>
    Initializer<DT, Rank> typed_initializer(std::string_view path, IOKind k, const char* what) {
        require_init(what);
        auto* e = dynamic_cast<detail::TypedEntry<DT, Rank>*>(&entry_of_kind(path, k));
        if (e == nullptr)
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{path} +
                                        "' was asked for with the wrong dtype or rank");
        return e->make_initializer();
    }
    /// op(root, primary device, primary pointer) on the root's primary storage,
    /// then refresh the other placements; a no-op on an unallocated EMPTY root.
    template <class Op>
    void write_root(detail::IOEntry& e, const char* what, Op&& op) {
        require_init(what);
        detail::IOEntry& root = *e.root();
        if (!root.has_storage()) {
            if (root.flat_size() == 0) return;
            throw std::logic_error("naga::cospan: " + describe() + ": '" + std::string{e.name()} +
                                   "' has no storage to initialize");
        }
        const Placement pl = root.storage_placements().front();
        op(root, pl.device, root.storage_on(pl));
        root.sync_other_placements(queue_ref());
    }
    template <class T>
    void fill_entry(detail::IOEntry& e, T value, const char* what) {
        write_root(e, what, [&](detail::IOEntry& root, const sycl::device& d, void* p) {
            detail::dispatch_dtype(root.dtype(), [&](auto tag) {
                using E   = typename decltype(tag)::type;
                const E v = static_cast<E>(value);
                queue_ref().fill_in_order(d, p, v, root.flat_size());
            });
        });
    }
    void zero_entry(detail::IOEntry& e, const char* what) {
        write_root(e, what, [&](detail::IOEntry& root, const sycl::device& d, void* p) {
            queue_ref().memset_in_order(d, p, 0, root.flat_size() * dtype_size(root.dtype()));
        });
    }

    std::vector<detail::IOEntry*> owned_roots_;  ///< transient roots this System releases

private:
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
    Located locate(std::string_view path) {
        const auto seg = split_path(path);
        System*    s   = this;
        for (std::size_t i = 0; i + 1 < seg.size(); ++i) {
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

    std::string describe() const {
        return path().empty() ? std::string{"the root System"} : "System '" + path() + "'";
    }

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
        if (props.has_property<property::parallel>() || props.has_property<property::fused>())
            throw std::logic_error(
                "naga::cospan: property::parallel and property::fused are step 3b (ParallelSystem / "
                "FusedSystem are not implemented yet)");
        auto owned = std::make_unique<T>(std::forward<Args>(ctor_args)...);
        T&   ref   = *owned;
        attach(std::move(owned), std::move(name), props);
        ref.run_setup();  // fully declared before the caller's next structural call
        return ref;
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
        if (tr && dynamic_cast<System*>(c.get()) == nullptr)
            throw std::invalid_argument("naga::cospan: property::transient is for Systems only ('" + name + "')");
        Subsystem& b = *c;
        b.path_      = path().empty() ? name : path() + "." + name;
        b.name_      = name;
        b.owner_     = this;
        if (auto* s = dynamic_cast<System*>(c.get())) s->transient_ = tr;
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

    // ---- alias shape ---------------------------------------------------------------------
    /// A promotion's dims: fixed dims kept, derived dims become inert markers
    /// (System resolves no shapes); the alias exists for naming and marks.
    static std::vector<detail::DimRecord> inert_dims(const detail::IOEntry& e) {
        std::vector<detail::DimRecord> out;
        for (int d = 0; d < e.rank(); ++d) {
            detail::DimRecord r = e.dim(d);
            if (r.how != detail::DimRecord::How::fixed) {
                r.how = detail::DimRecord::How::from_input;
                r.input.clear();
                r.dim         = 0;
                r.value_index = 0;
                r.extent      = e.extents()[static_cast<std::size_t>(d)];
            }
            out.push_back(std::move(r));
        }
        return out;
    }

    static std::vector<detail::DimRecord> shape_dims(const detail::IOEntry& e) {
        std::vector<detail::DimRecord> out;
        for (int d = 0; d < e.rank(); ++d) out.push_back(e.dim(d));
        return out;
    }

    using Members = std::vector<std::pair<std::string_view, detail::IOEntry*>>;

    Members resolve_members(std::initializer_list<std::string_view> members, const char* what) {
        if (members.size() == 0) fail(what, "no members");
        Members out;
        for (const std::string_view m : members) {
            detail::IOEntry* e = &entry_at(m);
            for (const auto& [p, o] : out)
                if (o == e) fail(what, "'" + std::string{m} + "' is listed twice (as '" + std::string{p} + "')");
            out.emplace_back(m, e);
        }
        return out;
    }

    void check_same_dtype(const Members& ms, const char* what) const {
        for (const auto& [path, m] : ms)
            if (!dtype_compatible(m->dtype(), ms.front().second->dtype()))
                fail(what, "'" + std::string{path} + "' has a different value dtype than '" +
                               std::string{ms.front().first} + "'");
    }

    /// A tie alias is a root: it needs a fixed shape from some member (or that
    /// member's root); its dims are that member's.
    const detail::IOEntry& shape_source(const Members& ms, const char* what) const {
        for (const auto& [path, m] : ms) {
            (void)path;
            if (!m->is_dynamic()) return *m;
            if (!m->root()->is_dynamic()) return *m->root();
        }
        fail(what, "no member has a fixed shape; a tie alias is a root and needs one");
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
    bool in_init_   = false;
    bool transient_ = false;  ///< set by the owner from property::transient
    bool released_  = false;

    ExecCtx                                                            ctx_;
    std::vector<ExecCtx>                                               child_ctx_;  ///< per child: its device's substrate
    std::vector<std::unique_ptr<Subsystem>>                            children_;   ///< declared order = exec order
    std::vector<Subsystem*>                                            child_ptrs_;
    std::unordered_map<std::string, std::size_t>                       child_index_;
    std::vector<std::optional<sycl::device>>                           placed_;     ///< property::device per child
    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>>  aliases_;
    std::vector<detail::IOEntry*>                                      alias_order_;
    sk::queue*                                                         queue_ = nullptr;
    Substrates*                                                        subs_  = nullptr;
};

}  // namespace naga::cospan
