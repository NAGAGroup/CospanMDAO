#pragma once

// =============================================================================
// naga/cospan/system.hpp — System: structure and wiring
// =============================================================================
//
// A System never does work: it orders the work its children do. It has no I/O of its own beyond ALIAS instances:
// promotions and tie aliases, which it owns by shared_ptr (docs/design-review-2026-10-03.md, section 1).
//
// THE FORWARD (docs/design-review-2026-10-03.md, section 3). A System's forward is three framework-owned steps:
//   precompute   rebuilds transient children (this allocates nothing), then the user's pre_forward(), where rearm()
//                may re-run children by name;
//   compute      calls each live child's forward in declaration order, and nothing else: no materializing and no
//                submitting. A child whose run policy has ended is deleted right away, waiting on THAT child's events
//                only, so its memory is free before later children allocate while independent work keeps running.
//                (Users: add subsystems that can go out of scope as late as possible in setup(), since deleting one
//                makes its parent wait on that subsystem's events before continuing.) depends_on hands one child's
//                events to another for one forward;
//   postcompute  the pushes of the tie_outputs aliases this System owns to the variable's other placements, then the
//                user's post_forward().
//
// Everything structural (add_subsystem, connect, promote, tie_*, init_input, marks) is legal inside setup() only,
// and every rule is checked in its own call (section 2): the wiring rules live in wiring.hpp, which these verbs
// call after resolving paths to instances. Setup allocates nothing, submits nothing and compiles nothing.
//
// ADDING A CHILD. add_subsystem is the factory (sections 2 and 4): it builds the child from a descriptor, runs the
// child's setup, then calls set_device on it, first with the device it was given and then, if this System itself has
// been placed, with that one (the outer call wins). The descriptor carries the name and the constructor
// arguments only: constructors never take a device.
//
// REBUILDING. Each child lives in a SLOT that keeps its factory, its device and its run policy. Rebuilding a child
// is the same build (factory plus setup) followed by REPLAYING the wiring this System declared that names it: the
// rebuilt side rejoins the variables that survive, and nothing unchanged is validated again (section 2).

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <set>
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

#include "naga/cospan/init.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/shape_hook.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/wiring.hpp"

namespace naga::cospan {

// -----------------------------------------------------------------------------
// Properties (SYCL idiom): unordered, self-naming, type-checked
// -----------------------------------------------------------------------------

namespace property {
struct run_n { std::size_t n; };  ///< run policy: the child runs on the first n forwards of its owner, then is skipped (n >= 1)
struct do_once {};                ///< run policy: run_n{1}
struct transient {};              ///< given by an OWNER at registration: a transient registration
}  // namespace property

namespace detail {
template <class P>
inline constexpr bool is_property_v =
    std::is_same_v<std::remove_cvref_t<P>, property::transient> ||
    std::is_same_v<std::remove_cvref_t<P>, property::run_n> ||
    std::is_same_v<std::remove_cvref_t<P>, property::do_once>;
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

    std::tuple<std::optional<property::transient>, std::optional<property::run_n>, std::optional<property::do_once>> props_;
};

// -----------------------------------------------------------------------------
// Descriptors
// -----------------------------------------------------------------------------

/// What add_subsystem builds a child from: the NAME and the CONSTRUCTOR ARGUMENTS of a subsystem of type T, and
/// nothing else. It carries no device: the device is an add_subsystem argument, and constructors never take one
/// (docs/design-review-2026-10-03.md, section 2, and section 4, "Descriptors"). The arguments are decayed copies, so
/// the slot can build the child again.
template <class T, class... Args>
struct SubsystemDesc {
    std::string                       name;
    std::tuple<std::decay_t<Args>...> args;
};

/// subsystem_desc<T>("name", ctor args...): a descriptor of a Component or a user-defined System.
template <class T, class... Args>
    requires std::derived_from<T, Subsystem>
SubsystemDesc<T, Args...> subsystem_desc(std::string name, Args&&... args) {
    return SubsystemDesc<T, Args...>{std::move(name), std::tuple<std::decay_t<Args>...>{std::forward<Args>(args)...}};
}

namespace detail {
/// A factory building a fresh T from copies of the descriptor's arguments, for the first build and every rebuild.
template <class T, class Tuple>
std::function<std::unique_ptr<Subsystem>()> make_factory(Tuple args) {
    return [args = std::move(args)]() mutable -> std::unique_ptr<Subsystem> {
        return std::apply([](auto&... a) -> std::unique_ptr<Subsystem> { return std::make_unique<T>(a...); }, args);
    };
}
}  // namespace detail

// -----------------------------------------------------------------------------
// System
// -----------------------------------------------------------------------------

class System : public Subsystem {
public:
    ~System() override = default;

    // ---- HOOK ------------------------------------------------------------------
    /// Structure and wiring (legal here only). There is no init hook: initial values are declared with init_input.
    virtual void setup() {}
    /// Runs at the end of precompute: where rearm() may re-run children on a trigger only the author knows.
    virtual void pre_forward() {}
    /// Runs at the end of postcompute.
    virtual void post_forward() {}

    // ---- adding children (legal inside setup() only) ---------------------------
    /// The child is built, named, placed, and its run_setup has run BEFORE this returns; then the device (if any)
    /// is applied with set_device. Properties: the run policy, for the whole subsystem.
    template <class T, class... A>
    void add_subsystem(SubsystemDesc<T, A...> desc) {
        declare_child(std::move(desc.name), property_list{}, std::nullopt, detail::make_factory<T>(std::move(desc.args)));
    }
    template <class T, class... A>
    void add_subsystem(const property_list& props, SubsystemDesc<T, A...> desc) {
        declare_child(std::move(desc.name), props, std::nullopt, detail::make_factory<T>(std::move(desc.args)));
    }
    template <class T, class... A>
    void add_subsystem(const sycl::device& dev, SubsystemDesc<T, A...> desc) {
        declare_child(std::move(desc.name), property_list{}, dev, detail::make_factory<T>(std::move(desc.args)));
    }
    template <class T, class... A>
    void add_subsystem(const property_list& props, const sycl::device& dev, SubsystemDesc<T, A...> desc) {
        declare_child(std::move(desc.name), props, dev, detail::make_factory<T>(std::move(desc.args)));
    }

    // ---- structure: connect / promote / tie (legal inside setup() only) --------
    /// Feeds `dst` from `src`; the source's variable wins (wiring.hpp, connect).
    void connect(std::string_view src, std::string_view dst) {
        require_declaring("connect");
        const auto s = instance_at(src);
        const auto d = instance_at(dst);
        detail::connect(site(order_of(src), order_of(dst)), s, d);
        const std::size_t k = decls_.size();
        record({first_seg(src), first_seg(dst)},
               [k, a = std::string{src}, b = std::string{dst}](System& sys, std::size_t) { sys.replay_connect(k, a, b); },
               d->meta);
    }

    /// Kind-agnostic and always allowed: a PEER alias in this namespace joining the member's variable (section 1).
    /// DT must match the member's value dtype and Rank its rank. The alias takes the member's device now (an outer
    /// set_device can override it later).
    template <DType DT, int Rank>
    void promote(std::string_view child_path, std::string promoted_name) {
        static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
        require_declaring("promote");
        check_new_name(promoted_name, "alias");
        const auto member = instance_at(child_path);
        auto       a      = detail::promote<DT, Rank>(site(), member, promoted_name, this);
        store_alias(a);
        const std::size_t k = decls_.size();
        record({first_seg(child_path)},
               [k, cp = std::string{child_path}, nm = std::move(promoted_name)](System& sys, std::size_t) {
                   sys.replay_promote(k, cp, nm);
               },
               a->meta);
    }

    /// A NEW variable (target, not source); every member's variable merges into it and the alias wins. The members
    /// are inputs of dtype DT and rank Rank. The tie's shape is a DEFINITION like a port's (cs::shape: per dimension
    /// specified or unspecified; a ragged tie: cs::shape(), (n), (cs::counts(v)) or (cs::offsets(v))): the definitions of
    /// the members and of everything connected to it merge (unspecified absorbs specified, two specified values must
    /// agree), and a dimension still unspecified comes from the connected output, the initializer, or a shape_hook on the
    /// alias (design review, section 10). `device`: where the tie's initialization runs and its data stays (section 2).
    template <DType DT, int Rank>
    void tie_inputs(std::initializer_list<std::string_view> members, std::string name, ShapeSpec shape = {},
                    std::optional<sycl::device> device = {}) {
        static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
        require_declaring("tie_inputs");
        check_new_name(name, "alias");
        const detail::TieMembers ms = resolve_members(members);
        auto a = detail::tie_inputs<DT, Rank>(site(), ms, name, this, shape, device);
        store_alias(a);
        record_tie(members, name, /*is_input=*/true, a);
    }

    /// A NEW variable (source, not target); atomic member dtypes unless allow_plain. The members are outputs of dtype
    /// DT and rank Rank, all on one device; the alias owns the shape (an output alias defines the shape of its
    /// connections). Its definition is the members' declared dimensions merged with `shape` (cs::shape); any dimension
    /// still unspecified must be set by a shape_hook on the alias, and a member cannot reshape its output (section 10).
    template <DType DT, int Rank>
    void tie_outputs(std::initializer_list<std::string_view> members, std::string name, ShapeSpec shape = {},
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic) {
        static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
        require_declaring("tie_outputs");
        check_new_name(name, "alias");
        const detail::TieMembers ms = resolve_members(members);
        auto a = detail::tie_outputs<DT, Rank>(site(), ms, name, this, shape, policy);
        store_alias(a);
        record_tie(members, name, /*is_input=*/false, a);
    }

    // ---- shape hooks (legal inside setup() only) -------------------------------------
    /// A shape hook is this System's contract for the shape of an input, an input tie alias or an output tie alias
    /// (docs/design-review-2026-10-03.md, section 10; shape_hook.hpp). One descriptor per dimension (cs::dim::as_is,
    /// from_shape, from_value, custom, fixed_at_run), or for a ragged target ONE of cs::ragged::counts_from /
    /// offsets_from. Each hooked dimension must be unspecified in the definition and is owned by this hook only. A hook
    /// and a connection exclude each other: the target can no longer be connected or tied (and a connected or tied input
    /// takes no hook), but its tie alias can take one. The hook runs on the main thread immediately before the first
    /// child that has an instance of the target; every source must be produced by an EARLIER child (checked when this
    /// System's setup ends).
    template <HookDescriptor... D>
    void shape_hook(std::string_view target, D... descriptors) {
        require_declaring("shape_hook");
        std::vector<detail::HookDim> dims;
        (dims.push_back(detail::make_hook_dim(descriptors)), ...);
        const std::size_t h = install_hook(std::string{target}, std::move(dims));
        const std::size_t k = decls_.size();
        std::vector<std::string> refs{first_seg(target)};
        for (const detail::HookDim& d : hooks_[h].dims)
            if (d.names_source()) refs.push_back(first_seg(d.path));
        record(std::move(refs), [k, h](System& sys, std::size_t) { sys.replay_hook(k, h); }, hooks_[h].target.lock()->meta);
    }

    // ---- initial values (legal inside setup() only) -----------------------------
    /// Declares the initial value of a free input's variable: `path` is a child input path or one of this System's
    /// promoted or tied input aliases. The last declaration wins: this System's setup runs after its children's, so it
    /// replaces theirs (and a tie's or promotion's own). It is ignored when the input ends up with a source, and
    /// naming an output throws.
    void init_input(std::string_view target, InitSpec init) {
        require_declaring("init_input");
        const auto        inst = instance_at(target);
        detail::declare_init(site(), inst, init);
        const std::size_t k = decls_.size();
        record({first_seg(target)},
               [k, t = std::string{target}, init](System& sys, std::size_t) { sys.replay_init(k, t, init); },
               inst->meta);
    }

    // ---- ordering (legal inside setup() only) -------------------------------------
    /// Child `b` depends on child `a`: for one forward at a time, B's submissions wait on A's events (A's forward runs
    /// first; its events are collected and handed to B). If B is a System, every component inside it adds them to its
    /// submissions. `a` must have been added before `b`. A duplicate edge is ignored.
    void depends_on(std::string_view b, std::string_view a) {
        require_declaring("depends_on");
        const auto ib = child_index_.find(std::string{b});
        const auto ia = child_index_.find(std::string{a});
        if (ib == child_index_.end()) fail("depends_on", "'" + std::string{b} + "' is not a subsystem of " + describe());
        if (ia == child_index_.end()) fail("depends_on", "'" + std::string{a} + "' is not a subsystem of " + describe());
        if (ia->second >= ib->second)
            fail("depends_on", "'" + std::string{b} + "' can depend only on a subsystem added before it ('" + std::string{a} + "' was not)");
        for (const DepEdge& e : dep_edges_)
            if (e.from == ia->second && e.to == ib->second) return;
        dep_edges_.push_back(DepEdge{ia->second, ib->second});
    }

    // ---- optimization marks: the last of a kind on a variable wins ---------------
    /// A free input of fixed shape.
    void add_design_var(std::string_view path, Bounds b = {}, Scaling s = {}) {
        require_declaring("add_design_var");
        const auto inst = instance_at(path);
        const detail::VarMeta& v = *inst->meta;
        if (inst->kind != IOKind::input)
            fail("add_design_var", "'" + std::string{path} + "' is an output; a design variable is a free input");
        if (const auto fed = detail::first_of_kind(v, IOKind::output))
            fail("add_design_var", "connected inputs are computed, not designed ('" + std::string{path} + "' is fed by '" +
                                       fed->full_path() + "')");
        if (!all_fixed(v))
            fail("add_design_var", "'" + std::string{path} + "' is dynamically shaped; a dynamic design variable is ill-formed");
        check_sizes(flat_size(v), b, std::nullopt, s, "add_design_var", path);
        store_mark(MarkDecl::Kind::design_var, path, inst, MarkRecord{std::move(b), std::nullopt, std::move(s)});
    }

    /// An output; `equals` and bounds are exclusive.
    void add_constraint(std::string_view path, Bounds b = {}, std::optional<std::vector<double>> equals = {}, Scaling s = {}) {
        require_declaring("add_constraint");
        const auto inst = instance_at(path);
        const detail::VarMeta& v = *inst->meta;
        if (inst->kind != IOKind::output) fail("add_constraint", "'" + std::string{path} + "' is not an output");
        if (equals && (b.lower || b.upper)) fail("add_constraint", "'equals' and bounds are exclusive");
        if (all_fixed(v)) check_sizes(flat_size(v), b, equals, s, "add_constraint", path);
        store_mark(MarkDecl::Kind::constraint, path, inst, MarkRecord{std::move(b), std::move(equals), std::move(s)});
    }

    /// An output of fixed shape with flat size 1.
    void add_objective(std::string_view path, Scaling s = {}) {
        require_declaring("add_objective");
        const auto inst = instance_at(path);
        const detail::VarMeta& v = *inst->meta;
        if (inst->kind != IOKind::output) fail("add_objective", "'" + std::string{path} + "' is not an output");
        if (!all_fixed(v))
            fail("add_objective", "'" + std::string{path} + "' is not fixed in shape; an objective is one value (flat size 1)");
        const std::size_t n = flat_size(v);
        if (n != 1)
            fail("add_objective", "'" + std::string{path} + "' has flat size " + std::to_string(n) + "; an objective is one value");
        check_sizes(n, Bounds{}, std::nullopt, s, "add_objective", path);
        store_mark(MarkDecl::Kind::objective, path, inst, MarkRecord{Bounds{}, std::nullopt, std::move(s)});
    }

    /// One declared mark: the path is relative to the declaring System; the Model's finalize makes it absolute.
    struct MarkDecl {
        enum class Kind : uint8_t { design_var, constraint, objective };
        Kind                            kind;
        std::string                     path;
        std::weak_ptr<detail::Instance> inst;
        MarkRecord                      rec;
    };
    const std::vector<MarkDecl>& marks() const noexcept { return marks_; }

    // ---- resolution & enumeration -------------------------------------------------
    /// The instance a path names, from THIS namespace: dotted, descending children; the last segment is a
    /// component's input or output, or an alias of the System reached; a single segment is this System's own alias.
    /// For wiring and marks, never for host access.
    std::shared_ptr<detail::Instance> instance_at(std::string_view path) {
        const auto seg = split_path(path);
        System*    s   = this;
        for (std::size_t i = 0; i + 1 < seg.size(); ++i) {
            const auto it = s->child_index_.find(std::string{seg[i]});
            if (it == s->child_index_.end())
                throw std::invalid_argument("naga::cospan: " + s->describe() + " has no subsystem '" + std::string{seg[i]} +
                                            "' (resolving '" + std::string{path} + "')");
            Subsystem* c = s->child_ptrs_[it->second];
            if (c == nullptr)
                throw std::logic_error("naga::cospan: '" + std::string{path} + "': subsystem '" + std::string{seg[i]} +
                                       "' does not exist right now (it was deleted)");
            if (auto* sub = dynamic_cast<System*>(c)) {
                s = sub;
                continue;
            }
            if (i + 2 != seg.size())
                throw std::invalid_argument("naga::cospan: '" + std::string{seg[i]} + "' is a Component; '" +
                                            std::string{path} + "' descends past it");
            return found(c->find_instance(seg.back()), *s, path);
        }
        return found(s->find_instance(seg.back()), *s, path);
    }

    /// Chains through children by dotted path. Null if a slot on the path exists but is empty (a deleted child);
    /// throws if a name was never declared.
    Subsystem* get_subsystem(std::string_view path) {
        const auto seg = split_path(path);
        System*    s   = this;
        Subsystem* c   = nullptr;
        for (std::size_t i = 0; i < seg.size(); ++i) {
            if (!s->child_index_.contains(std::string{seg[i]}))
                throw std::out_of_range("naga::cospan: " + s->describe() + " has no subsystem '" + std::string{seg[i]} +
                                        "' (resolving '" + std::string{path} + "')");
            c = s->find_child(seg[i]);
            if (c == nullptr) return nullptr;  // declared, but its slot is empty (the child was deleted)
            if (i + 1 < seg.size()) {
                s = dynamic_cast<System*>(c);
                if (s == nullptr)
                    throw std::invalid_argument("naga::cospan: '" + std::string{seg[i]} + "' is a Component; '" +
                                                std::string{path} + "' descends into it");
            }
        }
        return c;
    }

    /// Declaration order = execution order. Null entries are deleted children.
    std::span<Subsystem* const> children() const noexcept { return child_ptrs_; }

    /// The run policy asked for the child named `name` (a single segment): -1 eternal, -2 transient, n turns (run_n n,
    /// do_once = 1). -1 also for a name that is no slot.
    long run_policy_of(std::string_view name) const {
        const auto it = child_index_.find(std::string{name});
        return it == child_index_.end() ? -1 : slots_[it->second].requested;
    }
    /// The live child named `name`, or null (never declared, or deleted).
    const Subsystem* child_of(std::string_view name) const {
        const auto it = child_index_.find(std::string{name});
        return it == child_index_.end() ? nullptr : child_ptrs_[it->second];
    }

    // ---- Subsystem verbs ---------------------------------------------------------
    /// Runs setup() with the declaring flag up. The Model's setup is the only other pass.
    void run_setup() override {
        DeclScope scope{declaring_};
        setup();
        compute_hook_slots();  // after setup() and its declarations: where each hook runs, and the order of its sources
    }

    /// Section 2: the device of each own alias, then of every live child, recursively. The OUTER call wins: a
    /// device given to add_subsystem (or to a parent's own set_device) overrides every placement inside this subtree,
    /// and an author's placements are recommendations.
    void set_device(const sycl::device& d) override {
        forced_device_ = d;
        for (const auto& inst : alias_order_) {
            inst->device = d;
            if (inst->role == detail::Role::tie && inst->kind == IOKind::input && inst->meta->init_device) inst->meta->init_device = d;
        }
        for (Subsystem* c : child_ptrs_)
            if (c != nullptr) c->set_device(d);
    }

    std::shared_ptr<detail::Instance> find_instance(std::string_view name) const override {
        const auto it = aliases_.find(std::string{name});
        return it == aliases_.end() ? nullptr : it->second;
    }
    void collect_instances(std::vector<std::shared_ptr<detail::Instance>>& out) const override {
        for (const auto& a : alias_order_) out.push_back(a);
    }

    // ---- the forward (section 3) ---------------------------------------------------
    /// depends_on: these events are handed on to every child's submissions for ONE forward.
    void add_dependencies(std::span<const sycl::event> deps) override {
        pending_deps_.insert(pending_deps_.end(), deps.begin(), deps.end());
    }

    /// The events of this forward: the children's, plus this System's own push events.
    std::vector<sycl::event> events() const override { return forward_events_; }

    /// Rebuilds the transient children that are gone (the build allocates nothing), then the user's pre_forward().
    void run_precompute(detail::Runtime& rt) override {
        (void)rt;
        for (std::size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].requested == -2 && child_ptrs_[i] == nullptr) build_child(i, /*rebuild=*/true);
        DeclScope scope{in_pre_forward_};
        pre_forward();
    }

    /// Calls each live child's forward in declaration order, and nothing else. A child that has run its last turn
    /// (run_n reached 0, or a transient's one forward) is deleted right away: this waits on ITS events only, not the
    /// queue, so memory is freed before later children allocate while independent work already submitted keeps
    /// running. Add subsystems that can go out of scope as late as possible in setup().
    void run_compute(detail::Runtime& rt) override {
        forward_events_.clear();
        std::vector<std::vector<sycl::event>> dep_in(slots_.size());
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            Subsystem* child = child_ptrs_[i];
            if (child == nullptr || slots_[i].counter == 0) continue;
            for (detail::HookStep& hs : hooks_)  // the hooks whose target first appears in this child (section 10)
                if (hs.run_slot == i) hs.run(rt, forward_events_);
            std::vector<sycl::event> deps = dep_in[i];
            deps.insert(deps.end(), pending_deps_.begin(), pending_deps_.end());
            if (!deps.empty()) child->add_dependencies(deps);
            child->run_precompute(rt);
            child->run_compute(rt);
            child->run_postcompute(rt);
            const std::vector<sycl::event> ev = child->events();
            forward_events_.insert(forward_events_.end(), ev.begin(), ev.end());
            for (const DepEdge& e : dep_edges_)
                if (e.from == i) dep_in[e.to].insert(dep_in[e.to].end(), ev.begin(), ev.end());
            SlotInfo& si = slots_[i];
            si.ran       = true;
            if (si.requested > 0) {
                if (--si.counter == 0) delete_child(i);
            } else if (si.requested == -2) {
                delete_child(i);
            }
        }
        pending_deps_.clear();
    }

    /// The System that owns a tie alias pushes the tied outputs to the variable's other placements after ALL members
    /// have submitted, depending on all their writes (section 3). Then the user's post_forward().
    void run_postcompute([[maybe_unused]] detail::Runtime& rt) override {
        for (const auto& alias : alias_order_) {
            if (alias->role != detail::Role::tie || alias->kind != IOKind::output) continue;
            const std::shared_ptr<detail::DeviceDataBase> src = alias->data;
            if (!src) continue;  // nothing was written this forward, or the variable is not materialized
            const std::shared_ptr<detail::VarMeta> m = alias->meta;
            for (const auto& entry : m->data) {
                const std::shared_ptr<detail::DeviceDataBase> dst = entry.second.lock();
                if (!dst || dst == src || entry.first == src->key) continue;
                std::vector<sycl::event> deps = m->events.read_deps();  // every member's write of this forward
                const std::vector<sycl::event> own = src->events();
                deps.insert(deps.end(), own.begin(), own.end());
                const sycl::event e = dst->copy_from(*src, deps);
                src->register_event(e);  // a later writer of the source waits for the read
                forward_events_.push_back(e);
            }
        }
        post_forward();
    }

protected:
    /// "the root System" or "System 'path'": the prefix of every System error.
    std::string describe() const {
        return path().empty() ? std::string{"the root System"} : "System '" + path() + "'";
    }

    /// One SLOT per child: its name, a factory for a fresh one, the device it was given, the run policy asked for, the
    /// live counter and whether it has taken a turn. The parent owns the lifecycle. A slot, its name and its
    /// declarations outlive its child.
    struct SlotInfo {
        std::string                                 name;
        std::function<std::unique_ptr<Subsystem>()> make;                ///< a fresh child
        std::optional<sycl::device>                 device;              ///< as given to add_subsystem
        long                                        requested = -1;      ///< -1 eternal, -2 transient, n turns
        long                                        counter   = -1;      ///< live: turns left (n), else = requested
        bool                                        ran       = false;   ///< took a turn in an earlier forward
    };

    /// Builds slot i's child: from its factory, named and placed in this namespace, its run_setup run BEFORE this
    /// returns, then its device applied (the slot's, then ours if this System was placed: the outer call wins), then,
    /// on a rebuild, the wiring that names it replayed (section 2).
    void build_child(std::size_t i, bool rebuild) {
        SlotInfo&                  si = slots_[i];
        std::unique_ptr<Subsystem> c  = si.make();
        Subsystem&                 b  = *c;
        b.name_                       = si.name;
        b.path_                       = path().empty() ? si.name : path() + "." + si.name;
        b.owner_                      = this;
        children_[i]                  = std::move(c);
        child_ptrs_[i]                = &b;
        b.run_setup();
        if (si.device) b.set_device(*si.device);
        if (forced_device_) b.set_device(*forced_device_);
        if (rebuild) {
            replay_slot(i);
            compute_hook_slots();
        }
    }

    /// Re-runs a child by name: deletes it if it is live, builds it again (the factory, its setup, its device, and the
    /// replay of the wiring that names it), and gives it its full run policy again. For triggers only the author
    /// knows, e.g. re-running a do_once child from the previous run's results. Legal inside pre_forward() only.
    void rearm(std::string_view name) {
        if (!in_pre_forward_) throw std::logic_error("naga::cospan: rearm is legal inside pre_forward() only (" + describe() + ")");
        const auto it = child_index_.find(std::string{name});
        if (it == child_index_.end())
            throw std::invalid_argument("naga::cospan: " + describe() + ": rearm('" + std::string{name} + "'): no such subsystem");
        const std::size_t i = it->second;
        if (child_ptrs_[i] != nullptr) delete_child(i);
        build_child(i, /*rebuild=*/true);
        slots_[i].counter = slots_[i].requested;
        slots_[i].ran     = false;
    }

    /// Deletes slot i's child. It waits on THAT child's events only, never the queue (section 3); the child's
    /// instances then release their data. The slot, its name and its declarations stay.
    void delete_child(std::size_t i) {
        if (children_[i]) {
            sycl::event::wait(children_[i]->events());
            children_[i].reset();
        }
        child_ptrs_[i] = nullptr;
    }

    System& root_system() {
        System* s = this;
        while (s->owner() != nullptr) s = const_cast<System*>(s->owner());
        return *s;
    }

    std::vector<SlotInfo>                    slots_;
    std::vector<std::unique_ptr<Subsystem>>  children_;
    std::vector<Subsystem*>                  child_ptrs_;  ///< null once deleted
    std::unordered_map<std::string, std::size_t> child_index_;

    std::optional<sycl::device> forced_device_;  ///< set by set_device: applied to every child built later

private:
    /// Raises the declaring flag for the duration of setup().
    struct DeclScope {
        explicit DeclScope(bool& f) : f_{f} { f_ = true; }
        ~DeclScope() { f_ = false; }
        DeclScope(const DeclScope&)            = delete;
        DeclScope& operator=(const DeclScope&) = delete;
        bool&      f_;
    };

    // ---- guards & names --------------------------------------------------------------
    void require_declaring(const char* what) const {
        if (!declaring_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside setup() only (" + describe() + ")");
    }

    [[noreturn]] void fail(const char* what, const std::string& why) const {
        throw std::invalid_argument(std::string{"naga::cospan: "} + describe() + ": " + what + ": " + why);
    }

    void check_new_name(const std::string& name, const char* what) const {
        if (name.empty() || name.find('.') != std::string::npos)
            throw std::invalid_argument("naga::cospan: " + describe() + ": the " + what + " name '" + name +
                                        "' must be non-empty and contain no '.'");
        if (child_index_.contains(name) || aliases_.contains(name))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + name + "' is already a subsystem or alias name here");
    }

    // ---- paths -------------------------------------------------------------------------
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
            if (s.empty()) throw std::invalid_argument("naga::cospan: the path '" + std::string{p} + "' has an empty segment");
        return out;
    }
    static std::string first_seg(std::string_view p) { return std::string{p.substr(0, p.find('.'))}; }

    Subsystem* find_child(std::string_view name) const {
        const auto it = child_index_.find(std::string{name});
        return it == child_index_.end() ? nullptr : child_ptrs_[it->second];
    }

    std::shared_ptr<detail::Instance> found(std::shared_ptr<detail::Instance> i, const System& in, std::string_view path) const {
        if (!i)
            throw std::invalid_argument("naga::cospan: " + describe() + ": no input, output or alias '" + std::string{path} +
                                        "'" + (&in == this ? std::string{} : " (in " + in.describe() + ")"));
        return i;
    }

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

    /// The position of the child a path starts with, or none when it names this System's own alias.
    std::optional<std::size_t> order_of(std::string_view path) const {
        const auto it = child_index_.find(first_seg(path));
        return it == child_index_.end() ? std::nullopt : std::optional<std::size_t>{it->second};
    }

    /// Where a wiring call happens: this System's text, the child positions of the two ends, the root's warning set.
    detail::WiringSite site(std::optional<std::size_t> src_order = std::nullopt,
                            std::optional<std::size_t> dst_order = std::nullopt) {
        return detail::WiringSite{describe(), src_order, dst_order, &root_system().warned_cycles_};
    }

    detail::TieMembers resolve_members(std::initializer_list<std::string_view> members) {
        detail::TieMembers out;
        for (const std::string_view m : members) out.emplace_back(std::string{m}, instance_at(m));
        return out;
    }

    void store_alias(const std::shared_ptr<detail::Instance>& a) {
        alias_order_.push_back(a);
        aliases_.emplace(a->name, a);
    }

    // ---- adding children ---------------------------------------------------------------
    void declare_child(std::string name, const property_list& props, std::optional<sycl::device> device,
                       std::function<std::unique_ptr<Subsystem>()> make) {
        require_declaring("add_subsystem");
        check_new_name(name, "subsystem");
        const bool tr = props.has_property<property::transient>();
        if (tr && (props.has_property<property::run_n>() || props.has_property<property::do_once>()))
            throw std::invalid_argument("naga::cospan: " + describe() + ": add_subsystem('" + name +
                                        "'): transient cannot be combined with run_n or do_once; a transient subsystem "
                                        "runs every forward");
        const bool has_n = props.has_property<property::run_n>();
        const bool once  = props.has_property<property::do_once>();
        SlotInfo   si;
        si.name      = name;
        si.make      = std::move(make);
        si.device    = std::move(device);
        si.requested = has_n ? static_cast<long>(props.get_property<property::run_n>().n) : once ? 1 : tr ? -2 : -1;
        si.counter   = si.requested;
        const std::size_t i = slots_.size();
        child_index_.emplace(std::move(name), i);
        slots_.push_back(std::move(si));
        children_.emplace_back();
        child_ptrs_.push_back(nullptr);
        build_child(i, /*rebuild=*/false);
    }

    // ---- replay: the declarations that name a rebuilt child ------------------------------
    /// A wiring declaration: the first segments of the paths it names (a slot name makes it replay when that slot is
    /// rebuilt), what to do, and the variable it produced (weak: it keeps a variable alive only through instances).
    struct Decl {
        std::vector<std::string>                  refs;
        std::function<void(System&, std::size_t)> replay;
        std::weak_ptr<detail::VarMeta>            var;
    };

    void record(std::vector<std::string> refs, std::function<void(System&, std::size_t)> replay,
                const std::shared_ptr<detail::VarMeta>& var) {
        decls_.push_back(Decl{std::move(refs), std::move(replay), var});
    }

    void record_tie(std::initializer_list<std::string_view> members, const std::string& name, bool is_input,
                    const std::shared_ptr<detail::Instance>& alias) {
        std::vector<std::string> ms, refs;
        for (const std::string_view m : members) {
            ms.emplace_back(m);
            refs.push_back(first_seg(m));
        }
        const std::size_t k = decls_.size();
        record(std::move(refs),
               [k, ms = std::move(ms), name, is_input](System& sys, std::size_t i) { sys.replay_tie(k, ms, name, is_input, i); },
               alias->meta);
    }

    /// REPLAY (section 2): runs, in declaration order, every declaration that names slot i's name. The rebuilt side
    /// rejoins the variables that survive; nothing unchanged is validated again.
    void replay_slot(std::size_t i) {
        for (std::size_t k = 0; k < decls_.size(); ++k) {
            const auto& refs = decls_[k].refs;
            if (std::find(refs.begin(), refs.end(), slots_[i].name) == refs.end()) continue;
            const auto fn = decls_[k].replay;  // a copy: the list is not touched by a replay, but the call is not made through it
            fn(*this, i);
        }
    }

    void replay_connect(std::size_t k, const std::string& src, const std::string& dst) {
        const bool src_alive = path_alive(src), dst_alive = path_alive(dst);
        if (!src_alive && !dst_alive) return;
        if (src_alive && dst_alive) {
            const auto s = instance_at(src);
            const auto d = instance_at(dst);
            if (s->meta == d->meta) {
                decls_[k].var = s->meta;
                return;
            }
            const detail::WiringSite st = site(order_of(src), order_of(dst));
            detail::merge_definitions(st, "connect", *s->meta, *d->meta, s.get(), d.get());
            decls_[k].var = detail::merge_into(st, s->meta, d->meta);  // a data-holding variable survives, per merge_into
            detail::warn_cycle(st, *s, *d);                            // a pair already reported is not repeated
            return;
        }
        // The other end is a deleted child: the rebuilt end rejoins the variable that still lives, if any. Otherwise
        // it stays fresh, and materializing it later is the shape-not-known error.
        const std::shared_ptr<detail::VarMeta> var = decls_[k].var.lock();
        if (!var) return;
        const auto inst = instance_at(src_alive ? src : dst);
        if (inst->meta == var) return;
        const detail::WiringSite st = site(order_of(src), order_of(dst));
        detail::merge_definitions(st, "connect", *var, *inst->meta, nullptr, inst.get());
        decls_[k].var = detail::merge_into(st, var, inst->meta);
    }

    void replay_promote(std::size_t k, const std::string& child_path, const std::string& name) {
        const auto alias  = aliases_.at(name);
        const auto member = instance_at(child_path);
        if (alias->meta != member->meta) {
            const detail::WiringSite st = site();
            detail::merge_definitions(st, "promote", *member->meta, *alias->meta, member.get(), alias.get());
            detail::merge_into(st, member->meta, alias->meta);  // the member's variable wins
        }
        decls_[k].var = alias->meta;
    }

    void replay_tie(std::size_t k, const std::vector<std::string>& members, const std::string& name, bool is_input,
                    std::size_t i) {
        const auto alias = aliases_.at(name);
        for (const std::string& m : members) {
            if (first_seg(m) != slots_[i].name) continue;
            const auto inst = instance_at(m);
            if (inst->meta != alias->meta) {
                const detail::WiringSite st = site();
                detail::merge_definitions(st, is_input ? "tie_inputs" : "tie_outputs", *alias->meta, *inst->meta, alias.get(), inst.get());
                detail::merge_into(st, alias->meta, inst->meta);  // the alias wins; read again at each member
            }
            (is_input ? inst->can_be_target : inst->can_be_source) = false;
        }
        decls_[k].var = alias->meta;
    }

    void replay_init(std::size_t k, const std::string& target, const InitSpec& init) {
        const auto inst = instance_at(target);
        detail::declare_init(site(), inst, init);  // replayed in declaration order: the last one wins again
        decls_[k].var = inst->meta;
    }

    // ---- shape hooks ---------------------------------------------------------------------
    /// The slot of the child of THIS System that (transitively) declares `i`, if any (aliases of this System, host views
    /// and instances elsewhere in the tree have none).
    std::optional<std::size_t> slot_of(const detail::Instance& i) const {
        const Subsystem* p = i.declarer;
        while (p != nullptr && p->owner() != this) p = p->owner();
        if (p == nullptr) return std::nullopt;
        const auto it = child_index_.find(std::string{p->name()});
        if (it == child_index_.end()) return std::nullopt;
        return it->second;
    }

    /// Gives `inst`'s variable the hook's marks: the dimensions it owns become `hook` (an unspecified one; a dimension
    /// already owned by this hook stays), fixed_at_run ones become fixed; an input target is no longer a target. Safe to
    /// repeat (a rebuilt target replays it).
    void mark_hooked(detail::Instance& inst, const detail::HookStep& hs) {
        using How = detail::DimRecord::How;
        detail::VarMeta& m   = *inst.meta;
        const auto       own = [&](std::size_t u) {
            detail::DimRecord& r = m.definition[u];
            if (r.how == How::unspecified) {
                r.how        = How::hook;
                r.hook_owner = hs.owner;
            }
        };
        if (hs.dims.size() == 1 && hs.dims[0].is_ragged()) {
            own(0);
            own(1);
        } else {
            for (std::size_t u = 0; u < hs.dims.size(); ++u) {
                const detail::HookDim& d = hs.dims[u];
                if (d.kind == detail::HookDim::Kind::as_is) continue;
                if (d.kind == detail::HookDim::Kind::fixed_at_run) {
                    detail::DimRecord& r = m.definition[u];
                    if (r.how == How::unspecified) {
                        r.how    = How::fixed;
                        r.extent = d.index;
                    }
                } else {
                    own(u);
                }
            }
        }
        if (inst.kind == IOKind::input) {
            inst.can_be_target = false;  // a hook and a connection exclude each other
            inst.hooked        = true;
        }
    }

    /// Validates a hook declaration, marks the definition and stores the step. Returns its index in hooks_.
    std::size_t install_hook(const std::string& path, std::vector<detail::HookDim> dims) {
        using How = detail::DimRecord::How;
        const auto inst = instance_at(path);
        const auto bad  = [&](const std::string& why) { fail("shape_hook", "'" + path + "': " + why); };
        if (inst->kind != IOKind::input && !detail::is_tie_output(*inst)) bad("a shape hook targets an input or a tie alias");
        detail::VarMeta& m = *inst->meta;
        // An input that is already hooked was never connected or tied (connect refuses it), so for it the per-dimension
        // ownership check below answers: a second hook on a dimension gets "only one hook owns it", not the connected message.
        if (inst->kind == IOKind::input && !inst->hooked && (!inst->can_be_target || detail::first_of_kind(m, IOKind::output)))
            bad("it is connected or tied; a shape hook and a connection exclude each other");
        const bool ragged_target = m.is_ragged() && m.rank == 2;
        if (ragged_target) {
            if (dims.size() != 1 || !dims[0].is_ragged())
                bad("a ragged target takes ONE ragged descriptor (cs::ragged::counts_from or offsets_from) for both dimensions");
        } else {
            if (dims.size() != static_cast<std::size_t>(m.rank))
                bad("it has rank " + std::to_string(m.rank) + " but " + std::to_string(dims.size()) + " descriptors were given");
            for (const detail::HookDim& d : dims)
                if (d.is_ragged()) bad("a ragged descriptor (counts_from, offsets_from) is for a ragged target");
        }
        const auto need_unspecified = [&](std::size_t u) {
            if (m.definition[u].how != How::unspecified)
                bad("dimension " + std::to_string(u) + " is " + m.declared_rule(static_cast<int>(u)) +
                    "; a hook can only own an unspecified dimension (and only one hook owns it)");
        };
        bool any = false;
        if (ragged_target) {
            need_unspecified(1);
            if (m.definition[0].how == How::hook) bad("dimension 0 is already set by a shape hook");
            any = true;
        } else {
            for (std::size_t u = 0; u < dims.size(); ++u) {
                if (dims[u].kind == detail::HookDim::Kind::as_is) continue;
                need_unspecified(u);
                any = true;
            }
        }
        if (!any) bad("every descriptor is as_is: nothing to hook");
        for (detail::HookDim& d : dims) {
            if (!d.names_source()) continue;
            d.src = instance_at(d.path);
            const std::shared_ptr<detail::Instance> up = d.src.lock();
            if (d.kind != detail::HookDim::Kind::from_shape && !detail::is_integer_value_dtype(up->meta->dtype))
                bad("'" + d.path + "' drives a size by value and must have an integer dtype");
            if (d.kind == detail::HookDim::Kind::from_shape && d.index >= static_cast<std::size_t>(up->meta->rank))
                bad("'" + d.path + "' has rank " + std::to_string(up->meta->rank) + ", no dimension " + std::to_string(d.index));
        }
        detail::HookStep hs;
        hs.target  = inst;
        hs.dims    = std::move(dims);
        hs.owner   = ++root_system().next_hook_owner_;
        hs.path    = path;
        hs.resolve = [this](std::string_view p) { return instance_at(p); };
        mark_hooked(*inst, hs);
        hooks_.push_back(std::move(hs));
        return hooks_.size() - 1;
    }

    /// Where each hook runs: before the first child that has an instance of its target's variable. Every source's
    /// producer must be in an EARLIER child, else a setup error naming the hook, the source and both children. Run at the
    /// end of setup and after a rebuild.
    void compute_hook_slots() {
        for (detail::HookStep& hs : hooks_) {
            const std::shared_ptr<detail::Instance> target = hs.target.lock();
            if (!target) continue;
            std::size_t first = slots_.size();  // none: the hook never runs
            for (const auto& i : target->meta->live_instances())
                if (const auto s = slot_of(*i)) first = std::min(first, *s);
            hs.run_slot = first;
            for (const detail::HookDim& d : hs.dims) {
                if (!d.names_source()) continue;
                const std::shared_ptr<detail::Instance> src = d.src.lock();
                if (!src) continue;
                for (const auto& i : src->meta->live_instances()) {
                    if (!detail::is_output_port(*i)) continue;
                    const auto j = slot_of(*i);
                    if (j && *j >= first)
                        fail("shape_hook", "'" + hs.path + "': its source '" + d.path + "' is produced by '" + slots_[*j].name +
                                               "', which does not come before '" + slots_[first].name +
                                               "', the first child that has the target: a hook's sources must be produced by an earlier child");
                }
            }
        }
    }

    /// REPLAY (section 2) of a hook: a rebuilt target is re-marked and re-pointed, and the sources that live are resolved
    /// again.
    void replay_hook(std::size_t k, std::size_t h) {
        detail::HookStep& hs = hooks_[h];
        for (detail::HookDim& d : hs.dims)
            if (d.names_source() && path_alive(d.path)) d.src = instance_at(d.path);
        if (!path_alive(hs.path)) return;
        const auto inst = instance_at(hs.path);
        hs.target       = inst;
        mark_hooked(*inst, hs);
        decls_[k].var = inst->meta;
    }

    // ---- marks ---------------------------------------------------------------------------
    /// Every dimension is specified (a ragged variable's counts are not a single extent: it is never all fixed).
    static bool all_fixed(const detail::VarMeta& v) {
        return !v.def_counts && std::all_of(v.definition.begin(), v.definition.end(),
                                            [](const detail::DimRecord& r) { return r.how == detail::DimRecord::How::fixed; });
    }
    static std::size_t flat_size(const detail::VarMeta& v) {
        std::size_t n = 1;
        for (const detail::DimRecord& r : v.definition) n *= r.extent;
        return n;
    }

    /// The last declaration of a kind on a variable replaces the earlier one.
    void store_mark(MarkDecl::Kind kind, std::string_view path, const std::shared_ptr<detail::Instance>& inst, MarkRecord rec) {
        MarkDecl d{kind, std::string{path}, inst, std::move(rec)};
        for (MarkDecl& e : marks_) {
            if (e.kind != kind) continue;
            const auto o = e.inst.lock();
            if (o && o->meta == inst->meta) {
                e = std::move(d);
                return;
            }
        }
        marks_.push_back(std::move(d));
    }

    /// Bounds, equals and scaling are per-element or scalar (size 1 broadcasts).
    void check_sizes(std::size_t n, const Bounds& b, const std::optional<std::vector<double>>& eq, const Scaling& s,
                     const char* what, std::string_view path) const {
        auto ok  = [&](const std::vector<double>& v) { return v.size() == 1 || v.size() == n; };
        auto bad = [&](const char* which) {
            fail(what, std::string{"'"} + std::string{path} + "': " + which + " must have 1 or " + std::to_string(n) +
                           " values (the variable's flat size)");
        };
        if (b.lower && !ok(*b.lower)) bad("lower");
        if (b.upper && !ok(*b.upper)) bad("upper");
        if (eq && !ok(*eq)) bad("equals");
        if (!ok(s.ref)) bad("scaling ref");
        if (!ok(s.ref0)) bad("scaling ref0");
    }

    // ---- state ---------------------------------------------------------------------------
    struct DepEdge {
        std::size_t from, to;  ///< slot indices: `to` waits on `from`'s events
    };

    bool declaring_      = false;
    bool in_pre_forward_ = false;
    std::vector<DepEdge>     dep_edges_;
    std::vector<sycl::event> pending_deps_;   ///< depends_on events for this System, for one forward
    std::vector<sycl::event> forward_events_; ///< this forward's: the children's and the pushes

    std::unordered_map<std::string, std::shared_ptr<detail::Instance>> aliases_;  ///< this System owns its aliases (section 1)
    std::vector<std::shared_ptr<detail::Instance>>                     alias_order_;
    std::vector<Decl>                                                  decls_;
    std::vector<detail::HookStep>                                      hooks_;  ///< the shape hooks this System declared
    std::uint32_t                                                      next_hook_owner_ = 0;  ///< (the root's) hook owner ids
    std::vector<MarkDecl>                                              marks_;
    std::set<std::string> warned_cycles_;  ///< the cycle warnings already printed (the root's is the one used)
};

}  // namespace naga::cospan
