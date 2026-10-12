#pragma once

// =============================================================================
// naga/cospan/run_policy.hpp — run policies, dearm and rearm (r04, Phase 7)
// =============================================================================
//
// A run policy is its own kind of object, attached to a tree NODE (a leaf or a system). It has three calls: `before`
// (after the node's before-operations, before the node itself), `after` (after the node's after-operations) and
// `rearm`. Both `before` and `after` are called whether the node is in scope or not; the policy decides what they do.
//
//   run_n{n} / do_once   count turns; when they run out, `after` dearms the node's subtree. `rearm` resets the counter.
//   transient            `before` rearms the subtree, `after` dearms it. Its own `rearm` does nothing.
//
// There is no default policy: a node with none just runs. The last caller wins (nothing compares old and new).
//
// DEARM walks a node's subtree: every node goes out of scope; every holder in it releases its device data and keeps its
// variable metadata (so the variable lives until the model is reset). The data itself is freed when the LAST holder of a
// placement lets go (each free waits on its own events). REARM walks the same way: every node is back in scope, every
// policy is rearmed, every holder takes a container again, and its variable joins the pending set, so that the start-of-
// run pass materializes it (and runs a free input's initializer again if all of its data had been freed).

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "naga/cospan/model_meta.hpp"

namespace naga::cospan {

// -----------------------------------------------------------------------------
// Properties (SYCL idiom): unordered, self-naming, type-checked
// -----------------------------------------------------------------------------

namespace property {
struct run_n { std::size_t n; };  ///< run policy: the node runs on its first n turns, then goes out of scope (n >= 1)
struct do_once {};                ///< run policy: run_n{1}
struct transient {};              ///< run policy: in scope for its turn only, rearmed before it and dearmed after it
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

namespace detail {

class RunPolicy {
public:
    virtual ~RunPolicy() = default;
    virtual void before(ModelMeta& m, Node& n) {
        (void)m, (void)n;
    }
    virtual void after(ModelMeta& m, Node& n) {
        (void)m, (void)n;
    }
    virtual void rearm() {}
};

// -----------------------------------------------------------------------------
// dearm / rearm of a subtree
// -----------------------------------------------------------------------------

/// Takes the subtree out of scope: every holder releases its device data (and keeps its variable metadata).
inline void dearm_subtree(ModelMeta& m, Node& root) {
    (void)m;
    std::vector<Node*> stack{&root};
    while (!stack.empty()) {
        Node* n = stack.back();
        stack.pop_back();
        n->in_scope = false;
        for (const std::weak_ptr<Instance>& w : n->holders)
            if (const std::shared_ptr<Instance> inst = w.lock()) {
                inst->dearmed = true;  // the data goes at the end of the run (release_dearmed): a later node of this run may still read it
            }
        m.release_later.push_back(n);
        for (Node* c : n->children) stack.push_back(c);
    }
}

/// The end of a run: the data of the holders of every node that is still out of scope is released (their variable metadata stays).
inline void release_dearmed(ModelMeta& m) {
    for (Node* n : m.release_later)
        if (!n->in_scope)
            for (const std::weak_ptr<Instance>& w : n->holders)
                if (const std::shared_ptr<Instance> inst = w.lock())
                    if (inst->dearmed) inst->data.reset();
    m.release_later.clear();
}

/// Puts the subtree back in scope: policies rearmed, every holder given a container again, every variable pending.
inline void rearm_subtree(ModelMeta& m, Node& root) {
    std::vector<Node*> stack{&root};
    while (!stack.empty()) {
        Node* n = stack.back();
        stack.pop_back();
        n->in_scope = true;
        if (n->policy) n->policy->rearm();
        for (const std::weak_ptr<Instance>& w : n->holders) {
            const std::shared_ptr<Instance> inst = w.lock();
            if (!inst) continue;
            inst->dearmed = false;
            inst->refresh_container();
            // A variable all of whose data was freed starts over: it materializes (and, for a free input, initializes)
            // again in the start-of-run pass.
            if (!inst->meta->has_live_data()) inst->meta->forget_materialization();
            m.mark_pending(inst->meta);
        }
        for (Node* c : n->children) stack.push_back(c);
    }
}

// -----------------------------------------------------------------------------
// The policies
// -----------------------------------------------------------------------------

/// run_n and do_once: count the turns the node takes; dearm the subtree after the last one.
class CountedPolicy final : public RunPolicy {
public:
    explicit CountedPolicy(std::size_t n) : n_{n}, left_{n} {}
    void after(ModelMeta& m, Node& node) override {
        if (!node.in_scope || left_ == 0) return;  // a turn is only counted while the node runs
        if (--left_ == 0) dearm_subtree(m, node);
    }
    void rearm() override { left_ = n_; }

private:
    std::size_t n_, left_;
};

/// transient: in scope for its turn only.
class TransientPolicy final : public RunPolicy {
public:
    void before(ModelMeta& m, Node& node) override {
        if (!node.in_scope) rearm_subtree(m, node);
    }
    void after(ModelMeta& m, Node& node) override { dearm_subtree(m, node); }
};

/// The policy a property_list asks for, or null for none. `where` prefixes the errors.
inline std::shared_ptr<RunPolicy> make_run_policy(const property_list& props, const std::string& where) {
    const bool tr    = props.has_property<property::transient>();
    const bool has_n = props.has_property<property::run_n>();
    const bool once  = props.has_property<property::do_once>();
    if (tr && (has_n || once))
        throw std::invalid_argument("naga::cospan: " + where + ": transient cannot be combined with run_n or do_once");
    if (tr) return std::make_shared<TransientPolicy>();
    if (has_n) return std::make_shared<CountedPolicy>(props.get_property<property::run_n>().n);
    if (once) return std::make_shared<CountedPolicy>(1);
    return nullptr;
}

}  // namespace detail
}  // namespace naga::cospan
