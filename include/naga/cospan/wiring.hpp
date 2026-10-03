#pragma once

// =============================================================================
// naga/cospan/wiring.hpp — the structural operations over instances
// =============================================================================
//
// connect, promote, tie_inputs, tie_outputs and the declaration of an initial value, as free functions in
// naga::cospan::detail over the instances of instance.hpp (docs/design-review-2026-10-03.md, sections 1 and 2).
// A call merges the shared variable metadata (VarMeta) of the instances it joins; there is no other record of the
// wiring, and nothing is ever unregistered (an instance that goes away just expires from its variable).
//
// THE THREE MERGE DIRECTIONS (section 1: whose metadata survives, and so whose shape definition and alias rule)
//   connect: the SOURCE's metadata wins. The target side's initializer is dropped: an initializer on a connected
//            input is accepted and ignored (the source supplies the values).
//   tie:     the ALIAS's metadata wins (it owns the shape). The members' initializers are folded into it by the
//            highest-level-wins rule (smaller depth wins; two at one depth from different declarers are an error).
//   promote: the MEMBER's metadata wins: the promotion joins the member's variable directly.
// One exception, in merge_into: when exactly one side already holds data (a rebuilt child rejoining live data),
// that side's object survives, carrying the rule-winner's shape definition and alias rule, and keeping its own
// resolved shape, initializer, stale flag, events and data. A merge never touches stale flags, events or data.
//
// EVERY RULE IS CHECKED IN ITS OWN CALL, BEFORE ANYTHING IS CHANGED: a call that throws has changed nothing
// that outlives it (a tie builds its alias first, but that is only reachable once the call returns).
//
// CALLER CONTRACT (the System, phase 4): it resolves paths to instances; it passes the WiringSite (its location
// text, the child positions of the two ends, its warning set); it passes the declaration depth and declarer
// path to declare_init; it keeps the returned alias instances alive (instances are owned by their declarer); and
// to wire a rebuilt child back in, it replays these same calls in declaration order. It checks names and the
// "legal inside setup()" guard itself: neither is a wiring rule.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/dtype.hpp"
#include "naga/cospan/init_spec.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/variable.hpp"

namespace naga::cospan {

/// How a tie_outputs treats non-atomic members.
enum class TieOutputsPolicy : uint8_t { require_atomic, allow_plain };

namespace detail {

/// Where a wiring call happens, supplied by the calling System.
struct WiringSite {
    std::string                where;                 ///< "the root System" or "System 'path'": the prefix of every error
    std::optional<std::size_t> src_order, dst_order;  ///< child positions in the calling System; nullopt = its own alias
    std::set<std::string>*     warned = nullptr;      ///< cycle warnings already printed (kept by the root System)
};

/// The error of a wiring rule: the format of System::fail ("naga::cospan: <where>: <verb>: <reason>").
[[noreturn]] inline void wiring_fail(const WiringSite& site, const char* verb, const std::string& reason) {
    throw std::invalid_argument(std::string{"naga::cospan: "} + site.where + ": " + verb + ": " + reason);
}

using TieMembers = std::vector<std::pair<std::string, std::shared_ptr<Instance>>>;
using ResolveFn  = std::function<std::shared_ptr<Instance>(std::string_view)>;

/// The live instances of a variable (const: expired entries are skipped, not pruned).
inline std::vector<std::shared_ptr<Instance>> live_of(const VarMeta& m) {
    std::vector<std::shared_ptr<Instance>> out;
    for (const auto& w : m.instances)
        if (std::shared_ptr<Instance> s = w.lock()) out.push_back(std::move(s));
    return out;
}

/// The first live instance of the variable of kind `k`, else null.
inline std::shared_ptr<Instance> first_of_kind(const VarMeta& m, IOKind k) {
    for (const auto& w : m.instances)
        if (std::shared_ptr<Instance> s = w.lock())
            if (s->kind == k) return s;
    return {};
}

// -----------------------------------------------------------------------------
// Compatibility
// -----------------------------------------------------------------------------

/// Two variables may merge: the value dtype, the rank, and every extent declared FIXED anywhere in either (the
/// definitions and every live instance's own dims) must agree, and a statically per-index dimension (PerIndex) meets
/// no fixed dense extent. Dimensions derived at run time (FromInputDim, FromInput, FromInputValues, the alias value
/// and dim specs) and unspecified ones are skipped: they are checked reactively, later. `ia` / `ib`: the instances
/// that are being joined, for the messages (else the variable's first live instance).
inline void check_compatible(const WiringSite& site, const char* verb, const VarMeta& a, const VarMeta& b,
                             const Instance* ia = nullptr, const Instance* ib = nullptr) {
    if (&a == &b) return;
    const std::string pa = ia != nullptr ? ia->full_path() : a.display_path();
    const std::string pb = ib != nullptr ? ib->full_path() : b.display_path();
    if (!dtype_compatible(a.dtype, b.dtype))
        wiring_fail(site, verb, "the value dtypes differ ('" + pa + "' and '" + pb + "')");
    if (a.rank != b.rank)
        wiring_fail(site, verb, "'" + pa + "' has rank " + std::to_string(a.rank) + " but '" + pb + "' has rank " +
                                    std::to_string(b.rank) + ": the rank must match");

    // Every dimension record that speaks about dimension d, with who declared it.
    struct Said {
        const DimRecord* rec;
        std::string      who;
    };
    for (int d = 0; d < a.rank; ++d) {
        const auto        u = static_cast<std::size_t>(d);
        std::vector<Said> said;
        auto              add_group = [&](const VarMeta& m, const std::string& name) {
            if (u < m.definition.size()) said.push_back({&m.definition[u], "'" + name + "' (the variable's shape definition)"});
        };
        add_group(a, pa);
        add_group(b, pb);
        const auto la = live_of(a);
        const auto lb = live_of(b);
        for (const auto* group : {&la, &lb})
            for (const auto& i : *group)
                if (u < i->dims.size()) said.push_back({&i->dims[u], "'" + i->full_path() + "'"});

        const Said* fixed = nullptr;
        const Said* per   = nullptr;
        for (const Said& s : said) {
            if (s.rec->how == DimRecord::How::fixed) {
                if (fixed != nullptr && fixed->rec->extent != s.rec->extent)
                    wiring_fail(site, verb, "dimension " + std::to_string(d) + ": " + fixed->who + " declares " +
                                                std::to_string(fixed->rec->extent) + " but " + s.who + " declares " +
                                                std::to_string(s.rec->extent) + ": the extents must match");
                if (fixed == nullptr) fixed = &s;
            } else if (s.rec->how == DimRecord::How::per_index_host && per == nullptr) {
                per = &s;
            }
        }
        if (fixed != nullptr && per != nullptr)
            wiring_fail(site, verb, "dimension " + std::to_string(d) + ": " + per->who + " gives per-index extents but " +
                                        fixed->who + " declares one extent (" + std::to_string(fixed->rec->extent) +
                                        ") for it");
    }
}

// -----------------------------------------------------------------------------
// Initializers
// -----------------------------------------------------------------------------

/// What the highest-level-wins rule needs to know of the declaration held so far.
struct InitState {
    bool        set   = false;
    std::size_t depth = 0;
    std::string by;
};

/// True when the incoming declaration (depth, by) replaces the one held: none held, a smaller depth, or the same depth
/// from the same declarer (a replay: the newer replaces). Throws when the same depth comes from a different declarer.
inline bool init_takes(const WiringSite& site, const char* verb, const InitState& held, std::size_t depth,
                       const std::string& by) {
    if (!held.set || depth < held.depth) return true;
    if (depth > held.depth) return false;
    if (by != held.by) {
        auto text = [](const std::string& s) { return s.empty() ? std::string{"the root System"} : s; };
        wiring_fail(site, verb, "two initializers for one variable at the same level ('" + text(held.by) + "' and '" +
                                    text(by) + "'); the higher level wins, so declare it once, higher up");
    }
    return true;
}

/// Folds the initializer of `from` into `into` by the highest-level-wins rule (init_takes). A taken initializer
/// brings its depth and declarer, and its device when `into` has none.
inline void fold_init(const WiringSite& site, VarMeta& into, const VarMeta& from, const char* verb = "init_input") {
    if (!from.init) return;
    const InitState held{into.init.has_value(), into.init_depth, into.init_by};
    if (!init_takes(site, verb, held, from.init_depth, from.init_by)) return;
    into.init       = from.init;
    into.init_depth = from.init_depth;
    into.init_by    = from.init_by;
    if (!into.init_device && from.init_device) into.init_device = from.init_device;
}

// -----------------------------------------------------------------------------
// The merge walk
// -----------------------------------------------------------------------------

/// The generic merge, after the call's own checks. Returns the surviving object; every live instance of the other
/// is rebound to it. See the header for who survives. Never touches stale flags, events or data.
inline std::shared_ptr<VarMeta> merge_into(const WiringSite& site, std::shared_ptr<VarMeta> winner,
                                           std::shared_ptr<VarMeta> loser) {
    if (winner == loser) return winner;
    const bool wm = winner->is_materialized();
    const bool lm = loser->is_materialized();
    if (wm && lm) throw std::logic_error("naga::cospan: " + site.where + ": both sides of a merge hold data");
    std::shared_ptr<VarMeta> survivor = winner;
    std::shared_ptr<VarMeta> other    = loser;
    if (lm) {  // a rebuilt child rejoining live data: its object survives, with the rule-winner's rules
        survivor             = loser;
        other                = winner;
        survivor->definition  = winner->definition;
        survivor->alias_shape = winner->alias_shape;
    }
    for (const std::shared_ptr<Instance>& i : other->live_instances()) i->rebind(survivor);
    return survivor;
}

/// Both-sides-hold-data, checked before a multi-step merge begins (merge_into checks it again, per step).
inline void check_one_live(const WiringSite& site, const std::vector<const VarMeta*>& metas) {
    std::vector<const VarMeta*> live;
    for (const VarMeta* m : metas) {
        if (!m->is_materialized()) continue;
        bool seen = false;
        for (const VarMeta* l : live) seen = seen || l == m;
        if (!seen) live.push_back(m);
    }
    if (live.size() > 1) throw std::logic_error("naga::cospan: " + site.where + ": both sides of a merge hold data");
}

// -----------------------------------------------------------------------------
// connect
// -----------------------------------------------------------------------------

/// The consumer reads a producer that runs after it: one line on std::cerr per (consumer, producer) pair, never an
/// error. Without a solver the consumer sees the producer's previous value (its initial state on the first forward).
inline void warn_cycle(const WiringSite& site, const Instance& src, const Instance& dst) {
    if (!site.src_order || !site.dst_order || *site.src_order <= *site.dst_order) return;
    const std::string line = "'" + dst.full_path() + "' reads '" + src.full_path() + "'";
    if (site.warned != nullptr && !site.warned->insert(line).second) return;
    std::cerr << "naga::cospan: warning: " << line
              << ", which runs after it: this is a cycle; without a solver the consumer sees the producer's "
                 "previous value (or its initial state on the first forward)\n";
}

/// Feeds `dst` from `src`; the source's variable wins. Initializers on a connected input are ignored. Flags are not
/// changed (an input has at most one source: the check below).
inline void connect(const WiringSite& site, const std::shared_ptr<Instance>& src, const std::shared_ptr<Instance>& dst) {
    const std::string what = "connect('" + src->full_path() + "', '" + dst->full_path() + "')";
    if (!src->can_be_source)
        wiring_fail(site, what.c_str(),
                    "'" + src->full_path() + "' cannot be a source (an input, or a tied output: connect from its tie alias)");
    if (!dst->can_be_target)
        wiring_fail(site, what.c_str(),
                    "'" + dst->full_path() + "' cannot be a target (an output, or a tied input: connect through its tie alias)");
    if (const std::shared_ptr<Instance> fed = first_of_kind(*dst->meta, IOKind::output))
        wiring_fail(site, what.c_str(),
                    "'" + dst->full_path() + "' already has a source (its variable is fed by '" + fed->full_path() +
                        "'); an input has at most one");
    if (src->meta == dst->meta)
        wiring_fail(site, what.c_str(),
                    "'" + src->full_path() + "' and '" + dst->full_path() + "' are already connected (they share a variable)");
    if (dst->is_dense_input_port() && src->meta->is_ragged())
        wiring_fail(site, what.c_str(),
                    "'" + src->full_path() + "' is ragged; a dense Input cannot see per-index extents; declare '" +
                        dst->full_path() + "' RaggedInput");
    check_compatible(site, what.c_str(), *src->meta, *dst->meta, src.get(), dst.get());
    check_one_live(site, {src->meta.get(), dst->meta.get()});

    merge_into(site, src->meta, dst->meta);
    warn_cycle(site, *src, *dst);
}

// -----------------------------------------------------------------------------
// promote
// -----------------------------------------------------------------------------

/// A promotion's dims. Its shape is read live from the variable, so the dims are only classification: fixed and
/// unspecified are kept, derived ones become inert from_input markers.
inline std::vector<DimRecord> promoted_dims(const std::vector<DimRecord>& member) {
    std::vector<DimRecord> out;
    for (DimRecord r : member) {
        if (r.how != DimRecord::How::fixed && r.how != DimRecord::How::unspecified) {
            r.how         = DimRecord::How::from_input;
            r.input.clear();
            r.dim         = 0;
            r.value_index = 0;
        }
        out.push_back(std::move(r));
    }
    return out;
}

/// Every instance is of dtype DT and rank Rank.
template <DType DT, int Rank>
void check_typed_instance(const WiringSite& site, const char* verb, const Instance& e) {
    if (!dtype_compatible(e.meta->dtype, DT))
        wiring_fail(site, verb, "'" + e.full_path() + "' has a different dtype than the alias (the alias is declared <DType " +
                                    std::to_string(static_cast<int>(DT)) + ", " + std::to_string(Rank) + ">)");
    if (e.meta->rank != Rank)
        wiring_fail(site, verb, "'" + e.full_path() + "' has rank " + std::to_string(e.meta->rank) +
                                    " but the alias is declared rank " + std::to_string(Rank));
}

/// A PEER alias of `member` in the declarer's namespace, with the member's flags. It joins the member's variable (the
/// member's metadata wins; nothing merges). Its device is the member's: an outer set_device may override it later.
template <DType DT, int Rank>
std::shared_ptr<Instance> promote(const WiringSite& site, const std::shared_ptr<Instance>& member, std::string name,
                                  const Subsystem* declarer) {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    check_typed_instance<DT, Rank>(site, "promote", *member);
    std::shared_ptr<Instance> a = make_alias_instance(DT, Rank, std::move(name), declarer, member->kind, Role::promotion,
                                                      promoted_dims(member->dims), member->can_be_source,
                                                      member->can_be_target, member->meta);
    a->device = member->device;
    return a;
}

// -----------------------------------------------------------------------------
// Ties
// -----------------------------------------------------------------------------

/// A tie alias's dims: fixed extents for alias::fixed and fixed specs; unspecified when the shape is left to what
/// supplies the tie; placeholders (resolved reactively) for the others.
template <int Rank>
std::vector<DimRecord> alias_dims(const AliasShape& s) {
    std::vector<DimRecord> out(static_cast<std::size_t>(Rank));
    for (int d = 0; d < Rank; ++d) {
        const auto u = static_cast<std::size_t>(d);
        DimRecord& r = out[u];
        if (s.kind == AliasShape::Kind::fixed) {
            r.how    = DimRecord::How::fixed;
            r.extent = s.extents[u];
        } else if (s.kind == AliasShape::Kind::unspecified) {
            r.how = DimRecord::How::unspecified;
        } else if (s.kind == AliasShape::Kind::dims && s.dim_specs[u].how == AliasDim::How::fixed) {
            r.how    = DimRecord::How::fixed;
            r.extent = s.dim_specs[u].extent;
        } else {
            r.how = DimRecord::How::from_input;
            if (s.kind == AliasShape::Kind::dims) {  // statically per-index dims (resolved reactively)
                const AliasDim& sp = s.dim_specs[u];
                if (sp.how == AliasDim::How::per_index_host) {
                    r.how = DimRecord::How::per_index_host;
                    r.per = true;
                } else if (sp.how == AliasDim::How::output_values) {
                    r.per = true;
                }
            }
        }
    }
    return out;
}

/// At least two members, none null, none listed twice.
inline void check_tie_members(const WiringSite& site, const char* verb, const TieMembers& ms) {
    if (ms.size() < 2) wiring_fail(site, verb, "a tie needs two or more members; use promote for one");
    for (std::size_t i = 0; i < ms.size(); ++i) {
        if (!ms[i].second) wiring_fail(site, verb, "'" + ms[i].first + "' names no instance");
        for (std::size_t j = 0; j < i; ++j)
            if (ms[j].second == ms[i].second)
                wiring_fail(site, verb, "'" + ms[i].first + "' is listed twice (as '" + ms[j].first + "')");
    }
}

/// Validates an AliasShape against the alias's Rank, and records on it the upstream instances it reads (AliasSrc).
template <int Rank>
void check_alias_shape(const WiringSite& site, AliasShape& s, const char* verb, bool is_input, const ResolveFn& resolve) {
    auto fail = [&](const std::string& why) { wiring_fail(site, verb, why); };
    auto up_of = [&](const std::string& path) {
        std::shared_ptr<Instance> up = resolve(path);
        if (!up) fail("the alias shape names '" + path + "', which is no instance of the declaring System");
        return up;
    };
    auto is_source_output = [](const Instance& up) { return up.kind == IOKind::output && up.can_be_source; };
    switch (s.kind) {
        case AliasShape::Kind::dims: {
            if (s.rank != Rank || static_cast<int>(s.dim_specs.size()) != Rank)
                fail("alias::dims gives " + std::to_string(s.dim_specs.size()) + " specs but the alias is rank " +
                     std::to_string(Rank));
            for (int d = 0; d < Rank; ++d) {
                AliasDim&         sp  = s.dim_specs[static_cast<std::size_t>(d)];
                const std::string dim = "alias::dims, dimension " + std::to_string(d) + ": ";
                if (sp.how == AliasDim::How::fixed) continue;
                if (sp.how == AliasDim::How::per_index_host) {
                    if (d == 0) fail(dim + "a per-index spec is for dimensions >= 1 (dimension 0 is the number of indices)");
                    const AliasDim& d0 = s.dim_specs[0];
                    if (d0.how == AliasDim::How::fixed && sp.values.size() != d0.extent)
                        fail(dim + "per_index has " + std::to_string(sp.values.size()) + " extents but dimension 0 is " +
                             std::to_string(d0.extent));
                    continue;
                }
                if (sp.how == AliasDim::How::output_values && d == 0)
                    fail(dim + "output_values gives per-index extents, which are for dimensions >= 1");
                const std::shared_ptr<Instance> up = up_of(sp.path);
                if (sp.input) {
                    if (up->kind != IOKind::input)
                        fail(dim + "'" + sp.path + "' is not an input (input_value / input_values / input_dim name an input of "
                                   "the declaring system: a promoted input or a tie_inputs alias)");
                } else if (!is_source_output(*up)) {
                    fail(dim + "'" + sp.path + "' is not an output that can be a source");
                }
                if (sp.how != AliasDim::How::output_dim && !is_integer_value_dtype(up->meta->dtype))
                    fail(dim + "'" + sp.path + "' is not of integer dtype");
                if (sp.how == AliasDim::How::output_values && up->meta->rank > 2)
                    fail(dim + "'" + sp.path + "' drives per-index extents by value and must be rank 1 [N] or rank 2 [N,k]");
                if (sp.how == AliasDim::How::output_dim && static_cast<int>(sp.index) >= up->meta->rank)
                    fail(dim + "'" + sp.path + "' has rank " + std::to_string(up->meta->rank) + ", no dimension " +
                         std::to_string(sp.index));
                sp.src = AliasSrc{sp.path, up};
            }
            return;
        }
        case AliasShape::Kind::fixed:
            if (s.rank != Rank)
                fail("alias::fixed gives " + std::to_string(s.rank) + " extents but the alias is rank " + std::to_string(Rank));
            return;
        case AliasShape::Kind::unspecified:
            if (!is_input)
                fail("an output alias needs a shape (alias::fixed, dims, from_output_shape, from_output_values or "
                     "from_input_shape): an output with unspecified dims is an error");
            return;
        case AliasShape::Kind::from_input_shape: {
            const std::shared_ptr<Instance> up = up_of(s.path);
            if (up->kind != IOKind::input) fail("alias::from_input_shape('" + s.path + "') names an entry that is not an input");
            if (up->meta->rank != Rank)
                fail("alias::from_input_shape('" + s.path + "') names an input of rank " + std::to_string(up->meta->rank) +
                     " but the alias is rank " + std::to_string(Rank));
            s.src = AliasSrc{s.path, up};
            return;
        }
        case AliasShape::Kind::from_output_shape:
        case AliasShape::Kind::from_output_values: {
            const std::shared_ptr<Instance> up = up_of(s.path);
            if (!is_source_output(*up)) fail("the alias shape names '" + s.path + "', which is not an output that can be a source");
            if (s.kind == AliasShape::Kind::from_output_shape && up->meta->rank != Rank)
                fail("alias::from_output_shape('" + s.path + "') names an output of rank " + std::to_string(up->meta->rank) +
                     " but the alias is rank " + std::to_string(Rank));
            if (s.kind == AliasShape::Kind::from_output_values && !is_integer_value_dtype(up->meta->dtype))
                fail("alias::from_output_values('" + s.path + "') names an output that is not of integer dtype");
            s.src = AliasSrc{s.path, up};
            return;
        }
    }
}

/// A fresh variable and alias instance for a tie. The variable is ragged when any dimension is per-index or any
/// member's variable is ragged (a ragged member's data must be of the ragged kind).
template <DType DT, int Rank>
std::shared_ptr<Instance> build_tie_alias(const TieMembers& ms, std::string name, const Subsystem* declarer,
                                          const AliasShape& shape, IOKind kind, std::optional<sycl::device> device) {
    bool ragged = false;
    for (const auto& m : ms) ragged = ragged || m.second->meta->is_ragged();
    std::vector<DimRecord> dims = alias_dims<Rank>(shape);
    ragged                      = ragged || any_per_index(dims);
    for (const DimRecord& r : dims) ragged = ragged || r.how == DimRecord::How::per_index_from_input;
    std::shared_ptr<VarMeta> meta = make_var_meta(DT, Rank, dims, ragged);
    meta->alias_shape             = shape;
    meta->init_device             = device;
    const bool input              = kind == IOKind::input;
    std::shared_ptr<Instance> alias = make_alias_instance(DT, Rank, std::move(name), declarer, kind,
                                                          Role::tie, std::move(dims), /*can_source=*/!input,
                                                          /*can_target=*/input, std::move(meta));
    if (device) alias->device = *device;
    return alias;
}

/// Everything a tie checks about the merge itself, before the first one: each member against the alias, the members
/// against each other, at most one side holding data, and (inputs) the initializers folding without a conflict.
inline void check_tie_merge(const WiringSite& site, const char* verb, const std::shared_ptr<Instance>& alias,
                            const TieMembers& ms, bool inputs) {
    std::vector<const VarMeta*> metas;
    for (const auto& m : ms) {
        check_compatible(site, verb, *alias->meta, *m.second->meta, alias.get(), m.second.get());
        metas.push_back(m.second->meta.get());
    }
    for (std::size_t i = 0; i < ms.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            check_compatible(site, verb, *ms[j].second->meta, *ms[i].second->meta, ms[j].second.get(), ms[i].second.get());
    check_one_live(site, metas);
    if (inputs) {
        InitState held{};
        for (const auto& m : ms) {
            const VarMeta& v = *m.second->meta;
            if (!v.init) continue;
            if (init_takes(site, verb, held, v.init_depth, v.init_by)) held = InitState{true, v.init_depth, v.init_by};
        }
    }
}

/// The merge of a tie, after check_tie_merge: the alias's variable wins each time (`alias->meta` is read again at
/// every step: it is the survivor of the one before).
inline void apply_tie_merge(const WiringSite& site, const char* verb, const std::shared_ptr<Instance>& alias,
                            const TieMembers& ms, bool inputs) {
    for (const auto& m : ms) {
        if (inputs && !alias->meta->is_materialized()) fold_init(site, *alias->meta, *m.second->meta, verb);
        merge_into(site, alias->meta, m.second->meta);
    }
}

/// A NEW variable (target, not source); every member's variable merges into it and the alias's metadata wins. The
/// members are inputs of dtype DT and rank Rank; their unspecified dims come from the tie and their specified dims
/// are constraints on it. The members' initializers fold into it (smallest depth wins). `device`: where the tie's
/// initialization runs. Members become non-targets: they connect through the alias.
template <DType DT, int Rank>
std::shared_ptr<Instance> tie_inputs(const WiringSite& site, const TieMembers& members, std::string name,
                                     const Subsystem* declarer, AliasShape shape, std::optional<sycl::device> device,
                                     const ResolveFn& resolve) {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    const char* verb = "tie_inputs";
    check_tie_members(site, verb, members);
    for (const auto& [path, m] : members) {
        if (m->kind != IOKind::input)
            wiring_fail(site, verb, "'" + path + "' is not an input (tie before connecting, or connect from the alias)");
        if (const std::shared_ptr<Instance> fed = first_of_kind(*m->meta, IOKind::output))
            wiring_fail(site, verb, "'" + path + "' is not a free input: its variable is fed by '" + fed->full_path() +
                                        "' (tie before connecting, or connect from the alias)");
        if (!m->can_be_target) wiring_fail(site, verb, "'" + path + "' is already tied: connect through its tie alias");
        check_typed_instance<DT, Rank>(site, verb, *m);
    }
    check_alias_shape<Rank>(site, shape, verb, /*is_input=*/true, resolve);

    std::shared_ptr<Instance> alias = build_tie_alias<DT, Rank>(members, std::move(name), declarer, shape, IOKind::input, device);
    check_tie_merge(site, verb, alias, members, /*inputs=*/true);
    apply_tie_merge(site, verb, alias, members, /*inputs=*/true);
    for (const auto& m : members) m.second->can_be_target = false;  // visible at every use site
    return alias;
}

/// A NEW variable (source, not target); atomic member dtypes unless allow_plain. The members are outputs of dtype DT
/// and rank Rank on ONE device; the alias owns the shape (an output alias defines the shape of its connections, so
/// it cannot take its shape from one). Members become non-sources: they connect from the alias.
template <DType DT, int Rank>
std::shared_ptr<Instance> tie_outputs(const WiringSite& site, const TieMembers& members, std::string name,
                                      const Subsystem* declarer, AliasShape shape, TieOutputsPolicy policy,
                                      const ResolveFn& resolve) {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    const char* verb = "tie_outputs";
    check_tie_members(site, verb, members);
    for (const auto& [path, m] : members) {
        if (m->kind != IOKind::output) wiring_fail(site, verb, "'" + path + "' is not an output");
        if (!m->can_be_source) wiring_fail(site, verb, "'" + path + "' is already tied: connect from its tie alias");
        if (const std::shared_ptr<Instance> down = first_of_kind(*m->meta, IOKind::input))
            wiring_fail(site, verb, "'" + path + "' is already connected downstream to '" + down->full_path() +
                                        "': tie before connecting, or connect from the alias");
        if (policy == TieOutputsPolicy::require_atomic && !is_atomic(m->meta->dtype))
            wiring_fail(site, verb, "'" + path + "' is not atomic (require_atomic; pass allow_plain to assert the producers do not race)");
        check_typed_instance<DT, Rank>(site, verb, *m);
    }
    const auto& first = members.front();
    for (const auto& m : members)
        if (m.second->device != first.second->device)
            wiring_fail(site, verb, "tie_outputs needs every member on one device: '" + first.first + "' is on " +
                                        first.second->device.get_info<sycl::info::device::name>() + " and '" + m.first +
                                        "' on " + m.second->device.get_info<sycl::info::device::name>());
    check_alias_shape<Rank>(site, shape, verb, /*is_input=*/false, resolve);

    std::shared_ptr<Instance> alias =
        build_tie_alias<DT, Rank>(members, std::move(name), declarer, shape, IOKind::output, first.second->device);
    check_tie_merge(site, verb, alias, members, /*inputs=*/false);
    apply_tie_merge(site, verb, alias, members, /*inputs=*/false);
    for (const auto& m : members) m.second->can_be_source = false;
    return alias;
}

// -----------------------------------------------------------------------------
// Initial values
// -----------------------------------------------------------------------------

/// Declares the initial value of a free input's variable, at `depth` by `by` (the declarer's path): the smallest depth
/// wins, an equal depth from a different declarer throws, an equal depth from the same declarer replaces. An output
/// throws. Declaring one on a connected input is accepted and has no effect (the variable's source wins), as before:
/// nothing is stored when the variable has a producer.
inline void declare_init(const WiringSite& site, const std::shared_ptr<Instance>& inst, InitSpec spec, std::size_t depth,
                         std::string by) {
    if (inst->kind != IOKind::input)
        wiring_fail(site, "init_input", "'" + inst->full_path() + "' is an output; outputs are never initialized");
    VarMeta& v = *inst->meta;
    if (first_of_kind(v, IOKind::output)) return;
    const InitState held{v.init.has_value(), v.init_depth, v.init_by};
    if (!init_takes(site, "init_input", held, depth, by)) return;
    v.init       = std::move(spec);
    v.init_depth = depth;
    v.init_by    = std::move(by);
}

}  // namespace detail
}  // namespace naga::cospan
