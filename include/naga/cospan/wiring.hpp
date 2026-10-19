#pragma once

// =============================================================================
// naga/cospan/wiring.hpp — the structural operations over instances
// =============================================================================
//
// connect, promote, tie_inputs, tie_outputs and the declaration of an initial value, as free functions in
// naga::cospan::detail over the instances of instance.hpp (internal/design-review-2026-10-03.md, sections 1 and 2).
// A call merges the shared variable metadata (VarMeta) of the instances it joins; there is no other record of the
// wiring, and nothing is ever unregistered (an instance that goes away just expires from its variable).
//
// THE THREE MERGE DIRECTIONS (section 1: whose metadata survives). The SHAPE DEFINITIONS of everything a call joins
// are merged (section 10, merge_definitions): unspecified absorbs specified, two specified values must agree, a hooked
// dimension meets only unspecified; promotions take no shape.
//   connect: the SOURCE's metadata wins. The target side's initializer is dropped: an initializer on a connected
//            input is accepted and ignored (the source supplies the values) -- except on a CYCLE (a consumer registered
//            before its producer): the variable is marked `cycle`, keeps the target's initializer, and runs it at its
//            first materialization (variable.hpp).
// Wiring calls are legal at any time (design review, 11.0: no phase restrictions without a real reason). A merge of two
// variables that are BOTH materialized is an error at the call: "merging two materialized variables is not defined (yet)".
//   tie:     the ALIAS's metadata wins (it owns the shape). The members' initializers are folded into it by the
//            last-declaration-wins rule (design review, section 9; a tie's own initializer replaces its members').
//            A member's initializer is carried only when the alias has none.
//   promote: the MEMBER's metadata wins: the promotion joins the member's variable directly.
// A merge never touches stale flags or the data of a placement that has some; it only moves HOLDERS: the survivor
// adopts the loser's containers for the keys it has none for, and every holder of the loser switches to the survivor's
// container for its own key (Instance::rebind). Declarations happen before the first run, so no variable holds data
// when a merge happens; two materialized variables are still refused, as a guard.
// "Holds data" below means the variable is materialized (VarMeta::materialized), not that a container exists:
// every holder has an empty container from its declaration.
//
// EVERY RULE IS CHECKED IN ITS OWN CALL, BEFORE ANYTHING IS CHANGED: a call that throws has changed nothing
// that outlives it (a tie builds its alias first, but that is only reachable once the call returns).
//
// CALLER CONTRACT (the System, r04): it resolves paths to instances through the model metadata; it passes the
// WiringSite (its location text, the leaf ordinals of the two ends, the model's warning set); it passes the
// initializer to declare_init (the last declaration wins); it keeps the returned alias instances alive (instances are
// owned by their declarer). Nothing is replayed. It checks names and the lifecycle guard itself: neither is a wiring
// rule. TIES OF TIES are allowed: a tie alias (or a promotion of one) may be a member of an outer tie; the System marks
// the inner alias ABSORBED, and the existing merge rules decide the rest.

#include <algorithm>
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
    std::optional<std::size_t> src_order, dst_order;  ///< depth-first leaf ordinals: the source's last producer, the consumer
    std::set<std::string>*     warned = nullptr;      ///< cycle warnings already printed (kept by the model metadata)
};

/// The error of a wiring rule: the format of System::fail ("naga::cospan: <where>: <verb>: <reason>").
[[noreturn]] inline void wiring_fail(const WiringSite& site, const char* verb, const std::string& reason) {
    throw std::invalid_argument(std::string{"naga::cospan: "} + site.where + ": " + verb + ": " + reason);
}

using TieMembers = std::vector<std::pair<std::string, std::shared_ptr<Instance>>>;

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
// Merging shape definitions (design review, section 10)
// -----------------------------------------------------------------------------

/// The result of merging two variables' shape definitions.
struct MergedDefinition {
    std::vector<DimRecord>                  dims;
    std::optional<std::vector<std::size_t>> counts;  ///< a ragged variable's declared per-index counts
};

/// What merging the definitions of `a` and `b` gives, or the error of the call. Nothing is changed. The value dtype and the
/// rank must match; per dimension, UNSPECIFIED absorbs SPECIFIED, and two specified values must be equal (a ragged
/// per-index dimension is specified by its counts, which must be equal; counts and a single extent do not meet). A
/// dimension owned by a shape hook accepts only an unspecified one (it keeps the hook); hook with anything else is an error
/// (a hook and a connection or tie exclude each other). `ia` / `ib`: the instances that are being joined, for the messages
/// (else the variable's first live instance).
inline MergedDefinition merged_definition(const WiringSite& site, const char* verb, const VarMeta& a, const VarMeta& b,
                                          const Instance* ia = nullptr, const Instance* ib = nullptr) {
    if (&a == &b) return MergedDefinition{a.definition, a.def_counts};
    const std::string pa = ia != nullptr ? ia->full_path() : a.display_path();
    const std::string pb = ib != nullptr ? ib->full_path() : b.display_path();
    if (!dtype_compatible(a.dtype, b.dtype))
        wiring_fail(site, verb, "the value dtypes differ ('" + pa + "' and '" + pb + "')");
    if (a.rank != b.rank)
        wiring_fail(site, verb, "'" + pa + "' has rank " + std::to_string(a.rank) + " but '" + pb + "' has rank " +
                                    std::to_string(b.rank) + ": the rank must match");

    using How = DimRecord::How;
    MergedDefinition out;
    for (int d = 0; d < a.rank; ++d) {
        const auto        u   = static_cast<std::size_t>(d);
        const DimRecord&  x   = a.definition[u];
        const DimRecord&  y   = b.definition[u];
        const bool        xc  = u == 1 && a.def_counts.has_value();
        const bool        yc  = u == 1 && b.def_counts.has_value();
        const std::string dim = "dimension " + std::to_string(d);
        if (x.how == How::hook || y.how == How::hook) {
            const bool       xh    = x.how == How::hook;
            const DimRecord& other = xh ? y : x;
            if (other.how != How::unspecified)
                wiring_fail(site, verb, dim + " of '" + (xh ? pa : pb) + "' is set by a shape hook; it cannot be connected or tied");
            out.dims.push_back(xh ? x : y);
        } else if (x.how == How::unspecified) {
            out.dims.push_back(y);
            if (yc) out.counts = b.def_counts;
        } else if (y.how == How::unspecified) {
            out.dims.push_back(x);
            if (xc) out.counts = a.def_counts;
        } else {  // both specified
            if (xc != yc)
                wiring_fail(site, verb, dim + ": '" + (xc ? pa : pb) + "' gives per-index counts but '" + (xc ? pb : pa) +
                                            "' declares one extent (" + std::to_string(xc ? y.extent : x.extent) + ") for it");
            if (xc) {
                if (*a.def_counts != *b.def_counts)
                    wiring_fail(site, verb, dim + ": '" + pa + "' declares the counts " + extents_str(*a.def_counts) + " but '" + pb +
                                                "' declares " + extents_str(*b.def_counts) + ": the counts must match");
                out.counts = a.def_counts;
            } else if (x.extent != y.extent) {
                wiring_fail(site, verb, dim + ": '" + pa + "' declares " + std::to_string(x.extent) + " but '" + pb + "' declares " +
                                            std::to_string(y.extent) + ": the extents must match");
            }
            out.dims.push_back(x);
        }
    }
    if (out.counts && !out.dims.empty() && out.dims[0].how == How::fixed && out.dims[0].extent != out.counts->size())
        wiring_fail(site, verb, "the per-index counts have " + std::to_string(out.counts->size()) + " entries but dimension 0 is " +
                                    std::to_string(out.dims[0].extent));
    return out;
}

/// merged_definition, written into `winner` (the survivor of the call, or the side whose rules the survivor takes: see
/// merge_into). Throws before anything is changed.
inline void merge_definitions(const WiringSite& site, const char* verb, VarMeta& winner, VarMeta& loser,
                              const Instance* a = nullptr, const Instance* b = nullptr) {
    if (&winner == &loser) return;
    MergedDefinition m = merged_definition(site, verb, winner, loser, a, b);
    winner.definition  = std::move(m.dims);
    winner.def_counts  = std::move(m.counts);
}

// -----------------------------------------------------------------------------
// Initializers
// -----------------------------------------------------------------------------

// THE LAST DECLARATION WINS (internal/design-review-2026-10-03.md, section 2). A parent's setup runs after its children's,
// so a tie's, a promotion's or a System's initializer replaces the members'; tied members and promotions may carry their
// own initializers. There is no level comparison and no conflict error: the incoming declaration always replaces what is
// held. Only DECLARED initializers take part (VarMeta::init is empty for an input that declared none, which gets the
// implicit zero at initialization time): every merge below copies an initializer only when the source one is set, so an
// implicit or empty one never replaces an explicit one, in either declaration order.

/// Folds the initializer of `from` (a tie member) into `into` (the tie's alias): the member's replaces what is held, so
/// later members replace earlier ones; a member with none declared leaves what is held. Its device comes along when
/// `into` has none.
inline void fold_init(VarMeta& into, const VarMeta& from) {
    if (!from.init) return;
    into.init = from.init;
    if (!into.init_device && from.init_device) into.init_device = from.init_device;
}

// -----------------------------------------------------------------------------
// The merge walk
// -----------------------------------------------------------------------------

/// The error of a merge whose two variables both hold data (deferred until a real use appears).
inline constexpr const char* kBothMaterialized = "merging two materialized variables is not defined (yet)";

/// The generic merge, after the call's own checks (and merge_definitions, which put the merged definition in `winner`).
/// Returns the surviving object; every live instance of the other is rebound to it. See the header for who survives.
/// Never touches stale flags, events or data.
inline std::shared_ptr<VarMeta> merge_into(const WiringSite& site, std::shared_ptr<VarMeta> winner,
                                           std::shared_ptr<VarMeta> loser) {
    if (winner == loser) return winner;
    if (winner->is_materialized() && loser->is_materialized()) wiring_fail(site, "merge", kBothMaterialized);
    const std::shared_ptr<VarMeta>& survivor = winner;
    const std::shared_ptr<VarMeta>& other    = loser;
    survivor->adopt_containers(*other);  // a key the survivor has no container for takes the loser's (identity kept)
    for (const std::shared_ptr<Instance>& i : other->live_instances()) i->rebind(survivor);
    // The sticky marks of the loser stay true of the variable: a tie alias's variable that loses to a source is still a
    // tie, a tied output still pushes through its alias, a cycle is still a cycle.
    survivor->push_suppressed = survivor->push_suppressed || other->push_suppressed;
    survivor->tied_input      = survivor->tied_input || other->tied_input;
    survivor->tied_output     = survivor->tied_output || other->tied_output;
    if (other->cycle && !survivor->cycle) {
        survivor->cycle      = true;
        survivor->cycle_note = other->cycle_note;
    }
    return survivor;
}

/// Both-sides-hold-data, checked before a multi-step merge begins (merge_into checks it again, per step).
inline void check_one_live(const WiringSite& site, const char* verb, const std::vector<const VarMeta*>& metas) {
    std::vector<const VarMeta*> live;
    for (const VarMeta* m : metas) {
        if (!m->is_materialized()) continue;
        bool seen = false;
        for (const VarMeta* l : live) seen = seen || l == m;
        if (!seen) live.push_back(m);
    }
    if (live.size() > 1) wiring_fail(site, verb, kBothMaterialized);
}

// -----------------------------------------------------------------------------
// connect
// -----------------------------------------------------------------------------

/// The consumer reads a producer that runs after it (both are children of the calling System).
inline bool is_cycle(const WiringSite& site) {
    return site.src_order && site.dst_order && *site.src_order > *site.dst_order;
}

/// The merge of a connect: the source's variable wins. On a cycle the survivor is marked `cycle` and
/// keeps the TARGET's initializer (and the device it runs on), which a plain connect drops; a survivor that already holds
/// data keeps its own. Returns the survivor.
inline std::shared_ptr<VarMeta> merge_connected(const WiringSite& site, const Instance& src, const Instance& dst,
                                                std::shared_ptr<VarMeta> src_meta, std::shared_ptr<VarMeta> dst_meta) {
    const bool                         cyc      = is_cycle(site);
    const std::optional<InitSpec>      dst_init = dst_meta->init;
    const std::optional<sycl::device>  dst_dev  = dst_meta->init_device;
    const std::shared_ptr<VarMeta>     survivor = merge_into(site, std::move(src_meta), std::move(dst_meta));
    if (cyc) {
        survivor->cycle      = true;
        survivor->cycle_note = "'" + dst.full_path() + "' reads '" + src.full_path() + "', which runs after it";
        if (dst_init && !survivor->is_materialized()) {
            survivor->init = dst_init;
            if (dst_dev) survivor->init_device = dst_dev;
        }
    }
    return survivor;
}

/// The consumer reads a producer that runs after it: one line on std::cerr per (consumer, producer) pair, never an
/// error. Without a solver the consumer sees the producer's previous value (its initial state on the first forward).
inline void warn_cycle(const WiringSite& site, const Instance& src, const Instance& dst) {
    if (!is_cycle(site)) return;
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
    if (!dst->can_be_target && dst->hooked)
        wiring_fail(site, what.c_str(),
                    "'" + dst->full_path() + "' is the target of a shape hook; a shape hook and a connection exclude each other");
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
    check_one_live(site, what.c_str(), {src->meta.get(), dst->meta.get()});
    // the merged shape definition goes into the source's variable (it throws before anything is changed)
    merge_definitions(site, what.c_str(), *src->meta, *dst->meta, src.get(), dst.get());

    merge_connected(site, *src, *dst, src->meta, dst->meta);
    warn_cycle(site, *src, *dst);
}

// -----------------------------------------------------------------------------
// promote
// -----------------------------------------------------------------------------

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
    std::shared_ptr<Instance> a = make_alias_instance(DT, Rank, std::move(name), declarer->path(), declarer, member->kind, Role::promotion,
                                                      member->dims, member->can_be_source,
                                                      member->can_be_target, member->meta);
    a->device = member->device;
    a->refresh_container();
    return a;
}

// -----------------------------------------------------------------------------
// Ties
// -----------------------------------------------------------------------------

/// A tie alias's dims from its ShapeSpec (cs::shape): per-dimension specified or unspecified, like a port. A ragged tie
/// (any member is ragged, or the spec holds counts / offsets) is rank 2 and takes cs::shape(), cs::shape(n),
/// cs::shape(cs::counts(v)) or cs::shape(cs::offsets(v)); `counts` is set for the last two. Promotions take no shape.
template <int Rank>
std::vector<DimRecord> tie_dims(const WiringSite& site, const char* verb, const ShapeSpec& spec, bool ragged,
                                std::optional<std::vector<std::size_t>>& counts) {
    using How = DimRecord::How;
    std::vector<DimRecord> out(static_cast<std::size_t>(Rank));
    for (DimRecord& r : out) r.how = How::unspecified;
    if (spec.ragged_counts) {
        if (Rank != 2) wiring_fail(site, verb, "cs::counts / cs::offsets declare a ragged shape, which is rank 2 (the tie is rank " + std::to_string(Rank) + ")");
        out[0]        = detail::make_dim_record(spec.ragged_counts->size());
        out[1].how    = How::fixed;
        out[1].extent = 0;
        counts        = spec.ragged_counts;
        return out;
    }
    if (spec.dims.empty()) return out;  // cs::shape(): every dimension unspecified
    if (ragged) {
        if (Rank != 2) wiring_fail(site, verb, "a ragged tie is rank 2 (the tie is rank " + std::to_string(Rank) + ")");
        if (spec.dims.size() == 1) {
            out[0] = spec.dims[0];
            return out;
        }
        if (spec.dims.size() == 2 && spec.dims[0].how == How::unspecified && spec.dims[1].how == How::unspecified) return out;
        wiring_fail(site, verb, "a ragged tie takes cs::shape(), cs::shape(n), cs::shape(cs::counts(v)) or cs::shape(cs::offsets(v))");
    }
    if (spec.dims.size() != static_cast<std::size_t>(Rank))
        wiring_fail(site, verb, "cs::shape gives " + std::to_string(spec.dims.size()) + " dimensions but the tie is rank " + std::to_string(Rank));
    return spec.dims;
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

/// A fresh variable and alias instance for a tie. The variable is ragged when the spec holds counts or any member's
/// variable is ragged (a ragged member's data must be of the ragged kind). Its definition is the spec's.
template <DType DT, int Rank>
std::shared_ptr<Instance> build_tie_alias(const WiringSite& site, const char* verb, const TieMembers& ms, std::string name,
                                          const Subsystem* declarer, const ShapeSpec& spec, IOKind kind,
                                          std::optional<sycl::device> device) {
    bool ragged = spec.ragged_counts.has_value();
    for (const auto& m : ms) ragged = ragged || m.second->meta->is_ragged();
    std::optional<std::vector<std::size_t>> counts;
    std::vector<DimRecord>                  dims = tie_dims<Rank>(site, verb, spec, ragged, counts);
    std::shared_ptr<VarMeta> meta = make_var_meta(DT, Rank, dims, ragged);
    meta->def_counts              = counts;
    meta->init_device             = device.value_or(default_device());  // the tie's device: the one passed, else the default
    const bool input              = kind == IOKind::input;
    std::shared_ptr<Instance> alias = make_alias_instance(DT, Rank, std::move(name), declarer->path(), declarer, kind,
                                                          Role::tie, std::move(dims), /*can_source=*/!input,
                                                          /*can_target=*/input, std::move(meta));
    alias->def_counts = std::move(counts);
    if (device) {
        alias->device = *device;
        alias->refresh_container();
    }
    return alias;
}

/// Everything a tie checks about the merge itself, before the first one: each member against the alias, the members
/// against each other, and at most one side holding data. (Initializers never conflict: the last declaration wins.)
inline void check_tie_merge(const WiringSite& site, const char* verb, const std::shared_ptr<Instance>& alias,
                            const TieMembers& ms) {
    std::vector<const VarMeta*> metas;
    for (const auto& m : ms) {
        (void)merged_definition(site, verb, *alias->meta, *m.second->meta, alias.get(), m.second.get());
        metas.push_back(m.second->meta.get());
    }
    for (std::size_t i = 0; i < ms.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            (void)merged_definition(site, verb, *ms[j].second->meta, *ms[i].second->meta, ms[j].second.get(), ms[i].second.get());
    check_one_live(site, verb, metas);
}

/// The merge of a tie, after check_tie_merge: the alias's variable wins each time (`alias->meta` is read again at
/// every step: it is the survivor of the one before).
inline void apply_tie_merge(const WiringSite& site, const char* verb, const std::shared_ptr<Instance>& alias,
                            const TieMembers& ms, bool inputs) {
    // The alias's own initializer, if it has one, is the later declaration and is kept over every member's; otherwise the
    // members' are folded in order, later members replacing earlier ones.
    const bool own_init = alias->meta->init.has_value();
    for (const auto& m : ms) {
        if (inputs && !own_init && !alias->meta->is_materialized()) fold_init(*alias->meta, *m.second->meta);
        merge_definitions(site, verb, *alias->meta, *m.second->meta, alias.get(), m.second.get());
        merge_into(site, alias->meta, m.second->meta);
    }
}

/// A NEW variable (target, not source); every member's variable merges into it and the alias's metadata wins. The
/// members are inputs of dtype DT and rank Rank; their unspecified dims come from the tie and their specified dims
/// are constraints on it. The members' initializers fold into it (the last declaration wins). `device`: where the tie's
/// initialization runs. Members become non-targets: they connect through the alias.
template <DType DT, int Rank>
std::shared_ptr<Instance> tie_inputs(const WiringSite& site, const TieMembers& members, std::string name,
                                     const Subsystem* declarer, const ShapeSpec& shape, std::optional<sycl::device> device) {
    static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
    const char* verb = "tie_inputs";
    check_tie_members(site, verb, members);
    for (const auto& [path, m] : members) {
        if (m->kind != IOKind::input)
            wiring_fail(site, verb, "'" + path + "' is not an input (tie before connecting, or connect from the alias)");
        if (const std::shared_ptr<Instance> fed = first_of_kind(*m->meta, IOKind::output))
            wiring_fail(site, verb, "'" + path + "' is not a free input: its variable is fed by '" + fed->full_path() +
                                        "' (tie before connecting, or connect from the alias)");
        if (!m->can_be_target && m->hooked)
            wiring_fail(site, verb, "'" + path + "' is the target of a shape hook; a shape hook and a tie exclude each other");
        if (!m->can_be_target) wiring_fail(site, verb, "'" + path + "' is already tied: connect through its tie alias");
        check_typed_instance<DT, Rank>(site, verb, *m);
    }

    std::shared_ptr<Instance> alias =
        build_tie_alias<DT, Rank>(site, verb, members, std::move(name), declarer, shape, IOKind::input, device);
    check_tie_merge(site, verb, alias, members);
    apply_tie_merge(site, verb, alias, members, /*inputs=*/true);
    alias->meta->tied_input = true;  // after the merge: the survivor is the alias's variable
    for (const auto& m : members) m.second->can_be_target = false;  // visible at every use site
    return alias;
}

/// A NEW variable (source, not target); atomic member dtypes unless allow_plain. The members are outputs of dtype DT
/// and rank Rank on ONE device; the alias owns the shape (an output alias defines the shape of its connections, so
/// it cannot take its shape from one). Members become non-sources: they connect from the alias.
template <DType DT, int Rank>
std::shared_ptr<Instance> tie_outputs(const WiringSite& site, const TieMembers& members, std::string name,
                                      const Subsystem* declarer, const ShapeSpec& shape, TieOutputsPolicy policy,
                                      const sycl::device& device) {
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
    // Every member writes the same memory in ways the framework cannot know (atomics, disjoint slices), so they all live
    // on the tie's device. Checked here, and again by the owning System at each forward (a member can be moved later).
    for (const auto& m : members)
        if (m.second->device != device)
            wiring_fail(site, verb, "tie_outputs needs every member on the tie's device (" +
                                        device.get_info<sycl::info::device::name>() + "): '" + m.first + "' is on " +
                                        m.second->device.get_info<sycl::info::device::name>());

    std::shared_ptr<Instance> alias =
        build_tie_alias<DT, Rank>(site, verb, members, std::move(name), declarer, shape, IOKind::output, device);
    check_tie_merge(site, verb, alias, members);
    apply_tie_merge(site, verb, alias, members, /*inputs=*/false);
    alias->meta->tied_output     = true;
    alias->meta->push_suppressed = true;  // the members only register their writes; the owning System pushes through the alias
    for (const auto& m : members) {
        m.second->can_be_source = false;
        m.second->tied_member   = true;  // its tie's System owns the shape
    }
    return alias;
}

// -----------------------------------------------------------------------------
// Initial values
// -----------------------------------------------------------------------------

/// Declares the initial value of a free input's variable: the last declaration wins, so it replaces any held (a parent's
/// setup runs after its children's). An output throws. Declaring one on a connected input is accepted and has no
/// effect (the variable's source wins), as before: nothing is stored when the variable has a producer.
inline void declare_init(const WiringSite& site, const std::shared_ptr<Instance>& inst, InitSpec spec, bool restale = false) {
    if (inst->kind != IOKind::input)
        wiring_fail(site, "init_input", "'" + inst->full_path() + "' is an output; outputs are never initialized");
    VarMeta& v = *inst->meta;
    if (first_of_kind(v, IOKind::output) && !v.cycle) return;  // a cycle variable is connected and still has an initializer
    if (spec.shape()) v.check_init_against_definition(*spec.shape());  // an initializer with a shape: checked at the call
    v.init = std::move(spec);
    // A declaration made after the variable holds data (System::init_input, legal anywhere) is stale-making, like
    // Model::update_input: the variable initializes again at the next forward.
    if (restale) v.mark_stale();
}

}  // namespace detail
}  // namespace naga::cospan
