#pragma once

// =============================================================================
// naga/cospan/variable.hpp — the shared Variable behind a set of ports
//                            (r01 §4.4, §4.5; D143)
// =============================================================================
//
// INCLUDED ONLY FROM metadata.hpp, at the point where every type the Variable
// holds (Placement, MarkRecord, InitSpec) is defined and IOEntry is not yet.
// metadata.hpp closes its namespaces around the include.
//
// A detail::Variable is what a set of connected entries share. It is owned by
// shared_ptr; every declared entry (a PORT) holds one. It carries the state
// that used to live on the set's root entry:
//   - the shape authority, the ragged host description, write_epoch,
//     shape_version and the host cache (through the SHADOW below);
//   - source / marks / init declarations and the resolved init;
//   - the chunk flags and the substrates;
//   - the per-Placement storage, as weak_ptr to a Bundle<T>. A port keeps its
//     placement's bundle alive with a shared_ptr; the last one dropping frees it.
//
// The SHADOW is a typed StorageEntry with no declarer and no wrapper, made by
// the first port's factory. It is internal plumbing: root() resolves to it, so
// shape resolution, validation and typed storage work unchanged. User-facing
// messages and public accessors name ports by their paths, never the shadow.
//
// Anything that holds a port across forwards holds the Variable (shared_ptr) or
// a path, never a raw IOEntry*: a deletion can take the port away.

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace naga::cospan {
namespace detail {

class IOEntry;

/// One placement's storage of a Variable: the data and, for a ragged shape, this
/// placement's offset array and per-index arrays (dims 1 and 2).
template <class T>
struct Bundle {
    Bundle(Placement p, sk::usm_vector<T> v) : pl{std::move(p)}, data{std::move(v)} {}
    Placement         pl;
    sk::usm_vector<T> data;
    std::optional<sk::usm_vector<std::uint64_t>>                offs;
    std::array<std::optional<sk::usm_vector<std::uint64_t>>, 2> per;
    std::uint64_t                                               version = 0;  ///< the ragged version the arrays were built for
};

/// An initial value declared on one port, with the depth of the declaring
/// subsystem (root System = 0). The smallest depth wins at resolve.
struct InitDecl {
    InitSpec    spec;
    std::size_t depth = 0;
    std::string by;  ///< the declaring subsystem's path, for errors
};

/// What a non-port user of a variable holds: the bundle at the variable's primary placement.
struct UserHold {
    std::shared_ptr<void> bundle;
};

class Variable : public std::enable_shared_from_this<Variable> {
public:
    Variable() = default;
    ~Variable();  // out of line (metadata.hpp): the shadow's type is incomplete here
    Variable(const Variable&)            = delete;
    Variable& operator=(const Variable&) = delete;

    // ---- ports ----------------------------------------------------------------
    /// Live ports, raw: a port registers on construction and on joining, and
    /// unregisters in its destructor.
    std::vector<IOEntry*> ports;
    void register_port(IOEntry* p) { ports.push_back(p); }
    void unregister_port(IOEntry* p);  // defined in metadata.hpp (it names the port)
    /// The port the shadow's shape is adopted from: the entry that was the set's root before the split
    /// (the source output, the tie alias, or the free input). Survives as the last adopted shape when it goes.
    IOEntry*    authority = nullptr;
    /// Set when this variable was folded into another (absorb): whoever still holds it (an AliasSrc) follows it.
    std::shared_ptr<Variable> merged_into;
    std::string authority_path;  ///< names it after it is gone, so a rebuilt port with that path takes over

    // ---- the shadow -----------------------------------------------------------
    std::unique_ptr<IOEntry> shadow;

    // ---- structure --------------------------------------------------------------
    /// The producer port, from the parent's connect declaration. STRUCTURAL: it
    /// stays after the producer's child is deleted; `source_path` names it then.
    IOEntry*    source_port = nullptr;
    std::string source_path;

    // ---- root-only metadata (moved off the root entry) ---------------------------
    std::optional<MarkRecord> design_var, constraint, objective;  ///< metadata; keep nothing alive
    std::vector<InitDecl>     init_decls;                         ///< declared on any port
    std::optional<InitSpec>   init_spec;                          ///< resolved, or Model::init_input
    bool                      model_init   = false;               ///< init_spec came from Model::init_input
    bool                      init_pending = false;               ///< apply at the next run
    bool                      chunk_outside_tie = false;
    /// USERS that are not ports: a System whose tie alias shape rule reads this variable. Like a port, a user holds
    /// the bundle (at the primary placement, where the host read goes) for as long as it exists. The System owns its
    /// UserHold; the variable only knows it weakly.
    std::vector<std::weak_ptr<UserHold>> users;
    bool                      chunk_rerooted    = false;
    std::string               chunk_internal_by;
    Substrates*               substrates = nullptr;

    // ---- storage ----------------------------------------------------------------
    /// Per placement, a weak reference to its Bundle<T> (type-erased here; the
    /// shadow's StorageEntry<T> knows T). Expired = not materialized there.
    struct Live {
        Placement             pl;
        std::weak_ptr<void>   bundle;
    };
    std::vector<Live> live;
    /// The registered placements of the ports (primary first).
    std::vector<Placement> placements;

    std::shared_ptr<void> live_bundle(const Placement& p) const {
        for (const Live& l : live)
            if (l.pl == p) return l.bundle.lock();
        return {};
    }
    void set_live(const Placement& p, const std::shared_ptr<void>& b) {
        for (Live& l : live)
            if (l.pl == p) { l.bundle = b; return; }
        live.push_back(Live{p, b});
    }
    bool any_live() const {
        for (const Live& l : live)
            if (!l.bundle.expired()) return true;
        return false;
    }

    // ---- operations (defined in metadata.hpp, once IOEntry is complete) -----------
    /// Fold `other` into this variable: move its ports onto this one (re-pointing
    /// them), fold flags, marks, init declarations and source with today's
    /// contradiction checks. `other` is left empty.
    void absorb(Variable& other);

    /// Materialize on need (r01 §4.5). Defined in init.hpp (it applies inits). `full_range`: the variable holds the
    /// whole run, not one chunk of a ChunkedSystem's replay.
    void materialize(sk::queue& q, Substrates* subs = nullptr, bool full_range = true, bool apply_init = true);
};

}  // namespace detail
}  // namespace naga::cospan
