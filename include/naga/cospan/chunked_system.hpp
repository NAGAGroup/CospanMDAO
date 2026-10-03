#pragma once

// =============================================================================
// naga/cospan/chunked_system.hpp — ChunkedSystem (r01 "Chunked systems")
// =============================================================================
//
// A System that runs its subtree once per CHUNK of the leading dimension, so the memory its internal
// ports need is bounded by a budget. The leading dimension of every port is never named: every port of
// every child, and every port crossing the boundary, has leading length N (the chunk length of the
// whole run).
//
//   classification (once, at the end of setup, System::resolve):
//     INTERNAL  every member of the root's aliasing set is declared inside the subtree: the root lives by
//               its ports hold it like any other variable; inside the chunk loop the counter moves and
//               deletions of its children are deferred to after the last chunk.
//     BOUNDARY  the set has members inside and outside (a producer outside, a consumer outside, a
//               promotion at this level or above). A boundary root that is itself INSIDE (an inside
//               output consumed outside, an inside free input promoted out) is RE-ROOTED: a new alias
//               root is inserted at this system's level (merged like a tie alias), so every inside
//               member becomes a non-root peer; this system allocates it, at full size, before the
//               first chunk, and applies a free input's init there. A boundary root that is already
//               outside needs nothing.
//
//   per forward:
//     1. full-range pass: the inside peers of outside boundary roots see their whole root, and every
//        inside Component resolves its shapes (host only, no pre_forward, no allocation); a shape driven
//        by a VALUE computed inside the chunked system is an error;
//     2. checks: all boundary roots and all inside entries share the leading length N;
//     3. pricing: cost[i] = sum over INTERNAL roots of sizeof(dtype) * block size at i (from the
//        declared shape rules; pre_forward is not run for pricing); a greedy scan cuts maximal
//        contiguous chunks with sum <= budget; one index over budget is an error;
//     4. the re-rooted roots are allocated at full size;
//     5. per chunk [a, b): every inside peer of every boundary root is pointed at [a, b)
//        (detail::LeadingRangeAccess), then the subtree runs once (the default loop);
//     6. the ranges are cleared and the shapes restored to the full range.
//   Children never see a or b.
//
//   TOPOLOGY CACHE AND REPLAY:
//     the first forward does the full-range pass and plans the chunks as above, and records what that pass
//     depended on: the VALUES it read (root, write epoch, and a copy of the values) and the SHAPES of every
//     outside boundary root (root, shape version), plus N. It then captures, for every inside entry and every
//     internal root, the full-range ShapeState and derives each chunk's state from it (dim 0 = b - a, per-index
//     arrays cut to [a, b), offsets rebased). A later forward whose recorded dependencies are unchanged skips the
//     pass and the planning. Inside the chunk loop the cached state is APPLIED (no rule is evaluated, no value is
//     read, nothing is validated or resized), the boundary peers get their leading range without a refresh, the
//     internal roots' device index arrays (uploaded once, full range, when the roots are held) are addressed
//     through meta_shift_ = a, and every inside Component runs in replay mode (no resolve_shapes, no ragged
//     upload, no validate_connection; it still runs pre_forward, allocation of what it owns, pending inits and
//     the re-pointing of its wrappers).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "naga/cospan/reactive_shape.hpp"
#include "naga/cospan/system.hpp"

namespace naga::cospan {

class ChunkedSystem : public System {
    std::size_t budget_;
    std::size_t last_chunks_ = 0;

    struct Boundary {
        detail::IOEntry*              root   = nullptr;  ///< the root the inside members share
        detail::IOEntry*              member = nullptr;  ///< re-rooted: the inside entry that was the root
        std::vector<detail::IOEntry*> peers;             ///< inside Component entries that are peers of root
        std::size_t                   prev_flat = 0;
    };
    std::vector<Boundary>           boundary_;
    std::vector<detail::IOEntry*>   internal_;
    std::vector<ComponentBase*>     comps_;
    std::vector<detail::IOEntry*>   outside_ties_;  ///< outside tie aliases that are boundary roots: resolved here, before the pass

    // ---- the topology cache ----
    using Chunks = std::vector<std::pair<std::size_t, std::size_t>>;
    struct EntryTopo {
        detail::IOEntry*                          e = nullptr;
        detail::IOEntry::ShapeState               full;   ///< the full-range state after the pass
        std::vector<detail::IOEntry::ShapeState>  chunk;  ///< one per chunk, cut from `full`
    };
    struct ValueDep {
        detail::IOEntry*           root = nullptr;
        std::uint64_t              epoch = 0;
        std::vector<std::uint64_t> values;  ///< the root's whole content when the pass read it
    };
    struct ShapeDep {
        const detail::IOEntry* root = nullptr;
        std::uint64_t          version = 0;
    };
    struct Topology {
        bool                   valid = false;
        std::size_t            n     = 0;
        Chunks                 chunks;
        std::vector<EntryTopo> entries;
        std::vector<ValueDep>  values;
        std::vector<ShapeDep>  shapes;
    };
    Topology                             topo_;
    bool                                 recording_ = false;
    std::vector<detail::IOEntry*>        dep_roots_;
    std::vector<const detail::IOEntry*>  shape_roots_;
    std::vector<detail::AliasEvaluator::DepRead> reads_;  ///< the reads of the recording pass with their readers (self-check)
    std::function<void(std::size_t)>     chunk_hook_, chunk_post_hook_;
    std::vector<System*>                 systems_;       ///< this system and every descendant System (chunk_replay_ is set on them)
    std::vector<std::pair<detail::IOEntry*, ComponentBase*>> internal_ties_;  ///< internal tie roots, with the first Component that uses each

public:
    /// `budget_bytes` bounds the sum, over one chunk, of the bytes of the INTERNAL ports.
    explicit ChunkedSystem(std::size_t budget_bytes) : budget_{budget_bytes} {
        if (budget_bytes == 0) throw std::invalid_argument("naga::cospan: ChunkedSystem needs a budget of at least 1 byte");
    }

    std::size_t budget() const noexcept { return budget_; }
    /// OPAQUE: from outside, only this system's own promotions are visible.
    bool is_opaque() const noexcept override { return true; }
    /// TEST HOOKS. `before_launch`: called for every chunk after its shapes are applied from the cache and before
    /// its Components run. `after_chunk`: called after the chunk's Components have run (wrappers pointed): this is
    /// where a test captures the shapes and pointers of every inside entry. Both receive the chunk index.
    void set_chunk_hooks(std::function<void(std::size_t)> before_launch, std::function<void(std::size_t)> after_chunk) {
        chunk_hook_      = std::move(before_launch);
        chunk_post_hook_ = std::move(after_chunk);
    }
    /// TEST ACCESS: every inside Component entry and every internal root (each once).
    std::vector<detail::IOEntry*> chunk_entries() const {
        std::vector<detail::IOEntry*>              out;
        std::unordered_set<const detail::IOEntry*> seen;
        for (ComponentBase* c : comps_)
            for (detail::IOEntry* e : c->order_)
                if (seen.insert(e).second) out.push_back(e);
        for (detail::IOEntry* r : internal_)
            if (seen.insert(r).second) out.push_back(r);
        return out;
    }
    /// Whether the cached topology is currently held (a test hook).
    bool topology_cached() const noexcept { return topo_.valid; }
    /// The number of chunks the last forward ran (0 before the first, and for N = 0).
    std::size_t last_chunk_count() const noexcept { return last_chunks_; }

    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        Chunks                     chunks;
        std::optional<std::size_t> known_n;
        build_missing_transients(/*deep=*/true);  // transient children below exist for this whole forward, chunks included
        if (stale_) {  // a child under this system was deleted or rebuilt: classify again, build the topology again
            stale_ = false;
            chunk_classify();
        }
        try {
            resolve_outside_ties(false);  // their shapes are known before the chunk length is
            const std::size_t n = leading_length();
            known_n             = n;
            if (topology_valid(n)) {
                // The recorded dependencies are unchanged: no full pass, no planning. The entries are put back at
                // the full range (they are, after the last forward; this makes sure).
                for (EntryTopo& t : topo_.entries) t.e->apply_shape(t.full);
                chunks       = topo_.chunks;
                last_chunks_ = chunks.size();
            } else {
                topo_.valid = false;
                recording_  = true;
                prepare_full(n, chunks);
                recording_ = false;
                build_topology(n, chunks);
            }
            for (Boundary& b : boundary_)
                if (b.member != nullptr || b.root->chunk_outside_tie_) allocate_rerooted(*b.root);
            hold_internal(chunks);
            // The internal roots' device metadata is uploaded ONCE, at the full range, before the window
            // (version-gated: unchanged metadata is not sent again). Per chunk only meta_shift_ moves.
            for (detail::IOEntry* r : internal_) r->upload_ragged(q());
            for (System* s : systems_) s->defer_finishes(true);  // the chunk loop: counter moves and deletions wait
            diag::window_begin();
            run_chunks(chunks);
            diag::window_end();
            for (EntryTopo& t : topo_.entries) t.e->apply_shape(t.full);  // the whole run again, no second pass
            for (detail::IOEntry* r : internal_) r->meta_shift_ = 0;
            for (Boundary& b : boundary_)
                for (detail::IOEntry* p : b.peers) detail::LeadingRangeAccess::clear_no_refresh(*p);
            release_held();
            clear_dim0();
            // The turns that ended, once, after the last chunk, and LAST: a deletion drops ports this system points
            // at (subtree_changed then clears them). Descendants first: a parent's deletion takes them with it.
            const std::vector<System*> flush = systems_;
            for (auto it = flush.rbegin(); it != flush.rend(); ++it) (*it)->flush_finishes();
        } catch (...) {
            topo_.valid = false;  // whatever failed, the next forward builds the topology again
            recording_  = false;
            use_cached_shapes(false);
            for (ComponentBase* c : comps_) {
                c->replay_         = false;
                c->ev_.dep_recorder_   = nullptr;
                c->ev_.shape_recorder_ = nullptr;
            }
            try {
                for (detail::IOEntry* r : internal_) r->meta_shift_ = 0;
                if (known_n) set_dim0(*known_n);
                clear_ranges();
                for (System* s : systems_) s->defer_finishes(false);  // a failed forward applies nothing deferred
                release_held();
                clear_dim0();
            } catch (...) {  // the original error is the one to report
            }
            throw;
        }
        return nullptr;
    }

protected:
    /// A child under this system was deleted or rebuilt: everything held by raw pointer goes, and the next forward
    /// classifies again and drops the topology cache. Not while the chunk loop runs (deletions are deferred past it).
    void subtree_changed() override {
        stale_      = true;
        topo_.valid = false;
        topo_.entries.clear();
        topo_.values.clear();
        topo_.shapes.clear();
        comps_.clear();
        boundary_.clear();
        internal_.clear();
        internal_ties_.clear();
        systems_.clear();
    }
    bool stale_ = false;

    // ---- classification --------------------------------------------------------------------
    void chunk_classify() override {
        boundary_.clear();
        internal_.clear();
        internal_ties_.clear();
        comps_.clear();
        if (has_nested_chunked(*this))
            throw std::invalid_argument("naga::cospan: " + describe() + ": nested chunked systems are not supported");
        collect_components(comps_);
        for (detail::IOEntry* a : outside_ties_) a->chunk_outside_tie_ = false;
        outside_ties_.clear();
        for (ComponentBase* c : comps_) c->chunk_inside_ = true;
        systems_.clear();
        collect_systems(*this);

        std::vector<detail::IOEntry*>                    roots;
        std::unordered_set<const detail::IOEntry*>       seen;
        for (ComponentBase* c : comps_)
            for (detail::IOEntry* e : c->order_)
                if (seen.insert(e->root()).second) roots.push_back(e->root());

        for (detail::IOEntry* r : roots) {
            std::size_t in = 0, out = 0;
            for (const detail::IOEntry* p : r->peers()) (inside(*p) ? in : out)++;  // the shadow itself is no participant
            if (out == 0) {
                r->chunk_internal_by_ = path();
                internal_.push_back(r);
                if (r->alias_shape()) internal_ties_.emplace_back(r, nullptr);
            } else if (const detail::IOEntry* auth = r->variable().authority; auth != nullptr && inside(*auth)) {
                // The variable's shape authority is inside: the SHADOW is the boundary root (it is neutral storage, so no
                // new root is needed) and the authority port is the member whose resolved shape it adopts. This system
                // takes its OWN port on the variable (what it materializes through, and what holds it for the loop).
                Boundary b;
                b.member = const_cast<detail::IOEntry*>(auth);
                b.root   = r;
                own_port(*r);
                boundary_.push_back(std::move(b));
            } else {
                Boundary b;
                b.root = r;
                boundary_.push_back(std::move(b));
                if (r->alias_shape() && r->local_shape()) {  // a tie declared above this system, with inside members
                    r->chunk_outside_tie_ = true;
                    outside_ties_.push_back(r);
                }
            }
        }
        for (auto& [r, first] : internal_ties_)  // the first Component (registration order) that uses the tie root
            for (ComponentBase* c : comps_) {
                if (first != nullptr) break;
                for (detail::IOEntry* e : c->order_)
                    if (e->root() == r) {
                        first = c;
                        break;
                    }
            }
        for (Boundary& b : boundary_) {
            for (detail::IOEntry* p : b.root->peers())
                if (inside(*p) && p->mirror_of_ == nullptr && dynamic_cast<const ComponentBase*>(p->declarer()) != nullptr)
                    b.peers.push_back(p);
        }
    }

private:
    /// Inside = DECLARED by this system or a descendant: a Component entry, a tie alias of a descendant, and a tie alias of
    /// this system itself (its own ties are internal and transient). This system's own PROMOTIONS (no alias_shape_) are its
    /// interface: they stay boundary ports, as does anything declared above it.
    bool inside(const detail::IOEntry& e) const {
        for (const Subsystem* s = e.declarer(); s != nullptr; s = s->owner())
            if (s == this) return e.declarer() != this || e.alias_shape_.has_value();
        return false;
    }

    static bool has_nested_chunked(const System& s) {
        for (const Subsystem* c : s.child_ptrs_)
            if (const auto* sub = dynamic_cast<const System*>(c)) {
                if (dynamic_cast<const ChunkedSystem*>(sub) != nullptr) return true;
                if (has_nested_chunked(*sub)) return true;
            }
        return false;
    }

    /// This system's own PORT on a boundary variable (spec §4.5, §7): an alias entry declared here, inert dims, joined to
    /// the variable (it keeps no shape of its own: the shadow adopts the authority's each forward). The shadow is flagged
    /// as a boundary variable materialized by this system.
    void own_port(detail::IOEntry& shadow) {
        std::vector<detail::DimRecord> dims(static_cast<std::size_t>(shadow.rank()));
        for (detail::DimRecord& d : dims) d.how = detail::DimRecord::How::from_input;
        const std::string base = "chunk_" + std::string{shadow.name()};
        std::string       name = base;
        for (int k = 1; aliases_.contains(name); ++k) name = base + "_" + std::to_string(k);
        auto a = detail::make_alias_entry(shadow.dtype(), name, shadow.kind(), dims, shadow.can_be_source(), shadow.can_be_target());
        a->declarer_ = this;
        detail::IOEntry* port = a.get();
        detail::IOEntry::merge(*shadow.variable().authority, *port);  // the existing variable survives (marks, init, source)
        shadow.chunk_rerooted_ = true;
        register_alias(std::move(a));
    }

    // ---- the topology cache -----------------------------------------------------------------
    /// The pre-loop steps up to (not including) the allocation of the re-rooted outputs: dim 0 of every
    /// inside port, the re-rooted free inputs, the full-range pass, the init of an input whose shape the pass
    /// changed, and the chunk plan.
    void prepare_full(std::size_t n, Chunks& chunks) {
        set_dim0(n);  // the framework sets dim 0 of every inside port, whatever rule declared it
        // Re-rooted free INPUT roots first: a shape-driving read may need their storage.
        for (Boundary& b : boundary_)
            if (b.member != nullptr && b.root->kind() == IOKind::input) {
                b.member->update_shape();  // dim 0 = N, the other dims as declared
                b.root->adopt_shape(*b.member);
                b.prev_flat = b.root->flat_size();
                allocate_rerooted(*b.root);
            }
        dep_roots_.clear();
        shape_roots_.clear();
        reads_.clear();
        for (ComponentBase* c : comps_) c->ev_.read_log_.clear();
        resolve_outside_ties(recording_);  // again, recording what their rules read (value / shape dependencies)
        full_pass();
        for (Boundary& b : boundary_)  // an input root whose shape the pass changed takes its init again
            if (b.member != nullptr && b.root->kind() == IOKind::input && b.root->flat_size() != b.prev_flat)
                b.root->init_pending_ = true;  // applied with the allocation step
        chunks       = plan_chunks(n);
        last_chunks_ = chunks.size();
    }

    /// Each outside tie alias that is a boundary root takes the shape its OWN rule gives (its sources are outside by
    /// construction: a path into this system is refused at setup, a value computed inside is refused here), through
    /// the alias resolution of its owner, independent of which Component owns it.
    void resolve_outside_ties(bool record) {
        for (detail::IOEntry* a : outside_ties_) {
            System& owner = tie_owner(*a);
            owner.ev_.read_log_.clear();
            owner.ev_.resolve_alias_root(*a, record ? &dep_roots_ : nullptr, record ? &shape_roots_ : nullptr);
            // The reads of an outside tie's OWN rule are on its outside sources (opacity guarantees they are not inside):
            // dependencies of the cache, but not crossings of this system's boundary. They are not self-checked.
            owner.ev_.read_log_.clear();
        }
    }

    static std::vector<std::uint64_t> snapshot_values(detail::IOEntry& root, sk::queue& q) {
        std::vector<std::uint64_t> out;
        detail::dispatch_dtype(root.dtype(), [&](auto tag) {
            using T       = typename decltype(tag)::type;
            const auto& v = detail::storage_of<T>(root).read_primary(q);
            out.reserve(v.size());
            for (const T x : v) out.push_back(static_cast<std::uint64_t>(x));
        });
        return out;
    }

    template <class V>
    static V slice_vec(const V& v, std::size_t a, std::size_t b, std::size_t n) {
        if (v.size() != n) return v;  // not one entry per index (a scalar dimension's empty array)
        return V(v.begin() + static_cast<std::ptrdiff_t>(a), v.begin() + static_cast<std::ptrdiff_t>(b));
    }

    /// The state of chunk [a, b) cut from the full-range state: dim 0 = b - a, per-index arrays cut to [a, b),
    /// offsets rebased to start at 0.
    static detail::IOEntry::ShapeState slice_state(const detail::IOEntry::ShapeState& f, std::size_t a, std::size_t b) {
        detail::IOEntry::ShapeState s = f;
        const std::size_t           n = f.extents[0];
        s.extents[0]                  = b - a;
        if (f.ragged && f.offsets.size() == n + 1) {
            s.offsets.assign(b - a + 1, 0);
            for (std::size_t i = 0; i <= b - a; ++i) s.offsets[i] = f.offsets[a + i] - f.offsets[a];
        }
        for (std::size_t d = 0; d < 3; ++d) {
            if (f.dims[d].per) s.dims[d].per_index = slice_vec(f.dims[d].per_index, a, b, n);
            s.applied_per[d] = slice_vec(f.applied_per[d], a, b, n);
        }
        return s;
    }

    /// OPACITY self-check: every dependency the full pass recorded is an INTERNAL root or a boundary root (one reached
    /// through a promotion, or an outside tie of this system). Anything else means something read across the boundary.
    void check_dependencies() const {
        for (const detail::AliasEvaluator::DepRead& r : reads_) {
            const detail::IOEntry* root = r.root->root();
            bool                   ok   = std::find(internal_.begin(), internal_.end(), root) != internal_.end();
            for (const Boundary& b : boundary_) ok = ok || b.root == root;
            if (ok) continue;
            throw std::logic_error(std::string{"naga::cospan: internal error: "} + (r.shape ? "shape" : "value") +
                                   " dependency crosses chunked system '" + path() + "' outside its promotions (root '" +
                                   root->full_path() + "', read by '" + (r.by != nullptr ? r.by->full_path() : std::string{"?"}) + "')");
        }
    }

    /// After the full pass and the plan: record what the pass depended on and capture the states.
    void build_topology(std::size_t n, const Chunks& chunks) {
        check_dependencies();
        Topology t;
        t.n      = n;
        t.chunks = chunks;
        for (detail::IOEntry* e : chunk_entries()) {
            EntryTopo et;
            et.e    = e;
            et.full = e->capture_shape();
            et.chunk.reserve(chunks.size());
            for (const auto& [a, b] : chunks) et.chunk.push_back(slice_state(et.full, a, b));
            t.entries.push_back(std::move(et));
        }
        std::unordered_set<const detail::IOEntry*> seen_v;
        for (detail::IOEntry* r : dep_roots_) {
            if (!seen_v.insert(r).second) continue;
            ValueDep vd;
            vd.root   = r;
            vd.epoch  = r->write_epoch();
            vd.values = snapshot_values(*r, q());
            t.values.push_back(std::move(vd));
        }
        std::unordered_set<const detail::IOEntry*> seen_s;
        const auto add_shape = [&](const detail::IOEntry* r) {
            if (std::find(internal_.begin(), internal_.end(), r) != internal_.end()) return;  // derived inside the pass
            if (seen_s.insert(r).second) t.shapes.push_back(ShapeDep{r, r->shape_version()});
        };
        for (const Boundary& b : boundary_)
            if (b.member == nullptr) add_shape(b.root);
        for (const detail::IOEntry* r : shape_roots_) add_shape(r);
        dep_roots_.clear();
        shape_roots_.clear();
        t.valid = true;
        topo_   = std::move(t);
    }

    /// Whether the cached topology still holds: N, the shape of every outside boundary root (and root whose
    /// shape was copied), and the VALUES the pass read. A value root written since (its epoch moved) is compared
    /// by content: only a changed value invalidates, not a producer that ran again.
    bool topology_valid(std::size_t n) {
        if (!topo_.valid || topo_.n != n) return false;
        for (const ShapeDep& s : topo_.shapes)
            if (s.root->shape_version() != s.version) return false;
        for (ValueDep& v : topo_.values) {
            if (!v.root->present()) return false;
            if (v.root->write_epoch() == v.epoch) continue;
            if (snapshot_values(*v.root, q()) != v.values) return false;
            v.epoch = v.root->write_epoch();  // same content: remember the new epoch
        }
        return true;
    }

    void run_chunks(const Chunks& chunks) {
        for (ComponentBase* c : comps_) c->replay_ = true;
        use_cached_shapes(true);  // the Systems too: their alias materialization takes the cached shapes
        try {
            for (std::size_t ci = 0; ci < chunks.size(); ++ci) {
                const auto [a, b] = chunks[ci];
                for (EntryTopo& t : topo_.entries) t.e->apply_shape(t.chunk[ci]);
                for (Boundary& bd : boundary_)
                    for (detail::IOEntry* p : bd.peers) detail::LeadingRangeAccess::set_no_refresh(*p, a, b);
                for (detail::IOEntry* r : internal_) {
                    r->meta_shift_ = a;
                    // Each chunk is a fresh computation: an internal free input takes its init again.
                    if (r->io_kind_ == IOKind::input) r->init_pending_ = true;
                }
                if (chunk_hook_) chunk_hook_(ci);
                System::run_compute_forward();
                if (chunk_post_hook_) chunk_post_hook_(ci);
            }
        } catch (...) {
            for (ComponentBase* c : comps_) c->replay_ = false;
            use_cached_shapes(false);
            throw;
        }
        for (ComponentBase* c : comps_) c->replay_ = false;
        use_cached_shapes(false);
    }

    // ---- the per-forward steps ---------------------------------------------------------------
    sk::queue& q() const { return queue_ref(); }

    /// A re-rooted root: materialize its variable on need (allocate at its current size if absent; a free input takes
    /// its init); the ragged index arrays follow.
    void allocate_rerooted(detail::IOEntry& root) {
        root.variable().materialize(q(), nullptr);
        root.upload_ragged(q());
    }

    /// Before the first chunk every INTERNAL root is present with capacity for its largest chunk (the sum
    /// of its block sizes over the chunk's indices); its capacity is RESERVED (held_, reserve_elems_), and per
    /// chunk it is resized within it (a usm_vector resize within capacity, up or down, never reallocates). An
    /// eternal root that is still present from an earlier forward is kept when it is big enough; one
    /// that is too small is freed behind one wait and allocated again.
    void hold_internal(const std::vector<std::pair<std::size_t, std::size_t>>& chunks) {
        if (chunks.empty()) return;
        for (detail::IOEntry* r : internal_) {
            std::size_t most = 0;
            for (const auto& [a, b] : chunks)
                most = std::max(most, static_cast<std::size_t>(r->block_offset_at(b) - r->block_offset_at(a)));
            if (r->present() && r->capacity_elems() < most) {
                // too small: drop it (the ports let go of their bundles; the last holder's free waits on the events
                // registered on that storage) and allocate again
                for (detail::IOEntry* p : r->variable().ports) p->bundle_.reset();
            }
            r->reserve_elems_ = most;
            r->held_          = true;
            r->variable().materialize(q(), nullptr, false, /*apply_init=*/false);  // each chunk's init is the only one
        }
    }

    /// After the last chunk: the reservation ends (the roots themselves live by their counts).
    void release_held() {
        for (detail::IOEntry* r : internal_) {
            r->held_          = false;
            r->reserve_elems_ = 0;
        }
    }

    void collect_systems(System& s) {
        systems_.push_back(&s);
        for (Subsystem* c : s.child_ptrs_)
            if (auto* sub = dynamic_cast<System*>(c)) collect_systems(*sub);
    }
    void use_cached_shapes(bool on) {
        for (System* s : systems_) s->chunk_replay_ = on;
    }

    /// The System that declared a tie alias evaluates its shape rule (it owns the root).
    static System& tie_owner(const detail::IOEntry& a) {
        auto* s = const_cast<System*>(dynamic_cast<const System*>(a.shape_holder()->declarer()));  // the tie alias PORT's declarer
        if (s == nullptr) throw std::logic_error("naga::cospan: internal error: tie alias '" + a.full_path() + "' has no declaring System");
        return *s;
    }

    /// An internal tie root takes the shape its rule gives at the full range, through its declaring System's evaluator
    /// (value and shape reads recorded for the topology cache, as a Component's are).
    void resolve_internal_tie(detail::IOEntry& r) {
        System& owner = tie_owner(r);
        owner.ev_.read_log_.clear();
        owner.ev_.resolve_alias_root(r, recording_ ? &dep_roots_ : nullptr, recording_ ? &shape_roots_ : nullptr);
        reads_.insert(reads_.end(), owner.ev_.read_log_.begin(), owner.ev_.read_log_.end());
        owner.ev_.read_log_.clear();
    }

    void clear_ranges() {
        for (Boundary& b : boundary_)
            for (detail::IOEntry* p : b.peers) detail::LeadingRangeAccess::clear(*p);
    }

    /// Host only: the inside peers of OUTSIDE boundary roots see their whole root; every inside Component
    /// resolves its shapes in registration order; a re-rooted root adopts the full shape of the entry
    /// it re-rooted as soon as that entry's Component has resolved.
    void full_pass() {
        diag::count_full_passes();
        for (Boundary& b : boundary_)
            if (b.member == nullptr)
                for (detail::IOEntry* p : b.peers) detail::LeadingRangeAccess::set(*p, 0, b.root->extents()[0]);
        for (ComponentBase* c : comps_) {
            for (const auto& [r, first] : internal_ties_)  // the internal tie roots this Component is the first user of (it was their owner)
                if (first == c) resolve_internal_tie(*r);
            c->ev_.chunk_pass_ = true;
            if (recording_) {
                c->ev_.dep_recorder_   = &dep_roots_;
                c->ev_.shape_recorder_ = &shape_roots_;
            }
            try {
                c->resolve_shapes();
            } catch (...) {
                c->ev_.chunk_pass_     = false;
                c->ev_.dep_recorder_   = nullptr;
                c->ev_.shape_recorder_ = nullptr;
                throw;
            }
            c->ev_.chunk_pass_     = false;
            c->ev_.dep_recorder_   = nullptr;
            c->ev_.shape_recorder_ = nullptr;
            reads_.insert(reads_.end(), c->ev_.read_log_.begin(), c->ev_.read_log_.end());
            c->ev_.read_log_.clear();
            // a SPECIFIED dim 0 is a constraint, checked here once against the full length, never per chunk.
            for (detail::IOEntry* e : c->order_)
                if (e->dim0_override_ && e->dims_[0].how == detail::DimRecord::How::fixed &&
                    e->dims_[0].extent != *e->dim0_override_)
                    throw std::invalid_argument("naga::cospan: shape mismatch: dimension 0 of '" + e->full_path() +
                                                "' is fixed = " + std::to_string(e->dims_[0].extent) + " but chunked system '" +
                                                path() + "' supplies the full length " + std::to_string(*e->dim0_override_) +
                                                " for it (a specified dimension 0 is checked at the full length)");
            for (Boundary& b : boundary_)
                if (b.member != nullptr && b.member->declarer() == c) b.root->adopt_shape(*b.member);
        }
    }

    /// N, the chunk length of the whole run, known before the pass (the framework then sets dim 0 of every
    /// inside port to it): the leading length of the boundary roots that exist outside, and the declared
    /// (fixed) leading length of a re-rooted free input. They must all agree.
    std::size_t leading_length() const {
        const detail::IOEntry* first = nullptr;
        std::size_t            n     = 0;
        for (const Boundary& b : boundary_) {
            std::size_t l = 0;
            if (b.member == nullptr) {
                l = b.root->extents()[0];
                // a derived scalar dimension >= 1 that no producer has resolved yet (extent 0 = unknown): a setup error
                for (int d = 1; d < b.root->rank(); ++d)
                    if (!b.root->dim_is_per(d) && b.root->extents()[static_cast<std::size_t>(d)] == 0 &&
                        b.root->dim(d).how != detail::DimRecord::How::fixed)
                        throw detail::shape_needed(path(), "dimension " + std::to_string(d), *b.root);
            } else if (b.root->kind() == IOKind::input && b.member->dim(0).how == detail::DimRecord::How::fixed) {
                l = b.member->dim(0).extent;
            } else if (b.root->kind() == IOKind::input && b.member->dim(0).how == detail::DimRecord::How::unspecified &&
                       b.root->init_spec() && b.root->init_spec()->shape() && b.root->init_spec()->shape()->rank == b.root->rank()) {
                l = b.root->init_spec()->shape()->extents[0];  // a free input shaped by its init: the init gives the length
            } else {
                continue;  // an inside output, or a value-driven input: no length of its own
            }
            if (first == nullptr) {
                first = b.root;
                n     = l;
            } else if (l != n) {
                throw std::invalid_argument("naga::cospan: " + describe() + ": boundary roots '" + std::string{first->name()} +
                                            "' (leading length " + std::to_string(n) + ") and '" + std::string{b.root->name()} +
                                            "' (leading length " + std::to_string(l) + ") disagree; the chunk length is one");
            }
        }
        if (first == nullptr)
            throw std::invalid_argument("naga::cospan: " + describe() +
                                        ": no port crossing the boundary gives the chunk length (an outside root or a promoted free input is needed)");
        return n;
    }

    /// Dim 0 of every port of every inside component is N, the length of the whole run, during the full pass. An INTERNAL tie alias root (declared by a descendant, owning no port) has no Component entry to carry the
    /// override: it gets it too.
    void set_dim0(std::size_t n) {
        for (ComponentBase* c : comps_)
            for (detail::IOEntry* e : c->order_) e->dim0_override_ = n;
        for (detail::IOEntry* r : internal_)
            if (r->alias_shape()) const_cast<detail::IOEntry*>(r->shape_holder())->dim0_override_ = n;  // the tie alias port
    }
    void clear_dim0() {
        for (ComponentBase* c : comps_)
            for (detail::IOEntry* e : c->order_) e->dim0_override_.reset();
        for (detail::IOEntry* r : internal_)
            if (r->alias_shape()) const_cast<detail::IOEntry*>(r->shape_holder())->dim0_override_.reset();
    }

    /// Pricing from the declared shape rules (resolved over the full range) and the greedy scan.
    std::vector<std::pair<std::size_t, std::size_t>> plan_chunks(std::size_t n) const {
        std::vector<std::size_t> cost(n, 0);
        for (const detail::IOEntry* r : internal_) {
            const std::size_t bytes = dtype_size(r->dtype());
            for (std::size_t i = 0; i < n; ++i)
                cost[i] += bytes * static_cast<std::size_t>(r->block_offset_at(i + 1) - r->block_offset_at(i));
        }
        std::vector<std::pair<std::size_t, std::size_t>> chunks;
        std::size_t                                      a = 0, acc = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (cost[i] > budget_)
                throw std::invalid_argument("naga::cospan: " + describe() + ": index " + std::to_string(i) + " needs " +
                                            std::to_string(cost[i]) + " bytes of internal ports, over the budget of " +
                                            std::to_string(budget_) + " bytes");
            if (acc + cost[i] > budget_) {  // i > a here: a single index never exceeds the budget
                chunks.emplace_back(a, i);
                a   = i;
                acc = 0;
            }
            acc += cost[i];
        }
        if (n > 0) chunks.emplace_back(a, n);
        return chunks;
    }
};

}  // namespace naga::cospan