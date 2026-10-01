#pragma once

// =============================================================================
// naga/cospan/chunked_system.hpp — ChunkedSystem (D140 section 2, r01 "Chunked systems")
// =============================================================================
//
// A System that runs its subtree once per CHUNK of the leading dimension, so the memory its internal
// ports need is bounded by a budget. The leading dimension of every port is never named: every port of
// every child, and every port crossing the boundary, has leading length N (the chunk length of the
// whole run).
//
//   classification (once, at the end of setup, System::resolve):
//     INTERNAL  every member of the root's aliasing set is declared inside the subtree: the root is
//               forced transient, so its last user inside releases it (per chunk).
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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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

public:
    /// `budget_bytes` bounds the sum, over one chunk, of the bytes of the INTERNAL ports.
    explicit ChunkedSystem(std::size_t budget_bytes) : budget_{budget_bytes} {
        if (budget_bytes == 0) throw std::invalid_argument("naga::cospan: ChunkedSystem needs a budget of at least 1 byte");
    }

    std::size_t budget() const noexcept { return budget_; }
    /// The number of chunks the last forward ran (0 before the first, and for N = 0).
    std::size_t last_chunk_count() const noexcept { return last_chunks_; }

    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        std::vector<std::pair<std::size_t, std::size_t>> chunks;
        std::optional<std::size_t>                       known_n;
        try {
            const std::size_t n = leading_length();
            known_n             = n;
            set_dim0(n);  // the framework sets dim 0 of every inside port, whatever rule declared it
            // Re-rooted free INPUT roots first: a shape-driving read may need their storage.
            for (Boundary& b : boundary_)
                if (b.member != nullptr && b.root->kind() == IOKind::input) {
                    b.member->update_shape();  // dim 0 = N, the other dims as declared
                    b.root->adopt_shape(*b.member);
                    b.prev_flat = b.root->flat_size();
                    allocate_rerooted(*b.root);
                }
            full_pass();
            for (Boundary& b : boundary_)  // an input root whose shape the pass changed takes its init again
                if (b.member != nullptr && b.root->kind() == IOKind::input && b.root->flat_size() != b.prev_flat)
                    b.root->init_pending_ = true;  // applied with the allocation step below
            chunks       = plan_chunks(n);
            last_chunks_ = chunks.size();
            for (Boundary& b : boundary_)
                if (b.member != nullptr) allocate_rerooted(*b.root);
            hold_internal(chunks);
            for (const auto& [a, b] : chunks) {
                set_dim0(b - a);
                for (Boundary& bd : boundary_)
                    for (detail::IOEntry* p : bd.peers) detail::LeadingRangeAccess::set(*p, a, b);
                // Each chunk is a fresh computation: an internal free input takes its init again.
                for (detail::IOEntry* r : internal_)
                    if (r->io_kind_ == IOKind::input) r->init_pending_ = true;
                System::run_compute_forward();
            }
            set_dim0(n);  // clearing a range re-resolves the peer's shape: at the whole run's length
            clear_ranges();
            release_held();  // one queue wait, then every held root is freed
            full_pass();     // the shapes of the whole run again, for views from the host
            clear_ranges();
            clear_dim0();
        } catch (...) {
            try {
                if (known_n) set_dim0(*known_n);
                clear_ranges();
                release_held();
                clear_dim0();
            } catch (...) {  // the original error is the one to report
            }
            throw;
        }
        return nullptr;
    }

protected:
    // ---- classification --------------------------------------------------------------------
    void chunk_classify() override {
        boundary_.clear();
        internal_.clear();
        comps_.clear();
        if (has_nested_chunked(*this))
            throw std::invalid_argument("naga::cospan: " + describe() + ": nested chunked systems are not supported");
        collect_components(comps_);

        std::vector<detail::IOEntry*>                    roots;
        std::unordered_set<const detail::IOEntry*>       seen;
        for (ComponentBase* c : comps_)
            for (detail::IOEntry* e : c->order_)
                if (seen.insert(e->root()).second) roots.push_back(e->root());

        for (detail::IOEntry* r : roots) {
            std::size_t in = 0, out = 0;
            (inside(*r) ? in : out)++;
            for (const detail::IOEntry* p : r->peers_) (inside(*p) ? in : out)++;
            if (out == 0) {
                r->chunk_internal_by_ = path();
                internal_.push_back(r);
            } else if (inside(*r)) {
                Boundary b;
                b.member = r;
                b.root   = reroot(*r);
                boundary_.push_back(std::move(b));
            } else {
                Boundary b;
                b.root = r;
                boundary_.push_back(std::move(b));
            }
        }
        for (Boundary& b : boundary_) {
            for (detail::IOEntry* p : b.root->peers_)
                if (inside(*p) && p->mirror_of_ == nullptr && dynamic_cast<const ComponentBase*>(p->declarer()) != nullptr)
                    b.peers.push_back(p);
        }
    }

private:
    /// Declared in a strict descendant (an alias declared by this system itself is at its level: outside).
    bool inside(const detail::IOEntry& e) const {
        for (const Subsystem* s = e.declarer(); s != nullptr; s = s->owner())
            if (s == this) return e.declarer() != this;
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

    /// Insert a new root at this system's level and merge the old root's set into it: every inside member
    /// becomes a non-root peer, the outside members peers of the new root. The new root's dims are inert
    /// markers (the root is dynamic, and its shape is adopted from the old root each forward).
    detail::IOEntry* reroot(detail::IOEntry& r) {
        std::vector<detail::DimRecord> dims(static_cast<std::size_t>(r.rank()));
        for (detail::DimRecord& d : dims) d.how = detail::DimRecord::How::from_input;
        const std::string base = "chunk_" + std::string{r.name()};
        std::string       name = base;
        for (int k = 1; aliases_.contains(name); ++k) name = base + "_" + std::to_string(k);
        auto a = detail::make_alias_entry(r.dtype(), name, r.kind(), dims, r.can_be_source(), r.can_be_target());
        detail::IOEntry* p = a.get();
        p->declarer_       = this;
        const auto move_mark = [&](std::optional<MarkRecord>& from, std::optional<MarkRecord>& to) {
            if (!from) return;
            to        = std::move(from);
            to->root  = p;
            from.reset();
        };
        move_mark(r.design_var_, p->design_var_);
        move_mark(r.constraint_, p->constraint_);
        move_mark(r.objective_, p->objective_);
        detail::IOEntry::merge(*p, r);
        p->chunk_rerooted_ = true;
        register_alias(std::move(a));
        return p;
    }

    // ---- the per-forward steps ---------------------------------------------------------------
    sk::queue& q() const { return queue_ref(); }

    /// A re-rooted root: allocate at its current size if absent; apply a free input's pending init.
    void allocate_rerooted(detail::IOEntry& root) {
        root.allocate_if_absent(q());
        if (root.kind() == IOKind::input && root.init_pending()) {
            detail::apply_root_init(root, q());
            root.clear_init_pending();
        }
        root.upload_ragged(q());
    }

    /// D140 B3: before the first chunk every INTERNAL root is allocated once, with capacity for its largest
    /// chunk (the sum of its block sizes over the chunk's indices), and marked HELD: a last user does not
    /// release it, an owner finds it present, and per chunk it is resized within its capacity (a usm_vector
    /// resize within capacity, up or down, never reallocates).
    void hold_internal(const std::vector<std::pair<std::size_t, std::size_t>>& chunks) {
        if (chunks.empty()) return;
        for (detail::IOEntry* r : internal_) {
            if (r->present_) {  // left over from a failed forward
                q().wait();
                r->release_storage();
            }
            std::size_t most = 0;
            for (const auto& [a, b] : chunks)
                most = std::max(most, static_cast<std::size_t>(r->block_offset_at(b) - r->block_offset_at(a)));
            r->reserve_elems_ = most;
            r->held_          = true;
            r->allocate_if_absent(q());
        }
    }

    /// After the last chunk: one queue wait, then free every held root and clear HELD.
    void release_held() {
        bool any = false;
        for (const detail::IOEntry* r : internal_) any = any || r->held_;
        if (!any) return;
        q().wait();
        for (detail::IOEntry* r : internal_)
            if (r->held_) {
                r->release_storage();
                r->held_          = false;
                r->reserve_elems_ = 0;
            }
    }

    void clear_ranges() {
        for (Boundary& b : boundary_)
            for (detail::IOEntry* p : b.peers) detail::LeadingRangeAccess::clear(*p);
    }

    /// Host only: the inside peers of OUTSIDE boundary roots see their whole root; every inside Component
    /// resolves its shapes in registration order; a re-rooted root adopts the full shape of the entry
    /// that used to be its root as soon as that entry's Component has resolved.
    void full_pass() {
        for (Boundary& b : boundary_)
            if (b.member == nullptr)
                for (detail::IOEntry* p : b.peers) detail::LeadingRangeAccess::set(*p, 0, b.root->extents()[0]);
        for (ComponentBase* c : comps_) {
            c->chunk_pass_ = true;
            try {
                c->resolve_shapes();
            } catch (...) {
                c->chunk_pass_ = false;
                throw;
            }
            c->chunk_pass_ = false;
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
            } else if (b.root->kind() == IOKind::input && b.member->dim(0).how == detail::DimRecord::How::fixed) {
                l = b.member->dim(0).extent;
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

    /// Dim 0 of every port of every inside component is the current chunk length (D140 B3).
    void set_dim0(std::size_t n) {
        for (ComponentBase* c : comps_)
            for (detail::IOEntry* e : c->order_) e->dim0_override_ = n;
    }
    void clear_dim0() {
        for (ComponentBase* c : comps_)
            for (detail::IOEntry* e : c->order_) e->dim0_override_.reset();
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
