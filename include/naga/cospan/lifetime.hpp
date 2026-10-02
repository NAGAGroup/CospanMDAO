#pragma once
// =============================================================================
// naga/cospan/lifetime.hpp: variable lifetime by counting
// =============================================================================
//
// Every ROOT carries a count: 0 = out of scope (every root starts there), -1 = eternal, n > 0 =
// decrements owed. Its PARTNERS (the Components with an entry on it and the Systems with an alias
// entry on its set; recorded by System::resolve) each answer, at revive time, how many runs they
// will still take part in: their holder's LIVE run_counter(): -1 -> -1, -2 (transient) -> 1,
// n > 0 -> n, 0 -> 0. A partner decrements once after each run it takes part in; the root is
// released when the count reaches 0. A marked root (design variable, constraint, objective) is
// always eternal.
//
// THREADING: revive, decrement, release run on the thread driving the System loop (never in a
// kernel body). A batch is released with ONE queue wait.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "naga/cospan/diagnostics.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/subsystem.hpp"

namespace naga::cospan::detail {

struct LifetimeOps {
    /// What a partner answers: its holder's live run counter, a transient one counting as 1.
    static long answer(const Subsystem& holder) {
        const long c = holder.run_counter();
        return c == -2 ? 1 : c;
    }

    static bool marked(const IOEntry& r) { return r.design_var_ || r.constraint_ || r.objective_; }

    static void revive(IOEntry& root, const Subsystem& reviver) {
        if (root.count_ != 0) return;
        if (marked(root)) {
            root.count_ = -1;
            for (IOEntry::Partner& p : root.partners_) p.owed = 0;
            return;
        }
        std::vector<long> ans;
        ans.reserve(root.partners_.size());
        bool eternal = false;
        long sum     = 0;
        for (const IOEntry::Partner& p : root.partners_) {
            const long a = p.holder != nullptr ? answer(*p.holder) : 0;
            ans.push_back(a);
            if (a == -1)
                eternal = true;
            else
                sum += a;
        }
        // NO-WRITER ERROR (before the count is set): a root that is not a free input needs a writer that will run.
        if (root.kind() != IOKind::input) {
            bool                    any   = false;
            const IOEntry::Partner* first = nullptr;
            for (std::size_t i = 0; i < root.partners_.size(); ++i)
                if (root.partners_[i].writer) {
                    if (first == nullptr) first = &root.partners_[i];
                    if (ans[i] != 0) any = true;
                }
            if (!any)
                throw std::logic_error("naga::cospan: '" + root.full_path() + "' is revived by '" + reviver.path() +
                                       "' but nothing that will run writes it (its producer '" +
                                       (first != nullptr && first->holder != nullptr ? first->holder->path() : std::string{"<none>"}) +
                                       "' has stopped); promote it, or don't mark its holder transient");
        }
        if (eternal) {
            root.count_ = -1;
            for (IOEntry::Partner& p : root.partners_) p.owed = 0;
            return;
        }
        for (std::size_t i = 0; i < root.partners_.size(); ++i) root.partners_[i].owed = ans[i];
        root.count_ = sum;
    }

    static void decrement(IOEntry& root, const Subsystem& partner, std::vector<IOEntry*>& batch) {
        if (root.count_ <= 0) return;
        for (IOEntry::Partner& p : root.partners_) {
            if (p.holder != &partner || p.owed == 0) continue;  // a holder may appear once per use of the root
            --p.owed;
            --root.count_;
            if (root.count_ == 0) {
                root.released_by_ = partner.path();
                batch.push_back(&root);
            }
            return;
        }
    }

    /// One queue wait for the whole batch, then the storage of each root goes. A batch of roots with no
    /// storage is nothing to wait for.
    static void release(std::vector<IOEntry*>& batch, sk::queue& q) {
        bool any = false;
        for (const IOEntry* r : batch) any = any || r->present_;
        if (any) {
            diag::count_queue_waits();
            q.wait();
            for (IOEntry* r : batch) r->release_storage();
        }
        batch.clear();
    }

    /// Back to out of scope with nothing owed (rearm, a failed chunked forward).
    static void reset(IOEntry& root) {
        root.count_ = 0;
        for (IOEntry::Partner& p : root.partners_) p.owed = 0;
    }
    static void set_released_by(IOEntry& root, std::string by) { root.released_by_ = std::move(by); }
};

inline void revive(IOEntry& root, const Subsystem& reviver) { LifetimeOps::revive(root, reviver); }
inline void decrement(IOEntry& root, const Subsystem& partner, std::vector<IOEntry*>& batch) {
    LifetimeOps::decrement(root, partner, batch);
}
inline void release(std::vector<IOEntry*>& batch, sk::queue& q) { LifetimeOps::release(batch, q); }

/// A ChunkedSystem's deferred lifetime bookkeeping. While `active` (the chunk loop), decrements are
/// recorded (deduplicated per root and partner) and the run counters of the children that ran are not
/// touched; the ChunkedSystem applies both once, after the last chunk.
struct DeferSink {
    bool                                                 active = false;
    std::vector<std::pair<IOEntry*, const Subsystem*>>   items;
    std::vector<Subsystem*>                              ran;

    void record(IOEntry& r, const Subsystem& p) {
        const auto it = std::find(items.begin(), items.end(), std::pair<IOEntry*, const Subsystem*>{&r, &p});
        if (it == items.end()) items.emplace_back(&r, &p);
    }
    void note_ran(Subsystem* c) {
        if (std::find(ran.begin(), ran.end(), c) == ran.end()) ran.push_back(c);
    }
    void clear() {
        active = false;
        items.clear();
        ran.clear();
    }
};

/// Decrement now, or (inside a ChunkedSystem's chunk loop) defer.
inline void decrement_or_defer(IOEntry& root, const Subsystem& partner, std::vector<IOEntry*>& batch, DeferSink* sink) {
    if (sink != nullptr && sink->active)
        sink->record(root, partner);
    else
        LifetimeOps::decrement(root, partner, batch);
}

}  // namespace naga::cospan::detail