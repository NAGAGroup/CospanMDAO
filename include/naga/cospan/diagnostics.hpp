#pragma once
// Event counters (diagnostics): how often CoSpan itself waits on a queue, allocates, frees, reads back from the
// device, submits host tasks, uploads ragged index arrays, runs a ChunkedSystem full pass or resolves shapes.
// They only MEASURE: nothing in CoSpan reads them to decide anything. One process-wide object of relaxed atomics,
// always compiled in. ChunkedSystem brackets its chunk loop with window_begin() / window_end().

#include <atomic>
#include <cstdint>

namespace naga::cospan::diag {

struct EventCounts {
    std::uint64_t queue_waits    = 0;  ///< .wait() / .wait_and_throw() CoSpan calls on a queue
    std::uint64_t frees          = 0;  ///< storage slots dropped, metadata vectors destroyed by upload_ragged
    std::uint64_t allocations    = 0;  ///< storage slots allocated, growth past capacity, metadata vectors created
    std::uint64_t host_reads     = 0;  ///< read_primary calls (device-to-host reads CoSpan triggers)
    std::uint64_t host_tasks     = 0;  ///< from_host / write_primary / async_host(_in_order) CoSpan issues
    std::uint64_t ragged_uploads = 0;  ///< upload_ragged slot uploads that passed the version check
    std::uint64_t full_passes    = 0;  ///< ChunkedSystem full passes
    std::uint64_t resolve_shapes = 0;  ///< Component::resolve_shapes calls
};

inline EventCounts operator-(const EventCounts& a, const EventCounts& b) {
    return EventCounts{a.queue_waits - b.queue_waits,       a.frees - b.frees,
                       a.allocations - b.allocations,       a.host_reads - b.host_reads,
                       a.host_tasks - b.host_tasks,         a.ragged_uploads - b.ragged_uploads,
                       a.full_passes - b.full_passes,       a.resolve_shapes - b.resolve_shapes};
}

namespace detail {

struct Counters {
    std::atomic<std::uint64_t> queue_waits{0}, frees{0}, allocations{0}, host_reads{0}, host_tasks{0}, ragged_uploads{0},
        full_passes{0}, resolve_shapes{0};
    EventCounts win_begin{}, win_end{};  ///< the chunk loop's window (written by the one thread that runs it)
};

inline Counters& counters() {
    static Counters c;
    return c;
}

inline void bump(std::atomic<std::uint64_t>& a, std::uint64_t n) { a.fetch_add(n, std::memory_order_relaxed); }

}  // namespace detail

inline void count_queue_waits(std::uint64_t n = 1) { detail::bump(detail::counters().queue_waits, n); }
inline void count_frees(std::uint64_t n = 1) { detail::bump(detail::counters().frees, n); }
inline void count_allocations(std::uint64_t n = 1) { detail::bump(detail::counters().allocations, n); }
inline void count_host_reads(std::uint64_t n = 1) { detail::bump(detail::counters().host_reads, n); }
inline void count_host_tasks(std::uint64_t n = 1) { detail::bump(detail::counters().host_tasks, n); }
inline void count_ragged_uploads(std::uint64_t n = 1) { detail::bump(detail::counters().ragged_uploads, n); }
inline void count_full_passes(std::uint64_t n = 1) { detail::bump(detail::counters().full_passes, n); }
inline void count_resolve_shapes(std::uint64_t n = 1) { detail::bump(detail::counters().resolve_shapes, n); }

/// The counters now.
inline EventCounts snapshot() {
    const auto& c = detail::counters();
    const auto  r = std::memory_order_relaxed;
    return EventCounts{c.queue_waits.load(r), c.frees.load(r),          c.allocations.load(r), c.host_reads.load(r),
                       c.host_tasks.load(r),  c.ragged_uploads.load(r), c.full_passes.load(r), c.resolve_shapes.load(r)};
}

/// Zero every counter (and the window).
inline void reset() {
    auto& c       = detail::counters();
    const auto r  = std::memory_order_relaxed;
    c.queue_waits.store(0, r);
    c.frees.store(0, r);
    c.allocations.store(0, r);
    c.host_reads.store(0, r);
    c.host_tasks.store(0, r);
    c.ragged_uploads.store(0, r);
    c.full_passes.store(0, r);
    c.resolve_shapes.store(0, r);
    c.win_begin = EventCounts{};
    c.win_end   = EventCounts{};
}

/// ChunkedSystem: the snapshot at the start / end of the chunk loop; last_window() is end minus begin.
inline void window_begin() { detail::counters().win_begin = snapshot(); }
inline void window_end() { detail::counters().win_end = snapshot(); }
inline EventCounts last_window() { return detail::counters().win_end - detail::counters().win_begin; }

}  // namespace naga::cospan::diag
