#pragma once
// Event counters (diagnostics): how often CoSpan itself waits on a queue (Model::wait) and submits host tasks
// (host-body launches and a fused component's packing-buffer uploads). They only MEASURE: nothing in CoSpan reads them
// to decide anything. One process-wide object of relaxed atomics, always compiled in.
//
// Only counters with a live caller are kept (docs/design-review-2026-10-03.md: the allocation, free, host-read,
// ragged-upload, resize, full-pass and shape-resolution counts belonged to the arena, the host read cache, the
// ChunkedSystem and the setup-time shape pass, none of which exists now, and the chunk window went with the chunk
// loop). A new counter is added together with its caller.

#include <atomic>
#include <cstdint>

namespace naga::cospan::diag {

struct EventCounts {
    std::uint64_t queue_waits = 0;  ///< Model::wait calls (.wait_and_throw() on both queues)
    std::uint64_t host_tasks  = 0;  ///< host-body launches and fused-component packing-buffer uploads
};

inline EventCounts operator-(const EventCounts& a, const EventCounts& b) {
    return EventCounts{a.queue_waits - b.queue_waits, a.host_tasks - b.host_tasks};
}

namespace detail {

struct Counters {
    std::atomic<std::uint64_t> queue_waits{0}, host_tasks{0};
};

inline Counters& counters() {
    static Counters c;
    return c;
}

inline void bump(std::atomic<std::uint64_t>& a, std::uint64_t n) { a.fetch_add(n, std::memory_order_relaxed); }

}  // namespace detail

inline void count_queue_waits(std::uint64_t n = 1) { detail::bump(detail::counters().queue_waits, n); }
inline void count_host_tasks(std::uint64_t n = 1) { detail::bump(detail::counters().host_tasks, n); }

/// The counters now.
inline EventCounts snapshot() {
    const auto& c = detail::counters();
    const auto  r = std::memory_order_relaxed;
    return EventCounts{c.queue_waits.load(r), c.host_tasks.load(r)};
}

/// Zero every counter.
inline void reset() {
    auto& c      = detail::counters();
    const auto r = std::memory_order_relaxed;
    c.queue_waits.store(0, r);
    c.host_tasks.store(0, r);
}

}  // namespace naga::cospan::diag
