// =============================================================================
// include/naga/cospan/core/kernel_profile.hpp — opt-in per-kernel-type profiling
// =============================================================================
//
// FRAMEWORK_CLEANUP P5: instrumentation hung on the detail::submit choke point
// (kernel.hpp). Opt-in via NAGA_KERNEL_PROF=1, read ONCE into a static.
//
//   OFF (default): detail::submit takes one predictable branch on a cached
//   bool and otherwise compiles to exactly the uninstrumented launch. No
//   string ops, no lookups, no event retention.
//
//   ON: per-kernel-TYPE launch counts + accumulated device time. Identity is
//   a static per-type slot (stat_for<K>()) — the hot path is a counter
//   increment + event append, never a string/map lookup. Device time comes
//   from SYCL event profiling (command_end - command_start), which requires
//   queues created with property::queue::enable_profiling — Context creates
//   its queue pool with that property iff profiling is enabled (see
//   Context::make_pool_queue). Completed events are drained opportunistically
//   on each record; the remainder is waited at dump time.
//
// Honesty caveats (deliberate, known):
//   - Summed device times do NOT add up to wall time: kernels on different
//     queues may overlap, and gaps (host logic, transfers) are not counted.
//   - Transfer primitives (fill/copy/upload/download) and Context-internal
//     fills are memory ops, not kernels — they do not flow through submit and
//     are not counted here.
//   - Enabling profiling changes the run being measured (profiling queues,
//     event retention). The OFF path is undisturbed.
//   - Submission happens from a single host thread (context_execution layer
//     loops); counters are deliberately plain, not atomic.
//
// Dump: fires from ~Context() (stderr) while the runtime is still alive, or
// explicitly via naga::cospan::core::dump_kernel_profile(std::ostream&). NEVER dump
// from atexit: profiling-info queries after SYCL runtime teardown segfault.

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>

namespace naga::cospan::core::prof {

inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("NAGA_KERNEL_PROF");
        return e && e[0] == '1';
    }();
    return on;
}

struct KernelStat {
    std::string             name;
    std::uint64_t           launches  = 0;
    std::uint64_t           device_ns = 0;
    std::deque<sycl::event> pending;

    void drain(bool wait_all) {
        while (!pending.empty()) {
            auto& ev = pending.front();
            if (!wait_all &&
                ev.get_info<sycl::info::event::command_execution_status>() !=
                    sycl::info::event_command_status::complete)
                break;
            try {
                if (wait_all) ev.wait();  // profiling info needs completion
                const auto t0 = ev.get_profiling_info<
                    sycl::info::event_profiling::command_start>();
                const auto t1 = ev.get_profiling_info<
                    sycl::info::event_profiling::command_end>();
                device_ns += (t1 >= t0) ? (t1 - t0) : 0;
            } catch (const sycl::exception&) {
                // Profiling info unavailable on this backend/event: degrade to
                // launch counts only.
            }
            pending.pop_front();
        }
    }

    void record(sycl::event ev) {
        ++launches;
        pending.push_back(std::move(ev));
        drain(/*wait_all=*/false);
    }
};

namespace detail {
inline std::vector<KernelStat*>& registry() {
    static std::vector<KernelStat*> r;
    return r;
}

template <class K>
std::string_view type_name() {
#if defined(__clang__) || defined(__GNUC__)
    std::string_view p = __PRETTY_FUNCTION__;
    const auto b = p.find("K = ") + 4;
    const auto e = p.find_first_of(";]", b);
    return p.substr(b, e - b);
#else
    return typeid(K).name();
#endif
}
} // namespace detail

inline void dump_kernel_profile(std::ostream& os);

namespace detail {
inline KernelStat& make_stat(std::string_view name) {
    static std::deque<KernelStat> storage;  // stable addresses
    storage.emplace_back();
    storage.back().name = std::string(name);
    registry().push_back(&storage.back());
    return storage.back();
}
} // namespace detail

template <class K>
inline KernelStat& stat_for() {
    static KernelStat& s = detail::make_stat(detail::type_name<K>());
    return s;
}

template <class K>
inline void record(sycl::event ev) {
    stat_for<K>().record(std::move(ev));
}

inline void dump_kernel_profile(std::ostream& os) {
    auto reg = detail::registry();  // copy, then sort by device time desc
    for (auto* s : reg) s->drain(/*wait_all=*/true);
    std::sort(reg.begin(), reg.end(), [](const KernelStat* a, const KernelStat* b) {
        return a->device_ns > b->device_ns;
    });
    std::uint64_t tot_ns = 0, tot_n = 0;
    for (auto* s : reg) { tot_ns += s->device_ns; tot_n += s->launches; }
    os << "\n[KERNEL_PROF] per-kernel-type totals (device time; see header caveats)\n";
    for (auto* s : reg) {
        os << "[KERNEL_PROF] " << std::setw(12) << s->launches << "  "
           << std::setw(12) << std::fixed << std::setprecision(3)
           << double(s->device_ns) * 1e-6 << " ms  " << s->name << "\n";
    }
    os << "[KERNEL_PROF] TOTAL: " << tot_n << " launches, "
       << double(tot_ns) * 1e-6 << " ms device time\n";
}

// Dump then zero all stats ("profile since last dump"). Called from ~Context()
// while the SYCL runtime and queues are still alive — profiling-info queries
// after runtime teardown are use-after-free (learned the hard way: the original
// atexit dump SEGV'd inside get_profiling_info at __run_exit_handlers).
inline void dump_and_reset(std::ostream& os) {
    dump_kernel_profile(os);
    for (auto* s : detail::registry()) {
        s->launches  = 0;
        s->device_ns = 0;
        s->pending.clear();
    }
}

} // namespace naga::cospan::core::prof

namespace naga::cospan::core {
using prof::dump_kernel_profile;
} // namespace naga::cospan::core
