#pragma once

// The test-only root driver shared by smoke_system and smoke_parallel. It
// performs, per verb, what the Model will (r01 §11.2, §11.3); the library has
// no Model yet. Not public API.

#include <cstddef>
#include <unordered_map>
#include <utility>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace naga::cospan::detail {

struct RootDriver {
    sycl::device dev;
    sk::queue&   q;
    System&      root;
    Substrates   subs;
    /// What get_exec_info returned for the root: bytes per device.
    std::unordered_map<sycl::device, std::size_t> info;

    RootDriver(const sycl::device& d, sk::queue& queue, System& r) : dev{d}, q{queue}, root{r} {}

    /// run_setup -> get_exec_info -> substrates (arena from bytes if > 0, pool)
    /// -> prepare_exec -> compile -> run_init -> run_system_init.
    void start() {
        SubsystemAccess::place(root, "top", "top", nullptr);
        root.run_setup();
        info              = root.get_exec_info(dev);
        std::size_t total = 0;
        for (const auto& [d, bytes] : info) total += bytes;
        MemorySubstrate m;
        // Slack for the arena's per-allocation alignment.
        if (total > 0) m.arena.emplace(q, dev, sk::kind::device, total + 4096 * 64);
        m.pool.emplace(q, dev, sk::kind::device, 1 << 20);
        subs.emplace(dev, std::move(m));
        root.prepare_exec(q, subs);
        root.compile();
        ExecCtx ctx{dev, &q, nullptr, &subs.at(dev)};
        root.run_init(ctx);
        root.run_system_init();
    }

    /// Enqueue one forward; nothing waits (r01 §11.3: a loop of forwards need not).
    void enqueue() { (void)root.run_compute_forward(); }
    void wait() { q.wait(); }
    void forward() {
        enqueue();
        wait();
    }
};

}  // namespace naga::cospan::detail
