#pragma once

// =============================================================================
// naga/cospan/runtime.hpp — the run-time argument of materialize and initializers
// =============================================================================
//
// detail::Runtime is what the framework hands to the code that allocates and
// initializes a variable's data: the queues to submit to and the per-device pools
// to allocate from. It is NOT a setup context: nothing is passed down at setup
// (docs/design-review-2026-10-03.md, section 2), and an instance or a variable
// never stores one.
//
// The Model owns the Runtime and passes it down a forward. `forward` counts the Model's runs: the event lists on a variable's
// shared metadata roll over when it changes (variable.hpp, EventLists), so it is
// the one number that separates "this forward's writes" from the last one's.

#include <cstdint>
#include <unordered_map>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

namespace naga::cospan {

namespace sk = ::naga::slabkit;

namespace detail {

/// AdaptiveCpp's default device. Every instance takes it when it is declared; an outer
/// set_device overrides it (docs/design-review-2026-10-03.md, section 2).
inline sycl::device default_device() { return sycl::device{sycl::default_selector_v}; }

struct Runtime {
    sk::queue&                                  q;   ///< containers (sk::queue) and their memory work
    sk::unordered_queue&                        uq;  ///< kernels and host tasks, ordered by events only
    std::unordered_map<sycl::device, sk::DevicePool> pools;  ///< one pool per device, sized by the user (section 2)
    std::uint64_t                               forward = 0;  ///< the Model's run counter; drives event rollover

    /// The allocator for `dev` and `k`. Device-kind memory comes from the device's pool when it has one (SlabKit
    /// falls back to the OS itself when the pool is full or too fragmented); everything else is a direct
    /// allocation. A host-kind allocation names no device.
    template <class T>
    sk::allocator<T> allocator_for(const sycl::device& dev, sk::kind k) const {
        if (k == sk::kind::device) {
            const auto it = pools.find(dev);
            if (it != pools.end()) return sk::allocator<T>{q, it->second};
        }
        if (k == sk::kind::host) return sk::allocator<T>{q, k};
        return sk::allocator<T>{q, k, dev};
    }
};

}  // namespace detail
}  // namespace naga::cospan
