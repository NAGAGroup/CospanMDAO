#pragma once

// =============================================================================
// naga/cospan/model_config.hpp — configuring a Model by descriptors
// =============================================================================
//
//   cs::create_model<Sys>(args...)                              every device, no pool, no async handler
//   cs::create_model<Sys>(cs::model_config_desc(descs...), args...)
//
// model_config_desc takes its descriptors in any order:
//   cs::pool_desc(target, bytes[, cs::node_size_desc(n)])       zero or more
//   cs::async_handler_desc(handler)                              at most one
// A pool_desc target is a sycl::device, a std::vector<sycl::device> or cs::all_devices. A device that no pool_desc names
// gets a pool of 0 bytes (every allocation on it takes the OS route and is counted) with the default 1024-byte node.
//
// The Model's context is sk::context{}: EVERY available root device. There is no device gating: any device may be given to
// add_subsystem, whether or not a pool_desc names it. The descriptors resolve into detail::ResolvedConfig, which only the
// Model reads; the validation (a device covered twice, all_devices combined with another pool_desc, a device not in the
// context) happens when the Model is created and throws std::invalid_argument naming the device.

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

namespace naga::cospan {

class Model;

/// The granularity of a pool's pages, in bytes.
class node_size_desc {
public:
    explicit node_size_desc(std::size_t bytes) : bytes_{bytes} {}
    std::size_t bytes() const noexcept { return bytes_; }

private:
    std::size_t bytes_;
};

/// The pool_desc target for every device of the Model's context.
struct all_devices_t {
    explicit all_devices_t() = default;
};
inline constexpr all_devices_t all_devices{};

namespace detail {
/// One pool_desc, resolved: the devices it names (or all of them) with its size and node size.
struct PoolEntry {
    bool                      all = false;
    std::vector<sycl::device> devices;
    std::size_t               bytes     = 0;
    std::size_t               node_size = 1024;
};

/// A model_config_desc, resolved. Not public: only the Model reads it.
struct ResolvedConfig {
    std::vector<PoolEntry>             pools;
    std::optional<sycl::async_handler> async_handler;
};
}  // namespace detail

/// A pool over a target: (target, bytes[, node_size_desc]). The target is one device, a list of devices (each gets a pool of
/// `bytes`), or cs::all_devices.
class pool_desc {
public:
    pool_desc(const sycl::device& device, std::size_t bytes, node_size_desc node_size = node_size_desc{1024}) {
        e_.devices   = {device};
        e_.bytes     = bytes;
        e_.node_size = node_size.bytes();
    }
    pool_desc(std::vector<sycl::device> devices, std::size_t bytes, node_size_desc node_size = node_size_desc{1024}) {
        e_.devices   = std::move(devices);
        e_.bytes     = bytes;
        e_.node_size = node_size.bytes();
    }
    pool_desc(all_devices_t, std::size_t bytes, node_size_desc node_size = node_size_desc{1024}) {
        e_.all       = true;
        e_.bytes     = bytes;
        e_.node_size = node_size.bytes();
    }

private:
    friend class model_config_desc;
    detail::PoolEntry e_;
};

/// The handler given to both of the Model's queues; asynchronous errors reach it at wait().
class async_handler_desc {
public:
    explicit async_handler_desc(sycl::async_handler handler) : h_{std::move(handler)} {}

private:
    friend class model_config_desc;
    sycl::async_handler h_;
};

/// The Model's configuration: zero or more pool_desc and at most one async_handler_desc, in any order.
class model_config_desc {
    template <class D>
    static constexpr bool is_descriptor_v = std::is_same_v<D, pool_desc> || std::is_same_v<D, async_handler_desc>;
    template <class... Ds>
    static constexpr int count_handlers_v = (0 + ... + (std::is_same_v<Ds, async_handler_desc> ? 1 : 0));

public:
    template <class... Ds>
        requires(!(sizeof...(Ds) == 1 && (std::is_same_v<std::remove_cvref_t<Ds>, model_config_desc> && ...)))
    explicit model_config_desc(Ds... descs) {
        static_assert((is_descriptor_v<Ds> && ...),
                      "naga::cospan::model_config_desc takes only cs::pool_desc and cs::async_handler_desc");
        static_assert(count_handlers_v<Ds...> <= 1, "naga::cospan::model_config_desc takes at most one cs::async_handler_desc");
        (add(std::move(descs)), ...);
    }

private:
    friend class Model;
    void add(pool_desc d) { data_.pools.push_back(std::move(d.e_)); }
    void add(async_handler_desc d) { data_.async_handler = std::move(d.h_); }

    detail::ResolvedConfig data_;
};

}  // namespace naga::cospan
