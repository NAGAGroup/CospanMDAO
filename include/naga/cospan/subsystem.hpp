#pragma once

// =============================================================================
// naga/cospan/subsystem.hpp — the owner/child contract (r01 §6.1, §7.1)
// =============================================================================
//
// Standalone: it needs the SlabKit types and nothing else from CoSpan, so
// metadata.hpp can include it for Substrates and kernel.hpp for SubmitTarget.
// It holds the memory substrates, the execution context, what a submission
// consumes, the Subsystem verbs, and LaunchDescription (which names KernelParts
// by forward declaration only).

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

namespace naga::cospan {

namespace sk = ::naga::slabkit;

class System;
struct KernelParts;

/// One per device: the device's arena (present iff it has static bytes) and
/// pool (present iff configured), held for the model's life. Every allocation
/// a subsystem makes goes through its device's substrate.
struct MemorySubstrate {
    std::optional<sk::DeviceArena> arena;
    std::optional<sk::DevicePool>  pool;
};
using Substrates = std::unordered_map<sycl::device, MemorySubstrate>;

/// What a submission consumes: the queue, the group when the submission
/// belongs to a Parallel owner's group, and the device it targets.
struct SubmitTarget {
    sk::queue*        queue = nullptr;
    sk::queue::group* group = nullptr;
    sycl::device      device;
};

/// System-owned execution state; children receive a reference only to USE it.
/// Carries no events: ordering is the stream's (r01 §7.3).
struct ExecCtx {
    sycl::device       device;
    sk::queue*         queue     = nullptr;  ///< the Model's ONE queue
    sk::queue::group*  group     = nullptr;  ///< non-null only under a Parallel owner
    MemorySubstrate*   substrate = nullptr;  ///< this device's arena (+ pool)

    SubmitTarget target() { return SubmitTarget{queue, group, device}; }
};

/// What a Component's compute turn yields. The owner routes on is_custom_op()
/// (and, for host bodies, is_host()): kernels take the outer-kernel path.
class LaunchDescription {
public:
    virtual ~LaunchDescription() = default;
    virtual bool is_custom_op() const = 0;
    virtual bool is_host() const      = 0;
    /// Ordering is the stream's (r01 §7.3). Returns the events of what it submitted: one for a
    /// kernel, custom-op or host launch, none when nothing was submitted (an empty range). The
    /// caller registers them on the storage of every port the launch touched (r01 §4.4,
    /// Subsystem::register_launch_events).
    virtual std::vector<sycl::event> submit(SubmitTarget& t) const = 0;
    /// Null for custom-op and host descriptions.
    virtual const KernelParts* kernel() const { return nullptr; }
};

namespace detail {
struct SubsystemAccess;
}

/// The ONE contract between an owner and what it owns.
class Subsystem {
public:
    virtual ~Subsystem() = default;
    Subsystem(const Subsystem&)            = delete;
    Subsystem& operator=(const Subsystem&) = delete;

    std::string_view name() const noexcept { return name_; }   ///< in the owner's namespace
    const std::string& path() const noexcept { return path_; } ///< dotted from the root
    const System*    owner() const noexcept { return owner_; }  ///< null at the root

    // ---- setup chain (Model drives; r01 §11) ---------------------------------
    virtual void run_setup() = 0;
    virtual std::unordered_map<sycl::device, std::size_t>
        get_exec_info(const sycl::device& parent_device) = 0;
    virtual void prepare_exec(sk::queue& queue, Substrates& substrates) = 0;
    virtual void compile() = 0;

    // ---- forward (per forward; r01 §6.5) -------------------------------------
    /// A Component: shapes, pre_forward, then MATERIALIZE every variable it touches that is not live; a free input
    /// takes its init at the allocation (r01 §4.5).
    virtual void run_pre_compute(ExecCtx& owner_ctx) = 0;
    /// Components return a description and never submit it; Systems submit
    /// their subtree's work themselves and return null.
    virtual std::unique_ptr<LaunchDescription> run_compute_forward() = 0;
    /// A Component: the epoch bumps and the other-placement sync. Nothing is freed here: storage lives while a
    /// port holds its bundle (r01 §4.5).
    virtual void run_post_compute(ExecCtx& owner_ctx) = 0;

    /// The owner calls this right after submitting this subsystem's launch, before run_post_compute:
    /// a Component registers the launch's events on the storage of every port it touched, at the
    /// placement the launch ran on (SlabKit usm_vector::register_events, r01 §4.4). Systems have
    /// no launch of their own: the default does nothing.
    virtual void register_launch_events(std::span<const sycl::event> events) { (void)events; }

    /// What a launch still in flight needs from this subsystem after it is deleted: the runtime reads a kernel's
    /// dynamic_function_config on a worker thread when the launch runs. Each entry is (the launch's events, the
    /// object to keep). The deleting parent hands them to the root's retire list, which holds each until its events
    /// complete (r01 §7.2). The default has nothing.
    using InflightKeep = std::pair<std::vector<sycl::event>, std::shared_ptr<void>>;
    virtual void collect_inflight(std::vector<InflightKeep>& out) const { (void)out; }

    /// The run policy the owner was asked for (-1 eternal, -2 transient, n > 0 turns). The LIVE counter belongs to
    /// the owner's slot, not to the child (r01 §7.2-7.4).
    long requested_run() const noexcept { return requested_run_; }
    /// Eternal: this subsystem and every System above it run every forward and are never deleted or rebuilt
    /// (the arena route needs it). Defined in system.hpp.
    bool runs_eternally() const noexcept;

protected:
    Subsystem() = default;

private:
    friend class System;
    friend class ChunkedSystem;
    friend struct detail::SubsystemAccess;

    std::string   name_;
    std::string   path_;
    const System* owner_ = nullptr;
    long          requested_run_ = -1;     ///< -1 eternal, -2 transient, n > 0 (as asked at registration)
    bool          rebuilt_       = false;  ///< built again by its owner (rearm / transient): its allocations take the pool
};

namespace detail {

/// INTERNAL / TEST-ONLY until System exists: places a subsystem in its owner's
/// namespace (System does this itself through friendship).
struct SubsystemAccess {
    static void place(Subsystem& s, std::string name, std::string path, const System* owner) {
        s.name_  = std::move(name);
        s.path_  = std::move(path);
        s.owner_ = owner;
    }
};

}  // namespace detail
}  // namespace naga::cospan
