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
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

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
    /// Ordering is the stream's (r01 §7.3).
    virtual void submit(SubmitTarget& t) const = 0;
    /// Null for custom-op and host descriptions.
    virtual const KernelParts* kernel() const { return nullptr; }
};

namespace detail {
struct SubsystemAccess;
struct DeferSink;  // lifetime.hpp: a ChunkedSystem's deferred decrements
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
    /// A Component: shapes, pre_forward, then ALLOCATE every root it touches that has no
    /// storage and apply its init when it is a free input root (r01 §4.4).
    virtual void run_pre_compute(ExecCtx& owner_ctx) = 0;
    /// Components return a description and never submit it; Systems submit
    /// their subtree's work themselves and return null.
    virtual std::unique_ptr<LaunchDescription> run_compute_forward() = 0;
    /// A Component: the other-placement sync, then RELEASE the transient roots it is the last
    /// user of (one stream wait for the batch; r01 §4.4).
    virtual void run_post_compute(ExecCtx& owner_ctx) = 0;

    /// Given by the OWNER at registration (property::transient): this subsystem, a Component or
    /// a System, and everything under it are transient registrations (r01 §4.4).
    bool transient() const noexcept { return transient_; }

    /// Run policy (given by the owner at registration): the LIVE counter (-1 eternal, -2 transient,
    /// n > 0 runs left, 0 skipped) and the value the owner asked for.
    long run_counter() const noexcept { return run_counter_; }
    long requested_run() const noexcept { return requested_run_; }
    long setup_run() const noexcept { return setup_run_; }  ///< the effective value derived at setup
    /// Inside a ChunkedSystem: where decrements are deferred to (null elsewhere).
    detail::DeferSink* defer_sink() const noexcept { return defer_sink_; }

    /// Resets run_counter_ to its setup value for this subsystem and every descendant.
    void rearm();

protected:
    Subsystem() = default;

private:
    friend class System;
    friend class ChunkedSystem;
    friend struct detail::SubsystemAccess;

    std::string   name_;
    std::string   path_;
    const System* owner_ = nullptr;
    bool          transient_ = false;
    long          requested_run_ = -1;     ///< -1 eternal, -2 transient, n > 0 (as asked at registration)
    bool          run_explicit_  = false;  ///< run_n / do_once was given
    long          setup_run_     = -1;     ///< the effective value, derived at setup
    long          run_counter_   = -1;     ///< the live value
    detail::DeferSink* defer_sink_ = nullptr;  ///< inside a ChunkedSystem: where decrements are deferred to
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
