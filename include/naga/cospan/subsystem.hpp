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
    /// Component pass of init_tree: a Component runs its init(); a System
    /// forwards to its children in order.
    virtual void run_init(const ExecCtx& owner_ctx) = 0;
    /// System pass of init_tree: child Systems first, then this System's
    /// init(). A Component's is a no-op.
    virtual void run_system_init() = 0;

    // ---- forward (per forward; r01 §6.5) -------------------------------------
    virtual void run_pre_compute(ExecCtx& owner_ctx) = 0;
    /// Components return a description and never submit it; Systems submit
    /// their subtree's work themselves and return null.
    virtual std::unique_ptr<LaunchDescription> run_compute_forward() = 0;
    virtual void run_post_compute(ExecCtx& owner_ctx) = 0;

protected:
    Subsystem() = default;

private:
    friend class System;
    friend struct detail::SubsystemAccess;

    std::string   name_;
    std::string   path_;
    const System* owner_ = nullptr;
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
