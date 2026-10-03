#pragma once

// =============================================================================
// naga/cospan/subsystem.hpp — the owner/child contract
// =============================================================================
//
// Standalone: it needs SYCL, SlabKit and the standard library, and nothing from CoSpan (the types it names are
// forward-declared). A Subsystem is what a System owns: a Component (the leaf that does work) or a System (the
// node that orders it).
//
// THE OWNER/CHILD CONTRACT. The OWNER (a System) builds a child, gives it its name, its dotted path and
// itself as owner (the private fields, set through friendship), and only then asks it to declare itself. The
// child never reaches back: it holds no pointer to its owner's data, and the owner holds the child alone (by
// unique_ptr, in a slot).
//
// TWO SETUP PASSES (docs/design-review-2026-10-03.md, section 2):
//   1. setup: run_setup() declares the child's inputs, outputs, aliases and children. It is recursive through
//      add_subsystem: each add_subsystem builds the child and runs its setup before it returns.
//   2. finalize: the Model collects the marks. A Subsystem has no part in it.
// NOTHING IS PASSED DOWN AT SETUP: no context, no queue, no device. A device reaches a subtree only through
// set_device, which the factory (System::add_subsystem) calls after the child's setup (section 2), and the outer
// call wins.
//
// THE FORWARD. The three steps and the event getter below belong to section 3 of the record. They are implemented
// by components (phase 5) and by Systems (phase 6); they are part of the contract now so that the owner's
// structure code (deleting a child waits on events()) is written against the final shape.

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

namespace naga::cospan {

namespace sk = ::naga::slabkit;

class System;
class Model;

namespace detail {
class Instance;  // instance.hpp
struct Runtime;  // runtime.hpp
}  // namespace detail

/// The ONE contract between an owner and what it owns.
class Subsystem {
public:
    virtual ~Subsystem() = default;
    Subsystem(const Subsystem&)            = delete;
    Subsystem& operator=(const Subsystem&) = delete;

    std::string_view   name() const noexcept { return name_; }   ///< in the owner's namespace
    const std::string& path() const noexcept { return path_; }   ///< dotted from the root
    const System*      owner() const noexcept { return owner_; } ///< null at the root

    // ---- setup (section 2) ---------------------------------------------------------
    /// Declares this subsystem. Called once by its owner, after it was named and placed, before add_subsystem returns.
    virtual void run_setup() = 0;

    /// Places this subsystem on `d`: a component sets the device of each of its instances; a System sets the
    /// device of each of its own aliases and calls this on each live child (section 2). The OUTER call always wins.
    virtual void set_device(const sycl::device& d) = 0;

    /// This subsystem's OWN instance by local name (a component's input or output, a System's alias), else null.
    virtual std::shared_ptr<detail::Instance> find_instance(std::string_view name) const = 0;

    /// Appends this subsystem's own instances (not its children's).
    virtual void collect_instances(std::vector<std::shared_ptr<detail::Instance>>& out) const = 0;

    // ---- forward (section 3): implemented by components (phase 5) and Systems (phase 6) ----------
    /// A component: resolve derived shapes, materialize inputs, then outputs. A System: rebuild transient children.
    virtual void run_precompute(detail::Runtime& rt) = 0;
    /// A component: submit its work. A System: call each child's forward in declaration order.
    virtual void run_compute(detail::Runtime& rt) = 0;
    /// A component: its postcompute work. A System: the same for its children.
    virtual void run_postcompute(detail::Runtime& rt) = 0;

    /// A component: its launch event(s). A System: its children's events plus its own push events. The same getter
    /// serves deletion (the parent waits on a child's events only) and depends_on.
    virtual std::vector<sycl::event> events() const = 0;

    /// depends_on: events added to every submission of this subsystem for ONE forward.
    virtual void add_dependencies(std::span<const sycl::event> deps) = 0;

protected:
    Subsystem() = default;

private:
    friend class System;  // places its children
    friend class Model;   // places the root

    std::string   name_;
    std::string   path_;
    const System* owner_ = nullptr;
};

}  // namespace naga::cospan
