#pragma once

// =============================================================================
// naga/cospan/subsystem.hpp — the contract between the model's tree and what it holds
// =============================================================================
//
// Standalone: it needs SYCL, SlabKit and the standard library, and nothing from CoSpan (the types it names are
// forward-declared). A Subsystem is a node of the model's tree: a Component or a FusedGroup (a LEAF: the node that does
// work) or a System (the node that groups leaves and declares wiring).
//
// r04 (docs/spec/revisions/r04-2026-10-11-flat-model-metadata.md). The model owns every subsystem object through its
// ModelMeta (model_meta.hpp). A subsystem is given its FULL RESOLVED NAME (the root's is "") and a pointer to the
// metadata when it is registered, BEFORE its setup runs; it holds no pointer to a parent. Names are never parsed.
//
// A leaf has ONE forward(rt): the run's tree walk calls it. A System has only setup() (run_setup here) and the
// declaration calls; it never gets a forward.

#include <memory>
#include <span>
#include <stdexcept>
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
class Instance;    // instance.hpp
struct Runtime;    // runtime.hpp
struct ModelMeta;  // model_meta.hpp
struct SubsystemAccess;
}  // namespace detail

/// The ONE contract between the model's tree and a node of it.
class Subsystem {
public:
    virtual ~Subsystem() = default;
    Subsystem(const Subsystem&)            = delete;
    Subsystem& operator=(const Subsystem&) = delete;

    /// The full resolved name ("" for the root). `name()` and `path()` are the same string: messages and launch
    /// descriptions use them.
    const std::string& name() const noexcept { return name_; }
    const std::string& path() const noexcept { return name_; }

    // ---- setup ---------------------------------------------------------------------
    /// Declares this subsystem. Called once, after it was named and registered, before add_subsystem returns. It is
    /// recursive through add_subsystem (depth-first).
    virtual void run_setup() = 0;

    /// A leaf (a component or a fused group) does work and has a forward; a System does not.
    virtual bool is_leaf() const noexcept = 0;

    // ---- the run (leaves only) -----------------------------------------------------
    /// One forward: materialize the inputs, run prepare_forward, materialize the outputs, point the wrappers, submit, record
    /// the events, push to the other placements. Called by the model's tree walk for a leaf in scope.
    virtual void forward(detail::Runtime& rt) {
        (void)rt;
        throw std::logic_error("naga::cospan: '" + name_ + "' is not a leaf; it has no forward");
    }

    /// A leaf: its launch event(s). What depends_on hands on. A System has none.
    virtual std::vector<sycl::event> events() const { return {}; }

    /// depends_on: events added to every submission of this leaf for ONE forward. A System ignores them.
    virtual void add_dependencies(std::span<const sycl::event> deps) { (void)deps; }

protected:
    Subsystem() = default;

    /// The metadata this subsystem is registered in (null before registration).
    detail::ModelMeta* meta_ = nullptr;

private:
    friend struct detail::SubsystemAccess;  // the metadata names and binds what it registers

    std::string name_;
};

}  // namespace naga::cospan
