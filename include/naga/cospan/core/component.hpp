#pragma once

// =============================================================================
// naga/cospan/core/component.hpp — Component: computational composable
// =============================================================================
//
// Component is the base class for all computational elements in the framework.
// It inherits from Composable, which provides declare() (pure virtual), the
// name protocol (set_name/name), and I/O-metadata virtuals.
//
// Component adds:
//   - The execution lifecycle: forward/backward (pure virtual), init/reset
//     (defaulted no-ops). Only Components participate in execution; Systems
//     compose Components and are never executed directly.
//   - Declaration helpers that automatically prepend the instance name
//     (from Composable::name()) plus a "/" separator to each entry's local
//     name before delegating to Context. Subclasses call these helpers inside
//     their declare() implementation instead of reaching for ctx directly,
//     ensuring name scoping is handled uniformly.
//   - I/O introspection via declared_inputs_/declared_outputs_, populated by
//     the declare helpers.

#include "naga/cospan/core/composable.hpp"
#include "naga/cospan/core/context.hpp"
#include "naga/cospan/core/io.hpp"
#include "naga/cospan/core/kernel.hpp"
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace naga::cospan::core {

// =============================================================================
// Component — computational composable with auto-namespacing helpers
// =============================================================================

class Component : public Composable {
public:
    // --- Execution lifecycle (only Components participate in execution) ----------
    //
    // Systems compose Components and are never executed directly. The context
    // executor drives these methods on every registered Component in topological
    // order.

    virtual void forward(Context& ctx, sycl::queue& q) = 0;
    virtual void backward(Context& ctx, sycl::queue& q) = 0;
    virtual void init(Context&, sycl::queue&) {}
    virtual void reset(Context&, sycl::queue&) {}

    // --- Declaration helpers (called during declare()) --------------------------

    /// Declare an input entry. Prepends name() + "/" to local_name.
    /// Records the declaration for introspection before delegating to ctx.
    /// The dtyped variant records the DType in entry metadata (S1); the
    /// Dim-only form is the f32 alias.
    template <DType DT, int Dim>
    void declare_input(Context& ctx, std::string local_name,
                       const sycl::range<Dim>& range) {
        declared_inputs_.push_back({local_name, range.size()});
        ctx.declare_input<DT, Dim>(name() + "/" + local_name, range);
    }

    template <int Dim>
    void declare_input(Context& ctx, std::string local_name,
                       const sycl::range<Dim>& range) {
        declare_input<DType::f32, Dim>(ctx, std::move(local_name), range);
    }

    /// Declare an output entry. Prepends name() + "/" to local_name.
    /// Records the declaration for introspection before delegating to ctx.
    template <DType DT, int Dim>
    void declare_output(Context& ctx, std::string local_name,
                        const sycl::range<Dim>& range) {
        declared_outputs_.push_back({local_name, range.size()});
        ctx.declare_output<DT, Dim>(name() + "/" + local_name, range);
    }

    template <int Dim>
    void declare_output(Context& ctx, std::string local_name,
                        const sycl::range<Dim>& range) {
        declare_output<DType::f32, Dim>(ctx, std::move(local_name), range);
    }

    /// Declare a passthrough variable. Acts as both input and output —
    /// the entry is readable and writable. Adds to both input and output metadata.
    template <int Dim>
    void declare_passthrough_var(Context& ctx, std::string local_name,
                                 const sycl::range<Dim>& range) {
        declared_inputs_.push_back({local_name, range.size()});
        declared_outputs_.push_back({local_name, range.size()});
        ctx.declare_passthrough_var<Dim>(name() + "/" + local_name, range);
    }

    /// Mark an output as an objective. Prepends name() + "/" to local_name.
    void assign_objective(Context& ctx, std::string local_name);

    /// Mark an input or passthrough var as a design variable. Prepends name() + "/" to local_name.
    void assign_design_var(Context& ctx, std::string local_name);

    // --- Handle access helpers (called during forward/backward) ----------------
    //
    // Hot-path resolution goes through a component-local lookup cache (S8):
    // the first call for a local name does the full concat + hash resolution
    // and appends (name, entry index) to a small flat vector; subsequent
    // calls linear-scan the vector (faster than any map at component-sized N,
    // cache-friendly). Entry indices are append-only stable post-finalize;
    // slot routing is still resolved per call inside Context, so views are
    // never cached — only indices.

    /// Get a typed input handle. Prepends name() + "/" to local_name.
    template <int Dim>
    Input<Dim> get_input(Context& ctx, std::string_view local_name) {
        return ctx.get_input<Dim>(cached_index(ctx, local_name));
    }

    /// Dtyped input access (S1): f32 delegates to the standard view; other
    /// dtypes mint data-only dtyped views (no grad methods on the data tier).
    template <DType DT, int Dim>
    auto get_input(Context& ctx, std::string_view local_name) {
        return ctx.get_input<DT, Dim>(cached_index(ctx, local_name));
    }

    /// Get a typed output handle. Prepends name() + "/" to local_name.
    template <int Dim>
    Output<Dim> get_output(Context& ctx, std::string_view local_name) {
        return ctx.get_output<Dim>(cached_index(ctx, local_name));
    }

    /// Dtyped output access (S1) — see get_input.
    template <DType DT, int Dim>
    auto get_output(Context& ctx, std::string_view local_name) {
        return ctx.get_output<DT, Dim>(cached_index(ctx, local_name));
    }

    /// Get a typed passthrough handle. Prepends name() + "/" to local_name.
    template <int Dim>
    Output<Dim> get_passthrough(Context& ctx, std::string_view local_name) {
        return ctx.get_passthrough<Dim>(cached_index(ctx, local_name));
    }

    /// Declare a state variable. Prepends name() + "/" to local_name.
    template <int Dim>
    void declare_state_var(Context& ctx, std::string local_name, const sycl::range<Dim>& range) {
        ctx.declare_state_var<Dim>(name() + "/" + local_name, range);
    }

    /// Get a typed state var handle. Prepends name() + "/" to local_name.
    template <int Dim>
    StateVar<Dim> get_state_var(Context& ctx, std::string_view local_name) {
        return ctx.get_state_var<Dim>(cached_index(ctx, local_name));
    }

    /// Get an initializer for a state var. Prepends name() + "/" to local_name.
    /// Returns IOInitializer (write-only) targeting both primal slots.
    template <int Dim>
    IOInitializer<Dim> get_state_var_initializer(Context& ctx, const std::string& local_name) {
        return ctx.get_initializer<Dim>(cached_index(ctx, local_name));
    }

    /// Reset a state var to zero. Prepends name() + "/" to local_name.
    void reset_state_var(Context& ctx, std::string_view local_name) {
        ctx.reset_state_var(name() + "/" + std::string(local_name));
    }

    /// Declare an internal buffer variable. Prepends name() + "/" to local_name.
    template <int Dim>
    void declare_buffer_var(Context& ctx, std::string local_name, const sycl::range<Dim>& range) {
        ctx.declare_buffer_var<Dim>(name() + "/" + local_name, range);
    }

    /// Get a typed internal buffer handle. Prepends name() + "/" to local_name.
    template <int Dim>
    InternalBuffer<Dim> get_buffer_var(Context& ctx, std::string_view local_name) {
        return ctx.get_buffer_var<Dim>(cached_index(ctx, local_name));
    }

    // --- Kernel submission (S5a) ------------------------------------------------
    //
    // Submit one or more kernel functors in sequence through the framework's
    // single submission choke point (core/kernel.hpp). The in-order queue
    // provides the ordering contract between them.
    template <AnyKernel... Ks>
    void execute_compute(sycl::queue& q, const Ks&... ks) const {
        core::execute_compute(q, ks...);
    }

    // --- I/O introspection (populated by declare helpers) ----------------------

    /// Return all declared inputs with their local names and metadata.
    const std::vector<IOMetadata>& input_metadata() const override;

    /// Return all declared outputs with their local names and metadata.
    const std::vector<IOMetadata>& output_metadata() const override;

private:
    /// S8 lookup cache: local name → stable entry index in ctx. Guarded by
    /// the owning Context pointer so a component reused against a different
    /// Context never serves stale indices.
    size_t cached_index(Context& ctx, std::string_view local_name) {
        if (cache_ctx_ != &ctx) {
            lookup_cache_.clear();
            cache_ctx_ = &ctx;
        }
        for (const auto& [n, idx] : lookup_cache_)
            if (n == local_name) return idx;
        const size_t idx =
            ctx.entry_index(name() + "/" + std::string(local_name));
        lookup_cache_.emplace_back(std::string(local_name), idx);
        return idx;
    }

    std::vector<IOMetadata> declared_inputs_;
    std::vector<IOMetadata> declared_outputs_;
    std::vector<std::pair<std::string, size_t>> lookup_cache_;
    const Context* cache_ctx_ = nullptr;
};

} // namespace naga::cospan::core
