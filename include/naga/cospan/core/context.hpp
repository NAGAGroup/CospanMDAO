#pragma once

// =============================================================================
// naga/cospan/core/context.hpp — Context: I/O allocation manager
// =============================================================================
//
// Two-phase lifecycle:
//
//   Phase::Declaring → finalize() → Phase::Finalized
//
//   Declaring phase:
//     - declare_input<Dim> / declare_output<Dim> — register named entries with
//     shapes
//     - connect / assign_objective — annotate entries
//
//   finalize():
//     - Allocates device memory for all entries (data + grad where needed)
//     - Applies connection aliasing: when an Output connects to an Input,
//       the Input's data/grad pointers alias the Output's data/grad.
//       Connected entries (targets) are skipped during allocation — they
//       receive pointers from their source during the aliasing pass.
//     - Transitions to Phase::Finalized
//
//   Finalized phase:
//     - get_input<Dim> / get_output<Dim> — resolve entries to typed handles
//     - zero_output_gradients / zero_all_input_gradients
//
// Connection aliasing semantics:
//   connect(source, target) records that the Output `source` feeds the
//   Input `target`. During finalize(), target's data pointer is set to
//   source's data pointer, and target's grad pointer is set to source's
//   grad pointer. This means the target shares memory with the source —
//   writes to one are visible through the other. The target does not
//   receive its own allocation; it lives entirely in the source's memory.
//
// Shape storage:
//   All shapes are stored internally as sycl::range<3> with 1-padding for
//   unused dimensions. Max supported Dim is 3 (SYCL standard limit).
//   Shapes are left-justified: actual dimensions occupy the leftmost slots,
//   with 1-padding on the right.
//     range<1>{N}     → range<3>{N, 1, 1}
//     range<2>{A, B}  → range<3>{A, B, 1}
//     range<3>{A, B, C} → range<3>{A, B, C}
//   When retrieved via get_input<Dim>/get_output<Dim>, the range<3> is
//   collapsed to sycl::range<Dim> using the left-justified collapse rule:
//     - First Dim-1 dimensions as-is
//     - Last dimension = product of remaining dimensions
//     - Dim == 1 → range<1>{total_size}
//   This correctly round-trips: range<2>{3,4} → range<3>{3,4,1} → range<2>{3,4}
//
// Callers must provide fully-qualified entry names — no prefix state.

// SYCL
#include <sycl/sycl.hpp>

// SlabKit
#include <slabkit/array.hpp>   // slabkit::DeviceView
#include <slabkit/context.hpp> // slabkit::Context
#include <slabkit/memory.hpp>  // slabkit::malloc_device, unique_device_ptr

// Naga Core
#include "naga/cospan/core/detail/composable.hpp" // Composable, IOMetadata
#include "naga/cospan/core/dtype.hpp"             // DType, dtype_size, dtype_t
#include "naga/cospan/core/fusion.hpp"            // HorizontalFuser (tier-0.5 fusion)
#include "naga/cospan/core/io.hpp"                // Input<Dim>, Output<Dim>

namespace naga::cospan::core { class Component; } // forward declaration for execution registry

// Standard Library
#include <algorithm>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace naga::cospan::core {

// =============================================================================
// BackEdge — identifies a cycle-forming connection in the component graph
// =============================================================================

struct BackEdge {
  std::string
      source_component_name; // forward-graph source of the back edge (u in u→v)
  std::string
      target_component_name; // forward-graph target of the back edge (v in u→v)
  std::string source_entry_name; // FQN of u's Output entry
  std::string target_entry_name; // FQN of v's Input entry (aliases source)
};

// =============================================================================
// Context — I/O allocation manager for the component framework
// =============================================================================

class Context {
public:
  // =========================================================================
  // Phase — lifecycle state
  // =========================================================================

  enum class Phase { Declaring, Finalized };

  // =========================================================================
  // EntrySummary — read-only introspection struct
  // =========================================================================

  struct EntrySummary {
    std::string name;
    sycl::range<3> shape; // stored internally as 3D range
    size_t size;          // flat element count (product of shape dims)
    bool is_input;        // true if input, false if output
    bool is_objective;
    DType dtype = DType::f32; // element type of the entry's storage
    float *data; // device data pointer (valid after finalize)
    float *grad; // device grad pointer (valid after finalize; nullptr for
                 // data-tier entries — they never carry gradients)
  };

  // =========================================================================
  // Construction
  // =========================================================================

  Context(std::shared_ptr<slabkit::Context> sk_ctx, size_t device_index = 0);

  // Flushes the kernel profile (NAGA_KERNEL_PROF=1) while the queue pool and
  // SYCL runtime are still alive — see kernel_profile.hpp for why this must
  // NOT happen at atexit.
  ~Context();

  // =========================================================================
  // Declaration methods (Phase::Declaring only)
  // =========================================================================

  // Dtyped declaration variants (FRAMEWORK_CLEANUP S1): the enum value is
  // recorded in entry metadata; storage is allocated as count × dtype_size
  // bytes at finalize. The <Dim>-only overloads below alias to DType::f32 so
  // existing call sites are unchanged.
  template <DType DT, int Dim>
  void declare_input(std::string name, const sycl::range<Dim> &range) {
    declare_entry_common(std::move(name), to_shape3(range), EntryKind::Input,
                         DT, "declare_input()");
  }

  template <int Dim>
  void declare_input(std::string name, const sycl::range<Dim> &range) {
    declare_input<DType::f32, Dim>(std::move(name), range);
  }

  template <DType DT, int Dim>
  void declare_output(std::string name, const sycl::range<Dim> &range) {
    declare_entry_common(std::move(name), to_shape3(range), EntryKind::Output,
                         DT, "declare_output()");
  }

  template <int Dim>
  void declare_output(std::string name, const sycl::range<Dim> &range) {
    declare_output<DType::f32, Dim>(std::move(name), range);
  }

  template <DType DT, int Dim>
  void declare_passthrough_var(std::string name,
                               const sycl::range<Dim> &range) {
    static_assert(DT == DType::f32,
                  "P1: PassthroughVars are f32-only (grad-slot machinery is "
                  "f32; dtyped passthroughs arrive with the P2 io dtyping)");
    declare_entry_common(std::move(name), to_shape3(range),
                         EntryKind::PassthroughVar, DT,
                         "declare_passthrough_var()");
  }

  template <int Dim>
  void declare_passthrough_var(std::string name,
                               const sycl::range<Dim> &range) {
    declare_passthrough_var<DType::f32, Dim>(std::move(name), range);
  }

  template <DType DT, int Dim>
  void declare_promotion(std::string name, const sycl::range<Dim> &range) {
    declare_entry_common(std::move(name), to_shape3(range),
                         EntryKind::Promotion, DT, "declare_promotion()");
  }

  template <int Dim>
  void declare_promotion(std::string name, const sycl::range<Dim> &range) {
    declare_promotion<DType::f32, Dim>(std::move(name), range);
  }

  // =========================================================================
  // Connection & annotation (Phase::Declaring only)
  // =========================================================================

  void connect(std::string_view source_name, std::string_view target_name);

  // Disconnect (Phase::Declaring). Removes the exact recorded connection
  // between a source and target entry (no promotion-chain resolution) and
  // restores the target's is_target flag so it can be reconnected. Errors if
  // the connection does not exist.
  //
  // This enables splice surgery: an external training driver can disconnect
  // a model output from its downstream consumer and splice a crossbar
  // component between them (model_output -> XB/in0, driver_value -> XB/in1,
  // XB/out0 -> consumer) — downstream cannot tell.
  void disconnect(std::string_view source_name, std::string_view target_name);

  void assign_objective(std::string_view name);

  void set_max_backward_iterations(size_t k);

  // Must be called before finalize(); sets the history depth for Anderson
  // workspace allocation.
  void set_anderson_depth(size_t depth);

  // Returns iteration count from the most recent backward_to_equilibrium()
  // call.
  size_t last_adj_iters() const;

  // Full adjoint fixed-point solve with Anderson Acceleration.
  // Internally runs single-pass backward per outer iteration (bypasses
  // K_backward loop).
  struct AdjEquilibriumResult {
    size_t iters = 0;
    float norm = 0.0f;
    std::string worst_fqn;
    bool converged = false; // false = exited at max_adj_iters with norm >= tol
  };

  AdjEquilibriumResult backward_to_equilibrium(float adj_tol,
                                               size_t max_adj_iters,
                                               size_t check_interval,
                                               float adj_clip_norm);

  // Diagnostic (seed-vs-operator): when enabled, backward_to_equilibrium
  // logs max|λ| over recurrent grad prev slots after every backward pass
  // (raw, pre-clip/pre-blend). Distinguishes huge first-pass loss gradients
  // (scale appears at k=1) from operator amplification (scale grows with k).
  void set_adj_scale_log(bool v) { adj_scale_log_ = v; }

  // Entry points are the framework's registers (synchronous-logic semantics):
  // marking a component as an entry point means ALL of its incoming
  // connections read the previous tick (deferred), and its outputs drive the
  // current tick. Entry points with no upstream connections are pure
  // execution roots (the degenerate register). Marking may happen before or
  // after the component's inputs are connected, any time during Declaring.
  void mark_entry_point(const std::string &component_name);
  void unmark_entry_point(const std::string &component_name);

  size_t current_backward_iteration() const {
    return current_backward_iteration_;
  }

  // Slot rotation — public so EquilibriumExecutor can call at the top of each
  // forward sweep / adjoint iteration.
  //   advance_primal_slots(): curr becomes prev (no zeroing — outputs are
  //     functions; the owner fully rewrites curr each sweep).
  //   advance_grad_slots(): prev becomes curr (the transported iterate
  //     arrives), and the NEW prev slot is zeroed for this iteration's
  //     back-edge accumulation.
  void advance_primal_slots();
  void advance_grad_slots();

  void assign_design_var(std::string_view name);

  // =========================================================================
  // Unassign methods (Phase::Declaring only)
  // =========================================================================

  void unassign_objective(std::string_view name);
  void unassign_design_var(std::string_view name);

  // =========================================================================
  // DV freeze/thaw (any phase after declare)
  // =========================================================================

  // Freeze a specific DV by fully-qualified name
  void freeze_dv(const std::string &name);

  // Thaw (unfreeze) a specific DV
  void thaw_dv(const std::string &name);

  // Freeze all DVs except those named in keep_thawed
  void freeze_all_dvs_except(const std::vector<std::string> &keep_thawed);

  // Thaw all DVs (return to full training mode)
  void thaw_all_dvs();

  // Query whether a DV is frozen (useful for testing)
  bool is_frozen(const std::string &name) const;

  // =========================================================================
  // Component registration (Phase::Declaring only)
  // =========================================================================

  void register_component(Component *component_ptr,
                          const std::string &component_name,
                          const std::vector<IOMetadata> &inputs,
                          const std::vector<IOMetadata> &outputs);

  // =========================================================================
  // Post-finalize handle resolution (Phase::Finalized only)
  //
  // Every getter family has two overloads:
  //   - by FQN (string): full hash-map resolution. System-level access and
  //     app/inspection code use this — deliberately uncached (S8).
  //   - by entry index (size_t): skips resolution. Entry indices are
  //     append-only stable for the lifetime of the Context, so callers may
  //     resolve once and cache the index (Component::cached_index() does —
  //     the S8 component-local lookup cache). Slot routing (back-edge
  //     rotation) is still resolved per call, so views must NOT be cached
  //     across sweeps — only indices.
  // =========================================================================

  /// Resolve an FQN to its stable entry index (Phase::Finalized only).
  size_t entry_index(std::string_view name,
                     const char *fn = "entry_index()") const {
    assert_phase(Phase::Finalized, fn);
    std::string n(name);
    auto it = entry_map_.find(n);
    if (it == entry_map_.end())
      throw std::invalid_argument(std::string("Context::") + fn + ": entry '" +
                                  n + "' not found");
    return it->second;
  }

  template <int Dim> Input<Dim> get_input(std::string_view name) {
    return get_input<Dim>(entry_index(name, "get_input()"));
  }

  template <int Dim> Input<Dim> get_input(size_t entry_idx) {
    assert_phase(Phase::Finalized, "get_input()");
    const Entry *entry = &entries_[entry_idx];
    const std::string &n = entry->name;
    if (entry->kind == EntryKind::PassthroughVar)
      throw std::invalid_argument("Context::get_input(): '" + n +
                                  "' is a passthrough var — "
                                  "passthrough vars cannot be used in compute "
                                  "kernels, use get_passthrough() instead");
    if (entry->kind == EntryKind::StateVar)
      throw std::invalid_argument(
          "Context::get_input(): '" + n +
          "' is a state var — use get_state_var() instead");
    if (entry->kind == EntryKind::InternalBuffer)
      throw std::invalid_argument(
          "Context::get_input(): '" + n +
          "' is an internal buffer — use get_buffer_var() instead");
    if (entry->kind != EntryKind::Input && entry->kind != EntryKind::Promotion)
      throw std::invalid_argument("Context::get_input(): '" + n +
                                  "' is not an input");
    require_dtype(*entry, DType::f32, "get_input()");

    // Collapse shape3 to range<Dim>
    sycl::range<Dim> range = collapse_shape3<Dim>(entry->shape3);

    // Resolve grad pointers (handles recurrent-output slot routing).
    float *rg = resolve_read_grad(*entry);

    // Gradient accumulation routes by connection role (the adjoint mirror of
    // the primal prev/curr discipline): a FRESH consumer accumulates into the
    // grad CURR slot — consumed by the owner's backward in the SAME adjoint
    // iteration; a BACK-EDGE (deferred) consumer accumulates into the grad
    // PREV slot — rotated into curr at the next iteration (the transport
    // t_{k+1} = Jᵀ t_k). Non-recurrent owners have a single grad buffer.
    float *accum_grad_ptr = resolve_accum_grad(*entry);

    slabkit::DeviceView<float, Dim> data_view(resolve_read_data(*entry), range);
    slabkit::DeviceView<float, Dim> read_grad_view(rg, range);
    slabkit::DeviceView<float, Dim> accum_grad_view(accum_grad_ptr, range);

    return Input<Dim>(data_view, read_grad_view, accum_grad_view);
  }

  template <int Dim> Output<Dim> get_output(std::string_view name) {
    return get_output<Dim>(entry_index(name, "get_output()"));
  }

  template <int Dim> Output<Dim> get_output(size_t entry_idx) {
    assert_phase(Phase::Finalized, "get_output()");
    const Entry *entry = &entries_[entry_idx];
    const std::string &n = entry->name;
    if (entry->kind == EntryKind::PassthroughVar)
      throw std::invalid_argument("Context::get_output(): '" + n +
                                  "' is a passthrough var — "
                                  "passthrough vars cannot be used in compute "
                                  "kernels, use get_passthrough() instead");
    if (entry->kind == EntryKind::StateVar)
      throw std::invalid_argument(
          "Context::get_output(): '" + n +
          "' is a state var — use get_state_var() instead");
    if (entry->kind == EntryKind::InternalBuffer)
      throw std::invalid_argument(
          "Context::get_output(): '" + n +
          "' is an internal buffer — use get_buffer_var() instead");
    if (entry->kind != EntryKind::Output && entry->kind != EntryKind::Promotion)
      throw std::invalid_argument("Context::get_output(): '" + n +
                                  "' is not an output");
    require_dtype(*entry, DType::f32, "get_output()");

    // Collapse shape3 to range<Dim>
    sycl::range<Dim> range = collapse_shape3<Dim>(entry->shape3);

    // Resolve grad read pointer — Output only exposes get_grad (grad curr
    // slot: the total gradient consumed this adjoint iteration).
    float *rg = resolve_read_grad(*entry);

    slabkit::DeviceView<float, Dim> data_view(resolve_write_data(*entry),
                                              range);
    slabkit::DeviceView<float, Dim> read_grad_view(rg, range);

    return Output<Dim>(data_view, read_grad_view);
  }

  // Dtyped access variants (FRAMEWORK_CLEANUP S1/S2). For f32 these delegate
  // to the standard view getters; for other dtypes they mint data-only dtyped
  // views over the type-erased storage (no grad slots exist off the f32 path
  // — data-tier views have no grad methods at all, and f64 entries carry no
  // grad storage until the adjoint machinery is dtyped). Runtime-checks the
  // entry's dtype tag.
  template <DType DT, int Dim> auto get_input(std::string_view name) {
    return get_input<DT, Dim>(entry_index(name, "get_input()"));
  }

  template <DType DT, int Dim> auto get_input(size_t entry_idx) {
    if constexpr (DT == DType::f32) {
      return get_input<Dim>(entry_idx);
    } else {
      assert_phase(Phase::Finalized, "get_input()");
      const Entry *entry = &entries_[entry_idx];
      const std::string &n = entry->name;
      if (entry->kind != EntryKind::Input &&
          entry->kind != EntryKind::Promotion)
        throw std::invalid_argument("Context::get_input(): '" + n +
                                    "' is not an input");
      require_dtype(*entry, DT, "get_input()");
      using T = dtype_t<DT>;
      sycl::range<Dim> range = collapse_shape3<Dim>(entry->shape3);
      return Input<Dim, DT>(slabkit::DeviceView<T, Dim>(
          reinterpret_cast<T *>(resolve_read_data(*entry)), range));
    }
  }

  template <DType DT, int Dim> auto get_output(std::string_view name) {
    return get_output<DT, Dim>(entry_index(name, "get_output()"));
  }

  template <DType DT, int Dim> auto get_output(size_t entry_idx) {
    if constexpr (DT == DType::f32) {
      return get_output<Dim>(entry_idx);
    } else {
      assert_phase(Phase::Finalized, "get_output()");
      const Entry *entry = &entries_[entry_idx];
      const std::string &n = entry->name;
      if (entry->kind != EntryKind::Output &&
          entry->kind != EntryKind::Promotion)
        throw std::invalid_argument("Context::get_output(): '" + n +
                                    "' is not an output");
      require_dtype(*entry, DT, "get_output()");
      using T = dtype_t<DT>;
      sycl::range<Dim> range = collapse_shape3<Dim>(entry->shape3);
      return Output<Dim, DT>(slabkit::DeviceView<T, Dim>(
          reinterpret_cast<T *>(resolve_write_data(*entry)), range));
    }
  }

  // get_initializer<DT, Dim>: returns an IOInitializer for any entry kind.
  // Intended exclusively for init() / reset() / host-seeding use — bypasses
  // compute-kernel kind enforcement so that PassthroughVar, StateVar, and
  // InternalBuffer entries can have their primal data zeroed or seeded.
  template <DType DT, int Dim>
  IOInitializer<Dim, DT> get_initializer(std::string_view name) {
    return get_initializer<DT, Dim>(entry_index(name, "get_initializer()"));
  }

  template <DType DT, int Dim>
  IOInitializer<Dim, DT> get_initializer(size_t entry_idx) {
    assert_phase(Phase::Finalized, "get_initializer()");
    const Entry *entry = &entries_[entry_idx];
    require_dtype(*entry, DT, "get_initializer()");
    using T = dtype_t<DT>;
    sycl::range<Dim> range = collapse_shape3<Dim>(entry->shape3);
    // Curr slot — the current-sweep buffer that forward() will read.
    T *data = reinterpret_cast<T *>(resolve_write_data(*entry));
    // Prev slot — for any recurrent output (StateVar or back-edge cut
    // source) this is the other slot, so that deferred readers and
    // prev()-window reads on the first forward sweep see the initialized
    // value. For single-slot entries both pointers are identical.
    // NOTE: resolved directly from the slot table (not resolve_read_data,
    // which applies per-connection read discipline and would return the
    // curr slot for the owning entry itself). Slot pairs only exist on the
    // f32 path (StateVars are f32-gated at declaration).
    T *prev = data;
    const Entry &owner = get_owning_entry(*entry);
    if (float *r = primal_prev_slot(owner.name))
      prev = reinterpret_cast<T *>(r);
    return IOInitializer<Dim, DT>{data, prev, range};
  }

  template <int Dim> IOInitializer<Dim> get_initializer(std::string_view name) {
    return get_initializer<DType::f32, Dim>(name);
  }

  template <int Dim> IOInitializer<Dim> get_initializer(size_t entry_idx) {
    return get_initializer<DType::f32, Dim>(entry_idx);
  }

  template <int Dim> Output<Dim> get_passthrough(std::string_view name) {
    return get_passthrough<Dim>(entry_index(name, "get_passthrough()"));
  }

  template <int Dim> Output<Dim> get_passthrough(size_t entry_idx) {
    assert_phase(Phase::Finalized, "get_passthrough()");
    const Entry *entry = &entries_[entry_idx];
    const std::string &n = entry->name;
    if (entry->kind != EntryKind::PassthroughVar)
      throw std::invalid_argument("Context::get_passthrough(): '" + n +
                                  "' is not a passthrough var");
    require_dtype(*entry, DType::f32, "get_passthrough()");
    sycl::range<Dim> range = collapse_shape3<Dim>(entry->shape3);
    float *rg = resolve_read_grad(*entry);
    slabkit::DeviceView<float, Dim> data_view(resolve_write_data(*entry),
                                              range);
    slabkit::DeviceView<float, Dim> read_grad_view(rg, range);
    return Output<Dim>(data_view, read_grad_view);
  }

  template <DType DT, int Dim>
  void declare_state_var(std::string name, const sycl::range<Dim> &range) {
    static_assert(DT == DType::f32,
                  "P1: StateVars are f32-only (the dual-slot / adjoint "
                  "machinery is f32; dtyped StateVars arrive with the P2 io "
                  "dtyping)");
    declare_entry_common(std::move(name), to_shape3(range), EntryKind::StateVar,
                         DT, "declare_state_var()");
  }

  template <int Dim>
  void declare_state_var(std::string name, const sycl::range<Dim> &range) {
    declare_state_var<DType::f32, Dim>(std::move(name), range);
  }

  template <int Dim> StateVar<Dim> get_state_var(std::string_view name) {
    return get_state_var<Dim>(entry_index(name, "get_state_var()"));
  }

  template <int Dim> StateVar<Dim> get_state_var(size_t entry_idx) {
    assert_phase(Phase::Finalized, "get_state_var()");
    const Entry *entry = &entries_[entry_idx];
    const std::string &n = entry->name;
    if (entry->kind != EntryKind::StateVar)
      throw std::invalid_argument("Context::get_state_var(): '" + n +
                                  "' is not a StateVar");
    require_dtype(*entry, DType::f32, "get_state_var()");
    sycl::range<Dim> range = collapse_shape3<Dim>(entry->shape3);
    // StateVar = the degenerate back-edge (owner is its own deferred
    // consumer). Directly index the slot pairs: current() writes the curr
    // slot, prev() reads the prev slot; current().get_grad reads the grad
    // CURR slot (total gradient consumed this adjoint iteration), and
    // prev().accum_grad writes the grad PREV slot (the self-back-edge's
    // transported contribution, consumed next iteration). resolve_accum_grad
    // is NOT used here: it routes by the entry's connection role, and the
    // statevar entry itself is not a connection target.
    const auto &pidx = primal_slot_idx_.at(n);
    float *prev_data    = primal_slots_.at(n)[pidx.prev].get();
    float *current_data = primal_slots_.at(n)[pidx.curr].get();
    const auto &gidx = grad_slot_idx_.at(n);
    float *grad_read = grad_slots_.at(n)[gidx.curr].get();
    float *adj_write = grad_slots_.at(n)[gidx.prev].get();
    return StateVar<Dim>(prev_data, current_data, grad_read, adj_write, range);
  }

  void reset_state_var(std::string_view fqn);

  // Share outputs (Phase::Declaring). promote_output minus the single-source
  // constraint: creates a shared-storage owner as a
  // Promotion entry named owner_fqn and aliases each member onto it. The
  // owner holds the single physical buffer (no slot pair unless a back-edge
  // cut later marks it a recurrent output); members are excluded from
  // allocation and resolve to the owner via get_owning_entry. This is the
  // ONLY share mechanism (shared statevars are dead, 2026-07-20: a "shared
  // statevar" was just parallel components co-writing distinct slices of one
  // output — i.e. this, plus a register cut when the result is recurrent).
  //
  // Members co-write disjoint slices with function-return semantics: plain
  // stores, no owner zeroing, no atomics needed, every element accounted for
  // each forward. A member may be:
  //   - a leaf Output entry;
  //   - a plain Promotion (from promote_output) — resolved through its
  //     sources[0] chain to the leaf Output, which is enrolled, so aliasing
  //     flows leaf -> ... -> owner for the entire chain;
  //   - the owner of a lower-level share_outputs group (nested re-share) —
  //     enrolled directly.
  //
  // Consumer ordering: connecting from the owner, a member, or any promotion
  // over a member orders the consumer after ALL members via the share-group
  // edge union in build_component_graph (generic over shareable Promotion
  // owners).
  void share_outputs(const std::string &owner_fqn,
                     const std::vector<std::string> &member_fqns);

  template <DType DT, int Dim>
  void declare_buffer_var(std::string name, const sycl::range<Dim> &range) {
    static_assert(DT == DType::f32,
                  "P1: InternalBuffers are f32-only (they carry grads; dtyped "
                  "buffers arrive with the P2 io dtyping)");
    declare_entry_common(std::move(name), to_shape3(range),
                         EntryKind::InternalBuffer, DT,
                         "declare_buffer_var()");
  }

  template <int Dim>
  void declare_buffer_var(std::string name, const sycl::range<Dim> &range) {
    declare_buffer_var<DType::f32, Dim>(std::move(name), range);
  }

  template <int Dim> InternalBuffer<Dim> get_buffer_var(std::string_view name) {
    return get_buffer_var<Dim>(entry_index(name, "get_buffer_var()"));
  }

  template <int Dim> InternalBuffer<Dim> get_buffer_var(size_t entry_idx) {
    assert_phase(Phase::Finalized, "get_buffer_var()");
    const Entry *entry = &entries_[entry_idx];
    const std::string &n = entry->name;
    if (entry->kind != EntryKind::InternalBuffer)
      throw std::invalid_argument("Context::get_buffer_var(): '" + n +
                                  "' is not an InternalBuffer");
    require_dtype(*entry, DType::f32, "get_buffer_var()");
    sycl::range<Dim> range = collapse_shape3<Dim>(entry->shape3);
    float *data = entry->data;
    float *grad = entry->grad;
    return InternalBuffer<Dim>(data, grad, range);
  }

  // =========================================================================
  // Gradient zeroing (Phase::Finalized only)
  // =========================================================================

  void zero_non_dv_grads();
  void zero_dv_grads();

  // Zeros all single-slot grad buffers between adjoint iterations. Recurrent
  // outputs are skipped (their slot pairs carry the transport iterate t_k);
  // DVs are skipped (they ACCUMULATE Σ_k (∂F/∂θ)ᵀ·t_k across iterations —
  // transport-form adjoint); objectives are NOT skipped (chain-rule-honest
  // losses contribute nothing after the single k=0 seeding).
  void zero_non_recurrent_grads();

  void seed_objective_grad();

  // =========================================================================
  // Serialization (Phase::Finalized only)
  // =========================================================================

  void save(const std::string &path);
  void load(const std::string &path);

  // =========================================================================
  // Pointer patching (any phase)
  // =========================================================================

  void patch_data(std::string_view name, float *new_data);
  void patch_grad(std::string_view name, float *new_grad);

  // =========================================================================
  // Entry introspection (any phase, read-only)
  // =========================================================================

  std::vector<EntrySummary> entries() const;

  // Direct const access to internal registries
  const auto &entry_map() const { return entry_map_; }
  const auto &entries_vec() const { return entries_; }

  // Check if an entry is a connection target (i.e., connected from upstream)
  bool is_connected(std::string_view name) const;

  // Returns true if the named component has at least one registered input or
  // output entry. A plain System (no I/O metadata) will not appear in
  // component_nodes_ and returns false.
  bool component_has_io(const std::string &name) const;

  // Walk up the FQN prefix chain of an entry name to find the nearest
  // registered component that owns it. Returns the component name if found, ""
  // otherwise. This resolves naming hierarchy (e.g.
  // "model/outputs/readout/u" → "model") and is NOT connection
  // tracing — it only looks at the naming structure.
  std::string owning_component(const std::string &fqn) const;

  // =========================================================================
  // Phase query (any phase)
  // =========================================================================

  Phase phase() const { return phase_; }

  // =========================================================================
  // Device access (any phase)
  // =========================================================================

  sycl::queue &queue() { return queue_pool_[0]; }

  const sycl::queue &queue() const { return queue_pool_[0]; }

  std::shared_ptr<slabkit::Context> sk_ctx_shared() { return sk_ctx_; }

  const std::shared_ptr<slabkit::Context> &sk_ctx_shared() const {
    return sk_ctx_;
  }

  size_t device_index() const { return device_index_; }

  std::shared_ptr<slabkit::Context> sk_ctx() const { return sk_ctx_; }

  // Back-edge traversal (for EquilibriumExecutor)
  const std::vector<BackEdge> &back_edges() const { return back_edges_; }

  // Returns FQNs of all non-target recurrent entries (for EquilibriumExecutor
  // convergence)
  std::vector<std::string> recurrent_entries() const;

  // =========================================================================
  // Shape utility functions (public for use by Component helpers)
  // =========================================================================

  /// Compute total element count from 3D shape range
  static size_t shape3_size(const sycl::range<3> &s) {
    return s[0] * s[1] * s[2];
  }

  /// Convert sycl::range<Dim> to sycl::range<3> with 1-padding
  template <int Dim>
  static sycl::range<3> to_shape3(const sycl::range<Dim> &r) {
    sycl::range<3> result{1, 1, 1};
    for (int i = 0; i < Dim && i < 3; ++i) {
      result[i] = r[i];
    }
    return result;
  }

  // =========================================================================
  // Entry — internal I/O entry descriptor (public for EquilibriumExecutor
  // access)
  // =========================================================================

  enum class EntryKind {
    Input,
    Output,
    PassthroughVar,
    Promotion,
    StateVar,
    InternalBuffer
  };

  struct Entry {
    std::string name;
    sycl::range<3> shape3; // stored shape
    EntryKind kind;
    DType dtype = DType::f32; // element type — storage is allocated as
                              // shape3_size × dtype_size(dtype) bytes
    bool is_objective = false;
    bool is_design_variable = false;
    bool is_frozen = false; // DV freeze flag — optimizer skips frozen DVs
    // Type-erased base addresses. For f32 entries these are directly usable
    // as float*; for data-tier entries they are byte allocations whose typed
    // interpretation goes through the dtype-checked accessors (dtyped views,
    // io.hpp). grad is nullptr for non-f32 entries — the adjoint machinery
    // is f32-only.
    float *data = nullptr;
    float *grad = nullptr;

    // StateVar uses the standard primal_slots_ pair for prev/curr primal
    // buffers. No separate state_prev_data field needed.

    std::vector<std::string>
        sources; // for inputs: sources that connect to this
    std::vector<std::string> targets; // for outputs: targets this connects to
    bool is_target = false;           // true if connected as target
    bool is_shareable = false;     // true iff a share_outputs owner Promotion
    bool is_share_member = false;  // true iff this entry was ENROLLED as a
                                   // share member (alias of an owner's
                                   // storage). Distinguishes members from
                                   // consumers connected FROM the owner —
                                   // both appear in owner.targets with
                                   // sources[0] == owner, but only members
                                   // may be flattened by the share-group
                                   // union pass; owner-sourced consumers get
                                   // synthesized member→consumer ordering
                                   // edges instead.
    size_t share_index = 0;        // position within its share group (0 when unshared)
    size_t share_count = 1;        // number of members in its share group (1 when unshared)
    bool is_back_edge_source =
        false; // true if this entry is the SOURCE of a cut connection (an
               // entry-point input or an emergent back-edge in an unmarked
               // cycle). Bookkeeping/introspection only — read routing is
               // per-connection via is_deferred_read on the TARGET entry.
    bool is_recurrent_output =
        false; // RECURRENT OUTPUT mark: this entry is the owning source of a
               // back-edge cut connection. Grants the prev/curr primal slot
               // pair and the grad slot pair. StateVar-kind entries are
               // recurrent outputs by construction (the degenerate self
               // back-edge) regardless of this flag. Everything else is
               // single-slot — including other outputs of a component that
               // owns a back-edge input.
    bool is_deferred_read =
        false; // Per-connection read discipline: true iff THIS target entry
               // reads the previous sweep (prev slot) — its connection was
               // cut, either because its component is a marked entry point
               // or because the traversal cut it to break an unmarked cycle.
               // All other consumers read fresh (curr slot, ordered by the
               // topological sweep).
  };

  // Entry lookup by FQN (placed after Entry definition to avoid
  // forward-reference issue)
  const Entry &get_entry(const std::string &fqn) const {
    return entries_[entry_map_.at(fqn)];
  }

  // Grad CURR slot: the iterate consumed this adjoint iteration (t_k, plus
  // any fresh-consumer contributions accumulated during the reverse sweep).
  float *grad_curr_slot(const std::string &fqn) {
    return grad_slots_.at(fqn)[grad_slot_idx_.at(fqn).curr].get();
  }

  // Grad PREV slot: the back-edge accumulation target (t_{k+1} — rotated
  // into curr at the next iteration).
  float *grad_prev_slot(const std::string &fqn) {
    return grad_slots_.at(fqn)[grad_slot_idx_.at(fqn).prev].get();
  }

  // Resolve an FQN through its alias chain (targets / share members) to the
  // owning non-target entry. Returns nullptr if the FQN is unknown.
  const Entry *owning_entry(const std::string &fqn) const {
    auto it = entry_map_.find(fqn);
    if (it == entry_map_.end())
      return nullptr;
    return &get_owning_entry(entries_[it->second]);
  }

  // Primal-slot getters resolve to the owning entry first: slot tables are
  // keyed by owner name, so a share-member or alias-target FQN previously
  // returned nullptr here and was silently dropped by callers measuring
  // settle residuals (audit 2026-07-11). Returns nullptr only when the OWNER
  // genuinely has no slot pair (not a recurrent output).
  float *primal_prev_slot(const std::string &fqn) const {
    const Entry *owner = owning_entry(fqn);
    if (!owner)
      return nullptr;
    auto it = primal_slot_idx_.find(owner->name);
    if (it == primal_slot_idx_.end())
      return nullptr;
    return primal_slots_.at(owner->name)[it->second.prev].get();
  }

  float *primal_curr_slot(const std::string &fqn) const {
    const Entry *owner = owning_entry(fqn);
    if (!owner)
      return nullptr;
    auto it = primal_slot_idx_.find(owner->name);
    if (it == primal_slot_idx_.end())
      return nullptr;
    return primal_slots_.at(owner->name)[it->second.curr].get();
  }

private:
  friend class Executor;
  friend class MockContext;

  // =========================================================================
  // Lifecycle & execution — Executor-only
  // =========================================================================

  void finalize();
  void forward();
  void backward();
  void init_components();
  void reset_components();
  // Single-pass backward: runs the full backward graph exactly once (k=0 active
  // set only). Caller is responsible for slot advancement.
  void backward_single_pass();

  // Scatter/gather between recurrent grad prev slots and anderson_flat_
  void scatter_recurrent_write_to_flat();
  void scatter_flat_to_recurrent_write();

  // Scatter the current recurrent grad iterate into
  // anderson_iterates_[ring_slot]. Must be called BEFORE backward_single_pass()
  // so we capture x_k (the iterate being fed in) before
  // advance_grad_slots() rotates the slots.
  void scatter_recurrent_write_to_iterates(size_t ring_slot);

  // Anderson kernel orchestration
  void anderson_compute_residual_and_store(size_t ring_slot);
  void anderson_update_gram_column(size_t h, size_t new_col_ring_slot);
  void anderson_solve_and_apply_blend(size_t h, size_t newest_ring_slot);

  // Recompute full h×h Gram matrix (used when ring buffer wraps)
  void anderson_recompute_gram(size_t h, size_t newest_ring_slot);

  // Adjoint transport residual: max |grad_prev - grad_curr| over recurrent outputs
  float backward_to_equilibrium_compute_norm(std::string &worst_out);

  // Adjoint scale: max|write| over recurrent grad entries (adjoint-scale diagnostic)
  float backward_to_equilibrium_compute_scale(std::string &worst_out);
  bool adj_scale_log_ = false;

  // =========================================================================
  // Internal types
  // =========================================================================

  struct ComponentNode {
    Component *ptr = nullptr;
    std::string name;
    std::vector<IOMetadata> inputs;
    std::vector<IOMetadata> outputs;
  };

  // =========================================================================
  // ComponentGraph — adjacency structure for SCC / back-edge analysis
  // =========================================================================

  struct ComponentGraph {
    int N = 0;
    std::vector<std::string> idx_to_name;
    std::unordered_map<std::string, int> name_to_idx;
    std::vector<std::vector<int>> adj;
    // Maps (src_idx, tgt_idx) → list of (src_entry_fqn, tgt_entry_fqn) pairs
    std::map<std::pair<int, int>,
             std::vector<std::pair<std::string, std::string>>>
        edge_entries;
  };

  // =========================================================================
  // Phase enforcement
  // =========================================================================

  void assert_phase(Phase expected, const char *fn) const;

  // =========================================================================
  // Dtype enforcement & common declaration (private helpers)
  // =========================================================================

  void require_dtype(const Entry &entry, DType expected, const char *fn) const {
    if (entry.dtype != expected)
      throw std::invalid_argument(
          std::string("Context::") + fn + ": entry '" + entry.name +
          "' has dtype " + dtype_name(entry.dtype) + ", expected " +
          dtype_name(expected) +
          " — use the dtyped accessor matching the entry's declared dtype");
  }

  void declare_entry_common(std::string name, const sycl::range<3> &shape3,
                            EntryKind kind, DType dt, const char *fn) {
    assert_phase(Phase::Declaring, fn);
    if (entry_map_.find(name) != entry_map_.end())
      throw std::invalid_argument(std::string("Context::") + fn + ": entry '" +
                                  name + "' already exists");
    Entry entry;
    entry.name = name;
    entry.shape3 = shape3;
    entry.kind = kind;
    entry.dtype = dt;
    entries_.emplace_back(std::move(entry));
    entry_map_[entries_.back().name] = entries_.size() - 1;
  }

  // =========================================================================
  // Shape helpers (private)
  // =========================================================================

  // Collapse sycl::range<3> to sycl::range<Dim>
  // Shapes are stored left-justified (actual data in leftmost slots,
  // 1-padding on the right). Collapse takes first Dim-1 dims as-is,
  // multiplies remaining dims into the last dimension.
  //   Dim == 1 → range<1>{total_size} (flat view)
  //   Dim == 2 → range<2>{s[0], s[1]*s[2]}
  //   Dim == 3 → range<3>{s[0], s[1], s[2]} (identity)
  template <int Dim>
  sycl::range<Dim> collapse_shape3(const sycl::range<3> &s) const {
    if constexpr (Dim == 1) {
      return sycl::range<1>{shape3_size(s)};
    } else {
      sycl::range<Dim> result;
      // First Dim-1 dimensions as-is
      for (int i = 0; i < Dim - 1; ++i) {
        result[i] = s[i];
      }
      // Last dimension = product of remaining dimensions
      size_t last_dim = 1;
      for (int j = Dim - 1; j < 3; ++j) {
        last_dim *= s[j];
      }
      result[Dim - 1] = last_dim;
      return result;
    }
  }

  // =========================================================================
  // Execution order computation (private)
  // =========================================================================

  void validate_entry_points();

  // =========================================================================
  // Graph building: declared cuts + SCC-condensation execution order +
  // emergent back-edge detection
  //
  // Algorithm overview:
  //   1. identify_declared_cuts() — every connection whose target component is
  //      a marked entry point is cut up front (declared register placement).
  //      Cut connections are registered as back-edges and their target entries
  //      are flagged is_deferred_read.
  //   2. compute_sccs() — Tarjan's SCC decomposition on the graph MINUS
  //      declared cuts. Any cycle passing through an entry point dissolves
  //      here; any non-trivial SCC that remains is a cycle with no register
  //      on it (delay placement will be emergent — warned about loudly).
  //   3. compute_dag_and_order() — for each residual non-trivial SCC, runs a
  //      single DFS from the SCC's entry nodes. This DFS simultaneously:
  //      a) identifies emergent back-edges (edges to DFS ancestors)
  //      b) produces intra-SCC execution order (reverse DFS postorder)
  //      Using the same DFS for both guarantees consistency — the back-edge
  //      set is always a valid feedback arc set for the ordering produced.
  //      At the SCC level, Kahn's is run on the condensation DAG.
  //   4. compute_backward_active_sets() — BFS backward from objective-owning
  //      components using ALL edges so every node on any path into an
  //      objective is in the active set.
  //
  // Back-edge semantics (declared or emergent — one mechanism):
  //   The back-edge IS the cut point IS the recurrence carrier. The owning
  //   SOURCE entry is marked a RECURRENT OUTPUT and gets the prev/curr slot
  //   pairs (primal + grad), rotated by advance_{primal,grad}_slots. The
  //   full role table (primal sweep / adjoint iteration):
  //
  //                         primal (per sweep)      adjoint (per iteration)
  //     owner               writes curr             reads grad-curr
  //     fresh consumer      reads curr              accumulates grad-curr
  //     back-edge consumer  reads prev              accumulates grad-prev
  //     rotation            curr becomes prev       prev becomes curr
  //
  //   Only the specific target entries of cut connections read the prev
  //   slot; every other consumer of the same source reads fresh
  //   (per-connection discipline, keyed on Entry::is_deferred_read). The
  //   same flag routes gradient accumulation to the mirrored slot. StateVars
  //   are the degenerate case (owner is its own back-edge consumer).
  // =========================================================================

  /// Cut all connections targeting marked entry-point components. Registers
  /// them in back_edges_, flags deferred reads and carrier sources, and
  /// returns the cut set as component-graph index pairs for exclusion from
  /// SCC computation and ordering.
  std::set<std::pair<int, int>> identify_declared_cuts(const ComponentGraph &g);

  /// Shared registration for a cut connection (declared or emergent):
  /// records the BackEdge, marks the owning source entry as carrier
  /// (is_back_edge_source; is_recurrent_output unless DV), and flags the
  /// target entry is_deferred_read.
  void register_back_edge(const ComponentGraph &g, int u, int v,
                          const std::string &src_fqn,
                          const std::string &tgt_fqn);

  // Helper: run intra-SCC DFS from given entry seeds, recording emergent
  // back-edges and returning nodes in reverse postorder (valid execution
  // order given the back-edges found by this same traversal). Edges in
  // `cuts` are never traversed (already cut).
  std::vector<int> scc_dfs_order(const ComponentGraph &g,
                                 const std::vector<int> &entry_seeds,
                                 const std::unordered_set<int> &in_scc,
                                 const std::set<std::pair<int, int>> &cuts);

  void compute_dag_and_order(ComponentGraph &g,
                             const std::vector<std::vector<int>> &sccs,
                             const std::set<std::pair<int, int>> &cuts);
  void compute_backward_active_sets(const ComponentGraph &g);

  // =========================================================================
  // Back-edge slot resolution helpers
  // =========================================================================

  /// Follow alias chain (via entry.sources[0]) until reaching a non-target
  /// entry.
  const Entry &get_owning_entry(const Entry &entry) const;

  float *resolve_read_data(const Entry &entry) const;
  float *resolve_write_data(const Entry &entry) const;

  /// Resolve the grad READ pointer for an entry. For recurrent-output owners
  /// returns the grad CURR slot (the total gradient consumed this adjoint
  /// iteration). For normal entries, returns owner.grad.
  float *resolve_read_grad(const Entry &entry) const;

  /// Resolve the grad ACCUMULATION pointer for an entry — the adjoint mirror
  /// of the primal read discipline, routed by the TARGET entry's connection
  /// role: a back-edge (deferred) target accumulates into the owner's grad
  /// PREV slot (transported to the next adjoint iteration); every other
  /// consumer accumulates into the grad CURR slot (consumed by the owner's
  /// backward in the SAME iteration). For normal entries, returns owner.grad.
  float *resolve_accum_grad(const Entry &entry) const;

  // =========================================================================
  // Entry lookup
  // =========================================================================

  Entry *find_entry(const std::string &name);
  const Entry *find_entry(const std::string &name) const;

  // =========================================================================
  // Connection chain resolution helpers
  // =========================================================================

  /// Finds the real source component by walking backward through system-level
  /// promotion chains. Returns the component name that owns the originating
  /// Output entry, or empty string if none is found.
  std::string resolve_source_component(const std::string &src_fqn) const;

  /// Finds all real target components by walking DOWN the targets chain.
  /// For system-level promotions, recursively follows entry.targets until
  /// reaching component-owned entries, returning all resolved component names.
  std::vector<std::string>
  resolve_target_components(const std::string &tgt_fqn) const;

  // =========================================================================
  // Graph analysis method declarations
  // =========================================================================

  void assign_component_layers(const ComponentGraph &g);

  /// Build a component-level directed graph from entry connections.
  /// Vertices = registered components. Edges = data-flow dependencies
  /// derived from entry connections (following passthrough chains).
  ComponentGraph build_component_graph() const;

  /// Iterative Tarjan's SCC algorithm on the graph minus `cuts` (declared
  /// entry-point cuts). Returns SCCs in reverse topological order.
  std::vector<std::vector<int>>
  compute_sccs(const ComponentGraph &g,
               const std::set<std::pair<int, int>> &cuts) const;

  // Creates one queue for queue_pool_: in-order always; with
  // property::queue::enable_profiling iff NAGA_KERNEL_PROF=1 (P5 kernel
  // instrumentation — see core/kernel_profile.hpp).
  sycl::queue make_pool_queue() const;

  // =========================================================================
  // Private members — declaration order matters for initialization
  // =========================================================================

  // sk_ctx_ and device_index_ must be declared before queue_pool_ so they are
  // initialized first (C++ initializes members in declaration order)
  std::shared_ptr<slabkit::Context> sk_ctx_;
  size_t device_index_;
  Phase phase_ = Phase::Declaring;
  std::vector<sycl::queue> queue_pool_;
  std::vector<size_t> component_layer_;
  std::vector<std::vector<Component *>> components_by_layer_;
  std::vector<size_t> component_queue_idx_;
  size_t max_layer_width_ = 0;

  // Tier-0.5 horizontal kernel fusion (core/fusion.hpp). Installed as the
  // active collector for the duration of forward()/backward(); arena is
  // allocated at finalize. Gated by NAGA_FUSION (=0 disables).
  fusion::HorizontalFuser fuser_;

  // Entry storage
  std::vector<Entry> entries_;
  std::unordered_map<std::string, size_t>
      entry_map_; // name → index in entries_

  // Memory ownership (type-erased: allocations are shape3_size × dtype_size
  // bytes; these vectors exist for lifetime only, never dereferenced)
  std::vector<std::shared_ptr<void>> buffers_;      // owns data allocations
  std::vector<std::shared_ptr<void>> grad_buffers_; // owns grad allocations

  // Component registry & execution order
  std::vector<ComponentNode> component_nodes_;
  std::vector<Component *> execution_order_;

  // Back-edge tracking (populated during finalize via identify_declared_cuts
  // and, for emergent cuts in registerless cycles, scc_dfs_order)
  std::vector<BackEdge> back_edges_;
  std::unordered_set<std::string> back_edge_source_entries_;

  // Slot-pair index bookkeeping for recurrent outputs. curr/prev name the
  // PRIMAL-time roles; in the adjoint the same two slots swap roles (the
  // owner consumes grad-curr, back-edge consumers accumulate grad-prev) —
  // rotation is the same mechanical index swap in both directions.
  struct SlotPair {
    size_t curr;
    size_t prev;
  };

  // Grad slot pairs for recurrent outputs (keyed by owner FQN; exactly 2
  // buffers each). Rotated once per adjoint iteration by advance_grad_slots.
  std::unordered_map<std::string, std::vector<std::shared_ptr<float>>>
      grad_slots_;
  std::unordered_map<std::string, SlotPair> grad_slot_idx_;

  // Primal slot pairs for recurrent outputs (keyed by owner FQN). Rotated
  // once per forward sweep by advance_primal_slots — independent of grad
  // rotation.
  std::unordered_map<std::string, std::vector<std::shared_ptr<float>>>
      primal_slots_;
  std::unordered_map<std::string, SlotPair> primal_slot_idx_;

  // Entry point tracking (registers — see mark_entry_point)
  std::unordered_set<std::string>
      entry_points_; // component names declared as entry points
  size_t K_backward_ = 2; // max backward iterations
  size_t current_backward_iteration_ = 0; // set during backward()
  std::vector<std::unordered_set<Component *>>
      backward_active_sets_; // [k] = active set for iteration k

  // Dummy float for null DeviceView construction (if needed)
  mutable float dummy_float_ = 0.0f;

  // Anderson Acceleration workspace (allocated at finalization if
  // anderson_depth_ > 0)
  size_t total_recurrent_size_ =
      0; // sum of shape3_size over non-target recurrent entries
  size_t anderson_depth_ = 5; // configurable before finalize(); default 5

  std::vector<std::shared_ptr<float>>
      anderson_history_; // length anderson_depth_; each size
                         // total_recurrent_size_
  std::vector<std::shared_ptr<float>>
      anderson_residuals_; // length anderson_depth_; each size
                           // total_recurrent_size_
  std::vector<std::shared_ptr<float>>
      anderson_iterates_; // length anderson_depth_; stores x_k (grad curr
                          // slot contents before backward)
  std::shared_ptr<float>
      anderson_flat_; // size total_recurrent_size_; working buffer
  std::shared_ptr<float>
      anderson_gram_; // size anderson_depth_ * anderson_depth_
  std::shared_ptr<float> anderson_coeffs_;   // size anderson_depth_
  std::shared_ptr<float> anderson_rhs_;      // size anderson_depth_; RHS for
                                             // Walker-Ni unconstrained solve
  std::shared_ptr<float> anderson_norm_acc_; // size 1; convergence accumulator

  // Perf (audit §4.2): batched fill tables for the hot per-adjoint-iteration
  // zero/seed paths. Span sets are static after finalize; built lazily on
  // first use. One kernel replaces ~O(entries) tiny fills.
  struct FillBatch {
    std::shared_ptr<float *> ptrs;  // device array: span base pointers [n]
    std::shared_ptr<size_t> prefix; // device prefix sums [n+1]
    size_t n = 0;                   // span count
    size_t total = 0;               // total elements
    bool built = false;
  };
  FillBatch znr_batch_;  // zero_non_recurrent_grads spans
  FillBatch seed_batch_; // seed_objective_grad spans
  void build_fill_batch(FillBatch &batch,
                        const std::vector<std::pair<float *, size_t>> &spans);
  void run_fill_batch(FillBatch &batch, float value);

  size_t last_adj_iters_ =
      0; // set by backward_to_equilibrium(), read by last_adj_iters()

  std::vector<std::string>
      recurrent_fqns_cache_; // populated at finalization, consistent order

}; // class Context

} // namespace naga::cospan::core
