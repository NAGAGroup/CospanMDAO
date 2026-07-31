#include "naga/core/context.hpp"
#include <cassert>
#include <iostream>

namespace naga::core {

void Context::finalize() {
  // Step (a): Validate phase
  assert_phase(Phase::Declaring, "finalize()");

  // Step (b): Validate entry points
  validate_entry_points();

  // Step (c): Build component graph
  ComponentGraph g = build_component_graph();

  // Step (d): Cut declared registers. Every connection into a marked entry
  // point is cut up front: registered as a back-edge (recurrence carrier)
  // with its target entry flagged is_deferred_read.
  auto cuts = identify_declared_cuts(g);

  // Step (e): Compute SCCs on the graph minus declared cuts. Cycles through
  // entry points dissolve here; any residual non-trivial SCC is a
  // registerless cycle (emergent cut — warned about in compute_dag_and_order).
  auto sccs = compute_sccs(g, cuts);

  // Step (f): Compute DAG execution order via SCC condensation + Kahn's
  compute_dag_and_order(g, sccs, cuts);

  // Step (f1): Assign component layers and grow queue pool
  assign_component_layers(g);

  // Grow queue pool to max_layer_width_. queue_pool_[0] already exists from the
  // constructor.
  while (queue_pool_.size() < max_layer_width_)
    queue_pool_.push_back(make_pool_queue());

  assert(queue_pool_.size() >= max_layer_width_ && max_layer_width_ > 0);

  // Tier-0.5 horizontal fusion: allocate the device param arena now that the
  // queue pool exists (no-op when NAGA_FUSION=0).
  fuser_.init(sk_ctx_, device_index_);

  // Step (f2): Log the recurrent-output set. Recurrent outputs are marked
  // ONLY by register_back_edge (the owning source of a cut connection,
  // declared or emergent) — plus StateVar-kind entries, which are recurrent
  // outputs by construction. SCC "passengers" (other entries of components
  // inside a cycle) are single-slot: their reads are fresh, their grads are
  // recomputed within each adjoint iteration from the transported seed
  // (prev/curr spec, 2026-07-20 — supersedes the old blanket SCC marking).
  {
    std::cerr << "[RECURRENT] ";
    for (const auto &entry : entries_)
      if (entry.is_recurrent_output ||
          (entry.kind == EntryKind::StateVar && !entry.is_target))
        std::cerr << entry.name << " ";
    std::cerr << "\n";
  }

  // Step (g): Compute backward active sets
  compute_backward_active_sets(g);

  // Step (h): Allocate data buffers for non-target entries
  for (auto &entry : entries_) {
    if (!entry.is_target) {
      size_t count = shape3_size(entry.shape3);

      // StateVar always gets the primal slot pair (it IS the degenerate
      // back-edge); other entries get it only when marked recurrent outputs
      // by register_back_edge.
      if (!entry.is_design_variable && (entry.is_recurrent_output ||
                                        entry.kind == EntryKind::StateVar)) {
        // Slot-pair entries are f32-only in P1 (declare_state_var enforces
        // this; register_back_edge guards cut sources) — defense in depth.
        if (entry.dtype != DType::f32)
          throw std::runtime_error(
              "Context::finalize(): entry '" + entry.name +
              "' requires a primal slot pair but has dtype " +
              dtype_name(entry.dtype) +
              " — recurrent outputs are f32-only in P1");
        // Allocate the 2 primal buffers. Initial index roles {curr=1,
        // prev=0}: the executor rotates at the top of the first sweep, so
        // the first sweep writes slot 0.
        std::vector<std::shared_ptr<float>> dslots;
        for (size_t s = 0; s < 2; ++s) {
          auto d = slabkit::malloc_device<float>(count, sk_ctx_, device_index_);
          queue_pool_[0].fill(d.get(), 0.0f, count);
          dslots.push_back(std::shared_ptr<float>(std::move(d)));
        }
        entry.data = dslots[0].get();
        primal_slots_[entry.name] = std::move(dslots);
        primal_slot_idx_[entry.name] = {/*curr=*/1, /*prev=*/0};
        // Do NOT add to buffers_ — owned by primal_slots_
      } else if (entry.dtype == DType::f32) {
        // f32 path unchanged (same pool call, same behavior as pre-dtype).
        auto unique =
            slabkit::malloc_device<float>(count, sk_ctx_, device_index_);
        entry.data = unique.get();
        buffers_.push_back(std::shared_ptr<float>(std::move(unique)));
      } else {
        // Type-erased allocation: count × dtype_size bytes, zero-initialized
        // (data-tier entries have no init()-driven seeding contract yet, and
        // slabkit device memory is uninitialized). The base address is stored
        // type-erased in entry.data; typed interpretation goes through the
        // dtype-checked accessors.
        const size_t bytes = count * dtype_size(entry.dtype);
        auto unique =
            slabkit::malloc_device<uint8_t>(bytes, sk_ctx_, device_index_);
        queue_pool_[0].memset(unique.get(), 0, bytes);
        entry.data = reinterpret_cast<float *>(unique.get());
        buffers_.push_back(std::shared_ptr<void>(std::move(unique)));
      }
    }
  }

  // Ensure all fill calls on new data slots are complete
  queue_pool_[0].wait();

  // Step (i): Allocate grad buffers
  // Rule: Every DIFFERENTIABLE-tier f32 entry (input and output) gets a grad
  // buffer. Exceptions: connection targets (aliased to the source's data+grad
  // in step (k), so they must not allocate here) and non-f32 entries — the
  // data tier never carries gradients, and f64 gradient support is pending
  // the P2 io dtyping. Their grad stays nullptr; all zeroing/adjoint paths
  // skip null grads.
  for (auto &entry : entries_) {
    if (entry.is_target)
      continue;
    if (entry.dtype != DType::f32)
      continue;
    size_t count = shape3_size(entry.shape3);

    // StateVar and recurrent outputs get the grad slot pair for the adjoint
    // transport (curr = consumed this iteration; prev = back-edge
    // accumulation for the next iteration).
    // DVs are never recurrent (guarded in register_back_edge), so they
    // always fall through to the single-slot path below.
    if (entry.is_recurrent_output || entry.kind == EntryKind::StateVar) {
      // Exactly 2 slots. K_backward_ controls how many backward active sets
      // run per backward() call — independent of the slot count.
      size_t num_slots = 2;
      std::vector<std::shared_ptr<float>> gslots;
      for (size_t k = 0; k < num_slots; ++k) {
        auto g = slabkit::malloc_device<float>(count, sk_ctx_, device_index_);
        queue_pool_[0].fill(
            g.get(), 0.0f,
            count); // explicit zero (malloc_device is uninitialized)
        gslots.push_back(std::shared_ptr<float>(std::move(g)));
      }
      entry.grad = gslots[0].get();
      grad_buffers_.push_back(
          gslots[0]); // retain ownership via grad_buffers_ too
      grad_slots_[entry.name] = std::move(gslots);

      // {curr=0, prev=1} — reset to the same roles by zero_non_dv_grads()
      grad_slot_idx_[entry.name] = {/*curr=*/0, /*prev=*/1};
    } else {
      // Existing single-slot allocation unchanged
      auto grad_unique =
          slabkit::malloc_device<float>(count, sk_ctx_, device_index_);
      entry.grad = grad_unique.get();
      grad_buffers_.push_back(std::shared_ptr<float>(std::move(grad_unique)));
    }
  }

  // Ensure all fill calls above are complete before continuing
  queue_pool_[0].wait();

  // Cache recurrent FQNs in a consistent order for Anderson scatter/gather
  recurrent_fqns_cache_ =
      recurrent_entries(); // recurrent outputs (+ StateVars), !is_target

  // Cache total adjoint state size
  total_recurrent_size_ = 0;
  for (const auto &fqn : recurrent_fqns_cache_) {
    total_recurrent_size_ += shape3_size(get_entry(fqn).shape3);
  }

  // Always allocate norm accumulator when there are recurrent entries
  // (needed by backward_to_equilibrium_compute_norm() regardless of Anderson
  // depth)
  if (total_recurrent_size_ > 0) {
    auto &q = queue_pool_[0];
    auto na = slabkit::malloc_device<float>(1, sk_ctx_, device_index_);
    q.fill(na.get(), 0.0f, 1);
    anderson_norm_acc_ = std::shared_ptr<float>(std::move(na));
    q.wait();
  }

  if (anderson_depth_ > 0 && total_recurrent_size_ > 0) {
    auto &q = queue_pool_[0];
    size_t n = total_recurrent_size_;
    size_t m = anderson_depth_;

    anderson_history_.resize(m);
    anderson_residuals_.resize(m);
    anderson_iterates_.resize(m);
    for (size_t i = 0; i < m; ++i) {
      auto h = slabkit::malloc_device<float>(n, sk_ctx_, device_index_);
      q.fill(h.get(), 0.0f, n);
      anderson_history_[i] = std::shared_ptr<float>(std::move(h));

      auto r = slabkit::malloc_device<float>(n, sk_ctx_, device_index_);
      q.fill(r.get(), 0.0f, n);
      anderson_residuals_[i] = std::shared_ptr<float>(std::move(r));

      auto it = slabkit::malloc_device<float>(n, sk_ctx_, device_index_);
      q.fill(it.get(), 0.0f, n);
      anderson_iterates_[i] = std::shared_ptr<float>(std::move(it));
    }

    {
      auto flat = slabkit::malloc_device<float>(n, sk_ctx_, device_index_);
      q.fill(flat.get(), 0.0f, n);
      anderson_flat_ = std::shared_ptr<float>(std::move(flat));
    }
    {
      auto gram = slabkit::malloc_device<float>(m * m, sk_ctx_, device_index_);
      q.fill(gram.get(), 0.0f, m * m);
      anderson_gram_ = std::shared_ptr<float>(std::move(gram));
    }
    {
      auto coeffs = slabkit::malloc_device<float>(m, sk_ctx_, device_index_);
      q.fill(coeffs.get(), 0.0f, m);
      anderson_coeffs_ = std::shared_ptr<float>(std::move(coeffs));
    }
    {
      auto rhs = slabkit::malloc_device<float>(m, sk_ctx_, device_index_);
      q.fill(rhs.get(), 0.0f, m);
      anderson_rhs_ = std::shared_ptr<float>(std::move(rhs));
    }
    q.wait();
  }

  // Step (k): For each connection, alias target's data and grad to source
  for (auto &entry : entries_) {
    if ((entry.kind == EntryKind::Output ||
         entry.kind == EntryKind::PassthroughVar ||
         entry.kind == EntryKind::Promotion ||
         entry.kind ==
             EntryKind::StateVar) // InternalBuffer intentionally excluded
        && !entry.targets.empty()) {
      Entry *source = &entry;
      for (const auto &target_name : entry.targets) {
        Entry *target = find_entry(target_name);
        if (target) {
          // Alias data and grad pointers
          target->data = source->data;
          target->grad = source->grad;
          // Downstream targets of StateVars/recurrent outputs inherit the
          // mark for introspection; runtime slot routing always resolves
          // through get_owning_entry, so this is bookkeeping only.
          if (source->is_recurrent_output || source->kind == EntryKind::StateVar)
              target->is_recurrent_output = true;
        }
      }
    }
  }

  // Step (l): Transition to Finalized phase
  phase_ = Phase::Finalized;
}

void Context::validate_entry_points() {
  if (entry_points_.empty())
    throw std::runtime_error("Context::finalize: no entry points declared. "
                             "Call System::add_entry_point() to mark at least "
                             "one component as an entry point.");

  // Exit points no longer exist: the backward active set is seeded from
  // objective-owning components. An empty objective set simply yields an
  // empty backward active set — a forward-only context is valid.
}

} // namespace naga::core
