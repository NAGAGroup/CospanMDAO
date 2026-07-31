#include "naga/core/context.hpp"

#include <unordered_set>

namespace naga::core {

void Context::zero_non_dv_grads() {
    assert_phase(Phase::Finalized, "zero_non_dv_grads()");
    for (auto& entry : entries_) {
        if (entry.is_design_variable || entry.is_target) continue;
        size_t count = shape3_size(entry.shape3);

        if (entry.is_recurrent_output || entry.kind == EntryKind::StateVar) {
            // Zero both slots and reset the roles to the initial state
            for (auto& slot : grad_slots_[entry.name])
                queue_pool_[0].fill(slot.get(), 0.0f, count);
            grad_slot_idx_[entry.name] = {/*curr=*/0, /*prev=*/1};
        } else {
            if (!entry.grad) continue;  // e.g. zero-size hiddens buffer
            queue_pool_[0].fill(entry.grad, 0.0f, count);
        }
    }
    queue_pool_[0].wait();
}

void Context::zero_dv_grads() {
    assert_phase(Phase::Finalized, "zero_dv_grads()");
    for (auto& entry : entries_) {
        if (entry.is_design_variable) {
            if (!entry.grad) continue;  // data-tier entries carry no grads
            size_t count = shape3_size(entry.shape3);
            queue_pool_[0].fill(entry.grad, 0.0f, count);
        }
    }
    queue_pool_[0].wait();
}

// Perf (audit §4.2): build the static device-side span table once. Called
// lazily post-finalize; the entry sets and grad pointers involved are static
// (non-recurrent grads never rotate slots).
void Context::build_fill_batch(FillBatch& batch,
                               const std::vector<std::pair<float*, size_t>>& spans) {
    batch.n = spans.size();
    batch.total = 0;
    std::vector<float*> h_ptrs(batch.n);
    std::vector<size_t> h_prefix(batch.n + 1);
    for (size_t i = 0; i < batch.n; ++i) {
        h_ptrs[i]   = spans[i].first;
        h_prefix[i] = batch.total;
        batch.total += spans[i].second;
    }
    h_prefix[batch.n] = batch.total;

    batch.ptrs = std::shared_ptr<float*>(
        std::move(slabkit::malloc_device<float*>(batch.n, sk_ctx_shared(), device_index_)));
    batch.prefix = std::shared_ptr<size_t>(
        std::move(slabkit::malloc_device<size_t>(batch.n + 1, sk_ctx_shared(), device_index_)));
    queue_pool_[0].memcpy(batch.ptrs.get(), h_ptrs.data(), batch.n * sizeof(float*));
    queue_pool_[0].memcpy(batch.prefix.get(), h_prefix.data(), (batch.n + 1) * sizeof(size_t));
    queue_pool_[0].wait();
    batch.built = true;
}

// One kernel over the whole span set: flat index → span via binary search on
// the prefix table (n ~O(10³) → ~10 probes), then ptr[local] = value.
void Context::run_fill_batch(FillBatch& batch, float value) {
    if (batch.total == 0) return;
    float** ptrs   = batch.ptrs.get();
    size_t* prefix = batch.prefix.get();
    const size_t n = batch.n;
    queue_pool_[0].parallel_for(sycl::range<1>{batch.total}, [=](sycl::id<1> idx) {
        const size_t g = idx[0];
        // upper_bound(prefix, g) - 1
        size_t lo = 0, hi = n;
        while (lo + 1 < hi) {
            const size_t mid = (lo + hi) / 2;
            if (prefix[mid] <= g) lo = mid; else hi = mid;
        }
        ptrs[lo][g - prefix[lo]] = value;
    });
    queue_pool_[0].wait();
}

void Context::zero_non_recurrent_grads() {
    assert_phase(Phase::Finalized, "zero_non_recurrent_grads()");
    if (!znr_batch_.built) {
        std::vector<std::pair<float*, size_t>> spans;
        std::unordered_set<float*> seen;  // aliased targets share grad pointers
        for (auto& entry : entries_) {
            if (entry.is_target) continue;

            // Skip recurrent outputs and StateVar entries — their grad slot
            // pairs carry the adjoint transport iterate t_k.
            if (entry.is_recurrent_output || entry.kind == EntryKind::StateVar) continue;

            // Skip DVs: under the transport-form adjoint (2026-07-19) DV
            // grads ACCUMULATE across adjoint iterations — the total
            // Σ_k (∂F/∂θ)ᵀ·t_k is the Neumann series. They are zeroed once
            // per settle by zero_dv_grads().
            if (entry.is_design_variable) continue;

            // Everything else is zeroed EVERY iteration and recomputed from
            // the transported seed within the iteration. (The old
            // feeds_recurrent skip — protecting buffers aliased toward
            // recurrent components — is dead under the prev/curr spec
            // (2026-07-20): fresh gradient paths now complete within one
            // iteration, so letting per-iteration contributions persist in
            // any single-slot buffer would double-count the transport.)

            // Objectives are NOT skipped: their grads must be zero on
            // transport iterations (k >= 1) so chain-rule-honest losses
            // contribute nothing after the single seeding pass. The seed
            // (1.0) is re-applied by seed_objective_grad() on iteration 0
            // only.

            if (!entry.grad) continue;
            if (!seen.insert(entry.grad).second) continue;
            spans.emplace_back(entry.grad, shape3_size(entry.shape3));
        }
        build_fill_batch(znr_batch_, spans);
    }
    run_fill_batch(znr_batch_, 0.0f);
}

void Context::seed_objective_grad() {
    assert_phase(Phase::Finalized, "seed_objective_grad()");
    if (!seed_batch_.built) {
        std::vector<std::pair<float*, size_t>> spans;
        for (auto& entry : entries_) {
            if (entry.is_objective && entry.grad)
                spans.emplace_back(entry.grad, shape3_size(entry.shape3));
        }
        build_fill_batch(seed_batch_, spans);
    }
    run_fill_batch(seed_batch_, 1.0f);
}

} // namespace naga::core
