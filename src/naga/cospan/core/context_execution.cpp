#include "naga/cospan/core/context.hpp"
#include "naga/cospan/core/component.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <utility>

namespace naga::cospan::core {

void Context::forward() {
  assert_phase(Phase::Finalized, "forward()");
  // No primal zeroing of any kind: outputs are functions of inputs — every
  // element is accounted for by its owning component each sweep (share
  // members plain-store their disjoint slices; StateVars fully rewrite
  // current()). Accumulation shares are dead (2026-07-20).
  //
  // Slot rotation is the caller's responsibility (advance_primal_slots at
  // the TOP of each sweep — see EquilibriumExecutor::settle).
  //
  // Tier-0.5 horizontal fusion: while the FuserScope is installed, component
  // kernel submissions (execute_compute) are collected instead of launched;
  // same-layer basic-tier kernels flush as fused waves on queue_pool_[0]
  // (core/fusion.hpp). Disabled (NAGA_FUSION=0) this is the historical path.
  const bool fuse = fusion::HorizontalFuser::enabled();
  fusion::FuserScope fuser_scope(fuser_);
  if (fuse)
    fuser_.begin_pass();
  for (auto &layer : components_by_layer_) {
    if (fuse)
      fuser_.begin_layer(queue_pool_[0]);
    for (size_t pos = 0; pos < layer.size(); ++pos) {
      if (fuse)
        fuser_.begin_component(queue_pool_[pos]);
      layer[pos]->forward(*this, queue_pool_[pos]);
      if (fuse)
        fuser_.end_component();
    }
    if (fuse)
      fuser_.end_layer();
    for (size_t i = 0; i < layer.size(); ++i)
      queue_pool_[i].wait();
  }
}

void Context::backward() {
  assert_phase(Phase::Finalized, "backward()");
  // Slot rotation is the caller's responsibility — advance_grad_slots
  // should be called AFTER computing and checking convergence, not before.
  const bool fuse = fusion::HorizontalFuser::enabled();
  fusion::FuserScope fuser_scope(fuser_);
  if (fuse)
    fuser_.begin_pass();
  for (size_t k = 0; k < K_backward_; ++k) {
    if (backward_active_sets_[k].empty())
      continue;
    current_backward_iteration_ = k;
    const auto &active = backward_active_sets_[k];
    for (int li = (int)components_by_layer_.size() - 1; li >= 0; --li) {
      auto &layer = components_by_layer_[li];
      if (fuse)
        fuser_.begin_layer(queue_pool_[0]);
      for (size_t pos = 0; pos < layer.size(); ++pos) {
        if (active.count(layer[pos])) {
          if (fuse)
            fuser_.begin_component(queue_pool_[pos]);
          layer[pos]->backward(*this, queue_pool_[pos]);
          if (fuse)
            fuser_.end_component();
        }
      }
      if (fuse)
        fuser_.end_layer();
      // Wait only on queues that had work submitted this layer/K-phase.
      // Safe because queue assignment is static (component position within
      // layer → queue index), so an inactive component never submits to its
      // queue slot. Fused waves land on queue_pool_[0], which must therefore
      // also be waited when fusion is active.
      for (size_t pos = 0; pos < layer.size(); ++pos) {
        if (active.count(layer[pos]))
          queue_pool_[pos].wait();
      }
      if (fuse)
        queue_pool_[0].wait();
    }
    // Advance grad slots between non-empty K iterations (not after the last)
    bool has_next = false;
    for (size_t j = k + 1; j < K_backward_; ++j) {
      if (!backward_active_sets_[j].empty()) {
        has_next = true;
        break;
      }
    }
    if (has_next)
      advance_grad_slots();
  }
  current_backward_iteration_ = 0;
}

void Context::init_components() {
  // Reverse layer order: components at layer 0 (sources) initialize last,
  // winning over downstream components that may have written defaults to shared
  // buffers. This mirrors the original reverse-execution-order semantics.
  for (int li = (int)components_by_layer_.size() - 1; li >= 0; --li) {
    auto &layer = components_by_layer_[li];
    for (size_t pos = 0; pos < layer.size(); ++pos)
      layer[pos]->init(*this, queue_pool_[pos]);
    for (size_t i = 0; i < layer.size(); ++i)
      queue_pool_[i].wait();
  }
}

void Context::reset_components() {
  // Reverse layer order: components at layer 0 (sources) initialize last,
  // winning over downstream components that may have written defaults to shared
  // buffers. This mirrors the original reverse-execution-order semantics.
  for (int li = (int)components_by_layer_.size() - 1; li >= 0; --li) {
    auto &layer = components_by_layer_[li];
    for (size_t pos = 0; pos < layer.size(); ++pos)
      layer[pos]->reset(*this, queue_pool_[pos]);
    for (size_t i = 0; i < layer.size(); ++i)
      queue_pool_[i].wait();
  }
}

void Context::advance_grad_slots() {
  // Adjoint rotation: prev becomes curr (the transported iterate t_{k+1}
  // arrives for consumption), and the NEW prev slot — just consumed as curr —
  // is zeroed so this iteration's back-edge consumers accumulate cleanly.
  for (auto &[fqn, idx] : grad_slot_idx_) {
    std::swap(idx.curr, idx.prev);

    auto it_g = grad_slots_.find(fqn);
    if (it_g != grad_slots_.end()) {
      auto eit = entry_map_.find(fqn);
      if (eit != entry_map_.end()) {
        size_t count = shape3_size(entries_[eit->second].shape3);
        queue_pool_[0].fill(it_g->second[idx.prev].get(), 0.0f, count);
      }
    }
  }
  queue_pool_[0].wait();
}

void Context::advance_primal_slots() {
  // Primal rotation: curr becomes prev. NO zeroing — outputs are functions
  // of inputs; the owning component fully rewrites curr each sweep and must
  // make no assumptions about output state at sweep start.
  for (auto &[fqn, idx] : primal_slot_idx_) {
    std::swap(idx.curr, idx.prev);
  }
}

void Context::set_anderson_depth(size_t depth) {
  assert(depth <= 5 || depth == 0);
  anderson_depth_ = depth;
}

size_t Context::last_adj_iters() const { return last_adj_iters_; }

void Context::backward_single_pass() {
  assert_phase(Phase::Finalized, "backward_single_pass()");
  // Slot advancement is the caller's responsibility.
  // Run exactly the k=0 active set backward pass
  if (!backward_active_sets_.empty() && !backward_active_sets_[0].empty()) {
    current_backward_iteration_ = 0;
    const auto &active = backward_active_sets_[0];
    const bool fuse = fusion::HorizontalFuser::enabled();
    fusion::FuserScope fuser_scope(fuser_);
    if (fuse)
      fuser_.begin_pass();
    for (int li = (int)components_by_layer_.size() - 1; li >= 0; --li) {
      auto &layer = components_by_layer_[li];
      if (fuse)
        fuser_.begin_layer(queue_pool_[0]);
      for (size_t pos = 0; pos < layer.size(); ++pos) {
        if (active.count(layer[pos])) {
          if (fuse)
            fuser_.begin_component(queue_pool_[pos]);
          layer[pos]->backward(*this, queue_pool_[pos]);
          if (fuse)
            fuser_.end_component();
        }
      }
      if (fuse)
        fuser_.end_layer();
      for (size_t pos = 0; pos < layer.size(); ++pos) {
        if (active.count(layer[pos]))
          queue_pool_[pos].wait();
      }
      if (fuse)
        queue_pool_[0].wait();
    }
  }
  current_backward_iteration_ = 0;
}

void Context::scatter_recurrent_write_to_flat() {
  float *flat = anderson_flat_.get();
  size_t offset = 0;
  auto &q = queue_pool_[0];
  for (const auto &fqn : recurrent_fqns_cache_) {
    size_t count = shape3_size(get_entry(fqn).shape3);
    float *wsrc = grad_prev_slot(fqn);
    float *dst = flat + offset;
    q.parallel_for(sycl::range<1>{count},
                   [=](sycl::id<1> id) { dst[id[0]] = wsrc[id[0]]; });
    offset += count;
  }
  q.wait();
}

void Context::scatter_flat_to_recurrent_write() {
  float *flat = anderson_flat_.get();
  size_t offset = 0;
  auto &q = queue_pool_[0];
  for (const auto &fqn : recurrent_fqns_cache_) {
    size_t count = shape3_size(get_entry(fqn).shape3);
    float *wdst = grad_prev_slot(fqn);
    float *src = flat + offset;
    q.parallel_for(sycl::range<1>{count},
                   [=](sycl::id<1> id) { wdst[id[0]] = src[id[0]]; });
    offset += count;
  }
  q.wait();
}

void Context::scatter_recurrent_write_to_iterates(size_t ring_slot) {
  float *dst = anderson_iterates_[ring_slot].get();
  size_t offset = 0;
  auto &q = queue_pool_[0];
  for (const auto &fqn : recurrent_fqns_cache_) {
    size_t count = shape3_size(get_entry(fqn).shape3);
    // x_k is the iterate the upcoming backward pass CONSUMES — the grad
    // curr slot (t_k, possibly Anderson-blended last iteration). The prev
    // slot was just zeroed by the advance at the top of the iteration;
    // capturing it recorded x_k ≡ 0, making the DIIS residual f_k = F_k − 0
    // = F_k — i.e. Anderson minimized the blended ITERATE norm, not the
    // residual (audit 2026-07-11).
    float *rsrc = grad_curr_slot(fqn);
    float *d = dst + offset;
    q.parallel_for(sycl::range<1>{count},
                   [=](sycl::id<1> id) { d[id[0]] = rsrc[id[0]]; });
    offset += count;
  }
  q.wait();
}

void Context::anderson_compute_residual_and_store(size_t ring_slot) {
  size_t n = total_recurrent_size_;

  float *cur = anderson_flat_.get(); // F_k = g(x_k)
  float *x_k =
      anderson_iterates_[ring_slot].get(); // x_k (captured before backward)
  float *h_k = anderson_history_[ring_slot].get(); // will store F_k
  float *r_k =
      anderson_residuals_[ring_slot].get(); // will store f_k = F_k - x_k

  auto &q = queue_pool_[0];
  q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> id) {
    size_t i = id[0];
    h_k[i] = cur[i];
    r_k[i] = cur[i] - x_k[i];
  });
  q.wait();
}

void Context::anderson_update_gram_column(size_t h, size_t new_col_ring_slot) {
  size_t m = anderson_depth_;
  size_t n = total_recurrent_size_;
  float *gram = anderson_gram_.get();
  auto &q = queue_pool_[0];
  constexpr size_t kWG = 256;

  float *r_new = anderson_residuals_[new_col_ring_slot].get();

  for (size_t col_idx = 0; col_idx < h; ++col_idx) {
    size_t slot_col = (new_col_ring_slot + m - (h - 1) + col_idx) % m;
    float *r_col = anderson_residuals_[slot_col].get();

    float *acc = gram + col_idx * m + (h - 1);
    float *acc_sym = gram + (h - 1) * m + col_idx;

    q.fill(acc, 0.0f, 1).wait();

    size_t rounded = ((n + kWG - 1) / kWG) * kWG;
    q.parallel_for(sycl::nd_range<1>{rounded, kWG}, [=](sycl::nd_item<1> it) {
      size_t gid = it.get_global_id(0);
      float val = 0.0f;
      if (gid < n) {
        val = r_col[gid] * r_new[gid]; // <f_col, f_new> — direct inner product
      }
      float wg_sum =
          sycl::reduce_over_group(it.get_group(), val, sycl::plus<float>());
      if (it.get_local_id(0) == 0) {
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(*acc);
        ref.fetch_add(wg_sum);
      }
    });
    q.wait();

    if (col_idx != h - 1) {
      q.memcpy(acc_sym, acc, sizeof(float)).wait();
    }
  }
}

void Context::anderson_recompute_gram(size_t h, size_t newest_ring_slot) {
  size_t m = anderson_depth_;
  auto &q = queue_pool_[0];
  q.fill(anderson_gram_.get(), 0.0f, m * m).wait();
  // Recompute columns 0..h-1 as if building fresh
  // Column j corresponds to ring slot: (newest_ring_slot + m - (h-1) + j) % m
  for (size_t col = 0; col < h; ++col) {
    // Temporarily treat col as the "new column" for a window of size col+1
    size_t col_ring = (newest_ring_slot + m - (h - 1) + col) % m;
    anderson_update_gram_column(col + 1, col_ring);
  }
}

void Context::anderson_solve_and_apply_blend(size_t h,
                                             size_t newest_ring_slot) {
  size_t m = anderson_depth_;
  size_t n = total_recurrent_size_;

  float *gram = anderson_gram_.get();
  float *coeffs = anderson_coeffs_.get();
  float *flat = anderson_flat_.get();
  auto &q = queue_pool_[0];

  // DIIS augmented solve: (h+1)×(h+1) system
  // [G  1][θ]   [0]
  // [1T 0][λ] = [1]
  // G is the DIIS Gram matrix: G[i,j] = <f_i, f_j>
  // h <= anderson_depth_ = 5, so augmented system is at most 6×6

  q.single_task([=]() {
    // Build augmented system of size (h+1)×(h+1)
    float A[36]; // (anderson_depth_+1)^2 = 6^2 = 36 max
    float b[6];  // (anderson_depth_+1) = 6 max
    size_t hsz = h + 1;

    // Fill G block from gram (rows 0..h-1, cols 0..h-1)
    for (size_t i = 0; i < h; ++i)
      for (size_t j = 0; j < h; ++j)
        A[i * hsz + j] = gram[i * m + j];

    // Fill constraint row/col: A[i,h] = A[h,i] = 1, A[h,h] = 0
    for (size_t i = 0; i < h; ++i) {
      A[i * hsz + h] = 1.0f;
      A[h * hsz + i] = 1.0f;
    }
    A[h * hsz + h] = 0.0f;

    // RHS: b[0..h-1] = 0, b[h] = 1
    for (size_t i = 0; i < h; ++i)
      b[i] = 0.0f;
    b[h] = 1.0f;

    // Relative regularization on G block diagonal (not the constraint row/col)
    float max_diag = 0.0f;
    for (size_t i = 0; i < h; ++i) {
      float d = A[i * hsz + i];
      if (d < 0.f)
        d = -d;
      if (d > max_diag)
        max_diag = d;
    }
    if (max_diag < 1e-30f)
      max_diag = 1.0f;
    float reg = 1e-4f * max_diag;
    for (size_t i = 0; i < h; ++i)
      A[i * hsz + i] += reg;

    // Gaussian elimination with partial pivoting on hsz×hsz system
    for (size_t col = 0; col < hsz; ++col) {
      size_t pivot = col;
      float best =
          A[col * hsz + col] < 0.f ? -A[col * hsz + col] : A[col * hsz + col];
      for (size_t r = col + 1; r < hsz; ++r) {
        float v = A[r * hsz + col] < 0.f ? -A[r * hsz + col] : A[r * hsz + col];
        if (v > best) {
          best = v;
          pivot = r;
        }
      }
      for (size_t j = col; j < hsz; ++j) {
        float tmp = A[col * hsz + j];
        A[col * hsz + j] = A[pivot * hsz + j];
        A[pivot * hsz + j] = tmp;
      }
      {
        float tmp = b[col];
        b[col] = b[pivot];
        b[pivot] = tmp;
      }

      float pval = A[col * hsz + col];
      float abs_pval = pval < 0.f ? -pval : pval;
      float rel_tol = max_diag > 0.f ? max_diag : 1.0f;
      if (abs_pval < 1e-8f * rel_tol) {
        // Ill-conditioned: uniform fallback
        float unif = 1.0f / (float)h;
        for (size_t i = 0; i < h; ++i)
          coeffs[i] = unif;
        return;
      }
      for (size_t r = col + 1; r < hsz; ++r) {
        float factor = A[r * hsz + col] / pval;
        for (size_t j = col; j < hsz; ++j)
          A[r * hsz + j] -= factor * A[col * hsz + j];
        b[r] -= factor * b[col];
      }
    }
    // Back-substitution
    for (int r = (int)hsz - 1; r >= 0; --r) {
      for (size_t c = (size_t)(r + 1); c < hsz; ++c)
        b[r] -= A[r * hsz + c] * b[c];
      float piv = A[r * hsz + r];
      float abs_piv = piv < 0.f ? -piv : piv;
      float rel_tol2 = max_diag > 0.f ? max_diag : 1.0f;
      if (abs_piv < 1e-8f * rel_tol2) {
        float unif = 1.0f / (float)h;
        for (size_t i = 0; i < h; ++i)
          coeffs[i] = unif;
        return;
      }
      b[r] /= piv;
    }
    // b[0..h-1] are the θ_j coefficients; b[h] is the Lagrange multiplier
    // (discard)
    for (size_t i = 0; i < h; ++i)
      coeffs[i] = b[i];
  });
  q.wait();

  // Apply blend: flat = Σ_j θ_j * history[slot_j]
  for (size_t j = 0; j < h; ++j) {
    size_t slot_j = (newest_ring_slot + m - (h - 1) + j) % m;
    float *hist_j = anderson_history_[slot_j].get();
    float *co = coeffs;
    size_t jj = j;
    if (j == 0) {
      q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> id) {
        flat[id[0]] = co[jj] * hist_j[id[0]];
      });
    } else {
      q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> id) {
        flat[id[0]] += co[jj] * hist_j[id[0]];
      });
    }
    q.wait();
  }
}

float Context::backward_to_equilibrium_compute_norm(std::string &worst_out) {
  worst_out.clear();
  if (total_recurrent_size_ == 0 || !anderson_norm_acc_)
    return 0.0f;

  auto &q = queue_pool_[0];
  float *acc = anderson_norm_acc_.get();
  float global_max = 0.0f;

  constexpr size_t kWG = 256;
  for (const auto &fqn : recurrent_fqns_cache_) {
    size_t count = shape3_size(get_entry(fqn).shape3);
    float *wslot = grad_prev_slot(fqn);
    float *rslot = grad_curr_slot(fqn);

    q.fill(acc, 0.0f, 1).wait();

    size_t rounded = ((count + kWG - 1) / kWG) * kWG;
    q.parallel_for(sycl::nd_range<1>{rounded, kWG}, [=](sycl::nd_item<1> it) {
      size_t gid = it.get_global_id(0);
      float val = (gid < count) ? sycl::fabs(wslot[gid] - rslot[gid]) : 0.0f;
      float wg_max =
          sycl::reduce_over_group(it.get_group(), val, sycl::maximum<float>());
      if (it.get_local_id(0) == 0) {
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(*acc);
        ref.fetch_max(wg_max);
      }
    });
    q.wait();

    float entry_max = 0.0f;
    q.memcpy(&entry_max, acc, sizeof(float)).wait();

    if (entry_max > global_max) {
      global_max = entry_max;
      worst_out = fqn;
    }
  }

  return global_max;
}

float Context::backward_to_equilibrium_compute_scale(std::string &worst_out) {
  worst_out.clear();
  if (total_recurrent_size_ == 0 || !anderson_norm_acc_)
    return 0.0f;

  auto &q = queue_pool_[0];
  float *acc = anderson_norm_acc_.get();
  float global_max = 0.0f;

  constexpr size_t kWG = 256;
  for (const auto &fqn : recurrent_fqns_cache_) {
    size_t count = shape3_size(get_entry(fqn).shape3);
    float *wslot = grad_prev_slot(fqn);

    q.fill(acc, 0.0f, 1).wait();

    size_t rounded = ((count + kWG - 1) / kWG) * kWG;
    q.parallel_for(sycl::nd_range<1>{rounded, kWG}, [=](sycl::nd_item<1> it) {
      size_t gid = it.get_global_id(0);
      float val = (gid < count) ? sycl::fabs(wslot[gid]) : 0.0f;
      float wg_max =
          sycl::reduce_over_group(it.get_group(), val, sycl::maximum<float>());
      if (it.get_local_id(0) == 0) {
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(*acc);
        ref.fetch_max(wg_max);
      }
    });
    q.wait();

    float entry_max = 0.0f;
    q.memcpy(&entry_max, acc, sizeof(float)).wait();

    if (entry_max > global_max) {
      global_max = entry_max;
      worst_out = fqn;
    }
  }

  return global_max;
}

Context::AdjEquilibriumResult
Context::backward_to_equilibrium(float adj_tol, size_t max_adj_iters,
                                 size_t check_interval, float adj_clip_norm) {
  // depth >= 2 required: the blend needs h >= 2 histories, so depth 1 pays the
  // full scatter/residual/Gram pipeline per iteration and can never blend
  // (audit 2026-07-11). depth <= 1 now runs plain Richardson with zero
  // Anderson overhead.
  // Anderson/DIIS is DISABLED under the transport-form adjoint: DIIS blends
  // the iterate sequence toward its limit, but the quantity of interest is
  // now the SUM of iterates (accumulated into DV grads per pass) — blended
  // transports no longer sum to the Neumann series, so acceleration would
  // silently corrupt gradients. Plain iteration is exact. Retarget-or-delete
  // decision pending (Jack).
  bool use_anderson = false;
  (void)anderson_depth_;

  size_t m = anderson_depth_;
  size_t history_count = 0;
  size_t adj_iters = 0;

  AdjEquilibriumResult result;

  if (use_anderson) {
    auto &q = queue_pool_[0];
    for (size_t i = 0; i < m; ++i) {
      q.fill(anderson_history_[i].get(), 0.0f, total_recurrent_size_);
      q.fill(anderson_residuals_[i].get(), 0.0f, total_recurrent_size_);
      q.fill(anderson_iterates_[i].get(), 0.0f, total_recurrent_size_);
    }
    q.fill(anderson_flat_.get(), 0.0f, total_recurrent_size_);
    q.fill(anderson_gram_.get(), 0.0f, m * m);
    q.fill(anderson_coeffs_.get(), 0.0f, m);
    q.fill(anderson_rhs_.get(), 0.0f, m);
    q.wait();
  }

  for (size_t k = 0; k < max_adj_iters; ++k) {
    // Advance BEFORE each pass: the prev slot (t_k, accumulated by
    // back-edge consumers last iteration) becomes curr for consumption; a
    // fresh zeroed prev slot receives this iteration's back-edge transport
    // t_{k+1}. Top-of-loop placement means BOTH exit paths — convergence
    // break and max_adj_iters — leave the prev slot holding the newest
    // iterate. First iteration: slot roles were reset by
    // zero_non_dv_grads(), so this advance starts from zeros.
    //
    // TRANSPORT-FORM ADJOINT (supersedes
    // the fixed-point form): objectives are seeded ONCE, on iteration 0.
    // Loss components obey the universal chain rule (grad_input =
    // ∂loss/∂input · grad_loss), so on transport iterations (k ≥ 1) their
    // zeroed output grads make them contribute nothing — no component
    // special-casing. The iterate is the pure transport t_k = (Jᵀ)^k·g
    // (NOT the partial sum): the residual |t_{k+1} − t_k| genuinely closes
    // on every coordinate — including neutrally-stable statevar coordinates
    // (∂curr/∂prev = 1), where the old re-seeded form ramped forever
    // (λ_k = k·g, residual pinned at ‖g‖). DV gradients ACCUMULATE across
    // iterations (zero_non_recurrent_grads now skips DVs): the total
    // Σ_k (∂F/∂θ)ᵀ·t_k equals (∂F/∂θ)ᵀ·Σt_k — the same Neumann series the
    // fixed-point form computed, identical at every truncation depth, but
    // with a well-behaved iterate.
    advance_grad_slots();
    zero_non_recurrent_grads();
    if (k == 0)
      seed_objective_grad();

    if (use_anderson) {
      size_t ring_slot = history_count % m;
      scatter_recurrent_write_to_iterates(ring_slot);
    }

    backward_single_pass();

    ++adj_iters;

    if (adj_scale_log_) {
      std::string scale_worst;
      float scale = backward_to_equilibrium_compute_scale(scale_worst);
      std::fprintf(stderr, "[ADJ-SCALE] k=%zu max_abs_lambda=%.4g worst=%s\n",
                   k + 1, scale, scale_worst.c_str());
    }

    if (adj_clip_norm > 0.0f && total_recurrent_size_ > 0) {
      auto &q = queue_pool_[0];
      float *acc = anderson_norm_acc_.get();
      q.fill(acc, 0.0f, 1).wait();
      constexpr size_t kWG = 256;
      for (const auto &fqn : recurrent_fqns_cache_) {
        size_t count = shape3_size(get_entry(fqn).shape3);
        float *wslot = grad_prev_slot(fqn);
        size_t rounded = ((count + kWG - 1) / kWG) * kWG;
        q.parallel_for(
            sycl::nd_range<1>{rounded, kWG}, [=](sycl::nd_item<1> it) {
              size_t gid = it.get_global_id(0);
              float val = (gid < count) ? wslot[gid] * wslot[gid] : 0.0f;
              float wg_sum = sycl::reduce_over_group(it.get_group(), val,
                                                     sycl::plus<float>());
              if (it.get_local_id(0) == 0) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    ref(*acc);
                ref.fetch_add(wg_sum);
              }
            });
      }
      q.wait();
      float rms = 0.0f;
      q.memcpy(&rms, acc, sizeof(float)).wait();
      rms = std::sqrt(rms / float(total_recurrent_size_));
      if (rms > adj_clip_norm) {
        float scale = adj_clip_norm / rms;
        for (const auto &fqn : recurrent_fqns_cache_) {
          size_t count = shape3_size(get_entry(fqn).shape3);
          float *wslot = grad_prev_slot(fqn);
          q.parallel_for(sycl::range<1>{count},
                         [=](sycl::id<1> id) { wslot[id[0]] *= scale; });
        }
        q.wait();
      }
    }

    if (use_anderson) {
      size_t ring_slot = history_count % m;

      scatter_recurrent_write_to_flat();
      anderson_compute_residual_and_store(ring_slot);

      size_t h = std::min(history_count + 1, m);

      anderson_recompute_gram(h, ring_slot);

      bool is_check =
          ((k + 1) % check_interval == 0) || (k + 1 == max_adj_iters);
      if (is_check) {
        std::string worst;
        float norm = backward_to_equilibrium_compute_norm(worst);
        result.norm = norm;
        result.worst_fqn = std::move(worst);
        if (norm < adj_tol) {
          ++history_count;
          result.converged = true;
          break; // prev slot holds the newest transport iterate
        }
      }

      if (h >= 2) {
        anderson_solve_and_apply_blend(h, ring_slot);
        scatter_flat_to_recurrent_write();
      }

      ++history_count;
    } else {
      bool is_check =
          ((k + 1) % check_interval == 0) || (k + 1 == max_adj_iters);
      if (is_check) {
        std::string worst;
        float norm = backward_to_equilibrium_compute_norm(worst);
        result.norm = norm;
        result.worst_fqn = std::move(worst);
        if (norm < adj_tol) {
          result.converged = true;
          break; // prev slot holds the newest transport iterate
        }
      }
    }
  }

  last_adj_iters_ = adj_iters;
  result.iters = adj_iters;
  return result;
}

} // namespace naga::cospan::core
