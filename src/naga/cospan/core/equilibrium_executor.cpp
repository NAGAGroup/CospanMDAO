#include "naga/cospan/core/equilibrium_executor.hpp"
#include "naga/cospan/core/unsafe.hpp" // detail::RawAccess (framework-internal path)

#include <cstdint>
#include <cstdlib>

namespace naga::cospan::core {

namespace {
// Per-settle non-convergence warnings are OPT-IN via NAGA_EQ_WARN=1: at
// production settle depths they flood stderr (one line per settle), and the
// same information is always available in last_stats_ / the verbose
// per-settle line. One-shot configuration warnings are unaffected.
bool eq_warnings_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("NAGA_EQ_WARN");
    return e && std::atoi(e) != 0;
  }();
  return on;
}
} // namespace

std::shared_ptr<EquilibriumExecutor>
EquilibriumExecutor::create(std::shared_ptr<System> system,
                            std::shared_ptr<Context> ctx,
                            EquilibriumConfig config) {
  return std::shared_ptr<EquilibriumExecutor>(
      new EquilibriumExecutor(std::move(system), std::move(ctx), config));
}

void EquilibriumExecutor::setup() {
  ctx_->set_anderson_depth(config_.anderson_depth); // must precede finalize()
  Executor::setup(); // triggers finalize(), which allocates Anderson workspace

  recurrent_fqns_ = ctx_->recurrent_entries();

  auto fa = slabkit::malloc_device<float>(1, ctx_->sk_ctx_shared(),
                                          ctx_->device_index());
  ctx_->queue().fill(fa.get(), 0.0f, 1);
  fwd_acc_ = std::shared_ptr<float>(std::move(fa));

  auto fp = slabkit::malloc_device<uint64_t>(1, ctx_->sk_ctx_shared(),
                                             ctx_->device_index());
  ctx_->queue().fill(fp.get(), uint64_t{0}, 1);
  fwd_argmax_acc_ = std::shared_ptr<uint64_t>(std::move(fp));

  ctx_->queue().wait();
}

void EquilibriumExecutor::run() {
  // Forward-only path (inference): settle, then apply the post-settle op —
  // the discrete map that mutates the conditions held constant through the
  // settle, initiating the next discrete behavior.
  settle();
  system_->post_settle(*ctx_);
}

void EquilibriumExecutor::settle() {
  size_t iters = 0;
  bool converged = false;
  // Advance BEFORE each forward: the write slot (previous tick's state)
  // becomes the read slot, and a fresh write slot is rotated in. With the
  // advance at the top of the loop, BOTH exit paths — convergence break and
  // max-iters — leave the write slot holding the newest state and the read
  // slot holding the previous tick. (The old advance-at-loop-end structure
  // inverted slot parity on the max-iters path: adjoint fresh/deferred reads
  // swapped, post-settle reads one sweep stale, and the next settle's initial
  // advance rewound persistent state one sweep. Audit 2026-07-11.)
  // E1: within-settle decision-commit trajectory. When enabled, snapshot the
  // first element of each traj_fqn output after every forward sweep, print at
  // settle end. Answers whether a discrete op settles within max_fwd_iters.
  const bool log_traj = config_.traj_interval > 0 &&
                        !config_.traj_fqns.empty() &&
                        (settle_count_ % config_.traj_interval == 0);
  std::vector<float*> traj_ptrs;
  std::vector<std::vector<float>> traj_vals;  // [fqn][step]
  if (log_traj) {
    for (const auto& fqn : config_.traj_fqns)
      traj_ptrs.push_back(
          detail::RawAccess::data(ctx_->get_output<1>(fqn)));
    traj_vals.resize(config_.traj_fqns.size());
  }

  for (size_t k = 0; k < config_.max_fwd_iters; ++k) {
    ctx_->advance_primal_slots();
    Executor::forward();
    ++iters;
    if (log_traj) {
      for (size_t j = 0; j < traj_ptrs.size(); ++j) {
        float v = 0.0f;
        ctx_->queue().memcpy(&v, traj_ptrs[j], sizeof(float)).wait();
        traj_vals[j].push_back(v);
      }
    }
    bool is_check = ((k + 1) % config_.check_interval == 0) ||
                    (k + 1 == config_.max_fwd_iters);
    if (is_check) {
      float norm = compute_fwd_norm();
      if (norm < config_.fwd_tol) {
        converged = true;
        break;
      }
    }
  }
  if (log_traj) {
    std::printf("[TRAJ] settle=%zu (per forward step):\n", settle_count_);
    for (size_t j = 0; j < config_.traj_fqns.size(); ++j) {
      std::printf("[TRAJ]   %s:", config_.traj_fqns[j].c_str());
      for (float v : traj_vals[j]) std::printf(" %.4g", v);
      std::printf("\n");
    }
  }
  ++settle_count_;
  last_stats_.fwd_iters = iters;
  last_stats_.fwd_converged = converged;
  // Non-convergence detection: only warn when the caller actually asked for
  // convergence (fwd_tol > 0). fwd_tol = 0 means deliberate fixed-depth
  // settling — the flag still records it, but it isn't a surprise.
  // Per-settle warnings are OPT-IN via NAGA_EQ_WARN=1 (they drown the log at
  // production settle depths; last_stats_ always records convergence state).
  if (!converged && config_.fwd_tol > 0.0f && eq_warnings_enabled()) {
    std::fprintf(stderr,
                 "[EquilibriumExecutor] WARNING: forward settle exited at "
                 "max_fwd_iters=%zu without converging (norm=%.4g tol=%.4g "
                 "worst=%s) — IFT gradient will be taken at a non-fixed point\n",
                 config_.max_fwd_iters, last_stats_.fwd_norm, config_.fwd_tol,
                 last_stats_.fwd_worst_fqn.c_str());
  }
}

void EquilibriumExecutor::run_with_backward() {
  ctx_->zero_dv_grads();
  ctx_->zero_non_dv_grads();

  // Settle WITHOUT the post-settle op: the adjoint must be solved at the
  // pre-commit equilibrium (gradients belong to the world the settle
  // actually saw). The commit runs after the adjoint, below.
  settle();

  auto adj_result = ctx_->backward_to_equilibrium(
      config_.adj_tol, config_.max_adj_iters, config_.check_interval,
      config_.adj_clip_norm);

  last_stats_.adj_iters = adj_result.iters;
  last_stats_.adj_norm = adj_result.norm;
  last_stats_.adj_worst_fqn = adj_result.worst_fqn;
  last_stats_.adj_converged = adj_result.converged;
  if (!adj_result.converged && config_.adj_tol > 0.0f &&
      eq_warnings_enabled()) {
    std::fprintf(stderr,
                 "[EquilibriumExecutor] WARNING: adjoint solve exited at "
                 "max_adj_iters=%zu without converging (norm=%.4g tol=%.4g "
                 "worst=%s) — DV gradients taken from a non-converged adjoint\n",
                 config_.max_adj_iters, adj_result.norm, config_.adj_tol,
                 adj_result.worst_fqn.c_str());
  }

  // DV gradients accumulate ACROSS the adjoint transport iterations
  // (transport-form adjoint, 2026-07-19): each pass adds (∂F/∂θ)ᵀ·t_k, so at
  // exit the DV grads hold (∂F/∂θ)ᵀ·Σt_k — the Neumann series — plus the
  // direct objective path from the single seeding pass. No extra backward
  // pass needed.

  // Post-settle op AFTER the adjoint: mutate the settle's held-constant
  // conditions (e.g. Mosaic scratch → committed) for the next settle.
  system_->post_settle(*ctx_);

  if (config_.verbose) {
    std::printf("[EquilibriumExecutor] fwd_iters=%zu adj_iters=%zu "
                "fwd_norm=%.4g fwd_worst=%s adj_norm=%.4g adj_worst=%s\n",
                last_stats_.fwd_iters, last_stats_.adj_iters,
                last_stats_.fwd_norm, last_stats_.fwd_worst_fqn.c_str(),
                last_stats_.adj_norm, last_stats_.adj_worst_fqn.c_str());
  }
}

EquilibriumExecutor::EquilibriumExecutor(std::shared_ptr<System> system,
                                         std::shared_ptr<Context> ctx,
                                         EquilibriumConfig config)
    : Executor(std::move(system), std::move(ctx)), config_(config) {}

// Max norm: max_i |cur_i - prev_i| across all recurrent state elements.
// More selective than RMS — captures worst-case un-converged element rather
// than averaging convergent and divergent elements together. Pairs naturally
// with a softmax/LSE-style energy regularizer that emphasizes the same outlier
// modes.
float EquilibriumExecutor::compute_fwd_norm() {
  auto &queue = ctx_->queue();
  float global_max = 0.0f;
  std::string worst_fqn;

  float *fwd_acc_ptr = fwd_acc_.get();

  // Use configured FQN list if set, otherwise check all recurrent entries
  const auto &fqns_to_check = config_.fwd_norm_fqns.empty()
      ? recurrent_fqns_
      : config_.fwd_norm_fqns;

  for (size_t i = 0; i < fqns_to_check.size(); ++i) {
    const auto &fqn = fqns_to_check[i];
    // Resolve to the owning entry: slots live on owners, and for a share
    // member the residual is measured over the owner's full buffer (which
    // contains the member's slice). Unknown FQNs throw.
    const Context::Entry *owner = ctx_->owning_entry(fqn);
    if (!owner)
      throw std::runtime_error("[EquilibriumExecutor] fwd_norm fqn '" + fqn +
                               "' does not name any registered entry");
    size_t count = Context::shape3_size(owner->shape3);

    float *cur = ctx_->primal_curr_slot(fqn);
    float *prev = ctx_->primal_prev_slot(fqn);

    if (!cur || !prev) {
      // Owner has no primal slot pair — not a recurrent output, so a settle
      // residual cannot be measured on it. Silent skipping here previously
      // let taught quantities drop out of the convergence check unnoticed
      // (audit 2026-07-11, flagged by three auditors).
      if (!config_.fwd_norm_fqns.empty())
        throw std::runtime_error(
            "[EquilibriumExecutor] fwd_norm_fqns entry '" + fqn +
            "' (owner '" + owner->name +
            "') has no primal slot pair (not a recurrent output) — it "
            "cannot participate in the settle residual");
      if (warned_skipped_fqns_.insert(fqn).second)
        std::fprintf(stderr,
                     "[EquilibriumExecutor] WARNING: fwd_norm skipping "
                     "recurrent fqn '%s' (owner '%s' has no primal slot "
                     "pair)\n",
                     fqn.c_str(), owner->name.c_str());
      continue;
    }

    queue.fill(fwd_acc_ptr, 0.0f, 1);

    // NAGA_FWD_DEBUG diagnostic: pack (|delta| bits << 32 | index) so one
    // fetch_max yields the max AND its coordinate (nonnegative float bit
    // patterns are monotone as uints).
    static const bool fwd_debug_env = std::getenv("NAGA_FWD_DEBUG") != nullptr;
    // Local copy: statics are ODR-used, not captured, by [=] lambdas — a
    // device kernel must capture a plain local by value.
    const bool fwd_debug = fwd_debug_env;
    uint64_t *argmax_ptr = fwd_argmax_acc_.get();
    if (fwd_debug)
      queue.fill(argmax_ptr, uint64_t{0}, 1);

    constexpr size_t kWG = 256;
    size_t rounded = ((count + kWG - 1) / kWG) * kWG;
    queue.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{rounded}, sycl::range<1>{kWG}},
        [=](sycl::nd_item<1> it) {
          size_t gid = it.get_global_id(0);
          float val = 0.0f;
          if (gid < count) {
            val = sycl::fabs(cur[gid] - prev[gid]);
          }
          float wg_max = sycl::reduce_over_group(it.get_group(), val,
                                                 sycl::maximum<float>());
          if (fwd_debug) {
            uint64_t packed =
                (static_cast<uint64_t>(sycl::bit_cast<uint32_t>(val)) << 32) |
                static_cast<uint64_t>(gid & 0xffffffffu);
            uint64_t wg_packed = sycl::reduce_over_group(
                it.get_group(), packed, sycl::maximum<uint64_t>());
            if (it.get_local_id(0) == 0) {
              sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  aref(*argmax_ptr);
              aref.fetch_max(wg_packed);
            }
          }
          if (it.get_local_id(0) == 0) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                ref(*fwd_acc_ptr);
            ref.fetch_max(wg_max);
          }
        });
    queue.wait();

    float entry_max = 0.0f;
    queue.memcpy(&entry_max, fwd_acc_ptr, sizeof(float)).wait();

    if (fwd_debug && entry_max > 0.0f) {
      uint64_t packed = 0;
      queue.memcpy(&packed, argmax_ptr, sizeof(uint64_t)).wait();
      size_t idx = static_cast<size_t>(packed & 0xffffffffu);
      float cv = 0.0f, pv = 0.0f;
      queue.memcpy(&cv, cur + idx, sizeof(float)).wait();
      queue.memcpy(&pv, prev + idx, sizeof(float)).wait();
      std::fprintf(stderr,
                   "[FWD-DBG] %s idx=%zu cur=%.6g prev=%.6g |d|=%.6g\n",
                   fqn.c_str(), idx, cv, pv, entry_max);
    }

    if (entry_max > global_max) {
      global_max = entry_max;
      worst_fqn = fqn;
    }
  }

  last_stats_.fwd_norm = global_max;
  last_stats_.fwd_worst_fqn = worst_fqn;
  return global_max;
}

} // namespace naga::cospan::core
