#pragma once

// =============================================================================
// naga/cospan/core/equilibrium_executor.hpp — EquilibriumExecutor: DEQ-style
// equilibrium optimization
// =============================================================================
//
// EquilibriumExecutor extends Executor to implement Deep Equilibrium Model
// (DEQ)-style optimization. It drives a recurrent system to a fixed point in
// both the forward (primal) and adjoint (dual) passes, instead of unrolling a
// fixed number of steps.
//
// Forward pass (run()):
//   Iterates ctx_->forward() until ||u - u_prev|| < fwd_tol (or max_fwd_iters).
//   Convergence is measured on recurrent-output primal slot pairs
//   (|curr - prev|).
//
// Backward pass (run_with_backward()):
//   After forward equilibrium, iterates the transport-form adjoint:
//     t_{k+1} = Jᵀ · t_k   (t_0 = g, seeded once at k=0)
//   until ||t_{k+1} - t_k|| < adj_tol (or max_adj_iters). DV grads accumulate
//   Σ_k (∂F/∂θ)ᵀ·t_k across iterations — the Neumann series.
//
//   Slot roles per adjoint iteration (the mirror of the primal discipline):
//   - advance_grad_slots() at the top: prev becomes curr (t_k arrives), new
//     prev is zeroed
//   - fresh consumers accumulate into grad curr (consumed same iteration);
//     back-edge consumers accumulate the transport into grad prev
//   - After the pass: grad prev = t_{k+1}, grad curr = t_k
//   - The residual compares grad prev vs grad curr.

#include "naga/cospan/core/context.hpp"
#include "naga/cospan/core/executor.hpp"

#include <naga/slabkit/memory.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace naga::cospan::core {

// =============================================================================
// EquilibriumConfig
// =============================================================================

struct EquilibriumConfig {
  float fwd_tol = 1e-4f;       // forward convergence: ||u - u_prev||
  float adj_tol = 1e-4f;       // adjoint convergence: ||λ_{k+1} - λ_k||
  size_t max_fwd_iters = 1000; // safety cap on forward iterations
  size_t max_adj_iters = 100;  // safety cap on adjoint iterations
  size_t check_interval = 10;  // check convergence every N iterations
  bool verbose = false;        // print fwd/adj iter counts each step
  size_t anderson_depth =
      5; // history depth for Anderson Acceleration (0 = plain Richardson)
  float adj_clip_norm =
      0.0f; // RMS norm cap on adjoint iterate per step (0 = disabled)

  // When non-empty, fwd_norm is computed only over these specific StateVar or
  // recurrent FQNs instead of all recurrent entries. Allows e.g. checking only
  // a model's output buffer rather than all recurrent state.
  std::vector<std::string> fwd_norm_fqns;

  // Settle-horizon telemetry: when traj_interval > 0, snapshot the first
  // element of each traj_fqn output AFTER every forward sweep and print the
  // per-step trajectory at settle end, every traj_interval-th settle. Answers
  // whether a discrete op (the decision readouts) commits within max_fwd_iters.
  std::vector<std::string> traj_fqns;
  size_t traj_interval = 0;
};

// =============================================================================
// EquilibriumStats
// =============================================================================

struct EquilibriumStats {
  size_t fwd_iters = 0;
  size_t adj_iters = 0;
  float fwd_norm = 0.0f;
  std::string fwd_worst_fqn;
  float adj_norm = 0.0f;
  std::string adj_worst_fqn;
  // Convergence flags: false = exited at the iteration cap with the residual
  // still >= tol. With fwd_tol = 0 (deliberate fixed-depth settle) fwd_converged
  // is always false by construction — treat it as "ran the full budget".
  bool fwd_converged = false;
  bool adj_converged = false;
};

// =============================================================================
// EquilibriumExecutor
// =============================================================================

constexpr size_t kEquilibriumWG = 256;

class EquilibriumExecutor : public Executor {
public:
  static std::shared_ptr<EquilibriumExecutor>
  create(std::shared_ptr<System> system, std::shared_ptr<Context> ctx,
         EquilibriumConfig config = {});

  virtual ~EquilibriumExecutor() = default;

  void setup() override;
  void run() override;
  void run_with_backward() override;

  const EquilibriumStats &last_stats() const { return last_stats_; }

  // Update convergence config without re-setup. Safe to call between optimizer
  // iterations.
  void update_config(const EquilibriumConfig &config) { config_ = config; }
  const EquilibriumConfig &config() const { return config_; }

protected:
  EquilibriumExecutor(std::shared_ptr<System> system,
                      std::shared_ptr<Context> ctx, EquilibriumConfig config);

private:
  // The settle loop proper (no post-settle op). run() = settle() +
  // post_settle; run_with_backward() = settle() + adjoint + post_settle —
  // the commit must see the adjoint finish first: gradients belong to the
  // pre-commit equilibrium (post-settle hook).
  void settle();

  float compute_fwd_norm();

  EquilibriumConfig config_;
  EquilibriumStats last_stats_;

  size_t settle_count_ = 0;  // E1: settles seen, for traj_interval gating

  // Recurrent entry FQNs (non-target entries in non-trivial SCCs)
  std::vector<std::string> recurrent_fqns_;

  // Scalar device accumulator for forward norm reduction
  std::shared_ptr<float> fwd_acc_;

  // Packed (|delta| bits << 32 | index) accumulator for the fwd-residual
  // argmax — NAGA_FWD_DEBUG diagnostic (nonnegative float bit patterns are
  // monotone as uints, so one fetch_max finds max AND its coordinate).
  std::shared_ptr<uint64_t> fwd_argmax_acc_;

  // FQNs already warned about as unmeasurable (no dual slots) — warn once.
  std::unordered_set<std::string> warned_skipped_fqns_;
};

} // namespace naga::cospan::core
