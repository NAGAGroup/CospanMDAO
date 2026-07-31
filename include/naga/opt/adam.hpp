#pragma once

#include "optimize_result.hpp"
#include "naga/core/executor.hpp"
#include "naga/core/context.hpp"
#include "naga/core/unsafe.hpp"
#include <sycl/sycl.hpp>
#include <slabkit/array.hpp>
#include <functional>
#include <vector>
#include <cmath>
#include <cstdio>

namespace naga::opt {

constexpr size_t kAdamWG = 256;

// =============================================================================
// ObjectiveReduction — how per-component objective values are reduced to the
// single scalar used for stop/report.  Gradient accumulation is NOT affected:
// each objective still seeds its own grad independently (sum semantics).
// =============================================================================
enum class ObjectiveReduction {
    Mean,     // arithmetic mean of per-component values
    Sum,      // total sum of per-component values
    SoftMax,  // soft-max-weighted mean (smooth max, temperature=1)
    MaxNorm,  // maximum component value (default — most conservative for stop)
};

struct AdamConfig {
    float  learning_rate  = 1e-3f;
    size_t lr_ramp_iters  = 0;         // linear LR warmup over this many
                                       // iterations (0 = off). For warm
                                       // starts with fresh moments: early
                                       // steps have no second-moment
                                       // history and are effectively
                                       // huge relative to a converged
                                       // equilibrium (warm-restart
                                       // chain-shock forensics).
    float  beta1          = 0.9f;      // first moment decay
    float  beta2          = 0.999f;    // second moment decay
    float  epsilon        = 1e-8f;     // numerical stability
    float  max_grad_norm  = 0.0f;      // global gradient clip (<=0 = disabled)
    float  obj_tolerance  = 0.0f;      // stop when reduced objective < this value (0 = disabled)
    size_t max_iterations = 1000;
    bool   reset_between_iterations = false;  // call exec.reset() before each run_with_backward()
                                              // for online optimization this should be false;
                                              // for equilibrium-style optimization it may be true
    size_t callback_interval = 10;  // call iter_callback every N iterations (default 10)
    std::function<void(naga::core::Executor&, size_t)> iter_callback;  // called each iter before reset (if any). args: executor, iter index

    // Step gate: return false to skip
    // the DV update for this iteration. Purpose: keep gradients well-posed
    // BY CONSTRUCTION instead of clipping magnitudes — e.g. gate on forward
    // settle convergence, because an IFT adjoint solved around a
    // non-fixed-point is meaningless (measured: P(adj blowup | unsettled
    // fwd) = 50% vs 3% settled; 86% of giant spikes sat on unsettled
    // forwards). Gated iterations skip the moment updates too, so Adam's
    // second-moment estimate is not poisoned by invalid gradients. The
    // forward run still happens — state evolution is untouched (always-on).
    std::function<bool(naga::core::Executor&)> step_gate;

    // Objective reduction mode for stop/report scalar.
    // Gradient accumulation is unaffected (each objective seeds its own grad).
    ObjectiveReduction obj_reduction = ObjectiveReduction::MaxNorm;

    // grad_tolerance: stop when grad norm < this value (0.0f = disabled).
    // Kept for backward compatibility with simple optimization problems.
    // For dynamic settling systems, set to 0.0f and use obj_tolerance instead:
    // grad-norm-based stopping is unreliable when the system does not settle
    // to a fixed point between optimizer steps.
    float  grad_tolerance = 0.0f;
};

class AdamOptimizer {
public:
    explicit AdamOptimizer(AdamConfig config = {}) : config_(config) {}

    OptimizeResult optimize(naga::core::Executor& exec) {
        aux_bufs_.clear();
        gate_skips_ = 0;
        auto& ctx   = exec.ctx();
        auto& queue = ctx.queue();

        // discover design variables
        struct DVInfo {
            float*  data;
            float*  grad;
            float*  m;     // first moment buffer
            float*  v;     // second moment buffer
            size_t  size;
        };

        std::vector<DVInfo> dvs;
        for (const auto& entry : ctx.entries_vec()) {
            if (!entry.is_design_variable || entry.is_frozen) continue;
            size_t sz = entry.shape3[0] * entry.shape3[1] * entry.shape3[2];

            auto m_unique = slabkit::malloc_device<float>(
                sz, ctx.sk_ctx_shared(), ctx.device_index());
            float* m_buf = m_unique.get();
            aux_bufs_.push_back(std::shared_ptr<float>(std::move(m_unique)));

            auto v_unique = slabkit::malloc_device<float>(
                sz, ctx.sk_ctx_shared(), ctx.device_index());
            float* v_buf = v_unique.get();
            aux_bufs_.push_back(std::shared_ptr<float>(std::move(v_unique)));

            queue.fill(m_buf, 0.0f, sz).wait();
            queue.fill(v_buf, 0.0f, sz).wait();

            dvs.push_back({entry.data, entry.grad, m_buf, v_buf, sz});
        }

        // allocate device scratch for grad norm squared
        auto gns_unique = slabkit::malloc_device<float>(
            1, ctx.sk_ctx_shared(), ctx.device_index());
        float* gns_ptr = gns_unique.get();
        aux_bufs_.push_back(std::shared_ptr<float>(std::move(gns_unique)));

        // find objective entry FQNs — use get_output<1> to read the write slot
        // (current tick's value) rather than the raw entry.data pointer.
        // Perf (audit §4.8): objective write-slot pointers are static for
        // non-recurrent outputs — cache them once; per iteration submit all
        // D2H copies then wait ONCE (was 2 chained blocking memcpys × N).
        std::vector<std::string> obj_fqns;
        for (const auto& entry : ctx.entries_vec())
            if (entry.is_objective) obj_fqns.push_back(entry.name);
        std::vector<float*> obj_ptrs;
        obj_ptrs.reserve(obj_fqns.size());
        for (const auto& fqn : obj_fqns)
            obj_ptrs.push_back(naga::unsafe_cast(ctx.get_output<1>(fqn)).data);

        // Objective-group telemetry: group objectives by FQN prefix so we see WHICH losses
        // descend (decision / memory / controller), not just the reduced scalar.
        // 0=decision, 1=memory, 2=controller, 3=other.
        auto obj_group = [](const std::string& f) -> int {
            if (f.find("/loss_dec_") != std::string::npos) return 0;
            if (f.find("/loss_mem_") != std::string::npos) return 1;
            if (f.find("/mosaic/loss") != std::string::npos) return 2;
            return 3;
        };
        std::vector<int> obj_grp(obj_fqns.size());
        for (size_t i = 0; i < obj_fqns.size(); ++i) obj_grp[i] = obj_group(obj_fqns[i]);
        static const char* kGrpName[4] = {"dec", "mem", "ctrl", "other"};

        const float beta1     = config_.beta1;
        const float beta2     = config_.beta2;
        const float epsilon   = config_.epsilon;
        const float lr        = config_.learning_rate;
        const float max_gn    = config_.max_grad_norm;
        const float tol       = config_.grad_tolerance;  // 0 = disabled (recommended for dynamic settling)
        const float obj_tol   = config_.obj_tolerance;
        const size_t max_iter = config_.max_iterations;
        const ObjectiveReduction reduction = config_.obj_reduction;

        OptimizeResult result{false, 0, 0.0f, 0.0f};

        for (size_t iter = 0; iter < max_iter; ++iter) {
            if (config_.iter_callback && (iter % config_.callback_interval == 0))
                config_.iter_callback(exec, iter);
            if (config_.reset_between_iterations)
                exec.reset();
            exec.run_with_backward();

            // read per-component objective values (cached write-slot pointers;
            // batched D2H, single wait — audit §4.8)
            std::vector<float> obj_vals(obj_ptrs.size(), 0.0f);
            for (size_t i = 0; i < obj_ptrs.size(); ++i)
                queue.memcpy(&obj_vals[i], obj_ptrs[i], sizeof(float));
            queue.wait();

            // Reduce per-component objective values to a single stop/report scalar.
            // Gradient accumulation is NOT affected — each objective seeds its own
            // grad independently (sum semantics, unchanged).
            float obj_val = 0.0f;
            if (!obj_vals.empty()) {
                const size_t n = obj_vals.size();
                switch (reduction) {
                    case ObjectiveReduction::Mean: {
                        float sum = 0.0f;
                        for (float v : obj_vals) sum += v;
                        obj_val = sum / static_cast<float>(n);
                        break;
                    }
                    case ObjectiveReduction::Sum: {
                        float sum = 0.0f;
                        for (float v : obj_vals) sum += v;
                        obj_val = sum;
                        break;
                    }
                    case ObjectiveReduction::SoftMax: {
                        // soft-max-weighted mean (smooth max, temperature=1)
                        float max_v = obj_vals[0];
                        for (float v : obj_vals) if (v > max_v) max_v = v;
                        float sum_exp = 0.0f;
                        for (float v : obj_vals) sum_exp += std::exp(v - max_v);
                        float weighted = 0.0f;
                        for (float v : obj_vals) weighted += v * std::exp(v - max_v);
                        obj_val = (sum_exp > 0.0f) ? weighted / sum_exp : 0.0f;
                        break;
                    }
                    case ObjectiveReduction::MaxNorm: {
                        float mx = obj_vals[0];
                        for (float v : obj_vals) if (v > mx) mx = v;
                        obj_val = mx;
                        break;
                    }
                }
            }

            // compute grad norm squared (for diagnostics and optional grad_tolerance stop)
            queue.fill(gns_ptr, 0.0f, 1);
            for (auto& dv : dvs) {
                float* g  = dv.grad;
                size_t sz = dv.size;
                size_t rounded = ((sz + kAdamWG - 1) / kAdamWG) * kAdamWG;
                queue.parallel_for(
                    sycl::nd_range<1>{sycl::range<1>{rounded}, sycl::range<1>{kAdamWG}},
                    [=](sycl::nd_item<1> it) {
                        size_t gid = it.get_global_id(0);
                        float val = (gid < sz) ? g[gid] * g[gid] : 0.0f;
                        float wg_sum = sycl::reduce_over_group(
                            it.get_group(), val, sycl::plus<float>());
                        if (it.get_local_id(0) == 0) {
                            sycl::atomic_ref<float,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space> ref(*gns_ptr);
                            ref.fetch_add(wg_sum);
                        }
                    });
            }
            float gns_host = 0.0f;
            queue.memcpy(&gns_host, gns_ptr, sizeof(float)).wait();
            float grad_norm = std::sqrt(gns_host);

            // Guard against NaN/Inf gradients — skip update entirely if degenerate.
            // This happens when loss=0 (e.g. warmup with perfect init), giving zero
            // gradients everywhere and a degenerate Adam second-moment denominator.
            const bool gate_reject =
                config_.step_gate && !config_.step_gate(exec);
            if (gate_reject) ++gate_skips_;
            bool skip_update = gate_reject || !std::isfinite(grad_norm) ||
                               grad_norm < 1e-30f;

            // log every 10 iterations
            if (iter % 10 == 0) {
                // E2: per-group means (dec/mem/ctrl/other)
                float gsum[4] = {0,0,0,0}; int gcnt[4] = {0,0,0,0};
                for (size_t i = 0; i < obj_vals.size(); ++i) {
                    gsum[obj_grp[i]] += obj_vals[i]; gcnt[obj_grp[i]]++;
                }
                std::printf("[Adam] iter=%zu obj=%.6g grad_norm=%.6g%s |",
                            iter, obj_val, grad_norm,
                            gate_reject       ? " (gate-skipped)"
                            : skip_update     ? " (skipped)"
                                              : "");
                for (int g = 0; g < 4; ++g)
                    if (gcnt[g])
                        std::printf(" %s=%.4g", kGrpName[g], gsum[g] / gcnt[g]);
                std::printf("\n");
            }

            // convergence check — primary: reduced objective value.
            // Secondary: grad_tolerance (backward-compatible; set to 0 to disable,
            // which is recommended for dynamic settling systems).
            if (obj_tol > 0.0f && obj_val < obj_tol) {
                result = {true, iter + 1, obj_val, grad_norm};
                break;
            }
            if (tol > 0.0f && grad_norm < tol) {
                result = {true, iter + 1, obj_val, grad_norm};
                break;
            }
            result = {false, iter + 1, obj_val, grad_norm};

            if (skip_update) continue;

            // gradient clipping scale
            float clip = 1.0f;
            if (max_gn > 0.0f && grad_norm > max_gn)
                clip = max_gn / grad_norm;

            // bias correction scalars (computed on host, captured by value)
            float t   = static_cast<float>(iter + 1);
            float bc1 = 1.0f - std::pow(beta1, t);
            float bc2 = 1.0f - std::pow(beta2, t);

            // Linear LR warmup (see lr_ramp_iters).
            const float lr_t =
                (config_.lr_ramp_iters > 0 &&
                 iter + 1 < config_.lr_ramp_iters)
                    ? lr * (static_cast<float>(iter + 1) /
                            static_cast<float>(config_.lr_ramp_iters))
                    : lr;

            // Adam update per DV
            for (auto& dv : dvs) {
                float* data = dv.data;
                float* grad = dv.grad;
                float* m    = dv.m;
                float* v    = dv.v;
                size_t sz   = dv.size;

                queue.parallel_for(sycl::range<1>{sz}, [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    float g = grad[i] * clip;

                    // update moments
                    float mi = beta1 * m[i] + (1.0f - beta1) * g;
                    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
                    m[i] = mi;
                    v[i] = vi;

                    // bias-corrected Adam step
                    float m_hat = mi / bc1;
                    float v_hat = vi / bc2;
                    data[i] -= lr_t * m_hat / (sycl::sqrt(v_hat) + epsilon);
                });
            }
            queue.wait();
        }

        if (gate_skips_ > 0)
            std::printf("[Adam] step-gate skipped %zu/%zu iterations "
                        "(unsettled forwards)\n",
                        gate_skips_, result.iterations);
        return result;
    }

    size_t gate_skips() const { return gate_skips_; }

private:
    AdamConfig config_;
    size_t gate_skips_ = 0;
    std::vector<std::shared_ptr<float>> aux_bufs_;  // owns device memory
};

} // namespace naga::opt
