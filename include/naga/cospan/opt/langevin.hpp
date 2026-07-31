#pragma once

// =============================================================================
// naga/cospan/opt/langevin.hpp — Langevin SGD optimizer
// =============================================================================
//
// Online optimizer: momentum SGD + decaying Gaussian noise (Langevin dynamics).
// Designed for single-step optimization where the optimizer updates DVs
// after each forward+backward pass through the component graph.
//
// Usage:
//   LangevinOptimizer opt(config);
//   auto result = opt.optimize(executor);
//

#include "optimize_result.hpp"
#include "naga/cospan/core/executor.hpp"
#include "naga/cospan/core/context.hpp"
#include "naga/cospan/math/rng.hpp"
#include <cmath>
#include <iostream>
#include <naga/slabkit/array.hpp>
#include <vector>

namespace naga::cospan::opt {

struct LangevinConfig {
    float learning_rate    = 0.01f;
    float noise_scale      = 0.1f;      // σ — initial noise magnitude
    float noise_decay      = 0.99f;     // σ *= noise_decay each step
    float min_noise        = 1e-6f;     // floor for noise
    float momentum         = 0.9f;      // 0 = no momentum
    size_t max_iterations  = 1000;
    float grad_tolerance   = 1e-6f;     // ‖∇f‖ < tol → converged
    float max_grad_norm    = 1.0f;      // global gradient clip (<=0 = disabled)
    uint32_t seed          = 42;
    float weight_decay     = 0.0f;
};

class LangevinOptimizer {
public:
    explicit LangevinOptimizer(LangevinConfig config = {})
        : config_(config) {}

    OptimizeResult optimize(naga::cospan::core::Executor& exec) {
        momentum_bufs_.clear();
        auto& ctx = exec.ctx();

        // Discover DVs — scan entries for is_design_variable
        struct DVInfo {
            std::string name;
            size_t count;         // flat element count
            float* data;          // device pointer to DV data
            float* grad;          // device pointer to DV grad
            float* momentum_buf;  // device pointer to momentum buffer
        };

        std::vector<DVInfo> dvs;
        for (const auto& entry : ctx.entries_vec()) {
            if (entry.is_design_variable && !entry.is_frozen) {
                size_t count = 1;
                for (int d = 0; d < 3; ++d) count *= entry.shape3[d];

                // Allocate momentum buffer on device
                auto mom_unique = naga::slabkit::malloc_device<float>(
                    count, ctx.sk_ctx_shared(), ctx.device_index());
                float* mom_ptr = mom_unique.get();
                momentum_bufs_.push_back(std::shared_ptr<float>(std::move(mom_unique)));

                // Zero momentum buffer
                ctx.queue().fill(mom_ptr, 0.0f, count).wait();

                dvs.push_back({entry.name, count, entry.data, entry.grad, mom_ptr});
            }
        }

        // Allocate device scalar for grad_norm_squared computation
        auto gns_buf = naga::slabkit::malloc_device<float>(
            1, ctx.sk_ctx_shared(), ctx.device_index());
        float* gns_ptr = gns_buf.get();
        momentum_bufs_.push_back(std::shared_ptr<float>(std::move(gns_buf)));

        // Allocate device scalar for objective value readback
        auto obj_buf = naga::slabkit::malloc_device<float>(
            1, ctx.sk_ctx_shared(), ctx.device_index());
        float* obj_ptr = obj_buf.get();
        momentum_bufs_.push_back(std::shared_ptr<float>(std::move(obj_buf)));

        // Cache objective data pointer once
        float* obj_data_ptr = nullptr;
        for (const auto& entry : ctx.entries_vec())
            if (entry.is_objective) { obj_data_ptr = entry.data; break; }

        float current_noise = config_.noise_scale;
        bool converged = false;
        size_t iter = 0;
        float final_grad_norm = 0.0f;
        float final_obj = 0.0f;

        for (iter = 0; iter < config_.max_iterations; ++iter) {
            // Run forward + backward (handles gradient lifecycle internally)
            exec.run_with_backward();

            // Read objective value
            if (obj_data_ptr)
                ctx.queue().memcpy(obj_ptr, obj_data_ptr, sizeof(float)).wait();

            // Compute grad norm squared on device (needed for clipping)
            ctx.queue().fill(gns_ptr, 0.0f, 1).wait();
            for (const auto& dv : dvs) {
                float* grad_ptr = dv.grad;
                size_t count = dv.count;
                ctx.queue().parallel_for(sycl::range<1>{count},
                    [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    float g = grad_ptr[i];
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device> ref(*gns_ptr);
                    ref.fetch_add(g * g);
                }).wait();
            }

            // Read grad norm and objective back to host
            float gns_host = 0.0f, obj_host = 0.0f;
            ctx.queue().memcpy(&gns_host, gns_ptr, sizeof(float)).wait();
            ctx.queue().memcpy(&obj_host, obj_ptr, sizeof(float)).wait();

            final_grad_norm = std::sqrt(gns_host);
            final_obj = obj_host;

            // Log progress every 10 iterations
            if (iter % 10 == 0) {
                std::cout << "[Langevin] iter=" << iter
                          << " obj=" << final_obj
                          << " grad_norm=" << final_grad_norm
                          << " noise=" << current_noise
                          << std::endl;
            }

            // Convergence check
            if (final_grad_norm < config_.grad_tolerance) {
                converged = true;
                break;
            }

            // Global gradient clipping: scale = min(1, max_grad_norm / grad_norm)
            float clip_scale = 1.0f;
            if (config_.max_grad_norm > 0.0f && final_grad_norm > config_.max_grad_norm)
                clip_scale = config_.max_grad_norm / final_grad_norm;

            // Update each DV on device: clip + momentum step + noise
            // Mix iter into seed with a hash to avoid period collisions on long runs
            uint32_t iter_u = static_cast<uint32_t>(iter);
            uint32_t iter_hash = iter_u ^ (iter_u << 13) ^ (iter_u >> 17) ^ (iter_u << 5);
            uint32_t seed = config_.seed ^ iter_hash;
            float lr = config_.learning_rate;
            float mom = config_.momentum;
            float noise = current_noise;
            float wd = config_.weight_decay;

            for (auto& dv : dvs) {
                float* dv_data = dv.data;
                float* grad_ptr = dv.grad;
                float* mom_ptr  = dv.momentum_buf;
                size_t count    = dv.count;

                ctx.queue().parallel_for(sycl::range<1>{count},
                    [=](sycl::id<1> idx) {
                    size_t i = idx[0];

                    // Clip gradient
                    float g = grad_ptr[i] * clip_scale;

                    // Momentum update
                    float m = mom * mom_ptr[i] + g;
                    mom_ptr[i] = m;

                    // Random noise
                    uint32_t state = seed + static_cast<uint32_t>(i);
                    float n = naga::cospan::math::gaussian_rand(state) * noise;

                    // Parameter update: x = x * (1 - lr*wd) - lr*m + noise
                    dv_data[i] = dv_data[i] * (1.0f - lr * wd) - lr * m + n;
                }).wait();
            }

            // Decay noise
            current_noise = std::max(config_.min_noise, current_noise * config_.noise_decay);
        }

        return {converged, iter + 1, final_obj, final_grad_norm};
    }

private:
    LangevinConfig config_;
    std::vector<std::shared_ptr<float>> momentum_bufs_;  // owns device memory
};

} // namespace naga::cospan::opt
