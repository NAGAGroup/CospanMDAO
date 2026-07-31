#pragma once

// =============================================================================
// naga/math/rng.hpp — Device-side random number generation for SYCL kernels
// =============================================================================
//
// Hash-based PRNG (PCG variant) + Box-Muller transform for Gaussian noise.
// Stateless per-call pattern: each work-item initializes a uint32_t state
// from (seed + global_id + iteration_offset), then calls pcg_rand/gaussian_rand.
// No global state, no synchronization, works on any SYCL backend.
//

#include <sycl/sycl.hpp>

namespace naga::math {

// PCG-style hash — returns uniform float in [0, 1), mutates state
inline float pcg_rand(uint32_t& state) {
    uint32_t v = state;
    state = state * 747796405u + 2891336453u;
    v = v ^ (v >> ((v >> 28u) + 4u));
    v = v * 277803737u;
    v = v ^ (v >> 22u);
    return static_cast<float>(v) / 4294967295.0f;
}

// Box-Muller transform — returns Gaussian N(0,1), consumes 2 uniform draws
inline float gaussian_rand(uint32_t& state) {
    float u1 = pcg_rand(state);
    float u2 = pcg_rand(state);
    u1 = sycl::max(u1, 1e-10f);  // clamp away from zero for log
    return sycl::sqrt(-2.0f * sycl::log(u1)) * sycl::cos(6.2831853f * u2);
}

} // namespace naga::math
