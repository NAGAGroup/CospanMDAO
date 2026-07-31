#pragma once

// =============================================================================
// naga/cospan/math/partials.hpp — Analytic value+derivative primitives for kernels
// =============================================================================
//
// Dumb, composable, device-callable float helpers (FRAMEWORK_CLEANUP spec S6).
// Backward kernels compose these named partials instead of re-deriving the
// arithmetic inline — the expressions are IDENTICAL to the historical inline
// forms (bit-identical numerics), just centralized.
//
// Deliberately NOT a tape, NOT functor machinery, NOT dtype-generic. Value +
// derivative are returned together where they share subexpressions.
//

#include <sycl/sycl.hpp>

namespace naga::cospan::math {

// Small value/derivative pair — structured-bindable: auto [v, d] = ...
struct vd {
    float v;
    float d;
};

// -----------------------------------------------------------------------------
// Sigmoid: s(x) = 1 / (1 + exp(-x)),  s'(x) = s(1-s)
// -----------------------------------------------------------------------------
inline float sigmoid(float x) {
    return 1.0f / (1.0f + sycl::exp(-x));
}

// s' recovered from an already-computed sigmoid VALUE (the common backward
// idiom: forward stored f = sigmoid(·), backward uses f*(1-f)).
inline float d_sigmoid_from_value(float s) {
    return s * (1.0f - s);
}

inline vd sigmoid_vd(float x) {
    const float s = sigmoid(x);
    return {s, d_sigmoid_from_value(s)};
}

// -----------------------------------------------------------------------------
// Softplus: p(x) = log(1 + exp(x)) in the numerically stable branch form,
//           p'(x) = sigmoid(x)
// -----------------------------------------------------------------------------
inline float softplus(float x) {
    return x > 0.0f ? x + sycl::log1p(sycl::exp(-x))
                    : sycl::log1p(sycl::exp(x));
}

inline vd softplus_vd(float x) {
    return {softplus(x), sigmoid(x)};
}

// -----------------------------------------------------------------------------
// Tanh: t(x), t'(x) = 1 - t²
// -----------------------------------------------------------------------------
inline vd tanh_vd(float x) {
    const float t = sycl::tanh(x);
    return {t, 1.0f - t * t};
}

// -----------------------------------------------------------------------------
// Gaussian in precomputed-inverse-width form: g = exp(-r² · inv2s2),
// where inv2s2 = 1/(2σ²).  dg/d(r²) = -inv2s2 · g.
// -----------------------------------------------------------------------------
inline float gaussian(float r2, float inv2s2) {
    return sycl::exp(-r2 * inv2s2);
}

inline vd gaussian_vd(float r2, float inv2s2) {
    const float g = gaussian(r2, inv2s2);
    return {g, -inv2s2 * g};
}

// -----------------------------------------------------------------------------
// Difference-of-Gaussians at precomputed gaussian values:
//   K = Ae·ge − Ai·gi
// The backward kernels need ge/gi individually (∂K/∂Ae = ge, ∂K/∂Ai = −gi),
// so the composition stays at the call site; this names the combination.
// -----------------------------------------------------------------------------
inline float dog(float Ae, float ge, float Ai, float gi) {
    return Ae * ge - Ai * gi;
}

// Width-chain factor of a DoG lobe in σ form (inv2s2 = 1/(2σ²)):
//   ∂(A·exp(-r²/2σ²))/∂σ = A·g·(r²/σ³)
// Sign for the inhibitory lobe is applied at the call site (matches the
// historical inline expressions exactly).
inline float dog_dwidth(float A, float g, float r2, float s) {
    return A * g * (r2 / (s * s * s));
}

// -----------------------------------------------------------------------------
// Guarded division: a / max(b, eps)
// -----------------------------------------------------------------------------
inline float safe_div(float a, float b, float eps) {
    return a / sycl::max(b, eps);
}

} // namespace naga::cospan::math
