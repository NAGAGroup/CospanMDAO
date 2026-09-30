#pragma once

// =============================================================================
// examples/gemver-chain/common/chain_data.hpp
// =============================================================================
//
// Plain C++ shared by the CoSpan program and the cuBLAS program: no SYCL and no
// CUDA in here. It holds
//   - the deterministic input generators (an integer hash of (seed, index)),
//   - a host double-precision reference of the whole chain,
//   - the relative-L2 comparison.
//
// The chain (all matrices row-major, alpha = beta = 1/sqrt(n)):
//   GEMVER(A, u1, v1, u2, v2, y, z):  B = A + u1 v1^T + u2 v2^T
//                                     x = beta * B^T y + z
//                                     w = alpha * B x
//   G1     : A = A0, y and z generated.       Outputs B1, x1, w1.
//   BiCGK  : on B1, q = B1 p (p = x1) and s = B1^T r (r = w1).
//   G2     : A = B1, the same u1..v2, y = q, z = s.   Outputs B2, x2, w2.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gemver_chain {

// ---- the deterministic hash -------------------------------------------------------------------
// One seed per array, so no two arrays share values.
enum Seed : std::uint32_t { seed_A = 1, seed_u1 = 2, seed_v1 = 3, seed_u2 = 4, seed_v2 = 5, seed_y = 6, seed_z = 7 };

inline std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/// A deterministic float in [-1, 1) from (seed, index): 24 hash bits, exact in f32.
inline float hash_unit(std::uint32_t seed, std::uint64_t index) {
    const std::uint32_t lo = static_cast<std::uint32_t>(index);
    const std::uint32_t hi = static_cast<std::uint32_t>(index >> 32);
    const std::uint32_t h  = mix32(mix32(seed * 0x9e3779b9u + 0x85ebca6bu) ^ mix32(lo + mix32(hi + seed)));
    return static_cast<float>(h >> 8) * (1.0f / 8388608.0f) - 1.0f;  // [0, 2) - 1
}

// ---- generators -----------------------------------------------------------------------------------
inline std::vector<float> make_vector(std::uint32_t seed, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = hash_unit(seed, i);
    return v;
}

/// A0, row-major n x n.
inline std::vector<float> make_matrix_A0(std::size_t n) {
    std::vector<float> a(n * n);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = hash_unit(seed_A, i);
    return a;
}

/// The six generated vectors of the chain.
struct Vectors {
    std::vector<float> u1, v1, u2, v2, y, z;
};
inline Vectors make_vectors(std::size_t n) {
    return Vectors{make_vector(seed_u1, n), make_vector(seed_v1, n), make_vector(seed_u2, n),
                   make_vector(seed_v2, n), make_vector(seed_y, n),  make_vector(seed_z, n)};
}

// ---- problem constraints ----------------------------------------------------------------------------
/// The demo needs n % 256 == 0 (work-group size) and n % R == 0 (strips of R rows).
inline void check_problem(std::size_t n, std::size_t R) {
    if (n == 0 || n % 256 != 0)
        throw std::invalid_argument("gemver-chain: n must be a positive multiple of 256 (got " + std::to_string(n) + ")");
    if (R == 0 || n % R != 0)
        throw std::invalid_argument("gemver-chain: n must be a multiple of R (n = " + std::to_string(n) +
                                    ", R = " + std::to_string(R) + ")");
}

// ---- the host reference (double precision) ----------------------------------------------------------
struct ChainResult {
    std::vector<double> x1, w1, q, s, x2, w2;
    std::vector<double> B2;  // row-major n x n, only when asked for
};

/// The whole chain in double precision, from the float inputs. B is never stored:
/// it is recomputed on the fly from A0 and the rank-2 update, so this needs O(n)
/// memory beyond the optional B2.
inline ChainResult reference_chain(std::size_t n, const std::vector<float>& A0, const Vectors& g, bool want_B2 = false) {
    const double      beta = 1.0 / std::sqrt(static_cast<double>(n)), alpha = beta;
    const std::size_t nn   = n;

    // B(i, j) = A0(i, j) + times * update(i, j): times = 1 for B1, 2 for B2 (G2 adds the
    // same rank-2 update to A = B1).
    const auto bij = [&](std::size_t i, std::size_t j, int times) {
        return static_cast<double>(A0[i * nn + j]) +
               times * (static_cast<double>(g.u1[i]) * g.v1[j] + static_cast<double>(g.u2[i]) * g.v2[j]);
    };
    // x = beta * B^T y + z, w = alpha * B x.
    const auto gemver = [&](int times, const std::vector<double>& y, const std::vector<double>& z,
                            std::vector<double>& x, std::vector<double>& w) {
        std::vector<double> acc(nn, 0.0);
        for (std::size_t i = 0; i < nn; ++i)
            for (std::size_t j = 0; j < nn; ++j) acc[j] += bij(i, j, times) * y[i];
        x.assign(nn, 0.0);
        for (std::size_t j = 0; j < nn; ++j) x[j] = z[j] + beta * acc[j];
        w.assign(nn, 0.0);
        for (std::size_t i = 0; i < nn; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < nn; ++j) sum += bij(i, j, times) * x[j];
            w[i] = alpha * sum;
        }
    };
    const auto widen = [](const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); };

    ChainResult r;
    gemver(1, widen(g.y), widen(g.z), r.x1, r.w1);

    // BiCGK on B1: q = B1 p (p = x1), s = B1^T r (r = w1).
    r.q.assign(nn, 0.0);
    r.s.assign(nn, 0.0);
    for (std::size_t i = 0; i < nn; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < nn; ++j) {
            const double b = bij(i, j, 1);
            sum += b * r.x1[j];
            r.s[j] += b * r.w1[i];
        }
        r.q[i] = sum;
    }

    gemver(2, r.q, r.s, r.x2, r.w2);  // G2: A = B1, y = q, z = s

    if (want_B2) {
        r.B2.resize(nn * nn);
        for (std::size_t i = 0; i < nn; ++i)
            for (std::size_t j = 0; j < nn; ++j) r.B2[i * nn + j] = bij(i, j, 2);
    }
    return r;
}

// ---- comparison ---------------------------------------------------------------------------------------
/// ||got - want||_2 / ||want||_2 (0 for two zero vectors; infinity if the sizes differ or got has a NaN).
template <class T, class U>
inline double relative_l2(const std::vector<T>& got, const std::vector<U>& want) {
    if (got.size() != want.size()) return INFINITY;
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double g = static_cast<double>(got[i]), w = static_cast<double>(want[i]);
        const double d = g - w;
        num += d * d;
        den += w * w;
    }
    if (std::isnan(num)) return INFINITY;
    if (den == 0.0) return num == 0.0 ? 0.0 : INFINITY;
    return std::sqrt(num / den);
}

/// The effective traffic both programs quote: 8 n^2 floats moved per chain, 4 bytes each.
inline double nominal_bytes(std::size_t n) { return 8.0 * static_cast<double>(n) * static_cast<double>(n) * 4.0; }

}  // namespace gemver_chain
