// =============================================================================
// include/naga/core/kernel.hpp — kernel functor contract + submission choke
// point
// =============================================================================
//
// FRAMEWORK_CLEANUP S5a: components stop hand-rolling q.parallel_for
// boilerplate. A kernel is a trivially-copyable functor that carries its own
// launch geometry:
//
//   Basic tier (Kernel):   sycl::range<Dim>   range()   + operator()(id<Dim>)
//   WG tier    (NdKernel): sycl::nd_range<Dim> nd_range() + operator()(nd_item<Dim>)
//
// Component code builds functor instances (views + scalars captured by value)
// and hands them to execute_compute(q, k1, k2, ...), which submits them in
// sequence — the in-order queue provides the ordering contract, exactly as
// the old consecutive parallel_for calls did.
//
// detail::submit() is THE submission choke point: every framework and
// component kernel launch flows through it, so instrumentation (per-kernel
// timing, launch counting — P5, kernel_profile.hpp) hangs on one function.
// It is header-only and fully inlined. Instrumentation is opt-in via
// NAGA_KERNEL_PROF=1; when off, the only cost is one branch on a cached bool.
//
// Hand-rolled work-group kernels that the functor tiers don't cover remain
// possible via naga::unsafe_cast + a custom NdKernel functor — available,
// never the default path.

#pragma once

#include <sycl/sycl.hpp>
#include <concepts>
#include <type_traits>

#include <naga/core/kernel_profile.hpp>

namespace naga::core {

// ── Kernel concepts ──────────────────────────────────────────────────────────

template <class K, int Dim>
concept KernelDim =
    std::is_trivially_copyable_v<std::remove_cvref_t<K>> &&
    requires(const std::remove_cvref_t<K> k, sycl::id<Dim> i) {
        { k.range() } -> std::convertible_to<sycl::range<Dim>>;
        k(i);
    };

template <class K>
concept Kernel = KernelDim<K, 1> || KernelDim<K, 2> || KernelDim<K, 3>;

template <class K, int Dim>
concept NdKernelDim =
    std::is_trivially_copyable_v<std::remove_cvref_t<K>> &&
    requires(const std::remove_cvref_t<K> k, sycl::nd_item<Dim> it) {
        { k.nd_range() } -> std::convertible_to<sycl::nd_range<Dim>>;
        k(it);
    };

template <class K>
concept NdKernel = NdKernelDim<K, 1> || NdKernelDim<K, 2> || NdKernelDim<K, 3>;

template <class K>
concept AnyKernel = Kernel<K> || NdKernel<K>;

// ── Submission choke point (S5a) ─────────────────────────────────────────────

namespace detail {

template <AnyKernel K>
inline sycl::event launch(sycl::queue& q, const K& k) {
    using KK = std::remove_cvref_t<K>;
    if constexpr (NdKernelDim<KK, 1>) {
        return q.parallel_for(k.nd_range(), k);
    } else if constexpr (NdKernelDim<KK, 2>) {
        return q.parallel_for(k.nd_range(), k);
    } else if constexpr (NdKernelDim<KK, 3>) {
        return q.parallel_for(k.nd_range(), k);
    } else if constexpr (KernelDim<KK, 1>) {
        return q.parallel_for(k.range(), k);
    } else if constexpr (KernelDim<KK, 2>) {
        return q.parallel_for(k.range(), k);
    } else {
        static_assert(KernelDim<KK, 3>);
        return q.parallel_for(k.range(), k);
    }
}

template <AnyKernel K>
inline void submit(sycl::queue& q, const K& k) {
    // OFF path: one predictable branch on a cached bool, then exactly the
    // uninstrumented launch (the returned event is discarded, as before).
    if (!prof::enabled()) {
        (void)launch(q, k);
        return;
    }
    prof::record<std::remove_cvref_t<K>>(launch(q, k));
}

} // namespace detail

} // namespace naga::core

// execute_compute lives in fusion.hpp (it is the collector's interception
// point); included here so every existing `#include <naga/core/kernel.hpp>`
// site keeps seeing the full choke point. The include is cycle-safe:
// fusion.hpp includes this header, whose guard is already set by this point,
// and everything fusion.hpp needs from it is defined above.
#include <naga/core/fusion.hpp>
