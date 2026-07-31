// =============================================================================
// include/naga/core/fusion.hpp — tier-0.5 horizontal kernel fusion
// =============================================================================
//
// Fuses kernels that ALREADY run concurrently into single launches.
//
// Components in the same execution layer have no dependency edges by
// construction — today they run on parallel in-order queues, so the hardware
// may interleave them arbitrarily. Fusing them into one kernel cannot
// introduce an ordering that was not already possible: this transform is
// semantics-preserving with no access-pattern metadata required. ("Tier 0.5"
// of the kernel-fusion plan; tiers 1/2 — seam-bounded vertical fusion and
// persistent megakernels — build on the access-pattern spec.)
//
// Mechanism: ACPP_EXT_DYNAMIC_FUNCTIONS (SSCP generic JIT only). The set of
// kernel types to fuse is only known at runtime, so the fused kernel body is
// a JIT-assembled call sequence of `hstage<K>` template instantiations; each
// stage reads its functor bytes from a device param arena through a
// ParamCursor and grid-strides its own index space. Sequences are cached by
// signature (the sequence of dynamic-function handles), so a given wave shape
// JIT-compiles exactly once per process (and hits AdaptiveCpp's persistent
// on-disk cache across runs). Application-driven submission branching just
// produces a handful of distinct signatures, each cached.
//
// Wave discipline (correctness):
//   * Only components ALL of whose kernels are fusable (basic tier, 1D)
//     participate; any component that submits an nd_range/2D/3D kernel is
//     replayed verbatim, in order, on its own queue — exactly today's path.
//   * Wave j = the j-th kernel of each participating component. A component's
//     own kernel sequence is dependent (in-order queue contract), so waves
//     flush sequentially on ONE in-order queue: wave j+1 starts after wave j.
//     Intra-component order is preserved; cross-component order was never
//     guaranteed to begin with.
//   * Launch geometry = max stage n; every stage grid-strides its own n, so
//     each element is computed exactly once with unchanged arithmetic —
//     bit-identical to the unfused launch for non-atomic writes; atomic
//     accumulation order changes only within the envelope already permitted
//     by concurrent queues.
//
// CONTRACT (framework rule, enforced by review not by code): components must
// route ALL device work in forward()/backward() through execute_compute().
// Raw queue operations (fill/memcpy/parallel_for) in those hooks are
// invisible to the collector and void the ordering guarantee. As of
// 2026-07-17 every component honors this (raw ops appear only in init()).
//
// Gate: NAGA_FUSION=0 disables collection entirely — the submission path is
// then bit-identical to the pre-fusion framework (one launch per kernel).
// =============================================================================

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <slabkit/memory.hpp>

#include <naga/core/kernel.hpp>

namespace naga::core::fusion {

namespace jit = ::sycl::AdaptiveCpp_jit;

// ── ParamCursor: per-thread walk over the functor-byte arena ────────────────
//
// Device-side. Each fused stage consumes its own functor's bytes in call
// order; the host assembles the arena with the same alignment walk
// (append_functor_bytes), so cursor positions agree by construction.

struct ParamCursor {
    const std::byte* p;

    template <class K>
    K next() {
        std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p);
        a = (a + alignof(K) - 1) & ~static_cast<std::uintptr_t>(alignof(K) - 1);
        K k;
        __builtin_memcpy(&k, reinterpret_cast<const void*>(a), sizeof(K));
        p = reinterpret_cast<const std::byte*>(a) + sizeof(K);
        return k;
    }
};

// Host-side mirror of the cursor's alignment walk.
inline void append_functor_bytes(std::vector<std::byte>& blob, const void* src,
                                 size_t size, size_t align) {
    size_t off = blob.size();
    off = (off + align - 1) & ~(align - 1);
    blob.resize(off + size);
    std::memcpy(blob.data() + off, src, size);
}

// ── The stage template: one dynamic-function definition per functor type ────
//
// Instantiated per basic-tier kernel functor type at the collection site;
// &hstage<K> is a compile-time function pointer, satisfying the extension's
// direct-pointer requirement. Grid-stride over the stage's own n: with the
// wave launch sized to max n, each element runs exactly once.

template <class K>
SYCL_EXTERNAL void hstage(sycl::item<1> it, ParamCursor& cur) {
    const K      k = cur.template next<K>();
    const size_t n = sycl::range<1>(k.range()).size();
    const size_t G = it.get_range(0);
    for (size_t i = it.get_id(0); i < n; i += G)
        k(sycl::id<1>{i});
}

// The dynamic function every fused kernel calls; its definition is replaced
// at JIT time by the wave's call sequence. The noinline default definition
// with arguments_are_used follows the extension's documented recommendation.
__attribute__((noinline)) inline void fused_ops(sycl::item<1> it,
                                                ParamCursor& cur) {
    jit::arguments_are_used(it, cur);
}

// Fused kernel functor: captures only the wave's arena slice.
struct FusedWaveKernel {
    const std::byte* arena;
    void operator()(sycl::item<1> it) const {
        ParamCursor cur{arena};
        fused_ops(it, cur);
    }
};

inline jit::dynamic_function_id fused_entry_id() {
    static const jit::dynamic_function<void, sycl::item<1>, ParamCursor&>
        entry{&fused_ops};
    return entry.id();
}

// ── Wave launcher (module-visibility anchor) ─────────────────────────────────
//
// The SSCP JIT resolves call-sequence targets within the DEVICE MODULE OF THE
// TU THAT OUTLINED THE KERNEL. hstage<K> instantiations are emitted into the
// translation units that instantiate collect<K> (the application TU compiling
// the component kernels); if the fused kernel were launched from a framework
// .cpp, its device module would not contain them ("Could not find function
// call specialization target"). collect<K> therefore takes the address of
// this inline function, ODR-using it — and thereby outlining FusedWaveKernel —
// in the SAME TU that emits the stages. flush_wave launches through the
// stored pointer.
//
// Consequence (documented constraint): all components fused into one wave
// must be instantiated from one SYCL TU. True for single-TU applications;
// multi-TU apps would need per-TU fuser partitioning.
inline void launch_fused_wave(sycl::queue& q, jit::dynamic_function_config& cfg,
                              const std::byte* arena, size_t max_n) {
    auto ev = q.parallel_for(sycl::range<1>{max_n},
                             cfg.apply(FusedWaveKernel{arena}));
    if (prof::enabled())
        prof::record<FusedWaveKernel>(ev);
    else
        (void)ev;
}

using WaveLauncherFn = void (*)(sycl::queue&, jit::dynamic_function_config&,
                                const std::byte*, size_t);

// ── HorizontalFuser ──────────────────────────────────────────────────────────

class HorizontalFuser {
public:
    // Allocate the device param arena. Call once, after the queue pool
    // exists (finalize). No-op when disabled.
    void init(const std::shared_ptr<slabkit::Context>& sk_ctx,
              size_t device_index) {
        if (!enabled() || arena_)
            return;
        arena_ = slabkit::malloc_device<uint8_t>(kArenaBytes, sk_ctx,
                                                 device_index);
        arena_cap_ = kArenaBytes;
    }

    static bool enabled() {
        static const bool on = [] {
            const char* e = std::getenv("NAGA_FUSION");
            return !(e && e[0] == '0' && e[1] == '\0');
        }();
        return on;
    }

    // Call at the top of a pass (forward()/backward()), i.e. after the
    // previous pass's terminal queue waits: recycles the arena and the
    // in-flight host blobs.
    void begin_pass() {
        arena_off_ = 0;
        inflight_blobs_.clear();
    }

    void begin_layer(sycl::queue& flush_q) { flush_queue_ = &flush_q; }

    void begin_component(sycl::queue& q) {
        comp_queue_ = &q;
        comp_records_.clear();
    }

    template <AnyKernel K>
    void collect(sycl::queue& q, const K& k) {
        using KK = std::remove_cvref_t<K>;
        StageRecord r;
        r.replay = [k](sycl::queue& rq) { detail::submit(rq, k); };
        if constexpr (KernelDim<KK, 1> && !NdKernel<KK>) {
            const size_t n = sycl::range<1>(k.range()).size();
            if (n == 0)
                return;  // empty basic launch: no-op either way
            static const jit::dynamic_function_definition<void, sycl::item<1>,
                                                          ParamCursor&>
                def{&hstage<KK>};
            // Anchor the wave launcher in THIS TU — the one emitting
            // hstage<KK> to device code (see launch_fused_wave).
            wave_launcher_ = &launch_fused_wave;
            r.def_id     = def.id();
            r.def_handle = def.id().get_handle();
            r.n          = n;
            r.align      = alignof(KK);
            r.bytes.resize(sizeof(KK));
            std::memcpy(r.bytes.data(), &k, sizeof(KK));
            r.fusable = true;
        } else {
            (void)q;
            r.fusable = false;
        }
        comp_records_.push_back(std::move(r));
    }

    void end_component() {
        bool all_fusable = true;
        for (const auto& r : comp_records_)
            all_fusable &= r.fusable;
        if (!all_fusable) {
            // Mixed component: replay verbatim on its own queue — exactly
            // the pre-fusion path, concurrency with other components intact.
            for (auto& r : comp_records_)
                r.replay(*comp_queue_);
        } else {
            for (size_t j = 0; j < comp_records_.size(); ++j) {
                if (waves_.size() <= j)
                    waves_.emplace_back();
                waves_[j].push_back(std::move(comp_records_[j]));
            }
        }
        comp_records_.clear();
        comp_queue_ = nullptr;
    }

    void end_layer() {
        for (auto& wave : waves_)
            flush_wave(wave);
        waves_.clear();
        flush_queue_ = nullptr;
    }

private:
    static constexpr size_t kArenaBytes = 4u << 20;  // 4 MiB ≫ per-pass usage

    struct StageRecord {
        jit::dynamic_function_id def_id{};
        const void*              def_handle = nullptr;
        std::vector<std::byte>   bytes;
        size_t                   n     = 0;
        size_t                   align = 1;
        bool                     fusable = false;
        std::function<void(sycl::queue&)> replay;
    };

    struct SigHash {
        size_t operator()(const std::vector<const void*>& v) const {
            size_t h = 1469598103934665603ull;
            for (const void* p : v) {
                h ^= reinterpret_cast<size_t>(p);
                h *= 1099511628211ull;
            }
            return h;
        }
    };

    void flush_wave(std::vector<StageRecord>& wave) {
        if (wave.empty())
            return;
        sycl::queue& q = *flush_queue_;
        // Singleton waves and arena overflow replay on the FLUSH queue, not
        // the component queue: once any of a component's kernels entered a
        // wave, all its later kernels must follow on the same in-order queue
        // to preserve intra-component order.
        if (wave.size() == 1) {
            wave[0].replay(q);
            return;
        }

        std::vector<std::byte>   blob;
        std::vector<const void*> sig;
        size_t                   max_n = 0;
        sig.reserve(wave.size());
        for (const auto& r : wave) {
            append_functor_bytes(blob, r.bytes.data(), r.bytes.size(),
                                 r.align);
            sig.push_back(r.def_handle);
            max_n = std::max(max_n, r.n);
        }

        if (!arena_ || !wave_launcher_ ||
            arena_off_ + blob.size() > arena_cap_) {
            for (auto& r : wave)
                r.replay(q);
            return;
        }

        auto& cfg = configs_[sig];
        if (!cfg) {
            cfg = std::make_unique<jit::dynamic_function_config>();
            std::vector<jit::dynamic_function_id> ids;
            ids.reserve(wave.size());
            for (const auto& r : wave)
                ids.push_back(r.def_id);
            cfg->define_as_call_sequence(fused_entry_id(), ids);
        }

        std::byte* dst =
            reinterpret_cast<std::byte*>(arena_.get()) + arena_off_;
        arena_off_ = (arena_off_ + blob.size() + 63u) & ~size_t{63u};

        inflight_blobs_.push_back(std::move(blob));
        const auto& live = inflight_blobs_.back();
        q.memcpy(dst, live.data(), live.size());  // in-order: precedes kernel

        wave_launcher_(q, *cfg, dst, max_n);
    }

    // Per-component staging.
    std::vector<StageRecord> comp_records_;
    sycl::queue*             comp_queue_ = nullptr;

    // Per-layer waves.
    std::vector<std::vector<StageRecord>> waves_;
    sycl::queue*                          flush_queue_ = nullptr;

    // Signature → JIT config (lives for the process; configs must outlive
    // all kernels launched with them).
    std::unordered_map<std::vector<const void*>,
                       std::unique_ptr<jit::dynamic_function_config>, SigHash>
        configs_;

    // Launcher anchored in the stage-emitting TU (see launch_fused_wave).
    WaveLauncherFn wave_launcher_ = nullptr;

    // Device param arena (ring reset per pass) + host blobs kept alive until
    // the pass's terminal waits.
    slabkit::unique_device_ptr<uint8_t> arena_;
    size_t                              arena_cap_ = 0;
    size_t                              arena_off_ = 0;
    std::deque<std::vector<std::byte>>  inflight_blobs_;
};

// ── Active-fuser slot + RAII scope ───────────────────────────────────────────
//
// The submission choke point consults this slot; the Context installs its
// fuser for the duration of forward()/backward(). Single host thread drives
// execution (all B instances live in one Context), so a plain pointer slot
// suffices.

inline HorizontalFuser*& active_fuser_slot() {
    static HorizontalFuser* p = nullptr;
    return p;
}

class FuserScope {
public:
    explicit FuserScope(HorizontalFuser& f) {
        if (HorizontalFuser::enabled()) {
            prev_                = active_fuser_slot();
            active_fuser_slot()  = &f;
            installed_           = true;
        }
    }
    ~FuserScope() {
        if (installed_)
            active_fuser_slot() = prev_;
    }
    FuserScope(const FuserScope&)            = delete;
    FuserScope& operator=(const FuserScope&) = delete;

private:
    HorizontalFuser* prev_      = nullptr;
    bool             installed_ = false;
};

}  // namespace naga::core::fusion

namespace naga::core {

// ── execute_compute — sequenced submission of one or more kernels ────────────
//
// THE submission choke point (moved here from kernel.hpp so the collector can
// intercept). Without an active fuser this is exactly the historical path:
// in-order submission through detail::submit.

template <AnyKernel... Ks>
inline void execute_compute(sycl::queue& q, const Ks&... ks) {
    if (fusion::HorizontalFuser* f = fusion::active_fuser_slot()) {
        (f->collect(q, ks), ...);
        return;
    }
    (detail::submit(q, ks), ...);
}

}  // namespace naga::core
