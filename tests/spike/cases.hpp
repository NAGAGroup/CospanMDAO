#pragma once
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "stages.hpp"

namespace spike {

using entry_fn = sycl::AdaptiveCpp_jit::dynamic_function<void, sycl::nd_item<1>, FnOffsetInfo&,
                                                         const std::byte*, std::byte*>;
using def_fn   = sycl::AdaptiveCpp_jit::dynamic_function_definition<void, sycl::nd_item<1>,
                                                                   FnOffsetInfo&, const std::byte*,
                                                                   std::byte*>;

inline int check(const char* name, const std::vector<float>& got, auto expect) {
    for (std::size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - expect(i)) > 1e-4f) {
            std::printf("FAIL %s at %zu: got %f want %f\n", name, i, got[i], expect(i));
            return 1;
        }
    std::printf("PASS %s\n", name);
    return 0;
}

template <class T>
T* upload(sycl::queue& q, const std::vector<T>& h) {
    T* d = sycl::malloc_device<T>(h.size(), q);
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}
inline std::vector<float> download(sycl::queue& q, const float* d, std::size_t n) {
    std::vector<float> h(n);
    q.memcpy(h.data(), d, n * sizeof(float)).wait();
    return h;
}
template <class... B>
std::vector<std::byte> pack(std::vector<std::uint32_t>& offs, const B&... bodies) {
    std::vector<std::byte> out;
    auto put = [&](const auto& b) {
        offs.push_back(static_cast<std::uint32_t>(out.size()));
        const auto* p = reinterpret_cast<const std::byte*>(&b);
        out.insert(out.end(), p, p + sizeof(b));
        while (out.size() % 16) out.push_back(std::byte{0});
    };
    (put(bodies), ...);
    return out;
}

inline int run_all(sycl::queue& q) {
    int fails = 0;
    constexpr std::size_t N = 1000, WG = 64, G = (N + WG - 1) / WG;
    float* a = sycl::malloc_device<float>(N, q);
    float* b = sycl::malloc_device<float>(N, q);
    std::vector<float> ramp(N);
    for (std::size_t i = 0; i < N; ++i) ramp[i] = float(i);
    float* in = upload(q, ramp);

    // 1. single define: Scale
    {
        sycl::AdaptiveCpp_jit::dynamic_function_config cfg;
        cfg.define(&kb_entry_r1, &kb_fn<Scale>);
        std::vector<std::uint32_t> offs;
        auto bodies = pack(offs, Scale{a, 3.f});
        std::vector<StageRecord> recs{{N, 0, G, offs[0]}};
        auto* dr = upload(q, recs); auto* db = upload(q, bodies);
        launch_r1(q, cfg, G, WG, dr, db, 0);
        fails += check("define/Scale", download(q, a, N), [](std::size_t i) { return 3.f * i; });
        sycl::free(dr, q); sycl::free(db, q);
    }
    // 2. sequential call sequence via type-erased ids: Scale(a) then AddOne(a->b), shared cursor
    {
        sycl::AdaptiveCpp_jit::dynamic_function_config cfg;
        cfg.define_as_call_sequence(entry_fn{&kb_entry_r1}.id(),
                                    {def_fn{&kb_fn<Scale>}.id(), def_fn{&kb_fn<AddOne>}.id()});
        std::vector<std::uint32_t> offs;
        auto bodies = pack(offs, Scale{a, 2.f}, AddOne{a, b});
        std::vector<StageRecord> recs{{N, 0, G, offs[0]}, {N, 0, G, offs[1]}};
        auto* dr = upload(q, recs); auto* db = upload(q, bodies);
        launch_r1(q, cfg, G, WG, dr, db, 0);
        fails += check("sequence/Scale+AddOne", download(q, b, N), [](std::size_t i) { return 2.f * i + 1.f; });
        sycl::free(dr, q); sycl::free(db, q);
    }
    // 3. parallel mode: two stages on disjoint group slices, same wave
    {
        sycl::AdaptiveCpp_jit::dynamic_function_config cfg;
        cfg.define_as_call_sequence(entry_fn{&kb_entry_r1}.id(),
                                    {def_fn{&kb_fn<Scale>}.id(), def_fn{&kb_fn<AddOne>}.id()});
        std::vector<std::uint32_t> offs;
        auto bodies = pack(offs, Scale{a, 5.f}, AddOne{in, b});
        std::vector<StageRecord> recs{{N, 0, G, offs[0]}, {N, G, G, offs[1]}};
        auto* dr = upload(q, recs); auto* db = upload(q, bodies);
        launch_r1(q, cfg, 2 * G, WG, dr, db, 0);
        fails += check("parallel/Scale", download(q, a, N), [](std::size_t i) { return 5.f * i; });
        fails += check("parallel/AddOne", download(q, b, N), [](std::size_t i) { return float(i) + 1.f; });
        sycl::free(dr, q); sycl::free(db, q);
    }
    // 4. group body with local memory and a barrier
    {
        sycl::AdaptiveCpp_jit::dynamic_function_config cfg;
        cfg.define(&kb_entry_r1, &kb_fn<Reverse>);
        std::vector<std::uint32_t> offs;
        auto bodies = pack(offs, Reverse{in, b});
        std::vector<StageRecord> recs{{N, 0, G, offs[0]}};
        auto* dr = upload(q, recs); auto* db = upload(q, bodies);
        launch_r1(q, cfg, G, WG, dr, db, WG * sizeof(float));
        fails += check("group/Reverse", download(q, b, N), [&](std::size_t i) {
            const std::size_t g = i / WG, l = i % WG, src = g * WG + (WG - 1 - l);
            return src < N ? float(src) : 0.f;
        });
        sycl::free(dr, q); sycl::free(db, q);
    }
    sycl::free(a, q); sycl::free(b, q); sycl::free(in, q);
    std::printf("%s\n", fails ? "SPIKE FAILED" : "SPIKE PASSED");
    return fails;
}

}  // namespace spike
