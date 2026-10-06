// =============================================================================
// test_acpp_dynfn.cpp — probe: AdaptiveCpp dynamic functions across translation units
// =============================================================================
//
// What the toolchain has to support for the library's kernels to use
// definitions from other translation units, and indexed call sequences:
//
//   (a) define                          -- one definition, in another TU
//   (b) define_as_call_sequence         -- two definitions, in another TU
//   (c) define_as_indexed_call_sequence -- three definitions, in another TU
//   (d) define_as_indexed_call_sequence -- one definition, in another TU
//   (e) the same indexed sequence of three, with the definitions in THIS TU
//       (the control: it separates "indexed" from "cross-TU" when something fails)
//
// The definitions of (a)-(d) are SYCL_EXTERNAL and live only in
// acpp_dynfn_defs.cpp; the kernels and the configurations live here. Every
// definition writes to a slot of its own, so a wrong mapping is visible.

#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <sycl/sycl.hpp>

// defined in acpp_dynfn_defs.cpp
SYCL_EXTERNAL void probe_ext_set(int* out, sycl::item<1> idx);
SYCL_EXTERNAL void probe_ext_seq0(int* out, sycl::item<1> idx);
SYCL_EXTERNAL void probe_ext_seq1(int* out, sycl::item<1> idx);
SYCL_EXTERNAL void probe_ext_idx0(int* out, int k, sycl::item<1> idx);
SYCL_EXTERNAL void probe_ext_idx1(int* out, int k, sycl::item<1> idx);
SYCL_EXTERNAL void probe_ext_idx2(int* out, int k, sycl::item<1> idx);
SYCL_EXTERNAL void probe_ext_one0(int* out, int k, sycl::item<1> idx);

// the dynamic functions, with no definition anywhere
void probe_dyn_define(int* out, sycl::item<1> idx);
void probe_dyn_sequence(int* out, sycl::item<1> idx);
void probe_dyn_indexed3(int* out, int k, sycl::item<1> idx);
void probe_dyn_indexed1(int* out, int k, sycl::item<1> idx);
void probe_dyn_local(int* out, int k, sycl::item<1> idx);

// (e) the control: the same shape as (c), defined in this TU
SYCL_EXTERNAL void probe_local0(int* out, int k, sycl::item<1>) { out[k] = k * 10 + 1; }
SYCL_EXTERNAL void probe_local1(int* out, int k, sycl::item<1>) { out[k] = k * 10 + 2; }
SYCL_EXTERNAL void probe_local2(int* out, int k, sycl::item<1>) { out[k] = k * 10 + 3; }

namespace {

constexpr int kSlots = 4;

/// Runs `launch(q, data)` over four zeroed ints and returns them.
///
/// The dynamic_function_config the launch uses is declared by the TEST_CASE, so
/// it outlives this call: AdaptiveCpp JIT-compiles on its worker thread after
/// submission and reads the config then, so the config must stay alive until
/// every kernel using it has completed -- which is the wait below.
template <class Launch>
std::vector<int> run(Launch&& launch) {
    sycl::queue q{sycl::default_selector_v};
    int*        data = sycl::malloc_device<int>(kSlots, q);
    q.memset(data, 0, kSlots * sizeof(int)).wait();
    launch(q, data);
    q.wait_and_throw();

    std::vector<int> host(kSlots);
    q.memcpy(host.data(), data, kSlots * sizeof(int)).wait();
    sycl::free(data, q);
    return host;
}

}  // namespace

TEST_CASE("dynamic functions: a define across translation units", "[acpp_dynfn]") {
    sycl::AdaptiveCpp_jit::dynamic_function_config cfg;  // outlives run()'s wait
    const auto out = run([&cfg](sycl::queue& q, int* data) {
        cfg.define(&probe_dyn_define, &probe_ext_set);
        q.parallel_for(sycl::range{1},
                       cfg.apply([=](sycl::item<1> idx) { probe_dyn_define(data, idx); }));
    });
    CHECK(out == std::vector<int>{0, 42, 0, 0});
}

TEST_CASE("dynamic functions: a call sequence across translation units", "[acpp_dynfn]") {
    sycl::AdaptiveCpp_jit::dynamic_function_config cfg;  // outlives run()'s wait
    const auto out = run([&cfg](sycl::queue& q, int* data) {
        cfg.define_as_call_sequence(&probe_dyn_sequence, {&probe_ext_seq0, &probe_ext_seq1});
        q.parallel_for(sycl::range{1},
                       cfg.apply([=](sycl::item<1> idx) { probe_dyn_sequence(data, idx); }));
    });
    CHECK(out == std::vector<int>{0, 0, 3, 4});
}

TEST_CASE("dynamic functions: an indexed call sequence of three across translation units",
          "[acpp_dynfn]") {
    sycl::AdaptiveCpp_jit::dynamic_function_config cfg;  // outlives run()'s wait
    const auto out = run([&cfg](sycl::queue& q, int* data) {
        cfg.define_as_indexed_call_sequence(
            &probe_dyn_indexed3, {&probe_ext_idx0, &probe_ext_idx1, &probe_ext_idx2}, 1);
        q.parallel_for(sycl::range{1}, cfg.apply([=](sycl::item<1> idx) {
                           probe_dyn_indexed3(data, 99, idx);  // 99 is replaced by k
                       }));
    });
    // out[k] = k*10 + (k+1); the last slot is untouched
    CHECK(out == std::vector<int>{1, 12, 23, 0});
}

TEST_CASE("dynamic functions: an indexed call sequence of one across translation units",
          "[acpp_dynfn]") {
    sycl::AdaptiveCpp_jit::dynamic_function_config cfg;  // outlives run()'s wait
    const auto out = run([&cfg](sycl::queue& q, int* data) {
        cfg.define_as_indexed_call_sequence(&probe_dyn_indexed1, {&probe_ext_one0}, 1);
        q.parallel_for(sycl::range{1}, cfg.apply([=](sycl::item<1> idx) {
                           probe_dyn_indexed1(data, 99, idx);  // 99 is replaced by 0
                       }));
    });
    CHECK(out == std::vector<int>{7, 0, 0, 0});
}

TEST_CASE("dynamic functions: an indexed call sequence in one translation unit (control)",
          "[acpp_dynfn]") {
    sycl::AdaptiveCpp_jit::dynamic_function_config cfg;  // outlives run()'s wait
    const auto out = run([&cfg](sycl::queue& q, int* data) {
        cfg.define_as_indexed_call_sequence(&probe_dyn_local,
                                            {&probe_local0, &probe_local1, &probe_local2}, 1);
        q.parallel_for(sycl::range{1}, cfg.apply([=](sycl::item<1> idx) {
                           probe_dyn_local(data, 99, idx);
                       }));
    });
    CHECK(out == std::vector<int>{1, 12, 23, 0});
}
