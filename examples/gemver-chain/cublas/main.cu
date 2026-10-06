// examples/gemver-chain/cublas/main.cu — the GEMVER -> BiCGK -> GEMVER chain on cuBLAS.
//
//   gemver-chain-cublas [--n 16384] [--R 64] [--iters 20] [--warmup 3] [--check]
//                       [--dump <dir>] [--compare <dir>]
//
// --dump writes x1 w1 q s x2 w2 as raw f32 (cublas_<name>.f32); --compare reads the files
// the CoSpan program dumped (<variant>_<name>.f32) and prints the relative L2 against them.
//
// The same generator, host reference, CLI and output format as the CoSpan program
// (--R and --variant are accepted for parity and unused).
//
// cuBLAS is column-major, and a row-major B is a column-major M = B^T. So
//   B += u v^T           is   M += v u^T           cublasSger(n, n, 1, v, 1, u, 1, M, n)
//   x = beta B^T y + z   is   x = beta M y + z     cublasSgemv OP_N (after x = z)
//   w = alpha B x        is   w = alpha M^T x      cublasSgemv OP_T
//   q = B p              is   q = M^T p            cublasSgemv OP_T
//   s = B^T r            is   s = M r              cublasSgemv OP_N
// sger works IN PLACE, so B = A + ... first needs a device-to-device copy of A into B:
// an extra full read + write of the matrix that a fused kernel does not pay (see the README).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "common/chain_data.hpp"

namespace gd = gemver_chain;

#define CUDA_OK(call)                                                                                     \
    do {                                                                                                  \
        const cudaError_t e_ = (call);                                                                    \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e_)); \
    } while (0)
#define CUBLAS_OK(call)                                                                                   \
    do {                                                                                                  \
        const cublasStatus_t s_ = (call);                                                                 \
        if (s_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#call) + ": cublas status " + std::to_string(int(s_))); \
    } while (0)

namespace {

struct Options {
    int         n       = 16384;
    int         R       = 64;
    int         iters   = 20;
    int         warmup  = 3;
    std::string variant = "all";
    bool        check   = false;
    bool        verbose = false;
    std::string dump, compare;
    double      tol     = 1e-4;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto        val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--n") o.n = std::stoi(val());
        else if (a == "--R") o.R = std::stoi(val());
        else if (a == "--iters") o.iters = std::stoi(val());
        else if (a == "--warmup") o.warmup = std::stoi(val());
        else if (a == "--variant") o.variant = val();
        else if (a == "--tol") o.tol = std::stod(val());
        else if (a == "--check") o.check = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--dump") o.dump = val();
        else if (a == "--compare") o.compare = val();
        else throw std::invalid_argument("unknown argument " + a);
    }
    return o;
}

/// A device array of floats.
struct Dev {
    float* p = nullptr;
    explicit Dev(std::size_t count) { CUDA_OK(cudaMalloc(&p, count * sizeof(float))); }
    Dev(std::size_t count, const std::vector<float>& host) : Dev(count) {
        CUDA_OK(cudaMemcpy(p, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&)            = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<float> to_host(std::size_t count) const {
        std::vector<float> h(count);
        CUDA_OK(cudaMemcpy(h.data(), p, count * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }
};

struct Chain {
    int                n;
    cublasHandle_t     h;
    cudaStream_t       stream;
    float              alpha, beta;
    Dev                A0, u1, v1, u2, v2, y, z;
    Dev                B1, x1, w1, q, s, B2, x2, w2;

    Chain(int n_, const std::vector<float>& a0, const gd::Vectors& v)
        : n{n_}, A0{std::size_t(n_) * n_, a0}, u1{std::size_t(n_), v.u1}, v1{std::size_t(n_), v.v1},
          u2{std::size_t(n_), v.u2}, v2{std::size_t(n_), v.v2}, y{std::size_t(n_), v.y}, z{std::size_t(n_), v.z},
          B1{std::size_t(n_) * n_}, x1{std::size_t(n_)}, w1{std::size_t(n_)}, q{std::size_t(n_)}, s{std::size_t(n_)},
          B2{std::size_t(n_) * n_}, x2{std::size_t(n_)}, w2{std::size_t(n_)} {
        alpha = beta = 1.0f / std::sqrt(static_cast<float>(n));
        CUDA_OK(cudaStreamCreate(&stream));
        CUBLAS_OK(cublasCreate(&h));
        CUBLAS_OK(cublasSetStream(h, stream));
    }
    ~Chain() {
        cublasDestroy(h);
        cudaStreamDestroy(stream);
    }

    /// B = A + u1 v1^T + u2 v2^T ; x = beta B^T y + z ; w = alpha B x   (M = B^T, column-major)
    void gemver(const float* A, float* B, const float* yy, const float* zz, float* x, float* w) {
        const float one = 1.0f, zero = 0.0f;
        const std::size_t bytes = std::size_t(n) * n * sizeof(float);
        CUDA_OK(cudaMemcpyAsync(B, A, bytes, cudaMemcpyDeviceToDevice, stream));  // sger is in place
        CUBLAS_OK(cublasSger(h, n, n, &one, v1.p, 1, u1.p, 1, B, n));
        CUBLAS_OK(cublasSger(h, n, n, &one, v2.p, 1, u2.p, 1, B, n));
        CUDA_OK(cudaMemcpyAsync(x, zz, std::size_t(n) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUBLAS_OK(cublasSgemv(h, CUBLAS_OP_N, n, n, &beta, B, n, yy, 1, &one, x, 1));   // x = beta M y + z
        CUBLAS_OK(cublasSgemv(h, CUBLAS_OP_T, n, n, &alpha, B, n, x, 1, &zero, w, 1));  // w = alpha M^T x
    }

    void run() {
        const float one = 1.0f, zero = 0.0f;
        gemver(A0.p, B1.p, y.p, z.p, x1.p, w1.p);
        CUBLAS_OK(cublasSgemv(h, CUBLAS_OP_T, n, n, &one, B1.p, n, x1.p, 1, &zero, q.p, 1));  // q = B1 p, p = x1
        CUBLAS_OK(cublasSgemv(h, CUBLAS_OP_N, n, n, &one, B1.p, n, w1.p, 1, &zero, s.p, 1));  // s = B1^T r, r = w1
        gemver(B1.p, B2.p, q.p, s.p, x2.p, w2.p);
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        gd::check_problem(static_cast<std::size_t>(o.n), static_cast<std::size_t>(o.R));
        if (o.variant != "all" && o.variant != "cublas") {
            std::printf("gemver-chain-cublas: --variant %s is not a cuBLAS variant; nothing to run\n", o.variant.c_str());
            return 0;
        }
        cudaDeviceProp prop{};
        CUDA_OK(cudaGetDeviceProperties(&prop, 0));
        std::printf("gemver-chain: n=%d R=%d iters=%d warmup=%d device=\"%s\"\n", o.n, o.R, o.iters, o.warmup, prop.name);
        std::printf("effective GB/s uses a NOMINAL traffic of 8*n^2*4 bytes = %.3f GB per chain (not measured)\n",
                    gd::nominal_bytes(static_cast<std::size_t>(o.n)) / 1e9);

        const std::size_t  n = static_cast<std::size_t>(o.n);
        const auto         A0 = gd::make_matrix_A0(n);
        const gd::Vectors  vec = gd::make_vectors(n);
        Chain              chain(o.n, A0, vec);

        for (int i = 0; i < o.warmup; ++i) {
            chain.run();
            CUDA_OK(cudaStreamSynchronize(chain.stream));
        }
        cudaEvent_t e0, e1;
        CUDA_OK(cudaEventCreate(&e0));
        CUDA_OK(cudaEventCreate(&e1));
        CUDA_OK(cudaEventRecord(e0, chain.stream));
        for (int i = 0; i < o.iters; ++i) chain.run();
        CUDA_OK(cudaEventRecord(e1, chain.stream));
        CUDA_OK(cudaEventSynchronize(e1));
        float total_ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&total_ms, e0, e1));
        const double ms = total_ms / o.iters;

        std::printf("\n%-9s %12s %10s %10s\n", "variant", "ms/chain", "speedup", "GB/s");
        std::printf("%-9s %12.4f %10s %10.1f\n", "cublas", ms, "-", gd::nominal_bytes(n) / (ms * 1e6));

        bool ok = true;
        if (!o.dump.empty() || !o.compare.empty()) {
            const char* names[] = {"x1", "w1", "q", "s", "x2", "w2"};
            const Dev* devs[]   = {&chain.x1, &chain.w1, &chain.q, &chain.s, &chain.x2, &chain.w2};
            if (!o.dump.empty()) {
                std::filesystem::create_directories(o.dump);
                for (int i = 0; i < 6; ++i) {
                    const auto h = devs[i]->to_host(n);
                    std::ofstream f(o.dump + "/cublas_" + names[i] + ".f32", std::ios::binary);
                    f.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(h.size() * sizeof(float)));
                }
            }
            if (!o.compare.empty()) {
                for (const char* v : {"unfused", "parallel", "fused", "onepass"}) {
                    std::printf("  cross-check cublas vs %-8s rel L2:", v);
                    bool found = true;
                    for (int i = 0; i < 6 && found; ++i) {
                        std::ifstream f(o.compare + "/" + v + "_" + names[i] + ".f32", std::ios::binary);
                        if (!f) { found = false; break; }
                        std::vector<float> other(n);
                        f.read(reinterpret_cast<char*>(other.data()), static_cast<std::streamsize>(n * sizeof(float)));
                        std::printf(" %s %.2e", names[i], gd::relative_l2(devs[i]->to_host(n), other));
                    }
                    std::printf(found ? "\n" : " (no dump)\n");
                }
            }
        }
        if (o.check) {
            const bool with_B2 = o.n <= 4096;
            std::printf("\ncheck (tolerance %.1e, B2 %s):\n", o.tol, with_B2 ? "included" : "skipped for n > 4096");
            const gd::ChainResult ref = gd::reference_chain(n, A0, vec, with_B2);
            const std::vector<double>* want[] = {&ref.x1, &ref.w1, &ref.q, &ref.s, &ref.x2, &ref.w2};
            const std::vector<float>   got[]  = {chain.x1.to_host(n), chain.w1.to_host(n), chain.q.to_host(n),
                                                 chain.s.to_host(n),  chain.x2.to_host(n), chain.w2.to_host(n)};
            const char* names[] = {"x1", "w1", "q", "s", "x2", "w2"};
            std::printf("  check cublas   vs reference, rel L2:");
            for (int i = 0; i < 6; ++i) {
                const double e = gd::relative_l2(got[i], *want[i]);
                std::printf(" %s %.2e", names[i], e);
                if (!(e <= o.tol)) ok = false;
            }
            if (with_B2) {
                const double e = gd::relative_l2(chain.B2.to_host(n * n), ref.B2);
                std::printf(" B2 %.2e", e);
                if (!(e <= o.tol)) ok = false;
            }
            std::printf("  %s\n", ok ? "OK" : "FAIL");
        }
        std::printf("\n%s\n", ok ? "ALL OK" : "FAILED");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemver-chain-cublas: error: %s\n", e.what());
        return 2;
    }
}
