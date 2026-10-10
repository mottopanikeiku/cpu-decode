// Read-only STREAM-style sweep, not a memory-controller measurement.
#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <omp.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static volatile uint64_t sink = 0;

#if defined(__x86_64__)
__attribute__((target("avx2"), noinline))
static uint64_t read256(const uint64_t* p, size_t n, int passes) {
    __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
    uint64_t rest = 0;
    for (int pass = 0; pass < passes; ++pass) {
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            a = _mm256_xor_si256(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i)));
            b = _mm256_xor_si256(b, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i+4)));
            c = _mm256_xor_si256(c, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i+8)));
            d = _mm256_xor_si256(d, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i+12)));
        }
        for (; i < n; ++i) rest ^= p[i];
        // Prevent the compiler folding identical even-numbered sweeps to zero.
        asm volatile("" : "+x"(a), "+x"(b), "+x"(c), "+x"(d) : : "memory");
    }
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),
                      _mm256_xor_si256(_mm256_xor_si256(a,b), _mm256_xor_si256(c,d)));
    return rest ^ lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3];
}

__attribute__((target("avx512f"), noinline))
static uint64_t read512(const uint64_t* p, size_t n, int passes) {
    __m512i a = _mm512_setzero_si512(), b = a, c = a, d = a;
    uint64_t rest = 0;
    for (int pass = 0; pass < passes; ++pass) {
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            a = _mm512_xor_si512(a, _mm512_loadu_si512(p+i));
            b = _mm512_xor_si512(b, _mm512_loadu_si512(p+i+8));
            c = _mm512_xor_si512(c, _mm512_loadu_si512(p+i+16));
            d = _mm512_xor_si512(d, _mm512_loadu_si512(p+i+24));
        }
        for (; i < n; ++i) rest ^= p[i];
        asm volatile("" : "+v"(a), "+v"(b), "+v"(c), "+v"(d) : : "memory");
    }
    alignas(64) uint64_t lanes[8];
    _mm512_store_si512(lanes, _mm512_xor_si512(_mm512_xor_si512(a,b), _mm512_xor_si512(c,d)));
    for (auto x : lanes) rest ^= x;
    return rest;
}
#elif defined(__aarch64__)
__attribute__((noinline))
static uint64_t read_neon(const uint64_t* p, size_t n, int passes) {
    uint64x2_t a = vdupq_n_u64(0), b = a, c = a, d = a;
    uint64_t rest = 0;
    for (int pass = 0; pass < passes; ++pass) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            a = veorq_u64(a, vld1q_u64(p+i));
            b = veorq_u64(b, vld1q_u64(p+i+2));
            c = veorq_u64(c, vld1q_u64(p+i+4));
            d = veorq_u64(d, vld1q_u64(p+i+6));
        }
        for (; i < n; ++i) rest ^= p[i];
        asm volatile("" : "+w"(a), "+w"(b), "+w"(c), "+w"(d) : : "memory");
    }
    uint64x2_t all = veorq_u64(veorq_u64(a, b), veorq_u64(c, d));
    return rest ^ vgetq_lane_u64(all, 0) ^ vgetq_lane_u64(all, 1);
}
#endif

static uint64_t read(const std::string& kernel, const uint64_t* p, size_t n, int passes) {
#if defined(__x86_64__)
    return kernel == "simd256" ? read256(p, n, passes) : read512(p, n, passes);
#elif defined(__aarch64__)
    (void)kernel;
    return read_neon(p, n, passes);
#endif
}

static bool available(const std::string& kernel) {
#if defined(__x86_64__)
    return (kernel == "simd256" && __builtin_cpu_supports("avx2")) || (kernel == "simd512" && __builtin_cpu_supports("avx512f"));
#elif defined(__aarch64__)
    return kernel == "neon";
#else
    return false;
#endif
}

int main(int argc, char** argv) {
    try {
        int threads = 6, mib = 256, passes = 128, repeats = 5;
        std::string kernel = "simd512";
        for (int i = 1; i < argc; i += 2) {
            if (i+1 == argc) throw std::runtime_error("missing option value");
            const std::string option = argv[i], value = argv[i+1];
            if (option == "--threads") threads = std::stoi(value);
            else if (option == "--mib") mib = std::stoi(value);
            else if (option == "--passes") passes = std::stoi(value);
            else if (option == "--repeats") repeats = std::stoi(value);
            else if (option == "--kernel") kernel = value;
            else throw std::runtime_error("unknown option: " + option);
        }
        if (threads < 1 || threads > 256 || mib < 32 || mib > 1024 || passes < 1 || repeats < 1)
            throw std::runtime_error("invalid sweep parameters");
        if (!available(kernel)) throw std::runtime_error("invalid or unavailable kernel: " + kernel);
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        const size_t bytes = static_cast<size_t>(mib) * 1024 * 1024, n = bytes / sizeof(uint64_t);
        std::unique_ptr<uint64_t, decltype(&std::free)> data(
            static_cast<uint64_t*>(std::aligned_alloc(64, bytes)), &std::free);
        if (!data) throw std::runtime_error("allocation failed");
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) data.get()[i] = i * UINT64_C(6364136223846793005) + 1;
        auto sweep = [&](int count) {
            uint64_t checksum = 0;
            #pragma omp parallel reduction(^:checksum)
            {
                const size_t id = omp_get_thread_num(), total = omp_get_num_threads();
                const size_t start = n*id/total, end = n*(id+1)/total;
                checksum ^= read(kernel, data.get()+start, end-start, count);
            }
            sink = checksum;
        };
        sweep(3);
        std::cout << "{\"kind\":\"read_bandwidth\",\"kernel\":\"" << kernel
                  << "\",\"threads\":" << threads << ",\"array_bytes\":" << bytes
                  << ",\"passes\":" << passes << ",\"samples\":[";
        for (int r = 0; r < repeats; ++r) {
            const auto start = std::chrono::steady_clock::now();
            sweep(passes);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if (r) std::cout << ',';
            std::cout << "{\"seconds\":" << seconds << ",\"GB_per_s\":"
                      << static_cast<double>(bytes)*passes/seconds/1e9 << '}';
        }
        std::cout << "],\"checksum\":" << sink << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
