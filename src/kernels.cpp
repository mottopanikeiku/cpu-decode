#include "decode.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

namespace decode {
namespace detail {
bool avx512_supported() {
#if defined(CPU_DECODE_EMULATE_AVX512)
    return true;
#elif defined(DECODE_X86)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}
bool neon_supported() {
#if defined(DECODE_NEON) && defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}
} // namespace detail

float bf16_float(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
float half_float(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16, exponent = (h >> 10) & 0x1fu, mantissa = h & 0x3ffu, bits;
    if (exponent == 0x1f) bits = sign | 0x7f800000u | (mantissa << 13);
    else if (exponent) bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    else {
        float magnitude = float(mantissa) * 0x1p-24f;  // zero or subnormal, exact
        return sign ? -magnitude : magnitude;
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
uint16_t float_half(float value) {
    // Round-to-nearest-even conversion (F. Giesen, float_to_half_fast3_rtne).
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    bits ^= sign;
    uint16_t out;
    if (bits >= (127u + 16u) << 23) out = bits > 0x7f800000u ? 0x7e00 : 0x7c00;
    else if (bits < 113u << 23) {
        float magnitude, magic;
        const uint32_t magic_bits = 126u << 23;
        std::memcpy(&magnitude, &bits, 4);
        std::memcpy(&magic, &magic_bits, 4);
        magnitude += magic;
        uint32_t rounded;
        std::memcpy(&rounded, &magnitude, 4);
        out = uint16_t(rounded - magic_bits);
    } else {
        const uint32_t odd = (bits >> 13) & 1u;
        bits += 0xc8000fffu + odd;  // rebias exponent (15 - 127) << 23, then round
        out = uint16_t(bits >> 13);
    }
    return uint16_t(out | (sign >> 16));
}

Kernel parse_kernel(const std::string& name) {
    if (name == "auto") return detail::avx512_supported() ? Kernel::avx512 : detail::neon_supported() ? Kernel::neon : Kernel::scalar;
    if (name == "scalar") return Kernel::scalar;
    if (name == "avx512" && detail::avx512_supported()) return Kernel::avx512;
    if (name == "neon" && detail::neon_supported()) return Kernel::neon;
    throw std::runtime_error("unknown or unsupported kernel: " + name);
}
std::string kernel_name(Kernel k) {
    if (k == Kernel::avx512) return "avx512";
    if (k == Kernel::neon) return "neon";
    return "scalar";
}
Format parse_format(const std::string& name) {
    if (name == "bf16") return Format::bf16;
    if (name == "q8") return Format::q8;
    if (name == "q4") return Format::q4;
    throw std::runtime_error("unknown weight format: " + name);
}
std::string format_name(Format f) { return f == Format::q8 ? "q8" : f == Format::q4 ? "q4" : "bf16"; }
KvType parse_kv(const std::string& name) {
    if (name == "f32") return KvType::f32;
    if (name == "f16") return KvType::f16;
    throw std::runtime_error("kv must be f32 or f16");
}
std::string kv_name(KvType kv) { return kv == KvType::f16 ? "f16" : "f32"; }
WeightMemory parse_weight_memory(const std::string& name) {
    if (name == "mmap") return WeightMemory::mmap;
    if (name == "hugepage") return WeightMemory::hugepage;
    throw std::runtime_error("weights must be mmap or hugepage");
}
std::string weight_memory_name(WeightMemory m) { return m == WeightMemory::hugepage ? "hugepage" : "mmap"; }

size_t Matrix::weight_bytes() const {
    return format == Format::bf16 ? rows * cols * 2 : format == Format::q8 ? rows * cols : rows * cols / 2;
}
size_t Matrix::scale_bytes() const { return format == Format::bf16 ? 0 : rows * (cols / block_size) * 2; }

void Activation::resize(size_t size) {
    n = size;
    q.assign(size, 0);
    d.assign(size / block_size, 0.0f);
    bias.assign((size + 2 * block_size - 1) / (2 * block_size) * 16, 0);
}

static void finite_or_throw(const float* x, size_t n) {
    for (size_t j = 0; j < n; ++j)
        if (!std::isfinite(x[j])) throw std::runtime_error("nonfinite model weight");
}
void quantize_q8(const float* source, size_t n, int8_t* out, uint16_t* scales) {
    if (!n || n % block_size) throw std::runtime_error("q8 rows must be a nonzero multiple of 32");
    finite_or_throw(source, n);
    for (size_t b = 0; b < n / block_size; ++b) {
        const float* x = source + b * block_size;
        float maximum = 0;
        for (size_t j = 0; j < block_size; ++j) maximum = std::max(maximum, std::abs(x[j]));
        scales[b] = float_half(maximum / 127.0f);
        float d = half_float(scales[b]), inverse = d ? 1.0f / d : 0.0f;
        for (size_t j = 0; j < block_size; ++j)
            out[b * block_size + j] = int8_t(std::clamp(std::nearbyint(x[j] * inverse), -127.0f, 127.0f));
    }
}
void quantize_q4(const float* source, size_t n, uint8_t* out, uint16_t* scales) {
    if (!n || n % (2 * block_size)) throw std::runtime_error("q4 rows must be a nonzero multiple of 64");
    finite_or_throw(source, n);
    uint8_t levels[2 * block_size];
    for (size_t g = 0; g < n / (2 * block_size); ++g) {
        for (size_t half = 0; half < 2; ++half) {
            size_t b = 2 * g + half;
            const float* x = source + b * block_size;
            float extreme = 0;  // signed value of largest magnitude maps to level 0
            for (size_t j = 0; j < block_size; ++j) if (std::abs(x[j]) > std::abs(extreme)) extreme = x[j];
            scales[b] = float_half(extreme / -8.0f);
            float d = half_float(scales[b]), inverse = d ? 1.0f / d : 0.0f;
            for (size_t j = 0; j < block_size; ++j)
                levels[half * block_size + j] = uint8_t(std::clamp(std::floor(x[j] * inverse + 8.5f), 0.0f, 15.0f));
        }
        for (size_t j = 0; j < block_size; ++j) out[g * block_size + j] = uint8_t(levels[j] | (levels[block_size + j] << 4));
    }
}
void dequantize_row(const Matrix& m, size_t row, float* out) {
    if (m.format == Format::bf16) {
        const uint16_t* w = static_cast<const uint16_t*>(m.data) + row * m.cols;
        for (size_t j = 0; j < m.cols; ++j) out[j] = bf16_float(w[j]);
        return;
    }
    const uint16_t* s = m.scales + row * (m.cols / block_size);
    if (m.format == Format::q8) {
        const int8_t* w = static_cast<const int8_t*>(m.data) + row * m.cols;
        for (size_t j = 0; j < m.cols; ++j) out[j] = float(w[j]) * half_float(s[j / block_size]);
        return;
    }
    const uint8_t* w = static_cast<const uint8_t*>(m.data) + row * m.cols / 2;
    for (size_t g = 0; g < m.cols / (2 * block_size); ++g)
        for (size_t j = 0; j < block_size; ++j) {
            uint8_t byte = w[g * block_size + j];
            out[2 * g * block_size + j] = float(int(byte & 15) - 8) * half_float(s[2 * g]);
            out[(2 * g + 1) * block_size + j] = float(int(byte >> 4) - 8) * half_float(s[2 * g + 1]);
        }
}
void quantize_activation(const float* x, Activation& a, size_t first, size_t last) {
    for (size_t b = first; b < last; ++b) {
        const float* v = x + b * block_size;
        float maximum = 0;
        for (size_t j = 0; j < block_size; ++j) maximum = std::max(maximum, std::abs(v[j]));
        float d = maximum / 127.0f, inverse = d ? 1.0f / d : 0.0f;
        int32_t sum = 0;
        for (size_t j = 0; j < block_size; ++j) {
            int8_t q = int8_t(std::nearbyint(v[j] * inverse));
            a.q[b * block_size + j] = q;
            sum += q;
        }
        a.d[b] = d;
        a.bias[(b / 2) * 16 + (b % 2) * 8] = -128 * sum;
    }
}

void rmsnorm(const float* x, const float* weight, float* out, size_t n, float epsilon) {
    float sum = 0;
    for (size_t j = 0; j < n; ++j) sum += x[j] * x[j];
    float inv = 1.0f / std::sqrt(sum / float(n) + epsilon);
    for (size_t j = 0; j < n; ++j) out[j] = x[j] * inv * weight[j];
}
void rope(float* x, size_t heads, size_t dim, const float* cos, const float* sin) {
    for (size_t h = 0; h < heads; ++h) {
        float* v = x + h * dim;
        for (size_t j = 0; j < dim / 2; ++j) {
            float a = v[j], b = v[j + dim / 2];
            v[j] = a * cos[j] - b * sin[j];
            v[j + dim / 2] = b * cos[j] + a * sin[j];
        }
    }
}

Profile::Profile() {
    for (const char* name : {"embedding", "rmsnorm", "qkv", "rope_kv", "attention", "attention_merge", "attention_output",
                             "mlp_gate_up", "silu", "mlp_down", "lm_head", "argmax", "timing_overhead_and_loop"})
        seconds.emplace(name, 0.0);
}
Json Profile::json() const {
    uint64_t minimum = matrix_weight_bytes + scale_bytes + norm_bias_bytes + embedding_bytes + kv_write_bytes + kv_read_bytes;
    return {{"operation_seconds", seconds}, {"steps", steps}, {"head_steps", head_steps},
            {"matrix_weight_bytes", matrix_weight_bytes}, {"scale_bytes", scale_bytes},
            {"lm_head_weight_bytes", lm_head_weight_bytes}, {"lm_head_scale_bytes", lm_head_scale_bytes},
            {"norm_bias_bytes", norm_bias_bytes}, {"embedding_bytes", embedding_bytes},
            {"kv_write_bytes", kv_write_bytes}, {"kv_read_bytes", kv_read_bytes}, {"minimum_bytes", minimum},
            {"byte_model", "algorithmic weight/scale reads plus embedding row, KV writes and one read of each cached K/V row per KV head; excludes activation/cache-line/allocator traffic"}};
}
} // namespace decode
