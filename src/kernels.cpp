#include "decode.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define DECODE_X86 1
#endif

namespace decode {
float bf16_float(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
static float value(const Matrix& m, size_t i) {
    if (m.dtype == DType::bf16) return bf16_float(static_cast<const uint16_t*>(m.data)[i]);
    if (m.dtype == DType::f32) return static_cast<const float*>(m.data)[i];
    return float(static_cast<const int8_t*>(m.data)[i]);
}
static float dot_scalar(const Matrix& m, size_t row, const float* x) {
    float sum = 0;
    for (size_t j = 0; j < m.cols; ++j) sum += value(m, row * m.cols + j) * x[j];
    return m.dtype == DType::i8 ? sum * m.scales[row] : sum;
}
#ifdef DECODE_X86
__attribute__((target("avx2")))
static float dot256(const Matrix& m, size_t row, const float* x) {
    size_t j = 0, start = row * m.cols;
    __m256 sum = _mm256_setzero_ps();
    for (; j + 8 <= m.cols; j += 8) {
        __m256 w;
        if (m.dtype == DType::bf16) {
            auto bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const uint16_t*>(m.data) + start + j));
            w = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
        } else if (m.dtype == DType::i8) {
            auto bits = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(static_cast<const int8_t*>(m.data) + start + j));
            w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bits));
        } else w = _mm256_loadu_ps(static_cast<const float*>(m.data) + start + j);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(w, _mm256_loadu_ps(x + j)));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float total = 0;
    for (float lane : lanes) total += lane;
    for (; j < m.cols; ++j) total += value(m, start + j) * x[j];
    return m.dtype == DType::i8 ? total * m.scales[row] : total;
}
__attribute__((target("avx512f,avx512bw")))
static float dot512(const Matrix& m, size_t row, const float* x) {
    size_t j = 0, start = row * m.cols;
    __m512 sum = _mm512_setzero_ps();
    for (; j + 16 <= m.cols; j += 16) {
        __m512 w;
        if (m.dtype == DType::bf16) {
            auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const uint16_t*>(m.data) + start + j));
            w = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bits), 16));
        } else if (m.dtype == DType::i8) {
            auto bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const int8_t*>(m.data) + start + j));
            w = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bits));
        } else w = _mm512_loadu_ps(static_cast<const float*>(m.data) + start + j);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(w, _mm512_loadu_ps(x + j)));
    }
    float total = _mm512_reduce_add_ps(sum);
    for (; j < m.cols; ++j) total += value(m, start + j) * x[j];
    return m.dtype == DType::i8 ? total * m.scales[row] : total;
}
#endif
Kernel parse_kernel(const std::string& name) {
    if (name == "scalar") return Kernel::scalar;
#ifdef DECODE_X86
    __builtin_cpu_init();
    if (name == "simd256" && __builtin_cpu_supports("avx2")) return Kernel::simd256;
    if (name == "simd512" && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) return Kernel::simd512;
#endif
    throw std::runtime_error("unknown or unsupported kernel: " + name);
}
std::string kernel_name(Kernel k) {
    return k == Kernel::scalar ? "scalar" : k == Kernel::simd256 ? "simd256" : "simd512";
}
void matvec(const Matrix& m, const float* x, float* y, Kernel kernel, int threads) {
    if (!m.data || !m.rows || !m.cols || (m.dtype == DType::i8 && !m.scales) || threads < 1)
        throw std::runtime_error("invalid matvec arguments");
    float (*dot)(const Matrix&, size_t, const float*) = dot_scalar;
#ifdef DECODE_X86
    if (kernel == Kernel::simd256) dot = dot256;
    if (kernel == Kernel::simd512) dot = dot512;
#else
    if (kernel != Kernel::scalar) throw std::runtime_error("SIMD requires x86");
#endif
    #pragma omp parallel for num_threads(threads) schedule(static) if(threads > 1)
    for (size_t row = 0; row < m.rows; ++row) y[row] = dot(m, row, x);
}
void quantize_row(const float* source, size_t n, int8_t* out, float& scale) {
    if (!n) throw std::runtime_error("empty quantization row");
    float maximum = 0;
    for (size_t j = 0; j < n; ++j) {
        if (!std::isfinite(source[j])) throw std::runtime_error("nonfinite model weight");
        maximum = std::max(maximum, std::abs(source[j]));
    }
    scale = maximum == 0 ? 1.0f : maximum / 127.0f;
    for (size_t j = 0; j < n; ++j)
        out[j] = static_cast<int8_t>(std::clamp(std::round(source[j] / scale), -127.0f, 127.0f));
}
void rmsnorm(const float* x, const float* weight, float* out, size_t n, float epsilon) {
    float sum = 0;
    for (size_t j = 0; j < n; ++j) sum += x[j] * x[j];
    float inv = 1.0f / std::sqrt(sum / float(n) + epsilon);
    for (size_t j = 0; j < n; ++j) out[j] = x[j] * inv * weight[j];
}
void rope(float* x, size_t heads, size_t dim, size_t position, float theta) {
    for (size_t j = 0; j < dim / 2; ++j) {
        float angle = float(position) / std::pow(theta, float(2 * j) / float(dim));
        float c = std::cos(angle), s = std::sin(angle);
        for (size_t h = 0; h < heads; ++h) {
            float* v = x + h * dim;
            float a = v[j], b = v[j + dim / 2];
            v[j] = a * c - b * s;
            v[j + dim / 2] = b * c + a * s;
        }
    }
}
void attention(const float* q, const float* keys, const float* values, float* out,
               float* scores, size_t length, size_t heads, size_t kv_heads,
               size_t dim, int threads) {
    if (!length || !kv_heads || heads % kv_heads || !dim || threads < 1)
        throw std::runtime_error("invalid attention dimensions");
    const size_t stride = kv_heads * dim;
    const float scale = 1.0f / std::sqrt(float(dim));
    #pragma omp parallel for num_threads(threads) schedule(static) if(threads > 1)
    for (size_t h = 0; h < heads; ++h) {
        size_t kv = h / (heads / kv_heads);
        float* probability = scores + h * length;
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t t = 0; t < length; ++t) {
            float dot = 0;
            for (size_t j = 0; j < dim; ++j) dot += q[h * dim + j] * keys[t * stride + kv * dim + j];
            probability[t] = dot * scale;
            maximum = std::max(maximum, probability[t]);
        }
        float total = 0;
        for (size_t t = 0; t < length; ++t) {
            probability[t] = std::exp(probability[t] - maximum);
            total += probability[t];
        }
        std::fill(out + h * dim, out + (h + 1) * dim, 0.0f);
        for (size_t t = 0; t < length; ++t) {
            float p = probability[t] / total;
            for (size_t j = 0; j < dim; ++j) out[h * dim + j] += p * values[t * stride + kv * dim + j];
        }
    }
}
Profile::Profile() {
    for (const char* name : {"embedding", "rmsnorm", "qkv", "rope", "kv_write", "attention", "attention_output", "residual", "mlp_gate_up", "silu", "mlp_down", "lm_head", "argmax", "timing_overhead_and_loop"})
        seconds.emplace(name, 0.0);
}
Json Profile::json() const {
    uint64_t minimum = matrix_weight_bytes + scale_bytes + norm_bias_bytes + embedding_bytes + kv_write_bytes + kv_read_min_bytes;
    return {{"operation_seconds", seconds}, {"steps", steps}, {"head_steps", head_steps},
            {"matrix_weight_bytes", matrix_weight_bytes}, {"scale_bytes", scale_bytes},
            {"lm_head_weight_bytes", lm_head_weight_bytes}, {"lm_head_scale_bytes", lm_head_scale_bytes},
            {"norm_bias_bytes", norm_bias_bytes}, {"embedding_bytes", embedding_bytes},
            {"kv_write_bytes", kv_write_bytes}, {"kv_read_min_bytes", kv_read_min_bytes},
            {"kv_read_logical_bytes", kv_read_logical_bytes}, {"minimum_bytes", minimum},
            {"byte_model", "algorithmic weight/scale reads plus embedding row, FP32 KV writes and minimum GQA KV reads; excludes activation/cache-line/allocator traffic"}};
}
} // namespace decode
