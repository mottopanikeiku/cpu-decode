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
    float result; std::memcpy(&result, &bits, 4); return result;
}
float half_float(uint16_t value) {
    uint32_t sign = uint32_t(value & 0x8000) << 16, exponent = (value >> 10) & 31, fraction = value & 1023;
    uint32_t bits;
    if (!exponent) {
        if (!fraction) bits = sign;
        else {
            int e = -14;
            while (!(fraction & 1024)) { fraction <<= 1; --e; }
            bits = sign | (uint32_t(e + 127) << 23) | ((fraction & 1023) << 13);
        }
    } else if (exponent == 31) bits = sign | 0x7f800000u | (fraction << 13);
    else bits = sign | ((exponent + 112) << 23) | (fraction << 13);
    float result; std::memcpy(&result, &bits, 4); return result;
}
uint16_t float_half(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    uint16_t sign = uint16_t((bits >> 16) & 0x8000);
    uint32_t exponent = (bits >> 23) & 255, fraction = bits & 0x7fffff;
    if (exponent == 255) return uint16_t(sign | 0x7c00 | (fraction ? 0x200 : 0));
    int e = int(exponent) - 127;
    if (e > 15) return uint16_t(sign | 0x7c00);
    if (e < -25) return sign;
    if (e < -14) {
        fraction |= 0x800000;
        unsigned shift = unsigned(-e - 1);
        uint32_t rounded = fraction >> shift, remainder = fraction & ((uint32_t(1) << shift) - 1);
        uint32_t halfway = uint32_t(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
        return uint16_t(sign | rounded);
    }
    uint32_t rounded = fraction >> 13, remainder = fraction & 8191;
    if (remainder > 4096 || (remainder == 4096 && (rounded & 1))) ++rounded;
    return uint16_t(sign | ((uint32_t(e + 15) << 10) + rounded));
}
float matrix_scale(const Matrix& m, size_t row, size_t column) {
    size_t groups = m.group_size ? m.cols / m.group_size : 1;
    size_t index = row * groups + (m.group_size ? column / m.group_size : 0);
    return m.scale_dtype == DType::f16 ? half_float(static_cast<const uint16_t*>(m.scales)[index]) : static_cast<const float*>(m.scales)[index];
}
uint64_t matrix_scale_bytes(const Matrix& m) {
    return m.dtype == DType::i8 ? m.rows * (m.group_size ? m.cols / m.group_size : 1) * (m.scale_dtype == DType::f16 ? 2 : 4) : 0;
}
static float value(const Matrix& m, size_t index) {
    if (m.dtype == DType::bf16) return bf16_float(static_cast<const uint16_t*>(m.data)[index]);
    if (m.dtype == DType::f32) return static_cast<const float*>(m.data)[index];
    return float(static_cast<const int8_t*>(m.data)[index]);
}
static float dot_scalar(const Matrix& m, size_t row, const float* x) {
    float sum = 0;
    if (m.dtype == DType::i8 && m.group_size) {
        for (size_t begin = 0; begin < m.cols; begin += m.group_size) {
            float scale = matrix_scale(m, row, begin);
            for (size_t j = begin; j < begin + m.group_size; ++j) sum += (value(m, row * m.cols + j) * scale) * x[j];
        }
    } else {
        for (size_t j = 0; j < m.cols; ++j) sum += value(m, row * m.cols + j) * x[j];
        if (m.dtype == DType::i8) sum *= matrix_scale(m, row, 0);
    }
    return sum;
}
static void batch_dot_scalar(const Matrix& m, size_t row, const float* x, size_t stride,
                             size_t columns, const Activation*, float* result) {
    std::fill_n(result, columns, 0.0f);
    if (m.dtype == DType::i8 && m.group_size) {
        for (size_t begin = 0; begin < m.cols; begin += m.group_size) {
            float scale = matrix_scale(m, row, begin);
            for (size_t j = begin; j < begin + m.group_size; ++j) {
                float w = value(m, row * m.cols + j) * scale;
                for (size_t c = 0; c < columns; ++c) result[c] += w * x[c * stride + j];
            }
        }
    } else for (size_t j = 0; j < m.cols; ++j) {
        float w = value(m, row * m.cols + j);
        for (size_t c = 0; c < columns; ++c) result[c] += w * x[c * stride + j];
    }
    if (m.dtype == DType::i8 && !m.group_size)
        for (size_t c = 0; c < columns; ++c) result[c] *= matrix_scale(m, row, 0);
}
#ifdef DECODE_X86
__attribute__((target("avx2,f16c"), always_inline))
static inline float scale_fast(const Matrix& m, size_t row, size_t column, unsigned shift) {
    size_t index = m.group_size ? row * (m.cols >> shift) + (column >> shift) : row;
    if (m.scale_dtype == DType::f16) return _cvtsh_ss(static_cast<const uint16_t*>(m.scales)[index]);
    return static_cast<const float*>(m.scales)[index];
}
__attribute__((target("avx2,f16c"), always_inline))
static inline float row_scale_fast(const Matrix& m, size_t row) {
    return m.scale_dtype == DType::f16 ? _cvtsh_ss(static_cast<const uint16_t*>(m.scales)[row]) : static_cast<const float*>(m.scales)[row];
}
template<bool Grouped>
__attribute__((target("avx2,f16c,fma"), noinline))
static float dot256_impl(const Matrix& m, size_t row, const float* x) {
    size_t j = 0, start = row * m.cols;
    unsigned shift = Grouped ? unsigned(__builtin_ctzll(m.group_size)) : 0;
    __m256 sum = _mm256_setzero_ps();
    for (; j + 8 <= m.cols; j += 8) {
        __m256 w;
        if (m.dtype == DType::bf16) {
            auto bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const uint16_t*>(m.data) + start + j));
            w = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
        } else if (m.dtype == DType::i8) {
            auto bits = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(static_cast<const int8_t*>(m.data) + start + j));
            w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bits));
            if constexpr (Grouped) w = _mm256_mul_ps(w, _mm256_set1_ps(scale_fast(m, row, j, shift)));
        } else w = _mm256_loadu_ps(static_cast<const float*>(m.data) + start + j);
        sum = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + j), sum);
    }
    alignas(32) float lanes[8]; _mm256_store_ps(lanes, sum);
    float total = 0; for (float lane : lanes) total += lane;
    for (; j < m.cols; ++j) total += value(m, start + j) * x[j] * (m.dtype == DType::i8 && Grouped ? matrix_scale(m, row, j) : 1);
    return m.dtype == DType::i8 && !Grouped ? total * row_scale_fast(m, row) : total;
}
template<bool Grouped>
__attribute__((target("avx512f,avx512bw,f16c"), always_inline))
static inline __m512 load512(const Matrix& m, size_t index, size_t row, size_t column, unsigned shift) {
    if (m.dtype == DType::bf16) {
        auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const uint16_t*>(m.data) + index));
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bits), 16));
    }
    if (m.dtype == DType::i8) {
        auto bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const int8_t*>(m.data) + index));
        auto w = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bits));
        if constexpr (Grouped) return _mm512_mul_ps(w, _mm512_set1_ps(scale_fast(m, row, column, shift)));
        return w;
    }
    return _mm512_loadu_ps(static_cast<const float*>(m.data) + index);
}
template<bool Grouped>
__attribute__((target("avx512f,avx512bw,f16c"), noinline))
static float dot512_impl(const Matrix& m, size_t row, const float* x) {
    size_t j = 0, start = row * m.cols;
    unsigned shift = Grouped ? unsigned(__builtin_ctzll(m.group_size)) : 0;
    __m512 sum = _mm512_setzero_ps();
    for (; j + 16 <= m.cols; j += 16) sum = _mm512_fmadd_ps(load512<Grouped>(m, start + j, row, j, shift), _mm512_loadu_ps(x + j), sum);
    float total = _mm512_reduce_add_ps(sum);
    for (; j < m.cols; ++j) total += value(m, start + j) * x[j] * (m.dtype == DType::i8 && Grouped ? matrix_scale(m, row, j) : 1);
    return m.dtype == DType::i8 && !Grouped ? total * row_scale_fast(m, row) : total;
}
template<bool Grouped>
__attribute__((target("avx512f,avx512bw,f16c"), noinline))
static float dot512x4_impl(const Matrix& m, size_t row, const float* x) {
    if constexpr (Grouped) if (m.dtype != DType::i8) return dot512x4_impl<false>(m, row, x);
    size_t j = 0, start = row * m.cols;
    unsigned shift = Grouped ? unsigned(__builtin_ctzll(m.group_size)) : 0;
    __m512 a = _mm512_setzero_ps(), b = a, c = a, d = a;
    for (; j + 64 <= m.cols; j += 64) {
        if constexpr (Grouped) {
            // Each group spans at least two vectors. Decode its scale once,
            // rather than repeating half conversion and broadcast per vector.
            __m512 first = _mm512_set1_ps(scale_fast(m, row, j, shift));
            __m512 second = m.group_size == 32 ? _mm512_set1_ps(scale_fast(m, row, j + 32, shift)) : first;
            a = _mm512_fmadd_ps(_mm512_mul_ps(load512<false>(m, start + j, row, j, shift), first), _mm512_loadu_ps(x + j), a);
            b = _mm512_fmadd_ps(_mm512_mul_ps(load512<false>(m, start + j + 16, row, j + 16, shift), first), _mm512_loadu_ps(x + j + 16), b);
            c = _mm512_fmadd_ps(_mm512_mul_ps(load512<false>(m, start + j + 32, row, j + 32, shift), second), _mm512_loadu_ps(x + j + 32), c);
            d = _mm512_fmadd_ps(_mm512_mul_ps(load512<false>(m, start + j + 48, row, j + 48, shift), second), _mm512_loadu_ps(x + j + 48), d);
        } else {
            a = _mm512_fmadd_ps(load512<false>(m, start + j, row, j, shift), _mm512_loadu_ps(x + j), a);
            b = _mm512_fmadd_ps(load512<false>(m, start + j + 16, row, j + 16, shift), _mm512_loadu_ps(x + j + 16), b);
            c = _mm512_fmadd_ps(load512<false>(m, start + j + 32, row, j + 32, shift), _mm512_loadu_ps(x + j + 32), c);
            d = _mm512_fmadd_ps(load512<false>(m, start + j + 48, row, j + 48, shift), _mm512_loadu_ps(x + j + 48), d);
        }
    }
    for (; j + 16 <= m.cols; j += 16) a = _mm512_fmadd_ps(load512<Grouped>(m, start + j, row, j, shift), _mm512_loadu_ps(x + j), a);
    float total = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a, b), _mm512_add_ps(c, d)));
    for (; j < m.cols; ++j) total += value(m, start + j) * x[j] * (m.dtype == DType::i8 && Grouped ? matrix_scale(m, row, j) : 1);
    return m.dtype == DType::i8 && !Grouped ? total * row_scale_fast(m, row) : total;
}
static float dot256(const Matrix& m, size_t row, const float* x) {
    return m.group_size ? dot256_impl<true>(m, row, x) : dot256_impl<false>(m, row, x);
}
static float dot512(const Matrix& m, size_t row, const float* x) {
    return m.group_size ? dot512_impl<true>(m, row, x) : dot512_impl<false>(m, row, x);
}
static float dot512x4(const Matrix& m, size_t row, const float* x) {
    return m.group_size ? dot512x4_impl<true>(m, row, x) : dot512x4_impl<false>(m, row, x);
}
__attribute__((target("avx512vnni,avx512vl,avx512bw,avx2,f16c")))
static float dot_vnni(const Matrix& m, size_t row, const Activation& x) {
    float total = 0;
    unsigned shift = m.group_size ? unsigned(__builtin_ctzll(m.group_size)) : 0;
    const auto* weights = static_cast<const int8_t*>(m.data) + row * m.cols;
    const __m256i correction = _mm256_set1_epi8(char(0x80));
    for (size_t j = 0; j < m.cols; j += 32) {
        auto w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + j));
        auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.bytes.data() + j));
        auto dot = _mm256_sub_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), a, w), _mm256_dpbusd_epi32(_mm256_setzero_si256(), correction, w));
        auto sum = _mm_add_epi32(_mm256_castsi256_si128(dot), _mm256_extracti128_si256(dot, 1));
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
        total += float(_mm_cvtsi128_si32(sum)) * (x.scales[j / 32] * scale_fast(m, row, j, shift));
    }
    return total;
}
static float dot_vnni16_wide(const Matrix& m, size_t row, const Activation& x) {
    const auto* weights = static_cast<const int8_t*>(m.data) + row * m.cols;
    double sum = 0;
    for (size_t begin = 0; begin < m.cols;) {
        size_t width = std::min(m.group_size == 32 ? size_t(32) : size_t(64), m.cols - begin);
        int64_t dot = 0;
        for (size_t j = begin; j < begin + width; ++j)
            dot += int64_t(weights[j]) * int64_t(x.words[j]);
        double scale = double(matrix_scale(m, row, begin)) * double(x.scales[begin / 64]);
        sum = std::fma(double(dot), scale, sum);
        begin += width;
    }
    return float(sum);
}
__attribute__((target("avx512f,avx512vnni,avx512bw,avx2,f16c"), noinline))
static float dot_vnni16(const Matrix& m, size_t row, const Activation& x) {
    const auto* weights = static_cast<const int8_t*>(m.data) + row * m.cols;
    unsigned shift = m.group_size ? unsigned(__builtin_ctzll(m.group_size)) : 0;
    __m512 sum = _mm512_setzero_ps();
    for (size_t begin = 0; begin < m.cols;) {
        // A 32-weight artifact has two different scales within an activation
        // group; row/64/128 artifacts use one integer reset per 64 inputs.
        size_t width = std::min(size_t(64), m.cols - begin);
        if (m.group_size == 32) width = std::min(width, size_t(32));
        __m512i dot = _mm512_setzero_si512();
        size_t j = 0;
        for (; j + 32 <= width; j += 32) {
            auto w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + begin + j)));
            auto a = _mm512_loadu_si512(static_cast<const void*>(x.words.data() + begin + j));
            dot = _mm512_dpwssd_epi32(dot, w, a);
        }
        if (j < width) {
            alignas(64) int32_t lanes[16];
            _mm512_store_si512(static_cast<void*>(lanes), dot);
            for (; j < width; ++j) lanes[(j % 32) / 2] += int32_t(weights[begin + j]) * int32_t(x.words[begin + j]);
            dot = _mm512_load_si512(static_cast<const void*>(lanes));
        }
        // At most four signed products per lane: even -128 weights cannot
        // overflow int32. Convert once, FMA in input order, reduce once/row.
        float weight_scale = scale_fast(m, row, begin, shift), activation_scale = x.scales[begin / 64];
        float scale = weight_scale * activation_scale;
        // This conservative constant also leaves accumulation range for any
        // addressable row. Subnormal/large coefficients use a wide whole-row
        // integer reference, avoiding both premature scale loss and inf-inf.
        constexpr float maximum = float(double(std::numeric_limits<float>::max()) /
            (double(std::numeric_limits<size_t>::max()) * 128.0 * 32767.0 * 2.0));
        if (!std::isnormal(scale) || scale > maximum) return dot_vnni16_wide(m, row, x);
        sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), _mm512_set1_ps(scale), sum);
        begin += width;
    }
    return _mm512_reduce_add_ps(sum);
}
template<bool Grouped>
__attribute__((target("avx2,f16c,fma"), noinline))
static void batch_dot256(const Matrix& m, size_t row, const float* x, size_t stride,
                         size_t columns, const Activation*, float* result) {
    __m256 sums[projection_columns];
    for (size_t c = 0; c < columns; ++c) sums[c] = _mm256_setzero_ps();
    size_t j = 0, start = row * m.cols;
    unsigned shift = Grouped ? unsigned(__builtin_ctzll(m.group_size)) : 0;
    for (; j + 8 <= m.cols; j += 8) {
        __m256 w;
        if (m.dtype == DType::bf16) {
            auto bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const uint16_t*>(m.data) + start + j));
            w = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
        } else if (m.dtype == DType::i8) {
            auto bits = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(static_cast<const int8_t*>(m.data) + start + j));
            w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bits));
            if constexpr (Grouped) w = _mm256_mul_ps(w, _mm256_set1_ps(scale_fast(m, row, j, shift)));
        } else w = _mm256_loadu_ps(static_cast<const float*>(m.data) + start + j);
        for (size_t c = 0; c < columns; ++c)
            sums[c] = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + c * stride + j), sums[c]);
    }
    for (size_t c = 0; c < columns; ++c) {
        alignas(32) float lanes[8]; _mm256_store_ps(lanes, sums[c]);
        float total = 0; for (float lane : lanes) total += lane;
        for (size_t t = j; t < m.cols; ++t)
            total += value(m, start + t) * x[c * stride + t] * (m.dtype == DType::i8 && Grouped ? matrix_scale(m, row, t) : 1);
        result[c] = m.dtype == DType::i8 && !Grouped ? total * row_scale_fast(m, row) : total;
    }
}
template<bool Grouped, bool Four>
__attribute__((target("avx512f,avx512bw,f16c"), noinline))
static void batch_dot512(const Matrix& m, size_t row, const float* x, size_t stride,
                         size_t columns, const Activation*, float* result) {
    __m512 a[projection_columns], b[projection_columns], csum[projection_columns], d[projection_columns];
    for (size_t c = 0; c < columns; ++c) {
        a[c] = _mm512_setzero_ps();
        if constexpr (Four) b[c] = csum[c] = d[c] = a[c];
    }
    size_t j = 0, start = row * m.cols;
    unsigned shift = Grouped ? unsigned(__builtin_ctzll(m.group_size)) : 0;
    if constexpr (Four) {
        for (; j + 64 <= m.cols; j += 64) {
            __m512 w0 = load512<false>(m, start + j, row, j, shift);
            __m512 w1 = load512<false>(m, start + j + 16, row, j + 16, shift);
            __m512 w2 = load512<false>(m, start + j + 32, row, j + 32, shift);
            __m512 w3 = load512<false>(m, start + j + 48, row, j + 48, shift);
            if constexpr (Grouped) {
                if (m.dtype == DType::i8) {
                    __m512 first = _mm512_set1_ps(scale_fast(m, row, j, shift));
                    __m512 second = m.group_size == 32 ? _mm512_set1_ps(scale_fast(m, row, j + 32, shift)) : first;
                    w0 = _mm512_mul_ps(w0, first); w1 = _mm512_mul_ps(w1, first);
                    w2 = _mm512_mul_ps(w2, second); w3 = _mm512_mul_ps(w3, second);
                }
            }
            for (size_t c = 0; c < columns; ++c) {
                const float* input = x + c * stride + j;
                a[c] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(input), a[c]);
                b[c] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(input + 16), b[c]);
                csum[c] = _mm512_fmadd_ps(w2, _mm512_loadu_ps(input + 32), csum[c]);
                d[c] = _mm512_fmadd_ps(w3, _mm512_loadu_ps(input + 48), d[c]);
            }
        }
    }
    for (; j + 16 <= m.cols; j += 16) {
        __m512 w = load512<Grouped>(m, start + j, row, j, shift);
        for (size_t c = 0; c < columns; ++c)
            a[c] = _mm512_fmadd_ps(w, _mm512_loadu_ps(x + c * stride + j), a[c]);
    }
    for (size_t c = 0; c < columns; ++c) {
        float total;
        if constexpr (Four) total = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a[c], b[c]), _mm512_add_ps(csum[c], d[c])));
        else total = _mm512_reduce_add_ps(a[c]);
        for (size_t t = j; t < m.cols; ++t)
            total += value(m, start + t) * x[c * stride + t] * (m.dtype == DType::i8 && Grouped ? matrix_scale(m, row, t) : 1);
        result[c] = m.dtype == DType::i8 && !Grouped ? total * row_scale_fast(m, row) : total;
    }
}
__attribute__((target("avx512vnni,avx512vl,avx512bw,avx2,f16c")))
static void batch_dot_vnni(const Matrix& m, size_t row, const float*, size_t,
                           size_t columns, const Activation* x, float* result) {
    std::fill_n(result, columns, 0.0f);
    unsigned shift = m.group_size ? unsigned(__builtin_ctzll(m.group_size)) : 0;
    const auto* weights = static_cast<const int8_t*>(m.data) + row * m.cols;
    const __m256i correction = _mm256_set1_epi8(char(0x80));
    for (size_t j = 0; j < m.cols; j += 32) {
        auto w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + j));
        auto offset = _mm256_dpbusd_epi32(_mm256_setzero_si256(), correction, w);
        float ws = scale_fast(m, row, j, shift);
        for (size_t c = 0; c < columns; ++c) {
            auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[c].bytes.data() + j));
            auto dot = _mm256_sub_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), a, w), offset);
            auto sum = _mm_add_epi32(_mm256_castsi256_si128(dot), _mm256_extracti128_si256(dot, 1));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
            result[c] += float(_mm_cvtsi128_si32(sum)) * (x[c].scales[j / 32] * ws);
        }
    }
}
__attribute__((target("avx512f,avx512vnni,avx512bw,avx2,f16c"), noinline))
static void batch_dot_vnni16(const Matrix& m, size_t row, const float*, size_t,
                             size_t columns, const Activation* x, float* result) {
    const auto* weights = static_cast<const int8_t*>(m.data) + row * m.cols;
    unsigned shift = m.group_size ? unsigned(__builtin_ctzll(m.group_size)) : 0;
    __m512 sums[projection_columns];
    bool wide[projection_columns]{};
    for (size_t c = 0; c < columns; ++c) sums[c] = _mm512_setzero_ps();
    for (size_t begin = 0; begin < m.cols;) {
        size_t width = std::min(m.group_size == 32 ? size_t(32) : size_t(64), m.cols - begin);
        __m512i dots[projection_columns];
        for (size_t c = 0; c < columns; ++c) dots[c] = _mm512_setzero_si512();
        size_t j = 0;
        for (; j + 32 <= width; j += 32) {
            auto w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + begin + j)));
            for (size_t c = 0; c < columns; ++c)
                dots[c] = _mm512_dpwssd_epi32(dots[c], w, _mm512_loadu_si512(static_cast<const void*>(x[c].words.data() + begin + j)));
        }
        if (j < width) for (size_t c = 0; c < columns; ++c) {
            alignas(64) int32_t lanes[16]; _mm512_store_si512(static_cast<void*>(lanes), dots[c]);
            for (size_t t = j; t < width; ++t)
                lanes[(t % 32) / 2] += int32_t(weights[begin + t]) * int32_t(x[c].words[begin + t]);
            dots[c] = _mm512_load_si512(static_cast<const void*>(lanes));
        }
        float ws = scale_fast(m, row, begin, shift);
        constexpr float maximum = float(double(std::numeric_limits<float>::max()) /
            (double(std::numeric_limits<size_t>::max()) * 128.0 * 32767.0 * 2.0));
        for (size_t c = 0; c < columns; ++c) {
            float scale = ws * x[c].scales[begin / 64];
            if (!std::isnormal(scale) || scale > maximum) wide[c] = true;
            if (!wide[c]) sums[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots[c]), _mm512_set1_ps(scale), sums[c]);
        }
        begin += width;
    }
    for (size_t c = 0; c < columns; ++c)
        result[c] = wide[c] ? dot_vnni16_wide(m, row, x[c]) : _mm512_reduce_add_ps(sums[c]);
}
#endif
Kernel parse_kernel(const std::string& name) {
    if (name == "scalar") return Kernel::scalar;
#ifdef DECODE_X86
    __builtin_cpu_init();
    if (name == "auto") {
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("f16c")) return Kernel::simd512x4;
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma")) return Kernel::simd256;
        return Kernel::scalar;
    }
    if (name == "simd256" && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma")) return Kernel::simd256;
    if (name == "simd512" && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("f16c")) return Kernel::simd512;
    if (name == "simd512x4" && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("f16c")) return Kernel::simd512x4;
    if (name == "vnni" && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("f16c")) return Kernel::vnni;
    if (name == "vnni16" && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) return Kernel::vnni16;
#else
    if (name == "auto") return Kernel::scalar;
#endif
    throw std::runtime_error("unknown or unsupported kernel: " + name);
}
std::string kernel_name(Kernel k) {
    if (k == Kernel::scalar) return "scalar";
    if (k == Kernel::simd256) return "simd256";
    if (k == Kernel::simd512) return "simd512";
    if (k == Kernel::simd512x4) return "simd512x4";
    if (k == Kernel::vnni) return "vnni";
    if (k == Kernel::vnni16) return "vnni16";
    throw std::runtime_error("invalid kernel enum");
}
Activation::Activation(size_t capacity, Kernel selected)
    : kernel(selected), bytes(selected == Kernel::vnni ? capacity : 0),
      words(selected == Kernel::vnni16 ? capacity : 0),
      scales(selected == Kernel::vnni ? capacity / 32 + (capacity % 32 != 0) :
             selected == Kernel::vnni16 ? capacity / 64 + (capacity % 64 != 0) : 0) {}
template<class Item>
static size_t validate_projections(const Item* items, size_t count, const float* x, Kernel kernel, bool swiglu) {
    if (!count || !items || !x) throw std::runtime_error("empty projection");
    for (size_t i = 0; i < count; ++i)
        if (!items[i].matrix) throw std::runtime_error("missing projection matrix");
    if (swiglu && (count != 2 || items[0].matrix->rows != items[1].matrix->rows || items[0].bias || items[1].bias)) throw std::runtime_error("SwiGLU requires an equal-row gate/up pair without bias");
    size_t tasks = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto& m = *items[i].matrix;
        if (!m.data || !m.rows || !m.cols || m.dtype == DType::f16 || (!items[i].output && !(swiglu && i == 1)) || (m.dtype == DType::i8 && !m.scales) || m.cols != items[0].matrix->cols ||
            (m.group_size && (m.group_size < 32 || (m.group_size & (m.group_size - 1)) || m.cols % m.group_size))) throw std::runtime_error("invalid projection dimensions");
        if (kernel == Kernel::vnni && (m.dtype != DType::i8 || m.cols % 32 || (m.group_size && m.group_size < 32))) throw std::runtime_error("VNNI requires int8 matrices and groups divisible by32");
        if (kernel == Kernel::vnni16 && (m.dtype != DType::i8 ||
            (m.group_size && m.group_size != 32 && m.group_size != 64 && m.group_size != 128) ||
            (m.scale_dtype != DType::f16 && m.scale_dtype != DType::f32)))
            throw std::runtime_error("VNNI16 requires int8 matrices with row or 32/64/128-group F16/F32 scales");
        tasks += (m.rows + 63) / 64;
    }
    if (swiglu) tasks = (items[0].matrix->rows + 63) / 64;
#ifndef DECODE_X86
    if (kernel == Kernel::vnni || kernel == Kernel::vnni16) throw std::runtime_error("VNNI kernels require x86 SIMD support");
#endif
    return tasks;
}
static void prepare_activation(const float* x, size_t n, Kernel kernel, Activation& activation) {
    if ((kernel == Kernel::vnni || kernel == Kernel::vnni16) && activation.kernel != kernel)
        throw std::runtime_error("activation representation does not match kernel");
    if (kernel == Kernel::vnni) {
        if (activation.bytes.size() < n || activation.scales.size() < n / 32)
            throw std::runtime_error("activation capacity exceeded");
        for (size_t begin = 0; begin < n; begin += 32) {
            float maximum = 0; for (size_t j = begin; j < begin + 32; ++j) maximum = std::max(maximum, std::abs(x[j]));
            float scale = maximum == 0 ? 1 : maximum / 127;
            activation.scales[begin / 32] = scale;
            for (size_t j = begin; j < begin + 32; ++j) activation.bytes[j] = uint8_t(int(std::clamp(std::round(x[j] / scale), -127.0f, 127.0f)) + 128);
        }
    }
    if (kernel == Kernel::vnni16) {
        if (activation.words.size() < n || activation.scales.size() < n / 64 + (n % 64 != 0))
            throw std::runtime_error("activation capacity exceeded");
        for (size_t begin = 0; begin < n; begin += 64) {
            size_t end = std::min(n, begin + 64);
            float maximum = 0;
            for (size_t j = begin; j < end; ++j) {
                if (!std::isfinite(x[j])) throw std::runtime_error("nonfinite VNNI16 activation");
                maximum = std::max(maximum, std::abs(x[j]));
            }
            // A normal-scale floor avoids underflow and reciprocal overflow
            // for tiny groups, including hosts that flush subnormals to zero.
            float scale = maximum == 0 ? 1 : std::max(maximum / 32767.0f, std::numeric_limits<float>::min());
            activation.scales[begin / 64] = scale;
            for (size_t j = begin; j < end; ++j)
                activation.words[j] = static_cast<int16_t>(std::clamp(std::round(x[j] / scale), -32767.0f, 32767.0f));
        }
    }
}
void projections(const Projection* items, size_t count, const float* x, Kernel kernel, ThreadPool& pool, Activation& activation, bool swiglu) {
    size_t tasks = validate_projections(items, count, x, kernel, swiglu);
    prepare_activation(x, items[0].matrix->cols, kernel, activation);
    float (*dot)(const Matrix&, size_t, const float*) = dot_scalar;
#ifdef DECODE_X86
    if (kernel == Kernel::simd256) dot = dot256;
    if (kernel == Kernel::simd512) dot = dot512;
    if (kernel == Kernel::simd512x4) dot = dot512x4;
#endif
    struct Work { const Projection* items; size_t count; const float* x; decltype(dot) function; const Activation* activation; Kernel kernel; bool swiglu; } work{items, count, x, dot, &activation, kernel, swiglu};
    pool.run(tasks, [](void* ptr, size_t task) noexcept {
        auto& w = *static_cast<Work*>(ptr);
        auto dot_row = [&](const Matrix& matrix, size_t row) noexcept {
    #ifdef DECODE_X86
            if (w.kernel == Kernel::vnni) return dot_vnni(matrix, row, *w.activation);
            if (w.kernel == Kernel::vnni16) return dot_vnni16(matrix, row, *w.activation);
    #endif
            return w.function(matrix, row, w.x);
        };
        if (w.swiglu) {
            size_t row = task * 64, end = std::min(w.items[0].matrix->rows, (task + 1) * 64);
            for (; row < end; ++row) {
                float gate = dot_row(*w.items[0].matrix, row), up = dot_row(*w.items[1].matrix, row);
                w.items[0].output[row] = (gate / (1 + std::exp(-gate))) * up;
            }
            return;
        }
        for (size_t i = 0; i < w.count; ++i) {
            const auto& m = *w.items[i].matrix;
            size_t blocks = (m.rows + 63) / 64;
            if (task >= blocks) { task -= blocks; continue; }
            size_t row = task * 64, end = std::min(m.rows, (task + 1) * 64);
            for (; row < end; ++row) {
                float result = dot_row(m, row);
                w.items[i].output[row] = result + (w.items[i].bias ? w.items[i].bias[row] : 0);
            }
            return;
        }
    }, &work);
}
using BatchDot = void (*)(const Matrix&, size_t, const float*, size_t, size_t, const Activation*, float*);
static BatchDot batch_dot(Kernel kernel, const Matrix& matrix) {
    (void)kernel; (void)matrix;
#ifdef DECODE_X86
    bool grouped = matrix.group_size != 0;
    if (kernel == Kernel::simd256) return grouped ? batch_dot256<true> : batch_dot256<false>;
    if (kernel == Kernel::simd512) return grouped ? batch_dot512<true, false> : batch_dot512<false, false>;
    if (kernel == Kernel::simd512x4) return grouped ? batch_dot512<true, true> : batch_dot512<false, true>;
    if (kernel == Kernel::vnni) return batch_dot_vnni;
    if (kernel == Kernel::vnni16) return batch_dot_vnni16;
#endif
    return batch_dot_scalar;
}
void batch_projections(const BatchProjection* items, size_t count, const float* x,
                       size_t columns, size_t input_stride, Kernel kernel,
                       ThreadPool& pool, Activation* activations, bool swiglu) {
    if (!columns || columns > projection_columns || !activations)
        throw std::runtime_error("invalid projection column count");
    size_t tasks = validate_projections(items, count, x, kernel, swiglu);
    if (input_stride < items[0].matrix->cols) throw std::runtime_error("invalid input stride");
    for (size_t i = 0; i < count; ++i)
        if (!(swiglu && i == 1) && items[i].output_stride < items[i].matrix->rows)
            throw std::runtime_error("invalid output stride");
    for (size_t c = 0; c < columns; ++c)
        prepare_activation(x + c * input_stride, items[0].matrix->cols, kernel, activations[c]);
    struct Work {
        const BatchProjection* items; size_t count;
        const float* x; size_t stride, columns;
        const Activation* activations; Kernel kernel; bool swiglu;
    } work{items, count, x, input_stride, columns, activations, kernel, swiglu};
    pool.run(tasks, [](void* ptr, size_t task) noexcept {
        const auto& w = *static_cast<Work*>(ptr);
        if (w.swiglu) {
            size_t end = std::min(w.items[0].matrix->rows, (task + 1) * 64);
            BatchDot gate_dot = batch_dot(w.kernel, *w.items[0].matrix);
            BatchDot up_dot = batch_dot(w.kernel, *w.items[1].matrix);
            for (size_t row = task * 64; row < end; ++row) {
                float gate[projection_columns], up[projection_columns];
                gate_dot(*w.items[0].matrix, row, w.x, w.stride, w.columns, w.activations, gate);
                up_dot(*w.items[1].matrix, row, w.x, w.stride, w.columns, w.activations, up);
                for (size_t c = 0; c < w.columns; ++c)
                    w.items[0].output[c * w.items[0].output_stride + row] = (gate[c] / (1 + std::exp(-gate[c]))) * up[c];
            }
            return;
        }
        for (size_t i = 0; i < w.count; ++i) {
            const auto& item = w.items[i];
            size_t blocks = (item.matrix->rows + 63) / 64;
            if (task >= blocks) { task -= blocks; continue; }
            size_t end = std::min(item.matrix->rows, (task + 1) * 64);
            BatchDot dot = batch_dot(w.kernel, *item.matrix);
            for (size_t row = task * 64; row < end; ++row) {
                float result[projection_columns];
                dot(*item.matrix, row, w.x, w.stride, w.columns, w.activations, result);
                for (size_t c = 0; c < w.columns; ++c)
                    item.output[c * item.output_stride + row] = result[c] + (item.bias ? item.bias[row] : 0);
            }
            return;
        }
    }, &work);
}
void matvec(const Matrix& m, const float* x, float* y, Kernel kernel, ThreadPool& pool) {
    // Standalone kernel tests; the engine owns and reuses its activation buffers.
    Activation activation(m.cols, kernel);
    Projection projection{&m, y}; projections(&projection, 1, x, kernel, pool, activation);
}
void quantize_row(const float* source, size_t n, int8_t* out, float& scale) {
    if (!n) throw std::runtime_error("empty quantization row");
    float maximum = 0;
    for (size_t j = 0; j < n; ++j) {
        if (!std::isfinite(source[j])) throw std::runtime_error("nonfinite model weight");
        maximum = std::max(maximum, std::abs(source[j]));
    }
    scale = maximum == 0 ? 1.0f : maximum / 127.0f;
    for (size_t j = 0; j < n; ++j) out[j] = static_cast<int8_t>(std::clamp(std::round(source[j] / scale), -127.0f, 127.0f));
}
void rmsnorm(const float* x, const float* weight, float* out, size_t n, float epsilon) {
    float sum = 0; for (size_t j = 0; j < n; ++j) sum += x[j] * x[j];
    float inv = 1.0f / std::sqrt(sum / float(n) + epsilon);
    for (size_t j = 0; j < n; ++j) out[j] = x[j] * inv * weight[j];
}
void residual_rmsnorm(float* x, const float* residual, const float* weight, float* out, size_t n, float epsilon) {
    float sum = 0;
    for (size_t j = 0; j < n; ++j) { x[j] += residual[j]; sum += x[j] * x[j]; }
    float inv = 1.0f / std::sqrt(sum / float(n) + epsilon);
    for (size_t j = 0; j < n; ++j) out[j] = x[j] * inv * weight[j];
}
void rope(float* x, size_t heads, size_t dim, size_t position, float theta) {
    for (size_t j = 0; j < dim / 2; ++j) {
        float angle = float(position) / std::pow(theta, float(2 * j) / float(dim));
        float c = std::cos(angle), s = std::sin(angle);
        for (size_t h = 0; h < heads; ++h) {
            float* v = x + h * dim;
            float a = v[j], b = v[j + dim / 2]; v[j] = a * c - b * s; v[j + dim / 2] = b * c + a * s;
        }
    }
}
Profile::Profile() {
    for (const char* name : {"embedding", "rmsnorm", "qkv", "rope", "kv_write", "attention", "attention_output", "residual", "mlp_gate_up", "mlp_down", "lm_head", "argmax", "timing_overhead_and_loop"}) seconds.emplace(name, 0.0);
}
Json Profile::json() const {
    uint64_t minimum = matrix_weight_bytes + scale_bytes + norm_bias_bytes + embedding_bytes + kv_write_bytes + kv_read_min_bytes;
    return {{"operation_seconds", seconds}, {"steps", steps}, {"head_steps", head_steps},
            {"matrix_weight_bytes", matrix_weight_bytes}, {"scale_bytes", scale_bytes},
            {"lm_head_weight_bytes", lm_head_weight_bytes}, {"lm_head_scale_bytes", lm_head_scale_bytes},
            {"norm_bias_bytes", norm_bias_bytes}, {"embedding_bytes", embedding_bytes},
            {"kv_write_bytes", kv_write_bytes}, {"kv_read_min_bytes", kv_read_min_bytes},
            {"kv_read_logical_bytes", kv_read_logical_bytes}, {"minimum_bytes", minimum},
            {"byte_model", "algorithmic weight/scale reads plus embedding row, actual KV writes and minimum GQA KV reads; excludes activation/cache-line/allocator traffic"}};
}
} // namespace decode
