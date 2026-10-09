#include "decode.hpp"
#include "simd.hpp"
#include <cstring>
#include <stdexcept>
#include <omp.h>

// q8/q4 rows are dotted with an int8 activation quantized in 32-element blocks.
// Each block contributes weight_scale * activation_scale * integer_dot.
namespace decode {
namespace {
using Dot = float (*)(const Matrix&, size_t row, const float* x, const Activation& a);

float bf16_scalar(const Matrix& m, size_t row, const float* x, const Activation&) {
    const uint16_t* w = static_cast<const uint16_t*>(m.data) + row * m.cols;
    float sum = 0;
    for (size_t j = 0; j < m.cols; ++j) sum += bf16_float(w[j]) * x[j];
    return sum;
}
float q8_scalar(const Matrix& m, size_t row, const float*, const Activation& a) {
    const size_t blocks = m.cols / block_size;
    const int8_t* w = static_cast<const int8_t*>(m.data) + row * m.cols;
    const uint16_t* s = m.scales + row * blocks;
    float total = 0;
    for (size_t b = 0; b < blocks; ++b) {
        int32_t sum = 0;
        for (size_t j = 0; j < block_size; ++j) sum += int32_t(w[b * block_size + j]) * a.q[b * block_size + j];
        total += float(sum) * (half_float(s[b]) * a.d[b]);
    }
    return total;
}
float q4_scalar(const Matrix& m, size_t row, const float*, const Activation& a) {
    const size_t blocks = m.cols / block_size;
    const uint8_t* w = static_cast<const uint8_t*>(m.data) + row * m.cols / 2;
    const uint16_t* s = m.scales + row * blocks;
    float total = 0;
    for (size_t g = 0; g < blocks / 2; ++g) {
        int32_t low = 0, high = 0;
        const int8_t* q = a.q.data() + 2 * g * block_size;
        for (size_t j = 0; j < block_size; ++j) {
            uint8_t byte = w[g * block_size + j];
            low += (int32_t(byte & 15) - 8) * q[j];
            high += (int32_t(byte >> 4) - 8) * q[block_size + j];
        }
        total += float(low) * (half_float(s[2 * g]) * a.d[2 * g]);
        total += float(high) * (half_float(s[2 * g + 1]) * a.d[2 * g + 1]);
    }
    return total;
}

#ifdef DECODE_X86
DECODE_AVX512 float bf16_avx512(const Matrix& m, size_t row, const float* x, const Activation&) {
    const uint16_t* w = static_cast<const uint16_t*>(m.data) + row * m.cols;
    auto load = [&](size_t j) DECODE_AVX512 {
        __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + j));
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bits), 16));
    };
    __m512 a = _mm512_setzero_ps(), b = a, c = a, d = a;
    size_t j = 0;
    for (; j + 64 <= m.cols; j += 64) {
        a = _mm512_fmadd_ps(load(j), _mm512_loadu_ps(x + j), a);
        b = _mm512_fmadd_ps(load(j + 16), _mm512_loadu_ps(x + j + 16), b);
        c = _mm512_fmadd_ps(load(j + 32), _mm512_loadu_ps(x + j + 32), c);
        d = _mm512_fmadd_ps(load(j + 48), _mm512_loadu_ps(x + j + 48), d);
    }
    for (; j + 16 <= m.cols; j += 16) a = _mm512_fmadd_ps(load(j), _mm512_loadu_ps(x + j), a);
    float total = detail::sum512(_mm512_add_ps(_mm512_add_ps(a, b), _mm512_add_ps(c, d)));
    for (; j < m.cols; ++j) total += bf16_float(w[j]) * x[j];
    return total;
}
// One 64-weight pair (two blocks) as unsigned bytes for VPDPBUSD. q8 flips the
// sign bit (w + 128); q4 levels are already 0..15 (w + 8). The activation bias
// lanes hold -128 * block_sum, which removes the offset inside the dot product.
template<bool Q4> DECODE_AVX512 inline __m512i unsigned_pair(const uint8_t* w, size_t pair) {
    if constexpr (Q4) {
        __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + pair * 32));
        const __m256i nibble = _mm256_set1_epi8(15);
        __m256i low = _mm256_and_si256(packed, nibble);
        __m256i high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble);
        return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
    } else {
        return _mm512_xor_si512(_mm512_loadu_si512(w + pair * 64), _mm512_set1_epi8(char(0x80)));
    }
}
template<bool Q4> DECODE_AVX512 inline __m512 pair_product(const uint8_t* w, size_t pair, const Activation& a) {
    __m512i bias = _mm512_load_si512(a.bias.data() + pair * 16);
    if constexpr (Q4) bias = _mm512_srai_epi32(bias, 4);  // -8 * block_sum
    __m512i x = _mm512_load_si512(a.q.data() + pair * 64);
    return _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(bias, unsigned_pair<Q4>(w, pair), x));
}
template<bool Q4> DECODE_AVX512 float blocks_avx512(const Matrix& m, size_t row, const float*, const Activation& a) {
    const size_t blocks = m.cols / block_size, pairs = blocks / 2;
    const uint8_t* w = static_cast<const uint8_t*>(m.data) + row * (Q4 ? m.cols / 2 : m.cols);
    const uint16_t* s = m.scales + row * blocks;
    const __m512i spread[4] = {
        _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        _mm512_set_epi32(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2),
        _mm512_set_epi32(5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4),
        _mm512_set_epi32(7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6)};
    __m512 acc0 = _mm512_setzero_ps(), acc1 = acc0;
    size_t p = 0;
    for (; p + 4 <= pairs; p += 4) {
        // Eight block scales (weight * activation), spread to 8-lane halves per pair.
        __m256 scale8 = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 2 * p))),
                                      _mm256_loadu_ps(a.d.data() + 2 * p));
        __m512 scale = _mm512_castps256_ps512(scale8);
        acc0 = _mm512_fmadd_ps(pair_product<Q4>(w, p, a), _mm512_permutexvar_ps(spread[0], scale), acc0);
        acc1 = _mm512_fmadd_ps(pair_product<Q4>(w, p + 1, a), _mm512_permutexvar_ps(spread[1], scale), acc1);
        acc0 = _mm512_fmadd_ps(pair_product<Q4>(w, p + 2, a), _mm512_permutexvar_ps(spread[2], scale), acc0);
        acc1 = _mm512_fmadd_ps(pair_product<Q4>(w, p + 3, a), _mm512_permutexvar_ps(spread[3], scale), acc1);
    }
    for (; p < pairs; ++p) {
        uint32_t two;
        double activation;
        std::memcpy(&two, s + 2 * p, 4);
        std::memcpy(&activation, a.d.data() + 2 * p, 8);
        __m128 scale2 = _mm_mul_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(int(two))), _mm_castpd_ps(_mm_set_sd(activation)));
        acc0 = _mm512_fmadd_ps(pair_product<Q4>(w, p, a), _mm512_permutexvar_ps(spread[0], _mm512_castps128_ps512(scale2)), acc0);
    }
    return detail::sum512(_mm512_add_ps(acc0, acc1));
}
#endif

#ifdef DECODE_NEON
float bf16_neon(const Matrix& m, size_t row, const float* x, const Activation&) {
    const uint16_t* w = static_cast<const uint16_t*>(m.data) + row * m.cols;
    auto load = [&](size_t j) { return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + j), 16)); };
    float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
    size_t j = 0;
    for (; j + 16 <= m.cols; j += 16) {
        a = vfmaq_f32(a, load(j), vld1q_f32(x + j));
        b = vfmaq_f32(b, load(j + 4), vld1q_f32(x + j + 4));
        c = vfmaq_f32(c, load(j + 8), vld1q_f32(x + j + 8));
        d = vfmaq_f32(d, load(j + 12), vld1q_f32(x + j + 12));
    }
    float total = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    for (; j < m.cols; ++j) total += bf16_float(w[j]) * x[j];
    return total;
}
// Four consecutive block scales (weight * activation).
inline float32x4_t scales4(const uint16_t* s, const float* d) {
    return vmulq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(s))), vld1q_f32(d));
}
DECODE_NEON_DOT float q8_neon(const Matrix& m, size_t row, const float*, const Activation& a) {
    const size_t blocks = m.cols / block_size;
    const int8_t* w = static_cast<const int8_t*>(m.data) + row * m.cols;
    const int8_t* q = a.q.data();
    const uint16_t* s = m.scales + row * blocks;
    auto block = [&](size_t b) DECODE_NEON_DOT {
        int32x4_t sum = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w + b * 32), vld1q_s8(q + b * 32));
        return vcvtq_f32_s32(vdotq_s32(sum, vld1q_s8(w + b * 32 + 16), vld1q_s8(q + b * 32 + 16)));
    };
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = acc0;
    size_t b = 0;
    for (; b + 4 <= blocks; b += 4) {
        float32x4_t scale = scales4(s + b, a.d.data() + b);
        acc0 = vfmaq_laneq_f32(acc0, block(b), scale, 0);
        acc1 = vfmaq_laneq_f32(acc1, block(b + 1), scale, 1);
        acc0 = vfmaq_laneq_f32(acc0, block(b + 2), scale, 2);
        acc1 = vfmaq_laneq_f32(acc1, block(b + 3), scale, 3);
    }
    for (; b < blocks; ++b) acc0 = vfmaq_n_f32(acc0, block(b), half_float(s[b]) * a.d[b]);
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}
DECODE_NEON_DOT float q4_neon(const Matrix& m, size_t row, const float*, const Activation& a) {
    const size_t blocks = m.cols / block_size;
    const uint8_t* w = static_cast<const uint8_t*>(m.data) + row * m.cols / 2;
    const int8_t* q = a.q.data();
    const uint16_t* s = m.scales + row * blocks;
    const uint8x16_t nibble = vdupq_n_u8(15);
    const int8x16_t eight = vdupq_n_s8(8);
    // Group g (64 weights): low nibbles are block 2g, high nibbles block 2g + 1.
    auto group = [&](size_t g, float32x4_t& low, float32x4_t& high) DECODE_NEON_DOT {
        uint8x16_t v0 = vld1q_u8(w + g * 32), v1 = vld1q_u8(w + g * 32 + 16);
        const int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, nibble)), eight);
        const int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, nibble)), eight);
        const int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), eight);
        const int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), eight);
        const int8_t* x = q + g * 64;
        low = vcvtq_f32_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), l0, vld1q_s8(x)), l1, vld1q_s8(x + 16)));
        high = vcvtq_f32_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), h0, vld1q_s8(x + 32)), h1, vld1q_s8(x + 48)));
    };
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = acc0;
    size_t g = 0;
    for (; 2 * g + 4 <= blocks; g += 2) {
        float32x4_t scale = scales4(s + 2 * g, a.d.data() + 2 * g), l, h;
        group(g, l, h);
        acc0 = vfmaq_laneq_f32(acc0, l, scale, 0);
        acc1 = vfmaq_laneq_f32(acc1, h, scale, 1);
        group(g + 1, l, h);
        acc0 = vfmaq_laneq_f32(acc0, l, scale, 2);
        acc1 = vfmaq_laneq_f32(acc1, h, scale, 3);
    }
    for (; 2 * g < blocks; ++g) {
        float32x4_t l, h;
        group(g, l, h);
        acc0 = vfmaq_n_f32(acc0, l, half_float(s[2 * g]) * a.d[2 * g]);
        acc1 = vfmaq_n_f32(acc1, h, half_float(s[2 * g + 1]) * a.d[2 * g + 1]);
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}
// Four rows at once: every activation load and block scale conversion is shared
// by four weight rows, and four independent accumulator chains stay in flight.
template<int J, size_t R> DECODE_NEON_DOT inline void q8_block(float32x4_t (&acc)[R], const int8_t* const (&w)[R], const int8_t* q,
                                                               size_t b, const float32x4_t (&scale)[R]) {
    const int8x16_t x0 = vld1q_s8(q + b * 32), x1 = vld1q_s8(q + b * 32 + 16);
    for (size_t r = 0; r < R; ++r) {
        const int32x4_t sum = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w[r] + b * 32), x0), vld1q_s8(w[r] + b * 32 + 16), x1);
        acc[r] = vfmaq_laneq_f32(acc[r], vcvtq_f32_s32(sum), scale[r], J);
    }
}
template<size_t R> DECODE_NEON_DOT void q8_neon_rows(const Matrix& m, size_t row, const Activation& a, float* out) {
    const size_t blocks = m.cols / block_size;
    const int8_t* w[R];
    const uint16_t* s[R];
    float32x4_t acc[R], scale[R];
    for (size_t r = 0; r < R; ++r) {
        w[r] = static_cast<const int8_t*>(m.data) + (row + r) * m.cols;
        s[r] = m.scales + (row + r) * blocks;
        acc[r] = vdupq_n_f32(0);
    }
    const int8_t* q = a.q.data();
    size_t b = 0;
    for (; b + 4 <= blocks; b += 4) {
        // Four interleaved row streams defeat the hardware prefetcher; fetch the same
        // 128 bytes two row groups ahead (rows are contiguous) by hand.
        for (size_t r = 0; r < R; ++r) {
            __builtin_prefetch(w[r] + 2 * R * m.cols + b * 32);
            __builtin_prefetch(w[r] + 2 * R * m.cols + b * 32 + 64);
        }
        const float32x4_t d = vld1q_f32(a.d.data() + b);
        for (size_t r = 0; r < R; ++r) scale[r] = vmulq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(s[r] + b))), d);
        q8_block<0>(acc, w, q, b, scale);
        q8_block<1>(acc, w, q, b + 1, scale);
        q8_block<2>(acc, w, q, b + 2, scale);
        q8_block<3>(acc, w, q, b + 3, scale);
    }
    for (; b < blocks; ++b)
        for (size_t r = 0; r < R; ++r) {
            const int32x4_t sum = vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(w[r] + b * 32), vld1q_s8(q + b * 32)),
                                            vld1q_s8(w[r] + b * 32 + 16), vld1q_s8(q + b * 32 + 16));
            acc[r] = vfmaq_n_f32(acc[r], vcvtq_f32_s32(sum), half_float(s[r][b]) * a.d[b]);
        }
    for (size_t r = 0; r < R; ++r) out[r] = vaddvq_f32(acc[r]);
}
// One 64-weight group (blocks 2g, 2g + 1) for R rows. Levels stay unsigned 0..15;
// the -8 offset enters through the activation bias (-8 * block sum) as the dot's start.
template<int J, size_t R> DECODE_NEON_DOT inline void q4_group(float32x4_t (&acc)[R], const uint8_t* const (&w)[R], const int8_t* q,
                                                               const int32_t* bias, size_t g, const float32x4_t (&scale)[R]) {
    const int8_t* x = q + g * 64;
    const int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16), x2 = vld1q_s8(x + 32), x3 = vld1q_s8(x + 48);
    const int32x4_t low_start = vshrq_n_s32(vld1q_s32(bias + g * 16), 4), high_start = vshrq_n_s32(vld1q_s32(bias + g * 16 + 8), 4);
    const uint8x16_t nibble = vdupq_n_u8(15);
    for (size_t r = 0; r < R; ++r) {
        const uint8x16_t v0 = vld1q_u8(w[r] + g * 32), v1 = vld1q_u8(w[r] + g * 32 + 16);
        const int32x4_t low = vdotq_s32(vdotq_s32(low_start, vreinterpretq_s8_u8(vandq_u8(v0, nibble)), x0),
                                        vreinterpretq_s8_u8(vandq_u8(v1, nibble)), x1);
        const int32x4_t high = vdotq_s32(vdotq_s32(high_start, vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), x2),
                                         vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), x3);
        acc[r] = vfmaq_laneq_f32(acc[r], vcvtq_f32_s32(low), scale[r], 2 * J);
        acc[r] = vfmaq_laneq_f32(acc[r], vcvtq_f32_s32(high), scale[r], 2 * J + 1);
    }
}
template<size_t R> DECODE_NEON_DOT void q4_neon_rows(const Matrix& m, size_t row, const Activation& a, float* out) {
    const size_t blocks = m.cols / block_size;
    const uint8_t* w[R];
    const uint16_t* s[R];
    float32x4_t acc[R], scale[R];
    for (size_t r = 0; r < R; ++r) {
        w[r] = static_cast<const uint8_t*>(m.data) + (row + r) * m.cols / 2;
        s[r] = m.scales + (row + r) * blocks;
        acc[r] = vdupq_n_f32(0);
    }
    size_t g = 0;
    for (; 2 * g + 4 <= blocks; g += 2) {
        for (size_t r = 0; r < R; ++r) __builtin_prefetch(w[r] + R * m.cols + g * 32);  // two row groups ahead
        const float32x4_t d = vld1q_f32(a.d.data() + 2 * g);
        for (size_t r = 0; r < R; ++r) scale[r] = vmulq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(s[r] + 2 * g))), d);
        q4_group<0>(acc, w, a.q.data(), a.bias.data(), g, scale);
        q4_group<1>(acc, w, a.q.data(), a.bias.data(), g + 1, scale);
    }
    for (; 2 * g < blocks; ++g) {  // one trailing group: its scales in lanes 0 and 1
        for (size_t r = 0; r < R; ++r) {
            const float lanes[4] = {half_float(s[r][2 * g]) * a.d[2 * g], half_float(s[r][2 * g + 1]) * a.d[2 * g + 1], 0.0f, 0.0f};
            scale[r] = vld1q_f32(lanes);
        }
        q4_group<0>(acc, w, a.q.data(), a.bias.data(), g, scale);
    }
    for (size_t r = 0; r < R; ++r) out[r] = vaddvq_f32(acc[r]);
}
#endif

Dot select(Format format, Kernel kernel) {
#ifdef DECODE_X86
    if (kernel == Kernel::avx512)
        return format == Format::bf16 ? bf16_avx512 : format == Format::q8 ? blocks_avx512<false> : blocks_avx512<true>;
#endif
#ifdef DECODE_NEON
    if (kernel == Kernel::neon) return format == Format::bf16 ? bf16_neon : format == Format::q8 ? q8_neon : q4_neon;
#endif
    if (kernel != Kernel::scalar) throw std::runtime_error("kernel not compiled for this architecture");
    return format == Format::bf16 ? bf16_scalar : format == Format::q8 ? q8_scalar : q4_scalar;
}
using Rows4 = void (*)(const Matrix&, size_t row, const Activation&, float* out);
Rows4 select_rows4(const Matrix& m, Kernel kernel) {
#ifdef DECODE_NEON
    // q8 rows longer than 2048 weights stream well one at a time (the prefetch two
    // groups ahead would reach past L1); q4 is compute-bound and always gains.
    if (kernel == Kernel::neon && m.format == Format::q8 && m.cols <= 2048) return q8_neon_rows<4>;
    if (kernel == Kernel::neon && m.format == Format::q4) return q4_neon_rows<4>;
#endif
    (void)m; (void)kernel;
    return nullptr;
}
} // namespace

void matvec_rows(const Matrix& m, const float* x, const Activation& a, float* y,
                 size_t begin, size_t end, Kernel kernel, bool accumulate) {
    const Dot dot = select(m.format, kernel);
    size_t r = begin;
    if (const Rows4 rows4 = select_rows4(m, kernel))
        for (; r + 4 <= end; r += 4) {
            float out[4];
            rows4(m, r, a, out);
            for (size_t i = 0; i < 4; ++i) y[r + i] = accumulate ? y[r + i] + out[i] : out[i];
        }
    if (accumulate) for (; r < end; ++r) y[r] += dot(m, r, x, a);
    else for (; r < end; ++r) y[r] = dot(m, r, x, a);
}
void matvec(const Matrix& m, const float* x, float* y, Kernel kernel, int threads) {
    if (!m.data || !m.rows || !m.cols || threads < 1 || (m.format != Format::bf16 && (!m.scales || m.cols % (2 * block_size))))
        throw std::runtime_error("invalid matvec arguments");
    select(m.format, kernel);
    Activation a;
    if (m.format != Format::bf16) {
        a.resize(m.cols);
        quantize_activation(x, a, 0, m.cols / block_size);
    }
    #pragma omp parallel num_threads(threads) if(threads > 1)
    {
        size_t t = size_t(omp_get_thread_num()), n = size_t(omp_get_num_threads());
        matvec_rows(m, x, a, y, m.rows * t / n, m.rows * (t + 1) / n, kernel, false);
    }
}
} // namespace decode
