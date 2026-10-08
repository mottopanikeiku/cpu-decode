#include "decode.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <omp.h>

// Flash decoding for one new query token. Work item = (KV head, time chunk).
// Rows are processed in tiles of 16: the K tile is scored against every query
// head of the group while it is hot, then an online softmax (running max m and
// sum l) folds the V tile into each head's output. Partials are merged per head.
namespace decode {
namespace {
constexpr size_t tile = 16;
constexpr float negative_infinity = -std::numeric_limits<float>::infinity();

// Pointer to cached rows [t, t + n) of one KV head as FP32: the cache itself
// for f32, otherwise a converted copy in `buffer`.
template<class Convert>
const float* rows_f32(const void* cache, KvType kv, size_t offset, size_t count, float* buffer, Convert convert) {
    if (kv == KvType::f32) return static_cast<const float*>(cache) + offset;
    convert(static_cast<const uint16_t*>(cache) + offset, count, buffer);
    return buffer;
}

void convert_scalar(const uint16_t* h, size_t n, float* out) { for (size_t j = 0; j < n; ++j) out[j] = half_float(h[j]); }
float dot_scalar(const float* a, const float* b, size_t n) {
    float sum = 0;
    for (size_t j = 0; j < n; ++j) sum += a[j] * b[j];
    return sum;
}
#ifdef DECODE_NEON
void convert_neon(const uint16_t* h, size_t n, float* out) {
    size_t j = 0;
    for (; j + 4 <= n; j += 4) vst1q_f32(out + j, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(h + j))));
    for (; j < n; ++j) out[j] = half_float(h[j]);
}
float dot_neon(const float* a, const float* b, size_t n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + j), vld1q_f32(b + j));
        s1 = vfmaq_f32(s1, vld1q_f32(a + j + 4), vld1q_f32(b + j + 4));
    }
    float sum = vaddvq_f32(vaddq_f32(s0, s1));
    for (; j < n; ++j) sum += a[j] * b[j];
    return sum;
}
// exp(x) for x <= 0, same reduction and polynomial as the AVX-512 version.
inline float32x4_t exp_nonpositive_neon(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.0f / 720);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t scale = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(scale));
}
// Requires dim % 32 == 0. Output chunks of 32 floats (8 independent FMA chains)
// stay in registers across a tile's rows.
void attend_neon(const float* q, const void* keys, const void* values, KvType kv, size_t begin, size_t end,
                 size_t group, size_t dim, float* scratch, float* out, size_t stride) {
    float* key_buffer = scratch;
    float* value_buffer = key_buffer + tile * dim;
    float* probabilities = value_buffer + tile * dim;
    for (size_t t = begin; t < end; t += tile) {
        const size_t n = std::min(tile, end - t);
        const float* k = rows_f32(keys, kv, t * dim, n * dim, key_buffer, convert_neon);
        const float* v = rows_f32(values, kv, t * dim, n * dim, value_buffer, convert_neon);
        for (size_t h = 0; h < group; ++h) {
            float* state = out + h * stride;
            float* o = state + 2;
            const float* qh = q + h * dim;
            float tile_max = negative_infinity;
            for (size_t i = 0; i < n; ++i) {
                const float* kr = k + i * dim;
                float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
                for (size_t c = 0; c < dim; c += 16) {
                    a0 = vfmaq_f32(a0, vld1q_f32(qh + c), vld1q_f32(kr + c));
                    a1 = vfmaq_f32(a1, vld1q_f32(qh + c + 4), vld1q_f32(kr + c + 4));
                    a2 = vfmaq_f32(a2, vld1q_f32(qh + c + 8), vld1q_f32(kr + c + 8));
                    a3 = vfmaq_f32(a3, vld1q_f32(qh + c + 12), vld1q_f32(kr + c + 12));
                }
                probabilities[i] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
                tile_max = std::max(tile_max, probabilities[i]);
            }
            for (size_t i = n; i < tile; ++i) probabilities[i] = negative_infinity;
            const float maximum = std::max(state[0], tile_max);
            const float alpha = std::exp(state[0] - maximum);
            float32x4_t total = vdupq_n_f32(0);
            for (size_t i = 0; i < tile; i += 4) {
                float32x4_t p = exp_nonpositive_neon(vsubq_f32(vld1q_f32(probabilities + i), vdupq_n_f32(maximum)));
                if (i + 4 > n) {  // zero the rows past the end of the chunk
                    const uint32_t lanes[4] = {i < n, i + 1 < n, i + 2 < n, i + 3 < n};
                    p = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(p), vcgtq_u32(vld1q_u32(lanes), vdupq_n_u32(0))));
                }
                vst1q_f32(probabilities + i, p);
                total = vaddq_f32(total, p);
            }
            state[0] = maximum;
            state[1] = state[1] * alpha + vaddvq_f32(total);
            for (size_t c = 0; c < dim; c += 32) {
                float32x4_t acc[8];
                for (size_t j = 0; j < 8; ++j) acc[j] = vmulq_n_f32(vld1q_f32(o + c + 4 * j), alpha);
                for (size_t i = 0; i < n; ++i) {
                    const float* vr = v + i * dim + c;
                    for (size_t j = 0; j < 8; ++j) acc[j] = vfmaq_n_f32(acc[j], vld1q_f32(vr + 4 * j), probabilities[i]);
                }
                for (size_t j = 0; j < 8; ++j) vst1q_f32(o + c + 4 * j, acc[j]);
            }
        }
    }
}
#endif

// Portable tile loop; `convert` and `dot` are the ISA-specific pieces.
template<class Convert, class DotFn>
void attend_generic(const float* q, const void* keys, const void* values, KvType kv, size_t begin, size_t end,
                    size_t group, size_t dim, float* scratch, float* out, size_t stride, Convert convert, DotFn dot) {
    float* key_buffer = scratch;
    float* value_buffer = key_buffer + tile * dim;
    float* scores = value_buffer + tile * dim;
    for (size_t t = begin; t < end; t += tile) {
        const size_t n = std::min(tile, end - t);
        const float* k = rows_f32(keys, kv, t * dim, n * dim, key_buffer, convert);
        const float* v = rows_f32(values, kv, t * dim, n * dim, value_buffer, convert);
        for (size_t h = 0; h < group; ++h) {
            float* state = out + h * stride;  // [max, sum, output...]
            float* o = state + 2;
            float tile_max = negative_infinity;
            for (size_t i = 0; i < n; ++i) {
                scores[i] = dot(q + h * dim, k + i * dim, dim);
                tile_max = std::max(tile_max, scores[i]);
            }
            const float maximum = std::max(state[0], tile_max);
            const float alpha = std::exp(state[0] - maximum);
            float sum = 0;
            for (size_t i = 0; i < n; ++i) { scores[i] = std::exp(scores[i] - maximum); sum += scores[i]; }
            state[0] = maximum;
            state[1] = state[1] * alpha + sum;
            for (size_t j = 0; j < dim; ++j) o[j] *= alpha;
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < dim; ++j) o[j] += scores[i] * v[i * dim + j];
        }
    }
}

#ifdef DECODE_X86
// exp(x) for x <= 0: Cody-Waite reduction to r in [-ln2/2, ln2/2], degree-6
// Taylor polynomial (relative error below 2e-7), then scale by 2^n.
DECODE_AVX512 inline __m512 exp_nonpositive(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.0f / 720);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}
DECODE_AVX512 void convert_avx512(const uint16_t* h, size_t n, float* out) {
    size_t j = 0;
    for (; j + 16 <= n; j += 16)
        _mm512_storeu_ps(out + j, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(h + j))));
    for (; j < n; ++j) out[j] = half_float(h[j]);
}
// Requires dim % 64 == 0. One score vector holds a whole tile (16 rows); the output
// is updated 64 floats at a time so four independent FMA chains run per head.
DECODE_AVX512 void attend_avx512(const float* q, const void* keys, const void* values, KvType kv, size_t begin, size_t end,
                                 size_t group, size_t dim, float* scratch, float* out, size_t stride) {
    float* key_buffer = scratch;
    float* value_buffer = key_buffer + tile * dim;
    float* probabilities = value_buffer + tile * dim;
    for (size_t t = begin; t < end; t += tile) {
        const size_t n = std::min(tile, end - t);
        const __mmask16 valid = __mmask16((1u << n) - 1u);
        const float* k = rows_f32(keys, kv, t * dim, n * dim, key_buffer, convert_avx512);
        const float* v = rows_f32(values, kv, t * dim, n * dim, value_buffer, convert_avx512);
        for (size_t h = 0; h < group; ++h) {
            float* state = out + h * stride;
            float* o = state + 2;
            const float* qh = q + h * dim;
            alignas(64) float raw[tile];
            for (size_t i = 0; i < n; ++i) {
                const float* kr = k + i * dim;
                __m512 acc = _mm512_setzero_ps();
                for (size_t c = 0; c < dim; c += 64) {
                    __m512 a = _mm512_fmadd_ps(_mm512_loadu_ps(qh + c), _mm512_loadu_ps(kr + c), _mm512_mul_ps(_mm512_loadu_ps(qh + c + 16), _mm512_loadu_ps(kr + c + 16)));
                    __m512 b = _mm512_fmadd_ps(_mm512_loadu_ps(qh + c + 32), _mm512_loadu_ps(kr + c + 32), _mm512_mul_ps(_mm512_loadu_ps(qh + c + 48), _mm512_loadu_ps(kr + c + 48)));
                    acc = _mm512_add_ps(acc, _mm512_add_ps(a, b));
                }
                raw[i] = detail::sum512(acc);
            }
            const __m512 scores = _mm512_mask_loadu_ps(_mm512_set1_ps(negative_infinity), valid, raw);
            const float maximum = std::max(state[0], _mm512_reduce_max_ps(scores));
            const float alpha = std::exp(state[0] - maximum);
            const __m512 p = _mm512_maskz_mov_ps(valid, exp_nonpositive(_mm512_sub_ps(scores, _mm512_set1_ps(maximum))));
            _mm512_store_ps(probabilities, p);
            state[0] = maximum;
            state[1] = state[1] * alpha + detail::sum512(p);
            const __m512 scale = _mm512_set1_ps(alpha);
            for (size_t c = 0; c < dim; c += 64) {
                __m512 a0 = _mm512_mul_ps(_mm512_loadu_ps(o + c), scale), a1 = _mm512_mul_ps(_mm512_loadu_ps(o + c + 16), scale);
                __m512 a2 = _mm512_mul_ps(_mm512_loadu_ps(o + c + 32), scale), a3 = _mm512_mul_ps(_mm512_loadu_ps(o + c + 48), scale);
                for (size_t i = 0; i < n; ++i) {
                    const __m512 w = _mm512_set1_ps(probabilities[i]);
                    const float* vr = v + i * dim + c;
                    a0 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vr), a0);
                    a1 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vr + 16), a1);
                    a2 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vr + 32), a2);
                    a3 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vr + 48), a3);
                }
                _mm512_storeu_ps(o + c, a0); _mm512_storeu_ps(o + c + 16, a1);
                _mm512_storeu_ps(o + c + 32, a2); _mm512_storeu_ps(o + c + 48, a3);
            }
        }
    }
}
#endif
} // namespace

AttentionPlan plan_attention(size_t length, size_t heads, size_t kv_heads, size_t dim, int threads) {
    if (!length || !kv_heads || !heads || heads % kv_heads || !dim || threads < 1)
        throw std::runtime_error("invalid attention dimensions");
    AttentionPlan plan;
    plan.length = length;
    plan.group = heads / kv_heads;
    // Enough chunks for every thread, but at least 64 rows per chunk.
    plan.chunks = threads == 1 ? 1 : std::max<size_t>(1, std::min<size_t>(size_t(threads), (length + 63) / 64));
    plan.chunk = (length + plan.chunks - 1) / plan.chunks;
    plan.items = kv_heads * plan.chunks;
    plan.partial_stride = dim + 2;
    return plan;
}
size_t attention_scratch_floats(size_t, size_t dim) { return 2 * tile * dim + tile + 16; }

void attention_item(const AttentionPlan& plan, size_t item, const float* q, const void* keys, const void* values,
                    KvType kv, size_t capacity, size_t dim, float* scratch, float* partials, Kernel kernel) {
    const size_t head = item / plan.chunks, chunk = item % plan.chunks;
    const size_t begin = std::min(plan.length, chunk * plan.chunk), end = std::min(plan.length, begin + plan.chunk);
    const size_t width = kv == KvType::f16 ? 2 : 4, offset = head * capacity * dim * width;
    const void* k = static_cast<const char*>(keys) + offset;
    const void* v = static_cast<const char*>(values) + offset;
    const float* qg = q + head * plan.group * dim;
    float* out = partials + item * plan.group * plan.partial_stride;
    for (size_t h = 0; h < plan.group; ++h) {
        float* state = out + h * plan.partial_stride;
        state[0] = negative_infinity;
        std::fill(state + 1, state + plan.partial_stride, 0.0f);
    }
#ifdef DECODE_X86
    if (kernel == Kernel::avx512 && dim % 64 == 0) {
        attend_avx512(qg, k, v, kv, begin, end, plan.group, dim, scratch, out, plan.partial_stride);
        return;
    }
#endif
#ifdef DECODE_NEON
    if (kernel == Kernel::neon) {
        if (dim % 32 == 0) attend_neon(qg, k, v, kv, begin, end, plan.group, dim, scratch, out, plan.partial_stride);
        else attend_generic(qg, k, v, kv, begin, end, plan.group, dim, scratch, out, plan.partial_stride, convert_neon, dot_neon);
        return;
    }
#endif
    (void)kernel;
    attend_generic(qg, k, v, kv, begin, end, plan.group, dim, scratch, out, plan.partial_stride, convert_scalar, dot_scalar);
}
void attention_merge(const AttentionPlan& plan, size_t head, const float* partials, size_t dim, float* out) {
    const size_t kv = head / plan.group, g = head % plan.group;
    auto state = [&](size_t chunk) { return partials + ((kv * plan.chunks + chunk) * plan.group + g) * plan.partial_stride; };
    float maximum = negative_infinity;
    for (size_t c = 0; c < plan.chunks; ++c) if (state(c)[1] > 0) maximum = std::max(maximum, state(c)[0]);
    float total = 0;
    std::fill(out, out + dim, 0.0f);
    for (size_t c = 0; c < plan.chunks; ++c) {
        const float* s = state(c);
        if (!(s[1] > 0)) continue;
        const float weight = std::exp(s[0] - maximum);
        total += s[1] * weight;
        for (size_t j = 0; j < dim; ++j) out[j] += s[2 + j] * weight;
    }
    const float inverse = 1.0f / total;
    for (size_t j = 0; j < dim; ++j) out[j] *= inverse;
}
void attention(const float* q, const void* keys, const void* values, KvType kv, float* out,
               size_t length, size_t capacity, size_t heads, size_t kv_heads, size_t dim, Kernel kernel, int threads) {
    if (length > capacity) throw std::runtime_error("attention length exceeds cache capacity");
    const AttentionPlan plan = plan_attention(length, heads, kv_heads, dim, threads);
    std::vector<float> partials(plan.items * plan.group * plan.partial_stride);
    const size_t scratch_floats = (attention_scratch_floats(plan.group, dim) + 15) / 16 * 16;
    AlignedVector<float> scratch(size_t(threads) * scratch_floats);
    #pragma omp parallel num_threads(threads) if(threads > 1)
    {
        const size_t t = size_t(omp_get_thread_num()), n = size_t(omp_get_num_threads());
        for (size_t item = t; item < plan.items; item += n)
            attention_item(plan, item, q, keys, values, kv, capacity, dim, scratch.data() + t * scratch_floats, partials.data(), kernel);
        #pragma omp barrier
        for (size_t h = t; h < heads; h += n) attention_merge(plan, h, partials.data(), dim, out + h * dim);
    }
}
} // namespace decode
