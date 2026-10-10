#include "decode.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <omp.h>

// Flash decoding for one new query token. Work item = (KV head, time chunk).
// Keys are cached in tiles of 16 positions stored dimension-major, so one
// vector holds one dimension across 16 positions: scoring a tile against all
// query heads of a KV group is broadcast-FMA work with no horizontal sums.
// An online softmax (running max m and sum l) then folds the row-major V tile
// into every head's output at once, so each V row is loaded once per group.
// Partial (m, l, output) states of the time chunks are merged per head.
namespace decode {
namespace {
constexpr size_t tile = kv_tile;
constexpr float negative_infinity = -std::numeric_limits<float>::infinity();

inline float load_kv(const void* base, KvType kv, size_t index) {
    return kv == KvType::f32 ? static_cast<const float*>(base)[index] : half_float(static_cast<const uint16_t*>(base)[index]);
}

// Online-softmax update of one head's (max, sum) for a tile whose first n scores
// are valid. Scores become probabilities (zero past n); returns the output rescale.
float softmax_tile(float* state, float* scores, size_t n) {
    float maximum = state[0];
    for (size_t i = 0; i < n; ++i) maximum = std::max(maximum, scores[i]);
    const float alpha = std::exp(state[0] - maximum);
    float sum = 0;
    for (size_t i = 0; i < tile; ++i) {
        scores[i] = i < n ? std::exp(scores[i] - maximum) : 0.0f;
        sum += scores[i];
    }
    state[0] = maximum;
    state[1] = state[1] * alpha + sum;
    return alpha;
}

void attend_scalar(const float* q, const void* keys, const void* values, KvType kv, size_t begin, size_t end,
                   size_t group, size_t dim, float* scores, float* out, size_t stride) {
    for (size_t t = begin; t < end; t += tile) {
        const size_t n = std::min(tile, end - t);
        for (size_t h = 0; h < group; ++h) {
            float* s = scores + h * tile;
            for (size_t i = 0; i < n; ++i) {
                float sum = 0;
                for (size_t d = 0; d < dim; ++d) sum += q[h * dim + d] * load_kv(keys, kv, t * dim + d * tile + i);
                s[i] = sum;
            }
            float* state = out + h * stride;
            float* o = state + 2;
            const float alpha = softmax_tile(state, s, n);
            for (size_t j = 0; j < dim; ++j) o[j] *= alpha;
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < dim; ++j) o[j] += s[i] * load_kv(values, kv, (t + i) * dim + j);
        }
    }
}

#ifdef DECODE_NEON
// Eight consecutive cache elements as FP32.
template<bool Half> inline void load8(const void* base, size_t index, float32x4_t& a, float32x4_t& b) {
    if constexpr (Half) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(static_cast<const uint16_t*>(base) + index));
        a = vcvt_f32_f16(vget_low_f16(h));
        b = vcvt_high_f32_f16(h);
    } else {
        const float* p = static_cast<const float*>(base) + index;
        a = vld1q_f32(p);
        b = vld1q_f32(p + 4);
    }
}
// acc[h] += (x0, x1) * lane J of w[h] for every head of the group.
template<int J, size_t G> inline void fma_lane(float32x4_t (&acc)[G][2], const float32x4_t (&w)[G], float32x4_t x0, float32x4_t x1) {
    for (size_t h = 0; h < G; ++h) {
        acc[h][0] = vfmaq_laneq_f32(acc[h][0], x0, w[h], J);
        acc[h][1] = vfmaq_laneq_f32(acc[h][1], x1, w[h], J);
    }
}
// Requires dim % 8 == 0. D > 0 fixes the head size at compile time: with the loops fully
// unrolled the accumulators stay in place instead of rotating through extra moves.
template<size_t G, bool Half, size_t D = 0>
void attend_neon(const float* q, const void* keys, const void* values, size_t begin, size_t end, size_t runtime_dim,
                 float* scores, float* out, size_t stride) {
    const size_t dim = D ? D : runtime_dim;
    float alpha[G];
    for (size_t t = begin; t < end; t += tile) {
        const size_t n = std::min(tile, end - t);
        // Scores: positions [8 * half, 8 * half + 8) of the tile for all G heads.
        for (size_t half = 0; half < 2; ++half) {
            float32x4_t acc[G][2];
            for (size_t h = 0; h < G; ++h) acc[h][0] = acc[h][1] = vdupq_n_f32(0);
            const size_t base = t * dim + half * 8;
            #pragma GCC unroll 16
            for (size_t d = 0; d < dim; d += 4) {
                float32x4_t qv[G], k0, k1;
                for (size_t h = 0; h < G; ++h) qv[h] = vld1q_f32(q + h * dim + d);
                load8<Half>(keys, base + d * tile, k0, k1); fma_lane<0>(acc, qv, k0, k1);
                load8<Half>(keys, base + (d + 1) * tile, k0, k1); fma_lane<1>(acc, qv, k0, k1);
                load8<Half>(keys, base + (d + 2) * tile, k0, k1); fma_lane<2>(acc, qv, k0, k1);
                load8<Half>(keys, base + (d + 3) * tile, k0, k1); fma_lane<3>(acc, qv, k0, k1);
            }
            for (size_t h = 0; h < G; ++h) {
                vst1q_f32(scores + h * tile + half * 8, acc[h][0]);
                vst1q_f32(scores + h * tile + half * 8 + 4, acc[h][1]);
            }
        }
        for (size_t h = 0; h < G; ++h) {
            float* s = scores + h * tile;
            float* state = out + h * stride;
            for (size_t i = n; i < tile; ++i) s[i] = negative_infinity;
            const float32x4_t s0 = vld1q_f32(s), s1 = vld1q_f32(s + 4), s2 = vld1q_f32(s + 8), s3 = vld1q_f32(s + 12);
            const float maximum = std::max(state[0], vmaxvq_f32(vmaxq_f32(vmaxq_f32(s0, s1), vmaxq_f32(s2, s3))));
            alpha[h] = std::exp(state[0] - maximum);
            const float32x4_t m = vdupq_n_f32(maximum);
            vst1q_f32(s, detail::exp_neon(vsubq_f32(s0, m)));
            vst1q_f32(s + 4, detail::exp_neon(vsubq_f32(s1, m)));
            vst1q_f32(s + 8, detail::exp_neon(vsubq_f32(s2, m)));
            vst1q_f32(s + 12, detail::exp_neon(vsubq_f32(s3, m)));
            for (size_t i = n; i < tile; ++i) s[i] = 0.0f;
            const float sum = vaddvq_f32(vaddq_f32(vaddq_f32(vld1q_f32(s), vld1q_f32(s + 4)), vaddq_f32(vld1q_f32(s + 8), vld1q_f32(s + 12))));
            state[0] = maximum;
            state[1] = state[1] * alpha[h] + sum;
        }
        // Rows past n have zero probability; their (finite) cache contents do not matter.
        for (size_t c = 0; c < dim; c += 8) {
            float32x4_t acc[G][2];
            for (size_t h = 0; h < G; ++h) {
                const float* o = out + h * stride + 2 + c;
                acc[h][0] = vmulq_n_f32(vld1q_f32(o), alpha[h]);
                acc[h][1] = vmulq_n_f32(vld1q_f32(o + 4), alpha[h]);
            }
            #pragma GCC unroll 4
            for (size_t i = 0; i < tile; i += 4) {
                float32x4_t p[G], v0, v1;
                for (size_t h = 0; h < G; ++h) p[h] = vld1q_f32(scores + h * tile + i);
                load8<Half>(values, (t + i) * dim + c, v0, v1); fma_lane<0>(acc, p, v0, v1);
                load8<Half>(values, (t + i + 1) * dim + c, v0, v1); fma_lane<1>(acc, p, v0, v1);
                load8<Half>(values, (t + i + 2) * dim + c, v0, v1); fma_lane<2>(acc, p, v0, v1);
                load8<Half>(values, (t + i + 3) * dim + c, v0, v1); fma_lane<3>(acc, p, v0, v1);
            }
            for (size_t h = 0; h < G; ++h) {
                float* o = out + h * stride + 2 + c;
                vst1q_f32(o, acc[h][0]);
                vst1q_f32(o + 4, acc[h][1]);
            }
        }
    }
}
// Head size of every Qwen2.5 model up to 7B.
template<size_t G, bool Half>
void attend_neon64(const float* q, const void* keys, const void* values, size_t begin, size_t end, size_t dim,
                   float* scores, float* out, size_t stride) {
    attend_neon<G, Half, 64>(q, keys, values, begin, end, dim, scores, out, stride);
}
#endif

#ifdef DECODE_X86
// Same reduction and polynomial as the NEON version.
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
// Sixteen consecutive cache elements as FP32.
template<bool Half> DECODE_AVX512 inline __m512 load16(const void* base, size_t index) {
    if constexpr (Half) return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const uint16_t*>(base) + index)));
    else return _mm512_loadu_ps(static_cast<const float*>(base) + index);
}
// Requires dim % 32 == 0. One vector holds a whole tile's scores for one head.
template<size_t G, bool Half>
DECODE_AVX512 void attend_avx512(const float* q, const void* keys, const void* values, size_t begin, size_t end, size_t dim,
                                 float* scores, float* out, size_t stride) {
    float alpha[G];
    for (size_t t = begin; t < end; t += tile) {
        const size_t n = std::min(tile, end - t);
        const __mmask16 valid = __mmask16((1u << n) - 1u);
        __m512 acc[G];
        for (size_t h = 0; h < G; ++h) acc[h] = _mm512_setzero_ps();
        for (size_t d = 0; d < dim; ++d) {
            const __m512 k = load16<Half>(keys, t * dim + d * tile);
            for (size_t h = 0; h < G; ++h) acc[h] = _mm512_fmadd_ps(_mm512_set1_ps(q[h * dim + d]), k, acc[h]);
        }
        for (size_t h = 0; h < G; ++h) {
            float* state = out + h * stride;
            const __m512 s = _mm512_mask_mov_ps(_mm512_set1_ps(negative_infinity), valid, acc[h]);
            const float maximum = std::max(state[0], _mm512_reduce_max_ps(s));
            alpha[h] = std::exp(state[0] - maximum);
            const __m512 p = _mm512_maskz_mov_ps(valid, exp_nonpositive(_mm512_sub_ps(s, _mm512_set1_ps(maximum))));
            _mm512_storeu_ps(scores + h * tile, p);
            state[0] = maximum;
            state[1] = state[1] * alpha[h] + detail::sum512(p);
        }
        for (size_t c = 0; c < dim; c += 32) {
            __m512 o[G][2];
            for (size_t h = 0; h < G; ++h) {
                const __m512 a = _mm512_set1_ps(alpha[h]);
                o[h][0] = _mm512_mul_ps(_mm512_loadu_ps(out + h * stride + 2 + c), a);
                o[h][1] = _mm512_mul_ps(_mm512_loadu_ps(out + h * stride + 2 + c + 16), a);
            }
            for (size_t i = 0; i < tile; ++i) {
                const __m512 v0 = load16<Half>(values, (t + i) * dim + c), v1 = load16<Half>(values, (t + i) * dim + c + 16);
                for (size_t h = 0; h < G; ++h) {
                    const __m512 w = _mm512_set1_ps(scores[h * tile + i]);
                    o[h][0] = _mm512_fmadd_ps(w, v0, o[h][0]);
                    o[h][1] = _mm512_fmadd_ps(w, v1, o[h][1]);
                }
            }
            for (size_t h = 0; h < G; ++h) {
                _mm512_storeu_ps(out + h * stride + 2 + c, o[h][0]);
                _mm512_storeu_ps(out + h * stride + 2 + c + 16, o[h][1]);
            }
        }
    }
}
#endif

// Instantiates a SIMD kernel for groups of 1-8 query heads per KV head; false if none applies.
#define DECODE_GROUP_CASES(KERNEL, ...)                                                        \
    switch (group) {                                                                         \
    case 1: half ? KERNEL<1, true>(__VA_ARGS__) : KERNEL<1, false>(__VA_ARGS__); return true; \
    case 2: half ? KERNEL<2, true>(__VA_ARGS__) : KERNEL<2, false>(__VA_ARGS__); return true; \
    case 3: half ? KERNEL<3, true>(__VA_ARGS__) : KERNEL<3, false>(__VA_ARGS__); return true; \
    case 4: half ? KERNEL<4, true>(__VA_ARGS__) : KERNEL<4, false>(__VA_ARGS__); return true; \
    case 5: half ? KERNEL<5, true>(__VA_ARGS__) : KERNEL<5, false>(__VA_ARGS__); return true; \
    case 6: half ? KERNEL<6, true>(__VA_ARGS__) : KERNEL<6, false>(__VA_ARGS__); return true; \
    case 7: half ? KERNEL<7, true>(__VA_ARGS__) : KERNEL<7, false>(__VA_ARGS__); return true; \
    case 8: half ? KERNEL<8, true>(__VA_ARGS__) : KERNEL<8, false>(__VA_ARGS__); return true; \
    default: return false;                                                                   \
    }
bool attend_simd(Kernel kernel, size_t group, bool half, const float* q, const void* keys, const void* values,
                 size_t begin, size_t end, size_t dim, float* scores, float* out, size_t stride) {
#ifdef DECODE_X86
    if (kernel == Kernel::avx512 && dim % 32 == 0) { DECODE_GROUP_CASES(attend_avx512, q, keys, values, begin, end, dim, scores, out, stride) }
#endif
#ifdef DECODE_NEON
    if (kernel == Kernel::neon && dim == 64) { DECODE_GROUP_CASES(attend_neon64, q, keys, values, begin, end, dim, scores, out, stride) }
    if (kernel == Kernel::neon && dim % 8 == 0) { DECODE_GROUP_CASES(attend_neon, q, keys, values, begin, end, dim, scores, out, stride) }
#endif
    (void)kernel; (void)group; (void)half; (void)q; (void)keys; (void)values; (void)begin; (void)end; (void)dim; (void)scores; (void)out; (void)stride;
    return false;
}
#undef DECODE_GROUP_CASES
} // namespace

size_t kv_rows(size_t capacity) { return (capacity + tile - 1) / tile * tile; }

void write_kv(void* keys, void* values, KvType kv, size_t capacity, size_t dim, size_t head, size_t position,
              const float* k, const float* v) {
    if (position >= capacity) throw std::runtime_error("KV position outside cache capacity");
    const size_t base = head * kv_rows(capacity) * dim;
    const size_t key = base + (position / tile) * tile * dim + position % tile, row = base + position * dim;
    if (kv == KvType::f16) {
        uint16_t* kh = static_cast<uint16_t*>(keys);
        uint16_t* vh = static_cast<uint16_t*>(values);
        for (size_t d = 0; d < dim; ++d) { kh[key + d * tile] = float_half(k[d]); vh[row + d] = float_half(v[d]); }
    } else {
        float* kf = static_cast<float*>(keys);
        for (size_t d = 0; d < dim; ++d) kf[key + d * tile] = k[d];
        std::memcpy(static_cast<float*>(values) + row, v, dim * sizeof(float));
    }
}

AttentionPlan plan_attention(size_t length, size_t heads, size_t kv_heads, size_t dim, int threads) {
    if (!length || !kv_heads || !heads || heads % kv_heads || !dim || threads < 1)
        throw std::runtime_error("invalid attention dimensions");
    AttentionPlan plan;
    plan.length = length;
    plan.group = heads / kv_heads;
    // Enough chunks for every thread, at least 64 rows each, whole tiles only.
    const size_t wanted = threads == 1 ? 1 : std::max<size_t>(1, std::min<size_t>(size_t(threads), (length + 63) / 64));
    plan.chunk = ((length + wanted - 1) / wanted + tile - 1) / tile * tile;
    plan.chunks = (length + plan.chunk - 1) / plan.chunk;
    plan.items = kv_heads * plan.chunks;
    plan.partial_stride = dim + 2;
    return plan;
}
size_t attention_partial_floats(size_t heads, size_t kv_heads, size_t dim, int threads) {
    return kv_heads * size_t(threads) * (heads / kv_heads) * (dim + 2);
}
size_t attention_scratch_floats(size_t group, size_t) { return group * tile + 16; }

void attention_item(const AttentionPlan& plan, size_t item, const float* q, const void* keys, const void* values,
                    KvType kv, size_t capacity, size_t dim, float* scratch, float* partials, Kernel kernel) {
    const size_t head = item / plan.chunks, chunk = item % plan.chunks;
    const size_t begin = std::min(plan.length, chunk * plan.chunk), end = std::min(plan.length, begin + plan.chunk);
    const size_t width = kv == KvType::f16 ? 2 : 4, offset = head * kv_rows(capacity) * dim * width;
    const void* k = static_cast<const char*>(keys) + offset;
    const void* v = static_cast<const char*>(values) + offset;
    const float* qg = q + head * plan.group * dim;
    float* out = partials + item * plan.group * plan.partial_stride;
    for (size_t h = 0; h < plan.group; ++h) {
        float* state = out + h * plan.partial_stride;
        state[0] = negative_infinity;
        std::fill(state + 1, state + plan.partial_stride, 0.0f);
    }
    if (!attend_simd(kernel, plan.group, kv == KvType::f16, qg, k, v, begin, end, dim, scratch, out, plan.partial_stride))
        attend_scalar(qg, k, v, kv, begin, end, plan.group, dim, scratch, out, plan.partial_stride);
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
    std::vector<float> partials(attention_partial_floats(heads, kv_heads, dim, threads));
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
