#include "kv_cache.hpp"
#include "simd_math.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace decode {
AttentionWorkspace::AttentionWorkspace(size_t cap, size_t nheads, size_t dimension)
    : capacity(cap), heads(nheads), dim(dimension), blocks((cap + block_size - 1) / block_size),
      scores(cap * nheads), maxima(blocks * nheads), sums(blocks * nheads), weighted(blocks * nheads * dimension) {}
namespace {
float cache_value(const void* data, CacheType type, size_t index) {
    return type == CacheType::f16 ? half_float(static_cast<const uint16_t*>(data)[index]) : static_cast<const float*>(data)[index];
}
struct Work {
    const float* q;
    const void* keys;
    const void* values;
    CacheType type;
    float* out;
    size_t length, heads, kv_heads, dim, active_blocks, key_blocks;
    AttentionWorkspace& workspace;
    const float* key_scales = nullptr;
    const float* value_scales = nullptr;
    const float* raw_prefix = nullptr;
};
size_t key_index(const Work& w, size_t kv, size_t position, size_t column) noexcept {
    constexpr size_t width = AttentionWorkspace::block_size;
    return ((kv * w.key_blocks + position / width) * w.dim + column) * width + position % width;
}
float key_value(const Work& w, size_t kv, size_t position, size_t column) noexcept {
    if (w.raw_prefix && w.length < KvCache::key_mean_prefix)
        return w.raw_prefix[(position * w.kv_heads + kv) * w.dim + column];
    size_t index = key_index(w, kv, position, column);
    if (w.key_scales)
        return float(static_cast<const int8_t*>(w.keys)[index]) * w.key_scales[position * w.kv_heads + kv];
    return cache_value(w.keys, w.type, index);
}
float value_value(const Work& w, size_t kv, size_t position, size_t column) noexcept {
    size_t index = (position * w.kv_heads + kv) * w.dim + column;
    if (w.value_scales)
        return float(static_cast<const int8_t*>(w.values)[index]) * w.value_scales[position * w.kv_heads + kv];
    return cache_value(w.values, w.type, index);
}
void scalar_head(void* context, size_t h) noexcept {
    auto& w = *static_cast<Work*>(context);
    size_t kv = h / (w.heads / w.kv_heads);
    float* probability = w.workspace.scores.data() + h * w.workspace.capacity;
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t t = 0; t < w.length; ++t) {
        float dot = 0;
        for (size_t j = 0; j < w.dim; ++j) dot += w.q[h * w.dim + j] * key_value(w, kv, t, j);
        probability[t] = dot / std::sqrt(float(w.dim)); maximum = std::max(maximum, probability[t]);
    }
    float sum = 0;
    for (size_t t = 0; t < w.length; ++t) { probability[t] = std::exp(probability[t] - maximum); sum += probability[t]; }
    std::fill(w.out + h * w.dim, w.out + (h + 1) * w.dim, 0);
    for (size_t t = 0; t < w.length; ++t) for (size_t j = 0; j < w.dim; ++j)
        w.out[h * w.dim + j] += (probability[t] / sum) * value_value(w, kv, t, j);
}
void block_scalar(Work& w, size_t task) noexcept {
    size_t kv = task / w.active_blocks, block = task % w.active_blocks, groups = w.heads / w.kv_heads;
    size_t begin = block * AttentionWorkspace::block_size, end = std::min(w.length, begin + AttentionWorkspace::block_size);
    size_t slot = (kv * w.workspace.blocks + block) * groups;
    float scale = 1 / std::sqrt(float(w.dim));
    for (size_t g = 0; g < groups; ++g) w.workspace.maxima[slot + g] = -std::numeric_limits<float>::infinity();
    for (size_t t = begin; t < end; ++t) {
        float dots[64]{};
        // One K load/conversion per element, reused across every query in this KV group.
        for (size_t j = 0; j < w.dim; ++j) {
            float key = key_value(w, kv, t, j);
            for (size_t g = 0; g < groups; ++g) dots[g] += key * w.q[(kv * groups + g) * w.dim + j];
        }
        for (size_t g = 0; g < groups; ++g) {
            float score = dots[g] * scale;
            w.workspace.scores[(kv * groups + g) * w.workspace.capacity + t] = score;
            w.workspace.maxima[slot + g] = std::max(w.workspace.maxima[slot + g], score);
        }
    }
    for (size_t g = 0; g < groups; ++g) {
        float* probability = w.workspace.scores.data() + (kv * groups + g) * w.workspace.capacity;
        // -inf is the neutral block maximum, not a subtraction operand.
        // NaN scores still propagate through exp and the merge.
        float maximum = w.workspace.maxima[slot + g];
        if (maximum == -std::numeric_limits<float>::infinity()) maximum = 0;
        float sum = 0;
        for (size_t t = begin; t < end; ++t) { probability[t] = std::exp(probability[t] - maximum); sum += probability[t]; }
        w.workspace.sums[slot + g] = sum;
        float inverse = sum == 0 ? 0 : 1 / sum;
        for (size_t t = begin; t < end; ++t) probability[t] *= inverse;
        std::fill(w.workspace.weighted.data() + (slot + g) * w.dim, w.workspace.weighted.data() + (slot + g + 1) * w.dim, 0);
    }
    for (size_t t = begin; t < end; ++t) for (size_t j = 0; j < w.dim; ++j) {
        float value = value_value(w, kv, t, j);
        for (size_t g = 0; g < groups; ++g) w.workspace.weighted[(slot + g) * w.dim + j] += w.workspace.scores[(kv * groups + g) * w.workspace.capacity + t] * value;
    }
}
#if defined(__x86_64__) || defined(__i386__)
template<CacheType Type>
__attribute__((target("avx512f"), always_inline))
inline __m512 load_cache(const void* data, size_t index) {
    if constexpr (Type == CacheType::f16) return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const uint16_t*>(data) + index)));
    else if constexpr (Type == CacheType::i8)
        return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const int8_t*>(data) + index))));
    else return _mm512_loadu_ps(static_cast<const float*>(data) + index);
}
template<size_t Groups, CacheType Type>
__attribute__((target("avx512f")))
void block_simd_typed(Work& w, size_t task) noexcept {
    size_t kv = task / w.active_blocks, block = task % w.active_blocks;
    size_t begin = block * AttentionWorkspace::block_size, end = std::min(w.length, begin + AttentionWorkspace::block_size);
    size_t slot = (kv * w.workspace.blocks + block) * Groups, stride = w.kv_heads * w.dim;
    float scale = 1 / std::sqrt(float(w.dim));
    for (size_t g = 0; g < Groups; ++g) w.workspace.maxima[slot + g] = -std::numeric_limits<float>::infinity();
    const size_t key_start = (kv * w.key_blocks + block) * w.dim * AttentionWorkspace::block_size;
    const __m512 multiplier = _mm512_set1_ps(scale), neutral = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (size_t t = begin; t < end; t += 16) {
        size_t valid = std::min(size_t(16), end - t);
        __mmask16 mask = valid == 16 ? __mmask16(0xffff) : __mmask16((1u << valid) - 1);
        __m512 key_scales = _mm512_set1_ps(1);
        if constexpr (Type == CacheType::i8) {
            __m512i offsets = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                                _mm512_set1_epi32(int(w.kv_heads)));
            key_scales = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, offsets, w.key_scales + t * w.kv_heads + kv, 4);
        }
        __m512 dots[Groups];
        for (auto& dot : dots) dot = _mm512_setzero_ps();
        for (size_t j = 0; j < w.dim; ++j) {
            // Each lane is a token; one transposed K load serves every query
            // in this KV group, without a horizontal dot-product reduction.
            __m512 key = load_cache<Type>(w.keys, key_start + j * AttentionWorkspace::block_size + t - begin);
            if constexpr (Type == CacheType::i8) key = _mm512_mul_ps(key, key_scales);
            for (size_t g = 0; g < Groups; ++g)
                dots[g] = _mm512_fmadd_ps(key, _mm512_set1_ps(w.q[(kv * Groups + g) * w.dim + j]), dots[g]);
        }
        for (size_t g = 0; g < Groups; ++g) {
            __m512 scores = _mm512_mul_ps(dots[g], multiplier);
            _mm512_mask_storeu_ps(w.workspace.scores.data() + (kv * Groups + g) * w.workspace.capacity + t, mask, scores);
            float maximum = _mm512_reduce_max_ps(_mm512_mask_mov_ps(neutral, mask, scores));
            w.workspace.maxima[slot + g] = std::max(w.workspace.maxima[slot + g], maximum);
        }
    }
    for (size_t g = 0; g < Groups; ++g) {
        float* probability = w.workspace.scores.data() + (kv * Groups + g) * w.workspace.capacity;
        float block_maximum = w.workspace.maxima[slot + g];
        if (block_maximum == -std::numeric_limits<float>::infinity()) block_maximum = 0;
        __m512 lanes = _mm512_setzero_ps(), maximum = _mm512_set1_ps(block_maximum);
        for (size_t t = begin; t < end; t += 16) {
            size_t valid = std::min(size_t(16), end - t);
            __mmask16 mask = __mmask16(0xffffu >> (16 - valid));
            __m512 scores = _mm512_mask_loadu_ps(neutral, mask, probability + t);
            __m512 p = _mm512_maskz_mov_ps(mask, detail::exp_nonpositive(_mm512_sub_ps(scores, maximum)));
            _mm512_mask_storeu_ps(probability + t, mask, p);
            lanes = _mm512_add_ps(lanes, p);
        }
        float sum = _mm512_reduce_add_ps(lanes);
        w.workspace.sums[slot + g] = sum;
        float scalar_inverse = sum == 0 ? 0 : 1 / sum;
        __m512 inverse = _mm512_set1_ps(scalar_inverse);
        for (size_t t = begin; t < end; t += 16) {
            size_t valid = std::min(size_t(16), end - t);
            __mmask16 mask = __mmask16(0xffffu >> (16 - valid));
            __m512 p = _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, probability + t), inverse);
            _mm512_mask_storeu_ps(probability + t, mask, p);
        }
    }
    for (size_t j = 0; j < w.dim; j += 16) {
        __m512 accumulators[Groups];
        for (auto& acc : accumulators) acc = _mm512_setzero_ps();
        for (size_t t = begin; t < end; ++t) {
            __m512 value = load_cache<Type>(w.values, t * stride + kv * w.dim + j);
            if constexpr (Type == CacheType::i8)
                value = _mm512_mul_ps(value, _mm512_set1_ps(w.value_scales[t * w.kv_heads + kv]));
            for (size_t g = 0; g < Groups; ++g) {
                float probability = w.workspace.scores[(kv * Groups + g) * w.workspace.capacity + t];
                accumulators[g] = _mm512_fmadd_ps(value, _mm512_set1_ps(probability), accumulators[g]);
            }
        }
        for (size_t g = 0; g < Groups; ++g) _mm512_storeu_ps(w.workspace.weighted.data() + (slot + g) * w.dim + j, accumulators[g]);
    }
}
template<size_t Groups>
void block_simd(Work& w, size_t task) noexcept {
    if (w.type == CacheType::f16) block_simd_typed<Groups, CacheType::f16>(w, task);
    else if (w.key_scales) block_simd_typed<Groups, CacheType::i8>(w, task);
    else block_simd_typed<Groups, CacheType::f32>(w, task);
}
__attribute__((target("avx512f")))
void merge_simd_head(Work& w, size_t h) noexcept {
    size_t groups = w.heads / w.kv_heads, kv = h / groups, g = h % groups;
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t b = 0; b < w.active_blocks; ++b)
        maximum = std::max(maximum, w.workspace.maxima[(kv * w.workspace.blocks + b) * groups + g]);
    float denominator = 0;
    std::fill(w.out + h * w.dim, w.out + (h + 1) * w.dim, 0);
    for (size_t b = 0; b < w.active_blocks; ++b) {
        size_t slot = (kv * w.workspace.blocks + b) * groups + g;
        float factor = std::exp(w.workspace.maxima[slot] - maximum);
        // Reuse each local sum as its globally rescaled mass.
        w.workspace.sums[slot] *= factor;
        denominator += w.workspace.sums[slot];
    }
    // Local weighted averages and normalized merge coefficients keep
    // finite averages from overflowing an unnormalized intermediate.
    for (size_t b = 0; b < w.active_blocks; ++b) {
        size_t slot = (kv * w.workspace.blocks + b) * groups + g;
        __m512 multiplier = _mm512_set1_ps(w.workspace.sums[slot] / denominator);
        for (size_t j = 0; j < w.dim; j += 16) {
            float* output = w.out + h * w.dim + j;
            _mm512_storeu_ps(output, _mm512_fmadd_ps(_mm512_loadu_ps(w.workspace.weighted.data() + slot * w.dim + j),
                                                   multiplier, _mm512_loadu_ps(output)));
        }
    }
}
#endif
void block(void* context, size_t task) noexcept {
    auto& w = *static_cast<Work*>(context);
#if defined(__x86_64__) || defined(__i386__)
    if (!(w.dim % 16) && !(w.raw_prefix && w.length < KvCache::key_mean_prefix) && __builtin_cpu_supports("avx512f")) {
        switch (w.heads / w.kv_heads) {
            case 1: block_simd<1>(w, task); return;
            case 2: block_simd<2>(w, task); return;
            case 4: block_simd<4>(w, task); return;
            case 6: block_simd<6>(w, task); return;
            case 7: block_simd<7>(w, task); return;
            case 8: block_simd<8>(w, task); return;
        }
    }
#endif
    block_scalar(w, task);
}
void merge(void* context, size_t h) noexcept {
    auto& w = *static_cast<Work*>(context);
#if defined(__x86_64__) || defined(__i386__)
    if (!(w.dim % 16) && __builtin_cpu_supports("avx512f")) { merge_simd_head(w, h); return; }
#endif
    size_t groups = w.heads / w.kv_heads, kv = h / groups, g = h % groups;
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t b = 0; b < w.active_blocks; ++b)
        maximum = std::max(maximum, w.workspace.maxima[(kv * w.workspace.blocks + b) * groups + g]);
    float denominator = 0;
    std::fill(w.out + h * w.dim, w.out + (h + 1) * w.dim, 0);
    for (size_t b = 0; b < w.active_blocks; ++b) {
        size_t slot = (kv * w.workspace.blocks + b) * groups + g;
        float factor = std::exp(w.workspace.maxima[slot] - maximum);
        w.workspace.sums[slot] *= factor;
        denominator += w.workspace.sums[slot];
    }
    for (size_t b = 0; b < w.active_blocks; ++b) {
        size_t slot = (kv * w.workspace.blocks + b) * groups + g;
        float factor = w.workspace.sums[slot] / denominator;
        for (size_t j = 0; j < w.dim; ++j)
            w.out[h * w.dim + j] += w.workspace.weighted[slot * w.dim + j] * factor;
    }
}
}
void attention(const float* q, const void* keys, const void* values, CacheType type,
               float* out, size_t length, size_t heads, size_t kv_heads, size_t dim, size_t key_blocks,
               ThreadPool& pool, AttentionWorkspace& workspace, bool scalar) {
    if (!length || !kv_heads || heads % kv_heads || !dim || heads / kv_heads > 64 ||
        workspace.capacity < length || workspace.heads != heads || workspace.dim != dim ||
        key_blocks < length / AttentionWorkspace::block_size + (length % AttentionWorkspace::block_size != 0))
        throw std::runtime_error("invalid attention dimensions");
    if (type != CacheType::f16 && type != CacheType::f32)
        throw std::runtime_error("quantized attention requires KV scales");
    Work work{q, keys, values, type, out, length, heads, kv_heads, dim, (length + AttentionWorkspace::block_size - 1) / AttentionWorkspace::block_size, key_blocks, workspace};
    if (scalar) { pool.run(heads, scalar_head, &work); return; }
    pool.run(kv_heads * work.active_blocks, block, &work);
    // Each head merges fixed blocks in ascending order, independent of workers.
    pool.run(heads, merge, &work);
}
namespace detail {
void attention_quantized(const float* q, const int8_t* keys, const int8_t* values,
                         const float* key_scales, const float* value_scales,
                         const float* raw_prefix, float* out, size_t length,
                         size_t heads, size_t kv_heads, size_t dim, size_t key_blocks,
                         ThreadPool& pool, AttentionWorkspace& workspace, bool scalar) {
    if (!length || !heads || !kv_heads || heads % kv_heads || !dim || heads / kv_heads > 64 ||
        workspace.capacity < length || workspace.heads != heads || workspace.dim != dim ||
        key_blocks < length / AttentionWorkspace::block_size + (length % AttentionWorkspace::block_size != 0))
        throw std::runtime_error("invalid quantized attention dimensions");
    Work work{q, keys, values, CacheType::i8, out, length, heads, kv_heads, dim,
              length / AttentionWorkspace::block_size + (length % AttentionWorkspace::block_size != 0),
              key_blocks, workspace, key_scales, value_scales, raw_prefix};
    if (scalar) { pool.run(heads, scalar_head, &work); return; }
    pool.run(kv_heads * work.active_blocks, block, &work);
    pool.run(heads, merge, &work);
}
} // namespace detail
} // namespace decode
