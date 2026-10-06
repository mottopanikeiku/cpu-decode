#include "decode.hpp"
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
    size_t length, heads, kv_heads, dim, active_blocks;
    AttentionWorkspace& workspace;
};
void scalar_head(void* context, size_t h) noexcept {
    auto& w = *static_cast<Work*>(context);
    size_t stride = w.kv_heads * w.dim, kv = h / (w.heads / w.kv_heads);
    float* probability = w.workspace.scores.data() + h * w.workspace.capacity;
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t t = 0; t < w.length; ++t) {
        float dot = 0;
        for (size_t j = 0; j < w.dim; ++j) dot += w.q[h * w.dim + j] * cache_value(w.keys, w.type, t * stride + kv * w.dim + j);
        probability[t] = dot / std::sqrt(float(w.dim)); maximum = std::max(maximum, probability[t]);
    }
    float sum = 0;
    for (size_t t = 0; t < w.length; ++t) { probability[t] = std::exp(probability[t] - maximum); sum += probability[t]; }
    std::fill(w.out + h * w.dim, w.out + (h + 1) * w.dim, 0);
    for (size_t t = 0; t < w.length; ++t) for (size_t j = 0; j < w.dim; ++j)
        w.out[h * w.dim + j] += (probability[t] / sum) * cache_value(w.values, w.type, t * stride + kv * w.dim + j);
}
void block_scalar(Work& w, size_t task) noexcept {
    size_t kv = task / w.active_blocks, block = task % w.active_blocks, groups = w.heads / w.kv_heads;
    size_t begin = block * AttentionWorkspace::block_size, end = std::min(w.length, begin + AttentionWorkspace::block_size);
    size_t slot = (kv * w.workspace.blocks + block) * groups, stride = w.kv_heads * w.dim;
    float scale = 1 / std::sqrt(float(w.dim));
    for (size_t g = 0; g < groups; ++g) w.workspace.maxima[slot + g] = -std::numeric_limits<float>::infinity();
    for (size_t t = begin; t < end; ++t) {
        float dots[64]{};
        // One K load/conversion per element, reused across every query in this KV group.
        for (size_t j = 0; j < w.dim; ++j) {
            float key = cache_value(w.keys, w.type, t * stride + kv * w.dim + j);
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
        float sum = 0;
        for (size_t t = begin; t < end; ++t) { probability[t] = std::exp(probability[t] - w.workspace.maxima[slot + g]); sum += probability[t]; }
        w.workspace.sums[slot + g] = sum;
        std::fill(w.workspace.weighted.data() + (slot + g) * w.dim, w.workspace.weighted.data() + (slot + g + 1) * w.dim, 0);
    }
    for (size_t t = begin; t < end; ++t) for (size_t j = 0; j < w.dim; ++j) {
        float value = cache_value(w.values, w.type, t * stride + kv * w.dim + j);
        for (size_t g = 0; g < groups; ++g) w.workspace.weighted[(slot + g) * w.dim + j] += w.workspace.scores[(kv * groups + g) * w.workspace.capacity + t] * value;
    }
}
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx512f"), always_inline))
inline __m512 load_cache(const void* data, CacheType type, size_t index) {
    if (type == CacheType::f16) return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const uint16_t*>(data) + index)));
    return _mm512_loadu_ps(static_cast<const float*>(data) + index);
}
template<size_t Groups>
__attribute__((target("avx512f")))
void block_simd(Work& w, size_t task) noexcept {
    size_t kv = task / w.active_blocks, block = task % w.active_blocks;
    size_t begin = block * AttentionWorkspace::block_size, end = std::min(w.length, begin + AttentionWorkspace::block_size);
    size_t slot = (kv * w.workspace.blocks + block) * Groups, stride = w.kv_heads * w.dim;
    float scale = 1 / std::sqrt(float(w.dim));
    for (size_t g = 0; g < Groups; ++g) w.workspace.maxima[slot + g] = -std::numeric_limits<float>::infinity();
    for (size_t t = begin; t < end; ++t) {
        __m512 dots[Groups];
        for (auto& dot : dots) dot = _mm512_setzero_ps();
        for (size_t j = 0; j < w.dim; j += 16) {
            __m512 key = load_cache(w.keys, w.type, t * stride + kv * w.dim + j);
            for (size_t g = 0; g < Groups; ++g) dots[g] = _mm512_add_ps(dots[g], _mm512_mul_ps(key, _mm512_loadu_ps(w.q + (kv * Groups + g) * w.dim + j)));
        }
        for (size_t g = 0; g < Groups; ++g) {
            float score = _mm512_reduce_add_ps(dots[g]) * scale;
            w.workspace.scores[(kv * Groups + g) * w.workspace.capacity + t] = score;
            w.workspace.maxima[slot + g] = std::max(w.workspace.maxima[slot + g], score);
        }
    }
    for (size_t g = 0; g < Groups; ++g) {
        float* probability = w.workspace.scores.data() + (kv * Groups + g) * w.workspace.capacity;
        float sum = 0;
        for (size_t t = begin; t < end; ++t) { probability[t] = std::exp(probability[t] - w.workspace.maxima[slot + g]); sum += probability[t]; }
        w.workspace.sums[slot + g] = sum;
    }
    for (size_t j = 0; j < w.dim; j += 16) {
        __m512 accumulators[Groups];
        for (auto& acc : accumulators) acc = _mm512_setzero_ps();
        for (size_t t = begin; t < end; ++t) {
            __m512 value = load_cache(w.values, w.type, t * stride + kv * w.dim + j);
            for (size_t g = 0; g < Groups; ++g) {
                float probability = w.workspace.scores[(kv * Groups + g) * w.workspace.capacity + t];
                accumulators[g] = _mm512_add_ps(accumulators[g], _mm512_mul_ps(value, _mm512_set1_ps(probability)));
            }
        }
        for (size_t g = 0; g < Groups; ++g) _mm512_storeu_ps(w.workspace.weighted.data() + (slot + g) * w.dim + j, accumulators[g]);
    }
}
#endif
void block(void* context, size_t task) noexcept {
    auto& w = *static_cast<Work*>(context);
#if defined(__x86_64__) || defined(__i386__)
    if (!(w.dim % 16) && __builtin_cpu_supports("avx512f")) {
        switch (w.heads / w.kv_heads) {
            case 1: block_simd<1>(w, task); return;
            case 2: block_simd<2>(w, task); return;
            case 4: block_simd<4>(w, task); return;
            case 7: block_simd<7>(w, task); return;
            case 8: block_simd<8>(w, task); return;
        }
    }
#endif
    block_scalar(w, task);
}
}
void attention(const float* q, const void* keys, const void* values, CacheType type,
               float* out, size_t length, size_t heads, size_t kv_heads, size_t dim,
               ThreadPool& pool, AttentionWorkspace& workspace, bool scalar) {
    if (!length || !kv_heads || heads % kv_heads || !dim || heads / kv_heads > 64 ||
        workspace.capacity < length || workspace.heads != heads || workspace.dim != dim)
        throw std::runtime_error("invalid attention dimensions");
    Work work{q, keys, values, type, out, length, heads, kv_heads, dim, (length + AttentionWorkspace::block_size - 1) / AttentionWorkspace::block_size, workspace};
    if (scalar) { pool.run(heads, scalar_head, &work); return; }
    pool.run(kv_heads * work.active_blocks, block, &work);
    const size_t groups = heads / kv_heads;
    // Block boundaries and this ascending merge order never depend on thread count.
    for (size_t h = 0; h < heads; ++h) {
        size_t kv = h / groups, g = h % groups;
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t b = 0; b < work.active_blocks; ++b) maximum = std::max(maximum, workspace.maxima[(kv * workspace.blocks + b) * groups + g]);
        float denominator = 0;
        std::fill(out + h * dim, out + (h + 1) * dim, 0);
        for (size_t b = 0; b < work.active_blocks; ++b) {
            size_t slot = (kv * workspace.blocks + b) * groups + g;
            float factor = std::exp(workspace.maxima[slot] - maximum);
            denominator += workspace.sums[slot] * factor;
            for (size_t j = 0; j < dim; ++j) out[h * dim + j] += workspace.weighted[slot * dim + j] * factor;
        }
        for (size_t j = 0; j < dim; ++j) out[h * dim + j] /= denominator;
    }
}
} // namespace decode
