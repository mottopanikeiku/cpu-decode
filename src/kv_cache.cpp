#include "kv_cache.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace decode {
namespace {
size_t product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error("KV cache dimensions overflow");
    return a * b;
}
size_t sum_bytes(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b)
        throw std::runtime_error("KV cache dimensions overflow");
    return a + b;
}
float safe_scale(double maximum) noexcept {
    // A normal positive floor avoids underflow and infinite reciprocals for
    // tiny/subnormal inputs. Zero rows use scale 1 and exact zero bytes.
    // Centered residuals beyond the F32 range saturate at the largest finite
    // representable int8 reconstruction instead of producing infinities.
    if (maximum == 0) return 1.0f;
    float scale = float(std::max(std::min(maximum, double(std::numeric_limits<float>::max())) / 127.0,
                                 double(std::numeric_limits<float>::min())));
    // Keep 127 * scale representable even at the F32 upper boundary.
    if (double(scale) * 127.0 > double(std::numeric_limits<float>::max()))
        scale = std::nextafter(scale, 0.0f);
    return scale;
}
int8_t quantized(double value, float scale) noexcept {
    return int8_t(std::clamp(std::round(value / double(scale)), -127.0, 127.0));
}
}
size_t KvCache::allocation_bytes(CacheType type, size_t capacity, size_t kv_heads, size_t dim) {
    size_t stride = product(kv_heads, dim);
    size_t blocks = capacity / AttentionWorkspace::block_size + (capacity % AttentionWorkspace::block_size != 0);
    size_t elements = sum_bytes(product(product(blocks, AttentionWorkspace::block_size), stride), product(capacity, stride));
    switch (type) {
        case CacheType::f16: return product(elements, sizeof(uint16_t));
        case CacheType::f32: return product(elements, sizeof(float));
        case CacheType::i8: return sum_bytes(elements, product(product(capacity, kv_heads), 2 * sizeof(float)));
        case CacheType::i8_centered:
            return sum_bytes(sum_bytes(elements, product(product(capacity, kv_heads), 2 * sizeof(float))),
                             product(product(sum_bytes(1, std::min(capacity, key_mean_prefix)), stride), sizeof(float)));
        default: throw std::runtime_error("invalid KV cache type");
    }
}
KvCache::KvCache(CacheType type, size_t capacity, size_t heads, size_t kv_heads, size_t dim)
    : type_(type), capacity_(capacity), heads_(heads), kv_heads_(kv_heads), dim_(dim),
      blocks_(capacity / AttentionWorkspace::block_size + (capacity % AttentionWorkspace::block_size != 0)) {
    if (!capacity || !heads || !kv_heads || heads % kv_heads || heads / kv_heads > 64 || !dim)
        throw std::runtime_error("invalid KV cache dimensions");
    (void)allocation_bytes(type, capacity, kv_heads, dim);
    size_t stride = product(kv_heads, dim);
    size_t keys = product(product(blocks_, AttentionWorkspace::block_size), stride);
    size_t values = product(capacity, stride);
    switch (type) {
        case CacheType::f16: half_keys_.resize(keys); half_values_.resize(values); break;
        case CacheType::f32: float_keys_.resize(keys); float_values_.resize(values); break;
        case CacheType::i8_centered:
            mean_.resize(stride);
            prefix_.resize(product(std::min(capacity, key_mean_prefix), stride));
            [[fallthrough]];
        case CacheType::i8:
            byte_keys_.resize(keys); byte_values_.resize(values);
            key_scales_.resize(product(capacity, kv_heads), 1.0f);
            value_scales_.resize(product(capacity, kv_heads), 1.0f);
            break;
        default: throw std::runtime_error("invalid KV cache type");
    }
}
size_t KvCache::key_index(size_t position, size_t head, size_t channel) const noexcept {
    constexpr size_t width = AttentionWorkspace::block_size;
    return ((head * blocks_ + position / width) * dim_ + channel) * width + position % width;
}
void KvCache::store_byte_key(size_t position, size_t head, const float* key) {
    const float* mean = type_ == CacheType::i8_centered ? mean_.data() + head * dim_ : nullptr;
    double maximum = 0;
    for (size_t j = 0; j < dim_; ++j)
        maximum = std::max(maximum, std::abs(double(key[j]) - (mean ? double(mean[j]) : 0.0)));
    float scale = safe_scale(maximum);
    key_scales_[position * kv_heads_ + head] = scale;
    for (size_t j = 0; j < dim_; ++j)
        byte_keys_[key_index(position, head, j)] = quantized(double(key[j]) - (mean ? double(mean[j]) : 0.0), scale);
}
void KvCache::store(size_t position, const float* keys, const float* values) {
    if (position >= capacity_ || !keys || !values) throw std::runtime_error("invalid KV cache store");
    size_t stride = kv_heads_ * dim_;
    // Validate the entire row before changing either K or V (including the
    // saved prefix). Rejected rows cannot corrupt a replay of the prefix.
    for (size_t j = 0; j < stride; ++j)
        if (!std::isfinite(keys[j]) || !std::isfinite(values[j]))
            throw std::runtime_error("KV cache store requires finite keys and values");
    if (type_ == CacheType::f16 || type_ == CacheType::f32) {
        for (size_t h = 0; h < kv_heads_; ++h) for (size_t j = 0; j < dim_; ++j) {
            size_t row = h * dim_ + j, index = key_index(position, h, j);
            if (type_ == CacheType::f16) {
                half_keys_[index] = float_half(keys[row]);
                half_values_[position * stride + row] = float_half(values[row]);
            } else {
                float_keys_[index] = keys[row];
                float_values_[position * stride + row] = values[row];
            }
        }
        return;
    }
    for (size_t h = 0; h < kv_heads_; ++h) {
        double maximum = 0;
        for (size_t j = 0; j < dim_; ++j) maximum = std::max(maximum, std::abs(double(values[h * dim_ + j])));
        float scale = safe_scale(maximum);
        value_scales_[position * kv_heads_ + h] = scale;
        for (size_t j = 0; j < dim_; ++j)
            byte_values_[position * stride + h * dim_ + j] = quantized(values[h * dim_ + j], scale);
    }
    if (type_ == CacheType::i8_centered && position < key_mean_prefix) {
        std::copy(keys, keys + stride, prefix_.data() + position * stride);
        if (position + 1 != key_mean_prefix) return;
        // Recompute at absolute position 63 on every replay, never evolve the
        // mean afterwards and never look at future tokens. Keep raw prefix K
        // for attention after a rewind below 64, even after quantization.
        for (size_t j = 0; j < stride; ++j) {
            double sum = 0;
            for (size_t t = 0; t < key_mean_prefix; ++t) sum += prefix_[t * stride + j];
            mean_[j] = float(sum / double(key_mean_prefix));
        }
        for (size_t t = 0; t < key_mean_prefix; ++t) for (size_t h = 0; h < kv_heads_; ++h)
            store_byte_key(t, h, prefix_.data() + t * stride + h * dim_);
    } else {
        for (size_t h = 0; h < kv_heads_; ++h) store_byte_key(position, h, keys + h * dim_);
    }
}
void KvCache::attend(const float* queries, float* output, size_t length, ThreadPool& pool,
                     AttentionWorkspace& workspace, bool scalar) const {
    if (!queries || !output || !length || length > capacity_)
        throw std::runtime_error("invalid KV cache attention");
    if (type_ == CacheType::f16)
        attention(queries, half_keys_.data(), half_values_.data(), type_, output, length, heads_, kv_heads_, dim_, blocks_, pool, workspace, scalar);
    else if (type_ == CacheType::f32)
        attention(queries, float_keys_.data(), float_values_.data(), type_, output, length, heads_, kv_heads_, dim_, blocks_, pool, workspace, scalar);
    else
        detail::attention_quantized(queries, byte_keys_.data(), byte_values_.data(), key_scales_.data(), value_scales_.data(),
                                    prefix_.empty() ? nullptr : prefix_.data(), output, length, heads_, kv_heads_, dim_, blocks_, pool, workspace, scalar);
}
size_t KvCache::data_bytes() const noexcept {
    return (half_keys_.capacity() + half_values_.capacity()) * sizeof(uint16_t)
         + (float_keys_.capacity() + float_values_.capacity()) * sizeof(float)
         + (byte_keys_.capacity() + byte_values_.capacity()) * sizeof(int8_t);
}
size_t KvCache::scale_bytes() const noexcept { return (key_scales_.capacity() + value_scales_.capacity()) * sizeof(float); }
size_t KvCache::mean_bytes() const noexcept { return mean_.capacity() * sizeof(float); }
size_t KvCache::prefix_bytes() const noexcept { return prefix_.capacity() * sizeof(float); }
size_t KvCache::bytes() const noexcept { return data_bytes() + scale_bytes() + mean_bytes() + prefix_bytes(); }
} // namespace decode
