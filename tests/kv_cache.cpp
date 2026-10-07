#include "kv_cache.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using decode::CacheType;
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance, const std::string& message) {
    require(std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= tolerance,
            message + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}
void rejects(const std::function<void()>& operation, const std::string& message) {
    bool rejected = false;
    try { operation(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, message);
}
// Independent token-major oracle: materialize only test reference data, never
// feed a whole-cache float mirror into the implementation under test.
std::vector<float> representation(const std::vector<float>& input, size_t length,
                                  size_t kv_heads, size_t dim, CacheType type,
                                  bool keys, bool restore_mean = false) {
    size_t stride = kv_heads * dim;
    std::vector<float> output(length * stride), mean(stride);
    bool centered = keys && type == CacheType::i8_centered && length >= 64;
    if (centered) for (size_t j = 0; j < stride; ++j) {
        double total = 0;
        for (size_t t = 0; t < 64; ++t) total += input[t * stride + j];
        mean[j] = float(total / 64);
    }
    for (size_t t = 0; t < length; ++t) for (size_t h = 0; h < kv_heads; ++h) {
        double largest = 0;
        for (size_t j = 0; j < dim; ++j) {
            size_t channel = h * dim + j;
            largest = std::max(largest, std::abs(double(input[t * stride + channel]) - (centered ? mean[channel] : 0.0)));
        }
        float scale = largest == 0 ? 1.0f : float(std::max(std::min(largest, double(std::numeric_limits<float>::max())) / 127,
                                                        double(std::numeric_limits<float>::min())));
        if (double(scale) * 127 > std::numeric_limits<float>::max()) scale = std::nextafter(scale, 0.0f);
        for (size_t j = 0; j < dim; ++j) {
            size_t channel = h * dim + j, index = t * stride + channel;
            if (type == CacheType::f32 || (keys && type == CacheType::i8_centered && length < 64)) output[index] = input[index];
            else if (type == CacheType::f16) output[index] = decode::half_float(decode::float_half(input[index]));
            else {
                double residual = double(input[index]) - (centered ? mean[channel] : 0.0);
                output[index] = float(std::clamp(std::round(residual / double(scale)), -127.0, 127.0)) * scale;
                if (centered && restore_mean) output[index] += mean[channel];
            }
        }
    }
    return output;
}
std::vector<double> reference(const std::vector<float>& q, const std::vector<float>& keys,
                              const std::vector<float>& values, size_t length,
                              size_t heads, size_t kv_heads, size_t dim) {
    std::vector<double> output(heads * dim), scores(length);
    for (size_t h = 0; h < heads; ++h) {
        size_t kv = h / (heads / kv_heads);
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t t = 0; t < length; ++t) {
            double dot = 0;
            for (size_t j = 0; j < dim; ++j) dot += double(q[h * dim + j]) * keys[(t * kv_heads + kv) * dim + j];
            scores[t] = dot / std::sqrt(double(dim));
            maximum = std::max(maximum, scores[t]);
        }
        double sum = 0;
        for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
        for (size_t t = 0; t < length; ++t) for (size_t j = 0; j < dim; ++j)
            output[h * dim + j] += scores[t] / sum * values[(t * kv_heads + kv) * dim + j];
    }
    return output;
}
void check(decode::KvCache& cache, CacheType type, const std::vector<float>& q,
           const std::vector<float>& keys, const std::vector<float>& values,
           size_t length, size_t heads, size_t kv_heads, size_t dim,
           decode::ThreadPool& pool, decode::AttentionWorkspace& workspace) {
    auto represented_keys = representation(keys, length, kv_heads, dim, type, true);
    auto represented_values = representation(values, length, kv_heads, dim, type, false);
    auto expected = reference(q, represented_keys, represented_values, length, heads, kv_heads, dim);
    std::vector<float> output(heads * dim);
    for (bool scalar : {true, false}) {
        cache.attend(q.data(), output.data(), length, pool, workspace, scalar);
        for (size_t j = 0; j < output.size(); ++j)
            near(output[j], expected[j], 8e-6, scalar ? "scalar represented KV oracle" : "blocked represented KV oracle");
    }
    if (type == CacheType::i8_centered && length >= 64) {
        auto restored = representation(keys, length, kv_heads, dim, type, true, true);
        auto with_mean = reference(q, restored, represented_values, length, heads, kv_heads, dim);
        for (size_t j = 0; j < expected.size(); ++j) near(with_mean[j], expected[j], 2e-6, "common mean softmax invariance");
    }
}
void edge_and_replay_tests() {
    constexpr size_t capacity = 131, kv_heads = 2;
    for (auto type : {CacheType::f16, CacheType::f32, CacheType::i8, CacheType::i8_centered})
    for (size_t dim : {size_t(5), size_t(16), size_t(32)})
    for (size_t groups : {size_t(2), size_t(3), size_t(7)})
    for (int threads : {1, 2}) {
        size_t heads = kv_heads * groups, stride = kv_heads * dim;
        decode::KvCache cache(type, capacity, heads, kv_heads, dim);
        decode::ThreadPool pool(threads);
        decode::AttentionWorkspace workspace(capacity, heads, dim);
        std::vector<float> q(heads * dim), keys(capacity * stride), values(capacity * stride);
        for (size_t j = 0; j < q.size(); ++j) q[j] = std::cos(float(j) * 0.43f) * 0.7f;
        for (size_t t = 0; t < capacity; ++t) {
            for (size_t j = 0; j < stride; ++j) {
                keys[t * stride + j] = 4 + float(j % 3) + std::sin(float(t * stride + j) * 0.31f) * 0.6f;
                values[t * stride + j] = std::cos(float(t * stride + j) * 0.17f) * 1.7f;
            }
            cache.store(t, keys.data() + t * stride, values.data() + t * stride);
            size_t length = t + 1;
            if (length == 1 || length == 15 || length == 16 || length == 17 || length == 63 || length == 64 ||
                length == 65 || length == 127 || length == 128 || length == 129 || length == capacity)
                check(cache, type, q, keys, values, length, heads, kv_heads, dim, pool, workspace);
        }
        // After a completed prefix, rewinding must read the saved raw keys,
        // not centered bytes or an evolving/future-dependent mean.
        check(cache, type, q, keys, values, 31, heads, kv_heads, dim, pool, workspace);
        for (size_t t = 27; t < capacity; ++t) {
            for (size_t j = 0; j < stride; ++j) {
                keys[t * stride + j] = -3 + std::cos(float(t + j) * 0.23f) * 0.8f;
                values[t * stride + j] = std::sin(float(t * stride + j) * 0.19f);
            }
            cache.store(t, keys.data() + t * stride, values.data() + t * stride);
            if (t == 30 || t == 62 || t == 63 || t == 64 || t == 128)
                check(cache, type, q, keys, values, t + 1, heads, kv_heads, dim, pool, workspace);
        }
        // Overwriting position 63 alone must also recompute all prefix bytes.
        for (size_t j = 0; j < stride; ++j) keys[63 * stride + j] += 0.9f;
        cache.store(63, keys.data() + 63 * stride, values.data() + 63 * stride);
        check(cache, type, q, keys, values, 64, heads, kv_heads, dim, pool, workspace);
    }
}
void zero_tiny_and_validation_tests() {
    constexpr size_t capacity = 65, heads = 4, kv_heads = 2, dim = 16, stride = kv_heads * dim;
    decode::ThreadPool pool(1);
    decode::AttentionWorkspace workspace(capacity, heads, dim);
    std::vector<float> q(heads * dim, 0.5f), keys(capacity * stride), values(capacity * stride), output(heads * dim);
    for (auto type : {CacheType::i8, CacheType::i8_centered}) {
        decode::KvCache cache(type, capacity, heads, kv_heads, dim);
        for (size_t t = 0; t < capacity; ++t) cache.store(t, keys.data() + t * stride, values.data() + t * stride);
        check(cache, type, q, keys, values, 63, heads, kv_heads, dim, pool, workspace);
        check(cache, type, q, keys, values, 64, heads, kv_heads, dim, pool, workspace);
        for (bool scalar : {true, false}) {
            cache.attend(q.data(), output.data(), capacity, pool, workspace, scalar);
            for (float value : output) near(value, 0, 0, "zero rows exact");
        }
        for (size_t j = 0; j < keys.size(); ++j) {
            keys[j] = (j % 2 ? -1 : 1) * std::numeric_limits<float>::denorm_min();
            values[j] = (j % 3 ? -1 : 1) * std::numeric_limits<float>::min() * 1e-3f;
        }
        for (size_t t = 0; t < capacity; ++t) cache.store(t, keys.data() + t * stride, values.data() + t * stride);
        for (bool scalar : {true, false}) {
            cache.attend(q.data(), output.data(), capacity, pool, workspace, scalar);
            for (float value : output) near(value, 0, 0, "tiny inputs quantize safely to zero");
        }
        std::fill(keys.begin(), keys.end(), 0);
        std::fill(values.begin(), values.end(), 0);
    }
    for (auto type : {CacheType::f16, CacheType::f32, CacheType::i8, CacheType::i8_centered}) {
        decode::KvCache cache(type, capacity, heads, kv_heads, dim);
        for (size_t t = 0; t < capacity; ++t) cache.store(t, keys.data() + t * stride, values.data() + t * stride);
        for (float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
            keys[stride - 1] = invalid;
            rejects([&] { cache.store(63, keys.data(), values.data()); }, "nonfinite K rejected");
            keys[stride - 1] = 0;
            values[stride - 1] = invalid;
            rejects([&] { cache.store(63, keys.data(), values.data()); }, "nonfinite V rejected");
            values[stride - 1] = 0;
            check(cache, type, q, keys, values, 64, heads, kv_heads, dim, pool, workspace);
        }
        rejects([&] { cache.store(capacity, keys.data(), values.data()); }, "out of capacity store rejected");
        rejects([&] { cache.store(0, nullptr, values.data()); }, "null store rejected");
        rejects([&] { cache.attend(q.data(), output.data(), 0, pool, workspace); }, "empty attention rejected");
        rejects([&] { cache.attend(q.data(), output.data(), capacity + 1, pool, workspace); }, "long attention rejected");
        decode::AttentionWorkspace wrong(capacity, heads, dim + 1);
        rejects([&] { cache.attend(q.data(), output.data(), 1, pool, wrong); }, "workspace mismatch rejected");
    }
    // Upper finite boundary must not create infinite decoded int8 values.
    decode::KvCache extreme(CacheType::i8, 1, 1, 1, 16);
    std::vector<float> extreme_keys(16), extreme_values(16, std::numeric_limits<float>::max()), extreme_q(16), extreme_output(16);
    decode::AttentionWorkspace extreme_workspace(1, 1, 16);
    extreme.store(0, extreme_keys.data(), extreme_values.data());
    for (bool scalar : {true, false}) {
        extreme.attend(extreme_q.data(), extreme_output.data(), 1, pool, extreme_workspace, scalar);
        for (float value : extreme_output) require(std::isfinite(value) && value > 0, "upper F32 scale boundary finite");
    }
}
void memory_tests() {
    for (size_t capacity : {size_t(1), size_t(63), size_t(64), size_t(65), size_t(131)})
    for (auto type : {CacheType::f16, CacheType::f32, CacheType::i8, CacheType::i8_centered}) {
        constexpr size_t kv_heads = 2, dim = 5, stride = kv_heads * dim;
        decode::KvCache cache(type, capacity, 4, kv_heads, dim);
        size_t key_elements = ((capacity + 63) / 64) * 64 * stride;
        size_t value_elements = capacity * stride;
        size_t element_bytes = type == CacheType::f16 ? 2 : type == CacheType::f32 ? 4 : 1;
        size_t scales = (type == CacheType::i8 || type == CacheType::i8_centered) ? 8 * capacity * kv_heads : 0;
        size_t mean = type == CacheType::i8_centered ? 4 * stride : 0;
        size_t prefix = type == CacheType::i8_centered ? 4 * std::min(capacity, size_t(64)) * stride : 0;
        require(cache.data_bytes() >= element_bytes * (key_elements + value_elements), "allocated data includes padding");
        require(cache.scale_bytes() >= scales && cache.mean_bytes() >= mean && cache.prefix_bytes() >= prefix, "all retained components counted");
        require(cache.bytes() == cache.data_bytes() + cache.scale_bytes() + cache.mean_bytes() + cache.prefix_bytes(), "byte components sum");
        require(decode::KvCache::allocation_bytes(type, capacity, kv_heads, dim) == element_bytes * (key_elements + value_elements) + scales + mean + prefix, "preflight formula");
        if (type == CacheType::i8) require(cache.mean_bytes() == 0 && cache.prefix_bytes() == 0, "plain mode has no float mirror");
        if (type == CacheType::f16 || type == CacheType::f32) require(cache.scale_bytes() == 0 && cache.mean_bytes() == 0 && cache.prefix_bytes() == 0, "floating modes allocate no quantized buffers");
    }
    rejects([] { decode::KvCache cache(CacheType::i8, 0, 1, 1, 1); }, "empty capacity rejected");
    rejects([] { decode::KvCache cache(CacheType::i8, 1, 3, 2, 1); }, "invalid GQA rejected");
    rejects([] { decode::KvCache::allocation_bytes(CacheType::i8, std::numeric_limits<size_t>::max(), 2, 16); }, "allocation overflow rejected");
    rejects([] { decode::KvCache::allocation_bytes(static_cast<CacheType>(99), 1, 1, 1); }, "unknown type rejected");
}
} // namespace
int main() {
    try {
        edge_and_replay_tests();
        zero_tiny_and_validation_tests();
        memory_tests();
        std::cout << "KV cache synthetic tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
