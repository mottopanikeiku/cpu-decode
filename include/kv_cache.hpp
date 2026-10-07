#pragma once
#include "decode.hpp"

namespace decode {
// Keys use [kv_head][64-token block][channel][token lane]; values use
// [token][kv_head][channel]. Only the selected storage type is allocated.
// Centered i8 keeps only the first min(capacity,64) raw post-RoPE key
// vectors. Visible lengths below 64 read these raw keys; store(63) freezes
// their channel-wise mean and requantizes the prefix on every replay.
// Later keys subtract that fixed mean; values are int8 from the first token.
// Rewind is expressed by attention length and subsequent absolute stores.
class KvCache {
public:
    static constexpr size_t key_mean_prefix = 64;
    KvCache(CacheType type, size_t capacity, size_t heads, size_t kv_heads, size_t dim);
    static size_t allocation_bytes(CacheType type, size_t capacity, size_t kv_heads, size_t dim);
    void store(size_t position, const float* keys, const float* values);
    void attend(const float* queries, float* output, size_t length, ThreadPool& pool,
                AttentionWorkspace& workspace, bool scalar = false) const;
    // Retained buffer allocation only, excluding the object and the caller's
    // attention workspace. Includes padded keys, scales, mean and raw prefix.
    size_t bytes() const noexcept;
    size_t data_bytes() const noexcept;
    size_t scale_bytes() const noexcept;
    size_t mean_bytes() const noexcept;
    size_t prefix_bytes() const noexcept;
private:
    CacheType type_;
    size_t capacity_, heads_, kv_heads_, dim_, blocks_;
    std::vector<uint16_t> half_keys_, half_values_;
    std::vector<float> float_keys_, float_values_;
    std::vector<int8_t> byte_keys_, byte_values_;
    // Scales are token-major, one F32 scale per token/KV head.
    std::vector<float> key_scales_, value_scales_, mean_, prefix_;
    size_t key_index(size_t position, size_t head, size_t channel) const noexcept;
    void store_byte_key(size_t position, size_t head, const float* key);
};

namespace detail {
// Internal quantized attention binding. A non-null raw_prefix is used only
// for causal lengths below 64; otherwise every visible key is centered and
// the common q dot mean term cancels in softmax. Values are always int8.
void attention_quantized(const float* queries, const int8_t* keys, const int8_t* values,
                         const float* key_scales, const float* value_scales,
                         const float* raw_prefix, float* output, size_t length,
                         size_t heads, size_t kv_heads, size_t dim, size_t key_blocks,
                         ThreadPool& pool, AttentionWorkspace& workspace, bool scalar);
}
} // namespace decode
