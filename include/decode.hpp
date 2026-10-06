#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace decode {
using Json = nlohmann::json;
enum class DType { bf16, f32, i8 };
enum class Kernel { scalar, simd256, simd512 };
Kernel parse_kernel(const std::string& name);
std::string kernel_name(Kernel kernel);
float bf16_float(uint16_t value);
struct Matrix {
    const void* data = nullptr;
    const float* scales = nullptr;
    size_t rows = 0, cols = 0;
    DType dtype = DType::bf16;
};
void matvec(const Matrix& matrix, const float* x, float* y, Kernel kernel, int threads);
void quantize_row(const float* source, size_t n, int8_t* out, float& scale);
void rmsnorm(const float* x, const float* weight, float* out, size_t n, float epsilon);
void rope(float* x, size_t heads, size_t head_dim, size_t position, float theta);
void attention(const float* q, const float* keys, const float* values, float* out,
               float* scores, size_t length, size_t heads, size_t kv_heads,
               size_t head_dim, int threads);
struct Profile {
    Profile();
    std::map<std::string, double, std::less<>> seconds;
    uint64_t matrix_weight_bytes = 0, scale_bytes = 0, norm_bias_bytes = 0;
    uint64_t embedding_bytes = 0, kv_write_bytes = 0, kv_read_min_bytes = 0;
    uint64_t kv_read_logical_bytes = 0;
    uint64_t lm_head_weight_bytes = 0, lm_head_scale_bytes = 0;
    size_t steps = 0, head_steps = 0;
    Json json() const;
};
class Engine {
public:
    Engine(const std::string& directory, Kernel kernel, int threads, size_t capacity, bool cached_rope = true);
    ~Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    void reset();
    const std::vector<float>& step(int token, bool head, Profile* profile = nullptr);
    size_t vocab_size() const;
    void rewind(size_t position);
    size_t position() const;
    Json metadata() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
void quantize_model(const std::string& source, const std::string& output);
} // namespace decode
