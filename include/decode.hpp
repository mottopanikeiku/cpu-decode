#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <sched.h>

namespace decode {
using Json = nlohmann::json;
enum class DType { bf16, f32, f16, i8 };
enum class Kernel { scalar, simd256, simd512, simd512x4, vnni };
enum class CacheType { f16, f32 };
Kernel parse_kernel(const std::string& name);
std::string kernel_name(Kernel kernel);
float bf16_float(uint16_t value);
float half_float(uint16_t value);
uint16_t float_half(float value);
Json cpu_topology();
class CpuBinding {
public:
    explicit CpuBinding(int cpu);
    ~CpuBinding();
    CpuBinding(const CpuBinding&) = delete;
    CpuBinding& operator=(const CpuBinding&) = delete;
private:
    cpu_set_t original;
    int previous;
    bool changed = false;
};
class ThreadPool {
public:
    using Function = void (*)(void*, size_t) noexcept;
    ThreadPool(int threads, const std::vector<int>& cpus = {}, bool persistent = true, bool strict_affinity = true);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    void run(size_t tasks, Function function, void* context);
    const std::vector<int>& cpus() const;
    int size() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
struct Matrix {
    const void* data = nullptr;
    const void* scales = nullptr;
    size_t rows = 0, cols = 0;
    DType dtype = DType::bf16;
    size_t group_size = 0;
    DType scale_dtype = DType::f32;
};
float matrix_scale(const Matrix& matrix, size_t row, size_t column);
uint64_t matrix_scale_bytes(const Matrix& matrix);
struct Activation {
    std::vector<uint8_t> bytes;
    std::vector<float> scales;
    explicit Activation(size_t capacity);
};
struct Projection { const Matrix* matrix; float* output; const float* bias = nullptr; };
void matvec(const Matrix& matrix, const float* x, float* y, Kernel kernel, ThreadPool& pool);
void projections(const Projection* items, size_t count, const float* x, Kernel kernel,
                 ThreadPool& pool, Activation& activation, bool swiglu = false);
void quantize_row(const float* source, size_t n, int8_t* out, float& scale);
void rmsnorm(const float* x, const float* weight, float* out, size_t n, float epsilon);
void residual_rmsnorm(float* x, const float* residual, const float* weight, float* out, size_t n, float epsilon);
void rope(float* x, size_t heads, size_t head_dim, size_t position, float theta);
struct AttentionWorkspace {
    static constexpr size_t block_size = 64;
    size_t capacity, heads, dim, blocks;
    std::vector<float> scores, maxima, sums, weighted;
    AttentionWorkspace(size_t capacity, size_t heads, size_t dim);
};
void attention(const float* q, const void* keys, const void* values, CacheType type,
               float* out, size_t length, size_t heads, size_t kv_heads,
               size_t head_dim, ThreadPool& pool, AttentionWorkspace& workspace, bool scalar = false);
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
struct EngineOptions {
    bool cached_rope = true, scalar_attention = false, persistent_pool = true;
    bool strict_affinity = true;
    CacheType cache_type = CacheType::f16;
    std::vector<int> cpus;
};
class Engine {
public:
    Engine(const std::string& directory, Kernel kernel, int threads, size_t capacity, EngineOptions options = {});
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
void quantize_model(const std::string& source, const std::string& output, size_t group_size = 0, DType scale_dtype = DType::f32);
} // namespace decode
