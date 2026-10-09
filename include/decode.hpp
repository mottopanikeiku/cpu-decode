#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace decode {
using Json = nlohmann::json;

// Safetensors storage types accepted by the loader.
enum class DType { bf16, f16, f32, i8, u8 };
// Matrix encodings. q8/q4 use 32-weight blocks with one F16 scale per block.
enum class Format { bf16, q8, q4 };
enum class Kernel { scalar, avx512, neon };
enum class KvType { f32, f16 };
// hugepage copies the weights into anonymous memory advised for transparent huge pages.
enum class WeightMemory { mmap, hugepage };

Kernel parse_kernel(const std::string& name);  // "auto" selects the best supported kernel
std::string kernel_name(Kernel kernel);
Format parse_format(const std::string& name);
std::string format_name(Format format);
KvType parse_kv(const std::string& name);
std::string kv_name(KvType kv);
WeightMemory parse_weight_memory(const std::string& name);
std::string weight_memory_name(WeightMemory memory);

float bf16_float(uint16_t value);
float half_float(uint16_t value);
uint16_t float_half(float value);  // round to nearest even

template<class T> struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template<class U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(64))); }
    void deallocate(T* p, size_t) noexcept { ::operator delete(p, std::align_val_t(64)); }
    template<class U> bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
    template<class U> bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }
};
template<class T> using AlignedVector = std::vector<T, AlignedAllocator<T>>;

constexpr size_t block_size = 32;

struct Matrix {
    const void* data = nullptr;
    const uint16_t* scales = nullptr;  // F16 [rows, cols / 32] for q8/q4
    size_t rows = 0, cols = 0;
    Format format = Format::bf16;
    size_t weight_bytes() const;
    size_t scale_bytes() const;
};

// FP32 vector quantized to int8 in 32-element blocks, as consumed by q8/q4 matrices.
struct Activation {
    size_t n = 0;
    AlignedVector<int8_t> q;
    std::vector<float> d;        // one scale per block
    AlignedVector<int32_t> bias; // per 64-element pair: -128 * block sum in lanes 0 and 8, zero elsewhere
    void resize(size_t size);
};

// Block quantizers. q8: n % 32 == 0. q4: n % 64 == 0; byte j of each 64-weight
// group holds weight j (low nibble) and weight 32 + j (high nibble), offset by 8.
void quantize_q8(const float* source, size_t n, int8_t* out, uint16_t* scales);
void quantize_q4(const float* source, size_t n, uint8_t* out, uint16_t* scales);
void dequantize_row(const Matrix& matrix, size_t row, float* out);
// Quantizes blocks [first, last) of x into a.
void quantize_activation(const float* x, Activation& a, size_t first_block, size_t last_block);

// Rows [begin, end) of y = m * x. bf16 reads x; q8/q4 read a. accumulate adds into y.
void matvec_rows(const Matrix& m, const float* x, const Activation& a, float* y,
                 size_t begin, size_t end, Kernel kernel, bool accumulate);
// Whole product with its own parallel region (tests and tools).
void matvec(const Matrix& m, const float* x, float* y, Kernel kernel, int threads);

void rmsnorm(const float* x, const float* weight, float* out, size_t n, float epsilon);
// Split-half rotation with precomputed cos/sin of length dim / 2.
void rope(float* x, size_t heads, size_t dim, const float* cos, const float* sin);

// KV cache, per layer and KV head, for kv_rows(capacity) positions in the KV dtype:
// keys in tiles of kv_tile positions stored dimension-major ([tile][dim][kv_tile]),
// values row-major ([position][dim]). write_kv is the only writer of this layout.
constexpr size_t kv_tile = 16;
size_t kv_rows(size_t capacity);
void write_kv(void* keys, void* values, KvType kv, size_t capacity, size_t dim, size_t head,
              size_t position, const float* k, const float* v);

// Grouped-query flash decoding. A work item is one KV head and one time chunk
// (whole tiles), and scores every query head of its group against each K/V row once.
struct AttentionPlan {
    size_t length = 0, chunks = 0, chunk = 0, items = 0, group = 0, partial_stride = 0;
};
AttentionPlan plan_attention(size_t length, size_t heads, size_t kv_heads, size_t dim, int threads);
// Partial buffer size sufficient for every length at this thread count.
size_t attention_partial_floats(size_t heads, size_t kv_heads, size_t dim, int threads);
size_t attention_scratch_floats(size_t group, size_t dim);
// q is pre-scaled by 1/sqrt(dim). Writes item's partial (max, sum, output) per head.
void attention_item(const AttentionPlan& plan, size_t item, const float* q, const void* keys,
                    const void* values, KvType kv, size_t capacity, size_t dim, float* scratch,
                    float* partials, Kernel kernel);
void attention_merge(const AttentionPlan& plan, size_t head, const float* partials, size_t dim, float* out);
// Convenience: complete attention with its own parallel region.
void attention(const float* q, const void* keys, const void* values, KvType kv, float* out,
               size_t length, size_t capacity, size_t heads, size_t kv_heads, size_t dim,
               Kernel kernel, int threads);

struct Profile {
    Profile();
    std::map<std::string, double, std::less<>> seconds;
    uint64_t matrix_weight_bytes = 0, scale_bytes = 0, norm_bias_bytes = 0;
    uint64_t embedding_bytes = 0, kv_write_bytes = 0, kv_read_bytes = 0;
    uint64_t lm_head_weight_bytes = 0, lm_head_scale_bytes = 0;
    size_t steps = 0, head_steps = 0;
    Json json() const;
};

struct EngineOptions {
    Kernel kernel = Kernel::scalar;
    int threads = 1;
    size_t capacity = 0;
    KvType kv = KvType::f16;
    WeightMemory weights = WeightMemory::hugepage;
    bool fuse = true;  // one pass/row range for Q|K|V and gate|up
};

class Engine {
public:
    Engine(const std::string& directory, const EngineOptions& options);
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

void quantize_model(const std::string& source, const std::string& output, Format format, Format head_format);
} // namespace decode
