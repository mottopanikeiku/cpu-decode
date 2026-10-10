#include "decode.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unistd.h>

namespace {
using decode::Json;
namespace fs = std::filesystem;
void require(bool good, const std::string& what) { if (!good) throw std::runtime_error(what); }
void near(double a, double b, double tolerance, const std::string& what) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance)
        throw std::runtime_error(what + ": " + std::to_string(a) + " vs " + std::to_string(b));
}
void fails(const std::function<void()>& operation, const std::string& what, const std::string& expected = "") {
    bool failed = false;
    try { operation(); } catch (const std::exception& error) {
        failed = true;
        require(expected.empty() || std::string(error.what()).find(expected) != std::string::npos, what + ": wrong exception: " + error.what());
    }
    require(failed, what);
}
uint16_t to_bf16(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    return uint16_t(bits >> 16);
}
std::vector<decode::Kernel> kernels() {
    std::vector<decode::Kernel> out{decode::Kernel::scalar};
    for (const auto& name : {"avx512", "neon"}) {
        try { out.push_back(decode::parse_kernel(name)); }
        catch (const std::exception&) { std::cout << "skip unavailable " << name << '\n'; }
    }
    return out;
}
float half_round(float x) { return decode::half_float(decode::float_half(x)); }

void conversion_tests() {
    for (uint32_t h = 0; h < 65536; ++h) {
        float f = decode::half_float(uint16_t(h));
        if (std::isnan(f)) continue;
        require(decode::float_half(f) == h, "half round trip " + std::to_string(h));
    }
    near(decode::half_float(0x3c00), 1.0, 0, "half one");
    near(decode::half_float(0x0001), std::ldexp(1.0, -24), 0, "smallest subnormal half");
    near(half_round(1.0f + std::ldexp(1.0f, -11)), 1.0, 0, "tie rounds to even (down)");
    near(half_round(1.0f + 3 * std::ldexp(1.0f, -11)), 1.0 + std::ldexp(1.0, -9), 0, "tie rounds to even (up)");
    require(decode::float_half(65520.0f) == 0x7c00 && decode::float_half(-1e9f) == 0xfc00, "half overflow to infinity");
    require(decode::float_half(65519.0f) == 0x7bff, "largest finite half");
}

void quantizer_tests() {
    constexpr size_t n = 128;
    std::vector<float> x(n);
    for (size_t j = 0; j < n; ++j) x[j] = std::sin(float(j) * 0.7f) * (j < 32 ? 0.0f : 0.4f);
    std::vector<int8_t> q8(n); std::vector<uint8_t> q4(n / 2); std::vector<uint16_t> s8(n / 32), s4(n / 32);
    decode::quantize_q8(x.data(), n, q8.data(), s8.data());
    decode::quantize_q4(x.data(), n, q4.data(), s4.data());
    decode::Matrix m8{q8.data(), s8.data(), 1, n, decode::Format::q8}, m4{q4.data(), s4.data(), 1, n, decode::Format::q4};
    std::vector<float> d8(n), d4(n);
    decode::dequantize_row(m8, 0, d8.data());
    decode::dequantize_row(m4, 0, d4.data());
    for (size_t b = 0; b < n / 32; ++b) {
        float maximum = 0, extreme = 0;
        for (size_t j = 0; j < 32; ++j) {
            maximum = std::max(maximum, std::abs(x[b * 32 + j]));
            if (std::abs(x[b * 32 + j]) > std::abs(extreme)) extreme = x[b * 32 + j];
        }
        near(decode::half_float(s8[b]), half_round(maximum / 127), 0, "q8 block scale");
        near(decode::half_float(s4[b]), half_round(extreme / -8), 0, "q4 block scale");
        float d8b = decode::half_float(s8[b]), d4b = std::abs(decode::half_float(s4[b]));
        for (size_t j = 0; j < 32; ++j) {
            size_t i = b * 32 + j;
            near(d8[i], x[i], d8b * 0.5 + 1e-3 * d8b + 1e-9, "q8 error bound");
            // Levels span [-8, 7] scales: the side opposite the extreme clamps at 7/8 of it.
            near(d4[i], x[i], (std::abs(x[i]) > 7 * d4b ? 1.0 : 0.5) * d4b + 1e-3 * d4b + 1e-9, "q4 error bound");
        }
    }
    // Packing: byte j of a 64-weight group holds weight j (low) and weight 32 + j (high).
    for (size_t j = 0; j < 32; ++j) {
        near(d4[64 + j], (int(q4[32 + j] & 15) - 8) * decode::half_float(s4[2]), 0, "q4 low nibble is first block");
        near(d4[96 + j], (int(q4[32 + j] >> 4) - 8) * decode::half_float(s4[3]), 0, "q4 high nibble is second block");
    }
    fails([&] { decode::quantize_q8(x.data(), 33, q8.data(), s8.data()); }, "q8 needs 32-multiple", "multiple of 32");
    fails([&] { decode::quantize_q4(x.data(), 96, q4.data(), s4.data()); }, "q4 needs 64-multiple", "multiple of 64");
    float bad[32]{}; bad[3] = NAN;
    fails([&] { decode::quantize_q8(bad, 32, q8.data(), s8.data()); }, "nonfinite quantization must fail", "nonfinite");
    // Activations: symmetric int8 blocks with scale max/127 and pair bias lanes.
    decode::Activation a; a.resize(n);
    decode::quantize_activation(x.data(), a, 0, n / 32);
    for (size_t b = 0; b < n / 32; ++b) {
        int sum = 0;
        for (size_t j = 0; j < 32; ++j) {
            sum += a.q[b * 32 + j];
            near(a.q[b * 32 + j] * a.d[b], x[b * 32 + j], a.d[b] * 0.5 + 1e-9, "activation error bound");
            const float inverse = a.d[b] ? 1.0f / a.d[b] : 0.0f;  // every path rounds x * (1/d) to nearest even
            require(a.q[b * 32 + j] == int8_t(std::nearbyint(x[b * 32 + j] * inverse)), "activation rounding is round-to-nearest-even");
        }
        require(a.bias[(b / 2) * 16 + (b % 2) * 8] == -128 * sum, "activation pair bias");
    }
}

void matvec_tests() {
    constexpr size_t rows = 9, cols = 320;  // 5 pairs: exercises the 4-pair loop and its tail
    std::vector<float> weights(rows * cols), input(cols);
    for (size_t j = 0; j < weights.size(); ++j) weights[j] = std::sin(float(j) * 0.7f) * 0.4f;
    std::fill(weights.begin(), weights.begin() + cols, 0.0f);
    for (size_t j = 0; j < cols; ++j) input[j] = std::cos(float(j) * 0.3f) * (j % 7 == 0 ? 3.0f : 1.0f);
    std::vector<uint16_t> bf16(weights.size());
    for (size_t j = 0; j < weights.size(); ++j) bf16[j] = to_bf16(weights[j]);
    std::vector<int8_t> q8(weights.size()); std::vector<uint8_t> q4(weights.size() / 2);
    std::vector<uint16_t> s8(rows * cols / 32), s4(rows * cols / 32);
    for (size_t r = 0; r < rows; ++r) {
        decode::quantize_q8(weights.data() + r * cols, cols, q8.data() + r * cols, s8.data() + r * cols / 32);
        decode::quantize_q4(weights.data() + r * cols, cols, q4.data() + r * cols / 2, s4.data() + r * cols / 32);
    }
    decode::Activation a; a.resize(cols);
    decode::quantize_activation(input.data(), a, 0, cols / 32);
    std::vector<decode::Matrix> matrices{{bf16.data(), nullptr, rows, cols, decode::Format::bf16},
                                         {q8.data(), s8.data(), rows, cols, decode::Format::q8},
                                         {q4.data(), s4.data(), rows, cols, decode::Format::q4}};
    for (const auto& m : matrices) {
        std::vector<double> expected(rows);
        std::vector<float> row(cols), result(rows);
        for (size_t r = 0; r < rows; ++r) {
            decode::dequantize_row(m, r, row.data());
            for (size_t j = 0; j < cols; ++j)
                expected[r] += double(row[j]) * (m.format == decode::Format::bf16 ? double(input[j]) : double(a.q[j]) * a.d[j / 32]);
        }
        for (auto kernel : kernels()) for (int threads : {1, 2, 4}) {
            decode::matvec(m, input.data(), result.data(), kernel, threads);
            for (size_t r = 0; r < rows; ++r)
                near(result[r], expected[r], 2e-5 * (1 + std::abs(expected[r])), "matvec " + decode::format_name(m.format) + "/" + decode::kernel_name(kernel));
            std::vector<float> accumulated(rows, 1.5f);
            decode::matvec_rows(m, input.data(), a, accumulated.data(), 2, 7, kernel, true);
            for (size_t r = 0; r < rows; ++r)
                near(accumulated[r], r >= 2 && r < 7 ? 1.5 + expected[r] : 1.5, 2e-5 * (2 + std::abs(expected[r])), "matvec_rows range/accumulate");
        }
    }
    fails([&] { decode::parse_kernel("fake"); }, "unknown kernel must fail");
    fails([&] { decode::Matrix m{q8.data(), s8.data(), rows, 96, decode::Format::q8}; float y[rows]; decode::matvec(m, input.data(), y, decode::Kernel::scalar, 1); }, "quantized matvec needs 64-multiple columns");
}

void attention_tests() {
    // Group sizes 3 and 7 (Qwen2.5-0.5B), a head size that only the scalar/NEON paths take (40),
    // lengths around tile boundaries, and finite stale rows past the length that must be ignored.
    for (auto [heads, kv_heads] : {std::pair<size_t, size_t>{6, 2}, {14, 2}})
    for (size_t dim : {32, 40, 64}) for (size_t length : {1, 15, 16, 17, 150}) for (auto kv : {decode::KvType::f32, decode::KvType::f16}) {
        const size_t group = heads / kv_heads, capacity = 155, rows = decode::kv_rows(capacity), width = kv == decode::KvType::f16 ? 2 : 4;
        std::vector<float> q(heads * dim), keys(kv_heads * capacity * dim), values(keys.size()), out(heads * dim);
        for (size_t j = 0; j < q.size(); ++j) q[j] = std::sin(float(j) * 1.1f) * 0.5f;
        for (size_t j = 0; j < keys.size(); ++j) { keys[j] = std::sin(float(j) * 0.37f) * 2; values[j] = std::cos(float(j) * 0.11f) * 3 - 1; }
        if (kv == decode::KvType::f16) for (size_t j = 0; j < keys.size(); ++j) { keys[j] = half_round(keys[j]); values[j] = half_round(values[j]); }
        std::vector<uint8_t> key_cache(kv_heads * rows * dim * width), value_cache(key_cache.size());
        for (size_t g = 0; g < kv_heads; ++g) for (size_t t = 0; t < capacity; ++t)
            decode::write_kv(key_cache.data(), value_cache.data(), kv, capacity, dim, g, t, &keys[(g * capacity + t) * dim], &values[(g * capacity + t) * dim]);
        for (auto kernel : kernels()) for (int threads : {1, 2, 3}) {
            decode::attention(q.data(), key_cache.data(), value_cache.data(), kv, out.data(), length, capacity, heads, kv_heads, dim, kernel, threads);
            for (size_t h = 0; h < heads; ++h) {
                size_t g = h / group;
                std::vector<double> p(length);
                double maximum = -1e300, sum = 0;
                for (size_t t = 0; t < length; ++t) {
                    double dot = 0;
                    for (size_t j = 0; j < dim; ++j) dot += double(q[h * dim + j]) * keys[(g * capacity + t) * dim + j];
                    p[t] = dot; maximum = std::max(maximum, dot);
                }
                for (double& x : p) { x = std::exp(x - maximum); sum += x; }
                for (size_t j = 0; j < dim; ++j) {
                    double expected = 0;
                    for (size_t t = 0; t < length; ++t) expected += p[t] / sum * values[(g * capacity + t) * dim + j];
                    near(out[h * dim + j], expected, 2e-5, "grouped flash attention " + decode::kernel_name(kernel) + " group " + std::to_string(group) +
                         " dim " + std::to_string(dim) + " length " + std::to_string(length) + " " + decode::kv_name(kv));
                }
            }
        }
    }
    fails([&] { float k[4]{}; uint8_t cache[16 * 4 * 4]; decode::write_kv(cache, cache, decode::KvType::f32, 16, 4, 0, 16, k, k); }, "KV write past capacity", "capacity");
    auto plan = decode::plan_attention(4096, 14, 2, 64, 6);
    require(plan.chunks == 6 && plan.items == 12 && plan.group == 7, "attention plan gives every thread work");
    require(decode::plan_attention(100, 14, 2, 64, 6).chunks == 2, "short contexts keep chunks of at least 64 rows");
    float dummy[64];
    fails([&] { decode::attention(dummy, dummy, dummy, decode::KvType::f32, dummy, 0, 4, 2, 1, 4, decode::Kernel::scalar, 1); }, "empty attention must fail");
    fails([&] { decode::plan_attention(4, 3, 2, 4, 1); }, "heads must divide by KV heads");
}

void elementwise_tests() {
    float x[]{1, -2, 3, -4}, w[]{1, 0.5f, -1, 2}, out[4];
    decode::rmsnorm(x, w, out, 4, 1e-6f);
    for (size_t i = 0; i < 4; ++i) near(out[i], x[i] * w[i] / std::sqrt(7.5 + 1e-6), 1e-6, "RMSNorm");
    float rotated[]{1, 2, 3, 4, -2, 3, -4, 5}, original[8], cosines[2], sines[2];
    std::copy(rotated, rotated + 8, original);
    for (size_t j = 0; j < 2; ++j) {
        double angle = 7 / std::pow(10000.0, double(2 * j) / 4);
        cosines[j] = float(std::cos(angle)); sines[j] = float(std::sin(angle));
    }
    decode::rope(rotated, 2, 4, cosines, sines);
    for (size_t h = 0; h < 2; ++h) for (size_t j = 0; j < 2; ++j) {
        near(rotated[4 * h + j], original[4 * h + j] * cosines[j] - original[4 * h + j + 2] * sines[j], 1e-6, "split-half RoPE first");
        near(rotated[4 * h + j + 2], original[4 * h + j] * sines[j] + original[4 * h + j + 2] * cosines[j], 1e-6, "split-half RoPE second");
    }
    // SiLU across the range the exp approximation clamps (|g| > 87) and ties to the scalar formula.
    std::vector<float> gate, up, silu;
    for (float g = -120.0f; g <= 120.0f; g += 0.37f) { gate.push_back(g); up.push_back(1.5f - g * 0.01f); }
    silu.resize(gate.size());
    for (auto kernel : kernels()) {
        decode::silu_multiply(gate.data(), up.data(), silu.data(), gate.size(), kernel);
        for (size_t j = 0; j < gate.size(); ++j) {
            const double expected = gate[j] / (1.0 + std::exp(-double(gate[j]))) * up[j];
            near(silu[j], expected, 2e-6 * std::abs(expected) + 1e-30, "SiLU " + decode::kernel_name(kernel));
        }
    }
}

constexpr size_t hidden = 64, heads = 2, dim = hidden / heads, intermediate = 128, vocab = 11;
struct Fixture {
    fs::path path;
    std::map<std::string, std::vector<float>> weights;
    std::map<std::string, std::vector<size_t>> shapes;
    explicit Fixture(const fs::path& directory, size_t width = hidden) : path(directory) {
        fs::create_directories(path);
        Json config{{"model_type", "qwen2"}, {"tie_word_embeddings", true}, {"hidden_act", "silu"},
            {"hidden_size", width}, {"intermediate_size", intermediate}, {"num_hidden_layers", 1}, {"num_attention_heads", heads},
            {"num_key_value_heads", 1}, {"vocab_size", vocab}, {"max_position_embeddings", 32}, {"rms_norm_eps", 1e-6}, {"rope_theta", 10000}};
        std::ofstream(path / "config.json") << config.dump();
        add("model.embed_tokens.weight", {vocab, width}); add("model.norm.weight", {width}, true);
        std::string b = "model.layers.0.";
        add(b + "input_layernorm.weight", {width}, true); add(b + "post_attention_layernorm.weight", {width}, true);
        for (const auto& name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
            size_t rows = std::string(name) == "k_proj" || std::string(name) == "v_proj" ? width / heads : width;
            add(b + "self_attn." + name + ".weight", {rows, width});
            if (std::string(name) != "o_proj") add(b + "self_attn." + name + ".bias", {rows});
        }
        add(b + "mlp.gate_proj.weight", {intermediate, width}); add(b + "mlp.up_proj.weight", {intermediate, width});
        add(b + "mlp.down_proj.weight", {width, intermediate});
        Json header; uint64_t offset = 0;
        for (const auto& item : weights) {
            uint64_t bytes = item.second.size() * 2;
            header[item.first] = {{"dtype", "BF16"}, {"shape", shapes[item.first]}, {"data_offsets", {offset, offset + bytes}}}; offset += bytes;
        }
        std::string encoded = header.dump(); while (encoded.size() % 8) encoded += ' ';
        std::ofstream output(path / "model.safetensors", std::ios::binary);
        uint64_t size = encoded.size(); output.write(reinterpret_cast<const char*>(&size), 8); output.write(encoded.data(), encoded.size());
        for (const auto& item : weights) for (float f : item.second) { uint16_t bf = to_bf16(f); output.write(reinterpret_cast<const char*>(&bf), 2); }
    }
    void add(const std::string& name, std::vector<size_t> shape, bool norm = false) {
        size_t n = 1; for (size_t d : shape) n *= d;
        std::vector<float> values(n);
        for (size_t j = 0; j < n; ++j) values[j] = decode::bf16_float(to_bf16(norm ? 1 + 0.05f * std::sin(float(j)) : 0.15f * std::sin(float(j + weights.size() * 17))));
        weights[name] = std::move(values); shapes[name] = std::move(shape);
    }
    // Replace matrices with their q8/q4 dequantized values.
    void quantize_reference(decode::Format format) {
        for (auto& item : weights) if (shapes[item.first].size() == 2) {
            size_t rows = shapes[item.first][0], cols = shapes[item.first][1];
            std::vector<int8_t> q8(cols); std::vector<uint8_t> q4(cols / 2); std::vector<uint16_t> scales(cols / 32);
            for (size_t r = 0; r < rows; ++r) {
                float* row = item.second.data() + r * cols;
                decode::Matrix m{nullptr, scales.data(), 1, cols, format};
                if (format == decode::Format::q8) { decode::quantize_q8(row, cols, q8.data(), scales.data()); m.data = q8.data(); }
                else { decode::quantize_q4(row, cols, q4.data(), scales.data()); m.data = q4.data(); }
                decode::dequantize_row(m, 0, row);
            }
        }
    }
};
using Vec = std::vector<double>;
// Double-precision forward pass. With `quantized`, matrix inputs are rounded through
// the engine's int8 activation blocks; with `half_kv`, cached K/V are rounded to F16.
struct Reference {
    Fixture& f;
    bool quantized, half_kv;
    std::vector<Vec> keys, values;
    Reference(Fixture& fixture, bool q, bool h) : f(fixture), quantized(q), half_kv(h) {}
    Vec multiply(const std::string& name, Vec x) {
        if (quantized) {
            std::vector<float> input(x.begin(), x.end());
            decode::Activation a; a.resize(input.size());
            decode::quantize_activation(input.data(), a, 0, input.size() / 32);
            for (size_t j = 0; j < x.size(); ++j) x[j] = double(a.q[j]) * a.d[j / 32];
        }
        const auto& w = f.weights.at(name); size_t rows = f.shapes.at(name)[0];
        Vec out(rows, 0); for (size_t r = 0; r < rows; ++r) for (size_t j = 0; j < x.size(); ++j) out[r] += w[r * x.size() + j] * x[j];
        return out;
    }
    Vec norm(const Vec& x, const std::string& name) {
        double sum = 0; for (double v : x) sum += v * v;
        Vec out(x.size()); for (size_t j = 0; j < x.size(); ++j) out[j] = x[j] / std::sqrt(sum / x.size() + 1e-6) * f.weights.at(name)[j];
        return out;
    }
    void rotate(Vec& x) {
        for (size_t h = 0; h < x.size() / dim; ++h) for (size_t j = 0; j < dim / 2; ++j) {
            double a = x[h * dim + j], b = x[h * dim + j + dim / 2];
            double angle = double(keys.size()) / std::pow(10000.0, double(2 * j) / dim);
            x[h * dim + j] = a * std::cos(angle) - b * std::sin(angle);
            x[h * dim + j + dim / 2] = b * std::cos(angle) + a * std::sin(angle);
        }
    }
    Vec step(int token) {
        Vec x(hidden); for (size_t j = 0; j < hidden; ++j) x[j] = f.weights.at("model.embed_tokens.weight")[token * hidden + j];
        std::string b = "model.layers.0.";
        Vec n = norm(x, b + "input_layernorm.weight");
        Vec q = multiply(b + "self_attn.q_proj.weight", n), k = multiply(b + "self_attn.k_proj.weight", n), v = multiply(b + "self_attn.v_proj.weight", n);
        for (size_t j = 0; j < hidden; ++j) q[j] += f.weights.at(b + "self_attn.q_proj.bias")[j];
        for (size_t j = 0; j < dim; ++j) { k[j] += f.weights.at(b + "self_attn.k_proj.bias")[j]; v[j] += f.weights.at(b + "self_attn.v_proj.bias")[j]; }
        rotate(q); rotate(k);
        if (half_kv) for (size_t j = 0; j < dim; ++j) { k[j] = half_round(float(k[j])); v[j] = half_round(float(v[j])); }
        keys.push_back(k); values.push_back(v);
        Vec a(hidden, 0);
        for (size_t h = 0; h < heads; ++h) {
            Vec scores(keys.size()); double sum = 0;
            for (size_t t = 0; t < keys.size(); ++t) {
                double dot = 0; for (size_t j = 0; j < dim; ++j) dot += q[h * dim + j] * keys[t][j];
                scores[t] = std::exp(dot / std::sqrt(double(dim))); sum += scores[t];
            }
            for (size_t t = 0; t < keys.size(); ++t) for (size_t j = 0; j < dim; ++j) a[h * dim + j] += scores[t] / sum * values[t][j];
        }
        Vec o = multiply(b + "self_attn.o_proj.weight", a); for (size_t j = 0; j < hidden; ++j) x[j] += o[j];
        n = norm(x, b + "post_attention_layernorm.weight");
        Vec gate = multiply(b + "mlp.gate_proj.weight", n), up = multiply(b + "mlp.up_proj.weight", n);
        for (size_t j = 0; j < gate.size(); ++j) gate[j] = gate[j] / (1 + std::exp(-gate[j])) * up[j];
        o = multiply(b + "mlp.down_proj.weight", gate); for (size_t j = 0; j < hidden; ++j) x[j] += o[j];
        return multiply("model.embed_tokens.weight", norm(x, "model.norm.weight"));
    }
};
decode::EngineOptions options(decode::Kernel kernel, int threads, size_t capacity, decode::KvType kv = decode::KvType::f32, bool fuse = true,
                              decode::WeightMemory memory = decode::WeightMemory::hugepage) {
    decode::EngineOptions o;
    o.kernel = kernel; o.threads = threads; o.capacity = capacity; o.kv = kv; o.fuse = fuse; o.weights = memory;
    return o;
}
void engine_tests(const fs::path& root) {
    Fixture fixture(root / "bf16");
    // The quantized reference rounds activations through the same int8 blocks, so it
    // agrees to ~1e-7 except when FP32-vs-double inputs land on opposite sides of a
    // rounding boundary; one flipped level moves a logit by ~5e-3 in this fixture.
    auto check = [&](const fs::path& model, bool quantized, double tolerance) {
        for (auto kv : {decode::KvType::f32, decode::KvType::f16}) for (auto kernel : kernels()) for (int threads : {1, 3}) {
            const bool fuse = threads == 1, half = kv == decode::KvType::f16;
            const auto memory = threads == 1 ? decode::WeightMemory::mmap : decode::WeightMemory::hugepage;
            const std::string label = model.filename().string() + "/" + decode::kernel_name(kernel) + "/" + decode::kv_name(kv) + "/t" + std::to_string(threads);
            decode::Engine engine(model.string(), options(kernel, threads, 8, kv, fuse, memory));
            Reference reference(fixture, quantized, half); decode::Profile profile;
            for (int token : {1, 4, 2}) {
                Vec expected = reference.step(token); const auto& actual = engine.step(token, true, &profile);
                for (size_t j = 0; j < expected.size(); ++j) near(actual[j], expected[j], tolerance, "complete tiny Qwen forward " + label);
            }
            const size_t width = half ? 2 : 4;
            require(profile.steps == 3 && profile.head_steps == 3, "profile step count");
            require(profile.kv_write_bytes == 3 * 2 * dim * width, "KV write bytes");
            require(profile.kv_read_bytes == (1 + 2 + 3) * 2 * dim * width, "KV read bytes: each row once per KV head");
            engine.rewind(2);
            Reference replay(fixture, quantized, half); replay.step(1); replay.step(4); Vec expected = replay.step(5);
            const auto& actual = engine.step(5, true);
            for (size_t j = 0; j < expected.size(); ++j) near(actual[j], expected[j], tolerance, "KV rewind and overwrite " + label);
            fails([&] { engine.step(int(vocab), true); }, "invalid token");
            fails([&] { engine.rewind(4); }, "invalid forward rewind");
            engine.reset(); Reference fresh(fixture, quantized, half); expected = fresh.step(2);
            const auto& reset = engine.step(2, true);
            for (size_t j = 0; j < expected.size(); ++j) near(reset[j], expected[j], tolerance, "reset cache prefix " + label);
            engine.step(1, false); require(engine.position() == 2, "head-skipped prefill advances cache");
        }
    };
    check(root / "bf16", false, 1e-5);
    decode::quantize_model((root / "bf16").string(), (root / "q8").string(), decode::Format::q8, decode::Format::q8);
    decode::quantize_model((root / "bf16").string(), (root / "q4").string(), decode::Format::q4, decode::Format::q4);
    Fixture q8 = fixture; q8.quantize_reference(decode::Format::q8);
    Fixture q4 = fixture; q4.quantize_reference(decode::Format::q4);
    std::swap(fixture.weights, q8.weights); check(root / "q8", true, 1e-2); std::swap(fixture.weights, q8.weights);
    std::swap(fixture.weights, q4.weights); check(root / "q4", true, 1e-2); std::swap(fixture.weights, q4.weights);
    {
        decode::Engine engine((root / "q8").string(), options(decode::Kernel::scalar, 1, 4));
        auto meta = engine.metadata();
        require(meta["weight_format"] == "q8" && meta["head_format"] == "q8" && meta["kv_dtype"] == "f32", "quantized metadata");
    }
    decode::quantize_model((root / "bf16").string(), (root / "q4h8").string(), decode::Format::q4, decode::Format::q8);
    {
        decode::Engine engine((root / "q4h8").string(), options(decode::Kernel::scalar, 1, 4));
        require(engine.metadata()["weight_format"] == "q4" && engine.metadata()["head_format"] == "q8", "separate head format");
    }
    fails([&] { decode::quantize_model((root / "bf16").string(), (root / "q8").string(), decode::Format::q8, decode::Format::q8); }, "output overwrite rejected");
    Fixture odd(root / "odd", 96);
    fails([&] { decode::quantize_model((root / "odd").string(), (root / "odd-q8").string(), decode::Format::q8, decode::Format::q8); }, "unalignable matrix rejected", "multiple of 64");
    require(!fs::exists(root / "odd-q8"), "unsupported shape must not leave output");
    fails([&] { decode::Engine engine((root / "missing").string(), options(decode::Kernel::scalar, 1, 4)); }, "missing model rejected");
    fails([&] { decode::Engine engine((root / "bf16").string(), options(decode::Kernel::scalar, 0, 4)); }, "zero threads rejected");
    fails([&] { decode::Engine engine((root / "bf16").string(), options(decode::Kernel::scalar, 1, 33)); }, "oversize capacity rejected");
    decode::Engine short_engine((root / "bf16").string(), options(decode::Kernel::scalar, 1, 1)); short_engine.step(1, false);
    fails([&] { short_engine.step(1, true); }, "capacity overflow rejected");
    fs::create_directories(root / "bad"); fs::copy_file(root / "bf16/config.json", root / "bad/config.json");
    std::ofstream(root / "bad/model.safetensors", std::ios::binary) << "bad";
    fails([&] { decode::Engine engine((root / "bad").string(), options(decode::Kernel::scalar, 1, 4)); }, "truncated file rejected");
    { std::ofstream bad(root / "bad/model.safetensors", std::ios::binary); uint64_t n = 1000000; bad.write(reinterpret_cast<const char*>(&n), 8); }
    fails([&] { decode::Engine engine((root / "bad").string(), options(decode::Kernel::scalar, 1, 4)); }, "oversize header rejected");
    auto malformed = [&](const Json& header, const std::string& message, const std::string& expected = "", size_t payload_bytes = 8) {
        std::string encoded = header.dump(); while (encoded.size() % 8) encoded += ' ';
        { std::ofstream bad(root / "bad/model.safetensors", std::ios::binary);
          uint64_t n = encoded.size(); bad.write(reinterpret_cast<const char*>(&n), 8);
          bad.write(encoded.data(), encoded.size()); std::string payload(payload_bytes, '\0'); bad.write(payload.data(), std::streamsize(payload.size())); }
        fails([&] { decode::Engine engine((root / "bad").string(), options(decode::Kernel::scalar, 1, 4)); }, message, expected);
    };
    malformed({{"x", {{"dtype", "F64"}, {"shape", {1}}, {"data_offsets", {0, 8}}}}}, "unsupported dtype rejected", "unsupported tensor dtype");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {-1}}, {"data_offsets", {0, 2}}}}}, "negative shape rejected");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {1}}, {"data_offsets", {0, 100}}}}}, "out-of-bounds offsets rejected");
    malformed({{"x", {{"dtype", "F32"}, {"shape", {1}}, {"data_offsets", {1, 5}}}}}, "unaligned offsets rejected");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {2}}, {"data_offsets", {0, 4}}}},
               {"y", {{"dtype", "BF16"}, {"shape", {2}}, {"data_offsets", {2, 6}}}}}, "overlapping tensors rejected");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {1}}, {"data_offsets", {2, 4}}}}}, "payload gap rejected", "gapped");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {1}}, {"data_offsets", {0, 2}}}}}, "unindexed trailing payload rejected", "unindexed");
    malformed({{"model.embed_tokens.weight", {{"dtype", "I8"}, {"shape", {vocab, hidden}}, {"data_offsets", {0, vocab * hidden}}}},
               {"model.embed_tokens.weight.scales", {{"dtype", "F32"}, {"shape", {vocab}}, {"data_offsets", {vocab * hidden, vocab * hidden + 4 * vocab}}}}},
              "v1 per-row int8 artifact rejected", "re-quantized", vocab * hidden + 4 * vocab);
    { Json config; std::ifstream(root / "bf16/config.json") >> config; config["num_attention_heads"] = 3; std::ofstream(root / "bad/config.json") << config.dump(); }
    fails([&] { decode::Engine engine((root / "bad").string(), options(decode::Kernel::scalar, 1, 4)); }, "invalid architecture rejected");
}
}
int main() {
    char name[] = "/tmp/cpu-decode-tests-XXXXXX";
    char* directory = mkdtemp(name);
    if (!directory) { std::cerr << "cannot create temporary directory\n"; return 1; }
    fs::path root(directory);
    try {
        conversion_tests(); quantizer_tests(); matvec_tests(); attention_tests(); elementwise_tests(); engine_tests(root); fs::remove_all(root);
        std::cout << "synthetic numerical and malformed-input tests passed\n"; return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root); std::cerr << error.what() << '\n'; return 1;
    }
}
