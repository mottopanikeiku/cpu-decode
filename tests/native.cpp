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
        require(expected.empty() || std::string(error.what()).find(expected) != std::string::npos, what + ": wrong exception");
    }
    require(failed, what);
}
uint16_t to_bf16(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    return uint16_t(bits >> 16);
}
std::vector<decode::Kernel> kernels() {
    std::vector<decode::Kernel> out{decode::Kernel::scalar};
    for (const auto& name : {"simd256", "simd512", "simd512x4"}) {
        try { out.push_back(decode::parse_kernel(name)); }
        catch (const std::exception&) { std::cout << "skip unavailable " << name << '\n'; }
    }
    return out;
}
void numeric_tests() {
    float x[]{1, -2, 3, -4}, w[]{1, 0.5f, -1, 2}, out[4];
    decode::rmsnorm(x, w, out, 4, 1e-6f);
    for (size_t i = 0; i < 4; ++i) near(out[i], x[i] * w[i] / std::sqrt(7.5 + 1e-6), 1e-6, "RMSNorm");
    float rotated[]{1, 2, 3, 4, -2, 3, -4, 5};
    float original[8]; std::copy(rotated, rotated + 8, original);
    decode::rope(rotated, 2, 4, 7, 10000);
    for (size_t h = 0; h < 2; ++h) for (size_t j = 0; j < 2; ++j) {
        double angle = 7 / std::pow(10000.0, double(2 * j) / 4);
        near(rotated[4 * h + j], original[4 * h + j] * std::cos(angle) - original[4 * h + j + 2] * std::sin(angle), 3e-6, "split-half RoPE first");
        near(rotated[4 * h + j + 2], original[4 * h + j] * std::sin(angle) + original[4 * h + j + 2] * std::cos(angle), 3e-6, "split-half RoPE second");
    }
    constexpr size_t rows = 9, cols = 147;
    std::vector<float> matrix(rows * cols), input(cols), baseline(rows), result(rows), scales(rows);
    std::vector<uint16_t> bf16(matrix.size()); std::vector<int8_t> quantized(matrix.size());
    for (size_t j = 0; j < matrix.size(); ++j) { matrix[j] = std::sin(float(j) * 0.7f) * 0.4f; bf16[j] = to_bf16(matrix[j]); }
    std::fill(matrix.begin(), matrix.begin() + cols, 0.0f);
    for (size_t j = 0; j < cols; ++j) input[j] = std::cos(float(j) * 0.3f);
    for (size_t r = 0; r < rows; ++r) {
        decode::quantize_row(matrix.data() + r * cols, cols, quantized.data() + r * cols, scales[r]);
        float maximum = 0; for (size_t j = 0; j < cols; ++j) maximum = std::max(maximum, std::abs(matrix[r * cols + j]));
        near(scales[r], maximum == 0 ? 1 : maximum / 127.0, 1e-8, "quantization scale");
        for (size_t j = 0; j < cols; ++j) {
            float expected = std::round(matrix[r * cols + j] / scales[r]);
            near(quantized[r * cols + j], expected, 0, "quantization rounding");
            near(quantized[r * cols + j] * scales[r], matrix[r * cols + j], scales[r] * 0.501 + 1e-7, "quantization error bound");
        }
    }
    for (auto type : {decode::DType::f32, decode::DType::bf16, decode::DType::i8}) {
        const void* data = type == decode::DType::f32 ? static_cast<const void*>(matrix.data()) : type == decode::DType::bf16 ? static_cast<const void*>(bf16.data()) : static_cast<const void*>(quantized.data());
        decode::Matrix m{data, scales.data(), rows, cols, type};
        for (size_t r = 0; r < rows; ++r) {
            double sum = 0;
            for (size_t j = 0; j < cols; ++j) {
                size_t i = r * cols + j;
                double weight = type == decode::DType::f32 ? matrix[i] : type == decode::DType::bf16 ? decode::bf16_float(bf16[i]) : quantized[i];
                sum += weight * input[j];
            }
            baseline[r] = float(type == decode::DType::i8 ? sum * scales[r] : sum);
        }
        for (auto kernel : kernels()) for (int threads : {1, 2}) {
            decode::matvec(m, input.data(), result.data(), kernel, threads);
            for (size_t r = 0; r < rows; ++r) near(result[r], baseline[r], 3e-6, "matvec tail and dtype");
        }
    }
    float bad[]{NAN}; int8_t byte; float scale;
    fails([&] { decode::quantize_row(bad, 1, &byte, scale); }, "nonfinite quantization must fail");
    fails([&] { decode::quantize_row(x, 0, &byte, scale); }, "empty quantization must fail");
    fails([&] { decode::parse_kernel("fake"); }, "unknown kernel must fail");
    constexpr size_t length = 3, heads = 4, kvheads = 2, dim = 4;
    float queries[heads * dim], keys[length * kvheads * dim], values[length * kvheads * dim];
    float attention_out[heads * dim], scores[length * heads];
    for (size_t j = 0; j < heads * dim; ++j) queries[j] = float(j % 5) - 2;
    for (size_t j = 0; j < length * kvheads * dim; ++j) { keys[j] = float(j % 7) * 0.1f; values[j] = float(j % 11) - 4; }
    for (int threads : {1, 2}) {
        decode::attention(queries, keys, values, attention_out, scores, length, heads, kvheads, dim, threads);
        for (size_t h = 0; h < heads; ++h) {
            double p[length], sum = 0;
            for (size_t t = 0; t < length; ++t) {
                double dot = 0;
                for (size_t j = 0; j < dim; ++j) dot += double(queries[h * dim + j]) * keys[t * kvheads * dim + (h / 2) * dim + j];
                p[t] = std::exp(dot / 2); sum += p[t];
            }
            for (size_t j = 0; j < dim; ++j) {
                double expected = 0;
                for (size_t t = 0; t < length; ++t) expected += p[t] / sum * values[t * kvheads * dim + (h / 2) * dim + j];
                near(attention_out[h * dim + j], expected, 2e-6, "GQA causal cached attention");
            }
        }
    }
    fails([&] { decode::attention(queries, keys, values, attention_out, scores, 0, heads, kvheads, dim, 1); }, "empty attention must fail");
}
struct Fixture {
    fs::path path;
    std::map<std::string, std::vector<float>> weights;
    std::map<std::string, std::vector<size_t>> shapes;
    explicit Fixture(const fs::path& directory, size_t hidden = 8, size_t heads = 2) : path(directory) {
        fs::create_directories(path);
        Json config{{"model_type", "qwen2"}, {"tie_word_embeddings", true}, {"hidden_act", "silu"},
            {"hidden_size", hidden}, {"intermediate_size", 12}, {"num_hidden_layers", 1}, {"num_attention_heads", heads},
            {"num_key_value_heads", 1}, {"vocab_size", 11}, {"max_position_embeddings", 32}, {"rms_norm_eps", 1e-6}, {"rope_theta", 10000}};
        std::ofstream(path / "config.json") << config.dump();
        add("model.embed_tokens.weight", {11, hidden}); add("model.norm.weight", {hidden}, true);
        std::string b = "model.layers.0.";
        add(b + "input_layernorm.weight", {hidden}, true); add(b + "post_attention_layernorm.weight", {hidden}, true);
        for (const auto& name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
            size_t rows = std::string(name) == "k_proj" || std::string(name) == "v_proj" ? hidden / heads : hidden;
            add(b + "self_attn." + name + ".weight", {rows, hidden});
            if (std::string(name) != "o_proj") add(b + "self_attn." + name + ".bias", {rows});
        }
        add(b + "mlp.gate_proj.weight", {12, hidden}); add(b + "mlp.up_proj.weight", {12, hidden}); add(b + "mlp.down_proj.weight", {hidden, 12});
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
    void quantize_reference() {
        for (auto& item : weights) if (shapes[item.first].size() == 2) {
            size_t cols = shapes[item.first][1];
            for (size_t r = 0; r < shapes[item.first][0]; ++r) {
                float maximum = 0;
                for (size_t j = 0; j < cols; ++j) maximum = std::max(maximum, std::abs(item.second[r * cols + j]));
                float scale = maximum == 0 ? 1 : maximum / 127;
                for (size_t j = 0; j < cols; ++j) item.second[r * cols + j] = std::round(item.second[r * cols + j] / scale) * scale;
            }
        }
    }
};
using Vec = std::vector<double>;
struct Reference {
    Fixture& f;
    std::vector<Vec> keys, values;
    explicit Reference(Fixture& fixture) : f(fixture) {}
    Vec multiply(const std::string& name, const Vec& x) {
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
        for (size_t h = 0; h < x.size() / 4; ++h) for (size_t j = 0; j < 2; ++j) {
            double a = x[h * 4 + j], b = x[h * 4 + j + 2];
            double angle = keys.size() / std::pow(10000.0, double(j) / 2);
            x[h * 4 + j] = a * std::cos(angle) - b * std::sin(angle);
            x[h * 4 + j + 2] = b * std::cos(angle) + a * std::sin(angle);
        }
    }
    Vec step(int token) {
        Vec x(8); for (size_t j = 0; j < 8; ++j) x[j] = f.weights.at("model.embed_tokens.weight")[token * 8 + j];
        std::string b = "model.layers.0.";
        Vec n = norm(x, b + "input_layernorm.weight");
        Vec q = multiply(b + "self_attn.q_proj.weight", n), k = multiply(b + "self_attn.k_proj.weight", n), v = multiply(b + "self_attn.v_proj.weight", n);
        for (size_t j = 0; j < 8; ++j) q[j] += f.weights.at(b + "self_attn.q_proj.bias")[j];
        for (size_t j = 0; j < 4; ++j) { k[j] += f.weights.at(b + "self_attn.k_proj.bias")[j]; v[j] += f.weights.at(b + "self_attn.v_proj.bias")[j]; }
        rotate(q); rotate(k); keys.push_back(k); values.push_back(v);
        Vec a(8, 0);
        for (size_t h = 0; h < 2; ++h) {
            Vec scores(keys.size()); double sum = 0;
            for (size_t t = 0; t < keys.size(); ++t) { double dot = 0; for (size_t j = 0; j < 4; ++j) dot += q[h * 4 + j] * keys[t][j]; scores[t] = std::exp(dot / 2); sum += scores[t]; }
            for (size_t t = 0; t < keys.size(); ++t) for (size_t j = 0; j < 4; ++j) a[h * 4 + j] += scores[t] / sum * values[t][j];
        }
        Vec o = multiply(b + "self_attn.o_proj.weight", a); for (size_t j = 0; j < 8; ++j) x[j] += o[j];
        n = norm(x, b + "post_attention_layernorm.weight");
        Vec gate = multiply(b + "mlp.gate_proj.weight", n), up = multiply(b + "mlp.up_proj.weight", n);
        for (size_t j = 0; j < gate.size(); ++j) gate[j] = gate[j] / (1 + std::exp(-gate[j])) * up[j];
        o = multiply(b + "mlp.down_proj.weight", gate); for (size_t j = 0; j < 8; ++j) x[j] += o[j];
        return multiply("model.embed_tokens.weight", norm(x, "model.norm.weight"));
    }
};
void engine_tests(const fs::path& root) {
    Fixture fixture(root / "bf16");
    auto check = [&](const fs::path& model) {
        for (bool cached : {true, false}) for (auto kernel : kernels()) {
            decode::Engine engine(model.string(), kernel, 1, 8, cached); Reference reference(fixture); decode::Profile profile;
            for (int token : {1, 4, 2}) {
                Vec expected = reference.step(token); const auto& actual = engine.step(token, true, &profile);
                for (size_t j = 0; j < expected.size(); ++j) near(actual[j], expected[j], 3e-6, "complete tiny Qwen forward");
            }
            require(profile.steps == 3 && profile.head_steps == 3, "profile step count");
            require(profile.kv_write_bytes == 3 * 2 * 4 * 4, "KV write bytes");
            require(profile.kv_read_min_bytes == (1 + 2 + 3) * 2 * 4 * 4, "minimum GQA bytes");
            require(profile.kv_read_logical_bytes == profile.kv_read_min_bytes * 2, "logical GQA bytes");
            engine.rewind(2);
            Reference replay(fixture); replay.step(1); replay.step(4); Vec expected = replay.step(5);
            const auto& actual = engine.step(5, true);
            for (size_t j = 0; j < expected.size(); ++j) near(actual[j], expected[j], 3e-6, "KV rewind and overwrite");
            fails([&] { engine.step(11, true); }, "invalid token");
            fails([&] { engine.rewind(4); }, "invalid forward rewind");
            engine.reset(); Reference fresh(fixture); expected = fresh.step(2);
            const auto& reset = engine.step(2, true); for (size_t j = 0; j < expected.size(); ++j) near(reset[j], expected[j], 3e-6, "reset cache prefix");
            engine.step(1, false); require(engine.position() == 2, "head-skipped prefill advances cache");
        }
    };
    check(root / "bf16");
    decode::quantize_model((root / "bf16").string(), (root / "int8").string());
    fixture.quantize_reference(); check(root / "int8");
    fails([&] { decode::quantize_model((root / "bf16").string(), (root / "int8").string()); }, "output overwrite rejected");
    Fixture odd(root / "odd", 6, 3);
    fails([&] { decode::quantize_model((root / "odd").string(), (root / "odd-int8").string()); }, "unalignable int8 matrix rejected", "multiple of four");
    require(!fs::exists(root / "odd-int8"), "unsupported shape must not leave output");
    fails([&] { decode::Engine engine((root / "missing").string(), decode::Kernel::scalar, 1, 4); }, "missing model rejected");
    fails([&] { decode::Engine engine((root / "bf16").string(), decode::Kernel::scalar, 0, 4); }, "zero threads rejected");
    fails([&] { decode::Engine engine((root / "bf16").string(), decode::Kernel::scalar, 1, 33); }, "oversize capacity rejected");
    decode::Engine short_engine((root / "bf16").string(), decode::Kernel::scalar, 1, 1); short_engine.step(1, false);
    fails([&] { short_engine.step(1, true); }, "capacity overflow rejected");
    fs::create_directories(root / "bad"); fs::copy_file(root / "bf16/config.json", root / "bad/config.json");
    std::ofstream(root / "bad/model.safetensors", std::ios::binary) << "bad";
    fails([&] { decode::Engine engine((root / "bad").string(), decode::Kernel::scalar, 1, 4); }, "truncated file rejected");
    { std::ofstream bad(root / "bad/model.safetensors", std::ios::binary); uint64_t n = 1000000; bad.write(reinterpret_cast<const char*>(&n), 8); }
    fails([&] { decode::Engine engine((root / "bad").string(), decode::Kernel::scalar, 1, 4); }, "oversize header rejected");
    auto malformed = [&](const Json& header, const std::string& message, const std::string& expected = "") {
        std::string encoded = header.dump(); while (encoded.size() % 8) encoded += ' ';
        { std::ofstream bad(root / "bad/model.safetensors", std::ios::binary);
          uint64_t n = encoded.size(); bad.write(reinterpret_cast<const char*>(&n), 8);
          bad.write(encoded.data(), encoded.size()); uint64_t payload = 0; bad.write(reinterpret_cast<const char*>(&payload), 8); }
        fails([&] { decode::Engine engine((root / "bad").string(), decode::Kernel::scalar, 1, 4); }, message, expected);
    };
    malformed({{"x", {{"dtype", "F16"}, {"shape", {1}}, {"data_offsets", {0, 2}}}}}, "unsupported dtype rejected");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {-1}}, {"data_offsets", {0, 2}}}}}, "negative shape rejected");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {1}}, {"data_offsets", {0, 100}}}}}, "out-of-bounds offsets rejected");
    malformed({{"x", {{"dtype", "F32"}, {"shape", {1}}, {"data_offsets", {1, 5}}}}}, "unaligned offsets rejected");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {2}}, {"data_offsets", {0, 4}}}},
               {"y", {{"dtype", "BF16"}, {"shape", {2}}, {"data_offsets", {2, 6}}}}}, "overlapping tensors rejected");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {1}}, {"data_offsets", {2, 4}}}}}, "payload gap rejected", "gapped");
    malformed({{"x", {{"dtype", "BF16"}, {"shape", {1}}, {"data_offsets", {0, 2}}}}}, "unindexed trailing payload rejected", "unindexed");
    { Json config; std::ifstream(root / "bf16/config.json") >> config; config["num_attention_heads"] = 3; std::ofstream(root / "bad/config.json") << config.dump(); }
    fails([&] { decode::Engine engine((root / "bad").string(), decode::Kernel::scalar, 1, 4); }, "invalid architecture rejected");
}
}
int main() {
    char name[] = "/tmp/cpu-decode-tests-XXXXXX";
    char* directory = mkdtemp(name);
    if (!directory) { std::cerr << "cannot create temporary directory\n"; return 1; }
    fs::path root(directory);
    try {
        numeric_tests(); engine_tests(root); fs::remove_all(root);
        std::cout << "synthetic numerical and malformed-input tests passed\n"; return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root); std::cerr << error.what() << '\n'; return 1;
    }
}
