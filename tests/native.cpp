#include "decode.hpp"
#include <algorithm>
#include <atomic>
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
            decode::ThreadPool pool(threads);
            decode::matvec(m, input.data(), result.data(), kernel, pool);
            for (size_t r = 0; r < rows; ++r) near(result[r], baseline[r], 3e-6, "matvec tail and dtype");
        }
    }
    float bad[]{NAN}; int8_t byte; float scale;
    fails([&] { decode::quantize_row(bad, 1, &byte, scale); }, "nonfinite quantization must fail");
    fails([&] { decode::quantize_row(x, 0, &byte, scale); }, "empty quantization must fail");
    fails([&] { decode::parse_kernel("fake"); }, "unknown kernel must fail");
    constexpr size_t length = 3, heads = 4, kvheads = 2, dim = 4;
    float queries[heads * dim], keys[length * kvheads * dim], values[length * kvheads * dim];
    float attention_out[heads * dim];
    for (size_t j = 0; j < heads * dim; ++j) queries[j] = float(j % 5) - 2;
    for (size_t j = 0; j < length * kvheads * dim; ++j) { keys[j] = float(j % 7) * 0.1f; values[j] = float(j % 11) - 4; }
    for (int threads : {1, 2}) {
        decode::ThreadPool pool(threads);
        decode::AttentionWorkspace workspace(length, heads, dim);
        decode::attention(queries, keys, values, decode::CacheType::f32, attention_out, length, heads, kvheads, dim, pool, workspace);
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
    decode::ThreadPool pool(1);
    decode::AttentionWorkspace workspace(length, heads, dim);
    fails([&] { decode::attention(queries, keys, values, decode::CacheType::f32, attention_out, 0, heads, kvheads, dim, pool, workspace); }, "empty attention must fail");
}
struct Fixture {
    fs::path path;
    std::map<std::string, std::vector<float>> weights;
    std::map<std::string, std::vector<size_t>> shapes;
    explicit Fixture(const fs::path& directory, size_t hidden = 8, size_t heads = 2, size_t intermediate = 12, size_t capacity = 32) : path(directory) {
        fs::create_directories(path);
        Json config{{"model_type", "qwen2"}, {"tie_word_embeddings", true}, {"hidden_act", "silu"},
            {"hidden_size", hidden}, {"intermediate_size", intermediate}, {"num_hidden_layers", 1}, {"num_attention_heads", heads},
            {"num_key_value_heads", 1}, {"vocab_size", 11}, {"max_position_embeddings", capacity}, {"rms_norm_eps", 1e-6}, {"rope_theta", 10000}};
        std::ofstream(path / "config.json") << config.dump();
        add("model.embed_tokens.weight", {11, hidden}); add("model.norm.weight", {hidden}, true);
        std::string b = "model.layers.0.";
        add(b + "input_layernorm.weight", {hidden}, true); add(b + "post_attention_layernorm.weight", {hidden}, true);
        for (const auto& name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
            size_t rows = std::string(name) == "k_proj" || std::string(name) == "v_proj" ? hidden / heads : hidden;
            add(b + "self_attn." + name + ".weight", {rows, hidden});
            if (std::string(name) != "o_proj") add(b + "self_attn." + name + ".bias", {rows});
        }
        add(b + "mlp.gate_proj.weight", {intermediate, hidden}); add(b + "mlp.up_proj.weight", {intermediate, hidden}); add(b + "mlp.down_proj.weight", {hidden, intermediate});
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
            decode::EngineOptions options; options.cached_rope = cached; options.cache_type = decode::CacheType::f32;
            decode::Engine engine(model.string(), kernel, 1, 8, options); Reference reference(fixture); decode::Profile profile;
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
    malformed({{"x", {{"dtype", "F64"}, {"shape", {1}}, {"data_offsets", {0, 8}}}}}, "unsupported dtype rejected");
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
void affinity_tests(const fs::path& root) {
    cpu_set_t original;
    require(sched_getaffinity(0, sizeof(original), &original) == 0, "read test affinity");
    auto preferred = decode::cpu_topology()["preferred_cpu_ids"].get<std::vector<int>>();
    auto check_restored = [&] {
        cpu_set_t current; require(sched_getaffinity(0, sizeof(current), &current) == 0, "read current test affinity");
        require(CPU_EQUAL(&original, &current), "restore exact caller affinity");
    };
    {
        decode::Engine first((root / "bf16").string(), decode::Kernel::scalar, 1, 8);
        decode::Engine second((root / "bf16").string(), decode::Kernel::scalar, 6, 8);
        std::vector<int> expected; for (size_t j = 0; j < 6; ++j) expected.push_back(preferred[j % preferred.size()]);
        require(second.metadata()["cpu_set"] == expected, "overlapping engines select original allowed cores");
        check_restored(); first.step(1, true); check_restored(); second.step(1, true); check_restored();
        fails([&] { second.step(11, true); }, "step failure still restores affinity"); check_restored();
        first = decode::Engine((root / "bf16").string(), decode::Kernel::scalar, 6, 8);
        require(first.metadata()["cpu_set"] == expected, "replacement engine preserves full core selection");
        check_restored();
    }
    check_restored();
    {
        decode::CpuBinding outer(preferred[0]);
        decode::ThreadPool nested(2);
        require(nested.cpus()[0] == preferred[0] && nested.cpus()[1] == preferred[1 % preferred.size()], "nested pool uses outer allowed mask");
        if (preferred.size() > 1) {
            decode::CpuBinding inner(preferred[1]);
            require(decode::cpu_topology()["preferred_cpu_ids"] == preferred, "nested binding retains original topology");
        }
    }
    check_restored();
    for (bool persistent : {true, false}) {
        decode::ThreadPool pool(2, {}, persistent);
        size_t counts[3]{};
        pool.run(3, [](void* ptr, size_t index) noexcept { static_cast<size_t*>(ptr)[index] = index + 1; }, counts);
        require(counts[0] == 1 && counts[1] == 2 && counts[2] == 3, "pool runs every fixed task");
        check_restored();
        fails([&] { pool.run(1, nullptr, nullptr); }, "invalid callback"); check_restored();
    }
    for (bool persistent : {true, false}) {
        struct Probe { cpu_set_t expected; std::atomic<bool> mismatch{false}; size_t counts[128]{}; } probe;
        probe.expected = original;
        {
            decode::CpuBinding outer(preferred[0]);
            decode::ThreadPool pool(2, {}, persistent, false);
            require(pool.cpus() == decode::cpu_topology()["allowed_cpu_ids"].get<std::vector<int>>(), "unpinned pool reports full allowed mask");
            pool.run(128, [](void* ptr, size_t index) noexcept {
                auto& p = *static_cast<Probe*>(ptr); cpu_set_t current;
                if (sched_getaffinity(0, sizeof(current), &current) || !CPU_EQUAL(&current, &p.expected)) p.mismatch.store(true);
                p.counts[index] = index + 1;
            }, &probe);
            require(!probe.mismatch.load(), "all unpinned tasks use original full allowed mask");
            cpu_set_t current; require(sched_getaffinity(0, sizeof(current), &current) == 0 && CPU_COUNT(&current) == 1 && CPU_ISSET(preferred[0], &current), "unpinned run restores enclosing strict binding");
        }
        for (size_t index = 0; index < 128; ++index) require(probe.counts[index] == index + 1, "unpinned fixed tasks complete");
        check_restored();
    }
    {
        decode::EngineOptions options; options.strict_affinity = false;
        decode::Engine unpinned((root / "bf16").string(), decode::Kernel::scalar, 2, 8, options);
        decode::Engine strict((root / "bf16").string(), decode::Kernel::scalar, 2, 8);
        require(unpinned.metadata()["affinity"] == "unpinned" && unpinned.metadata()["cpu_set"] == decode::cpu_topology()["allowed_cpu_ids"], "unpinned engine reports actual allowed CPUs");
        for (int token : {1, 4, 2}) {
            const auto& expected = strict.step(token, true); const auto& actual = unpinned.step(token, true);
            require(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) == 0, "unpinned and strict logits bitwise equal");
            check_restored();
        }
        fails([&] { decode::ThreadPool invalid(1, {preferred[0]}, true, false); }, "unpinned explicit CPU set rejected");
    }
}
void group_kernel_tests(const fs::path& root) {
    constexpr size_t rows = 5, cols = 128;
    {
        decode::ThreadPool pool(1);
        float input16[16]{}, output[1], scales8[2]{1, 2};
        uint16_t half_weights[16]{}; int8_t weights8[16]{};
        decode::Matrix unsupported{half_weights, nullptr, 1, 16, decode::DType::f16};
        fails([&] { decode::matvec(unsupported, input16, output, decode::Kernel::scalar, pool); }, "FP16 matrix rejected before access");
        decode::Matrix too_small{weights8, scales8, 1, 16, decode::DType::i8, 8, decode::DType::f32};
        for (auto kernel : kernels()) fails([&] { decode::matvec(too_small, input16, output, kernel, pool); }, "unsupported eight-element group rejected consistently");
    }
    std::vector<float> weights(rows * cols), input(cols), actual(rows), expected(rows);
    std::vector<int8_t> bytes(weights.size());
    for (size_t j = 0; j < weights.size(); ++j) weights[j] = std::sin(float(j) * .37f) * .4f;
    for (size_t j = 0; j < input.size(); ++j) input[j] = std::cos(float(j) * .19f) * .7f;
    for (size_t group : {size_t(32), size_t(64), size_t(128)}) for (auto dtype : {decode::DType::f16, decode::DType::f32}) {
        if (group == 32 && dtype == decode::DType::f32) continue;
        size_t groups = cols / group;
        std::vector<float> scales(rows * groups); std::vector<uint16_t> halves(scales.size());
        for (size_t row = 0; row < rows; ++row) for (size_t g = 0; g < groups; ++g) {
            float& scale = scales[row * groups + g];
            decode::quantize_row(weights.data() + row * cols + g * group, group, bytes.data() + row * cols + g * group, scale);
            if (dtype == decode::DType::f16) {
                halves[row * groups + g] = decode::float_half(scale); scale = decode::half_float(halves[row * groups + g]);
                for (size_t j = 0; j < group; ++j) bytes[row * cols + g * group + j] = int8_t(std::clamp(std::round(weights[row * cols + g * group + j] / scale), -127.0f, 127.0f));
            }
        }
        const void* stored = dtype == decode::DType::f16 ? static_cast<const void*>(halves.data()) : scales.data();
        decode::Matrix matrix{bytes.data(), stored, rows, cols, decode::DType::i8, group, dtype};
        for (size_t row = 0; row < rows; ++row) {
            double total = 0;
            for (size_t j = 0; j < cols; ++j) total += double(bytes[row * cols + j]) * scales[row * groups + j / group] * input[j];
            expected[row] = float(total);
        }
        for (auto kernel : kernels()) {
            decode::ThreadPool pool(2);
            decode::matvec(matrix, input.data(), actual.data(), kernel, pool);
            for (size_t row = 0; row < rows; ++row) near(actual[row], expected[row], 5e-6, "group-scale SIMD against slow dequantized dot");
        }
        bool available = true; decode::Kernel vnni = decode::Kernel::scalar;
        try { vnni = decode::parse_kernel("vnni"); } catch (const std::exception&) { available = false; }
        if (available) {
            std::vector<int> activations(cols); std::vector<float> activation_scales(cols / 32);
            for (size_t begin = 0; begin < cols; begin += 32) {
                float maximum = 0; for (size_t j = begin; j < begin + 32; ++j) maximum = std::max(maximum, std::abs(input[j]));
                float scale = maximum == 0 ? 1 : maximum / 127; activation_scales[begin / 32] = scale;
                for (size_t j = begin; j < begin + 32; ++j) activations[j] = int(std::clamp(std::round(input[j] / scale), -127.0f, 127.0f));
            }
            for (size_t row = 0; row < rows; ++row) {
                float total = 0;
                for (size_t begin = 0; begin < cols; begin += 32) {
                    int dot = 0;
                    for (size_t j = begin; j < begin + 32; ++j) dot += int(bytes[row * cols + j]) * activations[j];
                    total += float(dot) * (activation_scales[begin / 32] * scales[row * groups + begin / group]);
                }
                expected[row] = total;
            }
            decode::ThreadPool pool(2);
            decode::matvec(matrix, input.data(), actual.data(), vnni, pool);
            require(std::memcmp(actual.data(), expected.data(), rows * 4) == 0, "VNNI signed correction equals slow integer dot");
        }
    }
    Fixture odd_scales(root / "odd-scales", 32, 1, 64, 8);
    decode::quantize_model((root / "odd-scales").string(), (root / "odd-scales-int8").string(), 32, decode::DType::f16);
    decode::Engine engine((root / "odd-scales-int8").string(), decode::Kernel::scalar, 1, 8);
    for (float value : engine.step(5, true)) require(std::isfinite(value), "FP16 odd-scale count preserves FP32 tensor alignment");
}
void optimized_tests(const fs::path& root) {
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        if ((bits & 0x7c00) == 0x7c00 && (bits & 1023)) continue;
        require(decode::float_half(decode::half_float(uint16_t(bits))) == bits, "F16 exact finite/infinity roundtrip");
    }
    require(decode::float_half(1.9998f) == 0x4000, "F16 mantissa carry");
    require(decode::float_half(1 + std::ldexp(1.0f, -11)) == 0x3c00, "F16 ties to even");
    require(decode::float_half(std::ldexp(1.0f, -25)) == 0, "F16 subnormal ties to even");
    constexpr size_t length = 129, heads = 14, kvheads = 2, dim = 64;
    std::vector<float> q(heads * dim), k(length * kvheads * dim), v(k.size()), reference(heads * dim), result(reference.size()), first;
    std::vector<uint16_t> kh(k.size()), vh(v.size());
    for (size_t j = 0; j < q.size(); ++j) q[j] = std::sin(float(j) * .13f) * 5;
    for (size_t j = 0; j < k.size(); ++j) {
        k[j] = std::cos(float(j) * .073f); v[j] = std::sin(float(j) * .17f);
        kh[j] = decode::float_half(k[j]); vh[j] = decode::float_half(v[j]);
    }
    for (auto type : {decode::CacheType::f32, decode::CacheType::f16}) {
        const void* keys = type == decode::CacheType::f32 ? static_cast<const void*>(k.data()) : kh.data();
        const void* values = type == decode::CacheType::f32 ? static_cast<const void*>(v.data()) : vh.data();
        {
            decode::ThreadPool pool(1); decode::AttentionWorkspace workspace(length, heads, dim);
            decode::attention(q.data(), keys, values, type, reference.data(), length, heads, kvheads, dim, pool, workspace, true);
        }
        for (int threads : {1, 2, 4, 6, 12}) {
            decode::ThreadPool pool(threads); decode::AttentionWorkspace workspace(length, heads, dim);
            decode::attention(q.data(), keys, values, type, result.data(), length, heads, kvheads, dim, pool, workspace);
            for (size_t j = 0; j < result.size(); ++j) near(result[j], reference[j], 3e-6, "blocked GQA vs scalar reference across block tail");
            if (threads == 1) first = result;
            else require(std::memcmp(first.data(), result.data(), result.size() * 4) == 0, "attention bitwise thread invariant");
        }
    }
    Fixture fixture(root / "wide", 448, 7, 64, 160);
    decode::quantize_model((root / "wide").string(), (root / "grouped").string(), 32, decode::DType::f16);
    fails([&] { decode::quantize_model((root / "wide").string(), (root / "over-budget").string(), 32, decode::DType::f32); }, "group budget rejected");
    for (const auto& path : {root / "wide", root / "grouped"}) for (auto type : {decode::CacheType::f32, decode::CacheType::f16}) {
        auto available = kernels();
        if (path.filename() == "grouped") {
            try { available.push_back(decode::parse_kernel("vnni")); } catch (const std::exception&) { std::cout << "skip unavailable vnni\\n"; }
        }
        for (auto kernel : available) {
            std::vector<float> baseline;
            for (int threads : {1, 2, 4, 6, 12}) {
                std::vector<float> actual;
                {
                    decode::EngineOptions options; options.cache_type = type;
                    decode::Engine engine(path.string(), kernel, threads, 129, options);
                    for (size_t position = 0; position < 129; ++position) {
                        const auto& row = engine.step(int((position * 7 + 3) % 11), true);
                        actual.insert(actual.end(), row.begin(), row.end());
                    }
                }
                if (threads == 1) baseline = actual;
                else require(baseline.size() == actual.size() && std::memcmp(baseline.data(), actual.data(), actual.size() * 4) == 0, "complete logits bitwise identical at1,2,4,6,12 threads");
            }
        }
    }
}
}
int main() {
    char name[] = "/tmp/cpu-decode-tests-XXXXXX";
    char* directory = mkdtemp(name);
    if (!directory) { std::cerr << "cannot create temporary directory\n"; return 1; }
    fs::path root(directory);
    try {
        numeric_tests(); engine_tests(root); affinity_tests(root); group_kernel_tests(root); optimized_tests(root); fs::remove_all(root);
        std::cout << "synthetic numerical and malformed-input tests passed\n"; return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root); std::cerr << error.what() << '\n'; return 1;
    }
}
