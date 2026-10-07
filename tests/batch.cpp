#include "decode.hpp"
#include "generate.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using namespace decode;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void fails(const std::function<void()>& fn) {
    bool caught = false;
    try { fn(); } catch (const std::runtime_error&) { caught = true; }
    require(caught, "expected rejected input");
}
void exact(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "logit shape mismatch");
    require(a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
            "batch arithmetic differs from steps");
}
uint16_t bf16(float f) { uint32_t bits; std::memcpy(&bits, &f, 4); return uint16_t(bits >> 16); }
std::vector<Kernel> kernels(bool integer) {
    std::vector<Kernel> result{Kernel::scalar};
    for (const auto* name : {"simd256", "simd512", "simd512x4", "vnni", "vnni16"}) {
        if (!integer && (std::string(name) == "vnni" || std::string(name) == "vnni16")) continue;
        try { result.push_back(parse_kernel(name)); }
        catch (const std::runtime_error&) { std::cout << "skip unavailable " << name << '\n'; }
    }
    return result;
}
// Two layers and GQA, so a token-major shortcut or incorrect head strides are visible.
void fixture(const fs::path& path, bool constant) {
    fs::create_directories(path);
    constexpr size_t hidden = 128, intermediate = 128, vocab = 11;
    Json config{{"model_type", "qwen2"}, {"tie_word_embeddings", true}, {"hidden_act", "silu"},
        {"hidden_size", hidden}, {"intermediate_size", intermediate}, {"num_hidden_layers", 2},
        {"num_attention_heads", 4}, {"num_key_value_heads", 2}, {"vocab_size", vocab},
        {"max_position_embeddings", 160}, {"rms_norm_eps", 1e-6}, {"rope_theta", 10000}};
    std::ofstream(path / "config.json") << config.dump();
    std::map<std::string, std::vector<uint16_t>> tensors;
    Json header; uint64_t offset = 0;
    auto add = [&](const std::string& name, std::vector<size_t> shape, bool norm = false) {
        size_t count = 1; for (size_t n : shape) count *= n;
        auto& data = tensors[name]; data.resize(count);
        for (size_t j = 0; j < count; ++j) {
            float value = norm ? 1.0f + (constant ? 0.0f : .03f * std::sin(float(j))) :
                constant ? 0.0f : .025f * std::sin(float(j + tensors.size() * 17));
            if (constant && name == "model.embed_tokens.weight") value = j / hidden == 0 ? 1.0f : .125f;
            data[j] = bf16(value);
        }
        header[name] = {{"dtype", "BF16"}, {"shape", shape}};
    };
    add("model.embed_tokens.weight", {vocab, hidden}); add("model.norm.weight", {hidden}, true);
    for (size_t layer = 0; layer < 2; ++layer) {
        std::string b = "model.layers." + std::to_string(layer) + ".";
        add(b + "input_layernorm.weight", {hidden}, true); add(b + "post_attention_layernorm.weight", {hidden}, true);
        for (const auto* name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
            bool kv = std::string(name) == "k_proj" || std::string(name) == "v_proj";
            size_t rows = kv ? hidden / 2 : hidden;
            add(b + "self_attn." + name + ".weight", {rows, hidden});
            if (std::string(name) != "o_proj") add(b + "self_attn." + name + ".bias", {rows});
        }
        add(b + "mlp.gate_proj.weight", {intermediate, hidden});
        add(b + "mlp.up_proj.weight", {intermediate, hidden});
        add(b + "mlp.down_proj.weight", {hidden, intermediate});
    }
    for (const auto& item : tensors) {
        uint64_t bytes = item.second.size() * 2;
        header[item.first]["data_offsets"] = {offset, offset + bytes}; offset += bytes;
    }
    std::string encoded = header.dump(); while (encoded.size() % 8) encoded += ' ';
    std::ofstream out(path / "model.safetensors", std::ios::binary);
    uint64_t size = encoded.size(); out.write(reinterpret_cast<const char*>(&size), 8);
    out.write(encoded.data(), encoded.size());
    for (const auto& item : tensors)
        out.write(reinterpret_cast<const char*>(item.second.data()), item.second.size() * 2);
}
std::vector<float> steps(Engine& engine, const std::vector<int>& tokens, size_t start) {
    std::vector<float> result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& row = engine.step(tokens[i], i >= start);
        if (i >= start) result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}
std::vector<int> tokens(size_t n, int salt = 0) {
    std::vector<int> result(n);
    for (size_t i = 0; i < n; ++i) result[i] = int((i * 7 + size_t(salt)) % 11);
    return result;
}
void forward_tests(const fs::path& model, bool integer) {
    for (Kernel kernel : kernels(integer))
    for (CacheType cache : {CacheType::f16, CacheType::f32, CacheType::i8, CacheType::i8_centered})
    for (bool scalar : {false, true}) {
        EngineOptions options; options.cache_type = cache; options.scalar_attention = scalar;
        options.cached_rope = !scalar; options.strict_affinity = false;
        Engine batched(model.string(), kernel, 2, 160, options), single(model.string(), kernel, 2, 160, options);
        for (size_t n : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(4), size_t(5), size_t(9)}) {
            batched.reset(); single.reset();
            exact(batched.batch(tokens(n)), steps(single, tokens(n), 0));
            require(batched.position() == n, "batch position");
            batched.reset(); single.reset();
            size_t skip = n / 2;
            exact(batched.batch(tokens(n), skip), steps(single, tokens(n), skip));
            batched.reset(); single.reset();
            require(batched.batch(tokens(n), n).empty(), "skip all heads");
            steps(single, tokens(n), n);
            exact(batched.batch({3}), steps(single, {3}, 0));
        }
        // Cross mean-freezing position 63 with a multi-tile batch, then roll back
        // below it and change the warmup prefix before freezing again.
        batched.reset(); single.reset();
        auto prefix = tokens(61);
        batched.batch(prefix, prefix.size()); steps(single, prefix, prefix.size());
        auto draft = tokens(9, 2);
        exact(batched.batch(draft), steps(single, draft, 0));
        batched.rewind(62); single.rewind(62);
        exact(batched.batch(tokens(11, 4)), steps(single, tokens(11, 4), 0));
        batched.rewind(66); single.rewind(66);
        exact(batched.batch(tokens(7, 8)), steps(single, tokens(7, 8), 0));
        batched.rewind(60); single.rewind(60);
        exact(batched.batch({8, 7}), steps(single, {8, 7}, 0));
        // Compare causal batch rows while crossing 64 from an empty cache too.
        batched.reset(); single.reset();
        exact(batched.batch(tokens(65)), steps(single, tokens(65), 0));
        size_t position = batched.position();
        fails([&] { batched.batch({1, 11}); });
        fails([&] { batched.batch({1, -1}); });
        fails([&] { batched.batch({1}, 2); });
        fails([&] { batched.batch(tokens(96)); });
        Profile profile;
        fails([&] { batched.batch({1}, 0, &profile); });
        if (cache == CacheType::i8 || cache == CacheType::i8_centered)
            fails([&] { batched.step(1, true, &profile); });
        fails([&] { batched.rewind(position + 1); });
        require(batched.position() == position, "invalid input changes position");
        exact(batched.batch({6}), steps(single, {6}, 0));
        batched.reset(); single.reset();
        auto all = tokens(160); batched.batch(all, all.size()); steps(single, all, all.size());
        fails([&] { batched.batch({1}); });
        require(batched.batch({}).empty() && batched.position() == 160, "empty batch at capacity");
    }
}
void projection_tests() {
    ThreadPool pool(2, {}, true, false);
    constexpr size_t rows = 5;
    for (size_t cols : {size_t(17), size_t(65), size_t(128)}) {
        std::vector<float> weights(rows * cols), scales(rows), input(4 * (cols + 3));
        std::vector<uint16_t> halves(weights.size()); std::vector<int8_t> bytes(weights.size());
        for (size_t j = 0; j < weights.size(); ++j) { weights[j] = .2f * std::sin(float(j)); halves[j] = bf16(weights[j]); }
        for (size_t j = 0; j < input.size(); ++j) input[j] = .7f * std::cos(float(j) * .19f);
        for (size_t r = 0; r < rows; ++r) quantize_row(weights.data() + r * cols, cols, bytes.data() + r * cols, scales[r]);
        for (DType dtype : {DType::f32, DType::bf16, DType::i8}) {
            Matrix m{dtype == DType::f32 ? static_cast<const void*>(weights.data()) : dtype == DType::bf16 ? static_cast<const void*>(halves.data()) : bytes.data(), scales.data(), rows, cols, dtype};
            for (Kernel kernel : kernels(dtype == DType::i8)) {
                if (kernel == Kernel::vnni && cols % 32) continue;
                std::vector<Activation> activations;
                for (size_t c = 0; c < 4; ++c) activations.emplace_back(cols, kernel);
                for (size_t count : {size_t(1), size_t(2), size_t(3), size_t(4)}) {
                    std::vector<float> actual(count * (rows + 2), -99), expected = actual;
                    BatchProjection item{&m, actual.data(), rows + 2};
                    batch_projections(&item, 1, input.data(), count, cols + 3, kernel, pool, activations.data());
                    for (size_t c = 0; c < count; ++c) matvec(m, input.data() + c * (cols + 3), expected.data() + c * (rows + 2), kernel, pool);
                    exact(actual, expected);
                }
            }
        }
    }
    constexpr size_t cols = 128;
    std::vector<int8_t> weights(rows * cols);
    std::vector<float> input(cols * 4), bias(rows, .3f);
    for (size_t j = 0; j < weights.size(); ++j) weights[j] = int8_t(int(j % 255) - 128);
    for (size_t j = 0; j < input.size(); ++j) input[j] = std::sin(float(j) * .17f);
    for (size_t group : {size_t(0), size_t(32), size_t(64), size_t(128)})
    for (DType scale_type : {DType::f16, DType::f32}) {
        std::vector<float> scales(rows * (group ? cols / group : 1), .003f);
        std::vector<uint16_t> halves(scales.size(), float_half(.003f));
        Matrix m{weights.data(), scale_type == DType::f16 ? static_cast<const void*>(halves.data()) : scales.data(), rows, cols, DType::i8, group, scale_type};
        for (Kernel kernel : kernels(true)) {
            std::vector<Activation> activations;
            for (size_t c = 0; c < 4; ++c) activations.emplace_back(cols, kernel);
            std::vector<float> actual(4 * rows), expected(actual.size());
            BatchProjection item{&m, actual.data(), rows, bias.data()};
            batch_projections(&item, 1, input.data(), 4, cols, kernel, pool, activations.data());
            for (size_t c = 0; c < 4; ++c) {
                Projection one{&m, expected.data() + c * rows, bias.data()};
                projections(&one, 1, input.data() + c * cols, kernel, pool, activations[c]);
            }
            exact(actual, expected);
            BatchProjection pair[]{{&m, actual.data(), rows}, {&m, nullptr, 0}};
            batch_projections(pair, 2, input.data(), 4, cols, kernel, pool, activations.data(), true);
            for (size_t c = 0; c < 4; ++c) {
                Projection one[]{{&m, expected.data() + c * rows}, {&m, nullptr}};
                projections(one, 2, input.data() + c * cols, kernel, pool, activations[c], true);
            }
            exact(actual, expected);
            fails([&] { batch_projections(&item, 1, input.data(), 5, cols, kernel, pool, activations.data()); });
            fails([&] { batch_projections(&item, 1, input.data(), 0, cols, kernel, pool, activations.data()); });
            fails([&] { batch_projections(&item, 1, input.data(), 4, cols - 1, kernel, pool, activations.data()); });
        }
    }
}
void wide_projection_tests() {
    const auto available = kernels(true);
    if (std::find(available.begin(), available.end(), Kernel::vnni16) == available.end()) return;
    ThreadPool pool(2, {}, true, false);
    constexpr size_t cols = 65, rows = 3;
    std::vector<int8_t> weights(rows * cols);
    for (size_t j = 0; j < weights.size(); ++j) weights[j] = int8_t(int(j % 255) - 128);
    std::vector<float> input(4 * cols);
    for (size_t c = 0; c < 4; ++c) for (size_t j = 0; j < cols; ++j)
        input[c * cols + j] = std::sin(float(j) * .3f) * (c == 0 ? 1e-30f : .7f);
    std::vector<Activation> activations;
    for (size_t c = 0; c < 4; ++c) activations.emplace_back(cols, Kernel::vnni16);
    for (float scale : {1e-30f, 1e20f}) {
        std::vector<float> scales(rows, scale), actual(rows * 4), expected(actual.size());
        Matrix matrix{weights.data(), scales.data(), rows, cols, DType::i8};
        BatchProjection item{&matrix, actual.data(), rows};
        batch_projections(&item, 1, input.data(), 4, cols, Kernel::vnni16, pool, activations.data());
        for (size_t c = 0; c < 4; ++c)
            matvec(matrix, input.data() + c * cols, expected.data() + c * rows, Kernel::vnni16, pool);
        exact(actual, expected);
    }
}
void generation_tests(const fs::path& model, bool integer, bool constant) {
    for (Kernel kernel : kernels(integer))
    for (CacheType cache : {CacheType::f16, CacheType::f32, CacheType::i8, CacheType::i8_centered}) {
        EngineOptions options; options.cache_type = cache; options.strict_affinity = false;
        Engine plain(model.string(), kernel, 1, 160, options), lookup(model.string(), kernel, 1, 160, options);
        for (const auto& prompt : {std::vector<int>(65, 0), std::vector<int>{0, 2, 0, 2, 0, 2}}) {
            auto expected = generate(plain, prompt, 24, {0, 4, 1});
            auto actual = generate(lookup, prompt, 24, {4, 4, 8});
            require(actual.tokens == expected.tokens, "lookup output differs from greedy");
            require(plain.position() == lookup.position(), "lookup final position differs");
            if (constant) {
                require(actual.verification_batches > 0, "fixture must verify lookup drafts");
                require(actual.accepted > 0, "fixture must accept lookup drafts");
                if (prompt.size() == 6) require(actual.drafted > actual.accepted, "fixture must reject lookup drafts");
            }
        }
    }
}
} // namespace
int main() {
    char name[] = "/tmp/cpu-decode-batch-XXXXXX"; char* directory = mkdtemp(name);
    if (!directory) { std::cerr << "cannot create fixture directory\n"; return 1; }
    fs::path root(directory);
    try {
        fixture(root / "bf16", false); fixture(root / "constant", true);
        quantize_model((root / "bf16").string(), (root / "i8").string());
        quantize_model((root / "bf16").string(), (root / "group32").string(), 32, DType::f16);
        quantize_model((root / "constant").string(), (root / "constant-i8").string());
        projection_tests(); wide_projection_tests();
        forward_tests(root / "bf16", false); forward_tests(root / "i8", true); forward_tests(root / "group32", true);
        generation_tests(root / "bf16", false, false); generation_tests(root / "i8", true, false);
        generation_tests(root / "constant", false, true); generation_tests(root / "constant-i8", true, true);
        fs::remove_all(root); std::cout << "batch and lookup synthetic tests passed\n"; return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root); std::cerr << error.what() << '\n'; return 1;
    }
}
