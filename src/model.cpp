#include "decode.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>

namespace decode {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
size_t product(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) throw std::runtime_error("tensor size overflow");
    return a * b;
}
Json read_json(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    Json result;
    input >> result;
    return result;
}
struct Mapping {
    void* base = MAP_FAILED;
    size_t length = 0;
    explicit Mapping(const fs::path& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path.string());
        struct stat info{};
        if (fstat(fd, &info) || info.st_size < 8) { close(fd); throw std::runtime_error("invalid safetensors file"); }
        length = static_cast<size_t>(info.st_size);
        base = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (base == MAP_FAILED) throw std::runtime_error("mmap failed: " + path.string());
    }
    ~Mapping() { if (base != MAP_FAILED) munmap(base, length); }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
};
struct Tensor {
    const void* data;
    std::vector<size_t> shape;
    DType type;
    size_t bytes;
    size_t count() const { size_t n = 1; for (size_t d : shape) n = product(n, d); return n; }
    float at(size_t i) const {
        if (type == DType::bf16) return bf16_float(static_cast<const uint16_t*>(data)[i]);
        if (type == DType::f32) { float x; std::memcpy(&x, static_cast<const char*>(data) + 4 * i, 4); return x; }
        return float(static_cast<const int8_t*>(data)[i]);
    }
};
struct Store {
    std::vector<std::unique_ptr<Mapping>> mappings;
    std::map<std::string, Tensor> tensors;
    explicit Store(const fs::path& directory) {
        uint16_t endian = 1;
        if (*reinterpret_cast<uint8_t*>(&endian) != 1) throw std::runtime_error("little-endian host required");
        std::set<std::string> files;
        if (fs::exists(directory / "model.safetensors.index.json")) {
            Json index = read_json(directory / "model.safetensors.index.json");
            for (auto& item : index.at("weight_map").items()) {
                std::string name = item.value().get<std::string>();
                if (fs::path(name).filename() != name) throw std::runtime_error("unsafe shard filename");
                files.insert(name);
            }
        } else files.insert("model.safetensors");
        for (const auto& name : files) {
            auto mapping = std::make_unique<Mapping>(directory / name);
            const char* data = static_cast<const char*>(mapping->base);
            uint64_t header_length;
            std::memcpy(&header_length, data, 8);
            if (header_length > 64 * 1024 * 1024 || header_length > mapping->length - 8)
                throw std::runtime_error("invalid safetensors header length");
            size_t origin = 8 + size_t(header_length);
            Json header = Json::parse(data + 8, data + origin);
            std::vector<std::pair<size_t, size_t>> intervals;
            for (auto& item : header.items()) {
                if (item.key() == "__metadata__") continue;
                auto entry = item.value();
                std::string dtype = entry.at("dtype").get<std::string>();
                DType type;
                size_t width;
                if (dtype == "BF16") { type = DType::bf16; width = 2; }
                else if (dtype == "F32") { type = DType::f32; width = 4; }
                else if (dtype == "I8") { type = DType::i8; width = 1; }
                else throw std::runtime_error("unsupported tensor dtype: " + dtype);
                std::vector<size_t> shape;
                for (const auto& d : entry.at("shape")) {
                    if (!d.is_number_integer() || d.get<int64_t>() <= 0) throw std::runtime_error("invalid tensor shape");
                    shape.push_back(d.get<size_t>());
                }
                if (shape.empty() || shape.size() > 2) throw std::runtime_error("expected vector or matrix tensor");
                const auto& offsets = entry.at("data_offsets");
                if (offsets.size() != 2 || !offsets[0].is_number_unsigned() || !offsets[1].is_number_unsigned())
                    throw std::runtime_error("invalid tensor offsets");
                size_t begin = offsets[0].get<size_t>(), end = offsets[1].get<size_t>();
                Tensor tensor{nullptr, shape, type, 0};
                size_t bytes = product(tensor.count(), width);
                if (end < begin || end > mapping->length - origin || end - begin != bytes || (origin + begin) % width)
                    throw std::runtime_error("tensor bounds/alignment mismatch: " + item.key());
                tensor.data = data + origin + begin;
                tensor.bytes = bytes;
                if (!tensors.emplace(item.key(), tensor).second) throw std::runtime_error("duplicate tensor: " + item.key());
                intervals.emplace_back(begin, end);
            }
            std::sort(intervals.begin(), intervals.end());
            for (size_t j = 1; j < intervals.size(); ++j)
                if (intervals[j].first < intervals[j - 1].second) throw std::runtime_error("overlapping tensors");
            mappings.push_back(std::move(mapping));
        }
    }
    const Tensor& get(const std::string& name, std::vector<size_t> shape) const {
        auto found = tensors.find(name);
        if (found == tensors.end()) throw std::runtime_error("missing tensor: " + name);
        if (found->second.shape != shape) throw std::runtime_error("wrong shape: " + name);
        return found->second;
    }
};
struct Config {
    size_t hidden, intermediate, layers, heads, kv_heads, vocab, max_positions, dim;
    float epsilon, theta;
    explicit Config(const Json& json) {
        if (json.at("model_type") != "qwen2" || !json.value("tie_word_embeddings", false))
            throw std::runtime_error("requires tied-head Qwen2 architecture");
        if (json.contains("rope_scaling") && !json["rope_scaling"].is_null())
            throw std::runtime_error("scaled RoPE is not supported");
        if (json.value("hidden_act", std::string("silu")) != "silu") throw std::runtime_error("requires SiLU");
        auto positive = [&](const char* key) {
            int64_t n = json.at(key).get<int64_t>();
            if (n <= 0 || n > 1000000) throw std::runtime_error(std::string("invalid config: ") + key);
            return size_t(n);
        };
        hidden = positive("hidden_size"); intermediate = positive("intermediate_size");
        layers = positive("num_hidden_layers"); heads = positive("num_attention_heads");
        kv_heads = positive("num_key_value_heads"); vocab = positive("vocab_size");
        max_positions = positive("max_position_embeddings");
        epsilon = json.at("rms_norm_eps").get<float>(); theta = json.at("rope_theta").get<float>();
        if (hidden % heads || heads % kv_heads || (hidden / heads) % 2 || !std::isfinite(epsilon) ||
            epsilon <= 0 || !std::isfinite(theta) || theta <= 0) throw std::runtime_error("invalid Qwen dimensions or constants");
        dim = hidden / heads;
        if (json.contains("head_dim") && json["head_dim"].get<size_t>() != dim) throw std::runtime_error("incompatible head_dim");
    }
};
struct Layer {
    Matrix q, k, v, o, gate, up, down;
    std::vector<float> input_norm, post_norm, qbias, kbias, vbias;
    std::vector<float> keys, values;
};
template<class F> void timed(Profile* profile, const char* name, F&& fn) {
    if (!profile) { fn(); return; }
    auto start = Clock::now(); fn();
    auto found = profile->seconds.find(name);
    if (found == profile->seconds.end()) throw std::runtime_error("missing profile operation");
    found->second += std::chrono::duration<double>(Clock::now() - start).count();
}
}
struct Engine::Impl {
    Config config;
    Store store;
    Kernel kernel;
    int threads;
    bool cached_rope;
    size_t capacity, pos = 0;
    Matrix embedding;
    std::vector<Layer> layers;
    std::vector<float> final_norm, x, norm, q, k, v, attn, projected, gate, up, scores, logits;
    std::vector<float> inverse_frequency, rope_cos, rope_sin;
    uint64_t stored_bytes = 0, stored_scales = 0;
    std::string weight_dtype;
    Impl(const std::string& directory, Kernel ktype, int nthreads, size_t cap, bool cached)
        : config(read_json(fs::path(directory) / "config.json")), store(directory), kernel(ktype), threads(nthreads), cached_rope(cached), capacity(cap) {
        if (threads < 1 || threads > 1024 || !capacity || capacity > config.max_positions) throw std::runtime_error("invalid thread count or context capacity");
        parse_kernel(kernel_name(kernel));
        omp_set_dynamic(0);
        size_t cache = product(product(product(capacity, config.kv_heads * config.dim), config.layers), 2 * sizeof(float));
        if (cache > 768ull * 1024 * 1024) throw std::runtime_error("KV cache exceeds 768MiB safety limit");
        embedding = matrix("model.embed_tokens.weight", config.vocab, config.hidden);
        weight_dtype = embedding.dtype == DType::i8 ? "int8" : "bf16";
        if (embedding.dtype == DType::f32) throw std::runtime_error("full FP32 model is not supported");
        if (store.tensors.count("lm_head.weight")) throw std::runtime_error("tied model must not store duplicate lm_head.weight");
        for (size_t l = 0; l < config.layers; ++l) {
            std::string base = "model.layers." + std::to_string(l) + ".";
            Layer layer;
            layer.q = matrix(base + "self_attn.q_proj.weight", config.hidden, config.hidden);
            layer.k = matrix(base + "self_attn.k_proj.weight", config.kv_heads * config.dim, config.hidden);
            layer.v = matrix(base + "self_attn.v_proj.weight", config.kv_heads * config.dim, config.hidden);
            layer.o = matrix(base + "self_attn.o_proj.weight", config.hidden, config.hidden);
            layer.gate = matrix(base + "mlp.gate_proj.weight", config.intermediate, config.hidden);
            layer.up = matrix(base + "mlp.up_proj.weight", config.intermediate, config.hidden);
            layer.down = matrix(base + "mlp.down_proj.weight", config.hidden, config.intermediate);
            for (const Matrix* m : {&layer.q, &layer.k, &layer.v, &layer.o, &layer.gate, &layer.up, &layer.down})
                if (m->dtype != embedding.dtype) throw std::runtime_error("mixed matrix dtypes are not supported");
            layer.input_norm = vector(base + "input_layernorm.weight", config.hidden);
            layer.post_norm = vector(base + "post_attention_layernorm.weight", config.hidden);
            layer.qbias = vector(base + "self_attn.q_proj.bias", config.hidden);
            layer.kbias = vector(base + "self_attn.k_proj.bias", config.kv_heads * config.dim);
            layer.vbias = vector(base + "self_attn.v_proj.bias", config.kv_heads * config.dim);
            layer.keys.resize(product(capacity, config.kv_heads * config.dim));
            layer.values.resize(layer.keys.size());
            layers.push_back(std::move(layer));
        }
        final_norm = vector("model.norm.weight", config.hidden);
        x.resize(config.hidden); norm.resize(config.hidden); q.resize(config.hidden);
        k.resize(config.kv_heads * config.dim); v.resize(k.size()); attn.resize(config.hidden);
        projected.resize(config.hidden); gate.resize(config.intermediate); up.resize(config.intermediate);
        scores.resize(product(config.heads, capacity)); logits.resize(config.vocab);
        inverse_frequency.resize(config.dim / 2); rope_cos.resize(config.dim / 2); rope_sin.resize(config.dim / 2);
        for (size_t j = 0; j < inverse_frequency.size(); ++j)
            inverse_frequency[j] = 1.0f / std::pow(config.theta, float(2 * j) / float(config.dim));
        for (const auto& item : store.tensors) {
            if (item.first.size() >= 7 && item.first.substr(item.first.size() - 7) == ".scales") stored_scales += item.second.bytes;
            else stored_bytes += item.second.bytes;
        }
    }
    Matrix matrix(const std::string& name, size_t rows, size_t cols) {
        const auto& t = store.get(name, {rows, cols});
        Matrix result{t.data, nullptr, rows, cols, t.type};
        if (t.type == DType::i8) {
            const auto& s = store.get(name + ".scales", {rows});
            if (s.type != DType::f32) throw std::runtime_error("scales must be FP32");
            result.scales = static_cast<const float*>(s.data);
            for (size_t j = 0; j < rows; ++j) if (!std::isfinite(result.scales[j]) || result.scales[j] <= 0) throw std::runtime_error("invalid quantization scale");
        }
        return result;
    }
    std::vector<float> vector(const std::string& name, size_t n) {
        const auto& t = store.get(name, {n});
        if (t.type == DType::i8 || (embedding.dtype == DType::i8 && t.type != DType::f32)) throw std::runtime_error("norms/biases must be floating point (FP32 in quantized model)");
        std::vector<float> result(n);
        for (size_t j = 0; j < n; ++j) {
            result[j] = t.at(j);
            if (!std::isfinite(result[j])) throw std::runtime_error("nonfinite norm or bias");
        }
        return result;
    }
    void multiply(const Matrix& m, const float* input, float* out, Profile* p) {
        matvec(m, input, out, kernel, threads);
        if (p) {
            p->matrix_weight_bytes += m.rows * m.cols * (m.dtype == DType::bf16 ? 2 : m.dtype == DType::i8 ? 1 : 4);
            if (m.dtype == DType::i8) p->scale_bytes += m.rows * 4;
        }
    }
    const std::vector<float>& step(int token, bool head, Profile* p) {
        if (token < 0 || size_t(token) >= config.vocab) throw std::runtime_error("token ID outside vocabulary");
        if (pos >= capacity) throw std::runtime_error("context capacity exceeded");
        timed(p, "embedding", [&] {
            for (size_t j = 0; j < config.hidden; ++j) {
                size_t offset = size_t(token) * config.hidden + j;
                if (embedding.dtype == DType::bf16) x[j] = bf16_float(static_cast<const uint16_t*>(embedding.data)[offset]);
                else x[j] = float(static_cast<const int8_t*>(embedding.data)[offset]) * embedding.scales[token];
            }
        });
        if (p) { p->embedding_bytes += config.hidden * (embedding.dtype == DType::bf16 ? 2 : 1) + (embedding.dtype == DType::i8 ? 4 : 0); ++p->steps; }
        if (cached_rope) timed(p, "rope", [&] {
            for (size_t j = 0; j < inverse_frequency.size(); ++j) {
                float angle = float(pos) * inverse_frequency[j];
                rope_cos[j] = std::cos(angle); rope_sin[j] = std::sin(angle);
            }
        });
        for (Layer& l : layers) {
            timed(p, "rmsnorm", [&] { rmsnorm(x.data(), l.input_norm.data(), norm.data(), config.hidden, config.epsilon); });
            timed(p, "qkv", [&] {
                multiply(l.q, norm.data(), q.data(), p); multiply(l.k, norm.data(), k.data(), p); multiply(l.v, norm.data(), v.data(), p);
                for (size_t j = 0; j < q.size(); ++j) q[j] += l.qbias[j];
                for (size_t j = 0; j < k.size(); ++j) { k[j] += l.kbias[j]; v[j] += l.vbias[j]; }
            });
            timed(p, "rope", [&] {
                if (!cached_rope) {
                    rope(q.data(), config.heads, config.dim, pos, config.theta);
                    rope(k.data(), config.kv_heads, config.dim, pos, config.theta);
                    return;
                }
                auto rotate = [&](std::vector<float>& data) {
                    for (size_t h = 0; h < data.size() / config.dim; ++h)
                        for (size_t j = 0; j < config.dim / 2; ++j) {
                            size_t a = h * config.dim + j, b = a + config.dim / 2;
                            float first = data[a], second = data[b];
                            data[a] = first * rope_cos[j] - second * rope_sin[j];
                            data[b] = second * rope_cos[j] + first * rope_sin[j];
                        }
                };
                rotate(q); rotate(k);
            });
            timed(p, "kv_write", [&] {
                std::copy(k.begin(), k.end(), l.keys.begin() + pos * k.size());
                std::copy(v.begin(), v.end(), l.values.begin() + pos * v.size());
            });
            timed(p, "attention", [&] { attention(q.data(), l.keys.data(), l.values.data(), attn.data(), scores.data(), pos + 1, config.heads, config.kv_heads, config.dim, threads); });
            timed(p, "attention_output", [&] { multiply(l.o, attn.data(), projected.data(), p); });
            timed(p, "residual", [&] { for (size_t j = 0; j < x.size(); ++j) x[j] += projected[j]; });
            timed(p, "rmsnorm", [&] { rmsnorm(x.data(), l.post_norm.data(), norm.data(), config.hidden, config.epsilon); });
            timed(p, "mlp_gate_up", [&] { multiply(l.gate, norm.data(), gate.data(), p); multiply(l.up, norm.data(), up.data(), p); });
            timed(p, "silu", [&] { for (size_t j = 0; j < gate.size(); ++j) gate[j] = (gate[j] / (1.0f + std::exp(-gate[j]))) * up[j]; });
            timed(p, "mlp_down", [&] { multiply(l.down, gate.data(), projected.data(), p); });
            timed(p, "residual", [&] { for (size_t j = 0; j < x.size(); ++j) x[j] += projected[j]; });
            if (p) {
                p->norm_bias_bytes += (2 * config.hidden + q.size() + 2 * k.size()) * 4;
                p->kv_write_bytes += 2 * k.size() * 4;
                p->kv_read_min_bytes += 2 * (pos + 1) * k.size() * 4;
                p->kv_read_logical_bytes += 2 * (pos + 1) * q.size() * 4;
            }
        }
        if (head) {
            timed(p, "rmsnorm", [&] { rmsnorm(x.data(), final_norm.data(), norm.data(), config.hidden, config.epsilon); });
            timed(p, "lm_head", [&] { multiply(embedding, norm.data(), logits.data(), p); });
            if (p) {
                p->norm_bias_bytes += config.hidden * 4; ++p->head_steps;
                p->lm_head_weight_bytes += embedding.rows * embedding.cols * (embedding.dtype == DType::bf16 ? 2 : 1);
                if (embedding.dtype == DType::i8) p->lm_head_scale_bytes += embedding.rows * 4;
            }
        }
        ++pos;
        return logits;
    }
};
Engine::Engine(const std::string& dir, Kernel kernel, int threads, size_t capacity, bool cached_rope) : impl(std::make_unique<Impl>(dir, kernel, threads, capacity, cached_rope)) {}
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
void Engine::reset() { impl->pos = 0; }
void Engine::rewind(size_t position) {
    if (position > impl->pos) throw std::runtime_error("cannot rewind forward");
    impl->pos = position;
}
const std::vector<float>& Engine::step(int token, bool head, Profile* profile) { return impl->step(token, head, profile); }
size_t Engine::vocab_size() const { return impl->config.vocab; }
size_t Engine::position() const { return impl->pos; }
Json Engine::metadata() const {
    return {{"weight_dtype", impl->weight_dtype}, {"quantization", impl->weight_dtype == "int8" ? "symmetric per-row int8, float32 scales, float32 activations" : "none"},
            {"kernel", kernel_name(impl->kernel)}, {"kernel_scope", "matrix-vector products; FP32 norms, RoPE, attention and elementwise operations shared"},
            {"threads", impl->threads}, {"stored_weight_bytes", impl->stored_bytes},
            {"rope", impl->cached_rope ? "cached" : "direct"},
            {"thread_scheduling", "OpenMP static rows/heads, separate parallel region per operation"},
            {"stored_scale_bytes", impl->stored_scales}, {"kv_capacity", impl->capacity},
            {"auxiliary_fp32_weight_bytes", (impl->config.layers * (3 * impl->config.hidden + 2 * impl->config.kv_heads * impl->config.dim) + impl->config.hidden) * 4},
            {"kv_cache_bytes", impl->capacity * impl->config.kv_heads * impl->config.dim * impl->config.layers * 2 * 4},
            {"tied_head", true}, {"vocab_size", impl->config.vocab}};
}
void quantize_model(const std::string& source, const std::string& output) {
    fs::path src(source), dst(output);
    if (fs::weakly_canonical(src) == fs::weakly_canonical(dst)) throw std::runtime_error("quantization output must differ from input");
    { Engine validate(source, Kernel::scalar, 1, 1); }
    Store store(src);
    if (fs::exists(dst / "model.safetensors") || fs::exists(dst / "config.json")) throw std::runtime_error("quantization output already contains a model");
    for (const auto& item : store.tensors) if (item.second.type != DType::bf16) throw std::runtime_error("quantization input must be BF16");
    if (store.tensors.count("lm_head.weight")) throw std::runtime_error("duplicate tied head in input");
    Json header;
    uint64_t offset = 0;
    auto add = [&](const std::string& name, const std::string& dtype, const std::vector<size_t>& shape, uint64_t bytes) {
        header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + bytes}}}; offset += bytes;
    };
    // Preserve source iteration order in data emission, not JSON key order.
    for (const auto& item : store.tensors) {
        const Tensor& t = item.second;
        if (t.shape.size() == 2) {
            add(item.first, "I8", t.shape, t.count());
            // Float scales require alignment even for synthetic odd-size matrices.
            offset = (offset + 3) & ~uint64_t(3);
            add(item.first + ".scales", "F32", {t.shape[0]}, t.shape[0] * 4);
        } else add(item.first, "F32", t.shape, t.count() * 4);
    }
    header["__metadata__"] = {{"format", "pt"}, {"quantization", "symmetric-per-row-int8"}};
    std::string encoded = header.dump();
    while (encoded.size() % 8) encoded.push_back(' ');
    fs::create_directories(dst);
    fs::path temporary = dst / "model.safetensors.partial";
    try {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create quantized model");
        uint64_t header_size = encoded.size();
        out.write(reinterpret_cast<const char*>(&header_size), 8); out.write(encoded.data(), encoded.size());
        uint64_t written = 0;
        for (const auto& item : store.tensors) {
            const Tensor& t = item.second;
            if (t.shape.size() == 2) {
                std::vector<float> row(t.shape[1]), scales(t.shape[0]);
                std::vector<int8_t> quantized(t.shape[1]);
                for (size_t r = 0; r < t.shape[0]; ++r) {
                    for (size_t j = 0; j < row.size(); ++j) row[j] = t.at(r * row.size() + j);
                    quantize_row(row.data(), row.size(), quantized.data(), scales[r]);
                    out.write(reinterpret_cast<const char*>(quantized.data()), quantized.size());
                }
                written += t.count();
                while (written % 4) { out.put('\0'); ++written; }
                out.write(reinterpret_cast<const char*>(scales.data()), scales.size() * 4); written += scales.size() * 4;
            } else {
                for (size_t j = 0; j < t.count(); ++j) {
                    float value = t.at(j);
                    if (!std::isfinite(value)) throw std::runtime_error("nonfinite model vector");
                    out.write(reinterpret_cast<const char*>(&value), 4);
                }
                written += t.count() * 4;
            }
            if (!out) throw std::runtime_error("quantization write failed");
        }
        out.close();
        if (!out) throw std::runtime_error("quantization flush failed");
        fs::copy_file(src / "config.json", dst / "config.json");
        fs::rename(temporary, dst / "model.safetensors");
    } catch (...) {
        fs::remove(temporary);
        throw;
    }
}
} // namespace decode
