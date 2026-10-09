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
#include <sstream>
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
// Read-only weight file. hugepage copies it into 2 MiB-aligned anonymous memory
// advised for transparent huge pages; mmap maps the file and only advises.
struct Mapping {
    void* base = MAP_FAILED;
    size_t length = 0, mapped = 0;
    Mapping(const fs::path& path, WeightMemory memory) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path.string());
        struct stat info{};
        if (fstat(fd, &info) || info.st_size < 8) { close(fd); throw std::runtime_error("invalid safetensors file"); }
        length = static_cast<size_t>(info.st_size);
        if (memory == WeightMemory::mmap) {
            mapped = length;
            base = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (base == MAP_FAILED) throw std::runtime_error("mmap failed: " + path.string());
            madvise(base, length, MADV_HUGEPAGE);  // effective only where page-cache huge pages are supported
            return;
        }
        constexpr size_t huge = size_t(2) << 20;
        mapped = (length + huge - 1) / huge * huge;
        void* raw = mmap(nullptr, mapped + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) { close(fd); throw std::runtime_error("anonymous weight allocation failed"); }
        auto start = reinterpret_cast<uintptr_t>(raw), aligned = (start + huge - 1) / huge * huge;
        if (aligned > start) munmap(raw, aligned - start);
        if (start + huge > aligned) munmap(reinterpret_cast<void*>(aligned + mapped), start + huge - aligned);
        base = reinterpret_cast<void*>(aligned);
        madvise(base, mapped, MADV_HUGEPAGE);
        for (size_t done = 0; done < length;) {
            ssize_t n = pread(fd, static_cast<char*>(base) + done, length - done, off_t(done));
            if (n <= 0) { close(fd); munmap(base, mapped); base = MAP_FAILED; throw std::runtime_error("cannot read " + path.string()); }
            done += size_t(n);
        }
        close(fd);
        mprotect(base, mapped, PROT_READ);
    }
    ~Mapping() { if (base != MAP_FAILED) munmap(base, mapped); }
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
        if (type == DType::f16) return half_float(static_cast<const uint16_t*>(data)[i]);
        if (type == DType::f32) { float x; std::memcpy(&x, static_cast<const char*>(data) + 4 * i, 4); return x; }
        throw std::runtime_error("integer tensor read as float");
    }
};
struct Store {
    std::vector<std::unique_ptr<Mapping>> mappings;
    std::map<std::string, Tensor> tensors;
    Json metadata = Json::object();
    Store(const fs::path& directory, WeightMemory memory) {
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
            auto mapping = std::make_unique<Mapping>(directory / name, memory);
            const char* data = static_cast<const char*>(mapping->base);
            uint64_t header_length;
            std::memcpy(&header_length, data, 8);
            if (header_length > 64 * 1024 * 1024 || header_length > mapping->length - 8)
                throw std::runtime_error("invalid safetensors header length");
            size_t origin = 8 + size_t(header_length);
            Json header = Json::parse(data + 8, data + origin);
            std::vector<std::pair<size_t, size_t>> intervals;
            for (auto& item : header.items()) {
                if (item.key() == "__metadata__") { metadata = item.value(); continue; }
                auto entry = item.value();
                std::string dtype = entry.at("dtype").get<std::string>();
                DType type;
                size_t width;
                if (dtype == "BF16") { type = DType::bf16; width = 2; }
                else if (dtype == "F16") { type = DType::f16; width = 2; }
                else if (dtype == "F32") { type = DType::f32; width = 4; }
                else if (dtype == "I8") { type = DType::i8; width = 1; }
                else if (dtype == "U8") { type = DType::u8; width = 1; }
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
            size_t covered = 0;
            for (const auto& interval : intervals) {
                if (interval.first != covered) throw std::runtime_error("gapped or overlapping safetensors payload");
                covered = interval.second;
            }
            if (covered != mapping->length - origin) throw std::runtime_error("unindexed safetensors payload");
            mappings.push_back(std::move(mapping));
        }
    }
    const Tensor& get(const std::string& name) const {
        auto found = tensors.find(name);
        if (found == tensors.end()) throw std::runtime_error("missing tensor: " + name);
        return found->second;
    }
    const Tensor& get(const std::string& name, const std::vector<size_t>& shape) const {
        const Tensor& t = get(name);
        if (t.shape != shape) throw std::runtime_error("wrong shape: " + name);
        return t;
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
// Matrices whose outputs are concatenated and that read the same input.
struct Projection {
    std::vector<Matrix> parts;
    size_t rows = 0;
    void add(const Matrix& m) { parts.push_back(m); rows += m.rows; }
};
struct Layer {
    Projection qkv, o, gate_up, down;
    std::vector<float> input_norm, post_norm, qkv_bias;
    AlignedVector<uint8_t> keys, values;  // KV layout of write_kv, in the KV dtype
};
size_t huge_page_kib() {
    std::ifstream smaps("/proc/self/smaps_rollup");
    std::string line;
    size_t total = 0;
    while (std::getline(smaps, line)) {
        if (line.rfind("AnonHugePages:", 0) == 0 || line.rfind("FilePmdMapped:", 0) == 0) {
            std::istringstream fields(line.substr(line.find(':') + 1));
            size_t kib = 0;
            fields >> kib;
            total += kib;
        }
    }
    return total;
}
}
struct Engine::Impl {
    Config config;
    EngineOptions options;
    Store store;
    Format format = Format::bf16, head_format = Format::bf16;
    size_t pos = 0, kv_width, kv_dim;
    Matrix embedding;
    Projection head;
    std::vector<Layer> layers;
    std::vector<float> final_norm, x, qkv, attn, gate_up, hidden, logits, partials;
    std::vector<float> inverse_frequency, rope_cos, rope_sin;
    Activation act_attn, act_hidden;
    // Per-thread copies: every thread normalizes and quantizes the residual itself
    // (cheaper than a serial step plus a barrier) and prepares its own queries.
    struct Local {
        std::vector<float> norm, q, k, v;
        Activation act;
    };
    std::vector<Local> locals;
    AlignedVector<float> scratch;
    size_t scratch_floats;
    uint64_t stored_bytes = 0, stored_scales = 0;
    size_t huge_kib = 0;
    Impl(const std::string& directory, const EngineOptions& opts)
        : config(read_json(fs::path(directory) / "config.json")), options(opts), store(directory, opts.weights) {
        if (options.threads < 1 || options.threads > 1024 || !options.capacity || options.capacity > config.max_positions)
            throw std::runtime_error("invalid thread count or context capacity");
        parse_kernel(kernel_name(options.kernel));
        omp_set_dynamic(0);
        kv_width = options.kv == KvType::f16 ? 2 : 4;
        kv_dim = config.kv_heads * config.dim;
        size_t cache = product(product(product(kv_rows(options.capacity), kv_dim), config.layers), 2 * kv_width);
        if (cache > 768ull * 1024 * 1024) throw std::runtime_error("KV cache exceeds 768MiB safety limit");
        if (store.tensors.count("lm_head.weight")) throw std::runtime_error("tied model must not store duplicate lm_head.weight");
        embedding = matrix("model.embed_tokens.weight", config.vocab, config.hidden);
        head_format = embedding.format;
        head.add(embedding);
        for (size_t l = 0; l < config.layers; ++l) {
            std::string base = "model.layers." + std::to_string(l) + ".";
            Layer layer;
            for (const char* name : {"q_proj", "k_proj", "v_proj"}) {
                size_t rows = std::string(name) == "q_proj" ? config.hidden : kv_dim;
                layer.qkv.add(matrix(base + "self_attn." + name + ".weight", rows, config.hidden));
                auto bias = vector(base + "self_attn." + name + ".bias", rows);
                layer.qkv_bias.insert(layer.qkv_bias.end(), bias.begin(), bias.end());
            }
            layer.o.add(matrix(base + "self_attn.o_proj.weight", config.hidden, config.hidden));
            layer.gate_up.add(matrix(base + "mlp.gate_proj.weight", config.intermediate, config.hidden));
            layer.gate_up.add(matrix(base + "mlp.up_proj.weight", config.intermediate, config.hidden));
            layer.down.add(matrix(base + "mlp.down_proj.weight", config.hidden, config.intermediate));
            if (l == 0) format = layer.qkv.parts[0].format;
            for (const Projection* p : {&layer.qkv, &layer.o, &layer.gate_up, &layer.down})
                for (const Matrix& m : p->parts)
                    if (m.format != format) throw std::runtime_error("projection matrices must share one weight format");
            layer.input_norm = vector(base + "input_layernorm.weight", config.hidden);
            layer.post_norm = vector(base + "post_attention_layernorm.weight", config.hidden);
            layer.keys.assign(product(kv_rows(options.capacity), kv_dim) * kv_width, 0);
            layer.values.assign(layer.keys.size(), 0);
            layers.push_back(std::move(layer));
        }
        if ((format == Format::bf16) != (head_format == Format::bf16)) throw std::runtime_error("BF16 and quantized matrices cannot be mixed");
        if (format != Format::bf16 && config.dim % block_size)
            throw std::runtime_error("quantized models require head_dim to be a multiple of 32");
        final_norm = vector("model.norm.weight", config.hidden);
        x.resize(config.hidden); attn.resize(config.hidden);
        qkv.resize(config.hidden + 2 * kv_dim); gate_up.resize(2 * config.intermediate); hidden.resize(config.intermediate);
        logits.resize(config.vocab);
        act_attn.resize(config.hidden);
        locals.resize(size_t(options.threads));
        for (Local& local : locals) {
            local.norm.resize(config.hidden); local.q.resize(config.hidden);
            local.k.resize(config.dim); local.v.resize(config.dim);
            local.act.resize(config.hidden);
        }
        act_hidden.resize(config.intermediate);
        inverse_frequency.resize(config.dim / 2); rope_cos.resize(config.dim / 2); rope_sin.resize(config.dim / 2);
        for (size_t j = 0; j < inverse_frequency.size(); ++j)
            inverse_frequency[j] = 1.0f / std::pow(config.theta, float(2 * j) / float(config.dim));
        auto plan = plan_attention(options.capacity, config.heads, config.kv_heads, config.dim, options.threads);
        partials.resize(attention_partial_floats(config.heads, config.kv_heads, config.dim, options.threads));
        scratch_floats = (attention_scratch_floats(plan.group, config.dim) + 15) / 16 * 16;
        scratch.assign(size_t(options.threads) * scratch_floats, 0.0f);
        for (const auto& item : store.tensors) {
            if (item.first.size() >= 7 && item.first.substr(item.first.size() - 7) == ".scales") stored_scales += item.second.bytes;
            else stored_bytes += item.second.bytes;
        }
        huge_kib = huge_page_kib();
    }
    Matrix matrix(const std::string& name, size_t rows, size_t cols) {
        const Tensor& t = store.get(name);
        Matrix result{t.data, nullptr, rows, cols, Format::bf16};
        if (t.type == DType::bf16 && t.shape == std::vector<size_t>{rows, cols}) return result;
        if (t.type == DType::i8 && t.shape == std::vector<size_t>{rows, cols}) result.format = Format::q8;
        else if (t.type == DType::u8 && t.shape == std::vector<size_t>{rows, cols / 2}) result.format = Format::q4;
        else throw std::runtime_error("unsupported matrix dtype/shape: " + name);
        if (cols % (2 * block_size)) throw std::runtime_error("quantized matrices need a multiple of 64 columns: " + name);
        auto found = store.tensors.find(name + ".scales");
        if (found == store.tensors.end() || found->second.type != DType::f16 || found->second.shape != std::vector<size_t>{rows, cols / block_size})
            throw std::runtime_error("expected F16 block scales [rows, cols/32] for " + name + " (per-row int8 files from v1 must be re-quantized)");
        result.scales = static_cast<const uint16_t*>(found->second.data);
        for (size_t j = 0; j < rows * (cols / block_size); ++j)
            if (!std::isfinite(half_float(result.scales[j]))) throw std::runtime_error("invalid quantization scale");
        return result;
    }
    std::vector<float> vector(const std::string& name, size_t n) {
        const auto& t = store.get(name, {n});
        if (t.type == DType::i8 || t.type == DType::u8 || (embedding.format != Format::bf16 && t.type != DType::f32))
            throw std::runtime_error("norms/biases must be floating point (FP32 in quantized model)");
        std::vector<float> result(n);
        for (size_t j = 0; j < n; ++j) {
            result[j] = t.at(j);
            if (!std::isfinite(result[j])) throw std::runtime_error("nonfinite norm or bias");
        }
        return result;
    }
    void account(Profile* p, const Projection& projection) const {
        for (const Matrix& m : projection.parts) { p->matrix_weight_bytes += m.weight_bytes(); p->scale_bytes += m.scale_bytes(); }
    }
    const std::vector<float>& step(int token, bool want_head, Profile* p) {
        if (token < 0 || size_t(token) >= config.vocab) throw std::runtime_error("token ID outside vocabulary");
        if (pos >= options.capacity) throw std::runtime_error("context capacity exceeded");
        auto last = Clock::now();
        auto mark = [&](const char* name) {
            auto now = Clock::now();
            p->seconds.find(name)->second += std::chrono::duration<double>(now - last).count();
            last = now;
        };
        dequantize_row(embedding, size_t(token), x.data());
        for (size_t j = 0; j < inverse_frequency.size(); ++j) {
            float angle = float(pos) * inverse_frequency[j];
            rope_cos[j] = std::cos(angle); rope_sin[j] = std::sin(angle);
        }
        if (p) mark("embedding");
        const AttentionPlan plan = plan_attention(pos + 1, config.heads, config.kv_heads, config.dim, options.threads);
        const bool quantized = format != Format::bf16;
        const size_t dim = config.dim, hidden_blocks = config.hidden / block_size;
        const float query_scale = 1.0f / std::sqrt(float(dim));
        const Kernel kernel = options.kernel;
        // One parallel region per token: every phase is a static split followed by a
        // barrier. Seven barriers per layer: qkv, attention, merge, o, gate/up, silu, down.
        #pragma omp parallel num_threads(options.threads)
        {
            const size_t tid = size_t(omp_get_thread_num()), team = size_t(omp_get_num_threads());
            Local& my = locals[tid];
            auto end_phase = [&](const char* name) {
                #pragma omp barrier
                if (p && tid == 0) mark(name);
            };
            auto split = [&](size_t n, size_t& begin, size_t& end) { begin = n * tid / team; end = n * (tid + 1) / team; };
            // Fused: one row split over all parts. Unfused: a split and barrier per part.
            auto project = [&](const Projection& projection, const float* input, const Activation& a, float* y, bool accumulate, const char* name) {
                if (options.fuse) {
                    size_t begin, end, offset = 0;
                    split(projection.rows, begin, end);
                    for (const Matrix& m : projection.parts) {
                        size_t b = std::max(begin, offset), e = std::min(end, offset + m.rows);
                        if (b < e) matvec_rows(m, input, a, y + offset, b - offset, e - offset, kernel, accumulate);
                        offset += m.rows;
                    }
                    end_phase(name);
                    return;
                }
                size_t offset = 0;
                for (const Matrix& m : projection.parts) {
                    size_t begin, end;
                    split(m.rows, begin, end);
                    matvec_rows(m, input, a, y + offset, begin, end, kernel, accumulate);
                    offset += m.rows;
                    end_phase(name);
                }
            };
            // Redundant per thread: reads the completed residual, writes only thread-local buffers.
            auto normalize = [&](const std::vector<float>& weight, bool quantize) {
                rmsnorm(x.data(), weight.data(), my.norm.data(), config.hidden, config.epsilon);
                if (quantize) quantize_activation(my.norm.data(), my.act, 0, hidden_blocks);
            };
            const size_t group = plan.group, q_rows = config.hidden, k_rows = q_rows + kv_dim;
            for (Layer& l : layers) {
                normalize(l.input_norm, quantized);
                project(l.qkv, my.norm.data(), my.act, qkv.data(), false, "qkv");
                for (size_t item = tid; item < plan.items; item += team) {
                    const size_t kv_head = item / plan.chunks;
                    // This group's queries with bias, RoPE and the 1/sqrt(dim) scale.
                    for (size_t j = kv_head * group * dim; j < (kv_head + 1) * group * dim; ++j) my.q[j] = qkv[j] + l.qkv_bias[j];
                    rope(my.q.data() + kv_head * group * dim, group, dim, rope_cos.data(), rope_sin.data());
                    for (size_t j = kv_head * group * dim; j < (kv_head + 1) * group * dim; ++j) my.q[j] *= query_scale;
                    // The last chunk holds the new position; only it reads that row, so it writes it.
                    if (item % plan.chunks == plan.chunks - 1) {
                        for (size_t j = 0; j < dim; ++j) {
                            my.k[j] = qkv[q_rows + kv_head * dim + j] + l.qkv_bias[q_rows + kv_head * dim + j];
                            my.v[j] = qkv[k_rows + kv_head * dim + j] + l.qkv_bias[k_rows + kv_head * dim + j];
                        }
                        rope(my.k.data(), 1, dim, rope_cos.data(), rope_sin.data());
                        write_kv(l.keys.data(), l.values.data(), options.kv, options.capacity, dim, kv_head, pos, my.k.data(), my.v.data());
                    }
                    attention_item(plan, item, my.q.data(), l.keys.data(), l.values.data(), options.kv, options.capacity, dim,
                                   scratch.data() + tid * scratch_floats, partials.data(), kernel);
                }
                end_phase("attention");
                for (size_t h = tid; h < config.heads; h += team) {
                    attention_merge(plan, h, partials.data(), dim, attn.data() + h * dim);
                    if (quantized) quantize_activation(attn.data(), act_attn, h * dim / block_size, (h + 1) * dim / block_size);
                }
                end_phase("attention_merge");
                project(l.o, attn.data(), act_attn, x.data(), true, "attention_output");
                normalize(l.post_norm, quantized);
                project(l.gate_up, my.norm.data(), my.act, gate_up.data(), false, "mlp_gate_up");
                {
                    size_t begin, end;
                    split((config.intermediate + block_size - 1) / block_size, begin, end);
                    const float* g = gate_up.data();
                    const float* u = g + config.intermediate;
                    const size_t first = std::min(config.intermediate, begin * block_size), last = std::min(config.intermediate, end * block_size);
                    silu_multiply(g + first, u + first, hidden.data() + first, last - first, kernel);
                    if (quantized) quantize_activation(hidden.data(), act_hidden, begin, end);
                }
                end_phase("silu");
                project(l.down, hidden.data(), act_hidden, x.data(), true, "mlp_down");
            }
            if (want_head) {
                normalize(final_norm, head_format != Format::bf16);
                project(head, my.norm.data(), my.act, logits.data(), false, "lm_head");
            }
        }
        if (p) {
            ++p->steps;
            p->embedding_bytes += embedding.weight_bytes() / embedding.rows + embedding.scale_bytes() / embedding.rows;
            for (const Layer& l : layers) for (const Projection* projection : {&l.qkv, &l.o, &l.gate_up, &l.down}) account(p, *projection);
            p->norm_bias_bytes += config.layers * (2 * config.hidden + qkv.size()) * 4;
            p->kv_write_bytes += config.layers * 2 * kv_dim * kv_width;
            p->kv_read_bytes += config.layers * 2 * (pos + 1) * kv_dim * kv_width;
            if (want_head) {
                account(p, head);
                p->norm_bias_bytes += config.hidden * 4; ++p->head_steps;
                p->lm_head_weight_bytes += embedding.weight_bytes();
                p->lm_head_scale_bytes += embedding.scale_bytes();
            }
        }
        ++pos;
        return logits;
    }
};
Engine::Engine(const std::string& dir, const EngineOptions& options) : impl(std::make_unique<Impl>(dir, options)) {}
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
    const auto& c = impl->config;
    const auto& o = impl->options;
    return {{"weight_format", format_name(impl->format)}, {"head_format", format_name(impl->head_format)},
            {"quantization", impl->format == Format::bf16 ? "none" :
                "32-weight blocks with F16 scales; activations quantized to int8 in 32-element blocks; integer dot products"},
            {"kernel", kernel_name(o.kernel)}, {"threads", o.threads}, {"kv_dtype", kv_name(o.kv)},
            {"weight_memory", weight_memory_name(o.weights)}, {"huge_page_kib", impl->huge_kib}, {"fused_projections", o.fuse},
            {"thread_scheduling", "one OpenMP parallel region per token; static row ranges; barrier per phase"},
            {"attention", "grouped flash decoding: KV head x time-chunk items, online softmax, merged per query head"},
            {"stored_weight_bytes", impl->stored_bytes}, {"stored_scale_bytes", impl->stored_scales}, {"kv_capacity", o.capacity},
            {"auxiliary_fp32_weight_bytes", (c.layers * (3 * c.hidden + 2 * impl->kv_dim) + c.hidden) * 4},
            {"kv_cache_bytes", o.capacity * impl->kv_dim * c.layers * 2 * impl->kv_width},
            {"tied_head", true}, {"vocab_size", c.vocab}};
}

void quantize_model(const std::string& source, const std::string& output, Format format, Format head_format) {
    if (format == Format::bf16 || head_format == Format::bf16) throw std::runtime_error("quantization formats must be q8 or q4");
    fs::path src(source), dst(output);
    if (fs::weakly_canonical(src) == fs::weakly_canonical(dst)) throw std::runtime_error("quantization output must differ from input");
    { EngineOptions validate; validate.capacity = 1; validate.weights = WeightMemory::mmap; Engine engine(source, validate); }
    Store store(src, WeightMemory::mmap);
    if (fs::exists(dst / "model.safetensors") || fs::exists(dst / "config.json")) throw std::runtime_error("quantization output already contains a model");
    for (const auto& item : store.tensors) {
        if (item.second.type != DType::bf16) throw std::runtime_error("quantization input must be BF16");
        if (item.second.shape.size() == 2 && item.second.shape[1] % (2 * block_size))
            throw std::runtime_error("quantized matrices need a multiple of 64 columns: " + item.first);
    }
    auto chosen = [&](const std::string& name) { return name == "model.embed_tokens.weight" ? head_format : format; };
    Json header;
    uint64_t offset = 0;
    auto add = [&](const std::string& name, const std::string& dtype, const std::vector<size_t>& shape, uint64_t bytes) {
        header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + bytes}}}; offset += bytes;
    };
    // Data is emitted in the same (map) order as these offsets.
    for (const auto& item : store.tensors) {
        const Tensor& t = item.second;
        if (t.shape.size() == 2) {
            size_t rows = t.shape[0], cols = t.shape[1];
            if (chosen(item.first) == Format::q8) add(item.first, "I8", {rows, cols}, rows * cols);
            else add(item.first, "U8", {rows, cols / 2}, rows * cols / 2);
            add(item.first + ".scales", "F16", {rows, cols / block_size}, rows * (cols / block_size) * 2);
        } else add(item.first, "F32", t.shape, t.count() * 4);
    }
    header["__metadata__"] = {{"format", "pt"}, {"quantization", "block32"}, {"weight_format", format_name(format)}, {"head_format", format_name(head_format)}};
    std::string encoded = header.dump();
    while (encoded.size() % 8) encoded.push_back(' ');
    fs::create_directories(dst);
    fs::path temporary = dst / "model.safetensors.partial";
    try {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create quantized model");
        uint64_t header_size = encoded.size();
        out.write(reinterpret_cast<const char*>(&header_size), 8); out.write(encoded.data(), encoded.size());
        for (const auto& item : store.tensors) {
            const Tensor& t = item.second;
            if (t.shape.size() == 2) {
                const size_t rows = t.shape[0], cols = t.shape[1], blocks = cols / block_size;
                const bool q8 = chosen(item.first) == Format::q8;
                std::vector<float> row(cols);
                std::vector<uint8_t> quantized(q8 ? cols : cols / 2);
                std::vector<uint16_t> scales(rows * blocks);
                for (size_t r = 0; r < rows; ++r) {
                    for (size_t j = 0; j < cols; ++j) row[j] = t.at(r * cols + j);
                    if (q8) quantize_q8(row.data(), cols, reinterpret_cast<int8_t*>(quantized.data()), scales.data() + r * blocks);
                    else quantize_q4(row.data(), cols, quantized.data(), scales.data() + r * blocks);
                    out.write(reinterpret_cast<const char*>(quantized.data()), std::streamsize(quantized.size()));
                }
                out.write(reinterpret_cast<const char*>(scales.data()), std::streamsize(scales.size() * 2));
            } else {
                for (size_t j = 0; j < t.count(); ++j) {
                    float value = t.at(j);
                    if (!std::isfinite(value)) throw std::runtime_error("nonfinite model vector");
                    out.write(reinterpret_cast<const char*>(&value), 4);
                }
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
