// Public llama C API at commit 6c73b3e12dc501de35fe5f6979960d06921a2f6c.
#include <llama.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "logits output requires IEEE-754 float32");

int32_t number(const std::string& text) {
    int32_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos ||
        parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("invalid nonnegative integer: " + text);
    return value;
}

std::vector<llama_token> parse_tokens(const std::string& text) {
    std::vector<llama_token> result;
    size_t begin = 0;
    for (;;) {
        const size_t end = text.find(',', begin);
        result.push_back(number(text.substr(begin, end == std::string::npos ? end : end - begin)));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (result.size() > size_t(std::numeric_limits<llama_pos>::max()))
        throw std::runtime_error("too many token positions");
    return result;
}

// Inclusive "A-B" or "A" ranges, increasing and inside [0, count), as in cpu-decode logits.
std::vector<bool> parse_positions(const std::string& text, size_t count) {
    std::vector<bool> wanted(count, false);
    size_t begin = 0, next = 0;
    for (;;) {
        const size_t end = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
        const size_t dash = item.find('-');
        const size_t first = size_t(number(item.substr(0, dash)));
        const size_t last = dash == std::string::npos ? first : size_t(number(item.substr(dash + 1)));
        if (first < next || first > last || last >= count)
            throw std::runtime_error("positions must be increasing ranges inside the token list");
        std::fill(wanted.begin() + std::ptrdiff_t(first), wanted.begin() + std::ptrdiff_t(last) + 1, true);
        next = last + 1;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return wanted;
}

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
};

void usage() {
    std::cout << "llama-logits --model GGUF --tokens ID,ID,... --output PREFIX --threads N\n"
                 "             [--positions A-B,C-D,...] [--kv f16|f32]\n"
                 "CPU-only Q8_0 or Q4_0 model, one token per decode. K/V cache f16 (default; flash\n"
                 "attention AUTO) or f32 (flash attention disabled: llama.cpp's FA casts K/V to F16).\n"
                 "Writes PREFIX.bin little-endian float32 [positions,vocab] for\n"
                 "the listed inclusive positions (default: all) and PREFIX.json.\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || (argc == 2 && std::string(argv[1]) == "--help")) {
            usage();
            return argc < 2 ? 1 : 0;
        }
        const std::set<std::string> known{"--model", "--tokens", "--output", "--threads", "--positions", "--kv"};
        std::map<std::string, std::string> options;
        for (int i = 1; i < argc; i += 2) {
            const std::string key = argv[i];
            if (!known.count(key) || i + 1 >= argc || !options.emplace(key, argv[i + 1]).second)
                throw std::runtime_error("unknown, duplicate or incomplete option: " + key);
        }
        const auto required = [&](const std::string& key) -> const std::string& {
            const auto found = options.find(key);
            if (found == options.end() || found->second.empty())
                throw std::runtime_error("missing " + key);
            return found->second;
        };
        const std::string& model_path = required("--model");
        const std::string& prefix = required("--output");
        auto tokens = parse_tokens(required("--tokens"));
        const int32_t threads = number(required("--threads"));
        if (threads < 1 || threads > 1024) throw std::runtime_error("threads must be 1..1024");
        const auto positions_option = options.find("--positions");
        const std::vector<bool> wanted = positions_option == options.end() ?
            std::vector<bool>(tokens.size(), true) : parse_positions(positions_option->second, tokens.size());
        const auto kv_option = options.find("--kv");
        const std::string kv = kv_option == options.end() ? "f16" : kv_option->second;
        if (kv != "f16" && kv != "f32") throw std::runtime_error("--kv must be f16 or f32");
        const ggml_type kv_type = kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;

        Backend backend;
        auto model_params = llama_model_default_params();
        // An empty offload-device list excludes every accelerator, even if available.
        ggml_backend_dev_t devices[] = {nullptr};
        model_params.devices = devices;
        model_params.n_gpu_layers = 0;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), model_params), &llama_model_free);
        if (!model) throw std::runtime_error("cannot load model: " + model_path);
        const llama_ftype ftype = llama_model_ftype(model.get());
        if (ftype != LLAMA_FTYPE_MOSTLY_Q8_0 && ftype != LLAMA_FTYPE_MOSTLY_Q4_0)
            throw std::runtime_error("model must be a Q8_0 or Q4_0 file");
        const std::string ftype_name = ftype == LLAMA_FTYPE_MOSTLY_Q8_0 ? "MOSTLY_Q8_0" : "MOSTLY_Q4_0";
        const auto* vocab = llama_model_get_vocab(model.get());
        if (!vocab) throw std::runtime_error("model has no vocabulary");
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (n_vocab <= 0) throw std::runtime_error("invalid vocabulary size");
        for (llama_token token : tokens)
            if (token < 0 || token >= n_vocab) throw std::runtime_error("token ID outside vocabulary");

        auto context_params = llama_context_default_params();
        context_params.n_ctx = uint32_t(tokens.size());
        context_params.n_batch = 1;
        context_params.n_ubatch = 1;
        context_params.n_seq_max = 1;
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context_params.type_k = kv_type;
        context_params.type_v = kv_type;
        // llama.cpp's flash-attention graph casts an F32 K/V cache to F16, so a real
        // F32 cache path needs ordinary attention; F16 keeps the default AUTO choice.
        context_params.flash_attn_type = kv == "f32" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), &llama_free);
        if (!context) throw std::runtime_error("cannot create llama context");
        if (llama_n_ctx_seq(context.get()) < tokens.size())
            throw std::runtime_error("context is too small for supplied tokens");

        const size_t row_bytes = size_t(n_vocab) * sizeof(float);
        if (row_bytes > size_t(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("logits row exceeds output stream limit");
        const uint32_t endian_probe = 1;
        const bool little_endian = *reinterpret_cast<const unsigned char*>(&endian_probe) == 1;
        // The common little-endian path writes llama's buffer directly, without copying.
        std::vector<uint32_t> swapped(little_endian ? 0 : size_t(n_vocab));
        std::vector<llama_token> argmax;
        std::vector<size_t> emitted;
        const auto parent = std::filesystem::path(prefix).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream binary(prefix + ".bin", std::ios::binary);
        if (!binary) throw std::runtime_error("cannot open logits output: " + prefix + ".bin");

        llama_pos position = 0;
        int32_t n_seq_id = 1;
        llama_seq_id sequence = 0;
        llama_seq_id* sequence_ptr = &sequence;
        int8_t output_logits = 0;
        llama_batch batch{};
        batch.n_tokens = 1;
        batch.pos = &position;
        batch.n_seq_id = &n_seq_id;
        batch.seq_id = &sequence_ptr;
        batch.logits = &output_logits;
        for (size_t i = 0; i < tokens.size(); ++i) {
            position = llama_pos(i);
            batch.token = &tokens[i];
            // Unwanted positions skip llama's output projection and copy.
            output_logits = wanted[i] ? 1 : 0;
            const int32_t status = llama_decode(context.get(), batch);
            if (status != 0)
                throw std::runtime_error("llama_decode failed at position " + std::to_string(i) +
                                         " with status " + std::to_string(status));
            if (!wanted[i]) continue;
            const float* logits = llama_get_logits_ith(context.get(), 0);
            if (!logits) throw std::runtime_error("missing logits at position " + std::to_string(i));
            llama_token best = 0;
            for (int32_t j = 0; j < n_vocab; ++j) {
                if (!std::isfinite(logits[j]))
                    throw std::runtime_error("nonfinite logits at position " + std::to_string(i));
                if (logits[j] > logits[best]) best = j;
                if (!little_endian) {
                    uint32_t bits;
                    std::memcpy(&bits, &logits[j], sizeof(bits));
                    swapped[size_t(j)] = ((bits & 0x000000ffU) << 24) | ((bits & 0x0000ff00U) << 8) |
                                         ((bits & 0x00ff0000U) >> 8) | ((bits & 0xff000000U) >> 24);
                }
            }
            argmax.push_back(best);
            emitted.push_back(i);
            const auto* bytes = little_endian ? reinterpret_cast<const char*>(logits) :
                                               reinterpret_cast<const char*>(swapped.data());
            if (!binary.write(bytes, std::streamsize(row_bytes)))
                throw std::runtime_error("cannot write logits output: " + prefix + ".bin");
        }
        binary.close();
        if (!binary) throw std::runtime_error("cannot close logits output: " + prefix + ".bin");
        const Json metadata{
            {"shape", {emitted.size(), size_t(n_vocab)}},
            {"tokens", tokens},
            {"positions", emitted},
            {"argmax", argmax},
            {"threads", threads},
            {"kv_dtype", kv == "f32" ? "float32" : "float16"},
            {"ftype", ftype_name},
            {"flash_attention", kv == "f32" ? "disabled" : "auto"},
            {"model", std::filesystem::path(model_path).filename().string()}
        };
        std::ofstream json(prefix + ".json");
        if (!json || !(json << metadata.dump(2) << '\n'))
            throw std::runtime_error("cannot write metadata output: " + prefix + ".json");
        json.close();
        if (!json) throw std::runtime_error("cannot close metadata output: " + prefix + ".json");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "llama-logits: " << error.what() << '\n';
        return 1;
    }
}
