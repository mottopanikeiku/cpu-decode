#include "decode.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
using decode::Json;
using Clock = std::chrono::steady_clock;
size_t number(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("invalid nonnegative integer: " + value);
    size_t consumed = 0;
    auto n = std::stoull(value, &consumed);
    if (consumed != value.size() || n > std::numeric_limits<size_t>::max()) throw std::runtime_error("integer overflow");
    return size_t(n);
}
std::vector<int> tokens(const std::string& text) {
    std::vector<int> result;
    std::stringstream input(text);
    std::string item;
    if (text.empty() || text.back() == ',') throw std::runtime_error("empty token list/item");
    while (std::getline(input, item, ',')) {
        size_t n = number(item);
        if (n > size_t(std::numeric_limits<int>::max())) throw std::runtime_error("token ID overflow");
        result.push_back(int(n));
    }
    return result;
}
int argmax(const std::vector<float>& logits) {
    return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
}
void parent_dir(const std::string& path) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}
void emit(const Json& json, const std::string& path) {
    if (path.empty()) { std::cout << json.dump(2) << '\n'; return; }
    parent_dir(path);
    std::ofstream out(path);
    if (!out || !(out << json.dump(2) << '\n')) throw std::runtime_error("cannot write " + path);
}
void usage() {
    std::cout << "cpu-decode quantize --model HF_SNAPSHOT --output INT8_DIR\n"
                 "cpu-decode logits --model DIR --tokens ID,ID --output PREFIX [--threads N --kernel scalar|simd256|simd512|simd512x4]\n"
                 "cpu-decode generate --model DIR --tokens ID,ID --steps N [--output JSON --threads N --kernel ...]\n"
                 "cpu-decode bench --model DIR --tokens ID,ID --context N --steps N --repeats N [--output JSON --threads N --kernel ...]\n"
                 "All forward modes accept --rope cached|direct (default cached).\n"
                 "Token IDs only; generation does not stop at EOS. logits writes PREFIX.bin float32 [positions,vocab] and PREFIX.json.\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") { usage(); return argc < 2 ? 1 : 0; }
        std::string command(argv[1]);
        std::set<std::string> known{"--model", "--output", "--tokens", "--threads", "--kernel", "--steps", "--context", "--repeats", "--rope"};
        std::map<std::string, std::string> options;
        for (int i = 2; i < argc; i += 2) {
            std::string key(argv[i]);
            if (!known.count(key) || i + 1 >= argc || !options.emplace(key, argv[i + 1]).second) throw std::runtime_error("unknown, duplicate or incomplete option: " + key);
        }
        auto required = [&](const std::string& name) {
            auto found = options.find(name);
            if (found == options.end() || found->second.empty()) throw std::runtime_error("missing " + name);
            return found->second;
        };
        auto option = [&](const std::string& name, const std::string& fallback) {
            auto found = options.find(name); return found == options.end() ? fallback : found->second;
        };
        std::string model = required("--model"), output = option("--output", "");
        if (command == "quantize") {
            decode::quantize_model(model, required("--output"));
            emit({{"model", output}, {"weight_dtype", "int8"}, {"quantization", "symmetric per-row int8; FP32 scales, norms and biases; tied head"}}, "");
            return 0;
        }
        if (command != "logits" && command != "generate" && command != "bench") throw std::runtime_error("unknown command: " + command);
        auto prompt = tokens(required("--tokens"));
        size_t nthreads = number(option("--threads", "1"));
        if (!nthreads || nthreads > 1024) throw std::runtime_error("threads must be 1..1024");
        auto kernel = decode::parse_kernel(option("--kernel", "scalar"));
        std::string rope_mode = option("--rope", "cached");
        if (rope_mode != "cached" && rope_mode != "direct") throw std::runtime_error("rope must be cached or direct");
        size_t steps = number(option("--steps", "16"));
        if (steps > 1000000) throw std::runtime_error("too many steps");
        size_t context = command == "bench" ? number(required("--context")) : prompt.size();
        if (!context || context > 1000000) throw std::runtime_error("invalid context size");
        size_t capacity = context + (command == "logits" ? 0 : steps);
        decode::Engine engine(model, kernel, int(nthreads), capacity, rope_mode == "cached");
        // Validate every supplied ID even when a short benchmark context uses only its prefix.
        for (int id : prompt) if (size_t(id) >= engine.vocab_size()) throw std::runtime_error("token ID outside vocabulary");
        Json result = engine.metadata();
        result["model"] = model;
        result["prompt_tokens"] = prompt;
        if (command == "logits") {
            std::string prefix = required("--output");
            parent_dir(prefix);
            std::ofstream binary(prefix + ".bin", std::ios::binary);
            if (!binary) throw std::runtime_error("cannot open logits output");
            std::vector<int> maxima;
            for (int id : prompt) {
                const auto& logits = engine.step(id, true);
                binary.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
                maxima.push_back(argmax(logits));
            }
            binary.close();
            if (!binary) throw std::runtime_error("logits output failed");
            result["shape"] = {prompt.size(), engine.vocab_size()}; result["argmax"] = maxima;
            result["tokens"] = prompt; result["dtype"] = "float32"; result["layout"] = "row-major little-endian";
            emit(result, prefix + ".json");
        } else if (command == "generate") {
            int next = 0;
            for (size_t i = 0; i < prompt.size(); ++i) {
                const auto& logits = engine.step(prompt[i], i + 1 == prompt.size());
                if (i + 1 == prompt.size()) next = argmax(logits);
            }
            std::vector<int> generated;
            for (size_t i = 0; i < steps; ++i) {
                generated.push_back(next);
                if (i + 1 < steps) next = argmax(engine.step(next, true));
            }
            result["generated_tokens"] = generated; result["steps"] = steps;
            emit(result, output);
        } else {
            if (!steps) throw std::runtime_error("benchmark steps must be positive");
            size_t repeats = number(option("--repeats", "5"));
            if (!repeats || repeats > 10000) throw std::runtime_error("repeats must be 1..10000");
            auto prefill_start = Clock::now();
            int seed = 0;
            for (size_t i = 0; i < context; ++i) {
                const auto& logits = engine.step(prompt[i % prompt.size()], i + 1 == context);
                if (i + 1 == context) seed = argmax(logits);
            }
            result["prefill_seconds"] = std::chrono::duration<double>(Clock::now() - prefill_start).count();
            int next = seed;
            next = argmax(engine.step(next, true));
            engine.rewind(context);
            result["context"] = context; result["steps"] = steps; result["repeats"] = repeats;
            result["warmup_steps"] = 1; result["samples"] = Json::array();
            for (size_t r = 0; r < repeats; ++r) {
                engine.rewind(context);
                decode::Profile profile;
                std::vector<double> timings;
                std::vector<int> generated;
                timings.reserve(steps); generated.reserve(steps);
                next = seed;
                auto start = Clock::now();
                for (size_t i = 0; i < steps; ++i) {
                    auto step_start = Clock::now();
                    generated.push_back(next);
                    const auto& logits = engine.step(next, true, &profile);
                    auto argmax_start = Clock::now();
                    next = argmax(logits);
                    auto end = Clock::now();
                    profile.seconds["argmax"] += std::chrono::duration<double>(end - argmax_start).count();
                    timings.push_back(std::chrono::duration<double>(end - step_start).count());
                }
                double seconds = std::chrono::duration<double>(Clock::now() - start).count();
                double attributed = 0;
                for (const auto& operation : profile.seconds) attributed += operation.second;
                profile.seconds["timing_overhead_and_loop"] = std::max(0.0, seconds - attributed);
                Json bytes{{"matrix_weights", double(profile.matrix_weight_bytes) / steps}, {"scales", double(profile.scale_bytes) / steps},
                           {"norm_bias", double(profile.norm_bias_bytes) / steps}, {"embedding", double(profile.embedding_bytes) / steps},
                           {"kv_read_min", double(profile.kv_read_min_bytes) / steps}, {"kv_read_logical", double(profile.kv_read_logical_bytes) / steps},
                           {"kv_write", double(profile.kv_write_bytes) / steps}, {"total_min", profile.json()["minimum_bytes"].get<double>() / steps}};
                bytes["lm_head"] = double(profile.lm_head_weight_bytes) / steps;
                bytes["lm_head_scales"] = double(profile.lm_head_scale_bytes) / steps;
                bytes["projection_weights"] = double(profile.matrix_weight_bytes - profile.lm_head_weight_bytes) / steps;
                result["samples"].push_back({{"seconds", seconds}, {"tokens_per_second", steps / seconds}, {"step_seconds", timings},
                    {"generated_tokens", generated}, {"operation_seconds", profile.seconds}, {"profile", profile.json()}, {"bytes_per_token", bytes}});
            }
            result["timing_scope"] = "decode forward including LM head and greedy argmax; excludes one-time load/prefill and one warmup step; operation timers enabled";
            result["context_tokens"] = "provided token IDs repeated to context length";
            emit(result, output);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpu-decode: " << error.what() << '\n';
        return 1;
    }
}
