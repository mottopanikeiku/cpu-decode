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
// Inclusive "A-B" or "A" ranges, increasing and inside [0, count).
std::vector<bool> positions(const std::string& text, size_t count) {
    std::vector<bool> wanted(count, false);
    std::stringstream input(text);
    std::string item;
    size_t next = 0;
    if (text.empty() || text.back() == ',') throw std::runtime_error("empty position range");
    while (std::getline(input, item, ',')) {
        size_t dash = item.find('-');
        size_t first = number(item.substr(0, dash)), last = dash == std::string::npos ? first : number(item.substr(dash + 1));
        if (first < next || first > last || last >= count) throw std::runtime_error("positions must be increasing ranges inside the token list");
        std::fill(wanted.begin() + std::ptrdiff_t(first), wanted.begin() + std::ptrdiff_t(last) + 1, true);
        next = last + 1;
    }
    return wanted;
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
    std::cout << "cpu-decode quantize --model HF_SNAPSHOT --output DIR [--format q8|q4 --head-format q8|q4]\n"
                 "cpu-decode logits --model DIR --tokens ID,ID --output PREFIX [--positions A-B,C-D ...]\n"
                 "cpu-decode generate --model DIR --tokens ID,ID --steps N [--output JSON ...]\n"
                 "cpu-decode bench --model DIR --tokens ID,ID --context N --steps N --repeats N [--output JSON ...]\n"
                 "Forward options: --threads N --kernel auto|scalar|avx512|neon --kv f16|f32\n"
                 "                 --weights hugepage|mmap --fuse on|off (defaults: 1 auto f16 hugepage on).\n"
                 "Token IDs only; generation does not stop at EOS. logits writes PREFIX.bin float32 [positions,vocab]\n"
                 "and PREFIX.json; --positions limits output (and LM-head work) to the listed inclusive ranges.\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") { usage(); return argc < 2 ? 1 : 0; }
        std::string command(argv[1]);
        std::set<std::string> known{"--model", "--output", "--tokens", "--threads", "--kernel", "--steps", "--context", "--repeats",
                                    "--kv", "--weights", "--fuse", "--format", "--head-format", "--positions"};
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
            auto format = decode::parse_format(option("--format", "q8")), head = decode::parse_format(option("--head-format", option("--format", "q8")));
            decode::quantize_model(model, required("--output"), format, head);
            emit({{"model", output}, {"weight_format", decode::format_name(format)}, {"head_format", decode::format_name(head)},
                  {"quantization", "32-weight blocks, F16 scales; FP32 norms and biases; tied head"}}, "");
            return 0;
        }
        if (command != "logits" && command != "generate" && command != "bench") throw std::runtime_error("unknown command: " + command);
        auto prompt = tokens(required("--tokens"));
        size_t nthreads = number(option("--threads", "1"));
        if (!nthreads || nthreads > 1024) throw std::runtime_error("threads must be 1..1024");
        std::string fuse = option("--fuse", "on");
        if (fuse != "on" && fuse != "off") throw std::runtime_error("fuse must be on or off");
        size_t steps = number(option("--steps", "16"));
        if (steps > 1000000) throw std::runtime_error("too many steps");
        size_t context = command == "bench" ? number(required("--context")) : prompt.size();
        if (!context || context > 1000000) throw std::runtime_error("invalid context size");
        decode::EngineOptions settings;
        settings.kernel = decode::parse_kernel(option("--kernel", "auto"));
        settings.threads = int(nthreads);
        settings.capacity = context + (command == "logits" ? 0 : steps);
        settings.kv = decode::parse_kv(option("--kv", "f16"));
        settings.weights = decode::parse_weight_memory(option("--weights", "hugepage"));
        settings.fuse = fuse == "on";
        if (command != "logits" && options.count("--positions")) throw std::runtime_error("--positions applies to logits only");
        decode::Engine engine(model, settings);
        // Validate every supplied ID even when a short benchmark context uses only its prefix.
        for (int id : prompt) if (size_t(id) >= engine.vocab_size()) throw std::runtime_error("token ID outside vocabulary");
        Json result = engine.metadata();
        result["model"] = model;
        result["prompt_tokens"] = prompt;
        if (command == "logits") {
            std::string prefix = required("--output");
            auto wanted = options.count("--positions") ? positions(options["--positions"], prompt.size()) : std::vector<bool>(prompt.size(), true);
            parent_dir(prefix);
            std::ofstream binary(prefix + ".bin", std::ios::binary);
            if (!binary) throw std::runtime_error("cannot open logits output");
            std::vector<int> maxima;
            std::vector<size_t> emitted;
            for (size_t i = 0; i < prompt.size(); ++i) {
                const auto& logits = engine.step(prompt[i], wanted[i]);
                if (!wanted[i]) continue;
                binary.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
                maxima.push_back(argmax(logits));
                emitted.push_back(i);
            }
            binary.close();
            if (!binary) throw std::runtime_error("logits output failed");
            result["shape"] = {emitted.size(), engine.vocab_size()}; result["argmax"] = maxima; result["positions"] = emitted;
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
            result["context_fill_seconds"] = std::chrono::duration<double>(Clock::now() - prefill_start).count();
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
                           {"kv_read", double(profile.kv_read_bytes) / steps}, {"kv_write", double(profile.kv_write_bytes) / steps},
                           {"total_min", profile.json()["minimum_bytes"].get<double>() / steps}};
                bytes["lm_head"] = double(profile.lm_head_weight_bytes) / steps;
                bytes["lm_head_scales"] = double(profile.lm_head_scale_bytes) / steps;
                bytes["projection_weights"] = double(profile.matrix_weight_bytes - profile.lm_head_weight_bytes) / steps;
                result["samples"].push_back({{"seconds", seconds}, {"tokens_per_second", steps / seconds}, {"step_seconds", timings},
                    {"generated_tokens", generated}, {"operation_seconds", profile.seconds}, {"profile", profile.json()}, {"bytes_per_token", bytes}});
            }
            result["timing_scope"] = "decode forward including LM head and greedy argmax; excludes one-time load, context fill and one warmup step; operation timers enabled";
            result["context_fill"] = "one token at a time (no batched prefill); recorded for completeness, not a prefill benchmark";
            result["context_tokens"] = "provided token IDs repeated to context length";
            emit(result, output);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpu-decode: " << error.what() << '\n';
        return 1;
    }
}
