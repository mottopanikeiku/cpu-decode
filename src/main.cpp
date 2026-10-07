#include "decode.hpp"
#include "generate.hpp"
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
    std::cout << "cpu-decode cpus\n"
                 "cpu-decode quantize --model HF_SNAPSHOT --output INT8_DIR [--group-size 0|32|64|128 --scale-dtype f16|f32]\n"
                 "cpu-decode logits --model DIR --tokens ID,ID --output PREFIX [--threads N --kernel auto|scalar|simd256|simd512|simd512x4|vnni|vnni16]\n"
                 "cpu-decode generate --model DIR --tokens ID,ID --steps N [--lookup K --ngram N --prefill-batch K --output JSON --threads N --kernel ...]\n"
                 "cpu-decode time-generate --model DIR --tokens ID,ID --steps N --repeats N [same generation options]\n"
                 "cpu-decode bench --model DIR --tokens ID,ID --context N --steps N --repeats N [--output JSON --threads N --kernel ...]\n"
                 "Forward modes: --rope cached|direct --kv f16|f32|i8|i8-centered --attention blocked|scalar --scheduler pool|openmp --affinity strict|unpinned --cpu-set IDS.\n"
                 "Kernel defaults to auto (FP32 activations); vnni uses int8 activations/groups of32; vnni16 uses int16 activations/groups of64. Both require int8 weights.\n"
                 "logits accepts --logits-start N or --logits-positions N,N and --batch K (default 1).\n"
                 "Token IDs only; generation does not stop at EOS. logits writes PREFIX.bin float32 [positions,vocab] and PREFIX.json.\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") { usage(); return argc < 2 ? 1 : 0; }
        std::string command(argv[1]);
        if (command == "cpus") {
            if (argc != 2) throw std::runtime_error("cpus takes no arguments");
            emit(decode::cpu_topology(), ""); return 0;
        }
        std::set<std::string> known{"--model", "--output", "--tokens", "--threads", "--kernel", "--steps", "--context", "--repeats", "--rope", "--kv", "--attention", "--scheduler", "--affinity", "--cpu-set", "--group-size", "--scale-dtype", "--logits-start", "--logits-positions", "--batch", "--lookup", "--ngram", "--prefill-batch"};
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
            size_t group = number(option("--group-size", "0"));
            std::string scale = option("--scale-dtype", "f32");
            if (scale != "f16" && scale != "f32") throw std::runtime_error("scale dtype must be f16 or f32");
            decode::quantize_model(model, required("--output"), group, scale == "f16" ? decode::DType::f16 : decode::DType::f32);
            emit({{"model", output}, {"weight_dtype", "int8"}, {"group_size", group}, {"scale_dtype", scale}}, "");
            return 0;
        }
        if (command != "logits" && command != "generate" && command != "time-generate" && command != "bench") throw std::runtime_error("unknown command: " + command);
        auto prompt = tokens(required("--tokens"));
        size_t nthreads = number(option("--threads", "1"));
        if (!nthreads || nthreads > 1024) throw std::runtime_error("threads must be 1..1024");
        auto kernel = decode::parse_kernel(option("--kernel", "auto"));
        std::string rope_mode = option("--rope", "cached");
        if (rope_mode != "cached" && rope_mode != "direct") throw std::runtime_error("rope must be cached or direct");
        decode::EngineOptions engine_options;
        engine_options.cached_rope = rope_mode == "cached";
        std::string kv = option("--kv", "f16"), attention = option("--attention", "blocked"), scheduler = option("--scheduler", "pool"), affinity = option("--affinity", "strict");
        if (kv != "f16" && kv != "f32" && kv != "i8" && kv != "i8-centered") throw std::runtime_error("kv must be f16, f32, i8 or i8-centered");
        if (attention != "blocked" && attention != "scalar") throw std::runtime_error("attention must be blocked or scalar");
        if (scheduler != "pool" && scheduler != "openmp") throw std::runtime_error("scheduler must be pool or openmp");
        if (affinity != "strict" && affinity != "unpinned") throw std::runtime_error("affinity must be strict or unpinned");
        engine_options.strict_affinity = affinity == "strict";
        engine_options.cache_type = kv == "f16" ? decode::CacheType::f16 : kv == "f32" ? decode::CacheType::f32 :
                                    kv == "i8" ? decode::CacheType::i8 : decode::CacheType::i8_centered;
        engine_options.scalar_attention = attention == "scalar"; engine_options.persistent_pool = scheduler == "pool";
        if (!option("--cpu-set", "").empty()) engine_options.cpus = tokens(option("--cpu-set", ""));
        size_t steps = number(option("--steps", "16"));
        if (steps > 1000000) throw std::runtime_error("too many steps");
        size_t context = command == "bench" ? number(required("--context")) : prompt.size();
        if (!context || context > 1000000) throw std::runtime_error("invalid context size");
        size_t capacity = context + (command == "logits" ? 0 : steps);
        if (command == "bench" && (kv == "i8" || kv == "i8-centered"))
            throw std::runtime_error("int8 KV timing uses tools.time_v3; operation-level traffic profiling is not available");
        decode::Engine engine(model, kernel, int(nthreads), capacity, engine_options);
        // Validate every supplied ID even when a short benchmark context uses only its prefix.
        for (int id : prompt) if (size_t(id) >= engine.vocab_size()) throw std::runtime_error("token ID outside vocabulary");
        Json result = engine.metadata();
        result["model"] = model;
        result["prompt_tokens"] = prompt;
        if (command == "logits") {
            std::string prefix = required("--output");
            size_t start = number(option("--logits-start", "0"));
            if (start >= prompt.size()) throw std::runtime_error("logits start outside input positions");
            std::vector<int> selected;
            if (!option("--logits-positions", "").empty()) {
                if (options.count("--logits-start")) throw std::runtime_error("use logits start or positions, not both");
                selected = tokens(option("--logits-positions", ""));
                if (!std::is_sorted(selected.begin(), selected.end()) ||
                    std::adjacent_find(selected.begin(), selected.end()) != selected.end() ||
                    size_t(selected.back()) >= prompt.size()) throw std::runtime_error("logit positions must be increasing and within input");
            }
            auto wanted = [&](size_t position) {
                return selected.empty() ? position >= start : std::binary_search(selected.begin(), selected.end(), int(position));
            };
            result["logit_positions"] = Json::array();
            parent_dir(prefix);
            std::ofstream binary(prefix + ".bin", std::ios::binary);
            if (!binary) throw std::runtime_error("cannot open logits output");
            std::vector<int> maxima;
            size_t batch = number(option("--batch", "1"));
            if (!batch || batch > 64) throw std::runtime_error("batch must be 1..64");
            std::vector<int> input;
            input.reserve(batch);
            for (size_t position = 0; position < prompt.size(); position += batch) {
                size_t end = std::min(prompt.size(), position + batch);
                size_t first_head = 0;
                while (position + first_head < end && !wanted(position + first_head)) ++first_head;
                if (batch == 1) {
                    const auto& logits = engine.step(prompt[position], first_head == 0);
                    if (first_head) continue;
                    binary.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
                    maxima.push_back(argmax(logits)); result["logit_positions"].push_back(position);
                } else {
                    input.assign(prompt.begin() + position, prompt.begin() + end);
                    const auto& logits = engine.batch(input, first_head);
                    for (size_t row = first_head; row < input.size(); ++row) {
                        if (!wanted(position + row)) continue;
                        auto offset = logits.begin() + (row - first_head) * engine.vocab_size();
                        binary.write(reinterpret_cast<const char*>(&*offset), engine.vocab_size() * sizeof(float));
                        maxima.push_back(int(std::max_element(offset, offset + engine.vocab_size()) - offset));
                        result["logit_positions"].push_back(position + row);
                    }
                }
            }
            result["forward_batch"] = batch;
            binary.close();
            if (!binary) throw std::runtime_error("logits output failed");
            result["shape"] = {maxima.size(), engine.vocab_size()}; result["argmax"] = maxima;
            result["tokens"] = prompt; result["dtype"] = "float32"; result["layout"] = "row-major little-endian";
            emit(result, prefix + ".json");
        } else if (command == "generate" || command == "time-generate") {
            decode::LookupOptions lookup;
            lookup.draft_tokens = number(option("--lookup", "0"));
            lookup.max_ngram = number(option("--ngram", "4"));
            lookup.prefill_batch = number(option("--prefill-batch", "8"));
            if (lookup.prefill_batch > 64 || lookup.max_ngram > 64) throw std::runtime_error("prefill batch and ngram must be 1..64");
            if (command == "generate") {
                auto generated = decode::generate(engine, prompt, steps, lookup);
                result["generated_tokens"] = generated.tokens; result["steps"] = steps;
                result["lookup"] = generated.json();
            } else {
                size_t repeats = number(option("--repeats", "5"));
                if (!steps || !repeats || repeats > 10000) throw std::runtime_error("timing needs positive steps and 1..10000 repeats");
                auto warmup = decode::generate(engine, prompt, steps, lookup);
                result["samples"] = Json::array();
                for (size_t repeat = 0; repeat < repeats; ++repeat) {
                    decode::GenerationTiming timing;
                    auto generated = decode::generate(engine, prompt, steps, lookup, &timing);
                    if (generated.tokens != warmup.tokens) throw std::runtime_error("generation changed between timing repeats");
                    result["samples"].push_back({{"prefill_seconds", timing.prefill_seconds},
                        {"decode_seconds", timing.decode_seconds}, {"generated_tokens", generated.tokens}, {"lookup", generated.json()}});
                }
                result["warmup"] = "one complete untimed generation";
                result["timing_scope"] = "prefill and greedy decode measured separately; excludes load, JSON output, and warmup; decode includes lookup, verification, and rollback; the first generated token comes from prefill and the final token is not forwarded";
                result["steps"] = steps; result["repeats"] = repeats;
            }
            result["lookup_draft_limit"] = lookup.draft_tokens;
            result["lookup_max_ngram"] = lookup.max_ngram; result["prefill_batch"] = lookup.prefill_batch;
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
                Json profile_json = profile.json();
                Json bytes{{"matrix_weights", double(profile.matrix_weight_bytes) / steps}, {"scales", double(profile.scale_bytes) / steps},
                           {"norm_bias", double(profile.norm_bias_bytes) / steps}, {"embedding", double(profile.embedding_bytes) / steps},
                           {"kv_read_min", double(profile.kv_read_min_bytes) / steps}, {"kv_read_logical", double(profile.kv_read_logical_bytes) / steps},
                           {"kv_write", double(profile.kv_write_bytes) / steps}, {"total_min", profile_json["minimum_bytes"].get<double>() / steps}};
                bytes["lm_head"] = double(profile.lm_head_weight_bytes) / steps;
                bytes["lm_head_scales"] = double(profile.lm_head_scale_bytes) / steps;
                bytes["projection_weights"] = double(profile.matrix_weight_bytes - profile.lm_head_weight_bytes) / steps;
                result["samples"].push_back({{"seconds", seconds}, {"tokens_per_second", steps / seconds}, {"step_seconds", timings},
                    {"generated_tokens", generated}, {"operation_seconds", profile.seconds}, {"profile", std::move(profile_json)}, {"bytes_per_token", bytes}});
            }
            result["timing_scope"] = "decode forward including LM head and greedy argmax; excludes load/prefill and one warmup step; operation timers enabled";
            result["context_tokens"] = "provided token IDs repeated to context length";
            emit(result, output);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpu-decode: " << error.what() << '\n';
        return 1;
    }
}
