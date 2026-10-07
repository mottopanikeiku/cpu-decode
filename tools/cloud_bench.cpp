// Persistent CPU benchmark using v2 and the public llama.cpp C API at
// 6c73b3e12dc501de35fe5f6979960d06921a2f6c. Only protocol JSON goes to stdout.
// One configuration line, then {"command":"run"} or {"command":"exit"}.
#include "decode.hpp"
#include <llama.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr int steps = 128;
constexpr const char* llama_commit = "6c73b3e12dc501de35fe5f6979960d06921a2f6c";

Json line() {
    std::string text;
    if (!std::getline(std::cin, text)) throw std::runtime_error("unexpected EOF; send command exit");
    Json value = Json::parse(text);
    if (!value.is_object()) throw std::runtime_error("protocol line must be a JSON object");
    return value;
}

int integer(const Json& value, int maximum, const char* name) {
    if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<uint64_t>() > uint64_t(maximum)))
        throw std::runtime_error(std::string(name) + " must be a nonnegative integer in range");
    const int64_t n = value.get<int64_t>();
    if (n < 0 || n > maximum) throw std::runtime_error(std::string(name) + " outside range");
    return int(n);
}

struct Config {
    std::string backend, model, kernel, flash;
    int threads, context;
    std::vector<int> tokens, cpus;
    explicit Config(const Json& value) {
        const std::set<std::string> fields{"backend", "model", "threads", "context", "steps", "tokens", "cpu_set", "kernel", "flash", "poll"};
        for (const auto& item : value.items())
            if (!fields.count(item.key())) throw std::runtime_error("unknown configuration field: " + item.key());
        backend = value.at("backend").get<std::string>();
        model = value.at("model").get<std::string>();
        kernel = value.at("kernel").get<std::string>();
        flash = value.at("flash").get<std::string>();
        threads = integer(value.at("threads"), GGML_MAX_N_THREADS, "threads");
        context = integer(value.at("context"), 4096, "context");
        if (backend != "native" && backend != "llama") throw std::runtime_error("backend must be native or llama");
        if (model.empty()) throw std::runtime_error("model must not be empty");
        if (threads < 1) throw std::runtime_error("threads must be positive");
        if (context != 128 && context != 4096) throw std::runtime_error("context must be 128 or 4096");
        if (integer(value.at("steps"), steps, "steps") != steps) throw std::runtime_error("steps must be 128");
        if (integer(value.at("poll"), 100, "poll") != 50) throw std::runtime_error("poll must be 50");
        if (kernel != "vnni16" && kernel != "auto") throw std::runtime_error("kernel must be vnni16 or auto");
        if (flash != "on" && flash != "off" && flash != "auto") throw std::runtime_error("flash must be on, off or auto");
        if (!value.at("tokens").is_array() || value.at("tokens").empty()) throw std::runtime_error("tokens must be a nonempty array");
        for (const auto& token : value.at("tokens")) tokens.push_back(integer(token, std::numeric_limits<int>::max(), "token"));
        if (!value.at("cpu_set").is_array() || value.at("cpu_set").size() != size_t(threads))
            throw std::runtime_error("cpu_set requires one distinct allowed CPU per thread");
        std::set<int> unique;
        for (const auto& cpu : value.at("cpu_set")) {
            int id = integer(cpu, std::min(CPU_SETSIZE, GGML_MAX_N_THREADS) - 1, "CPU");
            if (!unique.insert(id).second) throw std::runtime_error("duplicate CPU");
            cpus.push_back(id);
        }
    }
    int capacity() const { return context + steps; }
};

// Limit all subsequently created threads to the selected set before either load.
// Pool implementations further pin their compute threads to individual CPUs.
class Affinity {
    cpu_set_t original_{};
public:
    explicit Affinity(const std::vector<int>& cpus) {
        if (sched_getaffinity(0, sizeof(original_), &original_)) throw std::runtime_error("cannot read caller affinity");
        cpu_set_t selected;
        CPU_ZERO(&selected);
        for (int cpu : cpus) {
            if (!CPU_ISSET(cpu, &original_)) throw std::runtime_error("requested CPU is outside allowed affinity");
            CPU_SET(cpu, &selected);
        }
        if (sched_setaffinity(0, sizeof(selected), &selected)) throw std::runtime_error("cannot set caller affinity");
    }
    ~Affinity() { if (sched_setaffinity(0, sizeof(original_), &original_)) std::terminate(); }
    Affinity(const Affinity&) = delete;
    Affinity& operator=(const Affinity&) = delete;
};

Json affinity_snapshot(const Config& config, bool llama) {
    Json result = Json::array();
    size_t singleton_threads = 0;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
        const int tid = std::stoi(entry.path().filename().string());
        cpu_set_t mask;
        if (sched_getaffinity(tid, sizeof(mask), &mask)) throw std::runtime_error("cannot read worker affinity");
        std::vector<int> cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &mask)) {
            if (std::find(config.cpus.begin(), config.cpus.end(), cpu) == config.cpus.end())
                throw std::runtime_error("thread affinity escapes requested CPU set");
            cpus.push_back(cpu);
        }
        if (cpus.empty()) throw std::runtime_error("empty thread affinity");
        if (cpus.size() == 1) ++singleton_threads;
        result.push_back({{"tid", tid}, {"cpu_set", cpus}});
    }
    // Native restores the caller's set between steps. GGML keeps its caller pinned.
    const size_t required = llama || config.threads == 1 ? size_t(config.threads) : size_t(config.threads - 1);
    if (singleton_threads < required) throw std::runtime_error("strict pool affinity was not applied");
    return result;
}

// Earliest token wins ties, exactly as the v2 benchmark. Both paths share this
// scan; neither adds a separate full-vocabulary finite-check pass.
int greedy(const float* logits, size_t size) {
    if (!logits || !size) throw std::runtime_error("missing logits");
    return int(std::max_element(logits, logits + size) - logits);
}

class Native {
    decode::Engine engine_;
public:
    explicit Native(const Config& config)
        : engine_(config.model, decode::parse_kernel(config.kernel), config.threads, size_t(config.capacity()), options(config)) {
        for (int token : config.tokens)
            if (size_t(token) >= engine_.vocab_size()) throw std::runtime_error("token ID outside native vocabulary");
    }
    static decode::EngineOptions options(const Config& config) {
        decode::EngineOptions value;
        value.cached_rope = true;
        value.scalar_attention = false;
        value.persistent_pool = true;
        value.strict_affinity = true;
        value.cache_type = decode::CacheType::f16;
        value.cpus = config.cpus;
        return value;
    }
    int step(int token, bool head) {
        const auto& logits = engine_.step(token, head, nullptr);
        return head ? greedy(logits.data(), logits.size()) : 0;
    }
    void rewind(int context) {
        // The native attention length is position+1: tail entries remain stored
        // but are excluded, then overwritten on the next run. Prefix is intact.
        engine_.rewind(size_t(context));
    }
    Json metadata() const {
        Json value = engine_.metadata();
        value["profile"] = nullptr;
        value["flash"] = "not_applicable";
        value["poll"] = nullptr;
        value["repack"] = "not_applicable";
        value["threadpool"] = "native persistent strict";
        value["worker_spin_rounds"] = 2048;
        return value;
    }
};

// Capture initialization's effective FA choice and actual model-buffer names.
// The callback is never allowed to throw across the C API. Capture is disabled
// before timing so it introduces no allocation into a decode run.
struct Logs {
    std::mutex mutex;
    bool capture = true;
    std::atomic<bool> allocation_error{false};
    int flash = -1;
    std::vector<std::string> buffers;
    static void callback(ggml_log_level, const char* text, void* user) noexcept {
        std::fputs(text, stderr);
        auto& self = *static_cast<Logs*>(user);
        try {
            std::lock_guard<std::mutex> lock(self.mutex);
            if (!self.capture) return;
            if (std::strstr(text, "Flash Attention enabled")) self.flash = 1;
            if (std::strstr(text, "Flash Attention not supported, set to disabled")) self.flash = 0;
            if (std::strstr(text, "model buffer size")) self.buffers.emplace_back(text);
        } catch (...) { self.allocation_error = true; }
    }
};

class LlamaBackend {
    ggml_log_callback old_log_ = nullptr;
    void* old_user_ = nullptr;
public:
    explicit LlamaBackend(Logs& logs) {
        llama_log_get(&old_log_, &old_user_);
        llama_log_set(&Logs::callback, &logs);
        llama_backend_init();
    }
    ~LlamaBackend() {
        llama_backend_free();
        llama_log_set(old_log_, old_user_);
    }
    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;
};

class Llama {
    // Declaration order ensures context dies before pool/model/backend/logs.
    Logs logs_;
    LlamaBackend backend_;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, &llama_model_free};
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool_{nullptr, &ggml_threadpool_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context_{nullptr, &llama_free};
    Json features_ = Json::object();
    int vocab_ = 0;
    int requested_flash_ = -1;
    llama_pos position_ = 0;
    llama_token token_ = 0;
    int32_t n_seq_id_ = 1;
    llama_seq_id sequence_ = 0;
    llama_seq_id* sequence_ptr_ = &sequence_;
    int8_t output_ = 1;
    llama_batch batch_{};
public:
    explicit Llama(const Config& config) : backend_(logs_) {
        const auto reg = ggml_backend_cpu_reg();
        const auto get_features = reinterpret_cast<ggml_backend_get_features_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features"));
        if (!get_features) throw std::runtime_error("CPU backend does not expose features");
        for (auto* feature = get_features(reg); feature && feature->name; ++feature)
            features_[feature->name] = feature->value;
        if (features_.contains("OPENMP")) throw std::runtime_error("build llama.cpp with GGML_OPENMP=OFF for persistent CPU pools");
        if (!features_.contains("REPACK")) throw std::runtime_error("build llama.cpp with GGML_CPU_REPACK=ON");
        auto model_params = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr};
        model_params.devices = devices;
        model_params.n_gpu_layers = 0;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.use_extra_bufts = true;
        model_.reset(llama_model_load_from_file(config.model.c_str(), model_params));
        if (!model_) throw std::runtime_error("cannot load llama model");
        if (llama_model_ftype(model_.get()) != LLAMA_FTYPE_MOSTLY_Q8_0)
            throw std::runtime_error("llama model must use Q8_0 weights");
        const auto* vocab = llama_model_get_vocab(model_.get());
        if (!vocab || (vocab_ = llama_vocab_n_tokens(vocab)) <= 0) throw std::runtime_error("invalid llama vocabulary");
        for (int token : config.tokens) if (token >= vocab_) throw std::runtime_error("token ID outside llama vocabulary");

        auto params = llama_context_default_params();
        params.n_ctx = uint32_t(config.capacity());
        params.n_batch = 1;
        params.n_ubatch = 1;
        params.n_seq_max = 1;
        params.n_threads = config.threads;
        params.n_threads_batch = config.threads;
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
        params.flash_attn_type = config.flash == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED :
                                 config.flash == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
        requested_flash_ = int(params.flash_attn_type);
        params.offload_kqv = false;
        params.op_offload = false;
        params.no_perf = true;
        context_.reset(llama_init_from_model(model_.get(), params));
        if (!context_) throw std::runtime_error("cannot create llama context");
        if (llama_n_ctx_seq(context_.get()) < uint32_t(config.capacity())) throw std::runtime_error("llama context capacity too small");
        if (!llama_get_memory(context_.get())) throw std::runtime_error("llama model has no KV memory");
        auto pool_params = ggml_threadpool_params_default(config.threads);
        std::fill(std::begin(pool_params.cpumask), std::end(pool_params.cpumask), false);
        for (int cpu : config.cpus) pool_params.cpumask[cpu] = true;
        pool_params.strict_cpu = true;
        pool_params.poll = 50;
        pool_params.paused = false;
        pool_.reset(ggml_threadpool_new(&pool_params));
        if (!pool_ || ggml_threadpool_get_n_threads(pool_.get()) != config.threads)
            throw std::runtime_error("cannot create requested llama CPU pool");
        // One persistent pool serves both generation and single-token prefill.
        llama_attach_threadpool(context_.get(), pool_.get(), pool_.get());
        batch_.n_tokens = 1;
        batch_.token = &token_;
        batch_.pos = &position_;
        batch_.n_seq_id = &n_seq_id_;
        batch_.seq_id = &sequence_ptr_;
        batch_.logits = &output_;
    }
    int step(int token, bool head) {
        token_ = llama_token(token);
        output_ = head ? 1 : 0;
        const int status = llama_decode(context_.get(), batch_);
        if (status) throw std::runtime_error("llama_decode failed at position " + std::to_string(position_) + " with status " + std::to_string(status));
        ++position_;
        // llama_get_logits_ith synchronizes the complete forward/output before
        // argmax; no asynchronous work is left outside the timing boundary.
        return head ? greedy(llama_get_logits_ith(context_.get(), 0), size_t(vocab_)) : 0;
    }
    void rewind(int context) {
        llama_synchronize(context_.get());
        const auto memory = llama_get_memory(context_.get());
        if (!llama_memory_seq_rm(memory, sequence_, llama_pos(context), -1))
            throw std::runtime_error("llama cannot remove generated cache suffix");
        if (llama_memory_seq_pos_min(memory, sequence_) != 0 ||
            llama_memory_seq_pos_max(memory, sequence_) != context - 1)
            throw std::runtime_error("llama rewind did not preserve exact prefix");
        position_ = llama_pos(context); // Never rely on automatic position tracking.
    }
    Json metadata() {
        std::lock_guard<std::mutex> lock(logs_.mutex);
        logs_.capture = false;
        if (logs_.allocation_error) throw std::runtime_error("cannot capture llama initialization metadata");
        const int flash = requested_flash_ == -1 ? logs_.flash : requested_flash_;
        if (flash < 0) throw std::runtime_error("AUTO flash resolution missing from upstream initialization log");
        bool repacked = false;
        for (const auto& buffer : logs_.buffers) if (buffer.find("CPU_REPACK") != std::string::npos) repacked = true;
        return {{"llama_commit", llama_commit}, {"weight_dtype", "Q8_0"}, {"kv_dtype", "f16"},
                {"kv_capacity", llama_n_ctx_seq(context_.get())}, {"n_ctx", llama_n_ctx(context_.get())},
                {"n_batch", llama_n_batch(context_.get())}, {"n_ubatch", llama_n_ubatch(context_.get())},
                {"n_seq_max", llama_n_seq_max(context_.get())}, {"threads", llama_n_threads(context_.get())},
                {"threads_batch", llama_n_threads_batch(context_.get())}, {"kernel", "upstream CPU dispatch"},
                {"flash", flash ? "on" : "off"}, {"poll", 50}, {"repack", true},
                {"repack_buffer_used", repacked}, {"model_buffer_logs", logs_.buffers}, {"cpu_features", features_},
                {"stored_weight_bytes", llama_model_size(model_.get())}, {"vocab_size", vocab_},
                {"profile", nullptr}, {"no_perf", true}, {"n_gpu_layers", 0}, {"offload_kqv", false},
                {"op_offload", false}, {"affinity", "strict"}, {"threadpool", "ggml persistent strict"},
                {"threadpool_threads", ggml_threadpool_get_n_threads(pool_.get())}, {"shared_batch_pool", true}};
    }
};

void emit(const Json& value) {
    std::cout << value.dump() << '\n' << std::flush;
    if (!std::cout) throw std::runtime_error("cannot write protocol reply");
}

template<class Backend>
void serve(const Config& config) {
    Affinity affinity(config.cpus);
    Backend backend(config);
    int seed = 0;
    for (int i = 0; i < config.context; ++i)
        seed = backend.step(config.tokens[size_t(i) % config.tokens.size()], i + 1 == config.context);
    int next = seed;
    // Full untimed 128-token forward + head + greedy warmup, final token included.
    for (int i = 0; i < steps; ++i) next = backend.step(next, true);
    backend.rewind(config.context);
    Json metadata = backend.metadata();
    metadata["backend"] = config.backend;
    metadata["model"] = std::filesystem::path(config.model).filename().string();
    metadata["cpu_set"] = config.cpus;
    metadata["thread_affinity_after_warmup"] = affinity_snapshot(config, config.backend == "llama");
    metadata["context"] = config.context;
    metadata["requested_capacity"] = config.capacity();
    metadata["steps"] = steps;
    metadata["warmup_steps"] = steps;
    metadata["requested_kernel"] = config.kernel;
    metadata["requested_flash"] = config.flash;
    metadata["requested_poll"] = 50;
    metadata["prompt_tokens"] = config.tokens;
    metadata["context_tokens"] = "provided token IDs repeated to context length";
    metadata["timing_scope"] = "128 complete single-token forwards including LM head and earliest-tie greedy argmax; final consumed token forwarded; excludes load, prefill, warmup, rewind and JSON; no operation timers";
    std::vector<int> generated(size_t(steps), 0); // Allocated once, outside all clocks.
    emit({{"event", "ready"}, {"metadata", metadata}, {"seed", seed}});
    for (;;) {
        const Json request = line();
        if (request.size() != 1 || !request.contains("command") || !request.at("command").is_string())
            throw std::runtime_error("expected only a string command");
        const std::string command = request.at("command").get<std::string>();
        if (command == "exit") return;
        if (command != "run") throw std::runtime_error("command must be run or exit");
        backend.rewind(config.context);
        next = seed;
        const auto start = Clock::now();
        for (int i = 0; i < steps; ++i) {
            generated[size_t(i)] = next;
            next = backend.step(next, true);
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        if (!std::isfinite(seconds) || seconds <= 0) throw std::runtime_error("nonfinite or nonpositive elapsed time");
        emit({{"seconds", seconds}, {"tokens", generated}, {"next_token", next}});
    }
}
} // namespace

int main(int argc, char**) {
    try {
        if (argc != 1) throw std::runtime_error("cloud-bench accepts configuration and commands on stdin, not arguments");
        const Config config(line());
        if (config.backend == "native") serve<Native>(config);
        else serve<Llama>(config);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cloud-bench: " << error.what() << '\n';
        return 1;
    }
}
