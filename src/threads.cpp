#include "decode.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace decode {
namespace {
thread_local int bound_cpu = -1, binding_depth = 0;
thread_local cpu_set_t outer_allowed;
cpu_set_t allowed_mask() {
    if (binding_depth) return outer_allowed;
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask)) throw std::runtime_error("cannot read CPU affinity");
    return mask;
}
std::vector<int> parse_cpu_list(const std::string& text) {
    std::vector<int> ids;
    std::istringstream input(text);
    std::string part;
    while (std::getline(input, part, ',')) {
        size_t dash = part.find('-');
        int first = std::stoi(part), last = dash == std::string::npos ? first : std::stoi(part.substr(dash + 1));
        for (int cpu = first; cpu <= last; ++cpu) ids.push_back(cpu);
    }
    return ids;
}
void pause_cpu() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}
void pin_mask(const cpu_set_t& mask) noexcept {
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask)) std::terminate();
}
void pin(int cpu) noexcept {
    cpu_set_t mask;
    CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    // Validated allowed IDs; unexpected OS failures terminate rather than silently unpin.
    pin_mask(mask);
}
} // namespace
CpuBinding::CpuBinding(int cpu) : previous(bound_cpu) {
    if (bound_cpu == cpu) return;
    if (sched_getaffinity(0, sizeof(original), &original)) throw std::runtime_error("cannot read caller CPU affinity");
    const cpu_set_t& allowed = binding_depth ? outer_allowed : original;
    if (cpu != -1 && (cpu < 0 || cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed))) throw std::runtime_error("CPU binding outside caller allowed set");
    if (!binding_depth) outer_allowed = original;
    if (cpu == -1) pin_mask(allowed); else pin(cpu);
    bound_cpu = cpu; ++binding_depth; changed = true;
}
CpuBinding::~CpuBinding() {
    if (!changed) return;
    if (pthread_setaffinity_np(pthread_self(), sizeof(original), &original)) std::terminate();
    bound_cpu = previous; --binding_depth;
}
Json cpu_topology() {
    cpu_set_t mask = allowed_mask();
    std::vector<int> allowed, primary, secondary;
    std::map<int, uint64_t> frequency;
    std::set<int> representatives;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &mask)) allowed.push_back(cpu);
    for (int cpu : allowed) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/";
        std::ifstream freq(base + "cpufreq/cpuinfo_max_freq");
        uint64_t maximum = 0; freq >> maximum; frequency[cpu] = maximum;
        std::ifstream siblings(base + "topology/thread_siblings_list");
        std::string list; siblings >> list;
        int representative = cpu;
        if (!list.empty()) for (int sibling : parse_cpu_list(list))
            if (std::binary_search(allowed.begin(), allowed.end(), sibling)) representative = std::min(representative, sibling);
        if (representatives.insert(representative).second) primary.push_back(cpu);
        else secondary.push_back(cpu);
    }
    auto faster = [&](int a, int b) { return frequency[a] == frequency[b] ? a < b : frequency[a] > frequency[b]; };
    std::sort(primary.begin(), primary.end(), faster); std::sort(secondary.begin(), secondary.end(), faster);
    primary.insert(primary.end(), secondary.begin(), secondary.end());
    Json clocks = Json::object();
    for (auto [cpu, maximum] : frequency) clocks[std::to_string(cpu)] = maximum;
    return {{"allowed_cpu_ids", allowed}, {"preferred_cpu_ids", primary}, {"max_frequency_khz", clocks},
            {"ordering", "physical cores by maximum frequency, then SMT siblings by maximum frequency"}};
}
struct ThreadPool::Impl {
    int threads;
    bool persistent, strict_affinity;
    std::vector<int> cpus;
    std::vector<std::thread> workers;
    cpu_set_t allowed;
    alignas(64) std::atomic<size_t> next{0};
    alignas(64) std::atomic<size_t> remaining{0};
    alignas(64) std::atomic<uint64_t> epoch{0};
    std::atomic<bool> stopping{false};
    size_t tasks = 0, claim_batch = 1;
    Function function = nullptr;
    void* context = nullptr;
    std::mutex wake_mutex, done_mutex;
    std::condition_variable wake, done;
    Impl(int count, const std::vector<int>& requested, bool use_pool, bool bind_threads)
        : threads(count), persistent(use_pool), strict_affinity(bind_threads) {
        if (threads < 1 || threads > 1024) throw std::runtime_error("invalid thread count");
        allowed = allowed_mask();
        auto preferred = cpu_topology()["preferred_cpu_ids"].get<std::vector<int>>();
        if (!strict_affinity) {
            if (!requested.empty()) throw std::runtime_error("CPU set requires strict affinity");
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
        } else if (!requested.empty()) {
            if (requested.size() != size_t(threads)) throw std::runtime_error("CPU set must have one distinct allowed CPU per thread");
            std::set<int> unique;
            for (int cpu : requested) if (cpu < 0 || cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed) || !unique.insert(cpu).second)
                throw std::runtime_error("invalid or unavailable CPU in set");
            cpus = requested;
        } else {
            if (preferred.empty()) throw std::runtime_error("empty allowed CPU set");
            for (int i = 0; i < threads; ++i) cpus.push_back(preferred[size_t(i) % preferred.size()]);
        }
        if (!persistent) { omp_set_dynamic(0); return; }
        try {
            for (int i = 1; i < threads; ++i) workers.emplace_back([this, i] { worker(i); });
        } catch (...) {
            stop();
            throw;
        }
    }
    void consume() noexcept {
        for (;;) {
            size_t first = next.fetch_add(claim_batch, std::memory_order_relaxed);
            if (first >= tasks) return;
            size_t end = std::min(tasks, first + claim_batch);
            for (size_t task = first; task < end; ++task) function(context, task);
        }
    }
    void worker(int id) noexcept {
        if (strict_affinity) pin(cpus[size_t(id)]); else pin_mask(allowed);
        uint64_t observed = 0;
        for (;;) {
            uint64_t current = epoch.load(std::memory_order_acquire);
            for (size_t spin = 0; current == observed && spin < 2048; ++spin) {
                pause_cpu(); current = epoch.load(std::memory_order_acquire);
            }
            if (current == observed) {
                std::unique_lock<std::mutex> lock(wake_mutex);
                wake.wait(lock, [&] { return epoch.load(std::memory_order_acquire) != observed; });
                current = epoch.load(std::memory_order_acquire);
            }
            if (stopping.load(std::memory_order_relaxed)) return;
            observed = current;
            consume();
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(done_mutex);
                done.notify_one();
            }
        }
    }
    void run(size_t count, Function callback, void* data) {
        if (!callback) throw std::runtime_error("missing task callback");
        if (!count) return;
        if (!persistent) {
            #pragma omp parallel num_threads(threads)
            {
                if (strict_affinity) pin(cpus[size_t(omp_get_thread_num())]); else pin_mask(allowed);
                #pragma omp for schedule(static)
                for (size_t task = 0; task < count; ++task) callback(data, task);
            }
            return;
        }
        tasks = count; function = callback; context = data;
        // Keep streaming rows contiguous without changing any logical task boundaries.
        claim_batch = count >= size_t(threads) * 16 ? 8 : 1;
        next.store(0, std::memory_order_relaxed); remaining.store(workers.size(), std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(wake_mutex);
            epoch.fetch_add(1, std::memory_order_release);
        }
        wake.notify_all();
        consume();
        for (size_t spin = 0; remaining.load(std::memory_order_acquire) && spin < 2048; ++spin) pause_cpu();
        if (remaining.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(done_mutex);
            done.wait(lock, [&] { return remaining.load(std::memory_order_acquire) == 0; });
        }
    }
    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(wake_mutex);
            stopping.store(true, std::memory_order_relaxed); epoch.fetch_add(1, std::memory_order_release);
        }
        wake.notify_all();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
    }
    ~Impl() { stop(); }
};
ThreadPool::ThreadPool(int threads, const std::vector<int>& cpus, bool persistent, bool strict_affinity)
    : impl(std::make_unique<Impl>(threads, cpus, persistent, strict_affinity)) {}
ThreadPool::~ThreadPool() = default;
void ThreadPool::run(size_t tasks, Function function, void* context) {
    CpuBinding binding(impl->strict_affinity ? impl->cpus[0] : -1);
    impl->run(tasks, function, context);
}
const std::vector<int>& ThreadPool::cpus() const { return impl->cpus; }
int ThreadPool::size() const { return impl->threads; }
} // namespace decode
