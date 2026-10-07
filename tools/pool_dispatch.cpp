#include "decode.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 7) throw std::runtime_error("usage: pool-dispatch THREADS TASKS ITERATIONS strict|unpinned [SERIAL_GAP_NS [OUTPUT_JSON]]");
        int threads = std::stoi(argv[1]);
        size_t tasks = std::stoull(argv[2]), iterations = std::stoull(argv[3]);
        std::string affinity = argv[4];
        uint64_t gap_ns = argc > 5 ? std::stoull(argv[5]) : 0;
        if (!tasks || !iterations || tasks > 1000000 || iterations > 1000000 || (affinity != "strict" && affinity != "unpinned"))
            throw std::runtime_error("invalid dispatch benchmark arguments");
        bool strict = affinity == "strict";
        decode::ThreadPool pool(threads, {}, true, strict);
        // Engine::step binds once around a whole forward, not around each job.
        decode::CpuBinding binding(strict ? pool.cpus()[0] : -1);
        std::vector<uint64_t> counters(tasks);
        auto callback = [](void* data, size_t task) noexcept { ++static_cast<uint64_t*>(data)[task]; };
        constexpr size_t warmup = 100, batches = 5;
        for (size_t i = 0; i < warmup; ++i) pool.run(tasks, callback, counters.data());
        decode::Json samples = decode::Json::array();
        for (size_t batch = 0; batch < batches; ++batch) {
            double seconds = 0, gap_seconds = 0;
            if (!gap_ns) {
                auto start = std::chrono::steady_clock::now();
                for (size_t i = 0; i < iterations; ++i) pool.run(tasks, callback, counters.data());
                seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            } else {
                for (size_t i = 0; i < iterations; ++i) {
                    auto start_gap = std::chrono::steady_clock::now();
                    auto deadline = start_gap + std::chrono::nanoseconds(gap_ns);
                    auto start = start_gap;
                    while (start < deadline) start = std::chrono::steady_clock::now();
                    gap_seconds += std::chrono::duration<double>(start - start_gap).count();
                    pool.run(tasks, callback, counters.data());
                    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                }
            }
            samples.push_back({{"seconds", seconds}, {"serial_gap_seconds", gap_seconds}, {"nanoseconds_per_run", seconds * 1e9 / iterations}});
        }
        uint64_t checksum = 0;
        for (auto count : counters) {
            if (count != warmup + batches * iterations) throw std::runtime_error("dispatch omitted or repeated a task");
            checksum += count;
        }
        decode::Json result{{"threads", threads}, {"tasks", tasks}, {"affinity", affinity},
            {"cpu_set", pool.cpus()}, {"iterations_per_batch", iterations}, {"warmup_runs", warmup},
            {"requested_serial_gap_ns", gap_ns}, {"dispatch_timer_includes_per_run_clock", gap_ns != 0},
            {"callback", "increment one independent counter per fixed task"}, {"checksum", checksum},
            {"samples", samples}};
        if (argc == 7) {
            std::ofstream output(argv[6]);
            output << result.dump(2) << '\n';
            if (!output) throw std::runtime_error("cannot write dispatch benchmark result");
        }
        std::cout << result.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
