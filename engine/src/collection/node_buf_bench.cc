// Throughput/latency benchmark for L0-tier node buffer: isolate & measure
// producer-side CAS contention, NOT consumer-side contention-free path.

#include <benchmark/benchmark.h>

#include <atomic>
#include <thread>

#include "collection/config.h"
#include "node_buf.h"

using namespace strix::collection;

// Benchmarked producer threads calling enqueue in a tight loop, while one
// extra, non-benchmarked consumer thread calling dequeue. The buffer gets
// drained continuously so it stays near state-state occupancy.
static void BenchNodeBuf_ConcurrentPush(benchmark::State& state) {
    static NodeBuf*           shared_buf = nullptr;
    static std::atomic<bool>* stop_flag  = nullptr;
    static std::thread*       consumer   = nullptr;

    if (state.thread_index() == 0) {
        shared_buf = new NodeBuf{Config::Standard().lvl0_capacity};
        stop_flag  = new std::atomic<bool>(false);
        consumer   = new std::thread([] {
            while (!stop_flag->load(std::memory_order_relaxed)) {
                shared_buf->TryDequeue();
            }
        });
    }

    auto     local_id = static_cast<uint32_t>(state.thread_index()) << 16;
    uint64_t local_successes = 0;
    for (auto _ : state) {
        local_successes += (shared_buf->TryEnqueue(local_id));
        ++local_id;
    }

    state.counters["pushes"] = benchmark::Counter(
        static_cast<double>(local_successes), benchmark::Counter::kIsRate
    );

    if (state.thread_index() == 0) {
        stop_flag->store(true, std::memory_order_relaxed);
        consumer->join();

        delete consumer;
        delete stop_flag;
        delete shared_buf;
    }
}

BENCHMARK(BenchNodeBuf_ConcurrentPush)->Threads(1);
BENCHMARK(BenchNodeBuf_ConcurrentPush)->Threads(2);
BENCHMARK(BenchNodeBuf_ConcurrentPush)->Threads(3);
BENCHMARK(BenchNodeBuf_ConcurrentPush)->Threads(4);
BENCHMARK(BenchNodeBuf_ConcurrentPush)->Threads(8);

BENCHMARK_MAIN();
