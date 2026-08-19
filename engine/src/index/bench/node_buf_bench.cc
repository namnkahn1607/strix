// Throughput/latency benchmark for L0-tier node buffer.
//
// The goal is to isolate producer-side CAS contention from consumer-side
// contention-free path.

#include <benchmark/benchmark.h>

#include <atomic>
#include <thread>

#include "node_buf.h"

namespace {

// Adapts to real usage capacity of the node buffer on production.
inline constexpr uint32_t kBenchCapacity = 4'096u;

}  // namespace

// BenchNodeBuf_ConcurrentPush measures contention shared by producers.
// Each benchmarked thread is a producer calling `TryPush` in a tight loop.
// One extra, non-benchmarked consumer thread calling `TryPop`, hence draining
// the buffer continuously so the buffer stays near steady-state occupancy
// instead of being saturated.
static void BenchNodeBuf_ConcurrentPush(benchmark::State& state) {
    static NodeBuf*           shared_buf = nullptr;
    static std::atomic<bool>* stop_flag  = nullptr;
    static std::thread*       consumer   = nullptr;

    if (state.thread_index() == 0) {
        shared_buf = new NodeBuf(kBenchCapacity);
        stop_flag  = new std::atomic<bool>(false);
        consumer   = new std::thread([] {
            while (!stop_flag->load(std::memory_order_relaxed)) {
                shared_buf->TryPop();
            }
        });
    }

    // Google Benchmark synchronizes all threads before the loop below begins,
    // so the consumer and node buf constructed on thread 0 are visible to every
    // benchmarked producer thread without an extra barrier.

    auto     local_id = static_cast<uint32_t>(state.thread_index()) << 16;
    uint64_t local_successes = 0;
    for (auto _ : state) {
        if (shared_buf->TryPush(local_id)) {
            ++local_successes;
        }

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
