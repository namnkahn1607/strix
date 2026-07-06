// Author: namnkahn1607
//
// Throughput benchmark for L0Indices, isolating Producer-side CAS
// contention on push_head_ from TryPop's contention-free path (TryPop
// has exactly one caller in production and uses no CAS at all).
// A background consumer thread drains continuously so the benchmarked
// producer threads measure true CAS contention rather than the cost of
// being rejected by a ring that has been allowed to fill solid.

#include <benchmark/benchmark.h>

#include <atomic>
#include <thread>

#include "level0_ring.h"

namespace {

inline constexpr uint32_t kBenchCapacity = 1 << 12;  // 4096

}  // namespace

// -----------------------------------------------------------------------------
// Bench_L0ConcurrentPush
// Each benchmarked thread is a Producer calling TryPush in a tight loop.
// One extra, non-benchmarked background thread continuously drains via
// TryPop so the ring stays near steady-state occupancy instead of
// saturating solid and measuring "rejection speed" instead of CAS cost.
// -----------------------------------------------------------------------------

static void Bench_L0ConcurrentPush(benchmark::State& state) {
    static L0Indices*         shared_ring = nullptr;
    static std::atomic<bool>* stop_flag   = nullptr;
    static std::thread*       drainer     = nullptr;

    if (state.thread_index() == 0) {
        shared_ring = new L0Indices(kBenchCapacity);
        stop_flag   = new std::atomic<bool>(false);
        drainer     = new std::thread([] {
            while (!stop_flag->load(std::memory_order_relaxed)) {
                shared_ring->TryPop();
            }
        });
    }

    // Google Benchmark synchronizes all threads before the loop below
    // begins, so the drainer and ring constructed on thread 0 are visible
    // to every benchmarked producer thread without an extra barrier.
    uint64_t local_id = static_cast<uint64_t>(state.thread_index()) << 32;
    uint64_t local_successes = 0;
    for (auto _ : state) {
        if (shared_ring->TryPush(static_cast<uint32_t>(local_id))) {
            ++local_successes;
        }

        ++local_id;
    }

    state.counters["pushes"] = benchmark::Counter(
        static_cast<double>(local_successes), benchmark::Counter::kIsRate);

    if (state.thread_index() == 0) {
        stop_flag->store(true, std::memory_order_relaxed);
        drainer->join();

        delete drainer;
        delete stop_flag;
        delete shared_ring;
    }
}

// Thread counts mirror the production concurrency cache-miss producers perform
// on this ring buffer: 1 as a contention-free baseline, 2-4 as the expected
// real range, 8 as a safety margin.
BENCHMARK(Bench_L0ConcurrentPush)->Threads(1);
BENCHMARK(Bench_L0ConcurrentPush)->Threads(2);
BENCHMARK(Bench_L0ConcurrentPush)->Threads(3);
BENCHMARK(Bench_L0ConcurrentPush)->Threads(4);
BENCHMARK(Bench_L0ConcurrentPush)->Threads(8);

BENCHMARK_MAIN();
