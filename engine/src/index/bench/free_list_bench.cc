// Author: namnkahn1607
//
// Throughput benchmark for FreeList under concurrent Pop/Push, isolating
// the Treiber Stack head's CAS contention from any other system behavior.
// Run with --benchmark_counters_tabular=true to compare per-thread
// throughput across thread counts; a non-scaling curve indicates the
// single cache line backing free_head_ has become the bottleneck.

#include <benchmark/benchmark.h>

#include "free_list.h"

namespace {

// Large enough that no single thread can plausibly drain it to kEmpty
// within one Pop call under contention; each Pop is immediately followed
// by a Push of the same id, so total live capacity never drifts and the
// list never empties mid-run regardless of how long the benchmark runs.
inline constexpr uint32_t kBenchCapacity = 1 << 16;

}  // namespace

// -----------------------------------------------------------------------------
// BenchFreeListConcurrentPopPush
// Each thread alternates Pop()/Push() in a tight loop, which keeps free_head_
// under sustained contention indefinitely instead of draining the stack after
// a few hundred microseconds. Google Benchmark's threaded mode runs the
// requested thread count concurrently and reports per-iteration time already
// normalized per thread.
// -----------------------------------------------------------------------------

static void BenchFreeListConcurrentPopPush(benchmark::State& state) {
    // One shared FreeList across all threads in this run, constructed
    // exactly once on thread 0 and torn down exactly once on thread 0,
    // per Google Benchmark's fixture-less multithreaded convention.
    static FreeList* shared_list = nullptr;

    if (state.thread_index() == 0) {
        shared_list = new FreeList(kBenchCapacity);
    }

    // Google Benchmark synchronizes all threads before the loop below
    // begins, so construction on thread 0 is visible to every thread
    // without an extra barrier here.
    uint64_t local_ops = 0;
    for (auto _ : state) {
        const uint32_t id = shared_list->Pop();
        if (id != FreeList::kEmpty) {
            shared_list->Push(id);
            ++local_ops;
        }
    }

    state.counters["pop_push_pairs"] = benchmark::Counter(
        static_cast<double>(local_ops), benchmark::Counter::kIsRate
    );

    if (state.thread_index() == 0) {
        delete shared_list;
    }
}

// Thread counts mirror the production allocation-path concurrency this
// FreeList actually sees: 1 as a contention-free baseline, then 2-4 as
// the expected real range, then 8 as a safety margin above it.
BENCHMARK(BenchFreeListConcurrentPopPush)->Threads(1);
BENCHMARK(BenchFreeListConcurrentPopPush)->Threads(2);
BENCHMARK(BenchFreeListConcurrentPopPush)->Threads(3);
BENCHMARK(BenchFreeListConcurrentPopPush)->Threads(4);
BENCHMARK(BenchFreeListConcurrentPopPush)->Threads(8);

BENCHMARK_MAIN();
