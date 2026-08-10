// Throughput benchmark for tagged Trieber Stack under concurrent Pop/Push,
// isolating the CAS contention at the stack head from any other subsystems.
//
// Run with --benchmark_counters_tabular=true to compare per-thread throughput
// across thread counts; a non-scaling curve indicates a single cache line
// backing the stack head has become the bottleneck.

#include <benchmark/benchmark.h>

#include "common/tagged_treiber.h"

namespace {

// Large enough so that none thread can plausibly drain the stack empty.
inline constexpr uint32_t kBenchCapacity = 1 << 16;

}  // namespace

// BenchTrieberStackConcurrentPopPush invokes each thread alternately performing
// pop/push in a tight loop, which keeps the stack head under sustained
// contention and therefore, never drain it empty.
static void BenchTrieberStackConcurrentPopPush(benchmark::State& state) {
    // One shared instance across all threads; both constructed once and
    // destructed once on thread 0.
    static TreiberStack* shared_stack = nullptr;

    if (state.thread_index() == 0) {
        shared_stack = new TreiberStack(kBenchCapacity);
    }

    uint64_t local_ops = 0;

    // All threads are synchronized before the loop below begins, so contruction
    // on thread 0 is visible to every other threads without an std::barrier.
    for (auto _ : state) {
        const uint32_t id = shared_stack->Pop();
        if (id != TreiberStack::kEmpty) {
            shared_stack->Push(id);
            ++local_ops;
        }
    }

    state.counters["pop_push_pairs"] = benchmark::Counter(
        static_cast<double>(local_ops), benchmark::Counter::kIsRate
    );

    if (state.thread_index() == 0) {
        delete shared_stack;
    }
}

BENCHMARK(BenchTrieberStackConcurrentPopPush)->Threads(1);
BENCHMARK(BenchTrieberStackConcurrentPopPush)->Threads(2);
BENCHMARK(BenchTrieberStackConcurrentPopPush)->Threads(3);
BENCHMARK(BenchTrieberStackConcurrentPopPush)->Threads(4);
BENCHMARK(BenchTrieberStackConcurrentPopPush)->Threads(8);

BENCHMARK_MAIN();
