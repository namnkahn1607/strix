// Throughput benchmark for tagged Trieber Stack: measure CAS contention at
// stack head under concurrent push/pop.
//
// Run with --benchmark_repetitions=10 and --benchmark_counters_tabular=true
// to compare per-thread throughput across thread counts.

#include <benchmark/benchmark.h>

#include "base/tagged_treiber.h"

using namespace strix;

static void BenchTreiberStackConcurrentPopPush(benchmark::State& state) {
    constexpr uint32_t kBenchCapacity = 1 << 16;

    TreiberStack* shared_stack = nullptr;
    if (state.thread_index() == 0) {
        shared_stack = new TreiberStack{kBenchCapacity};
    }

    uint64_t local_ops = 0;
    for (auto _ : state) {
        const auto id = shared_stack->Pop();
        if (id != TreiberStack::kEmpty) {
            shared_stack->Push(id);
            ++local_ops;
        }
    }

    state.counters["pop_push"] = benchmark::Counter(
        static_cast<double>(local_ops), benchmark::Counter::kIsRate
    );

    if (state.thread_index() == 0) {
        delete shared_stack;
    }
}

BENCHMARK(BenchTreiberStackConcurrentPopPush)->Threads(1);
BENCHMARK(BenchTreiberStackConcurrentPopPush)->Threads(2);
BENCHMARK(BenchTreiberStackConcurrentPopPush)->Threads(4);
BENCHMARK(BenchTreiberStackConcurrentPopPush)->Threads(8);

BENCHMARK_MAIN();
