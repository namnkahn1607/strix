// Author: namnkahn1607
//
// Throughput/latency benchmarks for VectorIndex's search subroutines.
// Consists of 2 benchmark style: single-threaded scan for raw latency and
// concurrent contention that mimics production scenarios.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <atomic>
#include <thread>

#include "common/constants.h"
#include "index/vector_index.h"
#include "level0_ring.h"
#include "memory/memory_arena.h"

namespace {

// `FillVector()` fill the specified vector with a dedicated value.
void FillVector(float* dst, const float value) {
    std::fill(dst, dst + kVectorDim, value);
}

}  // namespace

// -----------------------------------------------------------------------------
// BenchSearchL0_FixedOccupancy
// Single-threaded latency at a fixed, fully-packed L0 ring. This is the number
// to read as "the cost of one indirect AVX2 scan" isolated from any contention.
//
// Compare against bench/mpsc_bench.cc's raw L0Buffer throughput to see how
// much of SearchL0's cost is the AVX2 kernel itself versus ring bookkeeping.
// -----------------------------------------------------------------------------

static void BenchSearchL0_FixedOccupancy(benchmark::State& state) {
    MemoryArena arena(ArenaConfig::Compact(kL0Capacity));
    VectorIndex index(arena, kL0Capacity, IvfConfig::Compact(4, 4, 16));

    alignas(32) float node_vec[kVectorDim];
    FillVector(node_vec, 0.1f);
    for (size_t i = 0; i < kL0Capacity; ++i) {
        if (!index.AcquireNode(node_vec, /*now=*/0)) {
            state.SkipWithError("Failed to pack L0 to its fullest during setup"
            );
            return;
        }
    }

    alignas(32) float query[kVectorDim];
    FillVector(query, 0.2f);

    for (auto _ : state) {
        auto result = index.SearchL0(query);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BenchSearchL0_FixedOccupancy)->Unit(benchmark::kMicrosecond);

// -----------------------------------------------------------------------------
// BenchSearchL0_ConcurrentContention
// One measured SearchL0 caller running concurrently against production-shaped
// contention: one simulated Consumer and multiple simulated Producers.
//
// All threads running Consumer and Producers are hand-rolled, not part of
// Google Benchmark's ->Threads(N) mechanism, since only the searching caller
// is being timed - the others exist purely to generate realistic contention.
// -----------------------------------------------------------------------------

static void BenchSearchL0_ConcurrentContention(benchmark::State& state) {
    constexpr uint32_t kNumProducers = 2;
    constexpr uint32_t kNodeID       = 0;

    static MemoryArena* arena = nullptr;
    static VectorIndex* index = nullptr;

    static std::atomic<bool>*        stop    = nullptr;
    static std::vector<std::thread>* workers = nullptr;

    if (state.thread_index() == 0) {
        arena = new MemoryArena(ArenaConfig::Compact(kL0Capacity));
        index =
            new VectorIndex(*arena, kL0Capacity, IvfConfig::Compact(4, 4, 16));
        stop = new std::atomic<bool>(false);

        // Half-fill the L0 ring buffer as initial state to almost alwasy
        // account rooms for Consumer and Producers.
        alignas(32) float fill_vec[kVectorDim];
        FillVector(fill_vec, 0.1f);
        for (size_t i = 0; i < kL0Capacity / 2; ++i) {
            index->AcquireNode(fill_vec, 0);
        }

        L0Buffer& ring = VectorIndexBenchAccess::GetL0Buffer(*index);
        workers        = new std::vector<std::thread>();

        // Initiate one Consumer worker.
        workers->emplace_back([&ring] {
            while (!stop->load(std::memory_order_relaxed)) {
                ring.TryPop();
            }
        });

        // Initiate multiple Producer workers.
        for (uint32_t i = 0; i < kNumProducers; ++i) {
            workers->emplace_back([&ring] {
                while (!stop->load(std::memory_order_relaxed)) {
                    ring.TryPush(kNodeID);
                }
            });
        }
    }

    alignas(32) float query[kVectorDim];
    FillVector(query, 0.3f);

    for (auto _ : state) {
        auto result = index->SearchL0(query);
        benchmark::DoNotOptimize(result);
    }

    if (state.thread_index() == 0) {
        stop->store(true, std::memory_order_relaxed);
        for (auto& w : *workers) {
            w.join();
        }

        delete workers;
        delete stop;
        delete index;
        delete arena;
    }
}

BENCHMARK(BenchSearchL0_ConcurrentContention)
    ->Unit(benchmark::kMicrosecond)
    ->Threads(1);

BENCHMARK_MAIN();
