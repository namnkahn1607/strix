// Author: namnkahn1607
//
// SearchL0 is on the critical path of the system.
// It combines AVX2 dot product scoring with NodeState classification
// for all 1,000 L0 slots.

#include <benchmark/benchmark.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <random>

#include "arena.h"
#include "constants.h"
#include "meta_node.h"
#include "search.h"

namespace {

inline constexpr uint64_t kNow = 1000;

// GenQueryVector(): allocates a 32-byte aligned unit vector of size 384 along
// dimension 0. Ownership: Caller must `std::free()`.
float* GenQueryVector() {
    auto* vec =
        static_cast<float*>(std::aligned_alloc(32, kVectorDim * sizeof(float)));
    std::memset(vec, 0, kVectorDim * sizeof(float));
    vec[0] = 1.0f;

    return vec;
}

// SetReady(): set a specified Arena slot READY with a normalized random vector.
void SetReady(MemoryArena& arena, const size_t node_id, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float*                                vec = arena.GetVector(node_id);

    float norm = 0.0f;
    for (size_t i = 0; i < kVectorDim; ++i) {
        vec[i] = dist(rng);
        norm += vec[i] * vec[i];
    }

    norm = std::sqrt(norm);
    for (size_t i = 0; i < kVectorDim; ++i) {
        vec[i] /= norm;
    }

    arena.GetNode(node_id).control_block.store(
        PackControl(NodeState::kReady, EvictState::kHot, 0, 0),
        std::memory_order_relaxed);
}

}  // namespace

// -----------------------------------------------------------------------------
// BenchSearchL0_AllReady
// All 1,000 L0 slots READY. Models steady-state production workload.
// Every slot contributes a dot product; no slots are skipped.
// -----------------------------------------------------------------------------

static void BenchSearchL0_AllReady(benchmark::State& state) {
    MemoryArena  arena{ArenaConfig::BenchSearchL0()};
    std::mt19937 rng(42);

    for (size_t i = 0; i < 1024; ++i) {
        SetReady(arena, i, rng);
    }

    float* query = GenQueryVector();
    for ([[maybe_unused]] auto _ : state) {
        auto result = SearchL0(arena, query, kNow);
        benchmark::DoNotOptimize(result);
    }

    std::free(query);
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(1024));
}

BENCHMARK(BenchSearchL0_AllReady)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(10'000);

// -----------------------------------------------------------------------------
// BenchSearchL0_AllDead
// All 1,000 slots DEAD. Models a cold/empty cache.
// No scoring happens; measures pure state-check overhead.
// -----------------------------------------------------------------------------

static void BenchSearchL0_AllDead(benchmark::State& state) {
    // Default-initialized arena has all slots DEAD (mmap zero-fills).
    MemoryArena arena{ArenaConfig::BenchSearchL0()};

    float* query = GenQueryVector();
    for ([[maybe_unused]] auto _ : state) {
        auto result = SearchL0(arena, query, kNow);
        benchmark::DoNotOptimize(result);
    }

    std::free(query);
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(1024));
}

BENCHMARK(BenchSearchL0_AllDead)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(10'000);

// -----------------------------------------------------------------------------
// BenchSearchL0_Mixed
// 500 READY + 500 DEAD slots, interleaved (even=READY, odd=DEAD).
// Approximates a partially-warm cache with ongoing eviction.
// -----------------------------------------------------------------------------

static void BenchSearchL0_Mixed(benchmark::State& state) {
    MemoryArena  arena{ArenaConfig::BenchSearchL0()};
    std::mt19937 rng(99);

    for (size_t i = 0; i < 1024; ++i) {
        // Even slots are READY.
        if ((i & 1) == 0) {
            SetReady(arena, i, rng);
        }
        // Odd slots remains DEAD.
    }

    float* query = GenQueryVector();
    for ([[maybe_unused]] auto _ : state) {
        auto result = SearchL0(arena, query, kNow);
        benchmark::DoNotOptimize(result);
    }

    std::free(query);
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(1024));
}

BENCHMARK(BenchSearchL0_Mixed)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(10'000);

BENCHMARK_MAIN();
