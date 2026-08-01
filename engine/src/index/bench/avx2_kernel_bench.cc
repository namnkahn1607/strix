// Author: namnkahn1607
//
// Microbenchmarks for DotProductBatch - the AVX2 SIMD kernel that computes
// dot products between one query vector and a batch of 4 node vectors.
//
// Build with: -O3 -mavx2 -mfma
// Reported metrics per benchmark:
//   items_per_second : number of vectors scored per second
//   bytes_per_second : memory bandwidth consumed (read-only, no write)

#include <benchmark/benchmark.h>

#include <random>

#include "index/avx2_kernel.h"

namespace {

inline constexpr uint32_t kAlign = 32;
inline constexpr uint32_t kDim   = 384;

// FillRandom(): fills `count` floats with random uniform values in [-1, 1].
void FillRandom(float* dst, const int32_t count, const uint64_t seed = 42) {
    std::mt19937                          gen(seed);  // NOLINT(cert-msc51-cpp)
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int32_t i = 0; i < count; ++i) {
        dst[i] = dist(gen);
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// BenchBatch4_1K
// Simulates a full L0 Buffer scan: 1,000 vectors x 384 dimensions.
// Expected to run entirely from L2/L3 cache (~1.5 MB working set).
// -----------------------------------------------------------------------------

static void BenchBatch4_1K(benchmark::State& state) {
    constexpr uint32_t kNumVectors  = 1'000;
    constexpr uint32_t kTotalFloats = kDim * kNumVectors;

    auto* l0_cache =
        static_cast<float*>(_mm_malloc(kTotalFloats * sizeof(float), kAlign));
    auto* query = static_cast<float*>(_mm_malloc(kDim * sizeof(float), kAlign));

    FillRandom(l0_cache, kTotalFloats, 42);
    FillRandom(query, kDim, 99);

    for ([[maybe_unused]] auto _ : state) {
        for (uint32_t i = 0; i < kNumVectors; i += 4) {
            float* node_batch = l0_cache + i * kDim;
            float  scores[4]  = {};

            benchmark::DoNotOptimize(query);
            benchmark::DoNotOptimize(node_batch);
            DotProductBatch(query, node_batch, scores);
            benchmark::DoNotOptimize(scores);
        }
    }

    state.SetItemsProcessed(state.iterations() * kNumVectors);
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(kTotalFloats * sizeof(float)));

    _mm_free(l0_cache);
    _mm_free(query);
}

BENCHMARK(BenchBatch4_1K)->Unit(benchmark::kNanosecond)->Iterations(10'000);

// -----------------------------------------------------------------------------
// BenchBatch4_20K
// Cache-pressure workload: 20,000 vectors ~ 30 MB working set.
// Exceeds L3 cache on most CPUs, forcing fetches from main memory.
// Models worst-case throughput as the index grows toward L1 buffer size.
// -----------------------------------------------------------------------------

static void BenchBatch4_20K(benchmark::State& state) {
    constexpr uint32_t kNumVectors  = 20000;
    constexpr uint32_t kTotalFloats = kDim * kNumVectors;

    auto* l0_cache =
        static_cast<float*>(_mm_malloc(kTotalFloats * sizeof(float), kAlign));
    auto* query = static_cast<float*>(_mm_malloc(kDim * sizeof(float), kAlign));

    FillRandom(l0_cache, static_cast<int32_t>(kTotalFloats), 42);
    FillRandom(query, kDim, 99);

    for ([[maybe_unused]] auto _ : state) {
        for (uint32_t i = 0; i < kNumVectors; i += 4) {
            float* node_batch = l0_cache + i * kDim;
            float  scores[4]  = {};

            benchmark::DoNotOptimize(query);
            benchmark::DoNotOptimize(node_batch);
            DotProductBatch(query, node_batch, scores);
            benchmark::DoNotOptimize(scores);
        }
    }

    state.SetItemsProcessed(state.iterations() * kNumVectors);
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(kTotalFloats * sizeof(float)));

    _mm_free(l0_cache);
    _mm_free(query);
}

BENCHMARK(BenchBatch4_20K)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_MAIN();
