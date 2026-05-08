//
// Created by nlnk on Apr 26, 26.
//

#include <benchmark/benchmark.h>

#include <random>

#include "avx_math.hh"

constexpr uint32_t ALIGN = 32;
constexpr uint32_t DIM = 384;

static void BenchBatch4_1K(benchmark::State& state) {
    constexpr uint32_t NUM_VECTORS = 1000;
    constexpr uint32_t TOTAL_FLOATS = DIM * NUM_VECTORS;

    auto* l0_cache =
        static_cast<float*>(_mm_malloc(TOTAL_FLOATS * sizeof(float), 32));
    auto* query = static_cast<float*>(_mm_malloc(DIM * sizeof(float), 32));

    std::mt19937 gen(42);  // NOLINT(cert-msc51-cpp)
    std::uniform_real_distribution dist(-1.0f, 1.0f);

    for (int32_t i = 0; i < NUM_VECTORS; ++i) {
        l0_cache[i] = dist(gen);
    }

    for (int32_t i = 0; i < DIM; ++i) {
        query[i] = dist(gen);
    }

    for ([[maybe_unused]] auto _ : state) {
        for (uint32_t i = 0; i < NUM_VECTORS; i += 4) {
            float* node_batch = l0_cache + i * DIM;
            float scores[4];

            benchmark::DoNotOptimize(query);
            benchmark::DoNotOptimize(node_batch);

            CosineL0_Batch4(query, node_batch, scores);
            benchmark::DoNotOptimize(scores);
        }
    }

    state.SetItemsProcessed(state.iterations() * NUM_VECTORS);
    state
        .SetBytesProcessed(  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
            state.iterations() * TOTAL_FLOATS * sizeof(float));

    _mm_free(l0_cache);
    _mm_free(query);
}

BENCHMARK(BenchBatch4_1K)->Unit(benchmark::kNanosecond)->Iterations(10000);

static void BenchBatch4_20K(benchmark::State& state) {
    constexpr uint32_t NUM_VECTORS = 20000;
    constexpr uint32_t TOTAL_FLOATS = DIM * NUM_VECTORS;

    auto* l0_cache =
        static_cast<float*>(_mm_malloc(TOTAL_FLOATS * sizeof(float), ALIGN));
    auto* query = static_cast<float*>(_mm_malloc(DIM * sizeof(float), ALIGN));

    std::mt19937 gen(42);  // NOLINT(cert-msc51-cpp)
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    for (int32_t i = 0; i < TOTAL_FLOATS; ++i) l0_cache[i] = dist(gen);
    for (int32_t i = 0; i < DIM; ++i) query[i] = dist(gen);

    for ([[maybe_unused]] auto _ : state) {
        for (uint32_t i = 0; i < NUM_VECTORS; i += 4) {
            float* node_batch = l0_cache + i * DIM;
            float scores[4];

            benchmark::DoNotOptimize(query);
            benchmark::DoNotOptimize(node_batch);

            CosineL0_Batch4(query, node_batch, scores);

            benchmark::DoNotOptimize(scores);
        }
    }

    state.SetItemsProcessed(state.iterations() * NUM_VECTORS);
    state
        .SetBytesProcessed(  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
            state.iterations() * TOTAL_FLOATS * sizeof(float));

    _mm_free(l0_cache);
    _mm_free(query);
}

BENCHMARK(BenchBatch4_20K)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_MAIN();
