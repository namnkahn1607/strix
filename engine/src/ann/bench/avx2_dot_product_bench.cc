// AVX2 + FMA accelerated dot product kernel micro-benchmark: estimates
// performance delta between 2 computing APIs.

#include <benchmark/benchmark.h>
#include <mm_malloc.h>

#include <new>
#include <random>

#include "ann/avx2_dot_product.h"

namespace ann       = strix::ann;
namespace inference = strix::inference;

namespace {

// Both benchmark run against a contiguous buffer of 2048 vectors.
// Get the performance delta attributable not to memory layout, cache behavior
// distinction or data churn.
constexpr uint32_t kNumVectors = 2'048u;

float* AllocVecBuf(uint32_t num_vectors, uint32_t seed = 0xABC) {
    constexpr uint32_t kAlign    = 32;
    const uint32_t     kTotalDim = num_vectors * inference::kVectorDim;

    auto* raw = _mm_malloc(kTotalDim * sizeof(float), kAlign);
    if (raw == nullptr) {
        throw std::bad_alloc();
    }

    auto* buf = static_cast<float*>(raw);

    std::mt19937                   gen(seed);
    std::uniform_real_distribution dist{-1.0f, 1.0f};
    for (uint32_t i = 0; i < kTotalDim; ++i) {
        buf[i] = dist(gen);
    }

    return buf;
}

}  // namespace

static void BenchDotProduct_Contiguous(benchmark::State& state) {
    auto* buf   = AllocVecBuf(kNumVectors);
    auto* query = AllocVecBuf(1);

    for ([[maybe_unused]] auto _ : state) {
        for (uint32_t i = 0; i < kNumVectors; i += ann::kBatchSize) {
            float* batch_start = buf + i * inference::kVectorDim;

            float scores[ann::kBatchSize] = {};

            benchmark::DoNotOptimize(query);
            benchmark::DoNotOptimize(batch_start);
            ann::DotProductContiguousBatch(query, batch_start, scores);
            benchmark::DoNotOptimize(scores);
        }
    }

    state.SetItemsProcessed(state.iterations() * kNumVectors);
    _mm_free(query);
    _mm_free(buf);
}

BENCHMARK(BenchDotProduct_Contiguous)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(1'000u);

static void BenchDotProduct_Discrete(benchmark::State& state) {
    auto* buf   = AllocVecBuf(kNumVectors);
    auto* query = AllocVecBuf(1);

    for ([[maybe_unused]] auto _ : state) {
        for (uint32_t i = 0; i < kNumVectors; i += ann::kBatchSize) {
            float* v0 = buf + i * inference::kVectorDim;
            float* v1 = v0 + inference::kVectorDim;
            float* v2 = v1 + inference::kVectorDim;
            float* v3 = v2 + inference::kVectorDim;

            float scores[ann::kBatchSize] = {};

            benchmark::DoNotOptimize(query);
            benchmark::DoNotOptimize(v0);
            benchmark::DoNotOptimize(v1);
            benchmark::DoNotOptimize(v2);
            benchmark::DoNotOptimize(v3);
            ann::DotProductDiscreteBatch(query, v0, v1, v2, v3, scores);
            benchmark::DoNotOptimize(scores);
        }
    }

    state.SetItemsProcessed(state.iterations() * kNumVectors);
    _mm_free(query);
    _mm_free(buf);
}

BENCHMARK(BenchDotProduct_Discrete)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(1'000u);

BENCHMARK_MAIN();
