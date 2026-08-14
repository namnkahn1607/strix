// AVX2 + FMA accelerated dot product kernel micro-benchmark: isolate and
// measure the performance delta between 2 dot product kernel APIs.
//
// To make the delta attributable not to distinction in memory layout, cache
// behavior or data churn, both 2 benchmarks run against an identical harness,
// which is a contiguous 2'048-vector buffer.

#include <benchmark/benchmark.h>
#include <mm_malloc.h>

#include <new>
#include <random>

#include "index/avx2_kernel.h"

namespace {

constexpr uint32_t kNumVectors = 2'048u;  // 3072 KiB buf size

// AllocVectorBuf allocates a contiguous array of `num_vectors`, then generate
// arbitrary values for every dimension of each vector using `seed`.
float* AllocVectorBuf(uint32_t num_vectors, uint32_t seed = 0xABC) {
    constexpr uint32_t kAlign    = 32;
    const uint32_t     total_dim = num_vectors * kVectorDim;

    auto* raw = _mm_malloc(total_dim * sizeof(float), kAlign);
    if (raw == nullptr) {
        throw std::bad_alloc();
    }

    auto* buf = static_cast<float*>(raw);

    std::mt19937                   gen(seed);
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    for (uint32_t i = 0; i < total_dim; ++i) {
        buf[i] = dist(gen);
    }

    return buf;
}

}  // namespace

// BenchDotProduct_Contiguous benchmarks `DotProductContiguousBatch()`: consumes
// a vector batch per call via a single base pointer and fixed dimension stride.
static void BenchDotProduct_Contiguous(benchmark::State& state) {
    auto* buf   = AllocVectorBuf(kNumVectors);
    auto* query = AllocVectorBuf(1);

    for ([[maybe_unused]] auto _ : state) {
        for (uint32_t i = 0; i < kNumVectors; i += kBatchSize) {
            float* batch_start = buf + i * kVectorDim;

            float scores[kBatchSize] = {};

            benchmark::DoNotOptimize(query);
            benchmark::DoNotOptimize(batch_start);
            DotProductContiguousBatch(query, batch_start, scores);
            benchmark::DoNotOptimize(scores);
        }
    }

    state.SetItemsProcessed(state.iterations() * kNumVectors);

    _mm_free(query);
    _mm_free(buf);
}

BENCHMARK(BenchDotProduct_Contiguous)
    ->Unit(benchmark::kNanosecond)
    // ->Repetitions(10)
    // ->ReportAggregatesOnly(true)
    ->Iterations(1'000u);

// BenchDotProduct_Discrete benchmarks `DotProductDiscreteBatch()`: consumes a
// vector batch per call via independent pointers.
static void BenchDotProduct_Discrete(benchmark::State& state) {
    auto* buf   = AllocVectorBuf(kNumVectors);
    auto* query = AllocVectorBuf(1);

    for ([[maybe_unused]] auto _ : state) {
        for (uint32_t i = 0; i < kNumVectors; i += kBatchSize) {
            float* v0 = buf + i * kVectorDim;
            float* v1 = v0 + kVectorDim;
            float* v2 = v0 + 2 * kVectorDim;
            float* v3 = v0 + 3 * kVectorDim;

            float scores[kBatchSize] = {};

            benchmark::DoNotOptimize(query);
            benchmark::DoNotOptimize(v0);
            benchmark::DoNotOptimize(v1);
            benchmark::DoNotOptimize(v2);
            benchmark::DoNotOptimize(v3);
            DotProductDiscreteBatch(query, v0, v1, v2, v3, scores);
            benchmark::DoNotOptimize(scores);
        }
    }

    state.SetItemsProcessed(state.iterations() * kNumVectors);

    _mm_free(query);
    _mm_free(buf);
}

BENCHMARK(BenchDotProduct_Discrete)
    ->Unit(benchmark::kNanosecond)
    // ->Repetitions(10)
    // ->ReportAggregatesOnly(true)
    ->Iterations(1'000u);

BENCHMARK_MAIN();
