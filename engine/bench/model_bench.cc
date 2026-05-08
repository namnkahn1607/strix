//
// Created by nlnk on May 9, 26.
//

#include <benchmark/benchmark.h>

#include "embedder.hh"

static void BenchEncode_ShortSingle(benchmark::State& state) {
    const auto& emb = Embedder::GetInstance();
    benchmark::DoNotOptimize(emb.Encode("warmup"));

    for ([[maybe_unused]] auto _ : state) {
        auto vec = emb.Encode("What is the capital of France?");
        benchmark::DoNotOptimize(vec);
    }
}

BENCHMARK(BenchEncode_ShortSingle)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

static void BenchEncode_LongSingle(benchmark::State& state) {
    const auto& emb = Embedder::GetInstance();
    benchmark::DoNotOptimize(emb.Encode("warmup"));

    const std::string long_prompt(1000, 'x');
    for ([[maybe_unused]] auto _ : state) {
        auto vec = emb.Encode(long_prompt);
        benchmark::DoNotOptimize(vec);
    }
}

BENCHMARK(BenchEncode_LongSingle)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
