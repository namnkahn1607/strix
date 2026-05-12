//
// Created by nlnk on May 9, 26.
//

#include <benchmark/benchmark.h>

#include <iostream>

#include "embedder.hh"

constexpr size_t DIM = 384;

static void SanityCheck() {
    const auto& emb = Embedder::GetInstance();

    // Bit-identical results for multiple same prompts
    const auto v1 = emb.Encode("Paris is the capital of France.");
    const auto v2 = emb.Encode("Paris is the capital of France.");
    const auto v3 = emb.Encode("Paris is the capital of France.");

    for (size_t i = 0; i < DIM; ++i) {
        assert(v1.get()[i] == v2.get()[i] && v2.get()[i] == v3.get()[i]);
    }

    // Encoding semantically similar prompt - Cosine must be high
    const auto sim = emb.Encode("What is the capital of France?");
    const auto diff = emb.Encode("How to cook pasta?");

    float dot_sim = 0.0f;
    float dot_diff = 0.0f;
    for (size_t i = 0; i < DIM; ++i) {
        dot_sim += v1.get()[i] * sim.get()[i];
        dot_diff += v1.get()[i] * diff.get()[i];
    }

    // "Paris is capital" vs "What is capital" must be
    // semantically closer than "cook pasta"
    assert(dot_sim > dot_diff);
    std::cout << "Sanity OK - sim=" << dot_sim << " diff=" << dot_diff << "\n";
}

static void BenchEncode_ShortSingle(benchmark::State& state) {
    SanityCheck();

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
    SanityCheck();

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
