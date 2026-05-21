#include <benchmark/benchmark.h>

#include <memory>
#include <mutex>

#include "embedder.hh"

static std::unique_ptr<Embedder> shared_emb;
static std::once_flag init_flag;

static void BenchConcurrentInference(benchmark::State& state) {
    std::call_once(init_flag, []() {
        const char* tok_path{std::getenv("TOKENIZER_PATH")};
        const char* bert_path{std::getenv("TRANSFORMER_PATH")};
        if (tok_path == nullptr || bert_path == nullptr) {
            throw std::runtime_error(
                "Env-var TOKENIZER_PATH or TRANSFORMER_PATH is not set");
        }

        shared_emb = std::make_unique<Embedder>(tok_path, bert_path);
    });

    const std::string prompt = "The quick brown fox jumps over the lazy dog";

    for ([[maybe_unused]] auto _ : state) {
        auto vec = shared_emb->Encode(prompt);

        benchmark::DoNotOptimize(vec);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BenchConcurrentInference)
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(50)
    ->Threads(100)
    ->UseRealTime()
    ->MinTime(3.0);

BENCHMARK_MAIN();
