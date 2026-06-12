//
// bench/customops_bench.cpp
//
// Concurrency benchmark for Embedder::Encode() - specifically written to
// verify that onnxruntime-extensions custom ops do NOT serialize concurrent
// inference calls via an internal mutex.
//
// If a global mutex exists, throughput will NOT scale past 1 thread -
// items/sec will plateau or decrease as thread count increases.
// If there is no mutex, throughput should scale linearly (or near-linearly)
// up to the number of physical cores.
//
// Prerequisites:
//   export TOKENIZER_PATH=<absolute path to tokenizer.onnx>
//   export TRANSFORMER_PATH=<absolute path to model.onnx>
//

#include <benchmark/benchmark.h>

#include <memory>
#include <mutex>

#include "inference.hh"

namespace {

// Shared 'Embedder' instance, initialized once across all benchmark threads.
// Mirrors production usage: one Embedder, N concurrent gRPC handler threads.
static std::unique_ptr<Embedder> shared_emb;
static std::once_flag            init_flag;

void LoadEnv() {
    std::call_once(init_flag, []() {
        const char* tok_path{std::getenv("TOKENIZER_PATH")};
        const char* bert_path{std::getenv("TRANSFORMER_PATH")};
        if (tok_path == nullptr || bert_path == nullptr) {
            throw std::runtime_error(
                "Env-var TOKENIZER_PATH or TRANSFORMER_PATH is not set");
        }

        shared_emb = std::make_unique<Embedder>(tok_path, bert_path);
    });
}

};  // namespace

// ---------------------------------------------------------------------------
// BenchConcurrentInference
// Runs at thread counts: 1, 2, 4, 8, 50, 100.
// UseRealTime() measures wall-clock time so that thread serialization
// (mutex contention) shows up as reduced items/sec, not reduced CPU time.
// MinTime(3.0) ensures enough iterations for stable measurements at high
// thread counts where individual calls are fast.
// ---------------------------------------------------------------------------

static void BenchConcurrentInference(benchmark::State& state) {
    LoadEnv();

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
