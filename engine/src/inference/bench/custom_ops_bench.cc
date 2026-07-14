// Author: namnkahn1607
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

#include <benchmark/benchmark.h>

#include <memory>
#include <mutex>

#include "inference_model.h"

namespace {

// A realistic long prompt that perhaps reaching MAX_TOKENS (256).
inline constexpr const char* kLongPrompt =
    "Artificial intelligence has transformed many industries over the past "
    "decade. From natural language processing to computer vision, machine "
    "learning models are now capable of performing tasks that once required "
    "significant human expertise. Large language models in particular have "
    "demonstrated remarkable abilities in text generation, summarization, "
    "question answering, and code completion. However, challenges remain in "
    "areas such as reasoning, factual accuracy, and computational efficiency. "
    "Researchers continue to explore new architectures and training techniques "
    "to address these limitations and push the boundaries of what is possible.";

// A typical short prompt that reaches ~10 tokens.
inline constexpr const char* kShortPrompt =
    "The quick brown fox jumps over the lazy dog";

// Shared 'Embedder' instance, initialized once across all benchmark threads.
// Mirrors production usage: one Embedder, N concurrent gRPC handler threads.
static std::unique_ptr<Embedder> shared_emb;
static std::once_flag            init_flag;

// LoadEnv(): load environment variables TOKENIZER_PATH and TRANSFORMER_PATH
// onto process only once.
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

}  // namespace

// -----------------------------------------------------------------------------
// BenchConcurrentInference_ShortPrompt
// Runs at thread counts: 1, 2, 4, 8, 50, 100.
// UseRealTime() measures wall-clock time so that thread serialization
// (mutex contention) shows up as reduced items/sec, not reduced CPU time.
// MinTime(3.0) ensures enough iterations for stable measurements at high
// thread counts where individual calls are fast.
// -----------------------------------------------------------------------------

static void BenchConcurrentInference_ShortPrompt(benchmark::State& state) {
    LoadEnv();

    for ([[maybe_unused]] auto _ : state) {
        auto vec = shared_emb->Encode(kShortPrompt);

        benchmark::DoNotOptimize(vec);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BenchConcurrentInference_ShortPrompt)
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(50)
    ->Threads(100)
    ->UseRealTime()
    ->MinTime(3.0);

// -----------------------------------------------------------------------------
// BenchConcurrentInference_LongPrompt
// Runs at thread counts: 2, 4, 8, and 16.
// Reveals the worst-case bottleneck if the Inference model receives
// more (really) longer prompts than usual.
// Still uses the same UseRealTime() and MinTime(3.0).
// -----------------------------------------------------------------------------

static void BenchConcurrentInference_LongPrompt(benchmark::State& state) {
    LoadEnv();

    for ([[maybe_unused]] auto _ : state) {
        auto vec = shared_emb->Encode(kLongPrompt);

        benchmark::DoNotOptimize(vec);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BenchConcurrentInference_LongPrompt)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(16)
    ->UseRealTime()
    ->MinTime(3.0);

BENCHMARK_MAIN();
