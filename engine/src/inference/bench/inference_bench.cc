// Author: namnkahn1607
//
// Single-threaded latency benchmarks for Embedder::Encode().
//
// Prerequisites:
//   export TOKENIZER_PATH=<absolute path to tokenizer.onnx>
//   export TRANSFORMER_PATH=<absolute path to model.onnx>

#include <benchmark/benchmark.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include "inference/inference_model.h"

namespace {

inline constexpr size_t kDim = 384;

// A realistic long prompt that perhaps reaches MAX_TOKENS (256).
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

std::unique_ptr<Embedder> glob_emb;

// GetEmbedder(): retrieves the `Embedder` global instance.
Embedder& GetEmbedder() {
    if (!glob_emb) {
        const char* tok_path  = std::getenv("TOKENIZER_PATH");
        const char* bert_path = std::getenv("TRANSFORMER_PATH");
        if (tok_path == nullptr || bert_path == nullptr) {
            throw std::runtime_error(
                "TOKENIZER_PATH or TRANSFORMER_PATH is not set"
            );
        }

        glob_emb = std::make_unique<Embedder>(tok_path, bert_path);
    }

    return *glob_emb;
}

inline void CurseYouClangYouIgnoreVariablesInAssertions(
    [[maybe_unused]] const void* ptr
) {
    return;
}

// SanityCheck()
//
// Verifies two invariants before benchmarking:
//   1. Determinism: three calls with the same prompt produce bitwise-identical
//      output vectors. Catches session state leaks between calls.
//   2. Semantic ordering: "Paris is the capital" is more similar to
//      "What is the capital" than to "How to cook pasta".
void SanityCheck() {
    auto&             emb    = GetEmbedder();
    const std::string anchor = "Paris is the capital of France.";

    auto r1 = emb.Encode(anchor);
    auto r2 = emb.Encode(anchor);
    auto r3 = emb.Encode(anchor);

    assert(r1.ok() && r2.ok() && r3.ok() && "SanityCheck: Encode failed");

    const float* v1 = r1.value().get();
    const float* v2 = r2.value().get();
    const float* v3 = r3.value().get();

    CurseYouClangYouIgnoreVariablesInAssertions(v2);
    CurseYouClangYouIgnoreVariablesInAssertions(v3);

    for (size_t i = 0; i < kDim; ++i) {
        assert(
            v1[i] == v2[i] && v2[i] == v3[i] &&
            "SanityCheck: non-deterministic output"
        );
    }

    auto r_sim  = emb.Encode("What is the capital of France?");
    auto r_diff = emb.Encode("How to cook pasta?");

    assert(r_sim.ok() && r_diff.ok() && "SanityCheck: Encode failed");

    float dot_sim  = 0.0f;
    float dot_diff = 0.0f;
    for (size_t i = 0; i < kDim; ++i) {
        dot_sim += v1[i] * r_sim.value().get()[i];
        dot_diff += v1[i] * r_diff.value().get()[i];
    }

    assert(dot_sim > dot_diff && "SanityCheck: semantic ordering violated");

    std::cout << "Sanity Check: OK. sim=" << dot_sim << " diff=" << dot_diff
              << "\n";
}

}  // namespace

// -----------------------------------------------------------------------------
// BenchEncode_ShortSingle
// Short prompt (~10 tokens). Represents the common case in production.
// SanityCheck runs once before the first iteration on the benchmark thread.
// -----------------------------------------------------------------------------

static void BenchEncode_ShortSingle(benchmark::State& state) {
    if (state.thread_index() == 0) {
        SanityCheck();
    }

    auto& emb = GetEmbedder();

    // Warm up ORT's lazy initialization paths before measuring.
    benchmark::DoNotOptimize(emb.Encode("wake up its time for school"));

    for ([[maybe_unused]] auto _ : state) {
        auto result = emb.Encode("What is the capital of France?");
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BenchEncode_ShortSingle)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

// -----------------------------------------------------------------------------
// BenchEncode_LongSingle
// Long prompt (~200 tokens, near MAX_TOKENS=256). Represents worst-case
// latency for a valid prompt. Attention is O(seq_len^2) so latency scales
// super-linearly with token count.
// -----------------------------------------------------------------------------

static void BenchEncode_LongSingle(benchmark::State& state) {
    auto& emb = GetEmbedder();
    benchmark::DoNotOptimize(emb.Encode("please dont love somebody else"));

    for ([[maybe_unused]] auto _ : state) {
        auto result = emb.Encode(kLongPrompt);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BenchEncode_LongSingle)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
