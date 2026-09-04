// Concurrency benchmark for text embedding pipeline's encoding: verifies that
// ORT-EXT custom ops do NOT serialize concurrent calls via an internal mutex.
//
// If an internal mutex DOES exists, throughput will NOT scale past one thread,
// it will plateau or decrease as thread count increases.
// Otherwise, throughput scales (near-)linearly up to the physical cores count.

#include <absl/log/check.h>
#include <benchmark/benchmark.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

#include "inference/sentence_encoder.h"
#include "inference/simd_float_buf.h"

using namespace strix::inference;

namespace {

const char* kTokenizerVar   = "TOKENIZER_PATH";
const char* kTransformerVar = "TRANSFORMER_PATH";

constexpr uint32_t kTokCounts[5] = {16, 32, 64, 128, 256};

// Generates input prompt by joining the repetitions with delimiter in between.
//
// Repetition is a repeatable unit whether joining N copies of it always yields
// exactly N tokens. Delimiter must be whitespace in order to avoid the prompt
// exceeding `max_input_chars_per_word` (100 by default).
std::string SyntheticPrompt(size_t exp_tok) {
    CHECK(exp_tok >= 2) << "Expected token counts must account for [CLS]/[SEP]";

    constexpr std::string_view kRep   = "test";
    constexpr std::string_view kDelim = " ";

    // Excluding special control tokens: `[CLS]` and `[SEP]`.
    size_t real_tok = exp_tok - 2;
    if (real_tok == 0) {
        return "";
    }

    std::string result;
    result.reserve(kRep.size() * real_tok + kDelim.size() * (real_tok - 1));
    result += kRep;
    for (size_t i = 1; i < real_tok; ++i) {
        result += kDelim;
        result += kRep;
    }

    return result;
}

static std::unique_ptr<SentenceEncoder> shared_encoder;

static void BenchConcurrentInference(benchmark::State& state, size_t exp_tok) {
    std::string input_text = SyntheticPrompt(exp_tok);

    SimdFloatBuf buf;
    for ([[maybe_unused]] auto _ : state) {
        auto encode_err = shared_encoder->Encode(input_text, buf);
        benchmark::DoNotOptimize(encode_err);
        benchmark::ClobberMemory();
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    // NOLINTBEGIN(concurrency-mt-unsafe)
    const char* tok_path  = std::getenv(kTokenizerVar);
    const char* bert_path = std::getenv(kTransformerVar);
    // NOLINTEND(concurrency-mt-unsafe)

    CHECK(tok_path != nullptr && bert_path != nullptr);
    shared_encoder = std::make_unique<SentenceEncoder>(tok_path, bert_path);

    for (const size_t exp_tok : kTokCounts) {
        const std::string name =
            "BenchConcurrentInference_" + std::to_string(exp_tok) + "TokPrompt";

        benchmark::RegisterBenchmark(
            name.c_str(), [exp_tok](benchmark::State& state
                          ) { BenchConcurrentInference(state, exp_tok); }
        )
            ->Unit(benchmark::kMicrosecond)
            ->Threads(1)
            ->Threads(2)
            ->Threads(3)
            ->Threads(4)
            ->Threads(8)
            ->UseRealTime()
            ->MinTime(3.0);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
