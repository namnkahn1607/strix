// Concurrency benchmark for text embedding pipeline's encoding.
// Verify that ORT-EXT custom ops do NOT serialize concurrent calls via an
// internal mutex.
//
// If an internal mutex DOES exists, throughput will NOT scale past 1 thread,
// it will plateau or decrease as thread count increases.
// Otherwise, throughput should scale linearly (or near-linearly) up to the
// number of physical cores.

#include <benchmark/benchmark.h>

#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "inference/sentence_encoder.h"
#include "inference/simd_float_buf.h"

namespace {

// Shared `SentenceEncoder` instance, initialized once across benchmark threads.
// Mirrors production usage: one text embedding pipeline instance, N concurrent
// gRPC worker threads.
static std::unique_ptr<SentenceEncoder> shared_encoder;
static std::once_flag                   init_flag;

// SyntheticPrompt generates input text by joining the repetitions with a
// delimiter in between.
//
// `rep` must be a repeatable unit whether joining `n` copies of it always
// yield exactly `n` tokens. `delim` must be whitespace in order to avoid the
// prompt exceeding `max_input_chars_per_word` (100 by default).
std::string SyntheticPrompt(const size_t expected_tok) {
    assert(expected_tok >= 2 && "Must account for [CLS]/[SEP]");

    constexpr std::string_view kRep   = "test";
    constexpr std::string_view kDelim = " ";

    // Excluding special control tokens: `[CLS]` and `[SEP]`.
    size_t effective_tok = expected_tok - 2;

    if (effective_tok == 0) {
        return "";
    }

    std::string result;
    result.reserve(
        kRep.size() * effective_tok + kDelim.size() * (effective_tok - 1)
    );
    result += kRep;

    for (size_t i = 1; i < effective_tok; ++i) {
        result += kDelim;
        result += kRep;
    }

    return result;
}

// LoadEnv preloads env-vars `TOKENIZER_PATH` and `TRANSFORMER_PATH`
// onto caller process only once.
void LoadEnv() {
    std::call_once(init_flag, []() {
        const char* tok_path{std::getenv("TOKENIZER_PATH")};
        const char* bert_path{std::getenv("TRANSFORMER_PATH")};
        if (tok_path == nullptr || bert_path == nullptr) {
            throw std::runtime_error(
                "Env-var TOKENIZER_PATH or TRANSFORMER_PATH is not set"
            );
        }

        shared_encoder = std::make_unique<SentenceEncoder>(tok_path, bert_path);
    });
}

static void BenchConcurrentInference(
    benchmark::State& state, const size_t tok_count
) {
    LoadEnv();

    std::string  input_text = SyntheticPrompt(tok_count);
    SimdFloatBuf buf;

    for ([[maybe_unused]] auto _ : state) {
        auto encode_err = shared_encoder->Encode(
            input_text, std::span<float, kVectorDim>{buf.data(), kVectorDim}
        );

        benchmark::DoNotOptimize(encode_err);
        benchmark::ClobberMemory();
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    for (const size_t tok_count : {16U, 32U, 64U, 128U, 256U}) {
        const std::string name = "BenchConcurrentInference_" +
                                 std::to_string(tok_count) + "TokPrompt";

        benchmark::RegisterBenchmark(
            name.c_str(), [tok_count](benchmark::State& state
                          ) { BenchConcurrentInference(state, tok_count); }
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
