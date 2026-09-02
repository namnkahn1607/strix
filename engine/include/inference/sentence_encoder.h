// Text embedding pipeline.

#pragma once

#include <onnxruntime_cxx_api.h>

#include <optional>
#include <string>

#include "inference/simd_float_buf.h"

namespace strix::inference {

// Predictable, non-fatal encoding failures.
// Callers are expected to map these into gRPC status codes.
enum class EncodeError {
    kTokenLimitExceeded,  // Input exceeds the model's maximum token count.
    kDegeneratedVector,   // Output vector is zero-norm; unusable for searching.
};

// Consists of 2 ORT sessions: a tokenizer and a BERT transformer.
// Both sessions run sequentially to produce embedding vector of given prompt.
//
// Safe to invoke by multiple threads, as ORT sessions are stateless per-run
// provided `Ort::SessionOptions` is not shared across calls.
class SentenceEncoder final {
public:
    explicit SentenceEncoder(const char* tok_path, const char* bert_path);

    SentenceEncoder(const SentenceEncoder&)            = delete;
    SentenceEncoder& operator=(const SentenceEncoder&) = delete;
    SentenceEncoder(SentenceEncoder&&)                 = delete;
    SentenceEncoder& operator=(SentenceEncoder&&)      = delete;

    // Performs vectorization on a prompt string and writes the result vector
    // values to `out`. Returns `EncodeError` on predictable failures, throws on
    // session-level failures.
    std::optional<EncodeError> Encode(
        const std::string& prompt, SimdFloatBuf& out
    ) const;

private:
    static Ort::SessionOptions InitOptions();

    Ort::Env            env_;
    Ort::SessionOptions options_;

    mutable Ort::Session tok_session_;   // Tokenizer
    mutable Ort::Session bert_session_;  // Transformer
};

}  // namespace strix::inference
