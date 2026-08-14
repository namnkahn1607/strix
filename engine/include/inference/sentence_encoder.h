// Text embedding pipeline declaration and its encoding failure enum.

#pragma once

#include <onnxruntime_cxx_api.h>

#include <optional>
#include <span>
#include <string>

#include "info.h"

// EncodeError lists all predictable, non-fatal encoding failures. Callers are
// expected to translate these into gRPC status codes.
enum class EncodeError {
    kTokenLimitExceeded,  // Input exceeds the model's maximum token count.
    kDegeneratedVector,   // Output vector is zero-norm; unusable for search.
};

// SentenceEncoder owns 2 ORT sessions: a tokenizer and a transformer (BERT).
// Both sessions are run sequentially to produce a 32-byte aligned embedding
// vector for a given prompt.
//
// Concurrency: Safe to call for inferencing from multiple threads. ORT sessions
// are stateless per-run provided `Ort::SessionOptions` is not shared across
// calls. `env_` and `options_` are read-only after construction.
//
// Ownership: construct once, pass by const reference to consumers.
class SentenceEncoder {
public:
    explicit SentenceEncoder(const char* tok_path, const char* bert_path);

    SentenceEncoder(const SentenceEncoder&)            = delete;
    SentenceEncoder& operator=(const SentenceEncoder&) = delete;
    SentenceEncoder(SentenceEncoder&&)                 = delete;
    SentenceEncoder& operator=(SentenceEncoder&&)      = delete;

    // Encode performs vectorization on a specified `prompt` string and writes
    // result to `out` (`out` must be 32-byte aligned). Returns `EncodeError` on
    // predictable failures.
    // Throws `std::runtime_error` on session-level failures.
    std::optional<EncodeError> Encode(
        const std::string& prompt, std::span<float, kVectorDim> out
    ) const;

private:
    // InitOptions completes `Ort::SessionOptions` contruction before handling
    // to `SentenEncoder` ctor to initialize any `Ort::Session`.
    static Ort::SessionOptions InitOptions();

    Ort::Env            env_;
    Ort::SessionOptions options_;

    mutable Ort::Session tok_session_;   // Tokenizer
    mutable Ort::Session bert_session_;  // Transformer
};
