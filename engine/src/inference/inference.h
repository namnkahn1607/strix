// Author: namnkahn1607
//
// Result<T, E> backport, EncodeError enum, and Embedder class.
// Embedder owns two ONNX Runtime sessions (tokenizer + transformer)
// and produces 32-byte aligned embedding vectors via Encode().

#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <utility>
#include <variant>

#include "aligned_vec.h"

// Result<T, E>
//
// Lightweight `std::expected<T, E>` backport (a C++23 feature).
//
// Encodes success (T) or a predictable, non-fatal failure (E) without
// exceptions. Use Result for errors that callers are expected to handle
// at the call site (e.g. validation failures). Throw for unrecoverable
// failures (e.g. ORT session crash) where the process cannot continue.
template <typename T, typename E>
struct Result {
    std::variant<T, E> data;

    bool ok() const noexcept {
        return std::holds_alternative<T>(data);
    }

    T& value() {
        return std::get<T>(data);
    }
    const T& value() const {
        return std::get<T>(data);
    }

    E& error() {
        return std::get<E>(data);
    }
    const E& error() const {
        return std::get<E>(data);
    }

    static Result Ok(T val) {
        return Result{
            std::variant<T, E>{std::in_place_index<0>, std::move(val)}};
    }

    static Result Err(E err) {
        return Result{
            std::variant<T, E>{std::in_place_index<1>, std::move(err)}};
    }
};

// EncodeError: Predictable, non-fatal failures from `Embedder::Encode()`.
// Callers are expected to translate these into gRPC status codes.
enum class EncodeError {
    kTokenLimitExceeded,  // Input exceeds the model's maximum token count.
    kDegeneratedVector,   // Output vector is zero-norm; unusable for search.
};

// Embedder
//
// Owns two ONNX Runtime sessions: a tokenizer and a transformer (BERT).
// Its `Encode()` runs both sessions sequentially to produce a 32-byte
// aligned embedding vector for a given prompt.
//
// Thread safety: `Encode()` is safe to call concurrently from multiple threads.
// ORT sessions are stateless per-run provided `Ort::SessionOptions` is not
// shared across calls. `env_` and `options_` are read-only after construction.
//
// Ownership: construct once, pass by const reference to consumers.
// Not copyable, not movable.
class Embedder {
public:
    // Initialises both ORT sessions from the given model file paths.
    // Throws `std::runtime_error` if either session fails to load.
    explicit Embedder(const char* tok_path, const char* bert_path);

    Embedder(const Embedder&)            = delete;
    Embedder& operator=(const Embedder&) = delete;
    Embedder(Embedder&&)                 = delete;
    Embedder& operator=(Embedder&&)      = delete;

    // Tokenises `prompt` and runs inference, returning an `AlignedVec` on
    // success. Returns `EncodeError` on predictable failures.
    // Throws `std::runtime_error` on session-level failures.
    Result<AlignedVec, EncodeError> Encode(const std::string& prompt) const;

private:
    Ort::Env            env_;
    Ort::SessionOptions options_;

    // Ort::Session has no default constructor; unique_ptr allows deferred
    // construction inside the member initialiser list.
    std::unique_ptr<Ort::Session> tok_session_;
    std::unique_ptr<Ort::Session> bert_session_;
};
