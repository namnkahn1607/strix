//
// inference/inference.hh
//

#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <variant>

#include "aligned_vec.hh"

// all-MiniLM-L6-v2 truncates on input text longer than 256 word pieces.
constexpr size_t MAX_TOKENS = 256;

// --- Result<T, E> ---
// Lightweight std::expected backport used by Encode().
// Rule: use Result for PREDICTABLE failures (validation errors).
//       Use throw for UNRECOVERABLE failures (ORT session crash).
template <typename T, typename E>
struct Result {
    std::variant<T, E> data;

    bool ok() const noexcept { return std::holds_alternative<T>(data); }

    T&       value() { return std::get<T>(data); }
    const T& value() const { return std::get<T>(data); }

    E&       error() { return std::get<E>(data); }
    const E& error() const { return std::get<E>(data); }

    static Result Ok(T val) {
        return Result{
            std::variant<T, E>{std::in_place_index<0>, std::move(val)}};
    }

    static Result Err(E err) {
        return Result{
            std::variant<T, E>{std::in_place_index<1>, std::move(err)}};
    }
};

// --- EncodeError ---
// Predictable, non-fatal failures from Encode().
enum class EncodeError {
    TOKEN_LIMIT_EXCEEDED,  // Prompt exceeding MAX_TOKENS
    DEGENERATED_VECTOR,
};

// --- Embedder ---
// Has two ONNX sessions: tokenizer + transformer.
// Thread-safe for concurrent Encode() calls (ORT sessions are stateless
// per-run as long as RunOptions is not shared).
class Embedder {
public:
    explicit Embedder(const char* tok_path, const char* bert_path);

    // No Copy/Move/Assignment semantics
    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;
    Embedder(Embedder&&) = delete;
    Embedder& operator=(Embedder&&) = delete;

    Result<AlignedVec, EncodeError> Encode(const std::string& prompt) const;

private:
    Ort::Env            env;
    Ort::SessionOptions options;

    // Ort::Session has no default constructor.
    // Declared as unique_ptr to allow construction in initializer list.
    std::unique_ptr<Ort::Session> tok_session;
    std::unique_ptr<Ort::Session> bert_session;
};
