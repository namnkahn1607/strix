//
// Created by nlnk on Mar 1, 26.
//

#ifndef STRIX_ENGINE_EMBEDDER_HH
#define STRIX_ENGINE_EMBEDDER_HH

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "raii_vector.hh"

constexpr size_t MAX_TOKENS = 256;

class TokenLimitException : public std::runtime_error {
public:
    explicit TokenLimitException(const std::string& msg)
        : std::runtime_error(msg) {}
};

class Embedder {
public:
    explicit Embedder(const char* tok_path, const char* bert_path);

    // Disallow copy/move/assignment semantics
    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;
    Embedder(Embedder&&) = delete;
    Embedder& operator=(Embedder&&) = delete;

    [[nodiscard]] AlignedVector Encode(const std::string& prompt) const;

private:
    Ort::Env env;
    Ort::SessionOptions session_options;

    // Ort::Session has no default constructor, C++ will force construction in
    // initializer list if not declared as pointer => Use smart pointer.
    std::unique_ptr<Ort::Session> tok_session;
    std::unique_ptr<Ort::Session> bert_session;
};

#endif  // STRIX_ENGINE_EMBEDDER_HH
