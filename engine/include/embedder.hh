//
// Created by nlnk on Mar 1, 26.
//

#ifndef STRIX_ENGINE_EMBEDDER_HH
#define STRIX_ENGINE_EMBEDDER_HH

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "raii_vector.hh"

class Embedder {
public:
    explicit Embedder(const char* model_path);

    // Disallow copy/move/assignment semantics
    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;
    Embedder(Embedder&&) = delete;
    Embedder& operator=(Embedder&&) = delete;

    [[nodiscard]] AlignedVector Encode(const std::string& prompt) const;

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;

    // Ort::Session has no default constructor, C++ will force construction in
    // initializer list if not declared as pointer => Use smart pointer.
    std::unique_ptr<Ort::Session> session_;
};

#endif  // STRIX_ENGINE_EMBEDDER_HH
