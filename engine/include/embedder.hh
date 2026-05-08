//
// Created by nlnk on Mar 1, 26.
//

#ifndef STRIX_ENGINE_EMBEDDER_HH
#define STRIX_ENGINE_EMBEDDER_HH

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "raii_vector.hh"

class Embedder {  // Meyers Singleton
public:
    // Remove Copy Constructor & Copy Assignment Operator
    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;

    // getInstance() now is thread-safe. If multiple calls to it are made,
    // they'll have to wait for initialization to complete.
    static Embedder& GetInstance() {
        // Only get initialization once called
        static Embedder instance;  // C++11 Magic Statics (Thread-safe local
                                   // static initialization)
        return instance;
    }

    [[nodiscard]] AlignedVector Encode(const std::string& prompt) const;

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;

    // Ort::Session has no default constructor, C++ will force construction in
    // initializer list if not declared as pointer => Use smart pointer.
    std::unique_ptr<Ort::Session> session_;

    Ort::MemoryInfo mem_info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    Embedder();
    ~Embedder() = default;
};

#endif  // STRIX_ENGINE_EMBEDDER_HH
