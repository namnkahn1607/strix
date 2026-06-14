//
// inference/inference.cc
//

#include "inference.hh"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "aligned_vec.hh"
#include "constants.hh"

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

Embedder::Embedder(const char* tok_path, const char* bert_path)
    : env{Ort::Env(ORT_LOGGING_LEVEL_ERROR, "onnx-env")}
    , options{Ort::SessionOptions()} {
    // Highest level of graph optimization
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    options.SetInterOpNumThreads(1);
    options.SetIntraOpNumThreads(1);

    options.EnableOrtCustomOps();

    tok_session = std::make_unique<Ort::Session>(env, tok_path, options);
    bert_session = std::make_unique<Ort::Session>(env, bert_path, options);
}

// ------------------------------------------------------------
// Vectorization
// ------------------------------------------------------------

Result<AlignedVec, EncodeError> Embedder::Encode(
    const std::string& prompt) const {
    const Ort::AllocatorWithDefaultOptions allocator;

    // ------------------------------------------------------------
    // PHASE 1: Tokenization
    // ------------------------------------------------------------

    const std::vector<int64_t> input_shape{1};
    const char*                input_string = prompt.c_str();

    Ort::Value text_tensor = Ort::Value::CreateTensor(
        allocator, input_shape.data(), input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    text_tensor.FillStringTensor(&input_string, 1);

    const char* tok_input_names[]{"text"};
    const char* tok_output_names[]{"input_ids", "attention_mask",
                                   "token_type_ids"};

    auto tok_outputs =
        tok_session->Run(Ort::RunOptions{nullptr}, tok_input_names,
                         &text_tensor, 1, tok_output_names, 3);

    // ------------------------------------------------------------
    // PHASE 2: Validation
    // ------------------------------------------------------------

    // 'input_ids' from Tokenizer has a tensor format of int64[?]
    const Ort::Value& input_ids_tensor = tok_outputs[0];
    const auto shape = input_ids_tensor.GetTensorTypeAndShapeInfo().GetShape();

    // '?' means Dynamic Axes, representing sequence length
    const auto seq_length = static_cast<size_t>(shape[0]);
    if (seq_length > MAX_TOKENS) {
        return Result<AlignedVec, EncodeError>::Err(
            EncodeError::TOKEN_LIMIT_EXCEEDED);
    }

    // ------------------------------------------------------------
    // PHASE 3: Transformer inferencing
    // ------------------------------------------------------------

    const char* bert_input_names[]{"input_ids", "attention_mask",
                                   "token_type_ids"};
    const char* bert_output_names[]{"last_hidden_state"};

    // 'input_ids' needs to be of int64[batch_size, seq_length]
    // -> Format a Rank-2 shape [1, seq_length]
    const std::array<int64_t, 2> bert_input_shape{
        1, static_cast<int64_t>(seq_length)};

    // Retrieve Memory Allocator from old Tensor
    const auto mem_info = input_ids_tensor.GetTensorMemoryInfo();

    // Create new View(s) pointing to new data
    std::vector<Ort::Value> bert_inputs;
    bert_inputs.reserve(3);
    for (size_t i = 0; i < 3; ++i) {
        auto* raw_data = tok_outputs[i].GetTensorMutableData<int64_t>();
        bert_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem_info, raw_data, seq_length, bert_input_shape.data(),
            bert_input_shape.size()));
    }

    const auto bert_outputs =
        bert_session->Run(Ort::RunOptions{nullptr}, bert_input_names,
                          bert_inputs.data(), 3, bert_output_names, 1);

    // ------------------------------------------------------------
    // PHASE 4: Shape validation
    // Check for unrecoverable errors
    // ------------------------------------------------------------

    const Ort::Value& output_tensor = bert_outputs.front();

    // Output shape is always [1, N, 384].
    const auto output_shape =
        output_tensor.GetTensorTypeAndShapeInfo().GetShape();

    if (output_shape.size() != 3) {
        throw std::runtime_error("Unexpected output rank");
    }

    if (output_shape[0] != 1) {
        throw std::runtime_error("Batching from Transformer");
    }

    const size_t vec_dim = static_cast<size_t>(output_shape[2]);
    if (vec_dim != VECTOR_DIM) {
        throw std::runtime_error(
            "Transformer output dimension mismath: expected " +
            std::to_string(VECTOR_DIM) + ", got " + std::to_string(vec_dim));
    }

    // ------------------------------------------------------------
    // PHASE 5: Mean Pooling
    // Squeeze [1, seq_length, 384] into [384] by averaging.
    // ------------------------------------------------------------

    auto query_vec = CreateAlignedVector(vec_dim);
    const float* __restrict__ src = output_tensor.GetTensorData<float>();

    float* __restrict__ buf = query_vec.get();
    std::memset(buf, 0, vec_dim * sizeof(float));

    const float inv_seq_len = 1.0f / static_cast<float>(seq_length);
    for (size_t i = 0; i < seq_length; ++i) {
        for (size_t j = 0; j < vec_dim; ++j) {
            buf[i] += src[i * vec_dim + j] * inv_seq_len;
        }
    }

    // ------------------------------------------------------------
    // PHASE 6: L2 Normalization
    // Now ||buf|| == 1.0, so Dot Product == Cosine Similarity.
    // ------------------------------------------------------------

    float sum_sq = 0.0f;
    for (size_t i = 0; i < vec_dim; ++i) {
        sum_sq += buf[i] * buf[i];
    }

    if (sum_sq < 1e-9f) {
        return Result<AlignedVec, EncodeError>::Err(
            EncodeError::DEGENERATED_VECTOR);
    }

    const float inv_norm = 1.0f / std::sqrt(sum_sq);
    for (size_t i = 0; i < vec_dim; ++i) {
        buf[i] *= inv_norm;
    }

    return Result<AlignedVec, EncodeError>::Ok(std::move(query_vec));
}
