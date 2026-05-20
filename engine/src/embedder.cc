//
// Created by nlnk on Mar 1, 26.
//

#include "embedder.hh"

#include "constant.hh"

Embedder::Embedder(const char* tok_path, const char* bert_path)
    : env{Ort::Env(ORT_LOGGING_LEVEL_ERROR, "onnx-env")}
    , session_options{Ort::SessionOptions()} {
    // Highest level of graph optimization
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);

    session_options.EnableOrtCustomOps();

    tok_session =
        std::make_unique<Ort::Session>(env, tok_path, session_options);
    bert_session =
        std::make_unique<Ort::Session>(env, bert_path, session_options);
}

AlignedVector Embedder::Encode(const std::string& prompt) const {
    const Ort::AllocatorWithDefaultOptions allocator;

    // PHASE 1: Tokenization
    const std::vector<int64_t> input_shape{1};
    const char* input_string{prompt.c_str()};

    Ort::Value text_tensor{Ort::Value::CreateTensor(
        allocator, input_shape.data(), input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)};
    text_tensor.FillStringTensor(&input_string, 1);

    const char* tok_input_names[]{"text"};
    const char* tok_output_names[]{"input_ids", "attention_mask",
                                   "token_type_ids"};

    auto tok_outputs{tok_session->Run(Ort::RunOptions{nullptr}, tok_input_names,
                                      &text_tensor, 1, tok_output_names, 3)};

    // PHASE 2: Validation

    // 'input_ids' from Tokenizer has a tensor format of int64[?]
    const Ort::Value& input_ids_tensor = tok_outputs[0];
    const auto shape = input_ids_tensor.GetTensorTypeAndShapeInfo().GetShape();
    // '?' means Dynamic Axes, representing sequence length
    const auto seq_length = static_cast<size_t>(shape[0]);

    if (seq_length > MAX_TOKENS) {
        throw TokenLimitException("Prompt length " +
                                  std::to_string(seq_length) +
                                  " exceeds limit of 256");
    }

    // PHASE 3: Transformer inference
    const char* bert_input_names[]{"input_ids", "attention_mask",
                                   "token_type_ids"};
    const char* bert_output_names[]{"last_hidden_state"};

    // 'input_ids' to Transformer must be int64[batch_size,sequence_length]
    // Format a Rank-2 shape [1, seq_length]
    const std::array<int64_t, 2> bert_input_shape{
        1, static_cast<int64_t>(seq_length)};

    // Retrieve Memory Allocator from old Tensor
    const auto mem_info = input_ids_tensor.GetTensorMemoryInfo();

    // Create new View(s) pointing to old data
    std::vector<Ort::Value> bert_inputs;
    bert_inputs.reserve(3);
    for (size_t i = 0; i < 3; ++i) {
        auto* raw_data = tok_outputs[i].GetTensorMutableData<int64_t>();
        bert_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem_info, raw_data, seq_length, bert_input_shape.data(),
            bert_input_shape.size()));
    }

    const auto bert_outputs{
        bert_session->Run(Ort::RunOptions{nullptr}, bert_input_names,
                          bert_inputs.data(), 3, bert_output_names, 1)};

    // PHASE 4: Mean pooling & Normalization
    const Ort::Value& output_tensor{bert_outputs.front()};
    const auto type_info{output_tensor.GetTensorTypeAndShapeInfo()};

    // Output shape is always [1, N, 384].
    const std::vector output_shape{type_info.GetShape()};
    if (output_shape.size() != 3 || output_shape[0] != 1) {
        throw std::runtime_error("Unexpected output rank or batching detected");
    }

    const size_t vec_dimension{
        static_cast<size_t>(output_shape[2])};  // vector dimension: 384
    if (vec_dimension != engine::VECTOR_DIM) {
        throw std::runtime_error("Invalid sequence length or vector dimension");
    }

    auto query_vec = NewAlignedVector(vec_dimension);

    float* __restrict__ aligned_buffer = query_vec.get();
    const auto* __restrict__ float_array{output_tensor.GetTensorData<float>()};

    std::memset(aligned_buffer, 0, vec_dimension * sizeof(float));

    // Squeeze 2D array [N][384] into [384] array using Mean Pooling
    const float inv_seq_len = 1.0f / static_cast<float>(seq_length);
    for (size_t i = 0; i < seq_length; ++i) {
        for (size_t j = 0; j < vec_dimension; ++j) {
            aligned_buffer[j] +=
                float_array[i * vec_dimension + j] * inv_seq_len;
        }
    }

    // Perform L2 Normalization
    float sum_sq = 0.0f;
    for (size_t i = 0; i < vec_dimension; ++i) {
        sum_sq += aligned_buffer[i] * aligned_buffer[i];
    }

    if (sum_sq < 1e-9f) {
        throw std::runtime_error("Degenerate zero vector from model");
    }

    const float inv_magnitude{
        1.0f / std::sqrt(sum_sq)};  // Only perform square root once
    for (size_t i = 0; i < vec_dimension; ++i) {
        aligned_buffer[i] *= inv_magnitude;
    }

    return query_vec;
}
