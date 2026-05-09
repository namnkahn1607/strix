//
// Created by nlnk on Mar 1, 26.
//

#include "embedder.hh"

#include "constant.hh"

Embedder::Embedder(const char* model_path, const char* ext_path)
    : env_{Ort::Env(ORT_LOGGING_LEVEL_ERROR, "onnx-env")}
    , session_options_{Ort::SessionOptions()} {
    // Highest level of graph optimization
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);

    try {
        session_options_.RegisterCustomOpsLibrary(ext_path);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to register custom ops: ") + e.what());
    }

    session_ =
        std::make_unique<Ort::Session>(env_, model_path, session_options_);
}

AlignedVector Embedder::Encode(const std::string& prompt) const {
    const Ort::AllocatorWithDefaultOptions allocator_;

    // 1. Define Tensor Input structure
    const std::vector<int64_t> input_shape{1};
    const char* input_string{prompt.c_str()};

    // 2. Create String Tensor
    Ort::Value input_tensor{Ort::Value::CreateTensor(
        allocator_, input_shape.data(), input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)};
    input_tensor.FillStringTensor(&input_string, 1);

    // 3. Prepare Run configurations
    const char* input_names[]{"text"};
    const char* output_names[]{"last_hidden_state"};

    // 4. Use Neural Network to execute the Input Tensor
    const auto output_tensors{session_->Run(Ort::RunOptions{nullptr},
                                            input_names, &input_tensor, 1,
                                            output_names, 1)};

    // 5. Extract Output Data
    const Ort::Value& output_tensor{output_tensors.front()};
    const auto type_info{output_tensor.GetTensorTypeAndShapeInfo()};
    // Output shape is always [1, N, 384].
    const std::vector output_shape{type_info.GetShape()};

    if (output_shape.size() != 3 || output_shape[0] != 1) {
        throw std::runtime_error("Unexpected output rank or batching detected");
    }

    const size_t seq_length{
        static_cast<size_t>(output_shape[1])};  // number of tokens in sequence
    const size_t vec_dimension{
        static_cast<size_t>(output_shape[2])};  // vector dimension: 384

    if (seq_length == 0 || vec_dimension != engine::VECTOR_DIM) {
        throw std::runtime_error("Invalid sequence length or vector dimension");
    }

    auto query_vec = NewAlignedVector(vec_dimension);

    float* __restrict__ aligned_buffer = query_vec.get();
    const auto* __restrict__ float_array{output_tensor.GetTensorData<float>()};

    std::memset(aligned_buffer, 0, vec_dimension * sizeof(float));

    // 6. Squeeze 2D array [N][384] into [384] array using Mean Pooling
    const float inv_seq_len = 1.0f / static_cast<float>(seq_length);
    for (size_t i = 0; i < seq_length; ++i) {
        for (size_t j = 0; j < vec_dimension; ++j) {
            aligned_buffer[j] +=
                float_array[i * vec_dimension + j] * inv_seq_len;
        }
    }

    // 7. Perform L2 Normalization
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
