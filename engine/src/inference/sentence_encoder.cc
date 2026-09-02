// Text embedding pipeline.

#include "inference/sentence_encoder.h"

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#include "inference/info.h"
#include "inference/simd_float_buf.h"

namespace strix::inference {

Ort::SessionOptions SentenceEncoder::InitOptions() {
    Ort::SessionOptions options;

    options.EnableOrtCustomOps();
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // Single-threaded ORT execution: avoid spawning too many threads, which
    // introduces unnecessary context-switch overhead.
    options.SetInterOpNumThreads(1);
    options.SetIntraOpNumThreads(1);

    return options;
}

SentenceEncoder::SentenceEncoder(const char* tok_path, const char* bert_path)
    : env_{Ort::Env(ORT_LOGGING_LEVEL_ERROR, "onnx-env")}
    , options_{InitOptions()}
    , tok_session_{env_, tok_path, options_}
    , bert_session_{env_, bert_path, options_} {}

std::optional<EncodeError> SentenceEncoder::Encode(
    const std::string& prompt, SimdFloatBuf& out
) const {
    const Ort::AllocatorWithDefaultOptions allocator;

    // PHASE 1: Tokenization
    // Tokenizing the input text into a sequence of word pieces.

    const std::vector<int64_t> input_shape{1};
    const char*                input_string = prompt.c_str();

    auto text_tensor = Ort::Value::CreateTensor(
        allocator, input_shape.data(), input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    );
    text_tensor.FillStringTensor(&input_string, 1);

    const char* tok_input_names[]{"text"};
    const char* tok_output_names[]{
        "input_ids", "attention_mask", "token_type_ids"
    };
    auto tok_outputs = tok_session_.Run(
        Ort::RunOptions{nullptr}, tok_input_names, &text_tensor, 1,
        tok_output_names, 3
    );

    // PHASE 2: Boundary validation
    // input_ids has shape int64[seq_length] (dynamic axis).
    // Reject inputs that exceed the model's context window.

    const Ort::Value& input_ids_tensor = tok_outputs.front();
    const auto shape = input_ids_tensor.GetTensorTypeAndShapeInfo().GetShape();

    const auto seq_length = static_cast<size_t>(shape.front());
    if (seq_length > kMaxTokens) {
        return EncodeError::kTokenLimitExceeded;
    }

    // PHASE 3: Reformat transformer input
    // Reshape from int64[seq_length] to int64[1, seq_length] (batch_size = 1).

    const std::array<int64_t, 2> bert_input_shape{
        1, static_cast<int64_t>(seq_length)
    };

    // Reuse allocator info from existing tensor so the new views
    // point to the same memory.
    const auto mem_info = input_ids_tensor.GetTensorMemoryInfo();

    std::vector<Ort::Value> bert_inputs;
    bert_inputs.reserve(3);
    for (size_t i = 0; i < 3; ++i) {
        auto* raw_data = tok_outputs[i].GetTensorMutableData<int64_t>();
        bert_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem_info, raw_data, seq_length, bert_input_shape.data(),
            bert_input_shape.size()
        ));
    }

    // PHASE 4: Transformation
    // Turns the word piece sequence to array of embeddings.

    const char* bert_input_names[]{
        "input_ids", "attention_mask", "token_type_ids"
    };
    const char* bert_output_names[]{"last_hidden_state"};
    const auto  bert_outputs = bert_session_.Run(
        Ort::RunOptions{nullptr}, bert_input_names, bert_inputs.data(), 3,
        bert_output_names, 1
    );

    // PHASE 5: Output shape validation
    // Expected shape: float[1, seq_length, kVectorDim].

    const Ort::Value& output_tensor = bert_outputs.front();
    const auto        output_shape =
        output_tensor.GetTensorTypeAndShapeInfo().GetShape();

    if (output_shape.size() != 3) {
        throw std::runtime_error(
            "Unexpected output rank from transformer: expected 3, got " +
            std::to_string(output_shape.size())
        );
    }
    if (output_shape.front() != 1) {
        throw std::runtime_error(
            "Unexpected batch dimension from transformer: expected 1, got " +
            std::to_string(output_shape.front())
        );
    }

    const auto actual_dim = static_cast<size_t>(output_shape[2]);
    if (actual_dim != kVectorDim) {
        throw std::runtime_error(
            "Transformer output dimension mismatch: expected " +
            std::to_string(kVectorDim) + ", got " + std::to_string(actual_dim)
        );
    }

    // PHASE 6: Mean pooling
    // Collapses float[1, seq_length, vec_dim] -> float[vec_dim] by averaging
    // across the sequence dimension.

    const float* __restrict__ src = output_tensor.GetTensorData<float>();

    float* dst = out.data();
    std::memset(dst, 0, actual_dim * sizeof(float));

    const float inv_seq_len = 1.0f / static_cast<float>(seq_length);
    for (size_t i = 0; i < seq_length; ++i) {
        for (size_t j = 0; j < actual_dim; ++j) {
            dst[j] += src[i * actual_dim + j] * inv_seq_len;
        }
    }

    // PHASE 7: Normalization
    // Scales ||out|| to 1.0, making dot product == cosine similarity.

    float sum_sq = 0.0f;
    for (size_t i = 0; i < actual_dim; ++i) {
        sum_sq += dst[i] * dst[i];
    }

    if (sum_sq < 1e-9f) {
        // A near-zero norm indicates a degenerated vector.
        return EncodeError::kDegeneratedVector;
    }

    const float inv_norm = 1.0f / std::sqrt(sum_sq);
    for (size_t i = 0; i < actual_dim; ++i) {
        dst[i] *= inv_norm;
    }

    return std::nullopt;
}

}  // namespace strix::inference
