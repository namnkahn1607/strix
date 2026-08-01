// Author: namnkahn1607
//
// Embedder constructor and Encode() implementation.
// Encode() runs two sequential ORT sessions (tokenizer -> transformer)
// and returns a mean-pooled, L2-normalised embedding vector.

#include "inference/inference_model.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/constants.h"
#include "inference/aligned_vec.h"

namespace {

// Maximum token sequence length accepted by all-MiniLM-L6-v2.
// Inputs exceeding this limit are rejected with kTokenLimitExceeded before
// reaching the transformer session.
constexpr size_t kMaxTokens = 256;

}  // namespace

Embedder::Embedder(const char* tok_path, const char* bert_path)
    : env_{Ort::Env(ORT_LOGGING_LEVEL_ERROR, "onnx-env")}
    , options_{Ort::SessionOptions()} {
    // Maximum graph-level fusion and constant folding.
    options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // Single-threaded ORT execution: the RPC layer already runs one worker
    // thread per core, each calling Encode() independently. Allowing ORT
    // to spawn additional threads would push the total thread count above the
    // core count and introduce unnecessary context-switch overhead.
    options_.SetInterOpNumThreads(1);
    options_.SetIntraOpNumThreads(1);

    options_.EnableOrtCustomOps();

    tok_session_  = std::make_unique<Ort::Session>(env_, tok_path, options_);
    bert_session_ = std::make_unique<Ort::Session>(env_, bert_path, options_);
}

Result<AlignedVec, EncodeError> Embedder::Encode(const std::string& prompt
) const {
    const Ort::AllocatorWithDefaultOptions allocator;

    // -------------------------------------------------------------------------
    // PHASE 1: Tokenization
    // Run the tokenizer session on the raw prompt string.
    // -------------------------------------------------------------------------

    const std::vector<int64_t> input_shape{1};
    const char*                input_string = prompt.c_str();

    Ort::Value text_tensor = Ort::Value::CreateTensor(
        allocator, input_shape.data(), input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    );
    text_tensor.FillStringTensor(&input_string, 1);

    const char* tok_input_names[]{"text"};
    const char* tok_output_names[]{
        "input_ids", "attention_mask", "token_type_ids"
    };

    auto tok_outputs = tok_session_->Run(
        Ort::RunOptions{nullptr}, tok_input_names, &text_tensor, 1,
        tok_output_names, 3
    );

    // -------------------------------------------------------------------------
    // PHASE 2: Token count validation
    // input_ids has shape int64[seq_length] (dynamic axis).
    // Reject inputs that exceed the model's context window.
    // -------------------------------------------------------------------------

    const Ort::Value& input_ids_tensor = tok_outputs[0];
    const auto shape = input_ids_tensor.GetTensorTypeAndShapeInfo().GetShape();

    const auto seq_length = static_cast<size_t>(shape[0]);
    if (seq_length > kMaxTokens) {
        return Result<AlignedVec, EncodeError>::Err(
            EncodeError::kTokenLimitExceeded
        );
    }

    // -------------------------------------------------------------------------
    // PHASE 3: Transformer inference
    // Reshape tokenizer outputs from int64[seq_length] to
    // int64[1, seq_length] (batch_size=1) before passing to BERT.
    // -------------------------------------------------------------------------

    const char* bert_input_names[]{
        "input_ids", "attention_mask", "token_type_ids"
    };
    const char* bert_output_names[]{"last_hidden_state"};

    const std::array<int64_t, 2> bert_input_shape{
        1, static_cast<int64_t>(seq_length)
    };

    // Borrow the allocator info from the existing tensor so the new views
    // point into the same memory without copying.
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

    const auto bert_outputs = bert_session_->Run(
        Ort::RunOptions{nullptr}, bert_input_names, bert_inputs.data(), 3,
        bert_output_names, 1
    );

    // -------------------------------------------------------------------------
    // PHASE 4: Output shape validation
    // Expected shape: float[1, seq_length, kVectorDim].
    // Rank or dimension mismatches indicate a model file mismatch and are
    // unrecoverable; throw rather than return an error.
    // -------------------------------------------------------------------------

    const Ort::Value& output_tensor = bert_outputs.front();
    const auto        output_shape =
        output_tensor.GetTensorTypeAndShapeInfo().GetShape();

    if (output_shape.size() != 3) {
        throw std::runtime_error("Unexpected output rank from transformer");
    }

    if (output_shape[0] != 1) {
        throw std::runtime_error("Unexpected batch dimension from transformer");
    }

    const size_t vec_dim = static_cast<size_t>(output_shape[2]);
    if (vec_dim != kVectorDim) {
        throw std::runtime_error(
            "Transformer output dimension mismatch: expected " +
            std::to_string(kVectorDim) + ", got " + std::to_string(vec_dim)
        );
    }

    // -------------------------------------------------------------------------
    // PHASE 5: Mean pooling
    // Collapse float[1, seq_length, vec_dim] -> float[vec_dim] by averaging
    // across the sequence dimension.
    // -------------------------------------------------------------------------

    auto query_vec                = CreateAlignedVector(vec_dim);
    const float* __restrict__ src = output_tensor.GetTensorData<float>();

    float* __restrict__ buf = query_vec.get();
    std::memset(buf, 0, vec_dim * sizeof(float));

    const float inv_seq_len = 1.0f / static_cast<float>(seq_length);
    for (size_t i = 0; i < seq_length; ++i) {
        for (size_t j = 0; j < vec_dim; ++j) {
            buf[j] += src[i * vec_dim + j] * inv_seq_len;
        }
    }

    // -------------------------------------------------------------------------
    // PHASE 6: L2 normalisation
    // Scale buf so ||buf||_2 == 1.0, making dot product == cosine similarity.
    // A near-zero norm indicates a degenerate vector (e.g. all-padding input).
    // -------------------------------------------------------------------------

    float sum_sq = 0.0f;
    for (size_t i = 0; i < vec_dim; ++i) {
        sum_sq += buf[i] * buf[i];
    }

    if (sum_sq < 1e-9f) {
        return Result<AlignedVec, EncodeError>::Err(
            EncodeError::kDegeneratedVector
        );
    }

    const float inv_norm = 1.0f / std::sqrt(sum_sq);
    for (size_t i = 0; i < vec_dim; ++i) {
        buf[i] *= inv_norm;
    }

    return Result<AlignedVec, EncodeError>::Ok(std::move(query_vec));
}
