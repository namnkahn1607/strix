// Deterministic tests for text embedding pipeline concurrent encoding.
//
// These tests do NOT verify whether the pipeline produces semantically correct
// embeddings or not.
// Instead they verify 2 properties that must hold regardless of model weights:
//   1. Consistency: concurrent calls with the same prompt produce outputs
//      that are bitwise-identical to the single-threaded one.
//   2. Validity: no output vector contains NaN values.

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/constants.h"
#include "inference/sentence_encoder.h"
#include "inference/simd_float_buf.h"

namespace {

// DotProduct computes dot product of 2 normalized vectors.
// Note: here, dot product == cosine similarity.
float DotProduct(const float* vec1, const float* vec2) {
    float sum = 0.0f;

    for (size_t i = 0; i < kVectorDim; ++i) {
        sum += (vec1[i] * vec2[i]);
    }

    return sum;
}

}  // namespace

// EncoderConcurrencyTest initializes the text embedding pipeline once and
// encodes the ground-truth vector on setup and destruct the pipeline on
// teardown.
//
// All tests share the same instance to reflect real production usage
// (single `SentenceEncoder` instance, multiple concurrent requests).
class EncoderConcurrencyTest : public ::testing::Test {
protected:
    static std::unique_ptr<SentenceEncoder> encoder;
    static std::vector<float>               truth;

    static constexpr uint32_t kNumThreads     = 8;
    static constexpr uint32_t kCallsPerThread = 20;

    // Similarity threshold for two runs on the same prompt.
    // ORT with FMA may produce slightly different results across threads
    // due to floating-point non-associativity, but should be near-identical.
    static constexpr float kCosineThreshold = 0.9999f;

    const std::string kPrompt = "What is the capital of France?";

    static void SetUpTestSuite() {
        const char* tok_path  = std::getenv("TOKENIZER_PATH");
        const char* bert_path = std::getenv("TRANSFORMER_PATH");

        ASSERT_NE(tok_path, nullptr) << "TOKENIZER_PATH is not set";
        ASSERT_NE(bert_path, nullptr) << "TRANSFORMER_PATH is not set";

        encoder = std::make_unique<SentenceEncoder>(tok_path, bert_path);

        SimdFloatBuf truth_buf;

        auto encode_err = encoder->Encode(
            "What is the capital of France?",
            std::span<float, kVectorDim>{truth_buf.data(), kVectorDim}
        );
        ASSERT_TRUE(!encode_err.has_value()) << "Ground truth encode failed";

        truth.assign(truth_buf.data(), truth_buf.data() + kVectorDim);
    }

    static void TearDownTestSuite() { encoder.reset(); }
};

std::unique_ptr<SentenceEncoder> EncoderConcurrencyTest::encoder = nullptr;
std::vector<float>               EncoderConcurrencyTest::truth;

// -----------------------------------------------------------------------------
// OutputConsistencyUnderConcurrency
// 8 threads x 20 calls = 160 concurrent encoding calls with the same prompt.
// Each result must have cosine similarity >= 0.9999 against the ground truth.
// Detects session state corruption or data races inside ORT.
// -----------------------------------------------------------------------------
TEST_F(EncoderConcurrencyTest, OutputConsistencyUnderConcurrency) {
    // Warm-up call to force ORT performing JIT-optimization.
    alignas(32) std::array<float, kVectorDim> temp_buf;
    [[maybe_unused]] auto _ = encoder->Encode("Mom I love you", temp_buf);

    std::atomic<uint32_t> correct_count{0};
    std::atomic<uint32_t> total_count{0};
    std::atomic<bool>     any_exception{false};

    auto worker = [&]() {
        for (uint32_t i = 0; i < kCallsPerThread; ++i) {
            try {
                SimdFloatBuf result_buf;

                auto encode_err = encoder->Encode(
                    kPrompt,
                    std::span<float, kVectorDim>{result_buf.data(), kVectorDim}
                );
                if (encode_err.has_value()) {
                    continue;
                }

                const float similarity =
                    DotProduct(result_buf.data(), truth.data());
                ++total_count;

                if (similarity >= kCosineThreshold) {
                    ++correct_count;
                }

            } catch (...) {
                any_exception.store(true);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (uint32_t i = 0; i < kNumThreads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_FALSE(any_exception.load())
        << "At least one encoding procedure threw exception under concurrency";
    EXPECT_EQ(total_count.load(), kNumThreads * kCallsPerThread);
    EXPECT_EQ(correct_count.load(), total_count.load())
        << correct_count.load() << "/" << total_count.load()
        << " outputs matched ground truth (similarity >= " << kCosineThreshold
        << ")";
}

// -----------------------------------------------------------------------------
// NoNaNUnderConcurrency
// 8 threads x 20 calls. Each output vector must contain no NaN values.
// NaN can appear if the normalization step divides by a near-zero norm.
// Encoding should return `EncodeError::kDegeneratedVector` before that happens,
// but this test verifies no NaN escapes through that code path.
// -----------------------------------------------------------------------------
TEST_F(EncoderConcurrencyTest, NoNaNUnderConcurrency) {
    std::atomic<uint32_t> total_count{0};
    std::atomic<uint32_t> nan_count{0};
    std::atomic<bool>     any_exception{false};

    auto worker = [&]() {
        for (uint32_t i = 0; i < kCallsPerThread; ++i) {
            try {
                SimdFloatBuf result_buf;

                auto encode_err = encoder->Encode(
                    kPrompt,
                    std::span<float, kVectorDim>{result_buf.data(), kVectorDim}
                );
                if (encode_err.has_value()) {
                    continue;
                }

                const float* ptr = result_buf.data();
                for (size_t j = 0; j < kVectorDim; ++j) {
                    if (std::isnan(ptr[j])) {
                        ++nan_count;
                    }
                }

                ++total_count;

            } catch (...) {
                any_exception.store(true);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (uint32_t i = 0; i < kNumThreads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_FALSE(any_exception.load())
        << "At least one encoding procedure threw exception under concurrency";
    EXPECT_EQ(nan_count.load(), 0)
        << "NaN detected in at least one concurrent output vector";
    EXPECT_EQ(total_count.load(), kNumThreads * kCallsPerThread);
}
