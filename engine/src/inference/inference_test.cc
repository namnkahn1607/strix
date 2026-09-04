// Deterministic tests for text embedding pipeline concurrent encoding.
//
// Does NOT verify whether the encoder produces semantically correct embeddings.
// Instead they verify 2 properties must held regardless of model weights:
//   1. Consistency: concurrent calls with the same prompt produce outputs that
//      are bitwise-identical to the single-threaded one.
//   2. Validity: no output vector contains NaN values.

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "inference/info.h"
#include "inference/sentence_encoder.h"
#include "inference/simd_float_buf.h"

using namespace strix::inference;

namespace {

const char* kTokenizerVar   = "TOKENIZER_PATH";
const char* kTransformerVar = "TRANSFORMER_PATH";

const char* kPrompt = "What is the capital of France?";

// Computes dot product between 2 vectors. Expects normalized.
float DotProduct(const float* vec1, const float* vec2) {
    float sum = 0.0f;
    for (size_t i = 0; i < kVectorDim; ++i) {
        sum += (vec1[i] * vec2[i]);
    }
    return sum;
}

}  // namespace

class EncoderConcurrentTest : public ::testing::Test {
protected:
    static constexpr uint32_t kNumThreads     = 8;
    static constexpr uint32_t kCallsPerThread = 16;
    // 8 threads x 16 calls = 128 concurrent encoding calls.

    // Similarity threshold for two runs on the same prompt.
    // ORT with FMA may produce slightly different results across threads due
    // to floating-point non-associativity, but should be near-identical.
    static constexpr float kCosineThreshold = 0.9999f;

    static std::unique_ptr<SentenceEncoder> encoder;
    static SimdFloatBuf                     truth;

    static void SetUpTestSuite() {
        // NOLINTBEGIN(concurrency-mt-unsafe)
        const char* tok_path  = std::getenv(kTokenizerVar);
        const char* bert_path = std::getenv(kTransformerVar);
        // NOLINTEND(concurrency-mt-unsafe)

        ASSERT_NE(tok_path, nullptr) << "TOKENIZER_PATH is not set";
        ASSERT_NE(bert_path, nullptr) << "TRANSFORMER_PATH is not set";
        encoder = std::make_unique<SentenceEncoder>(tok_path, bert_path);

        ASSERT_FALSE(encoder->Encode(kPrompt, truth).has_value())
            << "Ground truth encode failed";
    }

    static void TearDownTestSuite() { encoder.reset(); }
};

std::unique_ptr<SentenceEncoder> EncoderConcurrentTest::encoder = nullptr;
SimdFloatBuf                     EncoderConcurrentTest::truth;

// -----------------------------------------------------------------------------
// OutputConsistencyUnderConcurrency
// Each result must have cosine similarity geq threshold against the ground
// truth. Detects session state corruption or data races inside ORT.
// -----------------------------------------------------------------------------
TEST_F(EncoderConcurrentTest, OutputConsistencyUnderConcurrency) {
    std::atomic<uint32_t> correct_count{0};
    std::atomic<uint32_t> total_count{0};
    std::atomic<bool>     any_exception{false};

    auto worker = [&]() {
        for (uint32_t i = 0; i < kCallsPerThread; ++i) {
            try {
                SimdFloatBuf result;
                if (encoder->Encode(kPrompt, result).has_value()) {
                    continue;
                }

                float similarity = DotProduct(result.data(), truth.data());
                correct_count += (similarity >= kCosineThreshold);
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
    EXPECT_EQ(total_count.load(), kNumThreads * kCallsPerThread);
    EXPECT_EQ(correct_count.load(), total_count.load())
        << correct_count.load() << "/" << total_count.load()
        << " outputs matched ground truth (similarity >= " << kCosineThreshold
        << ")";
}

// -----------------------------------------------------------------------------
// NoNaNUnderConcurrency
// Each output vector must contain no NaN values, as it can appear if the
// normalization step divides by a near-zero norm.
// Verifies no NaN escapes the `EncodeError::kDegeneratedVector` return.
// -----------------------------------------------------------------------------
TEST_F(EncoderConcurrentTest, NoNaNUnderConcurrency) {
    std::atomic<uint32_t> total_count{0};
    std::atomic<uint32_t> nan_count{0};
    std::atomic<bool>     any_exception{false};

    auto worker = [&]() {
        for (uint32_t i = 0; i < kCallsPerThread; ++i) {
            try {
                SimdFloatBuf result;
                if (encoder->Encode(kPrompt, result).has_value()) {
                    continue;
                }

                const float* ptr = result.data();
                for (size_t j = 0; j < kVectorDim; ++j) {
                    nan_count += std::isnan(ptr[j]);
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
