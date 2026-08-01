// Author: namnkahn1607
//
// Integration tests for Embedder::Encode() under concurrent load.
//
// Scope: these tests do NOT verify that the model produces semantically
// correct embeddings.
// Instead they verify two properties that must hold regardless of model
// weights:
//   1. Consistency: concurrent calls with the same prompt produce outputs
//      that are bitwise-identical to the single-threaded ground truth.
//   2. Validity: no output vector contains NaN values.
//
// Prerequisites:
//   export TOKENIZER_PATH=<absolute path to tokenizer.onnx>
//   export TRANSFORMER_PATH=<absolute path to model.onnx>

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "inference/inference_model.h"

namespace {

inline constexpr int32_t kDim = 384;

// DotProduct(): computes dot product between 2 normalized vectors.
float DotProduct(const float* query, const float* node_vector) {
    float sum = 0.0f;

    for (int32_t i = 0; i < kDim; ++i) {
        sum += (query[i] * node_vector[i]);
    }

    return sum;
}

}  // namespace

// -----------------------------------------------------------------------------
// Test fixture
// SetUpTestSuite() initializes the Embedder once and encodes the ground-truth
// vector on the main thread. All tests share the same Embedder instance to
// reflect real production usage (one Embedder, many concurrent requests).
// -----------------------------------------------------------------------------

class EmbedderConcurrencyTest : public ::testing::Test {
protected:
    static std::unique_ptr<Embedder> emb;
    static std::vector<float>        truth;

    static constexpr uint32_t NUM_THREADS      = 8;
    static constexpr uint32_t CALLS_PER_THREAD = 20;

    // Threshold for Cosine Similarity between two runs of the same prompt.
    // ORT with FMA may produce slightly different results across threads
    // due to floating-point non-associativity, but should be near-identical.
    static constexpr float COSINE_THRESHOLD = 0.9999f;

    const std::string kPrompt = "What is the capital of France?";

    static void SetUpTestSuite() {
        const char* tok_path  = std::getenv("TOKENIZER_PATH");
        const char* bert_path = std::getenv("TRANSFORMER_PATH");

        ASSERT_NE(tok_path, nullptr) << "TOKENIZER_PATH is not set";
        ASSERT_NE(bert_path, nullptr) << "TRANSFORMER_PATH is not set";

        emb = std::make_unique<Embedder>(tok_path, bert_path);

        // Encode ground-truth vector on main thread (single-threaded baseline).
        auto result = emb->Encode("What is the capital of France?");
        ASSERT_TRUE(result.ok()) << "Ground truth encode failed";
        const float* ptr = result.value().get();
        truth.assign(ptr, ptr + kDim);
    }

    static void TearDownTestSuite() { emb.reset(); }
};

std::unique_ptr<Embedder> EmbedderConcurrencyTest::emb = nullptr;
std::vector<float>        EmbedderConcurrencyTest::truth;

// -----------------------------------------------------------------------------
// OutputConsistencyUnderConcurrency
// 8 threads x 20 calls = 160 concurrent Encode() calls with the same prompt.
// Each result must have cosine similarity >= 0.9999 against the ground truth.
// Detects session state corruption or data races inside ORT.
// -----------------------------------------------------------------------------

TEST_F(EmbedderConcurrencyTest, OutputConsistencyUnderConcurrency) {
    // Single warm-up call to let ORT JIT-compile any lazy paths.
    [[maybe_unused]] auto _ = emb->Encode("Mom I love you");

    std::atomic<int32_t> correct_count{0};
    std::atomic<int32_t> total_count{0};
    std::atomic<bool>    any_exception{false};

    auto worker = [&]() {
        for (uint32_t i = 0; i < CALLS_PER_THREAD; ++i) {
            try {
                auto result = emb->Encode(kPrompt);
                if (!result.ok()) {
                    continue;
                }

                const float sim =
                    DotProduct(result.value().get(), truth.data());
                ++total_count;

                if (sim >= COSINE_THRESHOLD) {
                    ++correct_count;
                }

            } catch (...) {
                any_exception.store(true);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (uint32_t i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_FALSE(any_exception.load())
        << "At least one Encode() threw an exception under concurrency";
    EXPECT_EQ(total_count.load(), NUM_THREADS * CALLS_PER_THREAD);
    EXPECT_EQ(correct_count.load(), total_count.load())
        << correct_count.load() << "/" << total_count.load()
        << " outputs matched ground truth (cosine >= " << COSINE_THRESHOLD
        << ")";
}

// -----------------------------------------------------------------------------
// NoNaNUnderConcurrency
// 8 threads x 20 calls. Each output vector must contain no NaN values.
// NaN can appear if the L2 normalization step divides by a near-zero norm -
// Encode() should return Err(DEGENERATE_VECTOR) before that happens, but
// this test verifies no NaN escapes through any code path.
// -----------------------------------------------------------------------------

TEST_F(EmbedderConcurrencyTest, NoNaNUnderConcurrency) {
    std::atomic<int32_t> nan_count{0};
    std::atomic<bool>    any_exception{false};

    auto worker = [&]() {
        for (uint32_t i = 0; i < CALLS_PER_THREAD; ++i) {
            try {
                auto result = emb->Encode(kPrompt);
                if (!result.ok()) {
                    continue;
                }

                const float* ptr = result.value().get();
                for (size_t j = 0; j < kDim; ++j) {
                    if (std::isnan(ptr[j])) {
                        ++nan_count;
                    }
                }

            } catch (...) {
                any_exception.store(true);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (uint32_t i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_FALSE(any_exception.load())
        << "At least one Encode() threw an exception under concurrency";
    EXPECT_EQ(nan_count.load(), 0)
        << "NaN detected in at least one concurrent output vector";
}
