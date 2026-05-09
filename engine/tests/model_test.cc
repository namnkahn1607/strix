//
// Created by nlnk on May 9, 26.
//

#include <gtest/gtest.h>

#include "embedder.hh"

constexpr size_t DIM = 384;

static std::vector<float> GetGroundTruth(const std::string& prompt) {
    const auto vec = Embedder::GetInstance().Encode(prompt);
    return {vec.get(), vec.get() + DIM};
}

static float CosineSimilarity(const float* a, const float* b) {
    float dot = 0.0f;

    for (size_t i = 0; i < DIM; ++i) {
        dot += (a[i] * b[i]);
    }

    return dot;
}

class EmbedderConcurrencyTest : public ::testing::Test {
protected:
    static constexpr uint32_t NUM_THREADS = 8;
    static constexpr uint32_t CALLS_PER_THREAD = 20;
    static constexpr float COSINE_THRESHOLD = 0.9999f;

    const std::string kPrompt = "What is the capital of France?";
};

TEST_F(EmbedderConcurrencyTest, OutputConsistencyUnderConcurrency) {
    [[maybe_unused]] auto _ = Embedder::GetInstance().Encode("warmup");
    const auto truth = GetGroundTruth(kPrompt);

    std::atomic correct_count{0};
    std::atomic total_count{0};
    std::atomic any_exception{false};

    auto worker = [&]() {
        for (uint32_t i = 0; i < CALLS_PER_THREAD; ++i) {
            try {
                auto vec = Embedder::GetInstance().Encode(kPrompt);
                const float sim = CosineSimilarity(vec.get(), truth.data());

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
        << "At least one Encode() threw under concurrency";
    EXPECT_EQ(total_count.load(), NUM_THREADS * CALLS_PER_THREAD);
    EXPECT_EQ(correct_count.load(), total_count.load())
        << correct_count.load() << "/" << total_count.load()
        << " outputs matched ground truth (cosine >= " << COSINE_THRESHOLD
        << ")";
}

TEST_F(EmbedderConcurrencyTest, NoNaNUnderConcurrency) {
    std::atomic nan_count{0};

    auto worker = [&]() {
        for (int i = 0; i < CALLS_PER_THREAD; ++i) {
            auto vec = Embedder::GetInstance().Encode(kPrompt);
            for (size_t j = 0; j < DIM; ++j) {
                if (std::isnan(vec.get()[j])) {
                    ++nan_count;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(nan_count.load(), 0) << "NaN detected in concurrent output";
}
