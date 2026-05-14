//
// Created by nlnk on Apr 24, 26.
//

#include <gtest/gtest.h>
#include <immintrin.h>

#include <random>

#include "avx_math.hh"

constexpr int32_t ALIGN = 32;
constexpr int32_t DIM = 384;
constexpr int32_t BATCH = 4;

float ScalarCosineSimilarity(const float* query, const float* node_vector) {
    float sum = 0.0f;

    for (int32_t i = 0; i < DIM; ++i) {
        sum += (query[i] * node_vector[i]);
    }

    return sum;
}

class AVX2Batch4CosineTest : public ::testing::Test {
protected:
    float* query = nullptr;
    float* node_batch = nullptr;

    void SetUp() override {
        query = static_cast<float*>(_mm_malloc(DIM * sizeof(float), ALIGN));
        node_batch =
            static_cast<float*>(_mm_malloc(BATCH * DIM * sizeof(float), ALIGN));
    }

    void TearDown() override {
        _mm_free(query);
        _mm_free(node_batch);
    }

    void GenerateNormalizedBatch(const int32_t seed = 42) const {
        auto normalize = [](float* v, const int32_t dim, std::mt19937& rng) {
            std::uniform_real_distribution dist(-1.0f, 1.0f);
            float norm = 0.0f;

            for (int32_t i = 0; i < dim; ++i) {
                v[i] = dist(rng);
                norm += (v[i] * v[i]);
            }

            norm = std::sqrt(norm);

            for (int32_t i = 0; i < dim; ++i) {
                v[i] /= norm;
            }
        };

        std::mt19937 gen(seed);

        normalize(query, DIM, gen);
        for (int32_t b = 0; b < BATCH; ++b) {
            normalize(node_batch + b * DIM, DIM, gen);
        }
    }
};

TEST_F(AVX2Batch4CosineTest, CompareWithScalarOracle) {
    for (int iter = 0; iter < 1000; ++iter) {
        GenerateNormalizedBatch(iter);

        float scores[BATCH] = {};
        CosineL0_Batch4(query, node_batch, scores);

        for (int32_t b = 0; b < BATCH; ++b) {
            const float expected =
                ScalarCosineSimilarity(query, node_batch + b * DIM);
            EXPECT_NEAR(scores[b], expected, 1e-4f)
                << "lane " << b << " iter " << iter;
        }
    }
}

TEST_F(AVX2Batch4CosineTest, IdenticalVectors) {
    GenerateNormalizedBatch();
    for (int32_t b = 0; b < BATCH; ++b) {
        std::copy_n(query, DIM, node_batch + b * DIM);
    }

    float scores[BATCH] = {};
    CosineL0_Batch4(query, node_batch, scores);

    for (int32_t b = 0; b < BATCH; ++b) {
        EXPECT_NEAR(scores[b], 1.0f, 1e-4f) << "lane " << b;
    }
}

TEST_F(AVX2Batch4CosineTest, OppositeVectors) {
    GenerateNormalizedBatch();
    for (int32_t b = 0; b < BATCH; ++b) {
        float* node = node_batch + b * DIM;
        for (int32_t i = 0; i < DIM; ++i) {
            node[i] = -query[i];
        }
    }

    float scores[BATCH] = {};
    CosineL0_Batch4(query, node_batch, scores);

    for (int32_t b = 0; b < BATCH; ++b) {
        EXPECT_NEAR(scores[b], -1.0f, 1e-4f) << "lane " << b;
    }
}

TEST_F(AVX2Batch4CosineTest, MixedLanes) {
    GenerateNormalizedBatch();

    float scores[BATCH] = {};
    CosineL0_Batch4(query, node_batch, scores);

    for (int32_t b = 0; b < BATCH; ++b) {
        const float expected =
            ScalarCosineSimilarity(query, node_batch + b * DIM);
        EXPECT_NEAR(scores[b], expected, 1e-4f) << "lane " << b;
    }
}

TEST_F(AVX2Batch4CosineTest, Deterministic) {
    GenerateNormalizedBatch();

    float first[BATCH] = {};
    CosineL0_Batch4(query, node_batch, first);

    for (int c = 0; c < 10; ++c) {
        float scores[BATCH] = {};
        CosineL0_Batch4(query, node_batch, scores);
        for (int32_t b = 0; b < BATCH; ++b) {
            EXPECT_EQ(scores[b], first[b]) << "lane " << b << " call " << c;
        }
    }
}
