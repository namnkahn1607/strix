// Author: namnkahn1607
//
// Unit tests for DotProductL0_Batch4 - the AVX2 SIMD kernel that computes
// dot products between one query vector and a batch of 4 node vectors.
//
// Correctness strategy: a 'scalar' implementation is used as the oracle.
// Tolerance: 1e-4f accounts for FMA instruction reordering vs scalar addition.

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "avx2_kernel.h"

namespace {

inline constexpr float   kApprox3 = 1e-3f;
inline constexpr float   kApprox4 = 1e-4f;
inline constexpr int32_t kAlign   = 32;
inline constexpr int32_t kDim     = 384;
inline constexpr int32_t kBatch   = 4;

// ScalarDotProduct(): computes dot product between 2 normalized vectors in
// scalar mode; no SIMD optimizations. Used as correctness oracle.
float ScalarDotProduct(const float* query, const float* node_vector) {
    float sum = 0.0f;

    for (int32_t i = 0; i < kDim; ++i) {
        sum += (query[i] * node_vector[i]);
    }

    return sum;
}

// Normalize(): scaling a vector to a magnitude (length) of exactly 1.
void Normalize(float* vec, const int32_t dim) {
    float norm = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }

    norm = std::sqrt(norm);
    for (int32_t i = 0; i < dim; ++i) {
        vec[i] /= norm;
    }
}

};  // namespace

// -----------------------------------------------------------------------------
// Test fixture: allocates 32-byte aligned query and node_batch buffers.
// All tests that require normalized vectors call GenerateNormalizedBatch().
// -----------------------------------------------------------------------------

class DotProductBatch4Test : public ::testing::Test {
protected:
    float* query      = nullptr;
    float* node_batch = nullptr;

    void SetUp() override {
        query = static_cast<float*>(_mm_malloc(kDim * sizeof(float), kAlign));
        node_batch = static_cast<float*>(
            _mm_malloc(kBatch * kDim * sizeof(float), kAlign));
    }

    void TearDown() override {
        _mm_free(query);
        _mm_free(node_batch);
    }

    // Fills query and all 4 node vectors with random values, then normalizes
    // each to unit L2 norm. Random seeded for reproducibility.
    void GenerateNormalized(const uint64_t seed = 42) const {
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int32_t i = 0; i < kDim; ++i) {
            query[i] = dist(gen);
        }

        Normalize(query, kDim);

        for (int32_t b = 0; b < kBatch; ++b) {
            float* node = node_batch + b * kDim;

            for (int32_t i = 0; i < kDim; ++i) {
                node[i] = dist(gen);
            }

            Normalize(node, kDim);
        }
    }
};

// -----------------------------------------------------------------------------
// CompareWithScalarOracle
// Runs 1,000 iterations with different random seeds to verify that AVX2
// output matches those of 'scalar' Dot Product.
// -----------------------------------------------------------------------------

TEST_F(DotProductBatch4Test, CompareWithScalarOracle) {
    for (uint64_t iter = 0; iter < 1000; ++iter) {
        GenerateNormalized(iter);

        float scores[kBatch] = {};
        DotProductBatch(query, node_batch, scores);

        for (int32_t b = 0; b < kBatch; ++b) {
            const float expected =
                ScalarDotProduct(query, node_batch + b * kDim);
            EXPECT_NEAR(scores[b], expected, kApprox4)
                << "lane=" << b << " iter=" << iter;
        }
    }
}

// -----------------------------------------------------------------------------
// IdenticalVectors
// query == node for all 4 lanes. Dot Product of a unit vector with itself
// must equal 1.0.
// Tests the upper bound of the similarity range.
// -----------------------------------------------------------------------------

TEST_F(DotProductBatch4Test, IdenticalVectors) {
    GenerateNormalized();
    for (int32_t b = 0; b < kBatch; ++b) {
        std::copy_n(query, kDim, node_batch + b * kDim);
    }

    float scores[kBatch] = {};
    DotProductBatch(query, node_batch, scores);

    for (int32_t b = 0; b < kBatch; ++b) {
        EXPECT_NEAR(scores[b], 1.0f, kApprox4) << "lane=" << b;
    }
}

// -----------------------------------------------------------------------------
// OppositeVectors
// node = -query for all 4 lanes. Dot product must equal -1.0.
// Tests the lower bound of the similarity range.
// -----------------------------------------------------------------------------

TEST_F(DotProductBatch4Test, OppositeVectors) {
    GenerateNormalized();
    for (int32_t b = 0; b < kBatch; ++b) {
        float* node = node_batch + b * kDim;
        for (int32_t i = 0; i < kDim; ++i) {
            node[i] = -query[i];
        }
    }

    float scores[kBatch] = {};
    DotProductBatch(query, node_batch, scores);

    for (int32_t b = 0; b < kBatch; ++b) {
        EXPECT_NEAR(scores[b], -1.0f, kApprox4) << "lane=" << b;
    }
}

// -----------------------------------------------------------------------------
// MixedLanes
// Each of the 4 lanes holds a different vector relationship to query:
//   Lane 0: identical to query      -> score ~ +1.0
//   Lane 1: opposite to query       -> score ~ -1.0
//   Lane 2: orthogonal to query     -> score ~  0.0
//   Lane 3: independent random vec  -> score compared to scalar oracle
//
// Purpose: verify that AVX2 horizontal reduction does not bleed values
// across lanes. Each lane must be independent.
// -----------------------------------------------------------------------------

TEST_F(DotProductBatch4Test, MixedLanes) {
    // Lane 3: Independent random
    GenerateNormalized(7);

    // Lane 0: Identical
    std::copy_n(query, kDim, node_batch + 0 * kDim);

    // Lane 1: Opposite
    for (int32_t i = 0; i < kDim; ++i) {
        node_batch[1 * kDim + i] = -query[i];
    }

    // Lane 2: Orthogonal
    {
        float* node2 = node_batch + 2 * kDim;
        std::fill_n(node2, kDim, 0.0f);

        // Gram-Schmidt: node2 = e_1 - (e_1 \dot query) * query
        node2[0] = 1.0f;

        const float proj = query[0];  // e_1 \dot query = query[0]
        for (int32_t i = 0; i < kDim; ++i) {
            node2[i] -= proj * query[i];
        }

        Normalize(node2, kDim);
    }

    float scores[kBatch] = {};
    DotProductBatch(query, node_batch, scores);

    float scalar_lane3 = ScalarDotProduct(query, node_batch + 3 * kDim);

    EXPECT_NEAR(scores[0], 1.0f, kApprox4) << "lane=0 (identical)";
    EXPECT_NEAR(scores[1], -1.0f, kApprox4) << "lane=1 (opposite)";
    EXPECT_NEAR(scores[2], 0.0f, kApprox3) << "lane=2 (orthogonal)";
    EXPECT_NEAR(scores[3], scalar_lane3, kApprox4) << "lane=3 (random)";
}

// -----------------------------------------------------------------------------
// Deterministic
// Same input must produce bitwise-identical output across repeated calls.
// FMA is deterministic for identical inputs on the same CPU; this test
// catches any accidental use of non-deterministic intrinsics.
// -----------------------------------------------------------------------------

TEST_F(DotProductBatch4Test, Deterministic) {
    GenerateNormalized();

    float first[kBatch] = {};
    DotProductBatch(query, node_batch, first);

    for (int c = 0; c < 10; ++c) {
        float scores[kBatch] = {};
        DotProductBatch(query, node_batch, scores);
        for (int32_t b = 0; b < kBatch; ++b) {
            EXPECT_EQ(scores[b], first[b]) << "lane=" << b << " call=" << c;
        }
    }
}

// -----------------------------------------------------------------------------
// ZeroQuery
// Query vector is all zeros. All dot products must be exactly 0.0.
// Not a realistic production case (Embedder rejects degenerated vectors),
// but verifies the kernel does not produce NaN or garbage on zero input.
// -----------------------------------------------------------------------------

TEST_F(DotProductBatch4Test, ZeroQuery) {
    GenerateNormalized();
    std::fill_n(query, kDim, 0.0f);

    float scores[kBatch] = {};
    DotProductBatch(query, node_batch, scores);

    for (int32_t b = 0; b < kBatch; ++b) {
        EXPECT_FLOAT_EQ(scores[b], 0.0f) << "lane=" << b;
    }
}

// -----------------------------------------------------------------------------
// SingleHotVector
// query = e_k (unit vector along dimension k).
// node[b] = e_{b} for b in [0, 3].
// Expected: scores[b] = query[b] = delta(k, b).
// Isolates individual dimensions to catch indexing bugs in the SIMD loop.
// -----------------------------------------------------------------------------

TEST_F(DotProductBatch4Test, SingleHotVector) {
    std::fill_n(query, kDim, 0.0f);
    std::fill_n(node_batch, kBatch * kDim, 0.0f);

    // query = e_0
    query[0] = 1.0f;

    // node[b] = e_b
    for (int32_t b = 0; b < kBatch; ++b) {
        node_batch[b * kDim + b] = 1.0f;
    }

    float scores[kBatch] = {};
    DotProductBatch(query, node_batch, scores);

    // Only lane 0 should have score = 1.0, rest = 0.0
    EXPECT_FLOAT_EQ(scores[0], 1.0f) << "lane=0 (e_0 \\dot e_0)";
    for (int32_t b = 1; b < kBatch; ++b) {
        EXPECT_FLOAT_EQ(scores[b], 0.0f) << "lane=" << b << " (e_0 \\dot e_b)";
    }
}
