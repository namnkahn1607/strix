//
// tests/avx2_test.cc
//
// Unit tests for DotProductL0_Batch4 - the AVX2 SIMD kernel that computes
// dot products between one query vector and a batch of 4 node vectors.
//
// Correctness strategy: a 'scalar' implementation is used as the oracle.
// Tolerance: 1e-4f accounts for FMA instruction reordering vs scalar addition.
//

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "avx2_math.hh"

namespace {

inline constexpr float   APPROX3 = 1e-3f;
inline constexpr float   APPROX4 = 1e-4f;
inline constexpr int32_t ALIGN = 32;
inline constexpr int32_t DIM = 384;
inline constexpr int32_t BATCH = 4;

// --- ScalarDotProduct ---
// Used as correctness oracle.
float ScalarDotProduct(const float* query, const float* node_vector) {
    float sum = 0.0f;

    for (int32_t i = 0; i < DIM; ++i) {
        sum += (query[i] * node_vector[i]);
    }

    return sum;
}

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

// ---------------------------------------------------------------------------
// Test fixture: allocates 32-byte aligned query and node_batch buffers.
// All tests that require normalized vectors call GenerateNormalizedBatch().
// ---------------------------------------------------------------------------

class DotProductBatch4Test : public ::testing::Test {
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

    // Fills query and all 4 node vectors with random values, then normalizes
    // each to unit L2 norm. Random seeded for reproducibility.
    void GenerateNormalized(const uint64_t seed = 42) const {
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int32_t i = 0; i < DIM; ++i) {
            query[i] = dist(gen);
        }

        Normalize(query, DIM);

        for (int32_t b = 0; b < BATCH; ++b) {
            float* node = node_batch + b * DIM;

            for (int32_t i = 0; i < DIM; ++i) {
                node[i] = dist(gen);
            }

            Normalize(node, DIM);
        }
    }
};

// ---------------------------------------------------------------------------
// CompareWithScalarOracle
// Runs 1,000 iterations with different random seeds to verify that AVX2
// output matches those of 'scalar' Dot Product.
// ---------------------------------------------------------------------------
TEST_F(DotProductBatch4Test, CompareWithScalarOracle) {
    for (uint64_t iter = 0; iter < 1000; ++iter) {
        GenerateNormalized(iter);

        float scores[BATCH] = {};
        DotProductL0_Batch4(query, node_batch, scores);

        for (int32_t b = 0; b < BATCH; ++b) {
            const float expected =
                ScalarDotProduct(query, node_batch + b * DIM);
            EXPECT_NEAR(scores[b], expected, APPROX4)
                << "lane=" << b << " iter=" << iter;
        }
    }
}

// ---------------------------------------------------------------------------
// IdenticalVectors
// query == node for all 4 lanes. Dot Product of a unit vector with itself
// must equal 1.0.
// Tests the upper bound of the similarity range.
// ---------------------------------------------------------------------------
TEST_F(DotProductBatch4Test, IdenticalVectors) {
    GenerateNormalized();
    for (int32_t b = 0; b < BATCH; ++b) {
        std::copy_n(query, DIM, node_batch + b * DIM);
    }

    float scores[BATCH] = {};
    DotProductL0_Batch4(query, node_batch, scores);

    for (int32_t b = 0; b < BATCH; ++b) {
        EXPECT_NEAR(scores[b], 1.0f, APPROX4) << "lane=" << b;
    }
}

// ---------------------------------------------------------------------------
// OppositeVectors
// node = -query for all 4 lanes. Dot product must equal -1.0.
// Tests the lower bound of the similarity range.
// ---------------------------------------------------------------------------
TEST_F(DotProductBatch4Test, OppositeVectors) {
    GenerateNormalized();
    for (int32_t b = 0; b < BATCH; ++b) {
        float* node = node_batch + b * DIM;
        for (int32_t i = 0; i < DIM; ++i) {
            node[i] = -query[i];
        }
    }

    float scores[BATCH] = {};
    DotProductL0_Batch4(query, node_batch, scores);

    for (int32_t b = 0; b < BATCH; ++b) {
        EXPECT_NEAR(scores[b], -1.0f, APPROX4) << "lane=" << b;
    }
}

// ---------------------------------------------------------------------------
// MixedLanes
// Each of the 4 lanes holds a different vector relationship to query:
//   Lane 0: identical to query      -> score ~ +1.0
//   Lane 1: opposite to query       -> score ~ -1.0
//   Lane 2: orthogonal to query     -> score ~  0.0
//   Lane 3: independent random vec  -> score compared to scalar oracle
//
// Purpose: verify that AVX2 horizontal reduction does not bleed values
// across lanes. Each lane must be independent.
// ---------------------------------------------------------------------------
TEST_F(DotProductBatch4Test, MixedLanes) {
    // Lane 3: Independent random
    GenerateNormalized(7);

    // Lane 0: Identical
    std::copy_n(query, DIM, node_batch + 0 * DIM);

    // Lane 1: Opposite
    for (int32_t i = 0; i < DIM; ++i) {
        node_batch[1 * DIM + i] = -query[i];
    }

    // Lane 2: Orthogonal
    {
        float* node2 = node_batch + 2 * DIM;
        std::fill_n(node2, DIM, 0.0f);

        // Gram-Schmidt: node2 = e_1 - (e_1 \dot query) * query
        node2[0] = 1.0f;

        const float proj = query[0];  // e_1 \dot query = query[0]
        for (int32_t i = 0; i < DIM; ++i) {
            node2[i] -= proj * query[i];
        }

        Normalize(node2, DIM);
    }

    float scores[BATCH] = {};
    DotProductL0_Batch4(query, node_batch, scores);

    float scalar_lane3 = ScalarDotProduct(query, node_batch + 3 * DIM);

    EXPECT_NEAR(scores[0], 1.0f, APPROX4) << "lane=0 (identical)";
    EXPECT_NEAR(scores[1], -1.0f, APPROX4) << "lane=1 (opposite)";
    EXPECT_NEAR(scores[2], 0.0f, APPROX3) << "lane=2 (orthogonal)";
    EXPECT_NEAR(scores[3], scalar_lane3, APPROX4) << "lane=3 (random)";
}

// ---------------------------------------------------------------------------
// Deterministic
// Same input must produce bitwise-identical output across repeated calls.
// FMA is deterministic for identical inputs on the same CPU; this test
// catches any accidental use of non-deterministic intrinsics.
// ---------------------------------------------------------------------------
TEST_F(DotProductBatch4Test, Deterministic) {
    GenerateNormalized();

    float first[BATCH] = {};
    DotProductL0_Batch4(query, node_batch, first);

    for (int c = 0; c < 10; ++c) {
        float scores[BATCH] = {};
        DotProductL0_Batch4(query, node_batch, scores);
        for (int32_t b = 0; b < BATCH; ++b) {
            EXPECT_EQ(scores[b], first[b]) << "lane=" << b << " call=" << c;
        }
    }
}

// ---------------------------------------------------------------------------
// ZeroQuery
// Query vector is all zeros. All dot products must be exactly 0.0.
// Not a realistic production case (Embedder rejects degenerated vectors),
// but verifies the kernel does not produce NaN or garbage on zero input.
// ---------------------------------------------------------------------------
TEST_F(DotProductBatch4Test, ZeroQuery) {
    GenerateNormalized();
    std::fill_n(query, DIM, 0.0f);

    float scores[BATCH] = {};
    DotProductL0_Batch4(query, node_batch, scores);

    for (int32_t b = 0; b < BATCH; ++b) {
        EXPECT_FLOAT_EQ(scores[b], 0.0f) << "lane=" << b;
    }
}

// ---------------------------------------------------------------------------
// SingleHotVector
// query = e_k (unit vector along dimension k).
// node[b] = e_{b} for b in [0, 3].
// Expected: scores[b] = query[b] = delta(k, b).
// Isolates individual dimensions to catch indexing bugs in the SIMD loop.
// ---------------------------------------------------------------------------
TEST_F(DotProductBatch4Test, SingleHotVector) {
    std::fill_n(query, DIM, 0.0f);
    std::fill_n(node_batch, BATCH * DIM, 0.0f);

    // query = e_0
    query[0] = 1.0f;

    // node[b] = e_b
    for (int32_t b = 0; b < BATCH; ++b) {
        node_batch[b * DIM + b] = 1.0f;
    }

    float scores[BATCH] = {};
    DotProductL0_Batch4(query, node_batch, scores);

    // Only lane 0 should have score = 1.0, rest = 0.0
    EXPECT_FLOAT_EQ(scores[0], 1.0f) << "lane=0 (e_0 \\dot e_0)";
    for (int32_t b = 1; b < BATCH; ++b) {
        EXPECT_FLOAT_EQ(scores[b], 0.0f) << "lane=" << b << " (e_0 \\dot e_b)";
    }
}
