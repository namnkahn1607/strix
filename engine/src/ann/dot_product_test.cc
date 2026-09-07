// AVX2 + FMA accelerated dot product kernel unit test.

#include "ann/dot_product.h"

#include <gtest/gtest.h>
#include <mm_malloc.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <random>

#include "inference/info.h"

namespace ann       = strix::ann;
namespace inference = strix::inference;

namespace {

constexpr float kApprox4 = 1e-4f;

float* AllocVecBuf(size_t num_vec) {
    constexpr uint32_t kAlign = 32;

    auto* raw = _mm_malloc(num_vec * inference::kVectorMemsize, kAlign);
    if (raw == nullptr) {
        throw std::bad_alloc();
    }
    return static_cast<float*>(raw);
}

void Normalize(float* vec, const size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }

    norm = 1 / std::sqrt(norm);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] *= norm;
    }
}

// Computes dot product value between 2 vectors in a scalar way.
// Used as correctness oracle. Expects normalized vectors.
float DotProductScalar(float* query, float* node_vec) {
    float sum = 0.0f;
    for (size_t i = 0; i < inference::kVectorDim; ++i) {
        sum += (query[i] * node_vec[i]);
    }
    return sum;
}

}  // namespace

class DotProductBatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        query_ = AllocVecBuf(1);
        batch_ = AllocVecBuf(ann::kBatchSize);
    }

    void TearDown() override {
        _mm_free(query_);
        _mm_free(batch_);
    }

    // Renews candidates with random values and re-normalizes.
    void Refresh(uint64_t seed = 0x123) const noexcept {
        std::mt19937                   gen{seed};
        std::uniform_real_distribution dist{-1.0f, 1.0f};

        for (uint32_t i = 0; i < inference::kVectorDim; ++i) {
            query_[i] = dist(gen);
        }
        Normalize(query_, inference::kVectorDim);

        for (uint32_t b = 0; b < ann::kBatchSize; ++b) {
            float* node = batch_ + b * inference::kVectorDim;
            for (size_t i = 0; i < inference::kVectorDim; ++i) {
                node[i] = dist(gen);
            }
            Normalize(node, inference::kVectorDim);
        }
    }

    // Computes dot product values using API provided by kernel.
    void Compute(float* scores) const noexcept {
        ann::BatchDotProduct(query_, batch_, scores);
    }

    void GradeAgainstScalar(float* scores) const noexcept {
        for (uint32_t b = 0; b < ann::kBatchSize; ++b) {
            const float expected =
                DotProductScalar(query_, batch_ + b * inference::kVectorDim);
            EXPECT_NEAR(scores[b], expected, kApprox4) << "lane=" << b;
        }
    }

    float* query_ = nullptr;
    float* batch_ = nullptr;
};

// -----------------------------------------------------------------------------
// Deterministic
// Identical input must produce bitwise-identical output across repeated calls,
// as FMA is deterministic within same CPU.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, Deterministic) {
    constexpr uint32_t kTotalCalls = 128;

    Refresh(0xDEADBEEF);
    float init_scr[ann::kBatchSize] = {};
    Compute(init_scr);

    for (uint32_t c = 0; c < kTotalCalls; ++c) {
        float curr_scr[ann::kBatchSize] = {};
        Compute(curr_scr);
        for (uint32_t b = 0; b < ann::kBatchSize; ++b) {
            EXPECT_EQ(curr_scr[b], init_scr[b]) << "lane=" << b << "call=" << c;
        }
    }
}

// -----------------------------------------------------------------------------
// ZeroQuery
// When the query vector is all zeros, all dot products must be exactly 0.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, ZeroQuery) {
    Refresh(0xDEADC0DE);
    std::fill_n(query_, inference::kVectorDim, 0.0f);

    float scores[ann::kBatchSize] = {};
    Compute(scores);
    for (uint32_t b = 0; b < ann::kBatchSize; ++b) {
        EXPECT_FLOAT_EQ(scores[b], 0.0f) << "lane=" << b;
    }
}

// -----------------------------------------------------------------------------
// CompareWithScalarOracle
// Multiple iterations with random seed to ensure identical behavior between the
// FMA-accelerated kernel and the scalar one.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, CompareWithScalarOracle) {
    constexpr uint64_t kAttempt = 1'000;
    for (uint64_t atmp = 0; atmp < kAttempt; ++atmp) {
        Refresh(atmp);
        float scores[ann::kBatchSize] = {};
        Compute(scores);
        GradeAgainstScalar(scores);
    }
}

// -----------------------------------------------------------------------------
// IdenticalVectors
// All lanes are the same as the query vector, and dot product of a unit vector
// with itself must equal 1.0.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, IdenticalVectors) {
    Refresh();
    for (uint32_t b = 0; b < ann::kBatchSize; ++b) {
        std::copy_n(
            query_, inference::kVectorDim, batch_ + b * inference::kVectorDim
        );
    }

    float scores[ann::kBatchSize] = {};
    Compute(scores);
    GradeAgainstScalar(scores);
}

// -----------------------------------------------------------------------------
// OppositeVectors
// All lanes are the opposing to the query vector, and all dot products are
// expected to be -1.0.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, OppositeVectors) {
    constexpr float kExpected = -1.0f;

    Refresh();
    for (uint32_t b = 0; b < ann::kBatchSize; ++b) {
        float* node = batch_ + b * inference::kVectorDim;
        for (size_t i = 0; i < inference::kVectorDim; ++i) {
            node[i] = -query_[i];
        }
    }

    float scores[ann::kBatchSize] = {};
    Compute(scores);
    for (uint32_t b = 0; b < ann::kBatchSize; ++b) {
        EXPECT_NEAR(scores[b], kExpected, kApprox4) << "lane=" << b;
    }
}

// -----------------------------------------------------------------------------
// MixedLanes
// Each lane holds a different vector relationship to query:
//   Lane 0: identical to query   -> score ~ +1.0
//   Lane 1: opposite to query    -> score ~ -1.0
//   Lane 2: orthogonal to query  -> score ~  0.0
//   Lane 3: random vec           -> score against the scalar oracle
//
// Verifies that horizontal reduction does not bleed values across lanes.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, MixedLanes) {
    // Lane 3: Total random.
    Refresh(0xCAFEBABE);
    float res = DotProductScalar(query_, batch_ + 3 * inference::kVectorDim);

    // Lane 0: Identical.
    std::copy_n(query_, inference::kVectorDim, batch_);

    // Lane 1: Opposite.
    for (size_t i = 0; i < inference::kVectorDim; ++i) {
        batch_[inference::kVectorDim + i] = -query_[i];
    }

    // Lane 2: Orthogonal.
    {
        float* v2 = batch_ + 2 * inference::kVectorDim;
        std::fill_n(v2, inference::kVectorDim, 0.0f);

        // Gram-Schmidt: v2 = e_1 - (e_1 \dot query_) * query_.
        v2[0] = 1.0f;

        // e_1 \dot query_ = query_[0].
        const float proj = query_[0];
        for (size_t i = 0; i < inference::kVectorDim; ++i) {
            v2[i] -= proj * query_[i];
        }
        Normalize(v2, inference::kVectorDim);
    }

    float exp_scores[ann::kBatchSize] = {1.0f, -1.0f, 0.0f, res};
    float act_scores[ann::kBatchSize] = {};
    Compute(act_scores);

    const char* kLaneNames[ann::kBatchSize] = {
        "identical", "opposite", "orthogonal", "random"
    };
    for (uint32_t b = 0; b < ann::kBatchSize; ++b) {
        EXPECT_NEAR(act_scores[b], exp_scores[b], kApprox4)
            << "lane=" << b << "(" << kLaneNames[b] << ")";
    }
}
