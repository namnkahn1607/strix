// AVX2 + FMA accelerated dot product kernel unit test.
// A scalar-compute implementation is used as the correctness oracle.

#include "index/avx2_kernel.h"

#include <gtest/gtest.h>
#include <mm_malloc.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <random>

#include "common/constants.h"

namespace {

inline constexpr float kApprox4 = 1e-4f;

// Normalize scales a vector to a magnitude length of `1`.
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

// ScalarDotProduct computes dot product value between 2 normalized vectors
// in a scalar way. Used as correctness oracle.
// `query` and `node_vector` will be normalized.
float ScalarDotProduct(float* query, float* node_vector) {
    Normalize(query, kVectorDim);
    Normalize(node_vector, kVectorDim);

    float sum = 0.0f;
    for (size_t i = 0; i < kVectorDim; ++i) {
        sum += (query[i] * node_vector[i]);
    }
    return sum;
}

}  // namespace

class DotProductBatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        query_      = AllocVectorBuf(kVectorMemsize);
        node_batch_ = AllocVectorBuf(kBatchSize * kVectorMemsize);
    }

    void TearDown() override {
        _mm_free(query_);
        _mm_free(node_batch_);
    }

    // Refresh `query_` and `node_batch_` with randomized values, then
    // normalize them (again).
    void Initialize(const uint64_t seed = 0x123) const noexcept {
        std::mt19937                   gen(seed);
        std::uniform_real_distribution dist(-1.0f, 1.0f);

        for (uint32_t i = 0; i < kVectorDim; ++i) {
            query_[i] = dist(gen);
        }
        Normalize(query_, kVectorDim);

        for (uint32_t b = 0; b < kBatchSize; ++b) {
            float* node = node_batch_ + b * kVectorDim;
            for (size_t i = 0; i < kVectorDim; ++i) {
                node[i] = dist(gen);
            }
            Normalize(node, kVectorDim);
        }
    }

    // Compute using the API provided by the dot product kernel.
    // The results of each are written onto the buffers received, which must be
    // of size `kBatchSize`.
    void Compute(float* contig_scores, float* disc_scores) const noexcept {
        DotProductContiguousBatch(query_, node_batch_, contig_scores);
        DotProductDiscreteBatch(
            query_, node_batch_, node_batch_ + kVectorDim,
            node_batch_ + 2 * kVectorDim, node_batch_ + 3 * kVectorDim,
            disc_scores
        );
    }

    // Grade a `scores` buffer against the results produced by the scalar
    // oracle. Approximate by `10e-4`.
    void GradeScalar(float* scores) const noexcept {
        for (uint32_t b = 0; b < kBatchSize; ++b) {
            const float expected =
                ScalarDotProduct(query_, node_batch_ + b * kVectorDim);
            EXPECT_NEAR(scores[b], expected, kApprox4) << "lane=" << b;
        }
    }

    float* query_      = nullptr;
    float* node_batch_ = nullptr;

private:
    float* AllocVectorBuf(const size_t size) {
        constexpr uint32_t kAlign = 32;

        auto* raw = _mm_malloc(size, kAlign);
        if (raw == nullptr) {
            throw std::bad_alloc();
        }

        return static_cast<float*>(raw);
    }
};

// -----------------------------------------------------------------------------
// Deterministic
// Identical input must produce bitwise-identical output across repeated calls,
// as FMA is deterministic on the same CPU.
// Catches any accidental use of non-deterministic intrinsics.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, Deterministic) {
    constexpr uint32_t kTotalCalls = 100;
    Initialize(0xDEADBEEF);

    float init_contig[kBatchSize] = {};
    float init_disc[kBatchSize]   = {};
    Compute(init_contig, init_disc);

    for (uint32_t c = 0; c < kTotalCalls; ++c) {
        float curr_contig[kBatchSize] = {};
        float curr_disc[kBatchSize]   = {};
        Compute(curr_contig, curr_disc);

        for (uint32_t b = 0; b < kBatchSize; ++b) {
            EXPECT_EQ(curr_contig[b], init_contig[b])
                << "lane=" << b << "call=" << c;
            EXPECT_EQ(curr_disc[b], init_disc[b])
                << "lane=" << b << "call=" << c;

            // Cross-API deterministic
            EXPECT_NEAR(curr_contig[b], curr_disc[b], kApprox4)
                << "lane=" << b << "call=" << c;
        }
    }
}

// -----------------------------------------------------------------------------
// ZeroQuery
// When the query vector is all zeros, all dot products must be exactly 0.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, ZeroQuery) {
    Initialize(0xDEADC0DE);
    std::fill_n(query_, kVectorDim, 0.0f);

    float contig_scores[kBatchSize] = {};
    float disc_scores[kBatchSize]   = {};
    Compute(contig_scores, disc_scores);

    for (uint32_t b = 0; b < kBatchSize; ++b) {
        EXPECT_FLOAT_EQ(contig_scores[b], 0.0f) << "lane=" << b;
        EXPECT_FLOAT_EQ(disc_scores[b], 0.0f) << "lane=" << b;
    }
}

// -----------------------------------------------------------------------------
// CompareWithScalarOracle
// Multiple iterations with random seed to ensure identical behavior between the
// FMA-accelerated kernel and the scalar one.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, CompareWithScalarOracle) {
    constexpr uint64_t kAttempt = 1'000u;
    for (uint64_t atmp = 0; atmp < kAttempt; ++atmp) {
        Initialize(atmp);
        float contig_scores[kBatchSize] = {};
        float disc_scores[kBatchSize]   = {};
        Compute(contig_scores, disc_scores);
        GradeScalar(contig_scores);
        GradeScalar(disc_scores);
    }
}

// -----------------------------------------------------------------------------
// IdenticalVectors
// All lanes are the same as the query vector, and dot product of a unit vector
// with itself must equal 1.0.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, IdenticalVectors) {
    Initialize();
    for (uint32_t b = 0; b < kBatchSize; ++b) {
        std::copy_n(query_, kVectorDim, node_batch_ + b * kVectorDim);
    }

    float contig_scores[kBatchSize] = {};
    float disc_scores[kBatchSize]   = {};
    Compute(contig_scores, disc_scores);
    GradeScalar(contig_scores);
    GradeScalar(disc_scores);
}

// -----------------------------------------------------------------------------
// OppositeVectors
// All lanes are the opposing to the query vector, and all dot products are
// expected to be -1.0.
// -----------------------------------------------------------------------------
TEST_F(DotProductBatchTest, OppositeVectors) {
    constexpr float kExpectedDotProduct = -1.0f;

    Initialize();
    for (uint32_t b = 0; b < kBatchSize; ++b) {
        float* node = node_batch_ + b * kVectorDim;
        for (size_t i = 0; i < kVectorDim; ++i) {
            node[i] = -query_[i];
        }
    }

    float contig_scores[kBatchSize] = {};
    float disc_scores[kBatchSize]   = {};
    Compute(contig_scores, disc_scores);

    for (uint32_t b = 0; b < kBatchSize; ++b) {
        EXPECT_NEAR(contig_scores[b], kExpectedDotProduct, kApprox4)
            << "lane=" << b;
        EXPECT_NEAR(disc_scores[b], kExpectedDotProduct, kApprox4)
            << "lane=" << b;
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
    const char* kLaneNames[kBatchSize] = {
        "identical", "opposite", "orthogonal", "random"
    };

    // Lane 3: Total random.
    Initialize(0xCAFEBABE);

    // Lane 0: Identical.
    std::copy_n(query_, kVectorDim, node_batch_);

    // Lane 1: Opposite.
    for (size_t i = 0; i < kVectorDim; ++i) {
        node_batch_[kVectorDim + i] = -query_[i];
    }

    // Lane 2: Orthogonal.
    {
        float* v2 = node_batch_ + 2 * kVectorDim;
        std::fill_n(v2, kVectorDim, 0.0f);

        // Gram-Schmidt: v2 = e_1 - (e_1 \dot query_) * query_.
        v2[0] = 1.0f;
        // e_1 \dot query_ = query_[0].
        const float proj = query_[0];
        for (size_t i = 0; i < kVectorDim; ++i) {
            v2[i] -= proj * query_[i];
        }
        Normalize(v2, kVectorDim);
    }

    float expected_scores[kBatchSize] = {
        1.0f, -1.0f, 0.0f,
        ScalarDotProduct(query_, node_batch_ + 3 * kVectorDim)
    };
    float contig_scores[kBatchSize] = {};
    float disc_scores[kBatchSize]   = {};
    Compute(contig_scores, disc_scores);

    for (uint32_t b = 0; b < kBatchSize; ++b) {
        const float expected = expected_scores[b];
        EXPECT_NEAR(contig_scores[b], expected, kApprox4)
            << "lane=" << b << "(" << kLaneNames[b] << ")";
        EXPECT_NEAR(disc_scores[b], expected, kApprox4)
            << "lane=" << b << "(" << kLaneNames[b] << ")";
    }
}
