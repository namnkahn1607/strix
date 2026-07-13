// Author: namnkahn1607
//
// AVX2 + FMA accelerated dot product kernel.
// Falls back to a scalar loop on non-x86_64 targets.

#pragma once

#include "constants.h"

// Number of vectors processed per dot product kernel call. Hardwired to 4
// as each of the 4 accumulators maps to to one YMM lane pair, and the final
// horizontal reduction folds all 4 scores in a single _mm_add_ps pass.
// Changing this constant requires re-writing the kernel.
inline constexpr uint32_t kBatchSize = 4;

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

// `DotProductBatch()` computes dot product between one query vector and a
// contiguous batch of `kBatchSize` node vectors, writing result to `scores`.
//
// Memory layout:
//   1. `query`      : 32-bit aligned, length of `kVectorDim` floats.
//   2. `node_batch` : 32-bit aligned, `kBatchSize` * `kVectorDim` floats.
//   3. `scores`     : writable buffer of at least `kBatchSize` floats.
//
// `kVectorDim` must be a multiple of 16 (two AVX2 registers per iteration).
// The loop is fully unrolled accross 4 nodes to maximize ILP.
__attribute__((target("avx2,fma"))) inline void DotProductBatch(
    const float* __restrict query, const float* __restrict node_batch,
    float* __restrict scores) noexcept {
    const float* __restrict__ n0 = node_batch;
    const float* __restrict__ n1 = node_batch + kVectorDim;
    const float* __restrict__ n2 = node_batch + kVectorDim * 2;
    const float* __restrict__ n3 = node_batch + kVectorDim * 3;

    // 2 accumulator registers per node to hide FMA latency (~4 cycles).
    __m256 sA0 = _mm256_setzero_ps(), sA1 = _mm256_setzero_ps();
    __m256 sB0 = _mm256_setzero_ps(), sB1 = _mm256_setzero_ps();
    __m256 sC0 = _mm256_setzero_ps(), sC1 = _mm256_setzero_ps();
    __m256 sD0 = _mm256_setzero_ps(), sD1 = _mm256_setzero_ps();

    // Each iteration consumes 16 floats (2 x __m256) from query and each node.
    constexpr size_t kFloatsEach = 16;
    for (size_t i = 0; i < kVectorDim; i += kFloatsEach) {
        const __m256 q0 = _mm256_load_ps(query + i);
        const __m256 q1 = _mm256_load_ps(query + i + 8);

        sA0 = _mm256_fmadd_ps(q0, _mm256_load_ps(n0 + i), sA0);
        sA1 = _mm256_fmadd_ps(q1, _mm256_load_ps(n0 + i + 8), sA1);

        sB0 = _mm256_fmadd_ps(q0, _mm256_load_ps(n1 + i), sB0);
        sB1 = _mm256_fmadd_ps(q1, _mm256_load_ps(n1 + i + 8), sB1);

        sC0 = _mm256_fmadd_ps(q0, _mm256_load_ps(n2 + i), sC0);
        sC1 = _mm256_fmadd_ps(q1, _mm256_load_ps(n2 + i + 8), sC1);

        sD0 = _mm256_fmadd_ps(q0, _mm256_load_ps(n3 + i), sD0);
        sD1 = _mm256_fmadd_ps(q1, _mm256_load_ps(n3 + i + 8), sD1);
    }

    // Fold the 2 accumulators for each node into a single __m256.
    const __m256 sumA = _mm256_add_ps(sA0, sA1);
    const __m256 sumB = _mm256_add_ps(sB0, sB1);
    const __m256 sumC = _mm256_add_ps(sC0, sC1);
    const __m256 sumD = _mm256_add_ps(sD0, sD1);

    // Reduce each __m256 to a scalar by adding its two __m128 halves.
    const __m128 rA = _mm_add_ps(_mm256_castps256_ps128(sumA),
                                 _mm256_extractf128_ps(sumA, 1));
    const __m128 rB = _mm_add_ps(_mm256_castps256_ps128(sumB),
                                 _mm256_extractf128_ps(sumB, 1));
    const __m128 rC = _mm_add_ps(_mm256_castps256_ps128(sumC),
                                 _mm256_extractf128_ps(sumC, 1));
    const __m128 rD = _mm_add_ps(_mm256_castps256_ps128(sumD),
                                 _mm256_extractf128_ps(sumD, 1));

    // Interleave pairs so that the final _mm_add_ps produces [A, B, C, D].
    const __m128 ab_lo = _mm_unpacklo_ps(rA, rB);
    const __m128 ab_hi = _mm_unpackhi_ps(rA, rB);
    const __m128 cd_lo = _mm_unpacklo_ps(rC, rD);
    const __m128 cd_hi = _mm_unpackhi_ps(rC, rD);

    const __m128 ab = _mm_add_ps(ab_lo, ab_hi);
    const __m128 cd = _mm_add_ps(cd_lo, cd_hi);

    const __m128 lo = _mm_movelh_ps(ab, cd);
    const __m128 hi = _mm_movehl_ps(cd, ab);

    // `scores` need not be aligned; storeu is safe and has identical
    // throughput to storea.
    _mm_storeu_ps(scores, _mm_add_ps(lo, hi));
}

// `DotProductIndirectBatch()`; same register-blocked AVX2/FMA strategy as
// `DotProductBatch()`, but receives 4 independent pointers to vectors
__attribute__((target("avx2,fma"))) inline void DotProductIndirectBatch(
    const float* __restrict query, const float* __restrict v0,
    const float* __restrict v1, const float* __restrict v2,
    const float* __restrict v3, float* __restrict scores) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    // Each iteration consumes 8 floats (1 x __m256) from query and each vector.
    constexpr size_t kFloatsEach = 8;
    for (size_t i = 0; i < kVectorDim; i += kFloatsEach) {
        const __m256 q = _mm256_load_ps(query + i);

        acc0 = _mm256_fmadd_ps(q, _mm256_load_ps(v0 + i), acc0);
        acc1 = _mm256_fmadd_ps(q, _mm256_load_ps(v1 + i), acc1);
        acc2 = _mm256_fmadd_ps(q, _mm256_load_ps(v2 + i), acc2);
        acc3 = _mm256_fmadd_ps(q, _mm256_load_ps(v3 + i), acc3);
    }

    // Reduce each __m256 to a scalar by adding its two __m128 halves.
    const __m128 rA = _mm_add_ps(_mm256_castps256_ps128(acc0),
                                 _mm256_extractf128_ps(acc0, 1));
    const __m128 rB = _mm_add_ps(_mm256_castps256_ps128(acc1),
                                 _mm256_extractf128_ps(acc1, 1));
    const __m128 rC = _mm_add_ps(_mm256_castps256_ps128(acc2),
                                 _mm256_extractf128_ps(acc2, 1));
    const __m128 rD = _mm_add_ps(_mm256_castps256_ps128(acc3),
                                 _mm256_extractf128_ps(acc3, 1));

    // Interleave pairs so that the final _mm_add_ps produces [A, B, C, D].
    const __m128 ab_lo = _mm_unpacklo_ps(rA, rB);
    const __m128 ab_hi = _mm_unpackhi_ps(rA, rB);
    const __m128 cd_lo = _mm_unpacklo_ps(rC, rD);
    const __m128 cd_hi = _mm_unpackhi_ps(rC, rD);

    const __m128 ab = _mm_add_ps(ab_lo, ab_hi);
    const __m128 cd = _mm_add_ps(cd_lo, cd_hi);

    const __m128 lo = _mm_movelh_ps(ab, cd);
    const __m128 hi = _mm_movehl_ps(cd, ab);

    _mm_storeu_ps(scores, _mm_add_ps(lo, hi));
}

#else  // non-x86_64 scalar fallback

// Scalar fallback for `DotProductBatch()`. Row-major traversal over
// `kBatchSize` nodes; no SIMD, no alignment requirements.
inline void DotProductBatch(const float* __restrict__ query,
                            const float* __restrict__ node_batch,
                            float* __restrict__ scores) noexcept {
    for (size_t k = 0; k < kBatchSize; ++k) {
        float dot = 0.0f;

        for (size_t i = 0; i < kVectorDim; ++i) {
            dot += query[i] * node_batch[k * kVectorDim + i];
        }

        scores[k] = dot;
    }
}

// Scalar fallback for `DotProductIndirectBatch()`. Put all vectors in a batch
// of `kBatchSize` then perform row-major order traversal.
// No SIMD, no alignment requirements.
inline void DotProductIndirectBatch(const float* __restrict__ query,
                                    const float* __restrict__ v0,
                                    const float* __restrict__ v1,
                                    const float* __restrict__ v2,
                                    const float* __restrict__ v3,
                                    float* __restrict__ scores) noexcept {
    const float* __restrict__ batch_vec[kBatchSize]{v0, v1, v2, v3};
    for (size_t k = 0; k < kBatchSize; ++k) {
        float dot = 0.0f;

        for (size_t i = 0; i < kVectorDim; ++i) {
            dot += query[i] * batch_vec[k][i];
        }

        scores[k] = dot;
    }
}

#endif
