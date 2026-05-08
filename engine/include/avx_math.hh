//
// Created by nlnk on Apr 20, 26.
//

#ifndef STRIX_ENGINE_AVX_MATH_HH
#define STRIX_ENGINE_AVX_MATH_HH

/* AMD/Intel x86 */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include "constant.hh"

__attribute__((target("avx2,fma"))) inline void CosineL0_Batch4(
    const float* __restrict__ query, const float* __restrict__ node_batch,
    float* __restrict__ scores) noexcept {
    constexpr size_t DIM = engine::VECTOR_DIM;

    const float* __restrict__ n0 = node_batch;
    const float* __restrict__ n1 = node_batch + DIM;
    const float* __restrict__ n2 = node_batch + DIM * 2;
    const float* __restrict__ n3 = node_batch + DIM * 3;

    __m256 sA0 = _mm256_setzero_ps(), sA1 = _mm256_setzero_ps();
    __m256 sB0 = _mm256_setzero_ps(), sB1 = _mm256_setzero_ps();
    __m256 sC0 = _mm256_setzero_ps(), sC1 = _mm256_setzero_ps();
    __m256 sD0 = _mm256_setzero_ps(), sD1 = _mm256_setzero_ps();

    for (size_t i = 0; i < DIM; i += 16) {
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

    const __m256 sumA = _mm256_add_ps(sA0, sA1);  // NOLINT(*-simd-intrinsics)
    const __m256 sumB = _mm256_add_ps(sB0, sB1);  // NOLINT(*-simd-intrinsics)
    const __m256 sumC = _mm256_add_ps(sC0, sC1);  // NOLINT(*-simd-intrinsics)
    const __m256 sumD = _mm256_add_ps(sD0, sD1);  // NOLINT(*-simd-intrinsics)

    const __m128 rA =
        _mm_add_ps(_mm256_castps256_ps128(sumA),  // NOLINT(*-simd-intrinsics)
                   _mm256_extractf128_ps(sumA, 1));
    const __m128 rB =
        _mm_add_ps(_mm256_castps256_ps128(sumB),  // NOLINT(*-simd-intrinsics)
                   _mm256_extractf128_ps(sumB, 1));
    const __m128 rC =
        _mm_add_ps(_mm256_castps256_ps128(sumC),  // NOLINT(*-simd-intrinsics)
                   _mm256_extractf128_ps(sumC, 1));
    const __m128 rD =
        _mm_add_ps(_mm256_castps256_ps128(sumD),  // NOLINT(*-simd-intrinsics)
                   _mm256_extractf128_ps(sumD, 1));

    const __m128 ab_lo = _mm_unpacklo_ps(rA, rB);
    const __m128 ab_hi = _mm_unpackhi_ps(rA, rB);
    const __m128 cd_lo = _mm_unpacklo_ps(rC, rD);
    const __m128 cd_hi = _mm_unpackhi_ps(rC, rD);

    const __m128 ab = _mm_add_ps(ab_lo, ab_hi);  // NOLINT(*-simd-intrinsics)
    const __m128 cd = _mm_add_ps(cd_lo, cd_hi);  // NOLINT(*-simd-intrinsics)

    const __m128 lo = _mm_movelh_ps(ab, cd);
    const __m128 hi = _mm_movehl_ps(cd, ab);

    _mm_storeu_ps(scores, _mm_add_ps(lo, hi));  // NOLINT(*-simd-intrinsics)
}

/* Scalar Fallback */
#else
#include "constant.hh"

inline void CosineL0_Batch4(const float* __restrict__ query,
                            const float* __restrict__ node_batch,
                            float* __restrict__ scores) noexcept {
    constexpr size_t DIM = engine::VECTOR_DIM;

    for (size_t i = 0; i < DIM; ++i) {
        for (int k = 0; k < 4; ++k) {
            scores[k] += query[i] * node_batch[k * DIM + i];
        }
    }
}

#endif

#endif  // STRIX_ENGINE_AVX_MATH_HH
