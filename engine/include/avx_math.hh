//
// Created by nlnk on Apr 20, 26.
//

#ifndef STRIX_ENGINE_AVX_MATH_HH
#define STRIX_ENGINE_AVX_MATH_HH

/* AMD/Intel x86 */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include "constant.hh"

__attribute__((target("avx2,fma"))) inline void CosineL0_Batch4_AVX2(
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

inline float CosineSimilarity(const float* query,
                              const float* node_vector) noexcept {
    // Apply 4 256-bit registers holding eight 0.0f's (unroll_factor = 4).
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (size_t i = 0; i < engine::VECTOR_DIM; i += 32) {
        // Load 8 floats of 'query'
        const __m256 q0 = _mm256_load_ps(query + i);
        const __m256 q1 = _mm256_load_ps(query + i + 8);
        const __m256 q2 = _mm256_load_ps(query + i + 16);
        const __m256 q3 = _mm256_load_ps(query + i + 24);

        // Load 8 floats of 'node_vector'
        const __m256 n0 = _mm256_load_ps(node_vector + i);
        const __m256 n1 = _mm256_load_ps(node_vector + i + 8);
        const __m256 n2 = _mm256_load_ps(node_vector + i + 16);
        const __m256 n3 = _mm256_load_ps(node_vector + i + 24);

        // Parallelism Fused Multiply-Add (ILP)
        sum0 = _mm256_fmadd_ps(q0, n0, sum0);
        sum1 = _mm256_fmadd_ps(q1, n1, sum1);
        sum2 = _mm256_fmadd_ps(q2, n2, sum2);
        sum3 = _mm256_fmadd_ps(q3, n3, sum3);
    }

    // Add into a single 256-bit register
    __m256 sum_vec =
        _mm256_add_ps(_mm256_add_ps(sum0, sum1),   // NOLINT(*-simd-intrinsics)
                      _mm256_add_ps(sum2, sum3));  // NOLINT(*-simd-intrinsics)

    // AVX2 divides 256-bit into 2 128-bit lanes
    const __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    const auto sum_high = _mm256_extractf128_ps(sum_vec, 1);

    // Sum 2 128-bit lane into a 128-bit (4 floats) register
    __m128 sum_128 =
        _mm_add_ps(sum_low, sum_high);  // NOLINT(*-simd-intrinsics)

    // Continue horizontal add on that 128-bit
    sum_128 = _mm_hadd_ps(sum_128, sum_128);  // 2 floats
    sum_128 = _mm_hadd_ps(sum_128, sum_128);  // 1 duplicated float

    return _mm_cvtss_f32(sum_128);
}

/* Scalar Fallback */
#else
#include "constant.hh"

inline float CosineSimilarity(const float* query,
                              const float* node_vector) noexcept {
    float dot_product{0.0f};

    for (size_t i = 0; i < engine::VECTOR_DIM; ++i) {
        dot_product += query[i] * node_vector[i];
    }

    return dot_product;
}

#endif

#endif  // STRIX_ENGINE_AVX_MATH_HH
