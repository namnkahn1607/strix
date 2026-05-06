//
// Created by nlnk on Apr 20, 26.
//

#ifndef STRIX_ENGINE_AVX_MATH_HH
#define STRIX_ENGINE_AVX_MATH_HH

/* AMD/Intel x86 */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include "constant.hh"

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
