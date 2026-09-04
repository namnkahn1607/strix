// AVX2 + FMA accelerated dot product kernel.
// Fallback to scalar computing on non-x86_64 targets.

#pragma once

#include "inference/info.h"

namespace strix::ann {

// Number of vectors processed per dot product kernel call.
inline constexpr uint32_t kBatchSize = 4;

}  // namespace strix::ann

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

namespace strix::ann {

__attribute__((target("avx2,fma"))) inline void BatchDotProduct(
    const float* __restrict query, const float* __restrict v0,
    const float* __restrict v1, const float* __restrict v2,
    const float* __restrict v3, float* __restrict scores
) noexcept {
    // FMA has ~4 cycle latency and ~0.5 cycle throughput on target CPU.
    // Little's Law: latency / throughput = 8 in-flight accumulators.
    __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps();
    __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps();
    __m256 s20 = _mm256_setzero_ps(), s21 = _mm256_setzero_ps();
    __m256 s30 = _mm256_setzero_ps(), s31 = _mm256_setzero_ps();

    // 2 x `__m256` (16 floats) consumed each iteration - unroll factor = 2.
    // batch_size(4) x unroll_factor(2) = 8.
    constexpr size_t kFloatsEach = 16;
    for (size_t i = 0; i < inference::kVectorDim; i += kFloatsEach) {
        const __m256 q0 = _mm256_load_ps(query + i);
        const __m256 q1 = _mm256_load_ps(query + i + 8);

        s00 = _mm256_fmadd_ps(q0, _mm256_load_ps(v0 + i), s00);
        s01 = _mm256_fmadd_ps(q1, _mm256_load_ps(v0 + i + 8), s01);

        s10 = _mm256_fmadd_ps(q0, _mm256_load_ps(v1 + i), s10);
        s11 = _mm256_fmadd_ps(q1, _mm256_load_ps(v1 + i + 8), s11);

        s20 = _mm256_fmadd_ps(q0, _mm256_load_ps(v2 + i), s20);
        s21 = _mm256_fmadd_ps(q1, _mm256_load_ps(v2 + i + 8), s21);

        s30 = _mm256_fmadd_ps(q0, _mm256_load_ps(v3 + i), s30);
        s31 = _mm256_fmadd_ps(q1, _mm256_load_ps(v3 + i + 8), s31);
    }

    const __m256 acc0 = _mm256_add_ps(s00, s01);
    const __m256 acc1 = _mm256_add_ps(s10, s11);
    const __m256 acc2 = _mm256_add_ps(s20, s21);
    const __m256 acc3 = _mm256_add_ps(s30, s31);

    const __m128 ra = _mm_add_ps(
        _mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1)
    );
    const __m128 rb = _mm_add_ps(
        _mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1)
    );
    const __m128 rc = _mm_add_ps(
        _mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1)
    );
    const __m128 rd = _mm_add_ps(
        _mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1)
    );

    const __m128 ab_lo = _mm_unpacklo_ps(ra, rb);
    const __m128 ab_hi = _mm_unpackhi_ps(ra, rb);
    const __m128 cd_lo = _mm_unpacklo_ps(rc, rd);
    const __m128 cd_hi = _mm_unpackhi_ps(rc, rd);

    const __m128 ab = _mm_add_ps(ab_lo, ab_hi);
    const __m128 cd = _mm_add_ps(cd_lo, cd_hi);

    const __m128 lo = _mm_movelh_ps(ab, cd);
    const __m128 hi = _mm_movehl_ps(cd, ab);

    _mm_storeu_ps(scores, _mm_add_ps(lo, hi));
}

__attribute__((target("avx2,fma"))) inline void BatchDotProduct(
    const float* __restrict query, const float* __restrict batch,
    float* __restrict scores
) noexcept {
    const float* __restrict v0 = batch;
    const float* __restrict v1 = v0 + inference::kVectorDim;
    const float* __restrict v2 = v1 + inference::kVectorDim;
    const float* __restrict v3 = v2 + inference::kVectorDim;
    BatchDotProduct(query, v0, v1, v2, v3, scores);
}

}  // namespace strix::ann
#else
namespace strix::ann {

inline void BatchDotProduct(
    const float* __restrict query, const float* __restrict batch,
    float* __restrict scores
) noexcept {
    for (size_t k = 0; k < kBatchSize; ++k) {
        float dot = 0.0f;
        for (size_t i = 0; i < inference::kVectorDim; ++i) {
            dot += query[i] * node_batch[k * inference::kVectorDim + i];
        }
        scores[k] = dot;
    }
}

inline void BatchDotProduct(
    const float* __restrict query, const float* __restrict v0,
    const float* __restrict v1, const float* __restrict v2,
    const float* __restrict v3, float* __restrict scores
) noexcept {
    const float* __restrict batch[kBatchSize]{v0, v1, v2, v3};
    BatchDotProduct(query, batch, scores);
}

}  // namespace strix::ann
#endif
