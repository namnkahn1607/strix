// AVX2 + FMA accelerated dot product kernel.
// Fallback to scalar computing on non-x86_64 targets.

#pragma once

#include "inference/info.h"

namespace strix::ann {

// Number of vectors processed per dot product kernel call.
// On target hardware, FMA has ~4 cycle latency and ~0.5 cycle throughput, which
// needs 8 in-flight accumulator chains (Little's Law: latency / thoughput).
inline constexpr uint32_t kBatchSize = 4;

}  // namespace strix::ann

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

namespace strix::ann {

namespace {

// 2 x `__m256` (16 floats) consumed each iteration - unroll factor = 2.
// `batch(4) x unroll(2) = 8` matches the specification.
constexpr size_t kFloatsEach = 16;

static_assert(
    inference::kVectorDim % kFloatsEach == 0,
    "Vector dimension should divisible by number of floats each iteration"
);

}  // namespace

__attribute__((target("avx2,fma"))) inline __m128 PostProcess(
    __m256 acc0, __m256 acc1, __m256 acc2, __m256 acc3
) noexcept {
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

    return _mm_add_ps(lo, hi);
}

__attribute__((target("avx2,fma"))) inline void DotProductContiguousBatch(
    float* __restrict query, float* __restrict node_batch,
    float* __restrict scores
) noexcept {
    const float* __restrict n0 = node_batch;
    const float* __restrict n1 = n0 + inference::kVectorDim;
    const float* __restrict n2 = n1 + inference::kVectorDim;
    const float* __restrict n3 = n2 + inference::kVectorDim;

    __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps();
    __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps();
    __m256 s20 = _mm256_setzero_ps(), s21 = _mm256_setzero_ps();
    __m256 s30 = _mm256_setzero_ps(), s31 = _mm256_setzero_ps();

    for (size_t i = 0; i < inference::kVectorDim; i += kFloatsEach) {
        const __m256 q0 = _mm256_load_ps(query + i);
        const __m256 q1 = _mm256_load_ps(query + i + 8);

        s00 = _mm256_fmadd_ps(q0, _mm256_load_ps(n0 + i), s00);
        s01 = _mm256_fmadd_ps(q1, _mm256_load_ps(n0 + i + 8), s01);

        s10 = _mm256_fmadd_ps(q0, _mm256_load_ps(n1 + i), s10);
        s11 = _mm256_fmadd_ps(q1, _mm256_load_ps(n1 + i + 8), s11);

        s20 = _mm256_fmadd_ps(q0, _mm256_load_ps(n2 + i), s20);
        s21 = _mm256_fmadd_ps(q1, _mm256_load_ps(n2 + i + 8), s21);

        s30 = _mm256_fmadd_ps(q0, _mm256_load_ps(n3 + i), s30);
        s31 = _mm256_fmadd_ps(q1, _mm256_load_ps(n3 + i + 8), s31);
    }

    const __m256 acc0 = _mm256_add_ps(s00, s01);
    const __m256 acc1 = _mm256_add_ps(s10, s11);
    const __m256 acc2 = _mm256_add_ps(s20, s21);
    const __m256 acc3 = _mm256_add_ps(s30, s31);

    _mm_storeu_ps(scores, PostProcess(acc0, acc1, acc2, acc3));
}

__attribute__((target("avx2,fma"))) inline void DotProductDiscreteBatch(
    float* __restrict query, float* __restrict v0, float* __restrict v1,
    float* __restrict v2, float* __restrict v3, float* __restrict scores
) noexcept {
    __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps();
    __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps();
    __m256 s20 = _mm256_setzero_ps(), s21 = _mm256_setzero_ps();
    __m256 s30 = _mm256_setzero_ps(), s31 = _mm256_setzero_ps();

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

    _mm_storeu_ps(scores, PostProcess(acc0, acc1, acc2, acc3));
}

}  // namespace strix::ann
#else
namespace strix::ann {

inline void DotProductContiguousBatch(
    float* __restrict query, float* __restrict node_batch,
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

inline void DotProductDiscreteBatch(
    float* __restrict query, float* __restrict v0, float* __restrict v1,
    float* __restrict v2, float* __restrict v3, float* __restrict scores
) noexcept {
    const float* __restrict batch_vec[kBatchSize]{v0, v1, v2, v3};
    for (size_t k = 0; k < kBatchSize; ++k) {
        float dot = 0.0f;
        for (size_t i = 0; i < inference::kVectorDim; ++i) {
            dot += query[i] * batch_vec[k][i];
        }
        scores[k] = dot;
    }
}

}  // namespace strix::ann
#endif
