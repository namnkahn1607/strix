// Property constants of BERT Transformer all-MiniLM-L6-v2.

#pragma once

constexpr size_t kVectorDim = 384;

constexpr size_t kVectorMemsize = kVectorDim * sizeof(float);

static_assert(
    kVectorDim % 8 == 0, "Vector dimension must be AVX2-register aligned."
);
