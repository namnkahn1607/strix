// Properties of BERT transformer all-MiniLM-L6-v2.

#pragma once

#include <cstddef>

namespace strix::inference {

// Maximum word piece sequence length accepted by `all-MiniLM-L6-v2`.
inline constexpr size_t kMaxTokens = 256;

// Output embedding vector dimension.
inline constexpr size_t kVectorDim     = 384;
inline constexpr size_t kVectorMemsize = kVectorDim * sizeof(float);

static_assert(
    kVectorDim % 8 == 0, "Vector dimension must fit in AVX2 registers."
);

}  // namespace strix::inference
