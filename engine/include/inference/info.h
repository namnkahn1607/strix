// Property constants of BERT transformer all-MiniLM-L6-v2.

#pragma once

namespace strix::inference {

// Maximum word piece sequence length accepted by `all-MiniLM-L6-v2`.
inline constexpr size_t kMaxTokens = 256;

inline constexpr size_t kVectorDim     = 384;
inline constexpr size_t kVectorMemsize = kVectorDim * sizeof(float);

}  // namespace strix::inference
