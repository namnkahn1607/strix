#pragma once

#include <array>

namespace strix::collection {

// Minimum cosine similarity threshold for a cache hit.
inline constexpr float kSimilarityThreshold = 0.85f;

inline constexpr uint32_t kTopK = 10;
static_assert(
    kTopK >= 1 && kTopK <= 16,
    "K > 16 requires a heap-based accumulator, not yet implemented."
);

struct SearchRecord {
    uint32_t node_id;
    uint8_t  version;
};

template <uint32_t K>
struct TopKResult {
    std::array<SearchRecord, K> records;
    uint32_t                    count = 0;
};

}  // namespace strix::collection
