// Author: namnkahn1607
//
// Cross-module compile-time constants.
// Only constants needed by MORE THAN ONE module live here.
// Module-private constants belong in their own translation units.

#pragma once

#include <cstddef>

// Embedding vector dimension produced by all-MiniLM-L6-v2.
// kVectorMemsize is the corresponding byte footprint of one vector.
inline constexpr size_t kVectorDim     = 384;
inline constexpr size_t kVectorMemsize = kVectorDim * sizeof(float);

// Minimum cosine similarity for a cache hit.
inline constexpr float kSimilarityThreshold = 0.85f;

// Per-tier slot limits.
// kTotalMaxSlots is the sum; used to size the unified MemoryArena.
inline constexpr size_t kL0MaxSlots    = 1'024;
inline constexpr size_t kL1MaxSlots    = 524'288;
inline constexpr size_t kTotalMaxSlots = kL0MaxSlots + kL1MaxSlots;
