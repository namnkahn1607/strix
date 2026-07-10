// Author: namnkahn1607
//
// Cross-module compile-time constants.
// Only constants needed by MORE THAN ONE module live here.
// Module-private constants belong in their own translation units.

#pragma once

#include <cstdint>

// Embedding vector dimension produced by `all-MiniLM-L6-v2`.
inline constexpr uint32_t kVectorDim = 384;
// The corresponding byte footprint of one vector.
inline constexpr uint32_t kVectorMemsize = kVectorDim * sizeof(float);

// Minimum cosine similarity for a cache hit.
inline constexpr float kSimilarityThreshold = 0.85f;

// L0 buffer's slot limits.
inline constexpr uint32_t kL0Capacity = 1 << 12;
// Total number of slots in the unified `MemoryArena`.
inline constexpr uint32_t kTotalSlots = 1 << 19;

// Maximum lifetime of a `kPending` node in seconds.
// Nodes remain `kPending` beyond this deadline are treated as stale by the GC.
inline constexpr uint32_t kPendingLifespan = 30;
