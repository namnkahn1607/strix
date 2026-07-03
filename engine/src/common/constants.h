// Author: namnkahn1607
//
// Cross-module compile-time constants.
// Only constants needed by MORE THAN ONE module live here.
// Module-private constants belong in their own translation units.

#pragma once

#include <cstddef>
#include <cstdint>

// Embedding vector dimension produced by all-MiniLM-L6-v2.
// kVectorMemsize is the corresponding byte footprint of one vector.
inline constexpr size_t kVectorDim     = 384;
inline constexpr size_t kVectorMemsize = kVectorDim * sizeof(float);

// Minimum cosine similarity for a cache hit.
inline constexpr float kSimilarityThreshold = 0.85f;

// Per-tier slot limits.
// kTotalMaxSlots is the sum; used to size the unified MemoryArena.
inline constexpr size_t kTotalSlots = 1 << 19;  // 524'288

// Maximum lifetime of a PENDING node in seconds.
// Nodes that remain PENDING beyond this deadline are treated as stale by
// the GC sweeper and by SearchL0 (skipped during scan).
inline constexpr uint32_t kPendingLifespan = 30;
