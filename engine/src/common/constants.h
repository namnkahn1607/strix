// Cross-module compile-time constants.
// Only constants needed by MORE THAN ONE modules live here.
// Module-private constants belong in their own translation units.

#pragma once

#include <cstdint>
#include <cstdlib>

// Embedding vector dimension produced by `all-MiniLM-L6-v2`.
inline constexpr size_t kVectorDim = 384;
// Byte footprint of one vector, used for buffer allocation sizing.
inline constexpr size_t kVectorMemsize = kVectorDim * sizeof(float);

// Minimum cosine similarity for a cache hit.
// NOTE: single-threshold heuristic; should be a runtime configurable.
inline constexpr float kSimilarityThreshold = 0.85f;

// L0 (hot-tier) buffer capacity, in slots. Sized to to fit in L2/L3 cache
// footprint for the frontier dynamic working set.
inline constexpr uint32_t kL0Capacity = 1 << 12;
// Total number of slots in the unified `MemoryArena` (all tiers combined)
inline constexpr uint32_t kTotalSlots = 1 << 19;

// Maximum lifetime of a `kPending` node in seconds.
// Nodes remaining `kPending` beyond this deadline are treated as stale by GC.
inline constexpr uint32_t kPendingLifespan = 30;
