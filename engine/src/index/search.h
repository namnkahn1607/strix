// Author: namnkahn1607
//
// SearchResult type and SearchL0() declaration.
// SearchL0() performs a single-pass O(n) scan of the L0 buffer,
// returning the best matching node and an opportunistic free slot.

#pragma once

#include "arena.h"

// `SearchResult`
//
// Fields:
//   1. `best_node_id`     : index of a node with the highest dot-product score.
//                          -1 if no READY/MIGRATING node found during the scan.
//   2. `best_score`       : dot-product score of `best_node_id`.
//                          -1.0f when `best_node_id == -1`.
//   3. `reusable_node_id` : index of the first DEAD slot met during scan;
//                          opportunistically collected so the caller can
//                          reclaim it without a second pass.
//                          -1 if no DEAD slot exists.
struct SearchResult {
    int32_t best_node_id     = -1;
    float   best_score       = -1.0f;
    int32_t reusable_node_id = -1;
};

// SearchL0()
//
// Linear scan of the L0 buffer searching for the vector most similar to
// the vector `query`. Dot products are computed in batches of `kBatchSize` via
// the `DotProductBatch()` routine.
//
// `curr_time` is a Unix timestamp (seconds). PENDING nodes
// whose `(created_at + kPendingLifespan) <= curr_time` are treated as logically
// evicted and skipped; they must not influence the search result.
//
// The scan is a single pass: it simultaneously tracks the best READY node
// and the first DEAD slot (`reusable_node_id`), avoiding a second pass
// for slot allocation.
SearchResult SearchL0(MemoryArena& arena, const float* query,
                      uint64_t curr_time) noexcept;
