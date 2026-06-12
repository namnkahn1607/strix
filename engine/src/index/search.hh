//
// index/search.hh
//

#include "arena.hh"

// --- SearchResult ---
// Returned by MemoryArena::SearchL0().
//  best_node_id    : node with highest dot-product score. -1 if none found.
//  best_score      : score of best_node_id. -1.0f if none found.
//  reusable_node_id: first DEAD slot encountered during scan. -1 if L0 full.
struct SearchResult {
    int32_t best_node_id = -1;
    float   best_score = -1.0f;
    int32_t reusable_node_id = -1;
};

// Scan L0 Buffer in searching for similar vector.
// curr_time: unix seconds, used to filter expired PENDING nodes.
SearchResult SearchL0(MemoryArena& arena, const float* query,
                      const uint64_t curr_time) noexcept;
