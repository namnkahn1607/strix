//
// index/search.cc
//

#include "search.hh"

#include "avx2_math.hh"

// ------------------------------------------------------------
// Vector Searching
// ------------------------------------------------------------

SearchResult SearchL0(MemoryArena& arena, const float* query,
                      const uint64_t curr_time) noexcept {
    SearchResult result;

    for (size_t i = 0; i < L0_MAX_SLOTS; i += BATCH_SIZE) {
        float scores[BATCH_SIZE] = {};
        bool  valid[BATCH_SIZE] = {};

        for (size_t k = 0; k < BATCH_SIZE; ++k) {
            const MetaNode& node = arena.GetNode(i + k);
            const uint64_t  ctrl =
                node.control_block.load(std::memory_order_relaxed);
            const auto [state, ref_bit, length, offset] = UnpackControl(ctrl);

            switch (state) {
                case NodeState::DEAD:
                    // Found a DEAD node, note immediately for future use.
                    if (result.reusable_node_id == -1) {
                        result.reusable_node_id = static_cast<int32_t>(i + k);
                    }
                    break;

                case NodeState::CLAIMED:
                    // There's on-going data construction. Do not touch.
                    break;

                case NodeState::PENDING: {
                    const uint64_t ts =
                        node.created_at.load(std::memory_order_acquire);

                    // A node whose created_at = 0 means unfinished transition
                    // from CLAIMED -> PENDING. Skip to avoid racing.
                    if (ts == 0) {
                        break;
                    }

                    // Expired PENDING node. Skip since GC will kill it soon.
                    if (curr_time - ts > PENDING_LIFESPAN) {
                        break;
                    }

                    valid[k] = true;
                    break;
                }

                case NodeState::READY:
                case NodeState::MIGRATING:
                    valid[k] = true;
                    break;
            }
        }

        DotProductL0_Batch4(query, arena.GetVector(i), scores);

        for (size_t k = 0; k < BATCH_SIZE; ++k) {
            if (valid[k] && scores[k] > result.best_score) {
                result.best_score = scores[k];
                result.best_node_id = static_cast<int32_t>(i + k);
            }
        }
    }

    return result;
}