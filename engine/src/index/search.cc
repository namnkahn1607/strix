// Author: namnkahn1607
//
// SearchL0() implementation.
// Single-pass scan of the L0 buffer using batched AVX2 dot products.

#include "search.h"

#include "avx2_math.h"

SearchResult SearchL0(MemoryArena& arena, const float* query,
                      const uint64_t curr_time) noexcept {
    SearchResult result;

    for (size_t i = 0; i < kL0MaxSlots; i += kBatchSize) {
        // scores[] is populated unconditionally by DotProductBatch() below.
        // valid[] gates which scores are considered in the result update.
        // This is intentionally branchless: computing dot products for
        // invalid nodes is cheaper than branching in the hot loop.
        float scores[kBatchSize] = {};
        bool  valid[kBatchSize]  = {};

        for (size_t k = 0; k < kBatchSize; ++k) {
            const MetaNode&       node = arena.GetNode(i + k);
            const UnpackedControl ctrl =
                node.LoadControl(std::memory_order_acquire);

            switch (ctrl.state) {
                case NodeState::kDead:
                    // Opportunistically record the first free slot so the
                    // caller can reclaim it without a second pass.
                    if (result.reusable_node_id == -1) {
                        result.reusable_node_id = static_cast<int32_t>(i + k);
                    }

                    break;

                case NodeState::kClaimed:
                    // Actively copying vector data; do not touch.
                    break;

                case NodeState::kPending: {
                    const uint64_t ts =
                        node.created_at.load(std::memory_order_acquire);

                    // created_at == 0 indicates the writer has not yet
                    // committed the timestamp (CLAIMED -> PENDING transition
                    // is not atomic). Skip to avoid a data race on ts.
                    if (ts == 0) {
                        break;
                    }

                    // Skip nodes whose PENDING window has expired; the GC
                    // sweeper will transition them to kDead shortly.
                    if (curr_time - ts > PENDING_LIFESPAN) {
                        break;
                    }

                    valid[k] = true;
                    break;
                }

                case NodeState::kReady:
                case NodeState::kMigrating:
                    valid[k] = true;
                    break;
            }
        }

        // Compute dot products for all kBatchSize nodes unconditionally.
        // Results for invalid nodes are computed but never read below.
        DotProductBatch(query, arena.GetVector(i), scores);

        for (size_t k = 0; k < kBatchSize; ++k) {
            if (valid[k] && scores[k] > result.best_score) {
                result.best_score   = scores[k];
                result.best_node_id = static_cast<int32_t>(i + k);
            }
        }
    }

    return result;
}
