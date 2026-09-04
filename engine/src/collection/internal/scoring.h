#pragma once

#include <cstdint>

#include "accumulator.h"
#include "ann/dot_product.h"
#include "memory/arena.h"
#include "node_buf.h"

namespace strix::collection {

// Lookahead distance for software prefetcher: 2 batch ahead.
inline constexpr uint32_t kPrefetchDistance = 2 * ann::kBatchSize;

template <uint32_t K, bool kBoundsSafe, typename NodeAt, typename CountAt>
void ScoreCandidate(
    const memory::Arena& arena, const float* query, NodeAt&& node_at,
    CountAt&& snap_count, TopKAccumulator<K>& acc
) noexcept {
    uint8_t  vers[ann::kBatchSize];
    uint32_t ids[ann::kBatchSize];
    uint32_t batch_count = 0;

    auto score_batch = [&]() __attribute__((always_inline)) noexcept {
        if (batch_count == 0) {
            return;
        }

        // Pad unused lanes by duplicating the last valid item.
        for (uint32_t i = batch_count; i < ann::kBatchSize; ++i) {
            ids[i] = ids[batch_count - 1];
        }

        float scores[ann::kBatchSize];
        ann::BatchDotProduct(
            query, arena.GetVector(ids[0]), arena.GetVector(ids[1]),
            arena.GetVector(ids[2]), arena.GetVector(ids[3]), scores
        );

        // Only consider non-padding items.
        for (uint32_t k = 0; k < batch_count; ++k) {
            const auto id    = ids[k];
            const auto ver   = vers[k];
            const auto score = scores[k];

            if (arena.GetMetaNode(id).LoadVersion() != ver) {
                // Dot product compute on torn vector. Skip.
                continue;
            }
            acc.Consider(id, ver, score);
        }

        batch_count = 0;
    };

    for (uint32_t i = 0;; ++i) {
        const uint32_t count = snap_count();
        if (i >= count) {
            break;
        }

        const uint32_t node_id = node_at(i);
        if constexpr (kBoundsSafe) {
            // L0-tier search falls here.
            const uint32_t pf_id = node_at(i + kPrefetchDistance);
            if (pf_id != NodeBuf::kEmpty) {
                const float* pf_vec = arena.GetVector(pf_id);
                __builtin_prefetch(pf_vec, /*rw=*/0, /*locality=*/3);
                __builtin_prefetch(
                    reinterpret_cast<const char*>(pf_vec) + 256,
                    /*rw=*/0, /*locality=*/3
                );
            }

            if (node_id == NodeBuf::kEmpty) {
                continue;
            }
        } else if (i + kPrefetchDistance < count) {
            // L1-tier search falls here.
            const uint32_t pf_id  = node_at(i + kPrefetchDistance);
            const float*   pf_vec = arena.GetVector(pf_id);
            __builtin_prefetch(pf_vec, /*rw=*/0, /*locality=*/3);
            __builtin_prefetch(
                reinterpret_cast<const char*>(pf_vec) + 256,
                /*rw=*/0, /*locality=*/3
            );
        }

        ids[batch_count]  = node_id;
        vers[batch_count] = arena.GetMetaNode(node_id).LoadVersion();

        if (++batch_count == ann::kBatchSize) {
            score_batch();
        }
    }

    score_batch();
}

}  // namespace strix::collection
