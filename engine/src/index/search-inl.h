// Batch-scoring kernel shared by both L0/L1-tier search routines.

#pragma once

#include <optional>

#include "dot_product/avx2_kernel.h"
#include "index/search_result.h"
#include "memory/memory_arena.h"
#include "node_buf.h"

// Lookahead distance for software prefetcher: `2` batch iterations ahead of
// the position currently being scored.
inline constexpr uint32_t kPrefetchDistance = 2 * kBatchSize;

// TopTwoAccumulator finalizes top 2 search result by simutaneously evaluating
// the dot product score.
// CAUTION: Not intended to use externally.
struct TopTwoAccumulator {
    float fst_score = -1.0f;
    float sec_score = -1.0f;

    uint32_t fst_id = 0;
    uint32_t sec_id = 0;

    uint8_t fst_ver = 0;
    uint8_t sec_ver = 0;

    void Consider(uint32_t id, uint8_t ver, float score) noexcept {
        if (score > fst_score) {
            sec_id    = fst_id;
            sec_score = fst_score;
            sec_ver   = fst_ver;

            fst_id    = id;
            fst_score = score;
            fst_ver   = ver;

        } else if (score > sec_score) {
            sec_id    = id;
            sec_score = score;
            sec_ver   = ver;
        }
    }

    std::optional<SearchResult> Finalize() const noexcept {
        if (fst_score < kSimilarityThreshold) {
            return std::nullopt;
        }

        SearchResult res;
        res.primary = {fst_id, fst_ver};
        if (sec_score >= kSimilarityThreshold) {
            res.secondary = {sec_id, sec_ver};
        }

        return res;
    }
};

// ScoreCandidates represents the shared batch-scoring kernel of both vector
// search sub-routines.
//
//   1. `kBoundsSafe == true`  : For ring buffer has natural wrap-around (using
//                               AND arithmetic). No manual bounds check needed.
//   2. `kBoundsSafe == false` : For flat array with fixed size. Ensure safety
//                               by evaluating bounds check.
template <bool kBoundsSafe, typename NodeAt, typename CountAt>
std::optional<SearchResult> ScoreCandidates(
    const MemoryArena& arena, const float* query, NodeAt&& node_at,
    CountAt&& count_at
) noexcept {
    uint8_t  batch_vers[kBatchSize];
    uint32_t batch_ids[kBatchSize];
    uint32_t batch_count = 0;

    TopTwoAccumulator acc;

    auto score_batch = [&]() __attribute__((always_inline)) noexcept {
        if (batch_count == 0) {
            return;
        }

        // Pad unused lanes by duplicating the last valid item.
        for (uint32_t i = batch_count; i < kBatchSize; ++i) {
            batch_ids[i] = batch_ids[batch_count - 1];
        }

        float scores[kBatchSize];
        DotProductDiscreteBatch(
            query, arena.GetVector(batch_ids[0]), arena.GetVector(batch_ids[1]),
            arena.GetVector(batch_ids[2]), arena.GetVector(batch_ids[3]), scores
        );

        // Only consider non-padding items.
        for (uint32_t k = 0; k < batch_count; ++k) {
            const uint32_t id = batch_ids[k];

            if (arena.GetNode(id).LoadVersion() != batch_vers[k]) {
                // Version mismatch: torn-read in computing dot product. Skip.
                continue;
            }

            acc.Consider(id, batch_vers[k], scores[k]);
        }

        batch_count = 0;
    };

    for (uint32_t i = 0;; ++i) {
        const uint32_t count = count_at();
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

        batch_ids[batch_count]  = node_id;
        batch_vers[batch_count] = arena.GetNode(node_id).LoadVersion();
        ++batch_count;

        if (batch_count == kBatchSize) {
            score_batch();
        }
    }

    score_batch();
    return acc.Finalize();
}
