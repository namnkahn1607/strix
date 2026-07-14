// Author: namnkahn1607
//
// The shared batch-scoring kernel used by both L0 and L1
// vector search subroutines.

#pragma once

#include <optional>

#include "avx2_kernel.h"
#include "constants.h"
#include "level0_ring.h"
#include "memory_arena.h"

// `SearchOutcome` describes result of a vector search against either tier.
// Only record information of a node whose similarity score exceeds the
// pre-defined `kSimiarityThreshold`.
struct SearchOutcome {
    uint32_t node_id;
    uint8_t  version;
};

// `SearchResult` tracks not just the champion but also the runner-up.
// The runner-up entry is optional.
struct SearchResult {
    SearchOutcome                primary;
    std::optional<SearchOutcome> secondary;
};

// Lookahead distance for the software prefetcher: 8 vectors, which is
// 2 batch-4 iterations ahead of the position currently being scored.
inline constexpr uint32_t kPrefetchDistance = 2 * kBatchSize;

// `TopTwoAccumulator` finalizes the vector search top 2 result by simutaneously
// evaluating the dot product score. 
// CAUTION: Not intended to use externally.
struct TopTwoAccumulator {
    float    fst_score = -1.0f, sec_score = -1.0f;
    uint32_t fst_id = 0, sec_id = 0;
    uint8_t  fst_ver = 0, sec_ver = 0;

    inline void Consider(uint32_t id, uint8_t ver, float score) noexcept {
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

    inline std::optional<SearchResult> Finalize() const noexcept {
        if (fst_score > kSimilarityThreshold) {
            return std::nullopt;
        }

        SearchResult res;
        res.primary = {fst_id, fst_ver};
        if (sec_score > kSimilarityThreshold) {
            res.secondary = {sec_id, sec_ver};
        }

        return res;
    }
};

// `ScoreCandidates()` is the shared batch-scoring routine of both vector
// search sub-routines. Takes `MemoryArena` explicitly as argument to access
// vector data.
//
//   1. `kBoundsSafe == true`  : For ring buffer has natural wrap-around (using
//                               AND arithmetic). No manual bounds check needed.
//   2. `kBoundsSafe == false` : For flat array with fixed size. Ensure safety
//                               by evaluating bounds check.
template <bool kBoundsSafe, typename NodeAt, typename CountAt>
std::optional<SearchResult> ScoreCandidates(const MemoryArena& arena,
                                            const float*       query,
                                            NodeAt&&           node_at,
                                            CountAt&& count_at) noexcept {
    uint8_t  batch_vers[kBatchSize];
    uint32_t batch_ids[kBatchSize];
    uint32_t batch_count = 0;

    TopTwoAccumulator acc;

    auto score_batch = [&]() __attribute__((always_inline)) noexcept {
        if (batch_count == 0) {
            return;
        }

        // Pad unused lanes by duplicating the last valid pointer so the batch-4
        // kernel never dereferences garbage.
        for (uint32_t i = batch_count; i < kBatchSize; ++i) {
            batch_ids[i] = batch_ids[batch_count - 1];
        }

        float scores[kBatchSize];
        DotProductIndirectBatch(query, arena.GetVector(batch_ids[0]),
                                arena.GetVector(batch_ids[1]),
                                arena.GetVector(batch_ids[2]),
                                arena.GetVector(batch_ids[3]), scores);

        // Cautious: Only consider non-padding items.
        for (uint32_t k = 0; k < batch_count; ++k) {
            const uint32_t id = batch_ids[k];

            const uint8_t curr_ver = arena.GetNode(id).LoadVersion();
            if (curr_ver != batch_vers[k]) {
                continue;
            }

            acc.Consider(id, curr_ver, scores[k]);
        }

        batch_count = 0;
    };

    for (uint32_t i = 0;; ++i) {
        const uint32_t count = count_at();
        if (i >= count) {
            break;
        }

        uint32_t record;
        if constexpr (kBoundsSafe) {
            // L0 vector search falls here. No branching check.
            const uint32_t pf_id = node_at(i + kPrefetchDistance);
            if (pf_id != L0Buffer::kEmpty) {
                const float* pf_vec = arena.GetVector(pf_id);
                __builtin_prefetch(pf_vec, /*rw=*/0, /*locality=*/3);
                __builtin_prefetch(reinterpret_cast<const char*>(pf_vec) + 256,
                                   /*rw=*/0, /*locality=*/3);
            }

            record = node_at(i);
            if (record == L0Buffer::kEmpty) {
                continue;
            }

        } else if (i + kPrefetchDistance < count) {
            // L1 vector search falls here.
            const uint32_t pf_id  = node_at(i + kPrefetchDistance);
            const float*   pf_vec = arena.GetVector(pf_id);
            __builtin_prefetch(pf_vec, /*rw=*/0, /*locality=*/3);
            __builtin_prefetch(reinterpret_cast<const char*>(pf_vec) + 256,
                               /*rw=*/0, /*locality=*/3);

            record = node_at(i);
        }

        const uint32_t& node_id = record;

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
