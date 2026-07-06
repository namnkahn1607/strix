// Author: namnkahn1607
//
// VectorIndex implementation. See index.h for the seqlock and Write-Ahead
// protocol invariants this relies on.

#include "index.h"

#include <atomic>
#include <cstring>
#include <exception>
#include <optional>

#include "avx2_math.h"
#include "constants.h"
#include "free_list.h"
#include "meta_node.h"

// `l0_cap` sizes the `L0Indices` ring; the capacity of `FreeList` is
// derived directly from `arena.MaxSlots()` so it can never drift out of
// sync with the arena it allocates `node_id` values into.
VectorIndex::VectorIndex(MemoryArena& arena, const size_t l0_cap)
    : arena_(arena), free_list_(arena.MaxSlots()), l0_indices_(l0_cap) {
}

std::optional<uint32_t> VectorIndex::AcquireNode(const float*   query,
                                                 const uint64_t now) noexcept {
    const uint32_t node_id = free_list_.Pop();
    if (node_id == FreeList::kEmpty) {
        return std::nullopt;
    }

    std::memcpy(arena_.GetVector(node_id), query, kVectorMemsize);

    MetaNode& node = arena_.GetNode(node_id);
    node.created_at.store(now, std::memory_order_relaxed);

    const uint8_t old_version = node.LoadVersion(std::memory_order_relaxed);
    const uint8_t new_version = NextVersion(old_version);

    const uint64_t published =
        PackControl(NodeState::kPending, EvictState::kHot, new_version, 0, 0);
    node.control_block.store(published, std::memory_order_release);

    if (!l0_indices_.TryPush(node_id)) {
        const uint64_t rollback =
            PackControl(NodeState::kDead, EvictState::kCold, new_version, 0, 0);
        node.control_block.store(rollback, std::memory_order_release);
        free_list_.Push(node_id);
        return std::nullopt;
    }

    return node_id;
}

void VectorIndex::ReleaseNode(const uint32_t node_id) noexcept {
    free_list_.Push(node_id);
}

namespace {

// Lookahead distance for the software prefetcher: 8 vectors, which is
// 2 batch-4 iterations ahead of the position currently being scored.
inline constexpr uint32_t kPrefetchDistance = 2 * kBatchSize;

}  // namespace

std::optional<SearchResult> VectorIndex::SearchL0(
    const float* query) const noexcept {
    const uint32_t right = l0_indices_.SnapPushHead();
    const uint32_t left  = l0_indices_.SnapPopTail();

    uint8_t  batch_vers[kBatchSize];
    uint32_t batch_ids[kBatchSize];
    uint32_t batch_count = 0;

    float    fst_score = -1.0f;
    uint8_t  fst_ver;
    uint32_t fst_id;
    float    sec_score = -1.0f;
    uint8_t  sec_ver;
    uint32_t sec_id;

    auto score_batch = [&]() noexcept {
        if (batch_count == 0) {
            return;
        }

        // Pad unused lanes by duplicating the last valid pointer so the batch-4
        // kernel never dereferences garbage.
        for (size_t i = batch_count; i < kBatchSize; ++i) {
            batch_ids[i] = batch_ids[batch_count - 1];
        }

        float scores[kBatchSize];
        DotProductIndirectBatch(query, arena_.GetVector(batch_ids[0]),
                                arena_.GetVector(batch_ids[1]),
                                arena_.GetVector(batch_ids[2]),
                                arena_.GetVector(batch_ids[3]), scores);

        // Cautious: Only consider non-padding items.
        for (size_t k = 0; k < batch_count; ++k) {
            const uint32_t id       = batch_ids[k];
            const uint8_t  curr_ver = arena_.GetNode(id).LoadVersion();
            if (curr_ver != batch_vers[k]) {
                continue;
            }

            if (scores[k] > fst_score) {
                sec_id    = fst_id;
                sec_score = fst_score;
                sec_ver   = fst_ver;

                fst_id    = id;
                fst_score = scores[k];
                fst_ver   = curr_ver;
            } else if (scores[k] > sec_score) {
                sec_id    = id;
                sec_score = scores[k];
                sec_ver   = curr_ver;
            }
        }

        batch_count = 0;
    };

    for (uint32_t i = left; i != right; ++i) {
        const uint32_t pf_id = l0_indices_.LoadSlot(i + kPrefetchDistance);
        if (pf_id != L0Indices::kEmpty) {
            const float* pf_vec = arena_.GetVector(pf_id);
            __builtin_prefetch(pf_vec, /*rw=*/0, /*locality=*/3);
            __builtin_prefetch(reinterpret_cast<const char*>(pf_vec) + 256,
                               /*rw=*/0, /*locality=*/3);
        }

        const uint32_t node_id = l0_indices_.LoadSlot(i);
        if (node_id == L0Indices::kEmpty) {
            continue;
        }

        batch_ids[batch_count]  = node_id;
        batch_vers[batch_count] = arena_.GetNode(node_id).LoadVersion();
        ++batch_count;

        const float* vec = arena_.GetVector(node_id);
        __builtin_prefetch(vec, /*rw=*/0, /*locality=*/3);
        __builtin_prefetch(reinterpret_cast<const char*>(vec) + 256, /*rw*/ 0,
                           /*locality=*/3);

        if (batch_count == 4) {
            score_batch();
        }
    }

    score_batch();

    if (fst_score < kSimilarityThreshold) {
        return std::nullopt;
    }

    SearchResult result;
    result.primary = {fst_id, fst_ver};
    if (sec_score >= kSimilarityThreshold) {
        result.secondary = {sec_id, sec_ver};
    }

    return result;
}

CacheOutcome VectorIndex::FetchPayload(const uint32_t node_id,
                                       const uint8_t  expected_version,
                                       const uint64_t curr_time,
                                       std::string*   out) const {
    MetaNode& node = arena_.GetNode(node_id);

    {
        const auto [state, ref_bit, version, length, v_offset] =
            node.LoadControl();

        if (version != expected_version || state == NodeState::kDead) {
            return CacheOutcome::kMiss;
        }

        if (state == NodeState::kPending) {
            const uint64_t ts = node.created_at.load(std::memory_order_acquire);
            if (curr_time - ts > kPendingLifespan) {
                return CacheOutcome::kMiss;
            }

            return CacheOutcome::kPendingHit;
        }

        try {
            out->resize(length);
        } catch (const std::exception&) {
            return CacheOutcome::kMiss;
        }

        arena_.ReadPayload(v_offset, length, out);
    }

    if (node.LoadVersion(std::memory_order_acquire) != expected_version) {
        out->clear();
        return CacheOutcome::kMiss;
    }

    return CacheOutcome::kHit;
}

bool VectorIndex::CommitPayload(const uint32_t node_id, const uint8_t* in,
                                const uint32_t length) noexcept {
    if (node_id >= arena_.MaxSlots()) {
        return false;
    }

    MetaNode& node = arena_.GetNode(node_id);
    const auto [state, ref, version, old_len, old_off] = node.LoadControl();

    if (state != NodeState::kPending) {
        return false;
    }

    const std::optional<uint64_t> offset_opt =
        arena_.WritePayload(node_id, in, length);
    if (!offset_opt.has_value()) {
        return false;
    }

    uint64_t expected = PackControl(state, ref, version, old_len, old_off);
    const uint64_t desired =
        PackControl(NodeState::kReady, ref, version, length, *offset_opt);
    return node.control_block.compare_exchange_strong(
        expected, desired, std::memory_order_release,
        std::memory_order_relaxed);
}
