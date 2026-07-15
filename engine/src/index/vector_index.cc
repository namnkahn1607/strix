// Author: namnkahn1607
//
// VectorIndex implementation. See its header for the seqlock and Write-Ahead
// protocol invariants this relies on.

#include "vector_index.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "constants.h"
#include "free_list.h"
#include "level0_ring.h"
#include "level1_ivf.h"
#include "meta_node.h"
#include "recalibration.h"
#include "search_inl.h"

// All nodes are intialized to `kUnclustered` (being a L0 node) at the start.
// The `L0Buffer` ring size is `l0_cap`; the capacity of `FreeList` is derived
// directly from `arena.MaxSlots()` so it can never drift out of sync with the
// arena it allocates `node_id` values into.
VectorIndex::VectorIndex(MemoryArena& arena, const uint32_t l0_cap,
                         const IvfConfig& config)
    : arena_(arena)
    , free_list_(arena.max_slots)
    , l0_buffer_(l0_cap)
    , node_owner_(std::make_unique_for_overwrite<std::atomic<uint32_t>[]>(
          arena.max_slots))
    , ivf_config_(config)
    , routes_(RoutingTable(config.num_clusters, config.max_cluster_size,
                           config.lazy_mapping),
              RoutingTable(config.num_clusters, config.max_cluster_size,
                           config.lazy_mapping))
    , recalibrator_(arena, routes_, &active_route_, config) {
    // Arguments check & sanitizing.
    if (config.kmeans_sample_size >= kUnclustered) {
        throw std::invalid_argument(
            "Number of clusters must be lower than kUnclustered identifier, "
            "which is " +
            std::to_string(kUnclustered));
    }

    if (config.kmeans_sample_size == 0) {
        throw std::invalid_argument(
            "K-means sample buffer size must be non-zero");
    }

    if (config.kmeans_sample_size % kBatchSize != 0) {
        throw std::invalid_argument(
            "K-means sample size must be a multiple of kernel batch size");
    }

    // Initialize all nodes (slots) as unclustered.
    for (size_t i = 0; i < arena.max_slots; ++i) {
        node_owner_[i].store(kUnclustered, std::memory_order_relaxed);
    }
}

VectorIndex::~VectorIndex() = default;

// -----------------------------------------------------------------------------
// Acquire/Release Node
// -----------------------------------------------------------------------------

//
std::optional<uint32_t> VectorIndex::AcquireNode(const float*   query,
                                                 const uint64_t now) noexcept {
    const uint32_t node_id = free_list_.Pop();
    if (node_id == FreeList::kEmpty) {
        return std::nullopt;
    }

    // Establish an anchor before anything else. The node ID might still may
    // still reside in its old cluster from previous life, even though GC
    // evicted it. Pinning a lock here signals Reassignment that "a node is
    // no longer belongs to this cluster, please purge it rightaway if you
    // encounter any".
    node_owner_[node_id].store(kUnclustered, std::memory_order_relaxed);

    std::memcpy(arena_.GetVector(node_id), query, kVectorMemsize);

    MetaNode& node = arena_.GetNode(node_id);
    node.created_at.store(now, std::memory_order_relaxed);

    const uint8_t old_version = node.LoadVersion(std::memory_order_relaxed);
    const uint8_t new_version = NextVersion(old_version);

    const uint64_t published =
        PackControl(NodeState::kPending, EvictState::kHot, new_version, 0, 0);
    node.control_block.store(published, std::memory_order_release);

    if (!l0_buffer_.TryPush(node_id)) {
        // Failed to register node to L0 buffer: the system is under high load.
        // Rollback to `kDead` and return to `FreeList`.
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

// -----------------------------------------------------------------------------
// Vector Searching
// -----------------------------------------------------------------------------

//
std::optional<SearchResult> VectorIndex::SearchL0(
    const float* query) const noexcept {
    const uint32_t right = l0_buffer_.SnapPushHead();
    const uint32_t left  = l0_buffer_.SnapPopTail();

    uint32_t count = right - left;
    return ScoreCandidates</*kBoundsSafe=*/true>(
        arena_, query,
        [&](uint32_t idx) noexcept { return l0_buffer_.LoadSlot(left + idx); },
        [count]() noexcept { return count; });
}

std::optional<SearchResult> VectorIndex::SearchL1(
    const float* query) const noexcept {
    const uint8_t       active = active_route_.load(std::memory_order_acquire);
    const RoutingTable& table  = routes_[active];

    const uint32_t cluster_id = table.MatchCluster(query);
    auto           members    = table.ClusterMemberIds(cluster_id);

    return ScoreCandidates</*kBoundsSafe=*/false>(
        arena_, query,
        [&](uint32_t idx) noexcept {
            return members[idx].load(std::memory_order_acquire);
        },
        [&table, cluster_id]() noexcept {
            return table.ClusterSize(cluster_id);
        });
}

// -----------------------------------------------------------------------------
// Fetch/Commit Payload
// -----------------------------------------------------------------------------

//
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
    if (node_id >= arena_.max_slots) {
        return false;
    }

    MetaNode& node = arena_.GetNode(node_id);
    const auto [state, ref_bit, version, old_len, old_off] = node.LoadControl();

    if (state != NodeState::kPending) {
        return false;
    }

    const auto offset_opt = arena_.WritePayload(node_id, in, length);
    if (!offset_opt.has_value()) {
        return false;
    }

    auto expected = PackControl(state, ref_bit, version, old_len, old_off);
    const auto desired =
        PackControl(NodeState::kReady, ref_bit, version, length, *offset_opt);

    // If this CAS fails, that means the node was evicted/reused between
    // LoadControl() and this point.
    // The payload bytes written becomes orphanated in the ring buffer, waiting
    // for GC to come and reclaim.
    return node.control_block.compare_exchange_strong(
        expected, desired, std::memory_order_release,
        std::memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// Background Coordinators
// -----------------------------------------------------------------------------

void VectorIndex::RunCompaction() noexcept {
    const uint32_t node_id = l0_buffer_.TryPop();
    if (node_id == L0Buffer::kEmpty) {
        return;
    }

    MetaNode& node = arena_.GetNode(node_id);
    const auto [state, ref_bit, version, length, offset] = node.LoadControl();

    if (state == NodeState::kDead) {
        return;
    }

    // Compaction only migrates `kPending` and `kReady` nodes.
    const float*   vec        = arena_.GetVector(node_id);
    const uint8_t  active     = active_route_.load(std::memory_order_acquire);
    const uint32_t cluster_id = routes_[active].MatchCluster(vec);

    // Re-validate before committing. If this generation was evicted and
    // reacquired during the process above, the score just computed may not
    // describe what's currently at node_id at all. Abandon rather than proceed.
    if (node.LoadVersion(std::memory_order_acquire) != version) {
        return;
    }

    if (!routes_[active].JoinCluster(node_id, cluster_id)) {
        // Cluster is full, which means it is semantically saturated, drop it.
        return;
    }

    node_owner_[node_id].store(cluster_id, std::memory_order_relaxed);
}

void VectorIndex::RunReassignment() noexcept {
    const uint8_t active = active_route_.load(std::memory_order_acquire);
    RoutingTable& table  = routes_[active];

    const uint32_t curr_cluster = reassignment_cursor_;
    reassignment_cursor_ = (reassignment_cursor_ + 1) % table.num_clusters;

    auto members = table.ClusterMemberIds(curr_cluster);

    uint32_t i = 0;
    while (i < table.ClusterSize(curr_cluster)) {
        const uint32_t node_id = members[i].load(std::memory_order_acquire);

        MetaNode& node = arena_.GetNode(node_id);
        const auto [state, ref_bit, version, length, offset] =
            node.LoadControl();
        const uint32_t owner =
            node_owner_[node_id].load(std::memory_order_relaxed);

        // Evicted by GC, or being re-acquired or being migrated to another
        // cluster by Compaction (no longer "agrees" with the current cluster
        // holding it).
        const bool is_zombie = (state == NodeState::kDead) ||
                               (owner == kUnclustered) ||
                               (owner != curr_cluster);

        if (is_zombie) {
            if (!table.LeaveCluster(curr_cluster, i, node_id)) {
                // Shouldn't happen since this thread is current cluster's only
                // Writer. If happens, reluctantly advance the iterator.
                ++i;
            }
            // Swap-with-tail brought an unexamined entry from tail into this
            // position; re-examine it next iteration.
            continue;
        }

        // Legitimate member, does it belong here?
        const float*   vec          = arena_.GetVector(node_id);
        const uint32_t home_cluster = table.MatchCluster(vec);

        if (home_cluster == curr_cluster) {
            ++i;
            continue;
        }

        // Seems like it does not belong here. Re-validate since we've
        // just read this node's vector data.
        if (node.LoadVersion() != version) {
            // A version mismatch happened. Not enough trust to continue
            // evaluating - skip.
            ++i;
            continue;
        }

        if (!table.JoinCluster(node_id, home_cluster)) {
            // Destination cluster full, stay. ANN's dot product guarantee
            // correctness for vector searching in IVF.
            ++i;
            continue;
        }

        node_owner_[node_id].store(home_cluster, std::memory_order_relaxed);

        if (!table.LeaveCluster(curr_cluster, i, node_id)) {
            // Shouldn't happen since this thread is current cluster's only
            // Writer. If happens, reluctantly advance the iterator.
            ++i;
        }
        // Swap-with-tail brought an unexamined entry from tail into this
        // position; re-examine it next iteration.
    }
}
