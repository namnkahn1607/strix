// Author: namnkahn1607
//
// VectorIndex implementation. See its header for the seqlock and Write-Ahead
// protocol invariants this relies on.

#include "index/vector_index.h"

#include <sys/types.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include "common/tagged_treiber.h"
#include "dot_product/avx2_kernel.h"
#include "index/ivf_config.h"
#include "index/search_result.h"
#include "level0_ring.h"
#include "level1_ivf.h"
#include "memory/memory_arena.h"
#include "memory/meta_node.h"
#include "recalibration.h"
#include "search-inl.h"

class VectorIndex::Impl {
public:
    Impl(MemoryArena& arena, uint32_t l0_cap, const IvfConfig& config)
        : arena_(arena)
        , free_node_list_(arena.max_slots)
        , l0_buffer_(l0_cap)
        , node_owner_(std::make_unique_for_overwrite<std::atomic<uint32_t>[]>(
              arena.max_slots
          ))
        , ivf_config_(config)
        , routes_(
              RoutingTable(
                  config.num_clusters, config.max_cluster_size,
                  config.lazy_mapping
              ),
              RoutingTable(
                  config.num_clusters, config.max_cluster_size,
                  config.lazy_mapping
              )
          )
        , recalibrator_(std::make_unique<Recalibrator>(
              arena, routes_, &active_route_, config
          )) {
        if (config.num_clusters >= kUnclustered) {
            throw std::invalid_argument(
                "Number of clusters must be lower than kUnclustered "
                "identifier, "
                "which is " +
                std::to_string(kUnclustered)
            );
        }

        if (config.kmeans_sample_size == 0) {
            throw std::invalid_argument(
                "K-means sample buffer size must be non-zero"
            );
        }

        if (config.kmeans_sample_size % kBatchSize != 0) {
            throw std::invalid_argument(
                "K-means sample size must be a multiple of kernel batch size"
            );
        }

        // Initialize all nodes (slots) as unclustered.
        for (size_t i = 0; i < arena.max_slots; ++i) {
            node_owner_[i].store(kUnclustered, std::memory_order_relaxed);
        }
    }

    MemoryArena& arena_;
    TreiberStack free_node_list_;
    L0Buffer     l0_buffer_;

    // `node_owner_`, a single-source-of-truth cluster ID tracker for every
    // clustered node. Unclustered nodes are considered `kUnclustered`.
    std::unique_ptr<std::atomic<uint32_t>[]> node_owner_;

    IvfConfig            ivf_config_;
    RoutingTable         routes_[2];
    std::atomic<uint8_t> active_route_{0};

    // Gates any IVF-related operations at cold start until the IVF's
    // centroid array is fully seeded.
    std::atomic<bool> ivf_enabled_{false};

    // `RunCompaction()` sequentially migrates nodes from L0 Buffer to L1 IVF.
    void RunCompaction() noexcept;

    // IVF's `Recalibrator` - the calibration state-machine controller.
    std::unique_ptr<Recalibrator> recalibrator_;

    // Round-robin cursor only for `RunReassignment()` to determine which
    // cluster to sweep on the next call.
    uint32_t reassignment_cursor_ = 0;

    // `RunReassignment()` sweeps each cluster per called. Remove `kDead` node
    // ID of that cluster and move datapoint to closer cluster.
    // A generation change (node be evicted and re-acquired) mid-process are
    // skipped to determine on another pass.
    void RunReassignment() noexcept;
};

// All nodes are intialized to `kUnclustered` (being a L0 node) at the start.
// The `L0Buffer` ring size is `l0_cap`; the capacity of `FreeList` is derived
// directly from `arena.MaxSlots()` so it can never drift out of sync with the
// arena it allocates `node_id` values into.
VectorIndex::VectorIndex(
    MemoryArena& arena, const uint32_t l0_cap, const IvfConfig& config
)
    : pimpl_(std::make_unique<Impl>(arena, l0_cap, config)) {}

VectorIndex::~VectorIndex() = default;

std::optional<uint32_t> VectorIndex::AcquireNode(
    const float* query, const uint64_t now
) noexcept {
    const uint32_t node_id = Inner()->free_node_list_.Pop();
    if (node_id == TreiberStack::kEmpty) {
        return std::nullopt;
    }

    // Establish an anchor before anything else. The node ID might still may
    // still reside in its old cluster from previous life, even though GC
    // evicted it. Pinning a lock here signals Reassignment that "a node is
    // no longer belongs to this cluster, please purge it rightaway if you
    // encounter any".
    Inner()->node_owner_[node_id].store(
        kUnclustered, std::memory_order_relaxed
    );

    std::memcpy(Inner()->arena_.GetVector(node_id), query, kVectorMemsize);

    MetaNode& node = Inner()->arena_.GetNode(node_id);
    node.created_at.store(now, std::memory_order_relaxed);

    const uint8_t old_version = node.LoadVersion(std::memory_order_relaxed);
    const uint8_t new_version = NextVersion(old_version);

    const uint64_t published =
        PackControl(NodeState::kPending, EvictState::kHot, new_version, 0, 0);
    node.control_block.store(published, std::memory_order_release);

    if (!Inner()->l0_buffer_.TryPush(node_id)) {
        // Failed to register node to L0 buffer: the system is under high load.
        // Rollback to `kDead` and return to `FreeList`.
        const uint64_t rollback =
            PackControl(NodeState::kDead, EvictState::kCold, new_version, 0, 0);
        node.control_block.store(rollback, std::memory_order_release);
        Inner()->free_node_list_.Push(node_id);
        return std::nullopt;
    }

    return node_id;
}

void VectorIndex::ReleaseNode(const uint32_t node_id) noexcept {
    Inner()->free_node_list_.Push(node_id);
}

std::optional<SearchResult> VectorIndex::SearchL0(const float* query
) const noexcept {
    const uint32_t right = Inner()->l0_buffer_.SnapPushHead();
    const uint32_t left  = Inner()->l0_buffer_.SnapPopTail();

    uint32_t count = right - left;
    return ScoreCandidates</*kBoundsSafe=*/true>(
        Inner()->arena_, query,
        [&](uint32_t idx) noexcept {
            return Inner()->l0_buffer_.LoadSlot(left + idx);
        },
        [count]() noexcept { return count; }
    );
}

std::optional<SearchResult> VectorIndex::SearchL1(const float* query
) const noexcept {
    const uint8_t active =
        Inner()->active_route_.load(std::memory_order_acquire);
    const RoutingTable& table = Inner()->routes_[active];

    const uint32_t cluster_id = table.MatchCluster(query);
    auto           members    = table.ClusterMemberIds(cluster_id);

    return ScoreCandidates</*kBoundsSafe=*/false>(
        Inner()->arena_, query,
        [&](uint32_t idx) noexcept {
            return members[idx].load(std::memory_order_acquire);
        },
        [&table, cluster_id]() noexcept {
            return table.ClusterSize(cluster_id);
        }
    );
}

CacheOutcome VectorIndex::Fetch(
    const uint32_t node_id, const uint8_t exp_ver, const uint64_t curr_time,
    std::string* out
) const noexcept {
    return Inner()->arena_.ReadPayload(node_id, exp_ver, curr_time, out);
}

bool VectorIndex::Commit(
    const uint32_t node_id, const uint8_t* in, const uint32_t length
) noexcept {
    return Inner()->arena_.WritePayload(node_id, in, length).has_value();
}

L0Buffer& VectorIndex::GetL0Buffer() noexcept { return Inner()->l0_buffer_; }

namespace {

// Backoff while the Coordinator's bootstrap is waiting for L0 traffic.
inline constexpr auto kBootstrapIdleBackoff = std::chrono::milliseconds(10);

// Fixed per-iteration tick for Coordinator's steady-state loop.
inline constexpr auto kCoordinatorTick = std::chrono::milliseconds(1);

}  // namespace

void VectorIndex::RunCoordinator(const std::atomic<bool>& shutdown_req) {
    // -------------------------------------------------------------------------
    // Bootstrap: seed routes_[inactive] directly from live L0 traffic.
    //
    // Escape the chicken-and-egg problem at cold start: all IVF-related
    // operations expected a valid-filled centroid array, which hasn't exist
    // yet. Bootstrap breaks the cycle by harvesting vectors straight from L0
    // buffer.
    // -------------------------------------------------------------------------
    {
        const uint8_t inactive =
            1 - Inner()->active_route_.load(std::memory_order_acquire);
        RoutingTable& table = Inner()->routes_[inactive];

        uint32_t seeded = 0;
        while (seeded < table.num_clusters) {
            if (shutdown_req.load(std::memory_order_relaxed)) {
                return;
            }

            const uint32_t node_id = Inner()->l0_buffer_.TryPop();
            if (node_id == L0Buffer::kEmpty) {
                std::this_thread::sleep_for(kBootstrapIdleBackoff);
                continue;
            }

            MetaNode& node = Inner()->arena_.GetNode(node_id);
            const auto [state, ref, version, length, v_offset] =
                node.LoadControl();
            if (state == NodeState::kDead) {
                // kDead node detected. Seed-in another vec for this centroid.
                continue;
            }

            const float* vec = Inner()->arena_.GetVector(node_id);
            table.SeedCentroid(seeded, vec);

            if (node.LoadVersion() != version) {
                // Torn-read detected. Seed-in another vec for this centroid.
                continue;
            }

            if (!table.JoinCluster(node_id, seeded)) {
                // Can't actually happen, as seeded is always this cluster's
                // fist member, so the cluster can never be full.
                continue;
            }
            Inner()->node_owner_[node_id].store(
                seeded, std::memory_order_relaxed
            );

            ++seeded;
        }

        Inner()->active_route_.store(inactive, std::memory_order_release);
        Inner()->ivf_enabled_.store(true, std::memory_order_release);
    }

    // -------------------------------------------------------------------------
    // Compaction-only phase: run until the first real Recalibration episode has
    // started. And since no episode has been published, no semantic drift
    // happens (yet), so Reassignment is meaningless this time.
    // -------------------------------------------------------------------------
    while (Inner()->recalibrator_->CurrentPhase() == Recalibrator::Phase::kIdle
    ) {
        if (shutdown_req.load(std::memory_order_relaxed)) {
            return;
        }

        Inner()->RunCompaction();
        Inner()->recalibrator_->NotifyCompactionSucceeded();
        std::this_thread::sleep_for(kCoordinatorTick);
    }

    // -------------------------------------------------------------------------
    // Steady-state. The Coordinator stays in this loop forever after.
    // -------------------------------------------------------------------------
    while (!shutdown_req.load(std::memory_order_relaxed)) {
        Inner()->RunCompaction();
        Inner()->recalibrator_->NotifyCompactionSucceeded();
        Inner()->recalibrator_->Tick();
        Inner()->RunReassignment();
        std::this_thread::sleep_for(kCoordinatorTick);
    }
}

void VectorIndex::Impl::RunCompaction() noexcept {
    const uint32_t node_id = l0_buffer_.TryPop();
    if (node_id == L0Buffer::kEmpty) {
        return;
    }

    MetaNode& node = arena_.GetNode(node_id);
    const auto [state, ref, version, length, v_offset] = node.LoadControl();

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

void VectorIndex::Impl::RunReassignment() noexcept {
    const uint8_t active = active_route_.load(std::memory_order_acquire);
    RoutingTable& table  = routes_[active];

    const uint32_t curr_cluster = reassignment_cursor_;
    reassignment_cursor_ = (reassignment_cursor_ + 1) % table.num_clusters;

    auto members = table.ClusterMemberIds(curr_cluster);

    uint32_t i = 0;
    while (i < table.ClusterSize(curr_cluster)) {
        const uint32_t node_id = members[i].load(std::memory_order_acquire);

        MetaNode& node = arena_.GetNode(node_id);
        const auto [state, ref, version, length, v_offset] = node.LoadControl();
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
