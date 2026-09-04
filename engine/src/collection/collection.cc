#include "collection/collection.h"

#include <absl/log/check.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>

#include "base/cache_state.h"
#include "base/tagged_treiber.h"
#include "collection/search.h"
#include "inference/info.h"
#include "internal/accumulator.h"
#include "internal/ivf_table.h"
#include "internal/node_buf.h"
#include "internal/recalibration.h"
#include "internal/scoring.h"
#include "memory/arena.h"
#include "memory/control_block.h"
#include "memory/state.h"

namespace strix::collection {

class Collection::Impl {
public:
    Impl(const Config& config, memory::Arena& arena)
        : config_{config}
        , arena_{arena}
        , node_free_list_{arena.max_slots}
        , node_buf_{config.lvl0_capacity}
        , tables_{IvfTable{config}, IvfTable{config}} {
        cluster_of = std::make_unique_for_overwrite<std::atomic<uint32_t>[]>(
            arena.max_slots
        );
        for (uint32_t node_id = 0; node_id < arena.max_slots; ++node_id) {
            cluster_of[node_id].store(kUnclustered, std::memory_order_relaxed);
        }
    }

    void Compaction() noexcept;
    void Reassignment() noexcept;

    Config         config_;
    memory::Arena& arena_;
    TreiberStack   node_free_list_;
    NodeBuf        node_buf_;

    std::unique_ptr<std::atomic<uint32_t>[]> cluster_of;

    std::atomic<bool>    ivf_enabled_{true};
    IvfTable             tables_[2];
    std::atomic<uint8_t> active_route_{0};

    std::unique_ptr<Recalibrator> recalibrator_;

    // Round-robin cursor of Reassignment tracking next target cluster
    // to operate upon invocation.
    uint32_t reassignment_cursor_ = 0;
};

Collection::Collection(const Config& config, memory::Arena& arena)
    : pimpl_{std::make_unique<Impl>(config, arena)} {}

Collection::~Collection() = default;

CacheState Collection::FetchCache(
    uint32_t node_id, uint8_t exp_ver, TimePoint now, std::string* out
) const noexcept {
    return Inner()->arena_.ReadPayload(node_id, exp_ver, now, out);
}

bool Collection::CommitEntry(
    uint32_t node_id, const uint8_t* in, uint32_t length
) noexcept {
    return Inner()->arena_.WritePayload(node_id, in, length).has_value();
}

std::optional<TopKResult<kTopK>> Collection::Search(const float* query
) const noexcept {
    TopKAccumulator<kTopK> acc;

    // L0-tier search.
    {
        const auto right = Inner()->node_buf_.SnapPushHead();
        const auto left  = Inner()->node_buf_.SnapPopTail();
        const auto count = right - left;
        ScoreCandidate<kTopK, /*kBoundsSafe=*/true>(
            Inner()->arena_, query,
            [&](uint32_t idx) noexcept {
                return Inner()->node_buf_.LoadSlot(left + idx);
            },
            [count]() noexcept { return count; }, acc
        );
    }

    // L1-tier search.
    {
        const uint8_t active =
            Inner()->active_route_.load(std::memory_order_acquire);
        const auto&    table      = Inner()->tables_[active];
        const uint32_t cluster_id = table.MatchCluster(query);

        auto members = table.ListMembers(cluster_id);
        ScoreCandidate<kTopK, /*kBoundsSafe=*/false>(
            Inner()->arena_, query,
            [&](uint32_t idx) noexcept {
                return members[idx].load(std::memory_order_acquire);
            },
            [&table, cluster_id]() noexcept {
                return table.ClusterSize(cluster_id);
            },
            acc
        );
    }

    return acc.Finalize();
}

std::optional<uint32_t> Collection::AcquireSlotFor(
    const float* query, TimePoint now
) noexcept {
    const auto node_id = Inner()->node_free_list_.Pop();
    if (node_id == TreiberStack::kEmpty) {
        return std::nullopt;
    }

    // The node ID might still reside in its old cluster from previous life even
    // though GC evicted it. Nail an anchor first for Reassignment to purge it
    // right away upon encountering.
    Inner()->cluster_of[node_id].store(kUnclustered, std::memory_order_relaxed);

    std::memcpy(
        Inner()->arena_.GetVector(node_id), query, inference::kVectorMemsize
    );

    auto& node = Inner()->arena_.GetMetaNode(node_id);
    node.created_at.store(now, std::memory_order_relaxed);

    const auto old_ver = node.LoadVersion(std::memory_order_relaxed);
    // Now that the vector data has been overwritten. Version MUST advance
    // regardless of successful acquistion or not.
    const auto new_ver = memory::NextVersion(old_ver);

    const auto published = memory::ControlBlock::Pack(
        memory::NodeState::kPending, memory::EvictState::kHot, new_ver, 0, 0
    );
    node.control_block.store(published, std::memory_order_release);

    if (!Inner()->node_buf_.TryEnqueue(node_id)) {
        // Failed to register acquired node slot.
        // Rollback to DEAD and return to Freelist.
        const auto rollback = memory::ControlBlock::Pack(
            memory::NodeState::kDead, memory::EvictState::kCold, new_ver, 0, 0
        );
        node.control_block.store(rollback, std::memory_order_release);
        Inner()->node_free_list_.Push(node_id);

        return std::nullopt;
    }

    return node_id;
}

void Collection::ReleaseSlot(uint32_t node_id) noexcept {
    Inner()->node_free_list_.Push(node_id);
}

void Collection::StartCoordinator(const std::atomic<bool>& shutdown_req
) noexcept {
    constexpr auto kBootstrapIdleBackoff = std::chrono::milliseconds{10};
    constexpr auto kCoordinatorTick      = std::chrono::milliseconds{1};

    // PHASE 1: IVF Bootstrap
    // Harvesting vectors from live L0-tier traffic and set them as IVF's first
    // centroids.

    const uint8_t inactive =
        1 - Inner()->active_route_.load(std::memory_order_acquire);
    auto& table = Inner()->tables_[inactive];

    uint32_t seeded = 0;
    while (seeded < table.num_clusters) {
        if (shutdown_req.load(std::memory_order_relaxed)) {
            return;
        }

        const auto node_id = Inner()->node_buf_.TryDequeue();
        if (node_id == NodeBuf::kEmpty) {
            std::this_thread::sleep_for(kBootstrapIdleBackoff);
            continue;
        }

        auto& node = Inner()->arena_.GetMetaNode(node_id);
        const auto [state, ref, ver, length, offset] = node.LoadControl();
        if (state == memory::NodeState::kDead) {
            continue;
        }

        const float* vec = Inner()->arena_.GetVector(node_id);
        table.SetCentroid(seeded, vec);

        if (node.LoadVersion() != ver) {
            continue;
        }

        // Shouldn't happen.
        // The seeding vector is always the cluster's first member.
        CHECK(!table.JoinCluster(node_id, seeded))
            << "First vector in cluster yields exhausted!?";

        Inner()->active_route_.store(inactive, std::memory_order_release);
        Inner()->ivf_enabled_.store(true, std::memory_order_release);
    }

    // PHASE 2: Compaction-only
    // Perform Compaction until first released Recalibrated episoide. And since
    // no semantic drift occurs (yet), so Reassignment is not needed.

    while (Inner()->recalibrator_->CurrentPhase() == Recalibrator::Phase::kIdle
    ) {
        if (shutdown_req.load(std::memory_order_relaxed)) {
            return;
        }

        Inner()->Compaction();
        Inner()->recalibrator_->NotifyCompactionSucceeded();
        std::this_thread::sleep_for(kCoordinatorTick);
    }

    // PHASE 3: Steady-state

    while (!shutdown_req.load(std::memory_order_relaxed)) {
        Inner()->Compaction();
        Inner()->recalibrator_->NotifyCompactionSucceeded();
        Inner()->recalibrator_->Tick();
        Inner()->Reassignment();
        std::this_thread::sleep_for(kCoordinatorTick);
    }
}

void Collection::Impl::Compaction() noexcept {
    const auto node_id = node_buf_.TryDequeue();
    if (node_id == NodeBuf::kEmpty) {
        return;
    }
    auto& node = arena_.GetMetaNode(node_id);

    const auto [state, ref, ver, length, offset] = node.LoadControl();
    // Compaction only migrates PENDING and READY nodes.
    if (state == memory::NodeState::kDead) {
        return;
    }

    const float*  vec        = arena_.GetVector(node_id);
    const uint8_t active     = active_route_.load(std::memory_order_acquire);
    const auto    cluster_id = tables_[active].MatchCluster(vec);

    // Re-validate before committing.
    if (node.LoadVersion() != ver) {
        // This node was evicted and reacquired during the process above, the
        // matching score justs computed may not describe what's currently at
        // that node slot. Skip.
        return;
    }

    if (!tables_[active].JoinCluster(node_id, cluster_id)) {
        // Cluster saturated. Drop.
        return;
    }

    cluster_of[node_id].store(cluster_id, std::memory_order_relaxed);
}

void Collection::Impl::Reassignment() noexcept {
    const uint8_t active = active_route_.load(std::memory_order_acquire);
    auto&         table  = tables_[active];

    const auto curr_cluster = reassignment_cursor_;
    reassignment_cursor_    = (reassignment_cursor_ + 1) % table.num_clusters;

    auto members = table.ListMembers(curr_cluster);

    uint32_t idx = 0;
    while (idx < table.ClusterSize(curr_cluster)) {
        const uint32_t node_id = members[idx].load(std::memory_order_acquire);

        auto& node = arena_.GetMetaNode(node_id);

        const auto [state, ref, ver, length, offset] = node.LoadControl();
        const uint32_t owner =
            cluster_of[node_id].load(std::memory_order_relaxed);

        const bool is_zombie = (state == memory::NodeState::kDead) ||
                               (owner == kUnclustered) ||
                               (owner != curr_cluster);
        if (is_zombie) {
            if (!table.LeaveCluster(curr_cluster, idx, node_id)) {
                // Shouldn't happen since this worker is the only writer.
                // Otherwise reluctantly advancing the iterator.
                ++idx;
            }

            // Swap-with-tail brought an unexamined node ID from tail to this
            // position. Examine it next iteration.
            continue;
        }

        // Legitimate member, does it belong here?
        const float* vec = arena_.GetVector(node_id);

        const auto home_cluster = table.MatchCluster(vec);
        if (home_cluster == curr_cluster || node.LoadVersion() != ver) {
            // Belong or not belong and corrupted read.
            ++idx;
            continue;
        }

        cluster_of[node_id].store(home_cluster, std::memory_order_relaxed);

        if (!table.LeaveCluster(curr_cluster, idx, node_id)) {
            // Shouldn't happen since this worker is the only writer.
            // Otherwise reluctantly advancing the iterator.
            ++idx;
        }
        // Swap-with-tail brought an unexamined node ID from tail to this
        // position. Examine it next iteration.
    }
}

}  // namespace strix::collection
