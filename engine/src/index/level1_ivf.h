// Author: namnkahn1607
//
// One generation of the L1 IVF layer.
// Owns its centroids and cluster membership.

#pragma once

#include <atomic>
#include <memory>

// A state which stands for "not yet assigned to any cluster".
// Hence, every L0 node is unclustered.
inline constexpr uint32_t kUnclustered = 0xFFFFFFFFU;

// `RoutingTable` acts as the "map" for L1 layer's IVF. Track and manage each
// cluster's centroid vector, size, members list (L1 Node's `node_id`).
class RoutingTable {
public:
    explicit RoutingTable(
        uint32_t num_clusters, uint32_t max_cluster_size, bool lazy_mapping
    );
    ~RoutingTable();

    RoutingTable(const RoutingTable&)            = delete;
    RoutingTable& operator=(const RoutingTable&) = delete;
    RoutingTable(RoutingTable&&)                 = delete;
    RoutingTable& operator=(RoutingTable&&)      = delete;

    const uint32_t num_clusters;
    const uint32_t max_cluster_size;

    // `SeedCentroid()` fills a specified cluster's centroid with given vector.
    // Used during the bootstrap phase of the system, while there're no initial
    // vectors as centroids.
    void SeedCentroid(uint32_t cluster_id, const float* vector) noexcept;

    // `CentroidRow()` returns a read-only access to a centroid vector. Used by
    // K-means++ seeding in Recalibration to measure sample distance to current
    // centroid and by the mini-batch step.
    const float* CentroidVector(uint32_t cluster_id) const noexcept;

    // `MatchCluster()` searches for most suitable cluster for an input vector.
    uint32_t MatchCluster(const float* query) const noexcept;

    // `JoinCluster()` appends `node_id` via swap-with-tail. Returns `false` if
    // the cluster is already at `MaxClusterSize()`, otherwise `true`.
    bool JoinCluster(uint32_t node_id, uint32_t cluster_id) noexcept;

    // `LeaveCluster()` remove the data point at `member_index` with cluster of
    // specified ID via swap-with-tail. `expected_node_id` must match the value
    // that's actually there, otherwise it's considered a bug and got rejected.
    bool LeaveCluster(
        uint32_t cluster_id, uint32_t member_index, uint32_t expected_node_id
    ) noexcept;

    // `ClusterMemberIds()` returns the pointer to the array containing all
    // member IDs of specified cluster.
    inline const std::atomic<uint32_t>* ClusterMemberIds(
        const uint32_t cluster_id
    ) const noexcept {
        return cluster_members_.get() +
               static_cast<size_t>(cluster_id) * max_cluster_size;
    }

    // `ClusterSize()` retrieves a cluster's current size through specified
    // memory order load.
    inline uint32_t ClusterSize(
        const uint32_t    cluster_id,
        std::memory_order order = std::memory_order_acquire
    ) const noexcept {
        return cluster_sizes_[cluster_id].load(order);
    }

private:
    float*                                   centroids_;
    std::unique_ptr<std::atomic<uint32_t>[]> cluster_sizes_;

    // Cluster datapoint tracker. Concurrency model: multiple Reader, single
    // Writer (Compaction, Reassignment).
    std::unique_ptr<std::atomic<uint32_t>[]> cluster_members_;
};
