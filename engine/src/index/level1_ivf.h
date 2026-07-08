// Author: namnkahn1607
//
// One generation of the L1 IVF layer.
// Owns its centroids and cluster membership.

#pragma once

#include <atomic>
#include <memory>

// `RoutingTable` acts as the "map" for L1 layer's IVF. Track and manage each
// cluster's centroid vector, size, members list (L1 Node's `node_id`).
class RoutingTable {
public:
    RoutingTable(size_t num_clusters_, size_t max_cluster_size_);
    ~RoutingTable();

    RoutingTable(const RoutingTable&)            = delete;
    RoutingTable& operator=(const RoutingTable&) = delete;
    RoutingTable(RoutingTable&&)                 = delete;
    RoutingTable& operator=(RoutingTable&&)      = delete;

    // `MatchCluster()` searches for most suitable cluster for an input vector.
    size_t MatchCluster(const float* query) const noexcept;

    // `JoinCluster()` appends `node_id` via swap-with-tail. Returns `false` if
    // the cluster is already at `MaxClusterSize()`, otherwise `true`.
    bool JoinCluster(uint32_t node_id, size_t cluster_id) noexcept;

    // `LeaveCluster()` remove the data point at `member_index` with cluster of
    // specified ID via swap-with-tail. `expected_node_id` must match the value
    // that's actually there, otherwise it's considered a bug and got rejected.
    bool LeaveCluster(size_t cluster_id, size_t member_index,
                      uint32_t expected_node_id) noexcept;

    // `ClusterMemberIds()` returns the pointer to the array containing all
    // member IDs of specified cluster.
    const std::atomic<uint32_t>* ClusterMemberIds(
        const size_t cluster_id) const noexcept {
        return cluster_members_.get() + cluster_id * max_cluster_size_;
    }

    uint16_t ClusterSize(
        const size_t      cluster_id,
        std::memory_order order = std::memory_order_acquire) const noexcept {
        return cluster_sizes_[cluster_id].load(order);
    }

    size_t NumClusters() const noexcept {
        return num_clusters_;
    }

    size_t MaxClusterSize() const noexcept {
        return max_cluster_size_;
    }

private:
    const size_t num_clusters_;
    const size_t max_cluster_size_;

    float*                                   centroids_;
    std::unique_ptr<std::atomic<uint16_t>[]> cluster_sizes_;

    // Cluster datapoint tracker. Concurrency model: multiple Reader, single
    // Writer (Compaction, Reassignment).
    std::unique_ptr<std::atomic<uint32_t>[]> cluster_members_;
};
