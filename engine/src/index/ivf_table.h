// L1 cache tier IVF table declaration.

#pragma once

#include <atomic>
#include <cstring>
#include <memory>

#include "index/ivf_config.h"
#include "inference/info.h"

// IvfTable acts as the "routing map" for L1-tier cache that tracks and manages
// each cluster's centroid vector, size, member node ID list.
//
// Thread-safe under multiple reader, single writer.
//
// Ownership model: construct once, pass by reference to consumers.
class IvfTable {
public:
    static constexpr uint32_t kUnclustered = 0xFFFFFFFFu;

    explicit IvfTable(const IvfConfig& ivf_config);
    ~IvfTable();

    IvfTable(const IvfTable&)            = delete;
    IvfTable& operator=(const IvfTable&) = delete;
    IvfTable(IvfTable&&)                 = delete;
    IvfTable& operator=(IvfTable&&)      = delete;

    // Returns nearest cluster's ID for specified vector.
    uint32_t MatchCluster(const float* query) const noexcept;

    // Appends `node_id` into a cluster via swap-with-tail technique.
    // Returns `false` if the cluster is already at max size, otherwise `true`.
    bool JoinCluster(uint32_t node_id, uint32_t cluster_id) noexcept;

    // Removes datapoint of a cluster using swap-with-tail technique.
    // Asserts that node ID at `member_idx` equals `exp_node_id`, otherwise the
    // call is rejected and do nothing.
    bool LeaveCluster(
        uint32_t cluster_id, uint32_t member_idx, uint32_t exp_node_id
    ) noexcept;

    // Fills specified centroid with given vector.
    // Used during IVF's bootstrap phase, when no initial centroid exists yet.
    void SeedCentroid(const uint32_t cluster_id, const float* vector) noexcept {
        std::memcpy(
            centroids_ + static_cast<size_t>(cluster_id) * kVectorDim, vector,
            kVectorMemsize
        );
    }

    // Read-only access to a centroid vector.
    const float* CentroidVector(const uint32_t cluster_id) const noexcept {
        return centroids_ + static_cast<size_t>(cluster_id) * kVectorDim;
    }

    // Read-only access to member IDs of a cluster.
    const std::atomic<uint32_t>* ClusterMember(const uint32_t cluster_id
    ) const noexcept {
        return cluster_members_.get() +
               static_cast<size_t>(cluster_id) * max_cluster_size;
    }

    // Retrieves a cluster's current size.
    // By default, an `std::memory_order_acquire` load is performed.
    uint32_t ClusterSize(
        const uint32_t    cluster_id,
        std::memory_order order = std::memory_order_acquire
    ) const noexcept {
        return cluster_sizes_[cluster_id].load(order);
    }

    const uint32_t num_clusters;
    const uint32_t max_cluster_size;

private:
    float* centroids_;

    std::unique_ptr<std::atomic<uint32_t>[]> cluster_sizes_;
    std::unique_ptr<std::atomic<uint32_t>[]> cluster_members_;
};
