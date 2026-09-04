// L1-tier IVF table.

#pragma once

#include <atomic>
#include <cstring>
#include <memory>

#include "collection/config.h"
#include "inference/info.h"

namespace strix::collection {

// Acts as the "routing map" for L1-tier cache that tracks and manages each
// cluster's centroid vector, size, member node ID list.
// Lock-free and thread-safe under multiple readers, single writer.
class IvfTable {
public:
    explicit IvfTable(const Config& config);
    ~IvfTable();

    IvfTable(const IvfTable&)            = delete;
    IvfTable& operator=(const IvfTable&) = delete;
    IvfTable(IvfTable&&)                 = delete;
    IvfTable& operator=(IvfTable&&)      = delete;

    // Decides the current nearest cluster to a vector.
    uint32_t MatchCluster(const float* vec) const noexcept;

    // Appends a node ID into a cluster via.
    // Returns `false` if the cluster is already saturated, otherwise `true`.
    bool JoinCluster(uint32_t node_id, uint32_t cluster_id) noexcept;

    // Removes a node ID from a cluster.
    // Expects caller to
    // Expects caller to provide node ID value and its position in cluster
    // buffer, otherwise the invocation is rejected.
    bool LeaveCluster(
        uint32_t cluster_id, uint32_t member_idx, uint32_t exp_node_id
    ) noexcept;

    // Retrieves a cluster's current size.
    // By default, an `std::memory_order_acquire` load is performed.
    uint32_t ClusterSize(
        uint32_t cluster_id, std::memory_order order = std::memory_order_acquire
    ) const noexcept {
        return cluster_sizes_[cluster_id].load(order);
    }

    const float* GetCentroid(uint32_t cluster_id) const noexcept {
        return centroids_ +
               static_cast<size_t>(cluster_id) * inference::kVectorDim;
    }

    void SetCentroid(uint32_t cluster_id, const float* vector) noexcept {
        std::memcpy(
            centroids_ +
                static_cast<size_t>(cluster_id) * inference::kVectorDim,
            vector, inference::kVectorMemsize
        );
    }

    const std::atomic<uint32_t>* ListMembers(uint32_t cluster_id
    ) const noexcept {
        return cluster_members_.get() +
               static_cast<size_t>(cluster_id) * max_cluster_size;
    }

    const uint32_t num_clusters;
    const uint32_t max_cluster_size;

private:
    float* centroids_;

    std::unique_ptr<std::atomic<uint32_t>[]> cluster_sizes_;
    std::unique_ptr<std::atomic<uint32_t>[]> cluster_members_;
};

}  // namespace strix::collection
