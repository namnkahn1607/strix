// Author: namnkahn1607
//
// One generation of the L1 IVF layer.
// Owns its centroids and cluster membership.

#include "level1_ivf.h"

#include <sys/mman.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "common/constants.h"
#include "common/syscall_utils.h"
#include "index/avx2_kernel.h"

RoutingTable::RoutingTable(
    const uint32_t num_clusters, const uint32_t max_cluster_size,
    const bool lazy_mapping
)
    : num_clusters(num_clusters), max_cluster_size(max_cluster_size) {
    if (num_clusters == 0 || max_cluster_size == 0) {
        throw std::invalid_argument(
            "Both number of clusters & cluster capacity must be non-zero"
        );
    }

    if (num_clusters % kBatchSize != 0) {
        throw std::invalid_argument(
            "Number of clusters must be a multiple of kernel batch size"
        );
    }

    centroids_ = static_cast<float*>(
        common::AllocMMap(num_clusters * kVectorMemsize, lazy_mapping)
    );
    cluster_sizes_   = std::make_unique<std::atomic<uint32_t>[]>(num_clusters);
    cluster_members_ = std::make_unique_for_overwrite<std::atomic<uint32_t>[]>(
        num_clusters * max_cluster_size
    );
}

RoutingTable::~RoutingTable() {
    common::DeallocMMap(centroids_, num_clusters * kVectorMemsize);
}

void RoutingTable::SeedCentroid(
    const uint32_t cluster_id, const float* vector
) noexcept {
    std::memcpy(centroids_ + cluster_id * kVectorDim, vector, kVectorMemsize);
}

const float* RoutingTable::CentroidVector(const uint32_t cluster_id
) const noexcept {
    return centroids_ + cluster_id * kVectorDim;
}

uint32_t RoutingTable::MatchCluster(const float* query) const noexcept {
    float    best_score    = -1.0f;
    uint32_t best_centroid = 0;

    for (uint32_t i = 0; i < num_clusters; i += kBatchSize) {
        float scores[kBatchSize];
        DotProductContiguousBatch(query, centroids_ + kVectorDim * i, scores);

        for (uint32_t k = 0; k < kBatchSize; ++k) {
            if (scores[k] > best_score) {
                best_score    = scores[k];
                best_centroid = i + k;
            }
        }
    }

    return best_centroid;
}

bool RoutingTable::JoinCluster(
    const uint32_t node_id, const uint32_t cluster_id
) noexcept {
    // Relaxed load due to concurrency model of `cluster_member_`.
    const uint32_t size =
        cluster_sizes_[cluster_id].load(std::memory_order_relaxed);
    if (size >= max_cluster_size) {
        return false;
    }

    // Write ahead, Publish later.
    cluster_members_[cluster_id * max_cluster_size + size].store(
        node_id, std::memory_order_relaxed
    );
    cluster_sizes_[cluster_id].store(size + 1, std::memory_order_release);
    return true;
}

bool RoutingTable::LeaveCluster(
    const uint32_t cluster_id, const uint32_t member_index,
    const uint32_t expected_node_id
) noexcept {
    const uint32_t base = cluster_id * max_cluster_size;

    // Relaxed load due to concurrency model of `cluster_member_`.
    const uint32_t len =
        cluster_sizes_[cluster_id].load(std::memory_order_relaxed);
    if (member_index >= len) {
        return false;
    }

    // Check if expected datapoint's value match actual one.
    const uint32_t actual_node_id =
        cluster_members_[base + member_index].load(std::memory_order_relaxed);
    if (actual_node_id != expected_node_id) {
        return false;
    }

    // Write ahead, Publish later.
    const uint32_t last_index = len - 1;
    if (member_index != last_index) {
        const uint32_t tail_value =
            cluster_members_[base + last_index].load(std::memory_order_relaxed);
        cluster_members_[base + member_index].store(
            tail_value, std::memory_order_relaxed
        );
    }

    cluster_sizes_[cluster_id].store(last_index, std::memory_order_release);
    return true;
}
