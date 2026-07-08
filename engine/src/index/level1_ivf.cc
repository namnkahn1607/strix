// Author: namnkahn1607
//
//

#include "level1_ivf.h"

#include <sys/mman.h>

#include <atomic>
#include <memory>
#include <stdexcept>

#include "avx2_kernel.h"
#include "constants.h"
#include "global_utils.h"

namespace {

static_assert(sizeof(std::atomic<uint16_t>) == sizeof(uint16_t));
static_assert(std::atomic<uint16_t>::is_always_lock_free);

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
static_assert(std::atomic<uint32_t>::is_always_lock_free);

}  // namespace

RoutingTable::RoutingTable(const size_t num_clusters_,
                           const size_t max_cluster_size_)
    : num_clusters_(num_clusters_), max_cluster_size_(max_cluster_size_) {
    if (num_clusters_ == 0 || max_cluster_size_ == 0) {
        throw std::invalid_argument(
            "Both number of clusters & cluster capacity must be non-zero");
    }

    if (num_clusters_ % kBatchSize != 0) {
        throw std::invalid_argument(
            "Number of clusters must be a multiple of kernel batch size");
    }

    centroids_ = static_cast<float*>(
        Alloc32(num_clusters_ * kVectorDim * sizeof(float), false));
    cluster_sizes_   = std::make_unique<std::atomic<uint16_t>[]>(num_clusters_);
    cluster_members_ = std::make_unique<std::atomic<uint32_t>[]>(
        num_clusters_ * max_cluster_size_);
}

RoutingTable::~RoutingTable() {
    munmap(centroids_, num_clusters_ * kVectorDim * sizeof(float));
}

size_t RoutingTable::MatchCluster(const float* query) const noexcept {
    float  best_score    = -1.0f;
    size_t best_centroid = 0;

    for (size_t i = 0; i < num_clusters_; i += kBatchSize) {
        float scores[kBatchSize];
        DotProductBatch(query, centroids_ + kVectorDim * i, scores);

        for (size_t k = 0; k < kBatchSize; ++k) {
            if (scores[k] > best_score) {
                best_score    = scores[k];
                best_centroid = i + k;
            }
        }
    }

    return best_centroid;
}

bool RoutingTable::JoinCluster(const uint32_t node_id,
                               const size_t   cluster_id) noexcept {
    // Relaxed load due to concurrency model of `cluster_member_`.
    const uint16_t size =
        cluster_sizes_[cluster_id].load(std::memory_order_relaxed);
    if (size >= max_cluster_size_) {
        return false;
    }

    // Write ahead, Publish later.
    cluster_members_[cluster_id * max_cluster_size_ + size].store(
        node_id, std::memory_order_relaxed);
    cluster_sizes_[cluster_id].store(size + 1, std::memory_order_release);
    return true;
}

bool RoutingTable::LeaveCluster(const size_t   cluster_id,
                                const size_t   member_index,
                                const uint32_t expected_node_id) noexcept {
    const size_t base = cluster_id * max_cluster_size_;

    // Relaxed load due to concurrency model of `cluster_member_`.
    const uint16_t size =
        cluster_sizes_[cluster_id].load(std::memory_order_relaxed);
    if (member_index >= size) {
        return false;
    }

    // Check if expected datapoint's value match actual one.
    const uint32_t actual_node_id =
        cluster_members_[base + member_index].load(std::memory_order_relaxed);
    if (actual_node_id != expected_node_id) {
        return false;
    }

    // Write ahead, Publish later.
    const uint16_t last_index = size - 1;
    if (member_index != last_index) {
        const uint32_t tail_value =
            cluster_members_[base + last_index].load(std::memory_order_relaxed);
        cluster_members_[base + member_index].store(tail_value,
                                                    std::memory_order_relaxed);
    }

    cluster_sizes_[cluster_id].store(last_index, std::memory_order_release);
    return true;
}
