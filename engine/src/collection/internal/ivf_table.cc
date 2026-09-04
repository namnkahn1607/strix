// L1-tier IVF table.

#include "ivf_table.h"

#include <sys/mman.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#include "ann/dot_product.h"
#include "collection/config.h"
#include "inference/info.h"
#include "memory/allocator.h"

namespace strix::collection {

IvfTable::IvfTable(const Config& config)
    : num_clusters{config.num_clusters}
    , max_cluster_size{config.max_cluster_size} {
    if (num_clusters == 0 || max_cluster_size == 0) {
        throw std::invalid_argument(
            "Number of clusters and its capacity must be non-zero"
        );
    }
    if (num_clusters % ann::kBatchSize != 0) {
        throw std::invalid_argument(
            "Number of clusters must be a multiple of " +
            std::to_string(ann::kBatchSize)
        );
    }

    centroids_ =
        memory::Alloc<float>(num_clusters * inference::kVectorDim, true);
    cluster_sizes_   = std::make_unique<std::atomic<uint32_t>[]>(num_clusters);
    cluster_members_ = std::make_unique<std::atomic<uint32_t>[]>(
        static_cast<size_t>(num_clusters) * max_cluster_size
    );
}

IvfTable::~IvfTable() {
    memory::Dealloc(centroids_, num_clusters * inference::kVectorDim);
}

uint32_t IvfTable::MatchCluster(const float* query) const noexcept {
    float    best_score    = -1.0f;
    uint32_t best_centroid = 0;

    for (uint32_t i = 0; i < num_clusters; i += ann::kBatchSize) {
        float scores[ann::kBatchSize];
        ann::BatchDotProduct(
            query, centroids_ + inference::kVectorDim * i, scores
        );

        for (uint32_t k = 0; k < ann::kBatchSize; ++k) {
            if (scores[k] > best_score) {
                best_score    = scores[k];
                best_centroid = i + k;
            }
        }
    }

    return best_centroid;
}

bool IvfTable::JoinCluster(uint32_t node_id, uint32_t cluster_id) noexcept {
    const uint32_t size =
        cluster_sizes_[cluster_id].load(std::memory_order_relaxed);
    if (size >= max_cluster_size) {
        return false;
    }

    cluster_members_[cluster_id * max_cluster_size + size].store(
        node_id, std::memory_order_relaxed
    );  // Write ahead
    cluster_sizes_[cluster_id].store(
        size + 1, std::memory_order_release
    );  // Publish later
    return true;
}

bool IvfTable::LeaveCluster(
    uint32_t cluster_id, uint32_t member_idx, uint32_t exp_node_id
) noexcept {
    const uint32_t base = cluster_id * max_cluster_size;

    const uint32_t len =
        cluster_sizes_[cluster_id].load(std::memory_order_relaxed);
    if (member_idx >= len) {
        return false;
    }

    const uint32_t actual_node_id =
        cluster_members_[base + member_idx].load(std::memory_order_relaxed);
    if (actual_node_id != exp_node_id) {
        return false;
    }

    const uint32_t last_idx = len - 1;
    if (member_idx != last_idx) {
        const uint32_t tail_value =
            cluster_members_[base + last_idx].load(std::memory_order_relaxed
            );  // Write ahead
        cluster_members_[base + member_idx].store(
            tail_value, std::memory_order_relaxed
        );  // Publish later
    }

    cluster_sizes_[cluster_id].store(last_idx, std::memory_order_release);
    return true;
}

}  // namespace strix::collection
