// L1-tier IVF table implementation: ctor, dtor, member methods.

#include "ivf_table.h"

#include <sys/mman.h>

#include <atomic>
#include <memory>
#include <stdexcept>

#include "common/syscall_utils.h"
#include "dot_product/avx2_kernel.h"
#include "index/ivf_config.h"
#include "inference/info.h"

IvfTable::IvfTable(const IvfConfig& config)
    : num_clusters{config.num_clusters}
    , max_cluster_size{config.max_cluster_size} {
    if (num_clusters == 0 || max_cluster_size == 0) {
        throw std::invalid_argument(
            "Number of cluster and its capacity must be non-zero"
        );
    }
    if (num_clusters % kBatchSize != 0) {
        throw std::invalid_argument(
            "Number of cluster must be a multiple of 'kBatchSize'"
        );
    }

    centroids_ = static_cast<float*>(
        common::AllocMMap(num_clusters * kVectorMemsize, config.prefault)
    );
    cluster_sizes_   = std::make_unique<std::atomic<uint32_t>[]>(num_clusters);
    cluster_members_ = std::make_unique<std::atomic<uint32_t>[]>(
        num_clusters * max_cluster_size
    );
}

IvfTable::~IvfTable() {
    common::DeallocMMap(centroids_, num_clusters * kVectorMemsize);
}

uint32_t IvfTable::MatchCluster(const float* query) const noexcept {
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

bool IvfTable::JoinCluster(
    const uint32_t node_id, const uint32_t cluster_id
) noexcept {
    const uint32_t size =
        cluster_sizes_[cluster_id].load(std::memory_order_relaxed);
    if (size >= max_cluster_size) {
        return false;
    }

    cluster_members_[cluster_id * max_cluster_size + size].store(
        node_id, std::memory_order_relaxed
    );  // Write ahead,
    cluster_sizes_[cluster_id].store(
        size + 1, std::memory_order_release
    );  // Publish later
    return true;
}

bool IvfTable::LeaveCluster(
    const uint32_t cluster_id, const uint32_t member_index,
    const uint32_t expected_node_id
) noexcept {
    const uint32_t base = cluster_id * max_cluster_size;

    const uint32_t len =
        cluster_sizes_[cluster_id].load(std::memory_order_relaxed);
    if (member_index >= len) {
        return false;
    }

    const uint32_t actual_node_id =
        cluster_members_[base + member_index].load(std::memory_order_relaxed);
    if (actual_node_id != expected_node_id) {
        return false;
    }

    const uint32_t last_index = len - 1;
    if (member_index != last_index) {
        const uint32_t tail_value =
            cluster_members_[base + last_index].load(std::memory_order_relaxed
            );  // Write ahead
        cluster_members_[base + member_index].store(
            tail_value, std::memory_order_relaxed
        );  // Publish later
    }

    cluster_sizes_[cluster_id].store(last_index, std::memory_order_release);
    return true;
}
