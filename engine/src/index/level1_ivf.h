// Author: namnkahn1607
//
// One generation of the L1 IVF layer.
// Owns its centroids and cluster membership.

#pragma once

#include <atomic>
#include <memory>

// `IvfConfig` describes the specifications of IVF at construction time. Note
// that all validation is performed inside the constructor.
//
// Fields:
//   1. `num_clusters`       : Total number of clusters. Must be non-zero and a
//                             multiple of `kBatchSize`.
//   2. `max_cluster_size`   : A cluster's maximal capacity. Must be greater
//                             than the average number of vectors per cluster.
//   3. `kmeans_sample_size` : The sample space size K-means in Recalibration
//                             process decides to take on.
//   4. `lazy_mapping`       : When `false`, `mmap` uses `MAP_POPULATE` to
//                             pre-fault all pages at construction, eliminating
//                             page-fault latency during operation.
//                             When `true`, pages are faulted on first access
//                             (lower startup cost, higher first-touch latency).
//
// The average number of vectors per cluster: `max_slots / num_clusters`.
struct IvfConfig {
    const uint32_t num_clusters;
    const uint32_t max_cluster_size;
    const uint32_t kmeans_sample_size;
    const bool     lazy_mapping;

    // `Production()` config: 1024 clusters, 1024 max size, 50'000 sample size
    // with pre-fault pages enabled.
    static IvfConfig Production() {
        return IvfConfig{1'024, 1'024, 50'000, false};
    }

    // `Compact()` config: user-specified sizes and dimensions with pre-fault
    // pages enabled.
    static IvfConfig Compact(uint32_t num_cluster, uint32_t max_cluster_size,
                             uint32_t kmeans_sample_size) {
        return IvfConfig{num_cluster, max_cluster_size, kmeans_sample_size,
                         false};
    }

    // `CompactLazy()` config: same as `Config()`, but lazily mapped.
    static IvfConfig CompactLazy(uint32_t num_cluster,
                                 uint32_t max_cluster_size,
                                 uint32_t kmeans_sample_size) {
        return IvfConfig{num_cluster, max_cluster_size, kmeans_sample_size,
                         true};
    }
};

// `RoutingTable` acts as the "map" for L1 layer's IVF. Track and manage each
// cluster's centroid vector, size, members list (L1 Node's `node_id`).
class RoutingTable {
public:
    explicit RoutingTable(uint32_t num_clusters, uint32_t max_cluster_size,
                          bool lazy_mapping);
    explicit RoutingTable(const IvfConfig& config);
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

    // `MatchCluster()` searches for most suitable cluster for an input vector.
    uint32_t MatchCluster(const float* query) const noexcept;

    // `JoinCluster()` appends `node_id` via swap-with-tail. Returns `false` if
    // the cluster is already at `MaxClusterSize()`, otherwise `true`.
    bool JoinCluster(uint32_t node_id, uint32_t cluster_id) noexcept;

    // `LeaveCluster()` remove the data point at `member_index` with cluster of
    // specified ID via swap-with-tail. `expected_node_id` must match the value
    // that's actually there, otherwise it's considered a bug and got rejected.
    bool LeaveCluster(uint32_t cluster_id, uint32_t member_index,
                      uint32_t expected_node_id) noexcept;

    // `ClusterMemberIds()` returns the pointer to the array containing all
    // member IDs of specified cluster.
    inline const std::atomic<uint32_t>* ClusterMemberIds(
        const uint32_t cluster_id) const noexcept {
        return cluster_members_.get() + cluster_id * max_cluster_size;
    }

    // `ClusterSize()` retrieves a cluster's current size through specified
    // memory order load.
    inline uint16_t ClusterSize(
        const uint32_t    cluster_id,
        std::memory_order order = std::memory_order_acquire) const noexcept {
        return cluster_sizes_[cluster_id].load(order);
    }

private:
    float*                                   centroids_;
    std::unique_ptr<std::atomic<uint16_t>[]> cluster_sizes_;

    // Cluster datapoint tracker. Concurrency model: multiple Reader, single
    // Writer (Compaction, Reassignment).
    std::unique_ptr<std::atomic<uint32_t>[]> cluster_members_;
};
