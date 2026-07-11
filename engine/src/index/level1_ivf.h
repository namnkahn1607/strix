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
//   1. `num_clusters` : Total number of clusters. Must be non-zero and a
//                       multiple of `kBatchSize`.
//   2. `max_cluster_size` : A cluster's maximal capacity. Must be greater than
//                           the average number of vectors per cluster.
//   3. `kmeans_sample_size` : The sample space size K-means in Recalibration
//                             process decides to take on.
//   4. `lazy_mapping` : When `false`, `mmap` uses `MAP_POPULATE` to pre-fault
//                       all pages at construction, eliminating page-fault
//                       latency during operation.
//   5. `recalibration_trigger_count` : Number of successful Compactions since
//                                      the last Recalibration publish before
//                                      starting a new episode.
//   6. `mini_batch_size` : Samples drawn per mini-batch step.
//   7. `max_lloyd_iterations` : Number of mini-batch steps per episode. Bounds
//                               worst-case episode duration regardless of
//                               whether `convergence_epsilon` is ever reached.
//   8. `convergence_epsilon` : Early termination signal if total centroid
//                              movement dips below this threshold.
//   9. `grace_period_seconds` : Minimal wait time after a publish before
//                               starting the next episode. Readers against the
//                               buffer just vacated have had a chance to finish
//                               before it's overwritten again.
//
// The average number of vectors per cluster: `max_slots / num_clusters`.
struct IvfConfig {
    const uint32_t num_clusters;
    const uint32_t max_cluster_size;
    const uint32_t kmeans_sample_size;
    const bool     lazy_mapping;
    const uint32_t recalibration_trigger_count;
    const uint32_t mini_batch_size;
    const uint32_t max_lloyd_iterations;
    const float    convergence_epsilon;
    const uint64_t grace_period_seconds;

    // `Production()` config: 1024 clusters, 1024 max size, 50'000 sample size
    // with pre-fault pages enabled.
    static IvfConfig Production() {
        return IvfConfig{
            /*num_clusters=*/1'024,
            /*max_cluster_size=*/1'024,
            /*kmeans_sample_size=*/65'536,
            /*lazy_mapping=*/false,
            /*recalibration_trigger_count=*/16'384,
            /*mini_batch_size=*/1024,
            /*max_lloyd_iterations=*/16,
            /*convergence_epsilon=*/1e-4f,
            /*grace_period_seconds=*/1,
        };
    }

    // `Compact()` config: user-specified sizes and dimensions with pre-fault
    // pages enabled.
    static IvfConfig Compact(uint32_t num_clusters, uint32_t max_cluster_size,
                             uint32_t kmeans_sample_size) {
        return IvfConfig{
            num_clusters,
            max_cluster_size,
            kmeans_sample_size,
            /*lazy_mapping=*/false,
            /*recalibration_trigger_count=*/16'384,
            /*mini_batch_size=*/1024,
            /*max_lloyd_iterations=*/16,
            /*convergence_epsilon=*/1e-4f,
            /*grace_period_seconds=*/1,
        };
    }

    // `CompactLazy()` config: same as `Config()`, but lazily mapped.
    static IvfConfig CompactLazy(uint32_t num_clusters,
                                 uint32_t max_cluster_size,
                                 uint32_t kmeans_sample_size) {
        return IvfConfig{
            num_clusters,
            max_cluster_size,
            kmeans_sample_size,
            /*lazy_mapping=*/true,
            /*recalibration_trigger_count=*/16'384,
            /*mini_batch_size=*/1024,
            /*max_lloyd_iterations=*/16,
            /*convergence_epsilon=*/1e-4f,
            /*grace_period_seconds=*/1,
        };
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

    // `CentroidRow()` returns a read-only access to a centroid vector. Used by
    // K-means++ seeding in Recalibration to measure sample distance to current
    // centroid and by the mini-batch step.
    const float* CentroidRow(uint32_t cluster_id) const noexcept;

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
