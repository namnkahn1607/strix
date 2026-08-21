// Vector Index configuration.

#pragma once

// Defines specification for `VectorIndex` at construction time.
//
// All validation are conducted inside the constructor.
struct IndexConfig {
    // L0-tier node buffer capacity.
    // Non-zero and a power of 2.
    const uint32_t l0_capacity;

    // Total number of clusters.
    // Non-zero and a multiple of `kBatchSize`.
    const uint32_t num_clusters;

    // A cluster's maximum capacity.
    // Greater than or equal to the average number of vectors per cluster.
    const uint32_t max_cluster_size;

    // Sample space size K-means decides to take on.
    // Non-zero and a multiple of `kBatchSize`.
    const uint32_t sample_size;

    // Ingestion threshold before starting a new calibarte session.
    const uint32_t recalibrate_trigger_count;

    // Samples drawn per mini-batch step.
    const uint32_t mini_batch_size;

    // Number of mini-batch step per episode-work.
    // Bounds worst-case episode duration regardless `min_convergence` is
    // ever reached.
    const uint32_t max_lloyd_iterations;

    // Early K-means termination threshold.
    // Activates once total centroid movement dips below it.
    const float min_convergence;

    // Minimum wait time after a publish before starting a new episode.
    const uint64_t grace_period_secs;

    static IndexConfig Standard() {
        // Sized to fit L2/L3 cache for the frontier dynamic workset.
        constexpr uint32_t kL0Capacity = 1 << 12;

        // `N = 524'288`, `K = 1'024` fits in range `[sqrt(N), 4 * sqrt(N)]`.
        constexpr uint32_t kNumClusters = 1'024u;

        // Sit inside diminishing-returns of Sculley (2010).
        constexpr uint32_t kMiniBatchSize = 1'024u;

        return {
            kL0Capacity,
            kNumClusters,
            /*max_cluster_size=*/1'024u,
            /*sample_size=*/65'536u,
            /*recalibrate_trigger_count=*/16'384u,
            kMiniBatchSize,
            /*max_lloyd_iterations=*/16,
            /*min_convergence=*/1e-4f,
            /*grace_period_secs=*/1,
        };
    }
};
