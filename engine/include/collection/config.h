#pragma once

#include <chrono>

namespace strix::collection {

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Config {
    // L0-tier node buffer capacity. Non-zero and a power of 2.
    const uint32_t lvl0_capacity;

    // Total number of clusters.
    // Non-zero and a multiple of dot product kernel batch size.
    const uint32_t num_clusters;

    // A cluster's maximum capacity.
    // Geq average number of vectors per cluster.
    const uint32_t max_cluster_size;

    // Sample space size K-means decides to take on.
    // Non-zero and a multiple of dot product kernel batch size.
    const uint32_t sample_size;

    // New vector ingestion threshold before starting a new calibrate session.
    const uint32_t recalibrate_trigger_count;

    // Samples drawn per mini-batch step.
    const uint32_t mini_batch_size;

    // Number of mini-batch step per episode-work.
    // Bounds worst-case episode duration regardless of reached
    // `min_convergence`.
    const uint32_t max_lloyd_iterations;

    // Early K-means early termination threshold.
    // Activates once total centroid movement dips below it.
    const float min_convergence;

    // Minimum wait time after a publish before starting a new episode.
    const std::chrono::seconds grace_period;

    static Config Standard() {
        // Sized to fit L2/L3 cache for the frontier dynamic workset.
        constexpr uint32_t kLvl0Capacity = 1 << 12;

        // `N = 524'288`, `K = 1'024` fits in range `[sqrt(N), 4 * sqrt(N)]`.
        constexpr uint32_t kNumClusters = 1'024u;

        // Sits inside diminishing-returns of Sculley (2010).
        constexpr uint32_t kMiniBatchSize = 1'024u;

        return {
            kLvl0Capacity,
            kNumClusters,
            /*max_cluster_size=*/1'024u,
            /*sample_size=*/65'536u,
            /*recalibrate_trigger_count=*/16'384u,
            kMiniBatchSize,
            /*max_lloyd_iterations=*/16,
            /*min_convergence=*/1e-4f,
            /*grace_period=*/std::chrono::seconds{1},
        };
    }
};

}  // namespace strix::collection
