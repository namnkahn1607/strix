// Inverted Index runtime configuration.

#pragma once

// IvfConfig defines the specification of IVF at construction time.
//
// All validation are performed inside the IVF constructor.
// NOTE: The average number of vectors per cluster: `max_slots / num_clusters`.
struct IvfConfig {
    // Total number of clusters.
    // Must be non-zero and a multiple of `kBatchSize`.
    const uint32_t num_clusters;

    // A cluster's maximum capacity.
    // Must be greater than the average number of vectors per cluster.
    const uint32_t max_cluster_size;

    // The sample space size K-means process decides to take on.
    const uint32_t kmeans_sample_size;

    // Page prefault enabled? (default: `true`).
    const bool prefault = true;

    // Number of successful Compactions since last episode published
    // before starting a new one.
    const uint32_t recalibration_trigger_count;

    // Samples drawn per mini-batch step.
    const uint32_t mini_batch_size;

    // Number of mini-batch step per episode.
    // Bounds worst-case episode duration regardless of `convergence_epsilon`
    // is ever reached or not.
    const uint32_t max_lloyd_iterations;

    // Early termination threshold if total centroid movement dips below it.
    const float convergence_epsilon;

    // Minimum wait time after a publish before starting a new episode.
    const uint64_t grace_period_seconds;

    // Configures `1024` clusters, `1024` max size, `65'536` sample size
    // with page prefault enabled.
    static IvfConfig Production() {
        return IvfConfig{
            /*num_clusters=*/1'024,
            /*max_cluster_size=*/1'024,
            /*kmeans_sample_size=*/65'536,
            /*prefault=*/true,
            /*recalibration_trigger_count=*/16'384,
            /*mini_batch_size=*/1024,
            /*max_lloyd_iterations=*/16,
            /*convergence_epsilon=*/1e-4f,
            /*grace_period_seconds=*/1,
        };
    }
};
