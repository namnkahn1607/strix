// K-means algorithm runtime configuration.

#pragma once

// KmeansConfig defines runtime specification for K-means in IVF.
//
// All validation are performed inside the constructor.
struct KmeansConfig {
    // The sample space K-means decides to take on.
    const uint32_t sample_size;

    // New node ingestion threshold before starting a new calibrate session.
    const uint32_t recalibrate_trigger_count;

    // Samples drawn per mini-batch step.
    const uint32_t mini_batch_size;

    // Number of mini-batch steps per episode-work.
    // Bounds worst-case episode duration regardless `convergence_epsilon`
    // is ever reached.
    const uint32_t max_lloyd_iterations;

    // Early K-means termination threshold.
    // Activates once total centroid movement dips below it.
    const float convergence_epsilon;

    // Minimum wait time after a publish before starting a new episode.
    const uint64_t grace_period_secs;

    // Standard configurations
    static KmeansConfig Standard() noexcept {
        return {
            /*sample_size=*/65'536u,
            /*recalibrate_trigger_count=*/16'384u,
            /*mini_batch_size=*/1'024u,
            /*max_lloyd_iterations=*/16,
            /*convergence_epsilon=*/1e-4f,
            /*grace_period_secs=*/1,
        };
    }
};
