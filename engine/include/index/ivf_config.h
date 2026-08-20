// Inverted Index runtime configuration.

#pragma once

// IvfConfig defines runtime specification of IVF at construction time.
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

    // Configures `1024` clusters, `1024` max size with page prefault.
    static IvfConfig Standard() {
        return {
            /*num_clusters=*/1'024,
            /*max_cluster_size=*/1'024,
        };
    }
};
