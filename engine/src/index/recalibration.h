// Author: namnkahn1607
//
// The K-means++ seeding and mini-batch Lloyd's iteration state
// machine that periodically rebuilds the IVF.

#pragma once

#include <atomic>
#include <memory>
#include <random>

#include "level1_ivf.h"

// `Recalibrator` operates the Recalibration phase and provides API to IVF's
// background coordination.
//
// Ownership model: intialized once, reference owned by `VectorIndex`.
class Recalibrator {
public:
    explicit Recalibrator(RoutingTable*         routes,
                          std::atomic<uint8_t>* active_route,
                          const IvfConfig&      config);
    ~Recalibrator();

    Recalibrator(const Recalibrator&)            = delete;
    Recalibrator& operator=(const Recalibrator&) = delete;
    Recalibrator(Recalibrator&&)                 = delete;
    Recalibrator& operator=(Recalibrator&&)      = delete;

    // `Phase` enum represents the state machine of Recalibration.
    enum class Phase : uint8_t {
        kIdle,
        kKMeansSeeding,
        kMiniBatch,
    };

    inline Phase CurrentPhase() const noexcept {
        return phase_;
    }

    // `NotifyCompactionSucceeded()` is called once per successful
    // compaction to notify `Recalibrator`.
    inline void NotifyCompactionSucceeded() noexcept {
        ++compaction_performed_;
    }

    // `Tick()` triggers one Recalibration routine that spreads across
    // many invocations.
    void Tick() noexcept;

private:
    // Non-owning (borrowed) fields from `VectorIndex`.
    RoutingTable*         routes_;
    std::atomic<uint8_t>* active_route_;
    IvfConfig             config_;

    std::mt19937 rng_;
    float*       kmeans_sample_;

    Phase    phase_                = Phase::kIdle;
    uint32_t recalibration_count_  = 0;  // Episode published so far
    uint32_t compaction_performed_ = 0;
    uint64_t last_publish_time_    = 0;  // Unix seconds

    // K-means++ seeding only-state.
    uint32_t                 kmeanspp_seeded_ = 0;
    std::unique_ptr<float[]> kmeanspp_min_dist_sq_;

    void StepKmeansPPSeeding() noexcept;

    // Mini-batch only-state.
    uint32_t                    mini_batch_iterations_done_ = 0;
    std::unique_ptr<uint32_t[]> kmeans_cluster_counts_;

    void StartMiniBatchPhase() noexcept;
    void StepMiniBatch() noexcept;

    void PublishRecalibration() noexcept;
};
