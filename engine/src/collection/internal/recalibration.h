#pragma once

#include <atomic>
#include <memory>
#include <random>

#include "collection/config.h"
#include "ivf_table.h"
#include "memory/arena.h"

namespace strix::collection {

// The K-means++ seeding and Lloyd mini-batch state machine.
class Recalibrator {
public:
    // Phase represents state machine of Recalibration process.
    enum class Phase : uint8_t {
        kIdle,
        kGathering,
        kKMeansSeeding,
        kMiniBatch,
    };

    explicit Recalibrator(
        const Config& config, memory::Arena& arena, IvfTable* table,
        std::atomic<uint8_t>* active_route
    );
    ~Recalibrator();

    Recalibrator(const Recalibrator&)            = delete;
    Recalibrator& operator=(const Recalibrator&) = delete;
    Recalibrator(Recalibrator&&)                 = delete;
    Recalibrator& operator=(Recalibrator&&)      = delete;

    // Notifies the Recalibrator about a node migrated to L1-tier.
    // Called per successful Compaction.
    void NotifyCompactionSucceeded() noexcept { ++ingestion_count_; }

    // Triggers one Recalibration step. Spread across multiple invocations.
    void Tick() noexcept;

    Phase CurrentPhase() const noexcept { return curr_phase_; }

private:
    void StepGathering() noexcept;
    void StepKmeansPPSeeding() noexcept;
    void StartMiniBatchPhase() noexcept;
    void StepMiniBatch() noexcept;
    void PublishRecalibration() noexcept;

    Config                config_;
    memory::Arena&        arena_;
    IvfTable*             table_;
    std::atomic<uint8_t>* active_route_;

    std::mt19937 rng_;
    float*       sample_buf_;

    Phase     curr_phase_          = Phase::kIdle;
    uint32_t  ingestion_count_     = 0;
    uint32_t  gathered_count_      = 0;
    bool      first_recalibration_ = true;
    TimePoint last_publish_time_   = TimePoint{};

    uint32_t kmeanspp_seeded_            = 0;
    uint32_t mini_batch_iterations_done_ = 0;

    std::unique_ptr<float[]>    kmeanspp_min_dist_sq_;
    std::unique_ptr<uint32_t[]> kmeans_cluster_counts_;
};

}  // namespace strix::collection
