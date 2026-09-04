#include "recalibration.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "ann/dot_product.h"
#include "collection/config.h"
#include "inference/info.h"
#include "memory/allocator.h"
#include "memory/arena.h"

namespace strix::collection {

Recalibrator::Recalibrator(
    const Config& config, memory::Arena& arena, IvfTable* table,
    std::atomic<uint8_t>* active_route
)
    : config_{config}
    , arena_{arena}
    , table_{table}
    , active_route_{active_route}
    , rng_{std::random_device{}()} {
    if (config.sample_size == 0) {
        throw std::invalid_argument(
            "K-means sample buffer size must be non-zero"
        );
    }
    if (config.sample_size % ann::kBatchSize != 0) {
        throw std::invalid_argument(
            "K-means sample size must be a multiple of " +
            std::to_string(ann::kBatchSize)
        );
    }

    sample_buf_ =
        memory::Alloc<float>(config.sample_size * inference::kVectorDim, true);
    kmeanspp_min_dist_sq_ =
        std::make_unique_for_overwrite<float[]>(config.sample_size);
    kmeans_cluster_counts_ =
        std::make_unique_for_overwrite<uint32_t[]>(table->num_clusters);
}

Recalibrator::~Recalibrator() {
    memory::Dealloc(sample_buf_, config_.sample_size * inference::kVectorDim);
}

void Recalibrator::Tick() noexcept {
    switch (curr_phase_) {
        case Phase::kIdle: {
            if (ingestion_count_ < config_.recalibrate_trigger_count) {
                return;
            }

            // Non-blocking time gate: wait for ongoing search routines to exit
            // active centroid before overwriting with vectors from new episode.
            if (Clock::now() - last_publish_time_ < config_.grace_period) {
                return;
            }

            ingestion_count_ = 0;
            gathered_count_  = 0;
            curr_phase_      = Phase::kGathering;
            return;
        }

        case Phase::kGathering: StepGathering(); return;
        case Phase::kKMeansSeeding: StepKmeansPPSeeding(); return;
        case Phase::kMiniBatch: StepMiniBatch(); return;
    }
}

void Recalibrator::StepGathering() noexcept {
    const uint8_t active = active_route_->load(std::memory_order_acquire);
    IvfTable&     table  = table_[active];

    std::uniform_int_distribution<uint32_t> cluster_dist(
        0, table.num_clusters - 1
    );

    for (uint32_t atmpt = 0; atmpt < config_.mini_batch_size; ++atmpt) {
        if (gathered_count_ >= config_.sample_size) {
            break;
        }

        const auto cluster_id   = cluster_dist(rng_);
        const auto cluster_size = table.ClusterSize(cluster_id);
        if (cluster_size == 0) {
            continue;
        }

        std::uniform_int_distribution<uint32_t> mem_dist(0, cluster_size - 1);

        const uint32_t node_id =
            table.ListMembers(cluster_id)[mem_dist(rng_)].load(
                std::memory_order_acquire
            );
        auto& node = arena_.GetMetaNode(node_id);
        const auto [state, ref, version, length, offset] = node.LoadControl();

        // PENDING is also a valid node state, as K-means only cares about
        // vector geomeotry.
        if (state == memory::NodeState::kDead) {
            continue;
        }

        const float* vec = arena_.GetVector(node_id);
        float* dst = sample_buf_ + gathered_count_ * inference::kVectorDim;
        std::memcpy(dst, vec, inference::kVectorMemsize);

        if (node.LoadVersion(std::memory_order_acquire) != version) {
            // Encountered a torn read. Try again.
            continue;
        }

        ++gathered_count_;
    }

    if (gathered_count_ < config_.sample_size) {
        // Not gathered enough. Resume on next invocation.
        return;
    }

    if (first_recalibration_) {
        // Nothing to warm-start from yet.
        kmeanspp_seeded_ = 0;
        curr_phase_      = Phase::kKMeansSeeding;
    } else {
        const uint8_t inactive = 1 - active;
        for (uint32_t c = 0; c < table_->num_clusters; ++c) {
            table_[inactive].SetCentroid(c, table_[active].GetCentroid(c));
        }

        StartMiniBatchPhase();
    }
}

void Recalibrator::StepKmeansPPSeeding() noexcept {
    const uint8_t inactive = 1 - active_route_->load(std::memory_order_acquire);
    IvfTable&     table    = table_[inactive];

    uint32_t chosen_idx;

    if (kmeanspp_seeded_ == 0) {
        // First centroid: nothing chosen yet to weight against. Pick uniformly
        // at random.
        std::uniform_int_distribution<uint32_t> dist(
            0, config_.sample_size - 1
        );
        chosen_idx = dist(rng_);
    } else {
        double total = 0.0;
        for (uint32_t i = 0; i < config_.sample_size; ++i) {
            total += kmeanspp_min_dist_sq_[i];
        }

        if (total <= 0.0) {
            // Every remaining sample is an exact duplicate of a centroid that
            // is already chosen, or the pool is degenerated.
            // Weighted sampling has nothing left to weight by.
            std::uniform_int_distribution<uint32_t> dist(
                0, config_.sample_size - 1
            );
            chosen_idx = dist(rng_);
        } else {
            std::uniform_real_distribution<double> dist(0.0, total);
            const double                           threshold = dist(rng_);

            double cumulative = 0.0;
            chosen_idx        = config_.sample_size - 1;
            for (uint32_t i = 0; i < config_.sample_size; ++i) {
                cumulative += kmeanspp_min_dist_sq_[i];
                if (cumulative >= threshold) {
                    chosen_idx = i;
                    break;
                }
            }
        }
    }

    const float* chosen_vec = sample_buf_ + chosen_idx * inference::kVectorDim;
    table.SetCentroid(kmeanspp_seeded_, chosen_vec);

    // Update every sample's distance to its nearest chosen centroid so far.
    const bool is_first_centroid = (kmeanspp_seeded_ == 0);
    for (uint32_t i = 0; i < config_.sample_size; i += ann::kBatchSize) {
        float scores[ann::kBatchSize];
        ann::BatchDotProduct(
            chosen_vec, sample_buf_ + inference::kVectorDim * i, scores
        );

        for (uint32_t k = 0; k < ann::kBatchSize; ++k) {
            const float dist_sq = 2.0f * (1.0f - scores[k]);
            if (is_first_centroid || dist_sq < kmeanspp_min_dist_sq_[i + k]) {
                kmeanspp_min_dist_sq_[i + k] = dist_sq;
            }
        }
    }

    ++kmeanspp_seeded_;
    if (kmeanspp_seeded_ >= table_->num_clusters) {
        StartMiniBatchPhase();
    }
}

void Recalibrator::StartMiniBatchPhase() noexcept {
    for (uint32_t c = 0; c < table_->num_clusters; ++c) {
        kmeans_cluster_counts_[c] = 0;
    }

    mini_batch_iterations_done_ = 0;
    curr_phase_                 = Phase::kMiniBatch;
}

void Recalibrator::StepMiniBatch() noexcept {
    const uint8_t inactive = 1 - active_route_->load(std::memory_order_acquire);
    IvfTable&     table    = table_[inactive];

    std::uniform_int_distribution<uint32_t> dist(0, config_.sample_size - 1);
    float                                   total_movement = 0.0f;

    for (uint32_t b = 0; b < config_.mini_batch_size; ++b) {
        const uint32_t sample_idx = dist(rng_);
        const float*   point = sample_buf_ + sample_idx * inference::kVectorDim;

        const uint32_t cluster_id   = table.MatchCluster(point);
        const float*   old_centroid = table.GetCentroid(cluster_id);

        ++kmeans_cluster_counts_[cluster_id];
        const float eta =
            1.0f / static_cast<float>(kmeans_cluster_counts_[cluster_id]);

        float updated[inference::kVectorDim];
        for (size_t d = 0; d < inference::kVectorDim; ++d) {
            updated[d] = (1.0f - eta) * old_centroid[d] + eta * point[d];
        }

        // The blend above does not preserve normalization.
        // Re-normalize the vector.
        constexpr float kFalseMargin = 1e-12f;

        float norm_sq = 0.0f;
        for (size_t d = 0; d < inference::kVectorDim; ++d) {
            norm_sq += updated[d] * updated[d];
        }

        const float norm = std::sqrt(norm_sq);
        if (norm > kFalseMargin) {
            for (size_t d = 0; d < inference::kVectorDim; ++d) {
                updated[d] /= norm;
            }
        }

        // Calculate total movement for convergence check.
        float dot_old_new = 0.0f;
        for (size_t d = 0; d < inference::kVectorDim; ++d) {
            dot_old_new += old_centroid[d] * updated[d];
        }
        total_movement += 2.0f * (1.0f - dot_old_new);

        table.SetCentroid(cluster_id, updated);
    }

    ++mini_batch_iterations_done_;

    if (mini_batch_iterations_done_ >= config_.max_lloyd_iterations ||
        total_movement < config_.min_convergence) {
        PublishRecalibration();
    }
}

void Recalibrator::PublishRecalibration() noexcept {
    const uint8_t next_active =
        1 - active_route_->load(std::memory_order_relaxed);
    active_route_->store(next_active, std::memory_order_release);

    last_publish_time_   = Clock::now();
    first_recalibration_ = false;
    curr_phase_          = Phase::kIdle;
}

}  // namespace strix::collection
