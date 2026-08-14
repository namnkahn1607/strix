// Author: namnkahn1607
//
// Recalibrator implementation: gathering, K-means++ seeding, mini-batch
// Lloyd's iteration, and publish.

#include "recalibration.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>

#include "common/syscall_utils.h"
#include "index/avx2_kernel.h"
#include "level1_ivf.h"
#include "memory/memory_arena.h"

Recalibrator::Recalibrator(
    MemoryArena& arena, RoutingTable* routes,
    std::atomic<uint8_t>* active_route, const IvfConfig& config
)
    : arena_(arena)
    , routes_(routes)
    , active_route_(active_route)
    , config_(config)
    , rng_(std::random_device{}())
    , kmeanspp_min_dist_sq_(
          std::make_unique_for_overwrite<float[]>(config.kmeans_sample_size)
      )
    , kmeans_cluster_counts_(
          std::make_unique_for_overwrite<uint32_t[]>(config.num_clusters)
      ) {
    kmeans_sample_ = static_cast<float*>(common::AllocMMap(
        config.kmeans_sample_size * kVectorMemsize, config.lazy_mapping
    ));
}

Recalibrator::~Recalibrator() {
    common::DeallocMMap(
        kmeans_sample_, config_.kmeans_sample_size * kVectorMemsize
    );
}

void Recalibrator::Tick() noexcept {
    switch (phase_) {
        case Phase::kIdle: {
            if (compaction_performed_ < config_.recalibration_trigger_count) {
                return;
            }

            const uint64_t now = common::MonotonicNow();
            if (now - last_publish_time_ < config_.grace_period_seconds) {
                // Non-blocking time gate: wait long enough (1 sec) for search
                // routines to exit the centroid buffer before overwriting it
                // with new vectors from Recalibration.
                return;
            }

            // Reset the epoch counter.
            compaction_performed_ = 0;

            gathered_count_ = 0;
            phase_          = Phase::kGathering;
            return;
        }

        case Phase::kGathering:
            StepGathering();
            return;

        case Phase::kKMeansSeeding:
            StepKmeansPPSeeding();
            return;

        case Phase::kMiniBatch:
            StepMiniBatch();
            return;
    }
}

void Recalibrator::StepGathering() noexcept {
    const uint8_t active = active_route_->load(std::memory_order_acquire);
    RoutingTable& table  = routes_[active];

    std::uniform_int_distribution<uint32_t> cluster_dist(
        0, table.num_clusters - 1
    );

    for (uint32_t attempt = 0; attempt < config_.mini_batch_size; ++attempt) {
        if (gathered_count_ >= config_.kmeans_sample_size) {
            break;
        }

        const uint32_t cluster_id   = cluster_dist(rng_);
        const uint32_t cluster_size = table.ClusterSize(cluster_id);
        if (cluster_size == 0) {
            continue;
        }

        std::uniform_int_distribution<uint32_t> mem_dist(0, cluster_size - 1);
        const uint32_t                          member_idx = mem_dist(rng_);

        const uint32_t node_id =
            table.ClusterMemberIds(cluster_id)[member_idx].load(
                std::memory_order_acquire
            );

        MetaNode& node = arena_.GetNode(node_id);
        const auto [state, ref_bit, version, length, offset] =
            node.LoadControl();

        if (state == NodeState::kDead) {
            continue;
        }

        // `kPending` is also a valid state. K-means only cares about vector
        // geomeotry, not payload status.
        const float* vec = arena_.GetVector(node_id);
        float*       dst = kmeans_sample_ + gathered_count_ * kVectorDim;
        std::memcpy(dst, vec, kVectorMemsize);

        if (node.LoadVersion(std::memory_order_acquire) != version) {
            // Encountered a torn read. Try again.
            continue;
        }

        ++gathered_count_;
    }

    if (gathered_count_ < config_.kmeans_sample_size) {
        // Not gathered enough. Resume on next call to Tick().
        return;
    }

    if (recalibration_count_ == 0) {
        // First recalibration phase. Nothing to warm-start from yet.
        kmeanspp_seeded_ = 0;
        phase_           = Phase::kKMeansSeeding;
    } else {
        const uint8_t inactive = 1 - active;
        for (uint32_t c = 0; c < config_.num_clusters; ++c) {
            routes_[inactive].SeedCentroid(
                c, routes_[active].CentroidVector(c)
            );
        }

        StartMiniBatchPhase();
    }
}

void Recalibrator::StepKmeansPPSeeding() noexcept {
    const uint8_t inactive = 1 - active_route_->load(std::memory_order_acquire);
    RoutingTable& table    = routes_[inactive];

    uint32_t chosen_idx;

    if (kmeanspp_seeded_ == 0) {
        // First centroid: nothing chosen yet to weight against. Pick uniformly
        // at random.
        std::uniform_int_distribution<uint32_t> dist(
            0, config_.kmeans_sample_size - 1
        );
        chosen_idx = dist(rng_);
    } else {
        double total = 0.0;
        for (uint32_t i = 0; i < config_.kmeans_sample_size; ++i) {
            total += kmeanspp_min_dist_sq_[i];
        }

        if (total <= 0.0) {
            // Every remaining sample is an exact duplicate of a centroid that
            // is already chosen, or the pool is degenerated.
            // Weighted sampling has nothing left to weight by.
            std::uniform_int_distribution<uint32_t> dist(
                0, config_.kmeans_sample_size - 1
            );
            chosen_idx = dist(rng_);
        } else {
            std::uniform_real_distribution<double> dist(0.0, total);
            const double                           threshold = dist(rng_);

            double cumulative = 0.0;
            chosen_idx        = config_.kmeans_sample_size - 1;
            for (uint32_t i = 0; i < config_.kmeans_sample_size; ++i) {
                cumulative += kmeanspp_min_dist_sq_[i];
                if (cumulative >= threshold) {
                    chosen_idx = i;
                    break;
                }
            }
        }
    }

    const float* chosen_vec = kmeans_sample_ + chosen_idx * kVectorDim;
    table.SeedCentroid(kmeanspp_seeded_, chosen_vec);

    // Update every sample's distance to its nearest chosen centroid so far.
    const bool is_first_centroid = (kmeanspp_seeded_ == 0);
    for (uint32_t i = 0; i < config_.kmeans_sample_size; i += kBatchSize) {
        float scores[kBatchSize];
        DotProductContiguousBatch(
            chosen_vec, kmeans_sample_ + kVectorDim * i, scores
        );

        for (uint32_t k = 0; k < kBatchSize; ++k) {
            const float dist_sq = 2.0f * (1.0f - scores[k]);
            if (is_first_centroid || dist_sq < kmeanspp_min_dist_sq_[i + k]) {
                kmeanspp_min_dist_sq_[i + k] = dist_sq;
            }
        }
    }

    ++kmeanspp_seeded_;
    if (kmeanspp_seeded_ >= config_.num_clusters) {
        StartMiniBatchPhase();
    }
}

void Recalibrator::StartMiniBatchPhase() noexcept {
    for (uint32_t c = 0; c < config_.num_clusters; ++c) {
        kmeans_cluster_counts_[c] = 0;
    }

    mini_batch_iterations_done_ = 0;
    phase_                      = Phase::kMiniBatch;
}

void Recalibrator::StepMiniBatch() noexcept {
    const uint8_t inactive = 1 - active_route_->load(std::memory_order_acquire);
    RoutingTable& table    = routes_[inactive];

    std::uniform_int_distribution<uint32_t> dist(
        0, config_.kmeans_sample_size - 1
    );
    float total_movement = 0.0f;

    for (uint32_t b = 0; b < config_.mini_batch_size; ++b) {
        const uint32_t sample_idx = dist(rng_);
        const float*   point      = kmeans_sample_ + sample_idx * kVectorDim;

        const uint32_t cluster_id   = table.MatchCluster(point);
        const float*   old_centroid = table.CentroidVector(cluster_id);

        ++kmeans_cluster_counts_[cluster_id];
        const float eta =
            1.0f / static_cast<float>(kmeans_cluster_counts_[cluster_id]);

        float updated[kVectorDim];
        for (size_t d = 0; d < kVectorDim; ++d) {
            updated[d] = (1.0f - eta) * old_centroid[d] + eta * point[d];
        }

        // The blend above does not preserve normalization.
        // Re-normalize the vector.
        constexpr float kFalseMargin = 1e-12f;

        float norm_sq = 0.0f;
        for (size_t d = 0; d < kVectorDim; ++d) {
            norm_sq += updated[d] * updated[d];
        }

        const float norm = std::sqrt(norm_sq);
        if (norm > kFalseMargin) {
            for (size_t d = 0; d < kVectorDim; ++d) {
                updated[d] /= norm;
            }
        }

        // Calculate total movement for convergence check.
        float dot_old_new = 0.0f;
        for (size_t d = 0; d < kVectorDim; ++d) {
            dot_old_new += old_centroid[d] * updated[d];
        }
        total_movement += 2.0f * (1.0f - dot_old_new);

        table.SeedCentroid(cluster_id, updated);
    }

    ++mini_batch_iterations_done_;

    if (mini_batch_iterations_done_ >= config_.max_lloyd_iterations ||
        total_movement < config_.convergence_epsilon) {
        PublishRecalibration();
    }
}

void Recalibrator::PublishRecalibration() noexcept {
    const uint8_t next_active =
        1 - active_route_->load(std::memory_order_relaxed);
    active_route_->store(next_active, std::memory_order_release);

    last_publish_time_ = common::MonotonicNow();
    ++recalibration_count_;
    phase_ = Phase::kIdle;
}
