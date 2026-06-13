//
// tests/searchl0_test.cc
//
// Unit tests for SearchL0() - the L0 buffer vector searching routine.
//
// SearchL0 has 3 responsibilities:
//   1. Score all VALID nodes via AVX2 Dot Product and return the best hit.
//   2. Track the first DEAD slot as a reusable candidate.
//   3. Correctly classify each NodeState as valid, skippable, or reusable.
//
// Timestamp convention: tests pass a fixed curr_time = 1000.
// PENDING nodes with created_at = 950 are fresh (age 50 < PENDING_LIFESPAN=30
// would be stale, so tests that want fresh nodes use age < 30).
//

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "arena.hh"
#include "constants.hh"
#include "meta_node.hh"
#include "search.hh"

namespace {

inline constexpr float    APPROX4 = 1e-4f;
inline constexpr uint64_t NOW = 1000;  // fixed "current time" for all tests
inline constexpr uint64_t FRESH_TS = NOW - 10;  // age 10s < lifespan (30s)
inline constexpr uint64_t STALE_TS = NOW - 60;  // age 60s > lifespan (30s)

// Allocates a 32-byte aligned float array of size 384, filled with a unit
// vector dimension 'hot_dim'. Ownership: Caller must std::free().
float* GenUnitVector(const size_t hot_dim) {
    auto* vec =
        static_cast<float*>(std::aligned_alloc(32, VECTOR_DIM * sizeof(float)));
    std::memset(vec, 0, VECTOR_DIM * sizeof(float));
    vec[hot_dim] = 1.0f;

    return vec;
}

// Allocates a normalized random-ish 32-byte aligned float array of size 384.
// Again, ownership: Caller must std::free().
float* GenRandomVector(const uint64_t seed) {
    auto* vec =
        static_cast<float*>(std::aligned_alloc(32, VECTOR_DIM * sizeof(float)));

    float norm = 0.0f;
    for (size_t i = 0; i < VECTOR_DIM; ++i) {
        vec[i] =
            static_cast<float>((seed * 1103515245 + i * 12345) & 0x7FFFFFFF) /
                static_cast<float>(0x7FFFFFFF) * 2.0f -
            1.0f;
        norm += vec[i] * vec[i];
    }

    norm = std::sqrt(norm);
    for (size_t i = 0; i < VECTOR_DIM; ++i) {
        vec[i] /= norm;
    }

    return vec;
}

// Set a specified Node READT with given vector.
void SetReady(MemoryArena& arena, const size_t node_id, const float* vec) {
    std::memcpy(arena.GetVector(node_id), vec, VECTOR_DIM * sizeof(float));
    const uint64_t ctrl = PackControl(NodeState::READY, EvictState::HOT, 0, 0);
    arena.GetNode(node_id).control_block.store(ctrl, std::memory_order_relaxed);
}

// Set a specified Node PENDING with given vector and timestamp.
void SetPending(MemoryArena& arena, const size_t node_id, const float* vec,
                const uint64_t ts) {
    std::memcpy(arena.GetVector(node_id), vec, VECTOR_DIM * sizeof(float));
    const uint64_t ctrl =
        PackControl(NodeState::PENDING, EvictState::HOT, 0, 0);

    MetaNode& node = arena.GetNode(node_id);
    node.control_block.store(ctrl, std::memory_order_relaxed);
    node.created_at.store(ts, std::memory_order_relaxed);
}

}  // namespace

// ---------------------------------------------------------------------------
// Test Fixture
// ---------------------------------------------------------------------------

class SearchL0Test : public ::testing::Test {
protected:
    MemoryArena arena{ArenaConfig::TestSearchL0()};
};

// ---------------------------------------------------------------------------
// EmptyArena
// All slots DEAD. No best node, reusable_node_id should be 0 (first slot).
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, EmptyArena) {
    float*             query = GenUnitVector(0);
    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);

    EXPECT_EQ(res.best_node_id, -1);
    EXPECT_FLOAT_EQ(res.best_score, -1.0f);
    EXPECT_EQ(res.reusable_node_id, 0);
}

// ---------------------------------------------------------------------------
// SingleReadyNode
// One READY node whose vector is identical to the query.
// best_score must be ~1.0, best_node_id must be that node.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, SingleReadyNode) {
    float* query = GenUnitVector(5);
    SetReady(arena, 0, query);

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);

    EXPECT_EQ(res.best_node_id, 0);
    EXPECT_NEAR(res.best_score, 1.0f, APPROX4);
}

// ---------------------------------------------------------------------------
// BestNodeIsHighestScore
// Three READY nodes with different similarity to query.
// SearchL0 must return the one with highest dot product.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, BestNodeIsHighestScore) {
    float* query = GenUnitVector(0);
    float* high = GenUnitVector(0);  // dot = 1.0
    float* med = GenUnitVector(1);   // dot = 0.0
    float* low = GenUnitVector(2);   // dot = 0.0

    // Put them in non-trivial positions (not slot 0) to test scan logic.
    SetReady(arena, 4, low);
    SetReady(arena, 8, med);
    SetReady(arena, 12, high);

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);
    std::free(high);
    std::free(med);
    std::free(low);

    EXPECT_EQ(res.best_node_id, 12);
    EXPECT_NEAR(res.best_score, 1.0f, APPROX4);
}

// ---------------------------------------------------------------------------
// DeadSlotsAreReusable
// DEAD slots must be tracked as reusable. First DEAD slot encountered wins.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, DeadSlotsAreReusable) {
    float* query = GenUnitVector(0);
    // Make slot 0 READY, slots 1-3 remain DEAD.
    SetReady(arena, 0, query);

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);

    // Slot 1 is the first DEAD slot.
    EXPECT_EQ(res.reusable_node_id, 1);
}

// ---------------------------------------------------------------------------
// ClaimedSlotsAreSkipped
// CLAIMED nodes must be excluded from scoring AND from reusable tracking.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, ClaimedSlotsAreSkipped) {
    float* query = GenUnitVector(0);

    // Slot 0: CLAIMED (writer is mid-copying, vector is garbage)
    std::memcpy(arena.GetVector(0), query, VECTOR_DIM * sizeof(float));
    arena.GetNode(0).control_block.store(
        PackControl(NodeState::CLAIMED, EvictState::HOT, 0, 0),
        std::memory_order_relaxed);

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);

    // CLAIMED slot must not be best_node_id or reusable_node_id.
    EXPECT_NE(res.best_node_id, 0);
    EXPECT_NE(res.reusable_node_id, 0);
    // Slot 1 (DEAD) should be the first reusable.
    EXPECT_EQ(res.reusable_node_id, 1);
}

// ---------------------------------------------------------------------------
// FreshPendingIsSearchable
// A PENDING node with a recent timestamp must be included in scoring.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, FreshPendingIsSearchable) {
    float* query = GenUnitVector(3);
    SetPending(arena, 0, query, FRESH_TS);

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);

    EXPECT_EQ(res.best_node_id, 0);
    EXPECT_NEAR(res.best_score, 1.0f, APPROX4);
}

// ---------------------------------------------------------------------------
// StalePendingIsSkipped
// A PENDING node whose age > lifespan of 30s must be excluded from scoring
// AND from reusable tracking.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, StalePendingIsSkipped) {
    float* query = GenUnitVector(3);
    SetPending(arena, 0, query, STALE_TS);

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);

    // Stale slot must not be best_node_id or reusable_node_id.
    EXPECT_NE(res.best_node_id, 0);
    EXPECT_NE(res.reusable_node_id, 0);
}

// ---------------------------------------------------------------------------
// PendingWithZeroTimestampIsSkipped
// A PENDING node with created_at == 0 means CLAIMED -> PENDING is not yet
// complete. Must be skipped entirely.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, PendingWithZeroTimestampIsSkipped) {
    float* query = GenUnitVector(7);
    SetPending(arena, 0, query, 0);

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);

    EXPECT_NE(res.best_node_id, 0);
}

// ---------------------------------------------------------------------------
// MigratingIsSearchable
// A MIGRATING node must be treated identically to READY for read operation.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, MigratingIsSearchable) {
    float* query = GenUnitVector(10);

    std::memcpy(arena.GetVector(0), query, VECTOR_DIM * sizeof(float));
    arena.GetNode(0).control_block.store(
        PackControl(NodeState::MIGRATING, EvictState::HOT, 0, 0),
        std::memory_order_relaxed);

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);

    EXPECT_EQ(res.best_node_id, 0);
    EXPECT_NEAR(res.best_score, 1.0f, APPROX4);
}

// ---------------------------------------------------------------------------
// NoReusableWhenAllSlotsOccupied
// Fill all L0 slots with READY nodes. reusable_node_id must be -1.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, NoReusableWhenAllSlotsOccupied) {
    float* query = GenUnitVector(0);

    for (size_t i = 0; i < L0_MAX_SLOTS; ++i) {
        SetReady(arena, i, query);
    }

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);

    EXPECT_EQ(res.reusable_node_id, -1);
}

// ---------------------------------------------------------------------------
// SimilarityThresholdReflected
// best_score must accurately reflect the dot product value, not be clamped
// or distorted. Caller uses this value to decide HIT/MISS.
// ---------------------------------------------------------------------------
TEST_F(SearchL0Test, SimilarityScoreIsAccurate) {
    float* query = GenRandomVector(42);
    float* node_vec = GenRandomVector(99);
    SetReady(arena, 0, node_vec);

    // Compute expected score manually
    float expected = 0.0f;
    for (size_t i = 0; i < VECTOR_DIM; ++i) {
        expected += query[i] * node_vec[i];
    }

    const SearchResult res = SearchL0(arena, query, NOW);
    std::free(query);
    std::free(node_vec);

    EXPECT_NEAR(res.best_score, expected, APPROX4);
}
