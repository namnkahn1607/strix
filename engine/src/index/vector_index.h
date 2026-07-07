// Author: namnkahn1607
//
// SearchOutcome, CacheOutcome, and the VectorIndex class: the Index Layer's
// entry point for vector search, node acquisition, and payload commit/fetch.
// Owns FreeList and L0Indices; indexes and holds a reference to MemoryArena.

#pragma once

#include <atomic>
#include <memory>
#include <optional>

#include "arena.h"
#include "free_list.h"
#include "level0_ring.h"

// `CacheOutcome` stimulates part of the cache states returned by Data plane to
// the Control plane.
// Used as classification result of `VectorIndex::FetchPayload()`.
enum class CacheOutcome : uint8_t { kMiss, kPendingHit, kHit };

// `SearchOutcome` describes result of a vector search against either tier.
// Only record information of a node whose similarity score exceeds the
// pre-defined `kSimiarityThreshold`.
struct SearchOutcome {
    uint32_t node_id;
    uint8_t  version;
};

// `SearchResult` tracks not just the champion but also the runner-up.
// The runner-up entry is optional.
struct SearchResult {
    SearchOutcome                primary;
    std::optional<SearchOutcome> secondary;
};

// `VectorIndexBenchAccess` grants benchmark code direct access to the private
// member `L0Indices` field of `VectorIndex`.
class VectorIndexBenchAccess;

// `VectorIndex` indexes vectors and payloads in a single `MemoryArena`, while
// surfaces the appropriate vector search and payload commit/fetch APIs to the
// gRPC class `CacheServiceImpl`.
//
// Ownership model: construct once, pass by reference to consumers.
// Not copyable, not movable.
class VectorIndex {
public:
    // `kUnclustered` is a `node_owner_` state which stands for "not yet
    // assigned to any cluster". Hence, every L0 node is unclustered.
    static constexpr uint16_t kUnclustered = 0xFFFF;

    // `VectorIndex` holds a reference to `MemoryArena`, and each instance of
    // type `FreeList` and `L0Indices`.
    explicit VectorIndex(MemoryArena& arena, const size_t l0_cap);

    VectorIndex(const VectorIndex&)            = delete;
    VectorIndex& operator=(const VectorIndex&) = delete;
    VectorIndex(VectorIndex&&)                 = delete;
    VectorIndex& operator=(VectorIndex&&)      = delete;

    // `AcquireNode()` is called by search path on a cache miss. Returns nothing
    // if `FreeList` is exhausted, or `L0Indices` is saturated - in the latter
    // case the node is rolled back to DEAD and returned to `FreeList`.
    std::optional<uint32_t> AcquireNode(const float* query,
                                        uint64_t     now) noexcept;

    // `ReleaseNode()` is wired up as `NodeFreedCallback` of `MemoryArena` so
    // GC can return an evicted `node_id` to `FreeList`.
    void ReleaseNode(uint32_t node_id) noexcept;

    // `SearchL0()` searches for the most similarity vector within L0 buffer.
    // A candidate whose version changes between being read and being scored
    // is considered 'torn read' and is discarded outright.
    // Similarity score must exceed `kSimilarityThreshold`, otherwise return
    // nothing.
    std::optional<SearchResult> SearchL0(const float* query) const noexcept;

    // `SearchL1()` searches for the most similarity vector within L1 buffer.
    // Similarity score must exceed `kSimilarityThreshold`, otherwise nothing
    // is returned.
    std::optional<SearchResult> SearchL1(const float* query) const noexcept;

    // `FetchPayload()` classifies the result `node_id` from search routines and
    // perform payload extracting.
    //   - `kHit`        : hit a `kReady` Node, `*out` holds the payload.
    //   - `kMiss`       : version mismatched (the Node is evicted and reused),
    //                     the Node is already `kDead`, a stale PENDING node, or
    //                     failed to allocate buffer for `*out`.
    //   - `kPendingHit` : hit a fresh `kPending` Node, `*out` is left empty.
    CacheOutcome FetchPayload(uint32_t node_id, uint8_t expected_version,
                              uint64_t curr_time, std::string* out) const;

    // `CommitPayload()` is called to write payload of a `kPending` Node that is
    // previously allocated by `AcquireNode()`.
    // Best effort, no retry: return false if the target Node is no longer
    // in state `kPending`, or the payload buffer of `MemoryArena` is saturated.
    bool CommitPayload(uint32_t node_id, const uint8_t* in,
                       uint32_t length) noexcept;

private:
    friend class VectorIndexBenchAccess;

    MemoryArena& arena_;
    FreeList     free_list_;
    L0Indices    l0_indices_;

    // `node_owner_`, a single-source-of-truth cluster ID tracker for every
    // clustered node. Unclustered nodes are considered `kUnclustered`.
    std::unique_ptr<std::atomic<uint16_t>[]> node_owner_;
};

// `VectorIndexBenchAccess` defined out-of-line so ordinary callers never see
// l0_indices_'s type requirements pulled into their translation unit through
// this accessor; only benchmark code that explicitly includes this and
// instantiates it pays for it.
class VectorIndexBenchAccess {
public:
    static L0Indices& GetL0Indices(VectorIndex& index) noexcept {
        return index.l0_indices_;
    }
};
