// Author: namnkahn1607
//
// SearchOutcome, CacheOutcome, and the VectorIndex class: the Index Layer's
// entry point for vector search, node acquisition, and payload commit/fetch.
// Owns FreeList and L0Buffer; indexes and holds a reference to MemoryArena.

#pragma once

#include <atomic>
#include <memory>
#include <optional>

#include "free_list.h"
#include "level0_ring.h"
#include "level1_ivf.h"
#include "memory_arena.h"
#include "search_inl.h"
#include "recalibration.h"

// `CacheOutcome` represents part of the cache states returned by Data plane to
// the Control plane.
// Used as classification result of `VectorIndex::FetchPayload()`.
enum class CacheOutcome : uint8_t { kMiss, kPendingHit, kHit };

// `VectorIndexBenchAccess` grants benchmark code direct access to the private
// member `L0Buffer` field of `VectorIndex`.
class VectorIndexBenchAccess;

// `VectorIndex` indexes vectors and payloads in a single `MemoryArena`, while
// surfaces the appropriate vector search and payload commit/fetch APIs to the
// gRPC class `CacheServiceImpl`.
//
// Ownership model: construct once, pass by reference to consumers.
// Not copyable, not movable.
class VectorIndex {
public:
    // `VectorIndex` holds a reference to `MemoryArena`, and each instance of
    // type `FreeList` and `L0Buffer`.
    explicit VectorIndex(MemoryArena& arena, uint32_t l0_cap,
                         const IvfConfig& config);
    ~VectorIndex();

    VectorIndex(const VectorIndex&)            = delete;
    VectorIndex& operator=(const VectorIndex&) = delete;
    VectorIndex(VectorIndex&&)                 = delete;
    VectorIndex& operator=(VectorIndex&&)      = delete;

    // `AcquireNode()` is called by search path on a cache miss. Returns nothing
    // if `FreeList` is exhausted, or `L0Buffer` is saturated - in the latter
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

    // `RunCoordinator()` triggers background coordinator worker that schedules
    // and performs Compaction, Recalibration, Reassignment.
    void RunCoordinator(const std::atomic<bool>& shutdown_req);

private:
    friend class VectorIndexBenchAccess;

    MemoryArena& arena_;
    FreeList     free_list_;
    L0Buffer     l0_buffer_;

    // `node_owner_`, a single-source-of-truth cluster ID tracker for every
    // clustered node. Unclustered nodes are considered `kUnclustered`.
    std::unique_ptr<std::atomic<uint32_t>[]> node_owner_;

    IvfConfig            ivf_config_;
    RoutingTable         routes_[2];
    std::atomic<uint8_t> active_route_{0};

    // `RunCompaction()` sequentially migrates nodes from L0 Buffer to L1 IVF.
    void RunCompaction() noexcept;

    // IVF's `Recalibrator` - the calibration state-machine controller.
    Recalibrator recalibrator_;

    // Round-robin cursor only for `RunReassignment()` to determine which
    // cluster to sweep on the next call.
    uint32_t reassignment_cursor_ = 0;

    // `RunReassignment()` sweeps each cluster per called. Remove `kDead` node
    // ID of that cluster and move datapoint to closer cluster.
    // A generation change (node be evicted and re-acquired) mid-process are
    // skipped to determine on another pass.
    void RunReassignment() noexcept;
};

// `VectorIndexBenchAccess` defined out-of-line so ordinary callers never see
// l0_indices_'s type requirements pulled into their translation unit through
// this accessor; only benchmark code that explicitly includes this and
// instantiates it pays for it.
class VectorIndexBenchAccess {
public:
    static L0Buffer& GetL0Buffer(VectorIndex& index) noexcept {
        return index.l0_buffer_;
    }
};
