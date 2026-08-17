// Memory Arena declaration.

#pragma once

#include <atomic>
#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "inference/info.h"
#include "memory/arena_config.h"
#include "memory/meta_node.h"
#include "worker/identity.h"

// MemoryArenaPrivateAccess grants user code the ability to access and interact
// with private fields of `MemoryArena`.
class MemoryArenaPrivateAccess;

template <size_t N>
class HazardTable;

// MemoryArena owns and manages the primary memory regions: metadata array,
// vector array and an optional payload ring buffer.
//
// Allocation is achieved via `mmap` at construction and released via `munmap`
// upon destruction. The ring buffer indexing relies on virtual-offset
// arithmetic with power-of-2 masking.
//
// Concurrency model:
//   1. Concurrent pipeline primitive: multi-threaded design with distinct
//      worker role: writer/reader and garbage collector.
//   2. Each individual method defines its own synchronization guarantees.
//   3. Thread-safety is NOT uniform across all APIs. Callers must consult the
//      safety contract of each API during invocation.
//
// Ownership model: construct once, pass by reference to consumers.
class MemoryArena final {
public:
    // NodeFreedCallback defines the signature for cross-layer eviction
    // notification.
    // Pass `node_id` that was just reclaimed by garbage collector.
    using NodeFreedCallback = std::function<void(uint32_t)>;

    explicit MemoryArena(const ArenaConfig& config);
    ~MemoryArena();

    MemoryArena(const MemoryArena&)            = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&)                 = delete;
    MemoryArena& operator=(MemoryArena&&)      = delete;

    // Copies `length` bytes from `offset` in the buffer into `out_payload`.
    // Asserts non-null `payload_buf`. Caller MUST pre-resize the destination
    // buffer before invocation.
    void ReadPayload(uint64_t offset, uint32_t length, std::string* out_payload)
        const noexcept;

    // Writes a `PayloadHeader` followed by `length` bytes from  `in_payload`
    // into the buffer.
    // The virtual offset of the written byte series is returned on success.
    // Asserts non-null `payload_buf`.
    std::optional<uint64_t> WritePayload(
        uint32_t node_id, const uint8_t* in_payload, uint32_t length
    ) noexcept;

    // Triggers the garbage collector that sweeps the node slot array to find
    // and evict COLD READY nodes while expiring stale PENDING nodes.
    // Operates until `shutdown_req` is set to `true`. Must be launched on ONE
    // dedicated thread; not re-entrant.
    void RunGarbageCollector(const std::atomic<bool>& shutdown_req);

    // Registers the dependency-inversion hook used by GC to notify the Freelist
    // that manages node ID.
    // Wired exactly ONCE during system init, before spawning the GC thread.
    void SetNodeFreedCallback(NodeFreedCallback cb) {
        on_node_freed_ = std::move(cb);
    }

    // Returns a reference to the `MetaNode` at `node_id`.
    // Caller must ensure `node_id < max_slots`.
    inline MetaNode& GetNode(const uint32_t node_id) const noexcept {
        return metadata_[static_cast<size_t>(node_id)];
    }

    // Returns a pointer to the vector's first float at `node_id`.
    // Caller must ensure `node_id < max_slots`.
    inline float* GetVector(const uint32_t node_id) const noexcept {
        return vectors_ + static_cast<size_t>(node_id) * kVectorDim;
    }

    inline uint64_t GetWriteHead() const noexcept {
        return write_head_.load(std::memory_order_acquire);
    }

    inline uint64_t GetReadTail() const noexcept {
        return read_tail_.load(std::memory_order_acquire);
    }

    const uint32_t max_slots;
    const size_t   payload_buf_size;

private:
    friend class MemoryArenaPrivateAccess;

    // Maps a virtual offset to its physical index in the ring buffer.
    size_t ActualIndex(const uint64_t offset) const noexcept {
        return offset & (payload_buf_size - 1);
    }

    // Reserves `length` bytes by advancing `write_head_`.
    // Returns the virtual offset at which the caller may begin writing.
    std::optional<uint64_t> AllocatePayload(uint32_t length) noexcept;

    // Scans the node slot array and expires all stale PENDING nodes to DEAD.
    void SweepStalePending(uint64_t curr_time) noexcept;

    // Attempts to reclaim space occuppied by payload of `node_id` using
    // `read_tail_` advancing.
    // If `release_node == true`, `node_id` will be released back to Freelist.
    void TryReclaimSpace(
        uint32_t node_id, uint64_t tail, uint32_t total_len, bool release_node
    ) noexcept;

    // Metadata array: one `MetaNode` per slot.
    MetaNode* metadata_;

    // Vector array: `kVectorDim` floats per slot.
    float* vectors_;

    // Payload ring buffer (`nullptr` when omitted).
    uint8_t* payload_buf_;

    std::atomic<uint64_t> write_head_;
    std::atomic<uint64_t> read_tail_;

    // Callback hook invoked by the GC when a node is evicted.
    NodeFreedCallback on_node_freed_;

    // Managing table of published hazard zones.
    std::unique_ptr<HazardTable<kNumRPCWorkers>> hazard_table_;
};

class MemoryArenaPrivateAccess final {
public:
    // PrefaultBuffer writes 2 bytes into the payload ring buffer: at index `0`
    // and at `write_head`, forcing the kernel to fault all pages immediately.
    //
    // Only call this when `MemoryArena` was constructed with `MAP_POPULATE`
    // disabled and non-null payload buffer.
    // Calling it inside concurrently writers or on an already-populated memory
    // wastes time and may introduce cache pressure.
    static void PrefaultBuffer(MemoryArena& arena) noexcept {
        assert(
            arena.payload_buf_ != nullptr &&
            "PrefaultBuffer requires a payload buffer"
        );

        const auto write_index          = arena.ActualIndex(arena.write_head_);
        arena.payload_buf_[write_index] = 0;
        arena.payload_buf_[0]           = 0;
    }
};
