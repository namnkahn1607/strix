// Memory Arena and its runtime configurator declarations.

#pragma once

#include <atomic>
#include <cassert>
#include <functional>
#include <optional>

#include "arena_config.h"
#include "common/constants.h"
#include "meta_node.h"

// MemoryArenaPrivateAccess grants user code the ability to access and interact
// with internal states/fields of `MemoryArena`.
class MemoryArenaPrivateAccess;

// MemoryArena owns and manages the primary memory regions: metadata array,
// vector array and an optional payload ring buffer.
//
// Allocation is achieved via `mmap` at construction and released via `munmap`
// upon destruction. The ring buffer indexing relies on virtual-offset
// arithmetic with power-of-2 masking.
//
// Concurrency model:
//   - Concurrent pipeline primitive: Multi-threaded design with distinct worker
//     role: writer/reader and garbage collector.
//   - Each individual method defines its own synchronization guarantees.
//   - Thread-safety is NOT uniform across all APIs. Callers must consult the
//     safety contract of each API during invocation.
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

    // ReadPayload copies `length` bytes starting at `v_offset` from the ring
    // buffer into `out_payload`.
    // Asserts non-null `payload_buf`. Caller MUST pre-resize the destination
    // buffer before invocation.
    void ReadPayload(
        uint64_t v_offset, uint32_t length, std::string* out_payload
    ) const noexcept;

    // WritePayload writes a `PayloadHeader` followed by payload of `length`
    // bytes from source `in_payload` into the ring buffer and returns the
    // virtual offset of the written header on success.
    // Asserts non-null `payload_buf`.
    std::optional<uint64_t> WritePayload(
        uint32_t node_id, const uint8_t* in_payload, uint32_t length
    ) noexcept;

    // RunGarbageCollector triggers the GC that sweeps the slot array to find
    // and evict `COLD READY` nodes while expiring stale `PENDING` nodes.
    // Operates until `g_shutdown_req` is set to true. Must be launched on ONE
    // dedicated thread; not re-entrant.
    void RunGarbageCollector(const std::atomic<bool>& g_shutdown_req);

    // SetNodeFreedCallback registers the dependency-inversion hook used by GC
    // to notify `FreeList`.
    // Must be wired exactly ONCE during system initialization, before spawning
    // the GC thread.
    void SetNodeFreedCallback(NodeFreedCallback cb) {
        on_node_freed_ = std::move(cb);
    }

    // GetNode returns a reference to the `MetaNode` at `node_id`.
    // Caller must ensure `node_id` < `max_slots`.
    inline MetaNode& GetNode(const uint32_t node_id) const noexcept {
        return metadata_[static_cast<size_t>(node_id)];
    }

    // GetVector returns a pointer to the vector's first float at `node_id`.
    // Caller must ensure `node_id` < `max_slots`.
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

    // ActualIndex maps a virtual offset to its physical index in the ring
    // buffer. Relies on `payload_buf_size_` being a power of 2.
    size_t ActualIndex(const uint64_t offset) const noexcept {
        return offset & (payload_buf_size - 1);
    }

    // AllocatePayload reserves `length` bytes by advancing `write_head_`.
    // Returns the virtual offset at which the caller may begin writing.
    std::optional<uint64_t> AllocatePayload(uint32_t length) noexcept;

    // SweepStalePending scans the `MetaNode` array and transitions all
    // stale `PENDING` nodes to `DEAD`.
    void SweepStalePending(uint64_t curr_time) noexcept;

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
