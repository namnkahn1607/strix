// Author: namnkahn1607
//
// PayloadHeader, ArenaConfig, and MemoryArena.
// MemoryArena owns the MetaNode metadata array, float vector arena,
// and payload ring buffer - all mmap-allocated at construction time.

#pragma once

#include <functional>

#include "constants.h"
#include "meta_node.h"

// PayloadHeader
//
// 12-byte header prepended to every payload written into the ring buffer.
// Enables constant reverse-lookup from a ring buffer position to its
// owning `MetaNode` without scanning the entire array.
//
// Fields:
//   1. `identifier` : caller-supplied tag; used to verify header integrity.
//   2. `node_id`    : index of the MetaNode that owns this payload.
//   3. `length`     : payload byte length, excluding this header.
struct alignas(4) PayloadHeader {
    uint32_t identifier;
    uint32_t node_id;
    uint32_t length;
};

// ArenaConfig
//
// Immutable parameters that govern memory layout of `MemoryArena` at
// construction time. All validation is performed inside the constructor.
//
// Fields:
//   1. `max_slots`        : total number of node + vector slots to allocate.
//                           Must be non-zero and multiple of `kBatchSize` (4).
//   2. `payload_buf_size` : ring buffer size in bytes. Must be a power of 2.
//                           Pass 0 to omit the payload buffer; payload methods
//                           will assert-fail).
//   3. `lazy_mapping`     : when `false`, `mmap` uses `MAP_POPULATE` to
//                           pre-fault all pages at construction, eliminating
//                           page-fault latency during operation.
//                           When `true`, pages are faulted on first access
//                           (lower startup cost, higher first-touch latency).
//   4. `start_point`      : initial value of the ring buffer write head;
//                           defaults to 0. Non-zero values are used in testing
//                           to exercise wrap-around behaviour.
struct ArenaConfig {
    size_t   max_slots;
    uint64_t payload_buf_size;
    bool     lazy_mapping;
    uint64_t start_point = 0;

    // Production(): `kTotalMaxSlots` slots, 4GB payload buffer, `MAP_POPULATE`.
    static ArenaConfig Production();

    // BenchSearchL0(): `kL0MaxSlots` slots, no payload buffer, `MAP_POPULATE`.
    // Use for isolated `SearchL0` throughput benchmarks.
    static ArenaConfig BenchSearchL0();

    // TestSearchL0(): `kL0MaxSlots` slots, no payload buffer, lazy mapping.
    // Use for unit tests where pre-faulting adds unnecessary latency.
    static ArenaConfig TestSearchL0();
};

// Maximum lifetime of a PENDING node in seconds.
// Nodes that remain PENDING beyond this deadline are treated as stale by
// the GC sweeper and by SearchL0 (skipped during scan).
inline constexpr uint32_t PENDING_LIFESPAN = 30;

// MemoryArena
//
// Owns all data-plane memory: the metadata array, the float vector arena,
// and the payload ring buffer.
//
// Memory is allocated via `mmap` at construction and released at destruction.
// The ring buffer uses virtual-offset arithmetic (power-of-2 masking) for
// lock-free head/tail management.
//
// Ownership model: construct once, pass by reference to consumers.
// Not copyable, not movable (owns raw `mmap` pointers and atomic members).
class MemoryArena {
public:
    explicit MemoryArena(const ArenaConfig& config);
    ~MemoryArena();

    MemoryArena(const MemoryArena&)            = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&)                 = delete;
    MemoryArena& operator=(MemoryArena&&)      = delete;

    // ReadPayload(): copies `length` bytes starting at `v_offset` from the
    // ring buffer into `out_payload`. Asserts that `payload_buf` is non-null.
    void ReadPayload(uint64_t v_offset, uint32_t length,
                     std::string* out_payload) const;

    // WritePayload(): writes a `PayloadHeader` followed by `in_payload` of
    // given `length` bytes into the ring buffer. Returns the virtual offset of
    // the written header. Asserts that `payload_buf` is non-null.
    uint64_t WritePayload(uint32_t node_id, const uint8_t* in_payload,
                          uint32_t length);

    // RunGarbageCollector()
    //
    // Snowplow GC: a single background thread that sweeps the slot array,
    // evicting COLD READY nodes and expiring stale PENDING nodes.
    // Runs until `g_shutdown_req` is set to true.
    // Must be launched on exactly one dedicated thread; not re-entrant.
    void RunGarbageCollector(const std::atomic<bool>& g_shutdown_req);

    // NodeFreedCallback
    //
    // `MemoryArena` (Storage Layer) must notify `VectorIndex` (Index Layer)
    // when a `node_id` becomes free again, but that would create a circular
    // dependency since Index Layer 'oversees' Storage Layer.
    using NodeFreedCallback = std::function<void(uint32_t)>;

    // SetNodeFreedCallback()
    //
    // GC use this to indirectly append `node_id` to `FreeList`.
    // The caller `main` wires this up to `VectorIndex::PushFreeNode`.
    void SetNodeFreedCallback(NodeFreedCallback cb) {
        on_node_freed_ = std::move(cb);
    }

    // GetNode(): returns a reference to the `MetaNode` at `node_id`.
    // Caller must ensure `node_id` < `max_slots`.
    inline MetaNode& GetNode(const size_t node_id) const noexcept {
        return metadata_[node_id];
    }

    // GetVector(): returns a pointer to the first float of the vector at
    // position `node_id`. The vector occupies `kVectorDim` contiguous floats.
    inline float* GetVector(const size_t node_id) const noexcept {
        return vectors_ + kVectorDim * node_id;
    }

    inline uint64_t GetWriteHead() const noexcept {
        return write_head_.load(std::memory_order_acquire);
    }

    inline uint64_t GetReadTail() const noexcept {
        return read_tail_.load(std::memory_order_acquire);
    }

    // PrefaultBuffer()
    //
    // Sequentially writes one byte per page across the payload ring buffer,
    // forcing the kernel to fault in all pages immediately.
    //
    // WARNING: only call this when the arena was constructed with
    // lazy_mapping = true and the payload buffer is non-null. Calling it
    // concurrently with writers or on an already-populated arena wastes time
    // and may introduce cache pressure.
    void PrefaultBuffer() const noexcept;

private:
    size_t   max_slots_;
    uint64_t payload_buf_size_;

    // Metadata array: one MetaNode per slot, mmap-allocated.
    MetaNode* metadata_;

    // Vector arena: kVectorDim floats per slot, mmap-allocated.
    // Row-major layout: slot i starts at vectors_ + kVectorDim * i.
    float* vectors_;

    // Payload ring buffer (nullptr when payload_buf_size_ == 0).
    uint8_t*              payload_buf_;
    std::atomic<uint64_t> write_head_;
    std::atomic<uint64_t> read_tail_;

    // Dependency-inversion callback to Index Layer.
    NodeFreedCallback on_node_freed_;

    // Maps a virtual offset to its physical index in the ring buffer.
    // Relies on `payload_buf_size_` being a power of 2.
    uint64_t ActualIndex(const uint64_t offset) const {
        return offset & (payload_buf_size_ - 1);
    }

    // Reserves `length` bytes in the ring buffer by advancing `write_head_`.
    // Returns the virtual offset at which the caller may begin writing.
    uint64_t AllocatePayload(uint32_t length);

    // Scans the metadata array and transitions stale PENDING nodes to DEAD.
    // A stale PENDING node has `created_at + kPendingLifespan <= curr_time`.
    void SweepStalePending(uint64_t curr_time) noexcept;
};
