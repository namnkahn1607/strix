// Author: namnkahn1607
//
// ArenaConfig, MemoryArena and PayloadHeader declarations.
// MemoryArena owns the MetaNode metadata array, float vector arena, and
// the payload ring buffer - all mmap-allocated at construction time.

#pragma once

#include <atomic>
#include <functional>
#include <optional>

#include "constants.h"
#include "meta_node.h"

// `ArenaConfig` describes the memory layout of `MemoryArena` at construction
// time. All validation is performed inside the constructor.
//
// Fields:
//   1. `max_slots`        : Total number of node + vector slots to allocate.
//                           Must be non-zero and multiple of `kBatchSize` (4).
//   2. `payload_buf_size` : Ring buffer size in bytes. Must be a power of 2.
//                           Pass 0 to omit the payload buffer; payload methods
//                           will assert-fail).
//   3. `lazy_mapping`     : When `false`, `mmap` uses `MAP_POPULATE` to
//                           pre-fault all pages at construction, eliminating
//                           page-fault latency during operation.
//                           When `true`, pages are faulted on first access
//                           (lower startup cost, higher first-touch latency).
//   4. `start_point`      : Initial value of the ring buffer write head;
//                           defaults to 0. Non-zero values are used in testing
//                           to exercise wrap-around behaviour.
struct ArenaConfig {
    const uint32_t max_slots;
    const uint64_t payload_buf_size;
    const bool     lazy_mapping;
    const uint64_t start_point = 0;

    // `Production()` config: `kTotalMaxSlots` slots, 4 GB payload buffer and
    // pre-fault pages enabled.
    static ArenaConfig Production() {
        // Default ring buffer capacity for `Production()` config.
        constexpr uint64_t kPayloadBufferSize = 0x100000000ULL;  // 4 GB
        return {kTotalSlots, kPayloadBufferSize, false};
    }

    // `Compact()` config: `slots` node/vector capacity, no payload buffer and
    // pre-fault pages enabled. For throughput benchmarks that need a smaller
    // arena but still pre-faulted. `MemoryArena` has no notion of L0/L1.
    static ArenaConfig Compact(uint32_t slots) {
        return {slots, 0, false};
    }

    // `CompactLazy()` config: same as `Compact()`, but lazily mapped. For unit
    // tests where pre-faulting only adds startup latency.
    static ArenaConfig CompactLazy(uint32_t slots) {
        return {slots, 0, true};
    }
};

// `MemoryArena` owns all memory: the metadata array, the vector array, and the
// payload ring buffer.
//
// Memory is allocated via `mmap` at construction and released at destruction.
// The ring buffer uses virtual-offset arithmetic (power-of-2 masking) for
// lock-free head/tail management. The payload buffer is optional; if omitted,
// all payload methods assert-fail.
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

    const uint32_t max_slots;
    const uint64_t payload_buf_size;

    // `ReadPayload()`: copies `length` bytes starting at `v_offset` from the
    // ring buffer into `out_payload`. Caller must resize the destination buffer
    // itself. Asserts that `payload_buf` is non-null.
    void ReadPayload(uint64_t v_offset, uint32_t length,
                     std::string* out_payload) const noexcept;

    // `WritePayload()`: writes a `PayloadHeader` followed by `in_payload` of
    // given `length` bytes into the ring buffer. Returns the virtual offset of
    // the written header. Asserts that `payload_buf` is non-null.
    std::optional<uint64_t> WritePayload(uint32_t       node_id,
                                         const uint8_t* in_payload,
                                         uint32_t       length) noexcept;

    // `RunGarbageCollector()` triggers Snowplow GC: a single background thread
    // that sweeps the slot array, evicting COLD READY nodes and expiring stale
    // PENDING nodes. Runs until `g_shutdown_req` is set to true.
    // Must be launched on exactly one dedicated thread; not re-entrant.
    void RunGarbageCollector(const std::atomic<bool>& g_shutdown_req);

    // `NodeFreedCallback`, a notifier to the Index Layer when a node_id is
    // released back to the FreeList.
    using NodeFreedCallback = std::function<void(uint32_t)>;

    // `SetNodeFreedCallback()` is used by GC to indirectly append `node_id`
    // back to `FreeList`. The `main()` wires this up to
    // `VectorIndex::ReleaseNode`.
    void SetNodeFreedCallback(NodeFreedCallback cb) {
        on_node_freed_ = std::move(cb);
    }

    // `GetNode()` returns a reference to the `MetaNode` at `node_id`.
    // Caller must ensure `node_id` < `max_slots`.
    inline MetaNode& GetNode(const uint32_t node_id) const noexcept {
        return metadata_[node_id];
    }

    // `GetVector()` returns a pointer to the first float of the vector at
    // position `node_id`. The vector occupies `kVectorDim` contiguous floats.
    inline float* GetVector(const uint32_t node_id) const noexcept {
        return vectors_ + kVectorDim * node_id;
    }

    // `GetWriteHead()` returns the current write head offset.
    inline uint64_t GetWriteHead() const noexcept {
        return write_head_.load(std::memory_order_acquire);
    }

    // `GetReadTail()` returns the current read tail offset.
    inline uint64_t GetReadTail() const noexcept {
        return read_tail_.load(std::memory_order_acquire);
    }

    // `PrefaultBuffer()` sequentially writes one byte per page across the
    // payload ring buffer, forcing the kernel to fault all pages immediately.
    //
    // WARNING: only call this when the arena was constructed with lazy mapping
    // enabled and the payload buffer is non-null. Calling it concurrently with
    // writers or on an already-populated arena wastes time and may introduce
    // cache pressure.
    void PrefaultBuffer() const noexcept;

private:
    // Metadata array: one MetaNode per slot, mmap-allocated.
    MetaNode* metadata_;

    // Vector arena: kVectorDim floats per slot, mmap-allocated.
    // Row-major layout: slot i starts at vectors_ + kVectorDim * i.
    float* vectors_;

    // Payload ring buffer (nullptr when payload_buf_size_ == 0).
    uint8_t* payload_buf_;

    std::atomic<uint64_t> write_head_;
    std::atomic<uint64_t> read_tail_;

    // Dependency-inversion callback to Index Layer.
    NodeFreedCallback on_node_freed_;

    // `ActualIndex()` maps a virtual offset to its physical index in the ring
    // buffer. Relies on `payload_buf_size_` being a power of 2.
    uint64_t ActualIndex(const uint64_t offset) const noexcept {
        return offset & (payload_buf_size - 1);
    }

    // `AllocatePayload()` reserves `length` bytes by advancing `write_head_`.
    // Returns the virtual offset at which the caller may begin writing.
    std::optional<uint64_t> AllocatePayload(uint32_t length) noexcept;

    // `SweepStalePending()` scans the metadata array and transitions stale
    // PENDING nodes to DEAD.
    void SweepStalePending(uint64_t curr_time) noexcept;
};

// `PayloadHeader`, a 12-byte header prepended to every payload written into the
// ring buffer. Enables constant reverse-lookup from a ring buffer position to
// its `MetaNode` without re-scanning the entire array.
//
// Fields:
//   1. `identifier` : Caller-supplied tag; used to verify header integrity.
//   2. `node_id`    : Index of the MetaNode that owns this payload.
//   3. `length`     : Payload byte length, excluding this header.
struct alignas(4) PayloadHeader {
    const uint32_t identifier;
    const uint32_t node_id;
    const uint32_t length;
};
