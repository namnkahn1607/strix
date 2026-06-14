//
// memory/arena.hh.
//

#pragma once

#include <atomic>
#include <cstdint>

#include "meta_node.hh"

// PayloadHeader.
// 12-byte header prepended to every payload in the ring buffer.
// Enables O(1) reverse-lookup from ring buffer position -> MetaNode.
struct alignas(4) PayloadHeader {
    uint32_t identifier;
    uint32_t node_id;
    uint32_t length;
};

// ArenaConfig.
// Controls memory layout of MemoryArena at construction time.
//
// 1. max_slots        : total number of slots to allocate - non-zero & a
//                       multiple of 4.
// 2. payload_buf_size : size of the ring buffer in bytes - a power of 2.
// 3. lazy_mapping     : whether to use MAP_POPULATE in mmap or not. Set
//                       false to use it.
//
// Pass 0 to omit the payload ring buffer (also made it nullptr).
// Non-null assertions will be made on relevant methods.
struct ArenaConfig {
    size_t   max_slots;
    size_t   payload_buf_size;
    bool     lazy_mapping;
    uint64_t start_point = 0;

    // Production-grade config:
    // 524'288 + 1'024 slots, 4GB payload buffer, populated.
    static ArenaConfig Production();

    // SearchL0 benchmark config:
    // 1'024 slots, no payload buffer, populated.
    static ArenaConfig BenchSearchL0();

    // SearchL0 testing config:
    // 1'024 slots, no payload buffer, lazy.
    static ArenaConfig TestSearchL0();
};

// Lifespan of a PENDING node in second(s)
inline constexpr uint32_t PENDING_LIFESPAN = 30;

// MemoryArena.
// Owns all memory: vector arena, payload ring buffer, and metadata array.
// Has trigger-once Garbage Collector. Single ownership.
class MemoryArena {
public:
    explicit MemoryArena(const ArenaConfig& config);
    ~MemoryArena();

    // No Copy/Move semantics
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    void ReadPayload(uint64_t v_offset, uint32_t length,
                     std::string* out_payload) const;

    uint64_t WritePayload(uint32_t node_id, const uint8_t* in_payload,
                          uint32_t length);

    // Snowplow garbage collector. Runs on a dedicated background thread.
    void RunGarbageCollector(const std::atomic<bool>& g_shutdown_req);

    inline MetaNode& GetNode(const size_t node_id) const noexcept {
        return metadata[node_id];
    }

    inline float* GetVector(const size_t node_id) const noexcept {
        return vectors + VECTOR_DIM_ARENA * node_id;
    }

    inline uint64_t GetWriteHead() const noexcept {
        return write_head.load(std::memory_order_acquire);
    }

    inline uint64_t GetReadTail() const noexcept {
        return read_tail.load(std::memory_order_acquire);
    }

private:
    constexpr static size_t VECTOR_DIM_ARENA = 384;

    // Runtime dimensions
    size_t max_slots;
    size_t payload_buf_size;

    // Vector arena
    MetaNode* metadata;
    float*    vectors;

    // Payload ring buffer (nullptr when payload_buf_size == 0)
    uint8_t*              payload_buf;
    std::atomic<uint64_t> write_head;
    std::atomic<uint64_t> read_tail;

    // Actual index of specified offset in ring buffer
    uint64_t ActualIndex(const uint64_t offset) const {
        return offset & (payload_buf_size - 1);
    }

    // Allocate helper
    uint64_t AllocatePayload(uint32_t length);

    // Periodic stale Node (overtimed PENDING) sweeper
    void SweepStalePending(uint64_t curr_time) noexcept;
};
