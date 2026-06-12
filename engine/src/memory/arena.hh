//
// memory/arena.hh.
//

#pragma once

#include <atomic>
#include <cstdint>

#include "meta_node.hh"

// --- PayloadHeader ---
// 12-byte header prepended to every payload in the ring buffer.
// Enables O(1) reverse-lookup from ring buffer position -> MetaNode.
struct alignas(4) PayloadHeader {
    uint32_t identifier;
    uint32_t node_id;
    uint32_t length;
};

inline constexpr uint64_t PAYLOAD_BUFFER_SIZE =
    4ULL * 1024 * 1024 * 1024;  // 4GB

// Lifespan of a PENDING node in second(s)
inline constexpr uint32_t PENDING_LIFESPAN = 30;

// --- MemoryArena ---
// Owns all memory: vector arena, payload ring buffer, and metadata array.
// Has trigger-once Garbage Collector. Single ownership.
class MemoryArena {
public:
    MemoryArena();
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

    // Vector arena
    MetaNode* metadata;
    float*    vectors;

    // Payload ring buffer
    uint8_t*              buffer_payload;
    std::atomic<uint64_t> write_head;
    std::atomic<uint64_t> read_tail;

    static uint64_t ActualIndex(const uint64_t offset) {
        return offset & (PAYLOAD_BUFFER_SIZE - 1);
    }

    // Allocate helper
    uint64_t AllocatePayload(uint32_t length);

    // Periodic stale Node (overtimed PENDING) sweeper
    void SweepStalePending(uint64_t curr_time) noexcept;
};
