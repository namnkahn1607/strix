//
// Created by nlnk on Apr 16, 26.
//

#ifndef STRIX_ENGINE_ARENA_HH
#define STRIX_ENGINE_ARENA_HH

#include <atomic>

#include "constant.hh"
#include "node.hh"

// 12-byte Payload Header supporting O(1) lookup to Vector Arena.
struct alignas(4) PayloadHeader {
    uint32_t identifier;
    uint32_t node_id;
    uint32_t length;
};

class MemoryArena {
public:
    MemoryArena();
    ~MemoryArena();

    // No Copy Constructor & Copy Assignment Operator
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    // The Watermark Snowplow
    void RunGarbageCollector(const std::atomic<bool>& g_shutdown_request);

    // Getters
    [[nodiscard]] MetaNode& GetNode(const size_t node_id) const noexcept {
        return metadata[node_id];
    }

    [[nodiscard]] float* GetVector(const size_t node_id) const noexcept {
        return vectors + (engine::VECTOR_DIM * node_id);
    };

    [[nodiscard]] uint64_t GetWriteHead() const noexcept {
        return write_head.load(std::memory_order_acquire);
    };

    [[nodiscard]] uint64_t GetReadTail() const noexcept {
        return read_tail.load(std::memory_order_acquire);
    }

    void ReadPayload(uint64_t v_offset, uint32_t length,
                     std::string* out_payload) const;

    uint64_t WritePayload(uint32_t node_id, const uint8_t* in_payload,
                          uint32_t length);

private:
    // Vector Arena
    MetaNode* metadata;
    float* vectors;

    // Payload Arena
    uint8_t* buffer_payload;
    std::atomic<uint64_t> write_head;
    std::atomic<uint64_t> read_tail;

    uint64_t AllocatePayload(uint32_t length);
};

#endif  // STRIX_ENGINE_ARENA_HH
