// L0 cache tier node buffer declaration.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

// NodeBuf is a MPSC ring buffer that tracks L0-tier's node ID(s).
// Write workers are producers, while Compaction worker is the sole consumer.
//
// Concurrency model: is lock-free and thread-safe.
class NodeBuf final {
public:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFU;

    explicit NodeBuf(uint32_t capacity);

    NodeBuf(const NodeBuf&)            = delete;
    NodeBuf& operator=(const NodeBuf&) = delete;
    NodeBuf(NodeBuf&&)                 = delete;
    NodeBuf& operator=(NodeBuf&&)      = delete;

    // Attempts to register specified `node_id` onto the buffer.
    // Used by write workers after finish writing vector data into the slot.
    bool TryPush(uint32_t node_id) noexcept;

    // Attempts to remove a node ID from the buffer.
    // Used by background Compaction worker to promote a node to L1-tier.
    uint32_t TryPop() noexcept;

    // Loads content of a slot at specified ring buffer position.
    uint32_t LoadSlot(const uint32_t ring_pos) const noexcept {
        return slots_[ring_pos & (capacity - 1)].load(std::memory_order_acquire
        );
    }

    // Snapshot accessors on `push_head_`.
    // Used by L0-tier search routine to define its rightmost search margin.
    uint32_t SnapPushHead() const noexcept {
        return push_head_.load(std::memory_order_relaxed);
    }

    // Snapshot accessors on `pop_tail_`.
    // Used by L0-tier search routine to define its leftmost search margin.
    uint32_t SnapPopTail() const noexcept {
        return pop_tail_.load(std::memory_order_relaxed);
    }

    const uint32_t capacity;

private:
    alignas(64) std::atomic<uint32_t> push_head_{0};
    alignas(64) std::atomic<uint32_t> pop_tail_{0};

    std::unique_ptr<std::atomic<uint32_t>[]> slots_;
};
