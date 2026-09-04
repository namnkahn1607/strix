// L0-tier node buffer.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

namespace strix::collection {

// Multi-producer single-consumer ring buffer that tracks L0-tier node ID(s).
// Acquiring workers are producers, while Compaction is the sole consumer.
class NodeBuf {
public:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;

    explicit NodeBuf(uint32_t capacity);

    NodeBuf(const NodeBuf&)            = delete;
    NodeBuf& operator=(const NodeBuf&) = delete;
    NodeBuf(NodeBuf&&)                 = delete;
    NodeBuf& operator=(NodeBuf&&)      = delete;

    // Attempts to push back an acquired node ID onto the buffer.
    bool TryEnqueue(uint32_t node_id) noexcept;

    // Attempts to pop front a node ID from the buffer.
    uint32_t TryDequeue() noexcept;

    // Loads content of a slot position.
    uint32_t LoadSlot(uint32_t pos) const noexcept {
        return slots_[pos & (capacity - 1)].load(std::memory_order_acquire);
    }

    // Snapshot accessors of `push_head_`.
    // Defines rightmost margin for the L0-tier search routine.
    uint32_t SnapPushHead() const noexcept {
        return push_head_.load(std::memory_order_relaxed);
    }

    // Snapshot accessors of `pop_tail_`.
    // Defines leftmost margin for the L0-tier search routine.
    uint32_t SnapPopTail() const noexcept {
        return pop_tail_.load(std::memory_order_relaxed);
    }

    const uint32_t capacity;

private:
    alignas(64) std::atomic<uint32_t> push_head_{0};
    alignas(64) std::atomic<uint32_t> pop_tail_{0};

    std::unique_ptr<std::atomic<uint32_t>[]> slots_;
};

}  // namespace strix::collection
