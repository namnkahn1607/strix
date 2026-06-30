// Author: namnkahn1607
//
// Multi-Producer Single-Consumer ring buffer of node_id values backing
// the L0 cache tier. Cache-miss search paths are producers; the
// background Compaction worker is the sole consumer.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

// L0Indices
//
// Multi-producer & Single-consumer for tracking L0's `node_id`.
class L0Indices {
public:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFU;

    explicit L0Indices(size_t capacity);

    L0Indices(const L0Indices&)            = delete;
    L0Indices& operator=(const L0Indices&) = delete;
    L0Indices(L0Indices&&)                 = delete;
    L0Indices& operator=(L0Indices&&)      = delete;

    // TryPush(): attempts to register a `node_id` onto L0 buffer.
    // Cache-miss search path get a `node_id` slot allocated from `FreeList`,
    // copy its vector data into that slot and register onto L0 buffer.
    bool TryPush(uint32_t node_id) noexcept;

    // TryPop(): attempts to remove a `node_id` from L0 buffer.
    // Used by background compaction worker to migrate a node from L0 to L1.
    uint32_t TryPop() noexcept;

    // LoadSlot(): loads slot's `node_id` content.
    // Caller must ensure `ring_pos` is within bound `[0, capacity_)`.
    uint32_t LoadSlot(const uint32_t ring_pos) const noexcept {
        return slots_[ring_pos].load(std::memory_order_acquire);
    }

    // SnapPushHead(): snapshot accessors on `push_head_`.
    // Used by `VectorIndex::SearchL0` to define its rightmost search margin.
    uint32_t SnapPushHead() const noexcept {
        return push_head_.load(std::memory_order_relaxed);
    }

    // SnapPopTail(): snapshot accessors on `pop_tail_`.
    // Used by `VectorIndex::SearchL0` to define its leftmost search margin.
    uint32_t SnapPopTail() const noexcept {
        return pop_tail_.load(std::memory_order_relaxed);
    }

    size_t Capacity() const noexcept {
        return capacity_;
    }

private:
    const size_t                             capacity_;
    std::unique_ptr<std::atomic<uint32_t>[]> slots_;

    // alignas(64): each isolated onto its own cache line to avoid
    // False Sharing with each other.
    alignas(64) std::atomic<uint32_t> push_head_{0};
    alignas(64) std::atomic<uint32_t> pop_tail_{0};
};
