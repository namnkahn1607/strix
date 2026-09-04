// L0-tier node buffer.

#include "node_buf.h"

#include <atomic>
#include <memory>
#include <stdexcept>

namespace strix::collection {

NodeBuf::NodeBuf(uint32_t capacity) : capacity{capacity} {
    if ((capacity & (capacity - 1)) != 0) {
        throw std::invalid_argument(
            "L0-tier node buffer capacity must be a power of 2"
        );
    }

    slots_ = std::make_unique_for_overwrite<std::atomic<uint32_t>[]>(capacity);
    for (uint32_t i = 0; i < capacity; ++i) {
        slots_[i].store(kEmpty, std::memory_order_relaxed);
    }
}

bool NodeBuf::TryEnqueue(uint32_t node_id) noexcept {
    uint32_t curr_push = push_head_.load(std::memory_order_relaxed);

    while (true) {
        if (curr_push - pop_tail_.load(std::memory_order_relaxed) >= capacity) {
            // Node buffer full, fails the registration anyway.
            return false;
        }

        const uint32_t next_push = curr_push + 1;
        if (push_head_.compare_exchange_weak(
                curr_push, next_push, std::memory_order_relaxed,
                std::memory_order_relaxed
            )) {
            break;
        }
        // CAS failure means another producer won the race.
        // curr_push is updated to its latest value; loop re-checks capacity
        // and retries.
    }

    const uint32_t ring_pos = curr_push & (capacity - 1);
    slots_[ring_pos].store(node_id, std::memory_order_release);
    return true;
}

uint32_t NodeBuf::TryDequeue() noexcept {
    const uint32_t curr_pop = pop_tail_.load(std::memory_order_relaxed);

    if (curr_pop == push_head_.load(std::memory_order_relaxed)) {
        // Node buffer empty, emit exhaustive signal.
        return kEmpty;
    }

    const uint32_t ring_pos = curr_pop & (capacity - 1);
    const uint32_t node_id  = slots_[ring_pos].load(std::memory_order_acquire);
    slots_[ring_pos].store(kEmpty, std::memory_order_relaxed);

    // Only one consumer. A relaxed store is sufficient.
    pop_tail_.store(curr_pop + 1, std::memory_order_relaxed);

    return node_id;
}

}  // namespace strix::collection
