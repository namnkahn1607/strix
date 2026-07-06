// Author: namnkahn1607
//
// L0Indices implementation. See its header for the monotonic-counter
// invariant this relies on.

#include "level0_ring.h"

#include <atomic>
#include <memory>
#include <stdexcept>

L0Indices::L0Indices(const size_t capacity) : capacity_(capacity) {
    if ((capacity & (capacity - 1)) != 0) {
        throw std::invalid_argument("L0 capacity must be a power of 2.");
    }

    slots_ = std::make_unique_for_overwrite<std::atomic<uint32_t>[]>(capacity);

    // Initialize all slots to empty.
    for (size_t i = 0; i < capacity_; ++i) {
        slots_[i].store(kEmpty, std::memory_order_relaxed);
    }
}

bool L0Indices::TryPush(const uint32_t node_id) noexcept {
    uint32_t curr_push = push_head_.load(std::memory_order_relaxed);

    while (true) {
        if (curr_push - pop_tail_.load(std::memory_order_relaxed) >=
            capacity_) {
            // L0 buffer full, fails the registration immediately.
            return false;
        }

        const uint32_t next_push = curr_push + 1;
        if (push_head_.compare_exchange_weak(curr_push, next_push,
                                             std::memory_order_relaxed,
                                             std::memory_order_relaxed)) {
            break;
        }
        // CAS failure means another Producer won the race;
        // curr_push has been refreshed to the current value by
        // compare_exchange_weak, loop re-checks capacity and
        // retries.
    }

    // This Producer won the CAS race.
    const uint32_t ring_pos = curr_push & (capacity_ - 1);
    slots_[ring_pos].store(node_id, std::memory_order_release);
    return true;
}

uint32_t L0Indices::TryPop() noexcept {
    const uint32_t curr_pop = pop_tail_.load(std::memory_order_relaxed);

    if (curr_pop == push_head_.load(std::memory_order_relaxed)) {
        // L0 buffer empty, return exhaustive signal.
        return kEmpty;
    }

    const uint32_t ring_pos = curr_pop & (capacity_ - 1);
    const uint32_t node_id  = slots_[ring_pos].load(std::memory_order_acquire);
    slots_[ring_pos].store(kEmpty, std::memory_order_relaxed);

    // There's only one Consumer, so a relaxed store is sufficient.
    pop_tail_.store(curr_pop + 1, std::memory_order_relaxed);

    return node_id;
}
