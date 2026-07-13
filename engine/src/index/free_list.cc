// Author: namnkahn1607
//
// FreeList implementation. See its header for the ownership and
// memory-ordering invariants this relies on.

#include "free_list.h"

#include <atomic>
#include <memory>

FreeList::FreeList(const uint32_t capacity) {
    if (capacity == 0) {
        free_head_.store({kEmpty, 0}, std::memory_order_relaxed);
        return;
    }

    free_next_ = std::make_unique_for_overwrite<uint32_t[]>(capacity);

    for (uint32_t i = 0; i + 1 < capacity; ++i) {
        free_next_[i] = i + 1;
    }

    free_next_[capacity - 1] = kEmpty;
    free_head_.store({0, 0}, std::memory_order_relaxed);
}

void FreeList::Push(const uint32_t node_id) noexcept {
    TaggedIndex old_head = free_head_.load(std::memory_order_relaxed);
    TaggedIndex new_head;

    // free_next_[node_id] is rewritten every retry against the latest
    // old_head, so a failed CAS never leaves a stale `next` pointer behind.
    do {
        free_next_[node_id] = old_head.head_id;
        new_head            = {node_id, old_head.tag + 1};
    } while (!free_head_.compare_exchange_weak(old_head, new_head,
                                               std::memory_order_release,
                                               std::memory_order_relaxed));
}

uint32_t FreeList::Pop() noexcept {
    TaggedIndex old_head = free_head_.load(std::memory_order_acquire);

    while (true) {
        if (old_head.head_id == kEmpty) {
            return kEmpty;
        }

        // Acquire-paired with the Push that wrote this head, so this read
        // of free_next_ is guaranteed to see that Push's write.
        const uint32_t    next_id  = free_next_[old_head.head_id];
        const TaggedIndex new_head = {next_id, old_head.tag + 1};

        if (free_head_.compare_exchange_weak(old_head, new_head,
                                             std::memory_order_acquire,
                                             std::memory_order_acquire)) {
            return old_head.head_id;
        }
        // old_head has been refreshed by compare_exchange_weak on
        // failure; loop retries with the up-to-date value.
    }
}
