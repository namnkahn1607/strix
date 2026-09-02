// Treiber Stack + tagging technique to avoid ABA problem.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace strix {

// Packs a Treiber Stack top with monotonic counter, forming a 8-byte word.
struct alignas(8) TaggedHead {
    uint32_t head_id;

    // Monotonic counter increments on every push/pop.
    uint32_t tag = 0;
    // Disambiguates a cycle back to the old `head_id`, hence resolving the ABA
    // problem of a plain Treiber Stack.
};

static_assert(
    sizeof(TaggedHead) == 8,
    "std::atomic shouldn't add memory overhead to TaggedEntry"
);
static_assert(
    std::atomic<TaggedHead>::is_always_lock_free,
    "TaggedEntry CAS must be hardware lock-free"
);

inline bool operator==(const TaggedHead& a, const TaggedHead& b) {
    return a.head_id == b.head_id && a.tag == b.tag;
}

// Lock-free data structure supporting multiple concurrent model of
// producers and consumers.
class TreiberStack {
public:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;

    explicit TreiberStack(uint32_t capacity) {
        if (capacity == 0) {
            throw std::invalid_argument(
                "Treiber Stack capacity must be positive"
            );
        }

        free_head_.store({0, 0}, std::memory_order_relaxed);

        free_next_ = std::make_unique_for_overwrite<uint32_t[]>(capacity);
        for (uint32_t i = 0; i < capacity - 1; ++i) {
            free_next_[i] = i + 1;
        }
        free_next_[capacity - 1] = kEmpty;
    }

    TreiberStack(const TreiberStack&)            = delete;
    TreiberStack& operator=(const TreiberStack&) = delete;
    TreiberStack(TreiberStack&&)                 = delete;
    TreiberStack& operator=(TreiberStack&&)      = delete;

    void Push(uint32_t id) noexcept {
        TaggedHead old_head = free_head_.load(std::memory_order_relaxed);
        TaggedHead new_head;

        do {
            free_next_[id] = old_head.head_id;
            new_head       = {id, old_head.tag + 1};
        } while (!free_head_.compare_exchange_weak(
            old_head, new_head, std::memory_order_release,
            std::memory_order_relaxed
        ));
    }

    uint32_t Pop() noexcept {
        TaggedHead old_head = free_head_.load(std::memory_order_acquire);

        while (true) {
            if (old_head.head_id == kEmpty) {
                return kEmpty;
            }

            const uint32_t   next_id  = free_next_[old_head.head_id];
            const TaggedHead new_head = {next_id, old_head.tag + 1};

            if (free_head_.compare_exchange_weak(
                    old_head, new_head, std::memory_order_acquire,
                    std::memory_order_acquire
                )) {
                return old_head.head_id;
            }
        }
    }

private:
    std::atomic<TaggedHead>     free_head_;
    std::unique_ptr<uint32_t[]> free_next_;
};

}  // namespace strix
