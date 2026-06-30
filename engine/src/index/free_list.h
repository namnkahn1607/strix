// Author: namnkahn1607
//
// Lock-free Treiber Stack of free node_id slots. GC pushes evicted
// node_id values back in; cache-miss vector search paths pop one out.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

// TaggedIndex
//
// 64-bit CAS for the Treiber Stack head.
struct alignas(8) TaggedIndex {
    uint32_t head_id;
    // `tag` is a monotonic counter incremented on every `Push` and `Pop`.
    // It disambiguates a head that cycles back to the same `head_id` after
    // an intervening `Pop` and `Push`, hence resolving the ABA problem
    // of a plain Treiber Stack.
    uint32_t tag = 0;
};

static_assert(std::atomic<TaggedIndex>::is_always_lock_free,
              "TaggedIndex CAS must be hardware lock-free.");

// FreeList
//
// Single-producer & Multi-consumer for managing freed `node_id`.
class FreeList {
public:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFU;

    explicit FreeList(const size_t capacity);

    FreeList(const FreeList&)            = delete;
    FreeList& operator=(const FreeList&) = delete;
    FreeList(FreeList&&)                 = delete;
    FreeList& operator=(FreeList&&)      = delete;

    // Push(): declares a `node_id` as freed. Used by the background GC.
    void Push(uint32_t node_id) noexcept;

    // Pop(): allocates a new `node_id`. Used by cache-miss Search path.
    uint32_t Pop() noexcept;

private:
    std::atomic<TaggedIndex> free_head_;
    // `free_next_` is a non-atomic array: a given `node_id` is only ever
    // written by the one GC thread that owns `Push`, so no two writers ever
    // race on the same slot. `Pop` only reads `free_next_` after observing it
    // via the CAS acquire/release pairing, so the read is synchronized
    // without needing `free_next_` itself to be atomic.
    std::unique_ptr<uint32_t[]> free_next_;
};
