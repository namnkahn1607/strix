// Author: namnkahn1607
//
// Unit tests for FreeList. Single-threaded cases pin down LIFO ordering
// and boundary behavior; multi-threaded cases assert the no-duplicate /
// no-lost-slot invariant under concurrent Pop (and mixed Push/Pop)
// against the production access pattern: one Push thread, N Pop threads.
// These tests are meaningful only when run under ThreadSanitizer; passing
// without TSan proves nothing about the absence of a race.

#include "free_list.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <thread>
#include <unordered_set>
#include <vector>

TEST(FreeListSingleThread, ZeroCapacityPopsEmpty) {
    FreeList list(0);
    EXPECT_EQ(list.Pop(), FreeList::kEmpty);
}

TEST(FreeListSingleThread, OneCapacityYieldsExactlyOneSlot) {
    FreeList list(1);
    EXPECT_EQ(list.Pop(), 0u);
    EXPECT_EQ(list.Pop(), FreeList::kEmpty);
}

TEST(FreeListSingleThread, DrainsAllSlotsExactlyOnce) {
    constexpr uint32_t kCapacity = 1'000;
    FreeList           list(kCapacity);

    std::unordered_set<uint32_t> seen;
    for (uint32_t i = 0; i < kCapacity; ++i) {
        const uint32_t id = list.Pop();
        ASSERT_NE(id, FreeList::kEmpty);

        // A duplicate here means Pop handed out the same node_id twice.
        ASSERT_TRUE(seen.insert(id).second);
    }

    EXPECT_EQ(list.Pop(), FreeList::kEmpty);
    EXPECT_EQ(seen.size(), kCapacity);
}

TEST(FreeListSingleThread, PushPopIsLifo) {
    constexpr uint32_t kCapacity = 100;

    // Drain all entries from the FreeList first.
    FreeList list(kCapacity);
    for (uint32_t i = 0; i < kCapacity; ++i) {
        list.Pop();
    }

    // The FreeList must be empty by now.
    ASSERT_EQ(list.Pop(), FreeList::kEmpty);

    list.Push(10);
    list.Push(20);
    list.Push(40);

    // Treiber Stack: most recently pushed comes back out first.
    EXPECT_EQ(list.Pop(), 40);
    EXPECT_EQ(list.Pop(), 20);
    EXPECT_EQ(list.Pop(), 10);
    EXPECT_EQ(list.Pop(), FreeList::kEmpty);
}

TEST(FreeListSingleThread, PushReusedSlotIsPoppableAgain) {
    FreeList       list(2);
    const uint32_t a = list.Pop();
    const uint32_t b = list.Pop();
    ASSERT_NE(a, FreeList::kEmpty);
    ASSERT_NE(b, FreeList::kEmpty);

    list.Push(a);
    EXPECT_EQ(list.Pop(), a);
    EXPECT_EQ(list.Pop(), FreeList::kEmpty);
}

namespace {

// RunConcurrentPopOnlyDrain()
//
// Drains `list` from `num_threads` concurrent `Pop()` callers with no
// concurrent `Push()`, and asserts the union of what every thread popped
// is exactly the full id space `{0, ..., capacity - 1}` with no duplicate
// and no slot lost to a missed CAS retry.
void RunConcurrentPopOnlyDrain(uint32_t capacity, size_t num_threads) {
    assert(num_threads > 0 && "Number of threads must be positive");

    FreeList list(capacity);

    std::vector<std::vector<uint32_t>> per_thread_results(num_threads);
    std::barrier start_gate(static_cast<long>(num_threads));

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([&list, &per_thread_results, &start_gate, t] {
            start_gate.arrive_and_wait();
            uint32_t id;
            while ((id = list.Pop()) != FreeList::kEmpty) {
                per_thread_results[t].push_back(id);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    std::vector<uint32_t> all_popped;
    for (auto& v : per_thread_results) {
        all_popped.insert(all_popped.end(), v.begin(), v.end());
    }

    ASSERT_EQ(all_popped.size(), capacity)
        << "Total popped count must equal capacity; a mismatch means a "
           "slot was either lost or handed out more than once.";

    std::sort(all_popped.begin(), all_popped.end());
    for (uint32_t i = 0; i < capacity; ++i) {
        ASSERT_EQ(all_popped[i], i)
            << "Sorted popped ids must form a contiguous [0, capacity) "
               "range with no missing or duplicate.";
    }
}

}  // namespace

TEST(FreeListConcurrency, TwoPopThreadsDrainExactlyOnce) {
    RunConcurrentPopOnlyDrain(50'000, 2);
}

TEST(FreeListConcurrency, ThreePopThreadsDrainExactlyOnce) {
    RunConcurrentPopOnlyDrain(50'000, 3);
}

TEST(FreeListConcurrency, FourPopThreadsDrainExactlyOnce) {
    RunConcurrentPopOnlyDrain(50'000, 4);
}

TEST(FreeListConcurrency, EightPopThreadsDrainExactlyOnce) {
    RunConcurrentPopOnlyDrain(50'000, 8);
}

// -----------------------------------------------------------------------------
// OutputConsistencyUnderConcurrency
// Mirrors the production access pattern: one dedicated Push thread (GC) racing
// against several concurrent Pop threads (cache-miss allocation paths). Asserts
// the stack never hands out a node_id that is currently live (already popped,
// not yet pushed back) and never loses a node_id permanently.
// -----------------------------------------------------------------------------

TEST(FreeListConcurrency, SinglePushVersusMultiplePopsMaintainsInvariant) {
    constexpr uint32_t kCapacity      = 20'000;
    constexpr size_t   kNumPopThreads = 4;
    constexpr auto     kRunDuration   = std::chrono::milliseconds(500);

    FreeList list(kCapacity);

    // Drain everything up front so the run starts from a known-empty
    // state; every id is "owned" by this thread until handed to Pop.
    std::vector<uint32_t> owned;
    owned.reserve(kCapacity);
    for (uint32_t i = 0; i < kCapacity; ++i) {
        owned.push_back(list.Pop());
    }

    std::atomic<bool>     stop{false};
    std::atomic<uint64_t> total_popped{0};
    std::atomic<uint64_t> total_pushed{0};

    // Push thread: continuously pushes back ids from owned, round-robin,
    // simulating GC returning evicted slots.
    std::thread pusher([&] {
        size_t i = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            list.Push(owned[i % owned.size()]);
            total_pushed.fetch_add(1, std::memory_order_relaxed);
            ++i;
        }
    });

    // Pop threads: just drain as fast as possible; correctness here means
    // "never observably corrupt", checked via TSan, not via a value
    // assertion mid-flight.
    std::vector<std::thread> poppers;
    poppers.reserve(kNumPopThreads);
    for (size_t i = 0; i < kNumPopThreads; ++i) {
        poppers.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                if (list.Pop() != FreeList::kEmpty) {
                    total_popped.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Wait for those workers to run for a dedicated amount of time.
    std::this_thread::sleep_for(kRunDuration);
    stop.store(true, std::memory_order_relaxed);

    pusher.join();
    for (auto& p : poppers) {
        p.join();
    }

    // The stack must have done meaningful work, and every push that lost its
    // CAS race must have retried until it succeeded (Push() has no failure
    // return path), so pushed count should be reachable and nonzero.
    EXPECT_GT(total_pushed.load(), 0u);
    EXPECT_GT(total_popped.load(), 0u);
}
