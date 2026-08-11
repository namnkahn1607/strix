// Unit tests for tagged Treiber Stack.
//
// Single-threaded cases ensure LIFO ordering, empty and boundary behavior.
// Multi-threaded cases assert correctness under concurrent operations against 3
// access model: multi-producers single consumer (MPSC), single-producer
// multi-consumers (SPMC) and multi-producers multi-consumers (MPMC).
//
// Encourage running under the advisory of ThreadSanitizer to catch specific
// data race cases.

#include "common/tagged_treiber.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

TEST(TaggedTreiberSingleThread, ZeroCapacityThrows) {
    EXPECT_THROW(TreiberStack(0), std::invalid_argument);
}

TEST(TaggedTreiberSingleThread, DrainsAllSlotsExactlyOnce) {
    constexpr uint32_t kCapacity = 1'000;
    TreiberStack       stack(kCapacity);

    std::unordered_set<uint32_t> seen;
    for (uint32_t i = 0; i < kCapacity; ++i) {
        const auto id = stack.Pop();
        ASSERT_NE(id, TreiberStack::kEmpty);

        // A duplicate here means Pop handed out same ID twice.
        ASSERT_TRUE(seen.insert(id).second);
    }

    EXPECT_EQ(stack.Pop(), TreiberStack::kEmpty);
    EXPECT_EQ(seen.size(), kCapacity);
}

TEST(TaggedTreiberSingleThread, PushPopIsLifo) {
    constexpr uint32_t kCapacity = 100;

    TreiberStack stack(kCapacity);
    for (uint32_t i = 0; i < kCapacity; ++i) {
        stack.Pop();
    }
    EXPECT_EQ(stack.Pop(), TreiberStack::kEmpty);

    stack.Push(10);
    stack.Push(20);
    stack.Push(40);
    EXPECT_EQ(stack.Pop(), 40);
    EXPECT_EQ(stack.Pop(), 20);
    EXPECT_EQ(stack.Pop(), 10);
    EXPECT_EQ(stack.Pop(), TreiberStack::kEmpty);
}

TEST(TaggedTreiberSingleThread, PushReusedSlotIsPoppableAgain) {
    TreiberStack stack(2);
    const auto   a = stack.Pop();
    const auto   b = stack.Pop();
    ASSERT_NE(a, TreiberStack::kEmpty);
    ASSERT_NE(b, TreiberStack::kEmpty);

    stack.Push(a);
    EXPECT_EQ(stack.Pop(), a);
    EXPECT_EQ(stack.Pop(), TreiberStack::kEmpty);
}

namespace {

// RunConcurrentPopOnlyDrain drains `TreiberStack` of `capacity` from
// `num_threads` concurrent pop(s) with no push.
void RunConcurrentPopOnlyDrain(uint32_t capacity, uint32_t num_threads) {
    if (num_threads <= 0) {
        throw std::invalid_argument("Number of threads must be positive");
    }

    TreiberStack stack(capacity);

    std::vector<std::vector<uint32_t>> per_thread_results(num_threads);
    std::barrier start_gate(static_cast<long>(num_threads));

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([&stack, &per_thread_results, &start_gate, t] {
            start_gate.arrive_and_wait();
            uint32_t id;
            while ((id = stack.Pop()) != TreiberStack::kEmpty) {
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
        << "Total popped count must equal capacity; a mismatch means a slot "
           "was either lost or handed out more than once.";

    std::sort(all_popped.begin(), all_popped.end());
    for (uint32_t i = 0; i < capacity; ++i) {
        ASSERT_EQ(all_popped[i], i)
            << "Sorted popped ids must form a contiguous [0, capacity) range "
               "with no missing or duplicate.";
    }
}

}  // namespace

TEST(TreiberStackConcurrency, TwoPopThreadsDrainExactlyOnce) {
    RunConcurrentPopOnlyDrain(20'000U, 2);
}

TEST(TreiberStackConcurrency, ThreePopThreadsDrainExactlyOnce) {
    RunConcurrentPopOnlyDrain(30'000U, 3);
}

TEST(TreiberStackConcurrency, FourPopThreadsDrainExactlyOnce) {
    RunConcurrentPopOnlyDrain(40'000U, 4);
}

TEST(TreiberStackConcurrency, EightPopThreadsDrainExactlyOnce) {
    RunConcurrentPopOnlyDrain(80'000U, 8);
}

// -----------------------------------------------------------------------------
// ConcurrentPushAndPopMaintainsInvariant
// Covers the multi-producers single consumer (MPSC) case: an ID must not be
// checked out by more than 2 workers at the same time.
// A stack that survives full contention of MPMC access model is safe under
// weaker models such as SPMC or MPSC as well.
// -----------------------------------------------------------------------------
TEST(TreiberStackConcurrency, ConcurrentPushAndPopMaintainsInvariant) {
    constexpr uint32_t kCapacity    = 20'000U;
    constexpr uint32_t kNumThreads  = 8;
    constexpr auto     kRunDuration = std::chrono::seconds(3);

    TreiberStack stack(kCapacity);

    // `state[id] == 0` -> `id` currently sits free inside the stack.
    // `state[id] == 1` -> `id` is being checked out by any thread.
    std::vector<std::atomic<uint8_t>> state(kCapacity);

    std::atomic<bool>     stop{false};
    std::atomic<bool>     violation{false};
    std::atomic<uint64_t> total_ops{0};
    std::barrier          start_gate(static_cast<long>(kNumThreads));

    auto worker = [&] {
        start_gate.arrive_and_wait();
        while (!stop.load(std::memory_order_relaxed)) {
            const auto id = stack.Pop();
            if (id == TreiberStack::kEmpty) {
                continue;
            }

            uint8_t expected = 0;
            if (!state[id].compare_exchange_strong(
                    expected, 1, std::memory_order_acq_rel,
                    std::memory_order_relaxed
                )) {
                violation.store(true, std::memory_order_relaxed);
            }

            // Write ahead, publish later.
            state[id].store(0, std::memory_order_release);
            stack.Push(id);

            total_ops.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(kNumThreads);
    for (uint32_t i = 0; i < kNumThreads; ++i) {
        workers.emplace_back(worker);
    }

    std::this_thread::sleep_for(kRunDuration);
    stop.store(true, std::memory_order_relaxed);
    for (auto& w : workers) {
        w.join();
    }

    ASSERT_FALSE(violation.load())
        << "A worker popped an ID that was being checked out by another "
           "worker; the stack handed out an ID twice concurrently.";
    EXPECT_GT(total_ops.load(), 0U)
        << "No push/pop cycle completed in the run window; check for livelock "
           "in the CAS retry loop.";

    std::unordered_set<uint32_t> drained;

    uint32_t id;
    while ((id = stack.Pop()) != TreiberStack::kEmpty) {
        ASSERT_TRUE(drained.insert(id).second)
            << "Duplicate ID " << id
            << " on final drain - a slot was lost/duplicated during the "
               "concurrent run.";
    }
    EXPECT_EQ(drained.size(), kCapacity);
}
