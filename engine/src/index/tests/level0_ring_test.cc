// Author: namnkahn1607
//
// Unit tests for L0Buffer. Single-threaded cases pin down FIFO ordering,
// capacity boundary behavior, and ring wraparound correctness; concurrent
// cases assert no-overcommit / no-corrupt-read invariants under the
// production access pattern: N concurrent Producer TryPush callers, one
// Consumer TryPop loop. Meaningful only when run under ThreadSanitizer.

#include "level0_ring.h"

#include <atomic>
#include <barrier>
#include <chrono>
#include <queue>
#include <stdexcept>
#include <thread>

#include "gtest/gtest.h"

TEST(L0BufferSingleThread, AcceptsPowerOfTwoCapacity) {
    EXPECT_NO_THROW(L0Buffer(64));
}

TEST(L0BufferSingleThread, RejectsNonPowerOfTwoCapacity) {
    EXPECT_THROW(L0Buffer(100), std::invalid_argument);
}

TEST(L0BufferSingleThread, PopOnEmptyReturnsEmpty) {
    L0Buffer ring(8);
    EXPECT_EQ(ring.TryPop(), L0Buffer::kEmpty);
}

TEST(L0BufferSingleThread, FillsExactlyToCapacityThenRejects) {
    constexpr uint32_t kCapacity = 8;
    L0Buffer           ring(kCapacity);

    for (uint32_t i = 0; i < kCapacity; ++i) {
        ASSERT_TRUE(ring.TryPush(i)) << "push " << i << " should succeed";
    }

    // Buffer is now full; the next push must be shed.
    EXPECT_FALSE(ring.TryPush(999));
}

TEST(L0BufferSingleThread, PushPopIsFifo) {
    L0Buffer ring(8);

    ASSERT_TRUE(ring.TryPush(10));
    ASSERT_TRUE(ring.TryPush(20));
    ASSERT_TRUE(ring.TryPush(30));

    // MPSC ring buffer is FIFO.
    EXPECT_EQ(ring.TryPop(), 10);
    EXPECT_EQ(ring.TryPop(), 20);
    EXPECT_EQ(ring.TryPop(), 30);
    EXPECT_EQ(ring.TryPop(), L0Buffer::kEmpty);
}

TEST(L0BufferSingleThread, SurvivesManyWrapsAroundCapacity) {
    // Regression test for the monotonic-counter fix: push_head_ and
    // pop_tail_ must remain correct as occupancy counters well past the
    // point where either counter wraps past capacity, not just within
    // the first lap around the ring.
    constexpr uint32_t kCapacity   = 4;
    constexpr uint32_t kIterations = 100'000;
    L0Buffer           ring(kCapacity);

    for (uint32_t i = 0; i < kIterations; ++i) {
        ASSERT_TRUE(ring.TryPush(i))
            << "push " << i
            << " unexpectedly shed on an empty ring; "
               "this indicates the occupancy calculation is wrapping "
               "incorrectly.";
        ASSERT_EQ(ring.TryPop(), i)
            << "pop after push " << i
            << " did not return FIFO order; "
               "ring wraparound has corrupted slot indexing.";
    }

    EXPECT_EQ(ring.TryPop(), L0Buffer::kEmpty);
}

TEST(L0BufferSingleThread, WrapsAroundWithPartiallyFullRing) {
    // Push/pop with the ring never fully drained between operations,
    // forcing push_head_ and pop_tail_ to wrap at different times.
    constexpr uint32_t kCapacity = 4;
    L0Buffer           ring(kCapacity);

    std::queue<uint32_t> expected_order;
    uint32_t             next_value = 0;

    for (uint32_t round = 0; round < 50'000; ++round) {
        if (ring.TryPush(next_value)) {
            expected_order.push(next_value);
            ++next_value;
        }

        if (round % 3 != 0) {
            // Pop slightly less often than push to keep the ring mostly
            // occupied and force push_head_ to lap pop_tail_ repeatedly.
            continue;
        }

        const uint32_t popped = ring.TryPop();
        if (popped != L0Buffer::kEmpty) {
            ASSERT_FALSE(expected_order.empty());
            ASSERT_EQ(popped, expected_order.front());
            expected_order.pop();
        }
    }
}

namespace {

// RunConcurrentProducersFillExactlyToCapacity()
//
// N producer threads TryPush concurrently until the ring buffer reports full,
// each recording its own successes. Asserts the total number of successful
// pushes  is exactly `capacity` - not more (overcommit, two producers winning
// the same slot) and not fewer (a missed slot due to a lost CAS retry).
void RunConcurrentProducersFillExactlyToCapacity(uint32_t capacity,
                                                 size_t   num_producers) {
    L0Buffer ring(capacity);

    std::vector<std::atomic<uint64_t>> per_thread_successes(num_producers);
    for (auto& c : per_thread_successes) {
        c.store(0, std::memory_order_relaxed);
    }

    std::barrier start_gate(static_cast<long>(num_producers));

    std::vector<std::thread> producers;
    producers.reserve(num_producers);
    for (size_t t = 0; t < num_producers; ++t) {
        producers.emplace_back([&ring, &per_thread_successes, &start_gate, t,
                                capacity] {
            start_gate.arrive_and_wait();
            // Each producer attempts well beyond capacity so the ring is
            // guaranteed to be driven to exactly full regardless of how
            // CAS retries get interleaved across threads.
            for (uint32_t i = 0; i < capacity * 4; ++i) {
                if (ring.TryPush(static_cast<uint32_t>(t) * 1'000'000 + i)) {
                    per_thread_successes[t].fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& p : producers) {
        p.join();
    }

    uint64_t total_successes = 0;
    for (auto& c : per_thread_successes) {
        total_successes += c.load(std::memory_order_relaxed);
    }

    ASSERT_EQ(total_successes, capacity)
        << "Total successful pushes must equal capacity exactly; a mismatch "
           "means producers either overcommitted a slot or lost "
           "one to a missed CAS retry.";
    EXPECT_FALSE(ring.TryPush(0))
        << "Ring reported as full by successful-push count must also reject "
           "a further push.";
}

}  // namespace

TEST(L0BufferConcurrency, TwoProducersFillExactlyToCapacity) {
    RunConcurrentProducersFillExactlyToCapacity(1024, 2);
}

TEST(L0BufferConcurrency, ThreeProducersFillExactlyToCapacity) {
    RunConcurrentProducersFillExactlyToCapacity(1024, 3);
}

TEST(L0BufferConcurrency, FourProducersFillExactlyToCapacity) {
    RunConcurrentProducersFillExactlyToCapacity(1024, 4);
}

TEST(L0BufferConcurrency, EightProducersFillExactlyToCapacity) {
    RunConcurrentProducersFillExactlyToCapacity(1024, 8);
}

// -----------------------------------------------------------------------------
// ProducersAndConsumerNeverYieldUnknownValue
// Mirrors the production access pattern: N concurrent producers racing TryPush
// against a single consumer thread draining via TryPop. Asserts every value
// the consumer observes was a value some producer actually pushed - catching
// a torn/garbage read between a producer's release store into a slot and the
// consumer's acquire load of that slot.
// -----------------------------------------------------------------------------

TEST(L0BufferConcurrency, ProducersAndConsumerNeverYieldUnknownValue) {
    constexpr uint32_t kCapacity     = 256;
    constexpr size_t   kNumProducers = 4;
    constexpr uint32_t kMagic        = 0x00FFFFFFu;
    constexpr auto     kRunDuration  = std::chrono::milliseconds(500);

    L0Buffer ring(kCapacity);

    std::atomic<bool>     stop{false};
    std::atomic<uint64_t> next_tag{0};

    // Tag every pushed value so the consumer can recognize "a value some
    // producer legitimately pushed" without needing a shared set guarded by a
    // mutex (which would mask the very race being tested for).
    std::vector<std::thread> producers;
    producers.reserve(kNumProducers);
    for (size_t i = 0; i < kNumProducers; ++i) {
        producers.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                const uint64_t tag =
                    next_tag.fetch_add(1, std::memory_order_relaxed);

                // Low 24 bits carry a tag instead of a real node_id, kept
                // distinct from kEmpty and within uint32_t range.
                const uint32_t value = static_cast<uint32_t>(tag & kMagic);
                ring.TryPush(value);
            }
        });
    }

    std::atomic<uint64_t> popped_count{0};
    std::atomic<uint64_t> bad_value_count{0};
    std::thread           consumer([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            const uint32_t v = ring.TryPop();
            if (v == L0Buffer::kEmpty) {
                continue;
            }

            popped_count.fetch_add(1, std::memory_order_relaxed);
            if (v >= kMagic) {
                // Outside the range any producer could have written;
                // indicates a torn read or uninitialized slot leaking out.
                bad_value_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    std::this_thread::sleep_for(kRunDuration);
    stop.store(true, std::memory_order_relaxed);

    for (auto& p : producers) {
        p.join();
    }

    consumer.join();

    EXPECT_EQ(bad_value_count, 0u);
    EXPECT_GT(popped_count.load(), 0u);
}
