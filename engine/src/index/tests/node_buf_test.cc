// Unit tests for L0-tier node buffer.
//
// Single-threaded cases check FIFO ordering, capacity boundary behavior, and
// ring buffer wrap-around correctness.
// Concurrent cases ensure no-overcommit / no-corrupt-read invariants under the
// MPSC access pattern: N concurrent producers, 1 consumer.
//
// Encourage running under the advisory of ThreadSanitizer to catch specific
// data race cases.

#include "node_buf.h"

#include <atomic>
#include <barrier>
#include <chrono>
#include <queue>
#include <stdexcept>
#include <thread>

#include "gtest/gtest.h"

TEST(NodeBufSingleThread, AcceptsPowerOfTwoCapacity) {
    EXPECT_NO_THROW(NodeBuf(64));
}

TEST(NodeBufSingleThread, RejectsNonPowerOfTwoCapacity) {
    EXPECT_THROW(NodeBuf(100), std::invalid_argument);
}

TEST(NodeBufSingleThread, PopOnEmptyReturnsEmpty) {
    NodeBuf buf(8);
    EXPECT_EQ(buf.TryPop(), NodeBuf::kEmpty);
}

TEST(NodeBufSingleThread, FillsExactlyToCapacityThenRejects) {
    constexpr uint32_t kCapacity = 8;

    NodeBuf buf(kCapacity);
    for (uint32_t i = 0; i < kCapacity; ++i) {
        ASSERT_TRUE(buf.TryPush(i)) << "push " << i << " should succeed";
    }

    // Node buffer is now full. Next push must be shed.
    EXPECT_FALSE(buf.TryPush(999));
}

TEST(NodeBufSingleThread, PushPopIsFifo) {
    NodeBuf ring(8);

    ASSERT_TRUE(ring.TryPush(10));
    ASSERT_TRUE(ring.TryPush(20));
    ASSERT_TRUE(ring.TryPush(30));

    EXPECT_EQ(ring.TryPop(), 10);
    EXPECT_EQ(ring.TryPop(), 20);
    EXPECT_EQ(ring.TryPop(), 30);
    EXPECT_EQ(ring.TryPop(), NodeBuf::kEmpty);
}

// -----------------------------------------------------------------------------
// SurvivesManyWrapsAroundCapacity
// Regression test for the monotonic-counter: push_head_ and pop_tail_ must
// remain correct as occupancy counters well past the point where either counter
// wraps past capacity.
// -----------------------------------------------------------------------------
TEST(NodeBufSingleThread, SurvivesManyWrapsAroundCapacity) {
    constexpr uint32_t kCapacity   = 4;
    constexpr uint32_t kIterations = 100'000u;

    NodeBuf ring(kCapacity);

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

    EXPECT_EQ(ring.TryPop(), NodeBuf::kEmpty);
}

// -----------------------------------------------------------------------------
// WrapsAroundWithPartiallyFullRing
// Push/pop with the ring buffer never fully drained between operations, forcing
// push_head_ and pop_tail_ to wrap at different times.
// -----------------------------------------------------------------------------
TEST(NodeBufSingleThread, WrapsAroundWithPartiallyFullRing) {
    constexpr uint32_t kCapacity    = 4;
    constexpr uint32_t kTotalRounds = 50'000u;
    constexpr uint32_t kDelayPopDiv = 3;

    NodeBuf ring(kCapacity);

    std::queue<uint32_t> expected_order;
    uint32_t             next_value = 0;

    for (uint32_t round = 0; round < kTotalRounds; ++round) {
        if (ring.TryPush(next_value)) {
            expected_order.push(next_value);
            ++next_value;
        }

        if (round % kDelayPopDiv != 0) {
            // Pop happens slightly less often than push to keep the buffer
            // mostly occupied and force push_head_ to lap pop_tail_ repeatedly.
            continue;
        }

        const auto popped = ring.TryPop();
        if (popped != NodeBuf::kEmpty) {
            ASSERT_FALSE(expected_order.empty());
            ASSERT_EQ(popped, expected_order.front());
            expected_order.pop();
        }
    }
}

namespace {

// RunConcurrentProducersSaturateBuffer prompts N producer calling `TryPush`
// concurrently until the buffer is saturated, each recording its own successes.
//
// Asserts total number of successful pushes equals buffer capacity - neither
// more (overcomit, multiple producers winning the same slot) nor fewer (missed
// slot due to a lost CAS retry).
void RunConcurrentProducersSaturateBuffer(
    const uint32_t capacity, const uint32_t num_producers
) {
    constexpr uint32_t kIDMultiplier = 1'000'000u;
    constexpr uint32_t kOverdoFactor = 4;

    std::vector<std::atomic<uint64_t>> per_thread_successes(num_producers);
    for (auto& c : per_thread_successes) {
        c.store(0, std::memory_order_relaxed);
    }

    NodeBuf      ring(capacity);
    std::barrier start_gate(num_producers);

    std::vector<std::thread> producers;
    producers.reserve(num_producers);
    for (uint32_t t = 0; t < num_producers; ++t) {
        producers.emplace_back([&ring, &per_thread_successes, &start_gate, t,
                                capacity] {
            start_gate.arrive_and_wait();

            // Each producer attempts well beyond capacity so the buffer is
            // guaranteed to be driven to saturated regardless of how CAS
            // retries get interleaved across threads.
            for (uint32_t i = 0; i < capacity * kOverdoFactor; ++i) {
                if (ring.TryPush(t * kIDMultiplier + i)) {
                    per_thread_successes[t].fetch_add(
                        1, std::memory_order_relaxed
                    );
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

    ASSERT_EQ(
        total_successes, capacity
    ) << "Total successful pushes must equal capacity; a mismatch means "
         "producers either overcommitted or lost slots to missed CAS retries.";

    EXPECT_FALSE(ring.TryPush(0))
        << "Ring reported as saturated must also reject further push.";
}

}  // namespace

TEST(NodeBufConcurrency, TwoProducersFillExactlyToCapacity) {
    RunConcurrentProducersSaturateBuffer(1'024u, 2);
}

TEST(NodeBufConcurrency, ThreeProducersFillExactlyToCapacity) {
    RunConcurrentProducersSaturateBuffer(1'024u, 3);
}

TEST(NodeBufConcurrency, FourProducersFillExactlyToCapacity) {
    RunConcurrentProducersSaturateBuffer(1'024u, 4);
}

TEST(NodeBufConcurrency, EightProducersFillExactlyToCapacity) {
    RunConcurrentProducersSaturateBuffer(1'024u, 8);
}

// -----------------------------------------------------------------------------
// ProducersAndConsumerNeverYieldUnknownValue
// Mirrors the MPSC access pattern: N concurrent producers racing against a
// single consumer.
// Asserts every value the consumer observes was a value any producer actually
// pushed - catching a torn/garbage read between a producer's release store into
// a slot and the consumer's acquire load of that slot.
// -----------------------------------------------------------------------------
TEST(NodeBufConcurrency, ProducersAndConsumerNeverYieldUnknownValue) {
    constexpr uint32_t kCapacity     = 256;
    constexpr uint32_t kNumProducers = 4;
    constexpr uint32_t kMagic        = 0x00FFFFFFu;
    constexpr auto     kRunDuration  = std::chrono::milliseconds(500);

    NodeBuf ring(kCapacity);

    std::atomic<bool>     stop{false};
    std::atomic<uint64_t> next_tag{0};

    // Tag every pushed value so the consumer can recognize "a value some
    // producer legitimately pushed" without the need of a shared mutex set.
    std::vector<std::thread> producers;
    producers.reserve(kNumProducers);
    for (uint32_t i = 0; i < kNumProducers; ++i) {
        producers.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                const uint64_t tag =
                    next_tag.fetch_add(1, std::memory_order_relaxed);

                // Low 24 bits carry a tag instead of a real `node_id`, kept
                // distinct from `NodeBuf::kEmpty` and within `uint32_t` range.
                const auto value = static_cast<uint32_t>(tag & kMagic);
                ring.TryPush(value);
            }
        });
    }

    std::atomic<uint64_t> popped_count{0};
    std::atomic<uint64_t> bad_value_count{0};
    std::thread           consumer([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            const auto val = ring.TryPop();
            if (val == NodeBuf::kEmpty) {
                continue;
            }

            popped_count.fetch_add(1, std::memory_order_relaxed);
            if (val >= kMagic) {
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
