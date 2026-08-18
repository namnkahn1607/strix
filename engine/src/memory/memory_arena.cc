// Memory Arena implementation: ctor, dtor, payload read/write,
// lock-free allocation, and garbage collection.

#include "memory/memory_arena.h"

#include <sys/mman.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <thread>

#include "common/cache_outcome.h"
#include "common/syscall_utils.h"
#include "dot_product/avx2_kernel.h"
#include "hazard_offset.h"
#include "memory/control_block.h"
#include "memory/meta_node.h"
#include "memory/state.h"
#include "payload_header.h"
#include "worker/identity.h"

MemoryArena::MemoryArena(const ArenaConfig& config)
    : max_slots(config.max_slots)
    , payload_buf_size(config.payload_buf_size)
    , write_head_(config.start_point)
    , read_tail_(config.start_point) {
    if (!(max_slots != 0 && max_slots % kBatchSize == 0)) {
        throw std::invalid_argument(
            "Arena slots must be non-zero and multiple of 'kBatchSize'"
        );
    }

    if (payload_buf_size > 0) {
        if ((payload_buf_size & (payload_buf_size - 1)) != 0) {
            throw std::invalid_argument(
                "Payload buffer size must be a power of 2"
            );
        }
        if (config.start_point >= payload_buf_size) {
            throw std::invalid_argument(
                "Start point must be smaller than payload buffer size"
            );
        }

    } else if (config.start_point != 0) {
        throw std::invalid_argument(
            "Start point must be 0 when there's no payload buffer"
        );
    }

    metadata_ = static_cast<MetaNode*>(
        common::AllocMMap(max_slots * sizeof(MetaNode), config.prefault)
    );
    vectors_ = static_cast<float*>(
        common::AllocMMap(max_slots * kVectorMemsize, config.prefault)
    );
    if (payload_buf_size > 0) {
        payload_buf_ = static_cast<uint8_t*>(
            common::AllocMMap(payload_buf_size, config.prefault)
        );
    } else {
        payload_buf_ = nullptr;
    }
}

MemoryArena::~MemoryArena() {
    common::DeallocMMap(metadata_, max_slots * sizeof(MetaNode));
    common::DeallocMMap(vectors_, max_slots * kVectorMemsize);
    if (payload_buf_ != nullptr) {
        common::DeallocMMap(payload_buf_, payload_buf_size);
    }
}

CacheOutcome MemoryArena::ReadPayload(
    const uint32_t node_id, const uint8_t exp_ver, const uint64_t curr_time,
    std::string* out
) const noexcept {
    assert(
        payload_buf_ != nullptr &&
        "ReadPayload requires a non-null payload buffer"
    );

    constexpr uint32_t kMaxReadAttempts = 2;

    const uint32_t slot = WorkerID();
    MetaNode&      node = metadata_[node_id];

    uint64_t     ctrl;
    ControlBlock cb;

    uint32_t attempt = 0;
    while (true) {
        ctrl = node.control_block.load(std::memory_order_acquire);
        cb   = UnpackControl(ctrl);

        if (cb.version != exp_ver || cb.state == NodeState::kDead) {
            return CacheOutcome::kMiss;
        }
        if (cb.state == NodeState::kPending) {
            const uint64_t ts = node.created_at.load(std::memory_order_acquire);
            return (curr_time - ts > kPendingLifespan)
                       ? CacheOutcome::kMiss
                       : CacheOutcome::kPendingHit;
        }

        hazard_table_->Publish(slot, cb.virtual_offset, cb.length);

        const auto cb2 = node.LoadControl();
        if (cb2.state == NodeState::kReady && cb2.version == cb.version) {
            if (cb2.virtual_offset == cb.virtual_offset) {
                // Stable virtual offset confirm. Proceed reading.
                break;
            }

            hazard_table_->Clear(slot);
            if (++attempt > kMaxReadAttempts) {
                return CacheOutcome::kMiss;
            }
            continue;
        }

        hazard_table_->Clear(slot);
        return CacheOutcome::kMiss;
    }

    try {
        out->resize(cb.length);
    } catch (const std::exception&) {
        hazard_table_->Clear(slot);
        return CacheOutcome::kMiss;
    }

    const auto text_index =
        ActualIndex(cb.virtual_offset + sizeof(PayloadHeader));
    if (payload_buf_size - text_index >= cb.length) {
        std::memcpy(out->data(), payload_buf_ + text_index, cb.length);
    } else {
        const size_t chunk1 = payload_buf_size - text_index;
        const size_t chunk2 = cb.length - chunk1;
        std::memcpy(out->data(), payload_buf_ + text_index, chunk1);
        std::memcpy(out->data() + chunk1, payload_buf_, chunk2);
    }

    hazard_table_->Clear(slot);

    if (exp_ver != node.LoadVersion()) {
        // Node was evicted mid-read. Unreliable data retrieval.
        out->clear();
        return CacheOutcome::kMiss;
    }

    auto       expected = ctrl;
    const auto desired  = PackControl(
        NodeState::kReady, EvictState::kHot, cb.version, cb.length,
        cb.virtual_offset
    );
    node.control_block.compare_exchange_strong(
        expected, desired, std::memory_order_release, std::memory_order_relaxed
    );

    return CacheOutcome::kHit;
}

std::optional<uint64_t> MemoryArena::WritePayload(
    const uint32_t node_id, const uint8_t* in, const uint32_t length
) noexcept {
    assert(
        payload_buf_ != nullptr &&
        "WritePayload requires a non-null payload buffer"
    );

    if (node_id >= max_slots) {
        return std::nullopt;
    }

    MetaNode& node = metadata_[node_id];

    const auto [state, ref, version, old_len, old_off] = node.LoadControl();
    if (state != NodeState::kPending) {
        return std::nullopt;
    }

    const auto opt_offset = AllocatePayload(length);
    if (!opt_offset.has_value()) {
        return std::nullopt;
    }

    const uint64_t header_offset = *opt_offset;
    const auto     header_index  = ActualIndex(header_offset);

    const PayloadHeader header{
        PayloadHeader::kValidIdentifier, node_id, length
    };
    std::memcpy(payload_buf_ + header_index, &header, sizeof(PayloadHeader));

    const auto text_index = ActualIndex(header_index + sizeof(PayloadHeader));
    if (payload_buf_size - text_index >= length) {
        std::memcpy(payload_buf_ + text_index, in, length);
    } else {
        const size_t chunk1 = payload_buf_size - text_index;
        const size_t chunk2 = length - chunk1;
        std::memcpy(payload_buf_ + text_index, in, chunk1);
        std::memcpy(payload_buf_, in + chunk1, chunk2);
    }

    auto expected = PackControl(
        NodeState::kPending, EvictState::kCold, version, old_len, old_off
    );
    const auto desired = PackControl(
        NodeState::kReady, EvictState::kHot, version, length, header_offset
    );
    if (!node.control_block.compare_exchange_strong(
            expected, desired, std::memory_order_release,
            std::memory_order_relaxed
        )) {
        // The node was either evicted or reacquired between metadata read and
        // this point. The written payload bytes becomes orphanated in the
        // buffer, waiting for GC to come and reclaim.
        return std::nullopt;
    }

    return header_offset;
}

void MemoryArena::RunGarbageCollector(const std::atomic<bool>& shutdown_req) {
    assert(
        payload_buf_ != nullptr &&
        "RunGarbageCollector requires a non-null payload buffer"
    );
    assert(
        on_node_freed_ != nullptr &&
        "Callback for free node releasing has not been wired"
    );

    constexpr uint64_t kHighWatermark   = 0xE0000000ull;  // 3.5 GB
    constexpr uint32_t kHighGCSleepMs   = 1;
    constexpr uint64_t kLowWatermark    = 0x80000000ull;  // 2 GB
    constexpr uint32_t kLowGCSleepMs    = 10;
    constexpr uint32_t kSweepIntervalMs = 5'000u;

    auto last_sweep = common::MonotonicNow();
    while (!shutdown_req.load(std::memory_order_relaxed)) {
        // Expire stale PENDING nodes periodically.
        auto now = common::MonotonicNow();
        if (now - last_sweep >= kSweepIntervalMs) {
            SweepStalePending(now);
            last_sweep = now;
        }

        // Adapt GC active rate to ring buffer occupancy.
        const uint64_t head = write_head_.load(std::memory_order_relaxed);
        const uint64_t tail = read_tail_.load(std::memory_order_relaxed);

        const uint64_t used_space = head - tail;
        if (used_space < kLowWatermark) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kLowGCSleepMs)
            );
            continue;
        }
        if (used_space < kHighWatermark) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kHighGCSleepMs
            ));
        }

        // Snowplow: advance read_tail_ once (per loop) at a time.
        const auto tail_index = ActualIndex(tail);

        // Not enough contiguous bytes remain before the buffer wraps to
        // hold a PayloadHeader. Restart from the beginning of the buffer.
        if (const auto padding = payload_buf_size - tail_index;
            padding < sizeof(PayloadHeader)) {
            read_tail_.fetch_add(padding, std::memory_order_relaxed);
            continue;
        }

        const auto* header =
            reinterpret_cast<const PayloadHeader*>(payload_buf_ + tail_index);

        // What's reading is not a valid PayloadHeader.
        // Reluctantly advance by one byte and retry.
        if (header->identifier != PayloadHeader::kValidIdentifier) {
            read_tail_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        const uint32_t node_id   = header->node_id;
        const uint32_t text_len  = header->length;
        const uint32_t total_len = sizeof(PayloadHeader) + text_len;

        MetaNode& node = metadata_[node_id];

        const auto [state, ref, version, length, offset] = node.LoadControl();

        // Virtual offset no longer matches its position in ring buffer.
        // Could be a consequence of payload rescuing. Check for published
        // hazards before advacing read_tail.
        if (offset != (tail & kVirtualOffsetMask)) {
            TryReclaimSpace(node_id, tail, total_len, false);
            continue;
        }

        // Node state was changed to DEAD in previous GC iteration.
        if (state == NodeState::kDead) {
            TryReclaimSpace(node_id, tail, total_len, true);
            continue;
        }

        // GC isn't responsible for expiring PENDING nodes.
        // A node owned this payload, was evicted, reacquired and becoming
        // PENDING. Its payload will be commited at write_head, not here.
        if (state == NodeState::kPending) {
            read_tail_.fetch_add(total_len, std::memory_order_relaxed);
            continue;
        }

        if (ref == EvictState::kCold) {
            // Cold node: evict immediately by transitioning to DEAD state.
            auto expected = PackControl(
                NodeState::kReady, EvictState::kCold, version, length, offset
            );
            const auto desired = PackControl(
                NodeState::kDead, EvictState::kCold, version, length, offset
            );
            if (node.control_block.compare_exchange_strong(
                    expected, desired, std::memory_order_release,
                    std::memory_order_relaxed
                )) {
                TryReclaimSpace(node_id, tail, total_len, true);
                continue;
            }
            // CAS failed. A worker might HIT this payload and promote the node
            // from COLD -> HOT. Fallthrough to the HOT case below.
        }

        // Hot node: grant it a Second Chance while cooling its reference bit.
        const auto opt_offset = AllocatePayload(text_len);
        if (!opt_offset.has_value()) {
            // No room to rescue. Retrying would livelock GC against itself:
            // rescue needs free ring space, and free space only comes from
            // GC advancing past this exact entry. Force eviction.
            auto expected = PackControl(
                NodeState::kReady, EvictState::kHot, version, length, offset
            );
            const auto desired = PackControl(
                NodeState::kDead, EvictState::kCold, version, length, offset
            );
            if (node.control_block.compare_exchange_strong(
                    expected, desired, std::memory_order_release,
                    std::memory_order_relaxed
                )) {
                TryReclaimSpace(node_id, tail, total_len, true);
                continue;
            }

            // The CAS above shouldn't fail.
            // Once the node is READY, only GC can change either its node state
            // or reference bit. Eviction hasn't happened yet for an aquisition
            // to change version, length and virtual offset. [*]
            assert(false && "Force-evict CAS should not fail");
            continue;
        }

        // Perform payload rescue: copy payload to a new position, cool the
        // reference bit, and update new virtual offset.
        const uint64_t rescued_offset = *opt_offset;
        const auto     rescued_index  = ActualIndex(rescued_offset);

        const PayloadHeader new_header{
            PayloadHeader::kValidIdentifier, node_id, text_len
        };
        std::memcpy(
            payload_buf_ + rescued_index, &new_header, sizeof(PayloadHeader)
        );

        auto src_idx = ActualIndex(tail + sizeof(PayloadHeader));
        auto dst_idx = ActualIndex(rescued_index + sizeof(PayloadHeader));

        uint64_t bytes_left = text_len;
        while (bytes_left > 0) {
            const uint64_t src_cont = payload_buf_size - src_idx;
            const uint64_t dst_cont = payload_buf_size - dst_idx;
            const uint64_t chunk = std::min({bytes_left, src_cont, dst_cont});

            std::memcpy(payload_buf_ + dst_idx, payload_buf_ + src_idx, chunk);
            bytes_left -= chunk;
            src_idx = ActualIndex(src_idx + chunk);
            dst_idx = ActualIndex(dst_idx + chunk);
        }

        auto expected = PackControl(
            NodeState::kReady, EvictState::kHot, version, length, offset
        );
        const auto desired = PackControl(
            NodeState::kReady, EvictState::kCold, version, length,
            rescued_offset
        );
        if (node.control_block.compare_exchange_strong(
                expected, desired, std::memory_order_release,
                std::memory_order_relaxed
            )) {
            TryReclaimSpace(node_id, tail, total_len, false);
            continue;
        }

        // The CAS above shouldn't fail for the same reasons as [*].
        assert(false && "Rescue CAS could not fail");
        continue;
    }
}

std::optional<uint64_t> MemoryArena::AllocatePayload(const uint32_t length
) noexcept {
    assert(
        payload_buf_ != nullptr &&
        "AllocatePayload requires a non-null payload buffer"
    );

    const size_t total_size = sizeof(PayloadHeader) + length;
    uint64_t     curr_write = write_head_.load(std::memory_order_relaxed);

    while (true) {
        if (curr_write + total_size -
                read_tail_.load(std::memory_order_relaxed) >=
            payload_buf_size) {
            // Expected allocation failure under high load.
            return std::nullopt;
        }

        const auto actual_index = ActualIndex(curr_write);

        uint64_t padding = 0;
        if (payload_buf_size - actual_index < sizeof(PayloadHeader)) {
            padding = payload_buf_size - actual_index;
        }

        const uint64_t allocated_offset = curr_write + padding;
        const uint64_t next_write       = allocated_offset + total_size;
        if (write_head_.compare_exchange_weak(
                curr_write, next_write, std::memory_order_relaxed
            )) {
            return allocated_offset;
        }
    }
}

void MemoryArena::SweepStalePending(const uint64_t curr_time) noexcept {
    for (uint32_t node_id = 0; node_id < max_slots; ++node_id) {
        MetaNode& node = metadata_[node_id];

        const uint64_t ctrl =
            node.control_block.load(std::memory_order_acquire);
        const auto [state, ref, version, length, offset] = UnpackControl(ctrl);

        if (state != NodeState::kPending) {
            continue;
        }

        const uint64_t ts = node.created_at.load(std::memory_order_acquire);
        if (curr_time - ts <= kPendingLifespan) {
            continue;
        }

        auto       expected = ctrl;
        const auto desired  = PackControl(
            NodeState::kDead, EvictState::kCold, version, length, offset
        );
        if (node.control_block.compare_exchange_strong(
                expected, desired, std::memory_order_release,
                std::memory_order_relaxed
            )) {
            on_node_freed_(node_id);
        }
        // CAS failed. A worker committed payload for this node and landed a
        // PENDING -> READY publish.
    }
}

void MemoryArena::TryReclaimSpace(
    const uint32_t node_id, const uint64_t tail, const uint32_t total_len,
    const bool release_node
) noexcept {
    if (hazard_table_->Overlaps(tail, total_len)) {
        return;
    }

    if (release_node) {
        on_node_freed_(node_id);
    }
    read_tail_.fetch_add(total_len, std::memory_order_relaxed);
}
