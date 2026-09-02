// Memory Arena implementation: ctor, dtor, payload read/write,
// lock-free allocation, and garbage collection.

#include "memory/arena.h"

#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include "absl/log/check.h"
#include "ann/avx2_dot_product.h"
#include "common/cache_state.h"
#include "hazard_offset.h"
#include "inference/info.h"
#include "memory/allocator.h"
#include "memory/control_block.h"
#include "memory/meta_node.h"
#include "memory/state.h"
#include "payload_header.h"
#include "worker/identity.h"

namespace strix::memory {

Arena::Arena(const ArenaConfig& config)
    : max_slots{config.max_slots}
    , payload_buf_size{config.payload_buf_size}
    , write_head_{config.start_point}
    , read_tail_{config.start_point}
    , hazard_table_{std::make_unique<HazardTable<worker::kNumRPCWorkers>>()} {
    if (!(max_slots != 0 && max_slots % ann::kBatchSize == 0)) {
        throw std::invalid_argument(
            "Arena slots must be non-zero and a multiple of " +
            std::to_string(ann::kBatchSize)
        );
    }

    metadata_ = Alloc<MetaNode>(max_slots, config.prefault);
    vectors_ = Alloc<float>(max_slots * inference::kVectorDim, config.prefault);

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

        payload_buf_ = Alloc<uint8_t>(payload_buf_size, config.prefault);

    } else if (config.start_point != 0) {
        throw std::invalid_argument(
            "Start point must be 0 when payload buffer is omitted"
        );
    }
}

Arena::~Arena() {
    Dealloc(metadata_, max_slots);
    Dealloc(vectors_, max_slots * inference::kVectorDim);
    if (payload_buf_ != nullptr) {
        Dealloc(payload_buf_, payload_buf_size);
    }
}

CacheState Arena::ReadPayload(
    uint32_t node_id, uint8_t exp_ver, Clock::time_point curr_time,
    std::string* out
) const noexcept {
    CHECK(payload_buf_ != nullptr) << "A non-null payload buffer is required";

    // Tolerate ONCE for the case of payload rescuing.
    constexpr uint32_t kMaxReadAttempts = 2;

    const uint32_t slot = worker::ThreadID();
    MetaNode&      node = metadata_[node_id];

    uint64_t     ctrl;
    ControlBlock cb;

    uint32_t attempt = 1;
    while (true) {
        ctrl = node.control_block.load(std::memory_order_acquire);
        cb   = ControlBlock::Unpack(ctrl);

        if (cb.version != exp_ver || cb.state == NodeState::kDead) {
            return CacheState::kMiss;
        }
        if (cb.state == NodeState::kPending) {
            const auto ts = node.created_at.load(std::memory_order_acquire);
            return (curr_time - ts > kPendingLifespan)
                       ? CacheState::kMiss
                       : CacheState::kPendingHit;
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
                return CacheState::kMiss;
            }
            continue;
        }

        hazard_table_->Clear(slot);
        return CacheState::kMiss;
    }

    try {
        out->resize(cb.length);
    } catch (const std::exception&) {
        hazard_table_->Clear(slot);
        return CacheState::kMiss;
    }

    Read(cb.virtual_offset, cb.length, out);
    hazard_table_->Clear(slot);

    if (exp_ver != node.LoadVersion()) {
        // Node was evicted mid-read. Unreliable data retrieval.
        out->clear();
        return CacheState::kMiss;
    }

    auto       expected = ctrl;
    const auto desired  = ControlBlock::Pack(
        NodeState::kReady, EvictState::kHot, cb.version, cb.length,
        cb.virtual_offset
    );
    // This CAS might fail due to 2 cases: another reader promoted this node OR
    // the payload itself has been relocated (rescued by GC).
    node.control_block.compare_exchange_strong(
        expected, desired, std::memory_order_release, std::memory_order_relaxed
    );

    return CacheState::kHit;
}

std::optional<uint64_t> Arena::WritePayload(
    uint32_t node_id, const uint8_t* in, uint32_t length
) noexcept {
    CHECK(payload_buf_ != nullptr) << "A non-null payload buffer is required";

    if (node_id >= max_slots) {
        return std::nullopt;
    }
    MetaNode& node = metadata_[node_id];

    const auto [state, ref, ver, old_len, old_off] = node.LoadControl();
    if (state != NodeState::kPending) {
        return std::nullopt;
    }

    const auto opt_offset = TryAllocateSpace(length);
    if (!opt_offset.has_value()) {
        return std::nullopt;
    }

    const uint64_t header_offset = opt_offset.value();
    const uint64_t header_index  = ActualIndex(header_offset);

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

    auto expected = ControlBlock::Pack(
        NodeState::kPending, EvictState::kCold, ver, old_len, old_off
    );
    const auto desired = ControlBlock::Pack(
        NodeState::kReady, EvictState::kHot, ver, length, header_offset
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

void Arena::RunGarbageCollector(const std::atomic<bool>& shutdown_req) {
    CHECK(payload_buf_ != nullptr) << "A non-null payload buffer is required";
    CHECK(on_node_freed_ != nullptr) << "Freed node callback hasn't been wired";

    constexpr uint64_t kHighWatermark = 0xE0000000ull;  // 3.5 GiB
    constexpr uint32_t kHighGCSleepMs = 1;
    constexpr uint64_t kLowWatermark  = 0x80000000ull;  // 2 GiB
    constexpr uint32_t kLowGCSleepMs  = 10;
    constexpr std::chrono::milliseconds kSweepInterval{5'000u};

    auto last_sweep = Clock::now();
    while (!shutdown_req.load(std::memory_order_relaxed)) {
        auto now = Clock::now();
        if (now - last_sweep >= kSweepInterval) {
            SweepStalePending(now);
            last_sweep = now;
        }

        const uint64_t head = write_head_.load(std::memory_order_relaxed);
        const uint64_t tail = read_tail_.load(std::memory_order_relaxed);

        // Adapt GC active rate to ring buffer occupancy.
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

        const auto tail_index = ActualIndex(tail);

        if (const auto padding = payload_buf_size - tail_index;
            padding < sizeof(PayloadHeader)) {
            read_tail_.fetch_add(padding, std::memory_order_relaxed);
            continue;
        }

        const auto* header =
            reinterpret_cast<const PayloadHeader*>(payload_buf_ + tail_index);
        if (header->identifier != PayloadHeader::kValidIdentifier) {
            // What's reading is not a valid PayloadHeader.
            // Reluctantly advance by one byte and retry.
            read_tail_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        const uint32_t node_id   = header->node_id;
        const uint32_t text_len  = header->length;
        const uint32_t total_len = sizeof(PayloadHeader) + text_len;

        MetaNode& node = metadata_[node_id];

        const auto [state, ref, ver, length, offset] = node.LoadControl();

        if (offset != (tail & kVirtualOffsetMask)) {
            // Virtual offset no longer matches its position in ring buffer.
            // Could be a consequence of payload rescuing, check for published
            // hazards before advacing read_tail.
            TryReclaimSpace(node_id, tail, total_len, false);
            continue;
        }

        if (state == NodeState::kDead) {
            // Node state was changed to DEAD in previous GC iteration.
            TryReclaimSpace(node_id, tail, total_len, true);
            continue;
        }

        // The node owns this payload was evicted, reacquired and becoming
        // PENDING. Its payload will be commited at write_head, not here.
        if (state == NodeState::kPending) {
            read_tail_.fetch_add(total_len, std::memory_order_relaxed);
            continue;
        }

        if (ref == EvictState::kCold) {
            // Cold node: evict immediately by transitioning to DEAD state.
            auto expected = ControlBlock::Pack(
                NodeState::kReady, EvictState::kCold, ver, length, offset
            );
            const auto desired = ControlBlock::Pack(
                NodeState::kDead, EvictState::kCold, ver, length, offset
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
        const auto opt_offset = TryAllocateSpace(text_len);
        if (!opt_offset.has_value()) {
            // No room to rescue. Retrying would livelock GC against itself:
            // rescue needs free ring space, and free space only comes from
            // GC advancing past this exact entry. Force eviction.
            auto expected = ControlBlock::Pack(
                NodeState::kReady, EvictState::kHot, ver, length, offset
            );
            const auto desired = ControlBlock::Pack(
                NodeState::kDead, EvictState::kCold, ver, length, offset
            );
            // This CAS shouldn't fail.
            // Once the node is READY, only GC can change either its node
            // state or reference bit. Eviction hasn't happened yet for an
            // aquisition to change version, length and virtual offset. [*]
            CHECK(node.control_block.compare_exchange_strong(
                expected, desired, std::memory_order_release,
                std::memory_order_relaxed
            )) << "Force-evict CAS shouldn't fail. Check concurrency "
                  "invariants.";
            TryReclaimSpace(node_id, tail, total_len, true);
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

        auto expected = ControlBlock::Pack(
            NodeState::kReady, EvictState::kHot, ver, length, offset
        );
        const auto desired = ControlBlock::Pack(
            NodeState::kReady, EvictState::kCold, ver, length,
            rescued_offset
        );
        // This CAS shouldn't fail for the same reasons as [*].
        CHECK(node.control_block.compare_exchange_strong(
            expected, desired, std::memory_order_release,
            std::memory_order_relaxed
        )) << "Rescue CAS shouldn't fail. Check concurrency invariants.";
        TryReclaimSpace(node_id, tail, total_len, false);
        continue;
    }
}

void Arena::Read(uint64_t offset, uint32_t length, std::string* out)
    const noexcept {
    const auto text_index = ActualIndex(offset + sizeof(PayloadHeader));
    if (payload_buf_size - text_index >= length) {
        std::memcpy(out->data(), payload_buf_ + text_index, length);
    } else {
        const size_t chunk1 = payload_buf_size - text_index;
        const size_t chunk2 = length - chunk1;
        std::memcpy(out->data(), payload_buf_ + text_index, chunk1);
        std::memcpy(out->data() + chunk1, payload_buf_, chunk2);
    }
}

std::optional<uint64_t> Arena::TryAllocateSpace(uint32_t length) noexcept {
    CHECK(payload_buf_ != nullptr) << "A non-null payload buffer is required";

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

void Arena::TryReclaimSpace(
    uint32_t node_id, uint64_t tail, uint32_t total_len, bool release_node
) noexcept {
    if (hazard_table_->Overlaps(tail, total_len)) {
        return;
    }

    if (release_node) {
        on_node_freed_(node_id);
    }
    read_tail_.fetch_add(total_len, std::memory_order_relaxed);
}

void Arena::SweepStalePending(Clock::time_point curr_time) noexcept {
    for (uint32_t node_id = 0; node_id < max_slots; ++node_id) {
        MetaNode& node = metadata_[node_id];

        const uint64_t ctrl =
            node.control_block.load(std::memory_order_acquire);
        const auto [state, ref, ver, length, offset] =
            ControlBlock::Unpack(ctrl);

        if (state != NodeState::kPending) {
            continue;
        }

        const auto ts = node.created_at.load(std::memory_order_acquire);
        if (curr_time - ts <= kPendingLifespan) {
            continue;
        }

        auto       expected = ctrl;
        const auto desired  = ControlBlock::Pack(
            NodeState::kDead, EvictState::kCold, ver, length, offset
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

}  // namespace strix::memory
