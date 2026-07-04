// Author: namnkahn1607
//
// MemoryArena implementation: construction, destruction, payload ring buffer
// read/write, lock-free allocation, and the Snowplow garbage collector.

#include "arena.h"

#include <sys/mman.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <thread>

#include "constants.h"
#include "meta_node.h"

namespace {

// Magic number written into every PayloadHeader to distinguish valid headers
// from stale or uninitialized bytes during GC snowplow traversal.
inline constexpr uint32_t kValidIdentifier = 0xDEADBEEF;

// Default ring buffer capacity for Production() config.
inline constexpr uint64_t kPayloadBufferSize = 0x100000000ULL;  // 4 GB

// Ring buffer occupancy thresholds that govern GC sleep intervals.
// Below kLowWatermark  : buffer pressure is low; GC sleeps longer.
// Above kHighWatermark : buffer pressure is high; GC runs near-continuously.
inline constexpr uint64_t kLowWatermark  = 0x80000000ULL;  // 2 GB
inline constexpr uint64_t kHighWatermark = 0xE0000000ULL;  // 3.5 GB

// GC polling intervals in milliseconds, selected by watermark level.
inline constexpr uint32_t kLowGCSleep  = 10;
inline constexpr uint32_t kHighGCSleep = 1;

// How often SweepStalePending() is invoked by the GC loop (milliseconds).
inline constexpr uint32_t kSweepInterval = 5'000;

// MmapRegion
//
// Allocates a private anonymous mapping of `size` bytes with read/write
// permissions. When `lazy` is false, `MAP_POPULATE` is added to instruct
// the kernel to pre-fault all pages during the mmap syscall, eliminating
// first-touch latency at the cost of longer construction time.
void* MmapRegion(const size_t size, const bool lazy) {
    const int flags = MAP_ANONYMOUS | MAP_PRIVATE | (lazy ? 0 : MAP_POPULATE);

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed for MemoryArena");
    }

    return ptr;
}

}  // namespace

ArenaConfig ArenaConfig::Production() {
    return {kTotalSlots, kPayloadBufferSize, false};
}

ArenaConfig ArenaConfig::Compact(const size_t slots) {
    return {slots, 0, false};
}

ArenaConfig ArenaConfig::CompactLazy(const size_t slots) {
    return {slots, 0, true};
}

MemoryArena::MemoryArena(const ArenaConfig& config)
    : max_slots_(config.max_slots)
    , payload_buf_size_(config.payload_buf_size)
    , write_head_(config.start_point)
    , read_tail_(config.start_point) {
    if (!(max_slots_ != 0 && max_slots_ % 4 == 0)) {
        throw std::invalid_argument(
            "Arena slots must be non-zero and multiple of 4");
    }

    if (payload_buf_size_ > 0) {
        if ((payload_buf_size_ & (payload_buf_size_ - 1)) != 0) {
            throw std::invalid_argument(
                "Payload buffer size must be a power of 2");
        }

        if (config.start_point >= payload_buf_size_) {
            throw std::invalid_argument(
                "Start point must be smaller than Payload buffer size");
        }

    } else if (config.start_point != 0) {
        throw std::invalid_argument(
            "Start point must be 0 when there's no Payload buffer");
    }

    metadata_ = static_cast<MetaNode*>(
        MmapRegion(max_slots_ * sizeof(MetaNode), config.lazy_mapping));
    vectors_ = static_cast<float*>(MmapRegion(
        max_slots_ * kVectorDim * sizeof(float), config.lazy_mapping));

    if (payload_buf_size_ > 0) {
        payload_buf_ = static_cast<uint8_t*>(
            MmapRegion(payload_buf_size_, config.lazy_mapping));
    } else {
        payload_buf_ = nullptr;
    }
}

MemoryArena::~MemoryArena() {
    munmap(metadata_, max_slots_ * sizeof(MetaNode));
    munmap(vectors_, max_slots_ * kVectorDim * sizeof(float));

    if (payload_buf_ != nullptr) {
        munmap(payload_buf_, payload_buf_size_);
    }
}

void MemoryArena::RunGarbageCollector(const std::atomic<bool>& g_shutdown_req) {
    assert(payload_buf_ != nullptr &&
           "Memory Arena is missing payload buffer for related operations");

    assert(on_node_freed_ != nullptr &&
           "Callback for free node releasing has not been wired");

    auto last_sweep = std::chrono::steady_clock::now();

    while (!g_shutdown_req.load(std::memory_order_relaxed)) {
        try {
            // Periodically expire stale PENDING nodes.
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_sweep)
                    .count() >= kSweepInterval) {
                const auto wall =
                    std::chrono::system_clock::now().time_since_epoch();
                const uint64_t curr_time = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::seconds>(wall)
                        .count());
                SweepStalePending(curr_time);
                last_sweep = now;
            }

            // Adapt sleep duration to ring buffer occupancy.
            const uint64_t head = write_head_.load(std::memory_order_relaxed);
            const uint64_t tail = read_tail_.load(std::memory_order_relaxed);
            const uint64_t used_space = head - tail;

            if (used_space < kLowWatermark) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(kLowGCSleep));
                continue;
            }

            if (used_space < kHighWatermark) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(kHighGCSleep));
            }

            // ------------------------------------------------------------------
            // Snowplow: advance read_tail_ one entry at a time.
            // ------------------------------------------------------------------

            const uint64_t tail_index = ActualIndex(tail);

            // Not enough contiguous bytes remain before the buffer wraps to
            // hold a PayloadHeader. Skip the padding bytes and restart from
            // the beginning of the buffer.
            if (const uint64_t padding = payload_buf_size_ - tail_index;
                padding < sizeof(PayloadHeader)) {
                read_tail_.fetch_add(padding, std::memory_order_relaxed);
                continue;
            }

            const auto* header = reinterpret_cast<const PayloadHeader*>(
                payload_buf_ + tail_index);

            // Identifier mismatch means we are pointing at padding or a
            // partially written header. Advance by one byte and retry.
            if (header->identifier != kValidIdentifier) {
                read_tail_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            const uint32_t node_id    = header->node_id;
            const uint32_t text_len   = header->length;
            const uint32_t total_size = sizeof(PayloadHeader) + text_len;

            MetaNode& node = metadata_[node_id];
            const auto [state, ref_bit, version, length, v_offset] =
                node.LoadControl();

            // Stale entry: node was already evicted or its virtual offset no
            // longer matches this ring buffer position.
            if (state == NodeState::kDead ||
                v_offset != (tail & kVirtualOffsetMask)) {
                read_tail_.fetch_add(total_size, std::memory_order_relaxed);
                continue;
            }

            // Payload bytes are already committed at this ring position
            // (v_offset matches), but the PENDING -> READY publish hasn't
            // landed yet. Leave it alone; evicting here would destroy an
            // in-flight commit.
            if (state == NodeState::kPending) {
                read_tail_.fetch_add(total_size, std::memory_order_relaxed);
                continue;
            }

            if (ref_bit == EvictState::kCold) {
                // Cold node: evict immediately by transitioning to kDead.
                // Eviction is a Release operation on this slot's identity:
                // version is carried through unchanged, never incremented.
                uint64_t expected =
                    PackControl(state, ref_bit, version, length, v_offset);
                const uint64_t desired = PackControl(NodeState::kDead, ref_bit,
                                                     version, length, v_offset);

                if (node.control_block.compare_exchange_strong(
                        expected, desired, std::memory_order_release,
                        std::memory_order_relaxed)) {
                    on_node_freed_(node_id);
                }

            } else {
                // Hot node: Second Chance - copy payload to a new ring buffer
                // position, cool the ref_bit, and update virtual_offset.
                //
                // If the CAS below fails (a concurrent writer already changed
                // the node's state), the rescued copy becomes an orphan. It
                // will be treated as 'staled' when the snowplow reaches its
                // new position and will be silently skipped. This is an
                // intentional lazy leak: under high CAS contention, orphaned
                // payloads accumulate between read_tail_ and write_head_,
                // reducing effective buffer capacity without corrupting data.
                const std::optional<uint64_t> opt_offset =
                    AllocatePayload(text_len);
                if (!opt_offset.has_value()) {
                    // No room to rescue. Retrying forever would livelock GC
                    // against itself: rescue needs free ring space, and free
                    // space only comes from GC advancing past this exact entry.
                    // Force-evict instead.
                    uint64_t expected =
                        PackControl(state, ref_bit, version, length, v_offset);
                    const uint64_t desired = PackControl(
                        NodeState::kDead, ref_bit, version, length, v_offset);

                    if (node.control_block.compare_exchange_strong(
                            expected, desired, std::memory_order_release,
                            std::memory_order_relaxed)) {
                        on_node_freed_(node_id);
                    }

                    read_tail_.fetch_add(total_size, std::memory_order_relaxed);
                    continue;
                }

                const uint64_t rescued_offset = *opt_offset;
                const uint64_t rescued_index  = ActualIndex(rescued_offset);

                const PayloadHeader new_header{kValidIdentifier, node_id,
                                               text_len};
                std::memcpy(payload_buf_ + rescued_index, &new_header,
                            sizeof(PayloadHeader));

                auto src_idx = ActualIndex(tail + sizeof(PayloadHeader));
                auto dst_idx =
                    ActualIndex(rescued_offset + sizeof(PayloadHeader));

                uint64_t bytes_left = text_len;
                while (bytes_left > 0) {
                    const uint64_t src_cont = payload_buf_size_ - src_idx;
                    const uint64_t dst_cont = payload_buf_size_ - dst_idx;
                    const uint64_t chunk =
                        std::min({bytes_left, src_cont, dst_cont});

                    std::memcpy(payload_buf_ + dst_idx, payload_buf_ + src_idx,
                                chunk);

                    bytes_left -= chunk;
                    src_idx = ActualIndex(src_idx + chunk);
                    dst_idx = ActualIndex(dst_idx + chunk);
                }

                // Rescue relocates a payload; it does not change node
                // identity, so version is carried through unchanged here too.
                uint64_t expected      = PackControl(state, EvictState::kHot,
                                                     version, length, v_offset);
                const uint64_t desired = PackControl(
                    state, EvictState::kCold, version, length, rescued_offset);

                node.control_block.compare_exchange_strong(
                    expected, desired, std::memory_order_release,
                    std::memory_order_relaxed);
            }

            read_tail_.fetch_add(total_size, std::memory_order_relaxed);

        } catch (const std::exception& e) {
            std::cerr << "[Vector Engine] GC WARNING: " << e.what() << "\n";
        }
    }
}

void MemoryArena::SweepStalePending(const uint64_t curr_time) noexcept {
    for (size_t i = 0; i < max_slots_; ++i) {
        MetaNode&      node = metadata_[i];
        const uint64_t ctrl =
            node.control_block.load(std::memory_order_acquire);
        const auto [state, ref_bit, version, length, offset] =
            UnpackControl(ctrl);

        if (state != NodeState::kPending) {
            continue;
        }

        const uint64_t ts = node.created_at.load(std::memory_order_acquire);
        if (curr_time - ts <= kPendingLifespan) {
            continue;
        }

        // Stale PENDING node: transition to kDead via CAS to avoid racing
        // with a concurrent writer that may have already advanced the state.
        uint64_t       expected = ctrl;
        const uint64_t desired =
            PackControl(NodeState::kDead, ref_bit, version, length, offset);

        if (node.control_block.compare_exchange_strong(
                expected, desired, std::memory_order_release,
                std::memory_order_relaxed)) {
            on_node_freed_(static_cast<uint32_t>(i));
        }
    }
}

void MemoryArena::ReadPayload(const uint64_t v_offset, const uint32_t length,
                              std::string* out_payload) const noexcept {
    assert(payload_buf_ != nullptr && "ReadPayload requires a payload buffer");
    assert(out_payload->size() == length &&
           "out_payload must be pre-sized to length by the caller");

    if (length == 0) {
        out_payload->clear();
        return;
    }

    const uint64_t text_index = ActualIndex(v_offset + sizeof(PayloadHeader));
    char*          dst        = out_payload->data();

    if (payload_buf_size_ - text_index >= length) {
        std::memcpy(dst, payload_buf_ + text_index, length);
    } else {
        const uint64_t chunk1 = payload_buf_size_ - text_index;
        const uint64_t chunk2 = length - chunk1;
        std::memcpy(dst, payload_buf_ + text_index, chunk1);
        std::memcpy(dst + chunk1, payload_buf_, chunk2);
    }
}

std::optional<uint64_t> MemoryArena::WritePayload(
    const uint32_t node_id, const uint8_t* in_payload,
    const uint32_t length) noexcept {
    assert(payload_buf_ != nullptr && "WritePayload requires a payload buffer");

    const std::optional<uint64_t> opt_offset = AllocatePayload(length);
    if (!opt_offset.has_value()) {
        return std::nullopt;
    }

    const uint64_t header_offset = *opt_offset;
    const uint64_t header_index  = ActualIndex(header_offset);

    const PayloadHeader header{kValidIdentifier, node_id, length};
    std::memcpy(payload_buf_ + header_index, &header, sizeof(PayloadHeader));

    // text_index is computed from header_index (physical), not header_offset
    // (virtual), to correctly handle the wrap-around case.
    const uint64_t text_index =
        ActualIndex(header_index + sizeof(PayloadHeader));

    if (payload_buf_size_ - text_index >= length) {
        std::memcpy(payload_buf_ + text_index, in_payload, length);
    } else {
        const uint64_t chunk1 = payload_buf_size_ - text_index;
        const uint64_t chunk2 = length - chunk1;
        std::memcpy(payload_buf_ + text_index, in_payload, chunk1);
        std::memcpy(payload_buf_, in_payload + chunk1, chunk2);
    }

    return header_offset;
}

std::optional<uint64_t> MemoryArena::AllocatePayload(
    const uint32_t length) noexcept {
    assert(payload_buf_ != nullptr &&
           "AllocatePayload requires a payload buffer");

    const size_t total_size = sizeof(PayloadHeader) + length;
    uint64_t     curr_write = write_head_.load(std::memory_order_relaxed);

    while (true) {
        if (curr_write + total_size -
                read_tail_.load(std::memory_order_relaxed) >=
            payload_buf_size_) {
            // Expected under high load; not a programming error.
            return std::nullopt;
        }

        const uint64_t actual_index = ActualIndex(curr_write);
        uint64_t       padding      = 0;

        // If the remaining contiguous bytes before the buffer wrap are not
        // enough to hold a PayloadHeader, skip them and restart from offset 0.
        if (payload_buf_size_ - actual_index < sizeof(PayloadHeader)) {
            padding = payload_buf_size_ - actual_index;
        }

        const uint64_t allocated_offset = curr_write + padding;
        const uint64_t next_write       = allocated_offset + total_size;

        if (write_head_.compare_exchange_weak(curr_write, next_write,
                                              std::memory_order_relaxed)) {
            return allocated_offset;
        }
        // CAS failure: curr_write was updated by a concurrent writer.
        // Loop reloads curr_write via the out-parameter and retries.
    }
}

void MemoryArena::PrefaultBuffer() const noexcept {
    assert(payload_buf_ != nullptr &&
           "PrefaultBuffer requires a payload buffer");

    payload_buf_[write_head_] = 0;
    payload_buf_[0]           = 0;
}
