//
// memory/arena.cc
//

#include "arena.hh"

#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "constants.hh"
#include "meta_node.hh"

namespace {

inline constexpr uint32_t VALID_IDENTIFIER = 0xDEADBEEF;

inline constexpr uint64_t LOW_WATERMARK = 2ULL * 1024 * 1024 * 1024;  // 2GB
inline constexpr uint64_t HIGH_WATERMARK =
    (3ULL * 1024 + 512ULL) * 1024 * 1024;  // 3.5GB

// Garbage collector LOW sleeping interval in millisecond(s)
inline constexpr uint32_t LOW_GC_SLEEP = 10;
// Garbage collector HIGH sleeping interval in millisecond(s)
inline constexpr uint32_t HIGH_GC_SLEEP = 1;

// How often a stale sweeper runs in millsecond(s)
inline constexpr uint32_t SWEEP_INTERVAL = 5'000;

// --- MmapAllocate ---
// Performs mmap an anonymous (MAP_ANONYMOUS) private (MAP_PRIVATE) region,
// MAP_POPULATE instructs the kernel to pre-faults all pages inside the syscall.
void* MmapAllocate(const size_t size) {
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_ANONYMOUS | MAP_PRIVATE | MAP_POPULATE, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed for MemoryArena");
    }

    return ptr;
}

}  // namespace

// ------------------------------------------------------------
// Constructor / Destructor
// ------------------------------------------------------------

MemoryArena::MemoryArena() : write_head(0), read_tail(0) {
    metadata = static_cast<MetaNode*>(
        MmapAllocate(TOTAL_MAX_SLOTS * sizeof(MetaNode)));
    vectors =
        static_cast<float*>(MmapAllocate(TOTAL_MAX_SLOTS * sizeof(float)));
    buffer_payload = static_cast<uint8_t*>(MmapAllocate(PAYLOAD_BUFFER_SIZE));
}

MemoryArena::~MemoryArena() {
    munmap(metadata, TOTAL_MAX_SLOTS * sizeof(MetaNode));
    munmap(vectors, TOTAL_MAX_SLOTS * sizeof(float));
    munmap(buffer_payload, PAYLOAD_BUFFER_SIZE);
}

// ------------------------------------------------------------
// Garbage Collector
// ------------------------------------------------------------

void MemoryArena::RunGarbageCollector(const std::atomic<bool>& g_shutdown_req) {
    try {
        auto last_sweep = std::chrono::steady_clock::now();

        while (!g_shutdown_req.load(std::memory_order_relaxed)) {
            // Trigger stale Node sweeper periodically
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_sweep)
                    .count() >= SWEEP_INTERVAL) {
                const auto wall =
                    std::chrono::system_clock::now().time_since_epoch();
                const uint64_t curr_time = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::seconds>(wall)
                        .count());
                SweepStalePending(curr_time);
                last_sweep = now;
            }

            const uint64_t head = write_head.load(std::memory_order_relaxed);
            const uint64_t tail = read_tail.load(std::memory_order_relaxed);
            const uint64_t used_space = head - tail;

            if (used_space < LOW_WATERMARK) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(LOW_GC_SLEEP));
                continue;
            }

            if (used_space < HIGH_WATERMARK) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(HIGH_GC_SLEEP));
            }

            // --- Snowplow ---
            const uint64_t tail_index = ActualIndex(tail);

            // Leaping in case not enough room for a valid Payload Header
            if (const uint64_t padding = PAYLOAD_BUFFER_SIZE - tail_index;
                padding < sizeof(PayloadHeader)) {
                read_tail.fetch_add(padding, std::memory_order_relaxed);
                continue;
            }

            const auto* header = reinterpret_cast<const PayloadHeader*>(
                buffer_payload + tail_index);

            // Reluctantly advancing read tail until found a valid header
            if (header->identifier != VALID_IDENTIFIER) {
                read_tail.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            const uint32_t node_id = header->node_id;
            const uint32_t text_len = header->length;
            const uint32_t total_size = sizeof(PayloadHeader) + text_len;

            MetaNode& node = metadata[node_id];
            const auto [state, ref_bit, length, v_offset] = node.LoadControl();

            // Skip if node is already DEAD or stale entry
            if (state == NodeState::DEAD ||
                v_offset != (tail & VIRTUAL_OFFSET_MASK)) {
                read_tail.fetch_add(total_size, std::memory_order_relaxed);
                continue;
            }

            // Do not evict CLAIMED (data constructing) or MIGRATING node
            if (state == NodeState::CLAIMED || state == NodeState::MIGRATING) {
                read_tail.fetch_add(total_size, std::memory_order_relaxed);
                continue;
            }

            if (ref_bit == EvictState::COLD) {
                // Cold node: evict rightaway
                uint64_t expected =
                    PackControl(state, ref_bit, length, v_offset);
                const uint64_t desired =
                    PackControl(NodeState::DEAD, ref_bit, length, v_offset);

                node.control_block.compare_exchange_strong(
                    expected, desired, std::memory_order_release,
                    std::memory_order_relaxed);
                node.created_at.store(0, std::memory_order_relaxed);

            } else {
                // Hot node: perform rescuing, giving them a Second Chance
                const uint64_t rescued_offset = AllocatePayload(text_len);
                const uint64_t rescued_index = ActualIndex(rescued_offset);

                const PayloadHeader new_header{VALID_IDENTIFIER, node_id,
                                               text_len};
                std::memcpy(buffer_payload + rescued_index, &new_header,
                            sizeof(PayloadHeader));

                uint64_t src_idx = ActualIndex(tail + sizeof(PayloadHeader));
                uint64_t dst_idx =
                    ActualIndex(rescued_offset + sizeof(PayloadHeader));

                uint64_t bytes_left = text_len;
                while (bytes_left > 0) {
                    const uint64_t src_cont = PAYLOAD_BUFFER_SIZE - src_idx;
                    const uint64_t dst_cont = PAYLOAD_BUFFER_SIZE - dst_idx;
                    const uint64_t chunk =
                        std::min({bytes_left, src_cont, dst_cont});

                    std::memcpy(buffer_payload + dst_idx,
                                buffer_payload + src_idx, chunk);

                    bytes_left -= chunk;
                    src_idx = ActualIndex(src_idx + chunk);
                    dst_idx = ActualIndex(dst_idx + chunk);
                }

                // Give node a second chance (to live)
                uint64_t expected =
                    PackControl(state, EvictState::HOT, length, v_offset);
                const uint64_t desired = PackControl(state, EvictState::COLD,
                                                     length, rescued_offset);

                node.control_block.compare_exchange_strong(
                    expected, desired, std::memory_order_release,
                    std::memory_order_relaxed);
            }

            read_tail.fetch_add(total_size, std::memory_order_relaxed);
        }

    } catch (const std::exception& e) {
        std::cerr << "[Engine] GC WARNING: " << e.what() << "\n";
    }
}

void MemoryArena::SweepStalePending(const uint64_t curr_time) noexcept {
    for (size_t i = 0; i < L0_MAX_SLOTS; ++i) {
        MetaNode&      node = metadata[i];
        const uint64_t ctrl =
            node.control_block.load(std::memory_order_acquire);
        const auto [state, ref_bit, length, offset] = UnpackControl(ctrl);

        if (state != NodeState::PENDING) {
            continue;
        }

        const uint64_t ts = node.created_at.load(std::memory_order_acquire);

        // Unfinished transition from CLAIMED -> PENDING
        // Do not touch then.
        if (ts == 0) {
            continue;
        }

        if (curr_time - ts <= PENDING_LIFESPAN) {
            continue;
        }

        // This is a stale Node, kill it rightaway
        uint64_t       expected = ctrl;
        const uint64_t desired =
            PackControl(NodeState::DEAD, ref_bit, length, offset);
        if (node.control_block.compare_exchange_strong(
                expected, desired, std::memory_order_release,
                std::memory_order_relaxed)) {
            node.created_at.store(0, std::memory_order_relaxed);
        }
    }
}

// ------------------------------------------------------------
// Read / Write payload
// ------------------------------------------------------------

void MemoryArena::ReadPayload(const uint64_t v_offset, const uint32_t length,
                              std::string* out_payload) const {
    if (length == 0) {
        out_payload->clear();
        return;
    }

    out_payload->resize(length);
    const uint64_t text_index = ActualIndex(v_offset + sizeof(PayloadHeader));
    char*          dst = out_payload->data();

    if (PAYLOAD_BUFFER_SIZE - text_index >= length) {
        std::memcpy(dst, buffer_payload + text_index, length);
    } else {
        const size_t chunk1 = PAYLOAD_BUFFER_SIZE - text_index;
        const size_t chunk2 = length - chunk1;
        std::memcpy(dst, buffer_payload + text_index, chunk1);
        std::memcpy(dst, buffer_payload, chunk2);
    }
}

uint64_t MemoryArena::WritePayload(const uint32_t node_id,
                                   const uint8_t* in_payload,
                                   const uint32_t length) {
    const uint64_t header_offset = AllocatePayload(length);
    const uint64_t header_index = ActualIndex(header_offset);

    const PayloadHeader header{VALID_IDENTIFIER, node_id, length};
    std::memcpy(buffer_payload + header_index, &header, sizeof(PayloadHeader));

    const uint64_t text_index =
        ActualIndex(header_index + sizeof(PayloadHeader));

    if (PAYLOAD_BUFFER_SIZE - text_index >= length) {
        std::memcpy(buffer_payload + text_index, in_payload, length);
    } else {
        const size_t chunk1 = PAYLOAD_BUFFER_SIZE - text_index;
        const size_t chunk2 = length - chunk1;
        std::memcpy(buffer_payload + text_index, in_payload, chunk1);
        std::memcpy(buffer_payload, in_payload + chunk1, chunk2);
    }

    return header_offset;
}

// ------------------------------------------------------------
// Private allocator
// ------------------------------------------------------------

uint64_t MemoryArena::AllocatePayload(const uint32_t length) {
    const size_t total_size = sizeof(PayloadHeader) + length;
    uint64_t     curr_write = write_head.load(std::memory_order_relaxed);

    while (true) {
        if (curr_write + total_size -
                read_tail.load(std::memory_order_relaxed) >=
            PAYLOAD_BUFFER_SIZE) {
            throw std::runtime_error("Resource Exhausted");
        }

        const uint64_t actual_index = ActualIndex(curr_write);
        uint64_t       padding = 0;

        if (PAYLOAD_BUFFER_SIZE - actual_index < sizeof(PayloadHeader)) {
            padding = PAYLOAD_BUFFER_SIZE - actual_index;
        }

        const uint64_t allocated_offset = curr_write + padding;
        const uint64_t next_write = allocated_offset + total_size;

        if (write_head.compare_exchange_weak(curr_write, next_write,
                                             std::memory_order_relaxed)) {
            return allocated_offset;
        }
    }
}
