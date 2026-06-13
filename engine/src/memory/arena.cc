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

// Default size of the Payload ring buffer
inline constexpr uint64_t PAYLOAD_BUFFER_SIZE =
    4ULL * 1024 * 1024 * 1024;  // 4GB

inline constexpr uint64_t LOW_WATERMARK = 2ULL * 1024 * 1024 * 1024;  // 2GB
inline constexpr uint64_t HIGH_WATERMARK =
    (3ULL * 1024 + 512ULL) * 1024 * 1024;  // 3.5GB

// Garbage collector LOW sleeping interval in millisecond(s)
inline constexpr uint32_t LOW_GC_SLEEP = 10;
// Garbage collector HIGH sleeping interval in millisecond(s)
inline constexpr uint32_t HIGH_GC_SLEEP = 1;

// How often a stale sweeper runs in millsecond(s)
inline constexpr uint32_t SWEEP_INTERVAL = 5'000;

// Allocates an anonymous (MAP_ANONYMOUS) private (MAP_PRIVATE) region,
// MAP_POPULATE instructs the kernel to pre-faults all pages inside the syscall.
void* MmapPopulate(const size_t size) {
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_ANONYMOUS | MAP_PRIVATE | MAP_POPULATE, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed for MemoryArena");
    }

    return ptr;
}

// Allocates without MAP_POPULATE - pages fault lazily on first access.
// Used for test configs where pre-faulting the full buffer is unnecessary.
void* MmapLazy(const size_t size) {
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed for MemoryArena");
    }

    return ptr;
}

}  // namespace

// ---------------------------------------------------------------------------
// ArenaConfig Factory methods
// ---------------------------------------------------------------------------

ArenaConfig ArenaConfig::Production() {
    return {TOTAL_MAX_SLOTS, PAYLOAD_BUFFER_SIZE, false};
}

ArenaConfig ArenaConfig::BenchSearchL0() {
    return {1'024, 0, false};
}

ArenaConfig ArenaConfig::TestSearchL0() {
    return {1'024, 0, true};
}

// ------------------------------------------------------------
// Constructor / Destructor
// ------------------------------------------------------------

MemoryArena::MemoryArena(const ArenaConfig& config)
    : max_slots(config.max_slots)
    , payload_buf_size(config.payload_buf_size)
    , write_head(0)
    , read_tail(0) {
    // --- Configuration validating ---
    if (!(max_slots != 0 && max_slots % 4 == 0)) {
        throw std::invalid_argument(
            "Arena slots must be non-zero and multiple of 4");
    }

    if (payload_buf_size > 0 &&
        (payload_buf_size & (payload_buf_size - 1)) != 0) {
        throw std::invalid_argument("Payload buffer size must be a power of 2");
    }

    auto mmap_fn = config.lazy_mapping ? MmapLazy : MmapPopulate;

    // --- Arena allocation ---
    metadata = static_cast<MetaNode*>(mmap_fn(max_slots * sizeof(MetaNode)));
    vectors = static_cast<float*>(
        mmap_fn(max_slots * VECTOR_DIM_ARENA * sizeof(float)));

    if (payload_buf_size > 0) {
        payload_buf = static_cast<uint8_t*>(mmap_fn(payload_buf_size));
    } else {
        payload_buf = nullptr;
    }

    std::cout << "[Vector Engine] Initialized Memory Arena (" << max_slots
              << " slots, " << payload_buf_size / (1024 * 1024) << "MB,"
              << " pre-fault=" << (config.lazy_mapping ? "false" : "true")
              << ")\n";
}

MemoryArena::~MemoryArena() {
    munmap(metadata, max_slots * sizeof(MetaNode));
    munmap(vectors, max_slots * sizeof(float));

    if (payload_buf != nullptr) {
        munmap(payload_buf, payload_buf_size);
    }
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
            if (const uint64_t padding = payload_buf_size - tail_index;
                padding < sizeof(PayloadHeader)) {
                read_tail.fetch_add(padding, std::memory_order_relaxed);
                continue;
            }

            const auto* header = reinterpret_cast<const PayloadHeader*>(
                payload_buf + tail_index);

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
                std::memcpy(payload_buf + rescued_index, &new_header,
                            sizeof(PayloadHeader));

                uint64_t src_idx = ActualIndex(tail + sizeof(PayloadHeader));
                uint64_t dst_idx =
                    ActualIndex(rescued_offset + sizeof(PayloadHeader));

                uint64_t bytes_left = text_len;
                while (bytes_left > 0) {
                    const uint64_t src_cont = payload_buf_size - src_idx;
                    const uint64_t dst_cont = payload_buf_size - dst_idx;
                    const uint64_t chunk =
                        std::min({bytes_left, src_cont, dst_cont});

                    std::memcpy(payload_buf + dst_idx, payload_buf + src_idx,
                                chunk);

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

    if (payload_buf_size - text_index >= length) {
        std::memcpy(dst, payload_buf + text_index, length);
    } else {
        const size_t chunk1 = payload_buf_size - text_index;
        const size_t chunk2 = length - chunk1;
        std::memcpy(dst, payload_buf + text_index, chunk1);
        std::memcpy(dst, payload_buf, chunk2);
    }
}

uint64_t MemoryArena::WritePayload(const uint32_t node_id,
                                   const uint8_t* in_payload,
                                   const uint32_t length) {
    const uint64_t header_offset = AllocatePayload(length);
    const uint64_t header_index = ActualIndex(header_offset);

    const PayloadHeader header{VALID_IDENTIFIER, node_id, length};
    std::memcpy(payload_buf + header_index, &header, sizeof(PayloadHeader));

    const uint64_t text_index =
        ActualIndex(header_index + sizeof(PayloadHeader));

    if (payload_buf_size - text_index >= length) {
        std::memcpy(payload_buf + text_index, in_payload, length);
    } else {
        const size_t chunk1 = payload_buf_size - text_index;
        const size_t chunk2 = length - chunk1;
        std::memcpy(payload_buf + text_index, in_payload, chunk1);
        std::memcpy(payload_buf, in_payload + chunk1, chunk2);
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
            payload_buf_size) {
            throw std::runtime_error("Resource Exhausted");
        }

        const uint64_t actual_index = ActualIndex(curr_write);
        uint64_t       padding = 0;

        if (payload_buf_size - actual_index < sizeof(PayloadHeader)) {
            padding = payload_buf_size - actual_index;
        }

        const uint64_t allocated_offset = curr_write + padding;
        const uint64_t next_write = allocated_offset + total_size;

        if (write_head.compare_exchange_weak(curr_write, next_write,
                                             std::memory_order_relaxed)) {
            return allocated_offset;
        }
    }
}
