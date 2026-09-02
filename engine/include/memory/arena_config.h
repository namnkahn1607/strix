#pragma once

namespace strix::memory {

// Construction-time specification of `MemoryArena`.
struct ArenaConfig {
    // Total number of node/vector slots.
    // Non-zero and a multiple of dot product kernel batch size.
    const uint32_t max_slots;

    // Payload ring buffer size in bytes.
    // Set `0` to omit; payload operations will assert-fail.
    const size_t payload_buf_size;

    // Page prefault option.
    // Default: `true`.
    const bool prefault = true;

    // Initial position of the payload buffer's write head.
    // Default: `0`.
    const uint64_t start_point = 0;

    static ArenaConfig Standard() {
        constexpr uint32_t kTotalSlots        = 1 << 19;         // 524'288
        constexpr size_t   kPayloadBufferSize = 0x100000000ull;  // 4 GiB
        return {kTotalSlots, kPayloadBufferSize};
    }
};

}  // namespace strix::memory
