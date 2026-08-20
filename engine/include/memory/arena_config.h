// Memory Arena runtime configuration.

#pragma once

// Total number of slots in the unified `MemoryArena` (all tiers combined).
inline constexpr uint32_t kTotalSlots = 1 << 19;

// ArenaConfig defines the specification of `MemoryArena` at construction time.
//
// All validation are performed inside `MemoryArena` constructor.
struct ArenaConfig {
    // Total number of node/vector slots.
    // Must be non-zero and a multiple of `kBatchSize`.
    const uint32_t max_slots;

    // Payload ring buffer size in bytes.
    // Pass `0` to omit; payload operations will assert-fail.
    const size_t payload_buf_size;

    // Page prefault enabled? (default: `true`).
    const bool prefault = true;

    // Initial value of the ring buffer's write head (defaults: `0`).
    // Non-zero values are used in testing to exercise wrap-around behavior.
    const uint64_t start_point = 0;

    // Configures `kTotalMaxSlots` slots, 4 GB payload buffer with
    // page prefault enabled.
    static ArenaConfig Production() {
        constexpr size_t kPayloadBufferSize = 0x100000000ULL;  // 4 GB
        return {kTotalSlots, kPayloadBufferSize};
    }

    // Configures dynamic slot capacity, no payload buffer with
    // page prefault enabled.
    // For throughput benchmarks that need a smaller capacity but prefaulted
    // pages. No notion of "tiers" is provided.
    static ArenaConfig Compact(uint32_t slots) { return {slots, 0}; }

    // Configures same as `Compact()` but page prefault disabled.
    // For unit tests where page prefaulting only adds startup latency.
    static ArenaConfig CompactLazy(uint32_t slots) { return {slots, 0, false}; }
};
