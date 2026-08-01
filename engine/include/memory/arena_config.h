// Memory Arena runtime configurator declaration.

#pragma once

#include "common/constants.h"

// ArenaConfig defines the specification of `MemoryArena` at construction time.
//
// All validation are performed inside the constructor.
struct ArenaConfig {
    // Total number of node/vector slots.
    // Must be non-zero and a multiple of `kBatchSize`.
    const uint32_t max_slots;

    // Payload ring buffer size in bytes.
    // Pass `0` to omit; payload operations will assert-fail.
    const size_t payload_buf_size;

    // `MAP_POPULATE` enable? (default: `true`).
    const bool prefault = true;

    // Initial value of the ring buffer's write head (defaults: `0`).
    // Non-zero values are used in testing to exercise wrap-around behavior.
    const uint64_t start_point = 0;

    // Production configs `kTotalMaxSlots` slots, 4 GB payload buffer with
    // `MAP_POPULATED` enabled.
    static ArenaConfig Production() {
        constexpr size_t kPayloadBufferSize = 0x100000000ULL;  // 4 GB
        return {kTotalSlots, kPayloadBufferSize};
    }

    // Compact configs dynamic slot capacity, no payload buffer with
    // `MAP_POPULATED` enabled.
    // For throughput benchmarks that need a smaller capacity but page still
    // prefaulted. No notion of "tiers" is provided.
    static ArenaConfig Compact(uint32_t slots) { return {slots, 0}; }

    // CompactLazy configs same as `Compact()` but `MAP_POPULATED` disabled.
    // For unit tests where page prefaulting only adds startup latency.
    static ArenaConfig CompactLazy(uint32_t slots) { return {slots, 0, false}; }
};
