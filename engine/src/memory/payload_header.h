// Payload header struct definition with valid identifier.

#pragma once

#include <cstdint>

// `PayloadHeader` is prepended to every payload in ring buffer.
// Enables constant reverse-lookup from a ring buffer payload to its `MetaNode`.
struct alignas(4) PayloadHeader {
    static constexpr uint32_t kValidIdentifier = 0xDEADBEEFU;

    // Caller-supplied tag; used to verify header integrity.
    const uint32_t identifier;
    // Index of the `MetaNode` that owns this payload.
    const uint32_t node_id;
    // Payload byte length, excluding this header.
    const uint32_t length;
};

// Compile-time assertion to ensure expected memory footprint.
static_assert(sizeof(PayloadHeader) == 12,
              "PayloadHeader should be 12-byte in memory");
