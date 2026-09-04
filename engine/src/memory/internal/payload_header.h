#pragma once

#include <cstdint>

namespace strix::memory {

// Header prepended to every payload in buffer.
// Enables constant reverse-lookup from a payload buffer to its `MetaNode`.
struct alignas(4) PayloadHeader {
    static constexpr uint32_t kValidIdentifier = 0xDEADBEEFu;

    // Caller-supplied tag; used to verify header integrity.
    const uint32_t identifier;

    // `MetaNode` ID that owns this payload.
    const uint32_t node_id;

    // Payload length in bytes, excluding this header.
    const uint32_t length;
};

static_assert(
    sizeof(PayloadHeader) == 12, "PayloadHeader must be 12-byte size in memory"
);

}  // namespace strix::memory
