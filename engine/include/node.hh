//
// Created by nlnk on Apr 19, 26.
//

#ifndef STRIX_ENGINE_NODE_HH
#define STRIX_ENGINE_NODE_HH

#include <atomic>

#include "constant.hh"

enum class NodeState : uint8_t {
    DEAD = 0,
    PENDING = 1,
    READY = 2,
    MIGRATING = 3
};

enum class EvictState : uint8_t { COLD = 0, HOT = 1 };

struct UnpackedControl {
    NodeState state;
    EvictState ref_bit;
    uint32_t length;
    uint64_t virtual_offset;
};

[[nodiscard]] inline uint64_t PackControl(NodeState state, EvictState ref_bit,
                                          const uint32_t length,
                                          const uint64_t offset) noexcept {
    return (static_cast<uint64_t>(state) << 62) |
           (static_cast<uint64_t>(ref_bit) << 61) |
           (static_cast<uint64_t>(length & engine::MAX_PAYLOAD_LENGTH) << 37) |
           (offset & engine::VIRTUAL_OFFSET_MASK);
}

[[nodiscard]] inline UnpackedControl UnpackControl(
    const uint64_t control) noexcept {
    return {static_cast<NodeState>(control >> 62),
            static_cast<EvictState>((control >> 61) & 0x1),
            static_cast<uint32_t>((control >> 37) & engine::MAX_PAYLOAD_LENGTH),
            control & engine::VIRTUAL_OFFSET_MASK};
}

// Avoid False Sharing using alignas(64) since cache line size for
// almost all modern x86 AMD and Intel CPUs is 64 bytes.
struct alignas(64) MetaNode {
    std::atomic<uint64_t> created_at;

    // Control Block Bit layout
    // Bits 62-63 (2 bits) : State (0=DEAD, 1=PENDING, 2=READY, 3=MIGRATING)
    // Bit  61    (1 bit)  : Ref_bit (0=Cold, 1=Hot)
    // Bits 37-60 (24 bits) : Length (Max 16 MB - Anti-DDoS Limit)
    // Bits 0-36  (37 bits) : Virtual Offset (Max 128 GB - Epoch Tracking)
    std::atomic<uint64_t> control_block;

    [[nodiscard]] UnpackedControl LoadControl(
        const std::memory_order order =
            std::memory_order_acquire) const noexcept {
        return UnpackControl(control_block.load(order));
    }
};

#endif  // STRIX_ENGINE_NODE_HH
