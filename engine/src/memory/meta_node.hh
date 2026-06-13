//
// memory/meta_node.hh
//
// MetaNode layout, state machine and control block pack/unpack logic
//
// Control Block Bit Layout (64 bits):
//   Bits 63-61 (3 bits) : NodeState
//   Bit  60    (1 bit)  : EvictState (CLOCK ref bit)
//   Bits 59-36 (24 bits): Payload length   (max 16 MB, anti-DDoS)
//   Bits 35-0  (36 bits): Virtual offset   (max 64 GB, epoch tracking)
//

#pragma once

#include <atomic>
#include <cstdint>

// State machine for Metadata Node:
//
// DEAD      : Slot is free, available for claiming.
// CLAIMED   : A writer is currently copying vector data into the slot.
// PENDING   : Vector data is fully written. Waiting to commit
//             the payload and transition to READY.
// READY     : Fully committed. Searchable and readable.
// MIGRATING : A compaction operation has claimed this node for L0 -> L1
//             migration. Readers may still serve it (data intact). GC skips.
enum class NodeState {
    DEAD = 0,
    CLAIMED = 1,
    PENDING = 2,
    READY = 3,
    MIGRATING = 4,
};

enum class EvictState : uint8_t {
    COLD = 0,
    HOT = 1,
};

struct UnpackedControl {
    NodeState  state;
    EvictState ref_bit;
    uint32_t   length;
    uint64_t   virtual_offset;
};

// Virtual offset mask: 36 bit -> max 64GB
inline constexpr uint64_t VIRTUAL_OFFSET_MASK = 0xFFFFFFFFFULL;

// Payload length mask: 24 bit -> max 16MB
inline constexpr uint32_t MAX_PAYLOAD_LENGTH = 0xFFFFFFU;

inline uint64_t PackControl(const NodeState state, const EvictState ref_bit,
                            const uint32_t length,
                            const uint64_t offset) noexcept {
    return (static_cast<uint64_t>(state) << 61) |
           (static_cast<uint64_t>(ref_bit) << 60) |
           (static_cast<uint64_t>(length & MAX_PAYLOAD_LENGTH) << 36) |
           (offset & VIRTUAL_OFFSET_MASK);
}

inline UnpackedControl UnpackControl(const uint64_t control) noexcept {
    return {static_cast<NodeState>(control >> 61),
            static_cast<EvictState>((control >> 60) & 0x1),
            static_cast<uint32_t>((control >> 36) & MAX_PAYLOAD_LENGTH),
            control & VIRTUAL_OFFSET_MASK};
}

// alignas(64): prevent false sharing across cache lines.
// Cache line size is 64 bytes on all modern x86 AMD/Intel CPUs.
struct alignas(64) MetaNode {
    std::atomic<uint64_t> created_at;
    std::atomic<uint64_t> control_block;

    UnpackedControl LoadControl(
        const std::memory_order order = std::memory_order_acquire) {
        return UnpackControl(control_block.load(order));
    }
};
