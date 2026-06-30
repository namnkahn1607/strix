// Author: namnkahn1607
//
// MetaNode layout, NodeState/EvictState enums, UnpackedControl struct,
// and PackControl/UnpackControl helpers.
//
// Control block bit layout (64 bits):
//   Bits 63–61  (3 bits) : NodeState
//   Bit  60     (1 bit)  : EvictState  (CLOCK algorithm reference bit)
//   Bits 59–36  (24 bits): Payload length  (max 16 MB; anti-DDoS cap)
//   Bits 35–0   (36 bits): Virtual offset  (max 64 GB; epoch-safe)

#pragma once

#include <atomic>
#include <cstdint>

// NodeState
//
// Lifecycle state machine for a single arena slot.
//
//   `DEAD`      -> `CLAIMED`  : writer atomically claims a free slot.
//   `CLAIMED`   -> `PENDING`  : vector data fully copied; awaiting payload
//                               to commit.
//   `PENDING`   -> `READY`    : payload committed; slot is searchable.
//   `READY`     -> `DEAD`     : cold node evicted; slot released
//                               back to pool.
//
// GC skips CLAIMED & MIGRATING nodes (data intact; ownership transferred to
// L1). Readers may still serve MIGRATING nodes during the migration window.
enum class NodeState : uint8_t {
    kDead      = 0,
    kClaimed   = 1,
    kPending   = 2,
    kReady     = 3,
};

// EvictState: CLOCK algorithm reference bit.
// HOT  -> node was accessed since the last GC sweep; spare from eviction.
// COLD -> node is a candidate for eviction on the next GC pass.
enum class EvictState : uint8_t {
    kCold = 0,
    kHot  = 1,
};

// UnpackedControl: decoded view of a MetaNode control block.
// Produced by `UnpackControl()`; consumed by any logical worker that interacts
// with control block of a node.
struct UnpackedControl {
    NodeState  state;
    EvictState ref_bit;
    uint32_t   length;
    uint64_t   virtual_offset;
};

// Bit-field masks for PackControl() / UnpackControl().
// kVirtualOffsetMask : 36-bit field -> addressable range up to 64 GB.
// kMaxPayloadLength  : 24-bit field -> maximum payload size of 16 MB.
inline constexpr uint64_t kVirtualOffsetMask = 0xF'FFFF'FFFFULL;
inline constexpr uint32_t kMaxPayloadLength  = 0xFF'FFFFU;

// PackControl() encodes a (`state`, `ref_bit`, `length`, `offset`) tuple into
// the 64-bit control block word. See the bit layout at the top of this file.
inline uint64_t PackControl(const NodeState state, const EvictState ref_bit,
                            const uint32_t length,
                            const uint64_t offset) noexcept {
    return (static_cast<uint64_t>(state) << 61) |
           (static_cast<uint64_t>(ref_bit) << 60) |
           (static_cast<uint64_t>(length & kMaxPayloadLength) << 36) |
           (offset & kVirtualOffsetMask);
}

// UnpackControl() decodes a raw 64-bit control block word into
// an `UnpackedControl` struct.
inline UnpackedControl UnpackControl(const uint64_t control) noexcept {
    return {static_cast<NodeState>(control >> 61),
            static_cast<EvictState>((control >> 60) & 0x1),
            static_cast<uint32_t>((control >> 36) & kMaxPayloadLength),
            control & kVirtualOffsetMask};
}

// MetaNode
//
// Metadata control block for one arena slot.
//
// alignas(64): each `MetaNode` occupies exactly one cache line, preventing
// false sharing between adjacent slots under concurrent access.
// Cache line size is 64 bytes on all modern x86 AMD/Intel microarchitectures.
struct alignas(64) MetaNode {
    // Unix timestamp (seconds) at which this slot was claimed.
    // Used by the GC to identify expired PENDING nodes.
    std::atomic<uint64_t> created_at;

    // Packed control block; encode/decode via PackControl/UnpackControl.
    std::atomic<uint64_t> control_block;

    // Atomically loads and decodes the control block.
    UnpackedControl LoadControl(
        const std::memory_order order = std::memory_order_acquire) const {
        return UnpackControl(control_block.load(order));
    }
};
