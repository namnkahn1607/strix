// Author: namnkahn1607
//
// MetaNode layout, NodeState/EvictState enums, UnpackedControl struct,
// and PackControl/UnpackControl helpers.
//
// Control block bit layout (64 bits):
//   Bits 63–62  (2 bits)  : NodeState
//   Bit  61     (1 bit)   : EvictState  (CLOCK algorithm reference bit)
//   Bits 60–57  (4 bits)  : Version     (seqlock-style read validation)
//   Bits 56–36  (21 bits) : Payload length  (max 2 MB - 1 byte)
//   Bits 35–0   (36 bits) : Virtual offset  (max 64 GB; epoch-safe)

#pragma once

#include <atomic>
#include <cstdint>

// `NodeState`, lifecycle state machine for a single arena slot.
//
//   - `kDead`    -> `kPending` : a writer holding a `node_id` from `FreeList`
//                                has finished copying vector data and stamped
//                                `created_at`; the slot is now searchable but
//                                has no payload yet.
//   - `kPending` -> `kReady`   : payload committed; slot is fully searchable.
//   - `kReady`   -> `kDead`    : cold node evicted; slot released back to pool.
enum class NodeState : uint8_t {
    kDead    = 0,
    kPending = 1,
    kReady   = 2,
};

// `EvictState`, CLOCK algorithm reference bit.
//
//   - `kHot`  : node was accessed since the last GC sweep; spare from eviction.
//   - `kCold` : node is a candidate for eviction on the next GC pass.
enum class EvictState : uint8_t {
    kCold = 0,
    kHot  = 1,
};

// `UnpackedControl` is decoded view of a MetaNode control block.
// Produced by `UnpackControl()`; consumed by any logical worker that interacts
// with control block of a node.
struct Control {
    NodeState  state;
    EvictState ref_bit;
    uint8_t    version;
    uint32_t   length;
    uint64_t   virtual_offset;
};

inline constexpr uint32_t kNodeStateShift = 62;
inline constexpr uint32_t kVersionShift   = 58;
inline constexpr uint32_t kEvictShift     = 57;
inline constexpr uint32_t kLengthShift    = 36;

inline constexpr uint64_t kNodeStateMask     = 0x3ULL;
inline constexpr uint64_t kVersionMask       = 0xFULL;
inline constexpr uint64_t kEvictMask         = 0x1ULL;
inline constexpr uint64_t kVirtualOffsetMask = 0xF'FFFF'FFFFULL;
inline constexpr uint32_t kMaxPayloadLength  = 0x1F'FFFFU;

// Assertions to ensure bit layout are configured correctly.
static_assert(kVersionShift + 4 == kNodeStateShift,
              "'version' field must occupy the 4 bits directly below 'state'");
static_assert(kEvictShift + 1 == kVersionShift,
              "'evict' bit must sit directly below the version 'field'");
static_assert(kLengthShift + 21 == kEvictShift,
              "'length' field must sit directly below the 'evict' bit");
static_assert(kLengthShift == 36,
              "'virtual offset' occupies bits [0, 36) unconditionally");

// `PackControl()` encodes a (`state`, `ref_bit`, `version`, `length`, `offset`)
// tuple into a 64-bit control block word.
inline uint64_t PackControl(const NodeState state, const EvictState ref_bit,
                            const uint8_t version, const uint32_t length,
                            const uint64_t offset) noexcept {
    return (static_cast<uint64_t>(state) << kNodeStateShift) |
           ((static_cast<uint64_t>(version) & kVersionMask) << kVersionShift) |
           (static_cast<uint64_t>(ref_bit) << kEvictShift) |
           (static_cast<uint64_t>(length & kMaxPayloadLength) << kLengthShift) |
           (offset & kVirtualOffsetMask);
}

// `UnpackControl()` decodes a raw 64-bit control block word into
// an `UnpackedControl` struct.
inline Control UnpackControl(const uint64_t control) noexcept {
    return {
        static_cast<NodeState>((control >> kNodeStateShift) & kNodeStateMask),
        static_cast<EvictState>((control >> kEvictShift) & kEvictMask),
        static_cast<uint8_t>((control >> kVersionShift) & kVersionMask),
        static_cast<uint32_t>((control >> kLengthShift) & kMaxPayloadLength),
        control & kVirtualOffsetMask};
}

// `NextVersion()` advances a 4-bit version counter with wraparound.
//
// Used only by the MISS-ed search path that acquires a fresh `node_id` from
// the `FreeList` and commits its vector data. GC releasing a node back to
// the `FreeList` must pass the version through unchanged.
//
// This asymmetry is what lets a seqlock-style read (snapshot version, read
// vector, snapshot version again) detect "someone acquired this slot during my
// read" without also tripping on ordinary eviction traffic.
inline uint8_t NextVersion(const uint8_t version) noexcept {
    return (version + 1) & kVersionMask;
}

// `MetaNode`, metadata control block for one arena slot.
//
// `alignas(64)`: each `MetaNode` occupies exactly one cache line, preventing
// False Sharing between adjacent slots under concurrent access.
struct alignas(64) MetaNode {
    // Unix timestamp (seconds) at which this slot was claimed.
    // Used by the GC to identify expired `kPending` nodes.
    // Always written by the acquiring writer before that writer publishes, so
    // any reader that observes `kPending` is guaranteed to see a valid,
    // freshly-written `created_at` - never a stale value left over from this
    // slot's previous occupant.
    std::atomic<uint64_t> created_at;

    // Packed control block; `PackControl()` / `UnpackControl()` is used to
    // encode/decode it.
    std::atomic<uint64_t> control_block;

    // `LoadControl()` atomically loads and decodes the control block.
    Control LoadControl(const std::memory_order order =
                            std::memory_order_acquire) const noexcept {
        return UnpackControl(control_block.load(order));
    }

    // `LoadVersion()` extracts only the version field without decoding the
    // rest of the control word.
    // Used on both sides of a seqlock-style vector read (snapshot before,
    // snapshot after) so the hot search path pays for one shift-and-mask
    // instead of a full `UnpackControl()` on each snapshot.
    uint8_t LoadVersion(const std::memory_order order =
                            std::memory_order_acquire) const noexcept {
        return static_cast<uint8_t>(
            (control_block.load(order) >> kVersionShift) & kVersionMask);
    }
};
