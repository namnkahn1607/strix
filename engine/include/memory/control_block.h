// Control block bit layout (64 bits):
//   Bits 63–62  (2 bits)  : NodeState
//   Bit  61     (1 bit)   : EvictState  (CLOCK reference bit)
//   Bits 60-57  (4 bits)  : Version     (seqlock-style versioning)
//   Bits 56–36  (21 bits) : Payload length
//   Bits 35–0   (36 bits) : Virtual offset

#pragma once

#include <cstdint>

#include "state.h"

namespace strix::memory {

inline constexpr uint32_t kNodeStateShift = 62;
inline constexpr uint32_t kEvictShift     = 61;
inline constexpr uint32_t kVersionShift   = 57;
inline constexpr uint32_t kLengthShift    = 36;

inline constexpr uint64_t kNodeStateMask     = 0x3;
inline constexpr uint64_t kEvictStateMask    = 0x1;
inline constexpr uint64_t kVersionMask       = 0xF;
inline constexpr uint32_t kMaxPayloadLength  = 0x1F'FFFFu;
inline constexpr uint64_t kVirtualOffsetMask = 0xF'FFFF'FFFFull;

// Advances a 4-bit version counter with wrap-around.
// Use only after a successful node acquisition and the owner has finished
// writing its new vector data onto that node.
inline uint8_t NextVersion(const uint8_t ver) noexcept {
    // This asymmetry lets a seqlock-style version check (snapshot -> read
    // -> snapshot again) detects ownership change mid-read.
    return (ver + 1) & kVersionMask;
}

// Decoded view of control block.
struct ControlBlock {
    // By default, overflow argument values are truncated to fit their field's
    // bit-width (e.g. X bits) by keeping only X LSB(s).
    static uint64_t Pack(
        NodeState state, EvictState ref, uint8_t ver, uint32_t length,
        uint64_t offset
    ) noexcept {
        auto state64 = static_cast<uint64_t>(state);
        auto ref64   = static_cast<uint64_t>(ref);
        auto ver64   = static_cast<uint64_t>(ver);
        auto len64   = static_cast<uint64_t>(length & kMaxPayloadLength);
        return ((state64 & kNodeStateMask) << kNodeStateShift) |
               ((ref64 & kEvictStateMask) << kEvictShift) |
               ((ver64 & kVersionMask) << kVersionShift) |
               (len64 << kLengthShift) | (offset & kVirtualOffsetMask);
    }

    static ControlBlock Unpack(uint64_t ctrl) noexcept {
        return {
            static_cast<NodeState>((ctrl >> kNodeStateShift) & kNodeStateMask),
            static_cast<EvictState>((ctrl >> kEvictShift) & kEvictStateMask),
            static_cast<uint8_t>((ctrl >> kVersionShift) & kVersionMask),
            static_cast<uint32_t>((ctrl >> kLengthShift) & kMaxPayloadLength),
            ctrl & kVirtualOffsetMask
        };
    }

    NodeState  state;           // Current lifecycle state of the node.
    EvictState ref;             // Reference bit used by CLOCK algorithm.
    uint8_t    version;         // Seqlock-style version counter.
    uint32_t   length;          // Payload length in bytes (max 2 MiB - 1 byte).
    uint64_t   virtual_offset;  // Virtual buffer offset (max 64 GiB).
};

static_assert(
    kEvictShift + 1 == kNodeStateShift,
    "'evict' bit must sit below the 'state' field"
);
static_assert(
    kVersionShift + 4 == kEvictShift,
    "'version' field must occupy the 4-bit below the 'evict' bit"
);
static_assert(
    kLengthShift + 21 == kVersionShift,
    "'length' field must occupy the 21-bit sit below the 'version' field"
);
static_assert(kLengthShift == 36, "'virtual offset' must occupy bits [0, 36)");

static_assert(
    ((kNodeStateMask << kNodeStateShift) | (kEvictStateMask << kEvictShift) |
     (kVersionMask << kVersionShift) |
     (static_cast<uint64_t>(kMaxPayloadLength) << kLengthShift) |
     kVirtualOffsetMask) == ~0ULL,
    "Bit fields must form an exact partition of 64 bits"
);

}  // namespace strix::memory
