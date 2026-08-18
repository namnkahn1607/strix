// Control block internal structure fields with its encoder/decoder.
//
// Control block bit layout (64 bits):
//   Bits 63–62  (2 bits)  : NodeState
//   Bit  61     (1 bit)   : EvictState  (CLOCK reference bit)
//   Bits 60-57  (4 bits)  : Version     (seqlock-style versioning)
//   Bits 56–36  (21 bits) : Payload length
//   Bits 35–0   (36 bits) : Virtual offset

#pragma once

#include <cstdint>

#include "state.h"

inline constexpr uint32_t kNodeStateShift = 62U;
inline constexpr uint32_t kEvictShift     = 61U;
inline constexpr uint32_t kVersionShift   = 57U;
inline constexpr uint32_t kLengthShift    = 36U;

static_assert(
    kEvictShift + 1 == kNodeStateShift,
    "'evict' bit must sit below the 'state' field"
);
static_assert(
    kVersionShift + 4 == kEvictShift,
    "'version' field must occupy the 4 bits below the 'evict' bit"
);
static_assert(
    kLengthShift + 21 == kVersionShift,
    "'length' field must sit below the 'version' field"
);
static_assert(
    kLengthShift == 36, "'virtual offset' occupies bits [0, 36) unconditionally"
);

inline constexpr uint64_t kNodeStateMask     = 0x3ULL;
inline constexpr uint64_t kEvictStateMask    = 0x1ULL;
inline constexpr uint64_t kVersionMask       = 0xFULL;
inline constexpr uint64_t kVirtualOffsetMask = 0xF'FFFF'FFFFULL;
inline constexpr uint32_t kMaxPayloadLength  = 0x1F'FFFFU;

// NextVersion advances a 4-bit version counter with wrap-around.
//
// Use only after a worker successfully acquired a fresh `node_id` and
// finished committing its new vector data onto the node.
//
// This asymmetry lets a seqlock-style version check: snapshot -> read data ->
// snapshot again, detect ownership change across multiple workers.
inline uint8_t NextVersion(const uint8_t version) noexcept {
    return (version + 1) & kVersionMask;
}

// ControlBlock provides the decoded view of control block of a `MetaNode`.
// Produced by `UnpackControl(uint64_t)`; consumed by any logical worker that
// needs to access the internal state/field of a node.
struct ControlBlock {
    NodeState  state;           // Current lifecycle state of the node.
    EvictState ref;             // Reference bit used by CLOCK algorithm.
    uint8_t    version;         // Seqlock-style version counter.
    uint32_t   length;          // Payload length in bytes (max 2 MB - 1 byte).
    uint64_t   virtual_offset;  // Virtual buffer offset (max 64 GB).
};

// PackControl encodes a `[state, ref, version, length, offset]` tuple into
// a 64-bit control block word.
//
// By default, overflow argument values are truncated to fit their field's
// bit-width (e.g. X bits) by keeping only X LSB(s).
inline uint64_t PackControl(
    const NodeState state, const EvictState ref, const uint8_t version,
    const uint32_t length, const uint64_t offset
) noexcept {
    return (static_cast<uint64_t>(state) << kNodeStateShift) |
           (static_cast<uint64_t>(ref) << kEvictShift) |
           ((static_cast<uint64_t>(version) & kVersionMask) << kVersionShift) |
           (static_cast<uint64_t>(length & kMaxPayloadLength) << kLengthShift) |
           (offset & kVirtualOffsetMask);
}

// UnpackControl decodes a raw 64-bit word into a single `ControlBlock`.
inline ControlBlock UnpackControl(const uint64_t control) noexcept {
    return {
        static_cast<NodeState>((control >> kNodeStateShift) & kNodeStateMask),
        static_cast<EvictState>((control >> kEvictShift) & kEvictStateMask),
        static_cast<uint8_t>((control >> kVersionShift) & kVersionMask),
        static_cast<uint32_t>((control >> kLengthShift) & kMaxPayloadLength),
        control & kVirtualOffsetMask
    };
}
