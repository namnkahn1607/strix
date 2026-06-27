// Author: namnkahn1607
//
// Unit tests for PackControl() and UnpackControl() - the bit-manip
// operations that encode/decode the 64-bit control block of MetaNode.

#include <gtest/gtest.h>

#include "meta_node.h"

namespace {

// RoundTrip(): pack and unpack immediately, return the UnpackedControl.
UnpackedControl RoundTrip(const NodeState state, const EvictState ref_bit,
                          const uint32_t length, const uint64_t offset) {
    return UnpackControl(PackControl(state, ref_bit, length, offset));
}

}  // namespace

// -----------------------------------------------------------------------------
// ZeroInputProducesZero
// PackControl with all-zero fields must produce a zero word.
// DEAD=0, COLD=0, length=0, offset=0 -> 0x0.
// -----------------------------------------------------------------------------

TEST(PackControlTest, ZeroInputProducesZero) {
    const uint64_t packed =
        PackControl(NodeState::kDead, EvictState::kCold, 0, 0);
    EXPECT_EQ(packed, 0x0ULL);
}

// -----------------------------------------------------------------------------
// RoundTrip_AllStates
// Every NodeState value round-trips correctly through Pack/Unpack.
// Verifies the 3-bit state field does not overflow into adjacent bits.
// -----------------------------------------------------------------------------

TEST(PackControlTest, RoundTrip_AllStates) {
    const NodeState states[] = {
        NodeState::kDead,  NodeState::kClaimed,   NodeState::kPending,
        NodeState::kReady, NodeState::kMigrating,
    };
    for (const NodeState state : states) {
        const auto result = RoundTrip(state, EvictState::kCold, 0, 0);
        EXPECT_EQ(result.state, state)
            << "state=" << static_cast<uint32_t>(state);
    }
}

// -----------------------------------------------------------------------------
// RoundTrip_BothEvictStates
// Both EvictState values (COLD=0, HOT=1) round-trip correctly.
// -----------------------------------------------------------------------------

TEST(PackControlTest, RoundTrip_BothEvictStates) {
    const auto cold = RoundTrip(NodeState::kReady, EvictState::kCold, 0, 0);
    EXPECT_EQ(cold.ref_bit, EvictState::kCold);

    const auto hot = RoundTrip(NodeState::kDead, EvictState::kHot, 0, 0);
    EXPECT_EQ(hot.ref_bit, EvictState::kHot);
}

// -----------------------------------------------------------------------------
// RoundTrip_LengthBoundaries
// Length field is 24 bits: valid range [0, 0xFFFFFF].
// Tests zero, one, midpoint (0x7FFFFF), and max.
// -----------------------------------------------------------------------------

TEST(PackControlTest, RoundTrip_LengthBoundaries) {
    const uint32_t lengths[] = {0, 1, 0x7FFFFF, kMaxPayloadLength};
    for (const uint32_t len : lengths) {
        const auto result =
            RoundTrip(NodeState::kClaimed, EvictState::kHot, len, 0);
        EXPECT_EQ(result.length, len) << "length=0x" << std::hex << len;
    }
}

// -----------------------------------------------------------------------------
// RoundTrip_OffsetBoundaries
// Offset field is 36 bits: valid range [0, 0xFFFFFFFFF].
// Tests zero, one, midpoint (0x7FFFFFFFF), and max.
// -----------------------------------------------------------------------------

TEST(PackControlTest, RoundTrip_OffsetBoundaries) {
    const uint64_t offsets[] = {0, 1, 0x7FFFFFFFFULL, kVirtualOffsetMask};
    for (const uint64_t offset : offsets) {
        const auto result =
            RoundTrip(NodeState::kPending, EvictState::kCold, 0, offset);
        EXPECT_EQ(result.virtual_offset, offset)
            << "offset=0x" << std::hex << offset;
    }
}

// -----------------------------------------------------------------------------
// RoundTrip_AllFieldsMaxed
// All fields at maximum simultaneously. Verifies no field bleeds into another
// when all bits are set.
// -----------------------------------------------------------------------------
TEST(PackControlTest, RoundTrip_AllFieldsMaxed) {
    const auto result =
        RoundTrip(NodeState::kMigrating,  // 0b100 = max 3-bit state
                  EvictState::kHot,       // 1
                  kMaxPayloadLength,      // 0xFFFFFF
                  kVirtualOffsetMask);    // 0xFFFFFFFFF

    EXPECT_EQ(result.state, NodeState::kMigrating);
    EXPECT_EQ(result.ref_bit, EvictState::kHot);
    EXPECT_EQ(result.length, kMaxPayloadLength);
    EXPECT_EQ(result.virtual_offset, kVirtualOffsetMask);
}

// -----------------------------------------------------------------------------
// Isolation_StateDoesNotCorruptOtherFields
// Changing only the state field must leave all other fields unchanged.
// -----------------------------------------------------------------------------
TEST(PackControlTest, Isolation_StateDoesNotCorruptOtherFields) {
    constexpr uint32_t LENGTH = 0xABCDEF;
    constexpr uint64_t OFFSET = 0x123456789ULL;

    const NodeState states[] = {
        NodeState::kDead,  NodeState::kClaimed,   NodeState::kPending,
        NodeState::kReady, NodeState::kMigrating,
    };
    for (const NodeState state : states) {
        const auto result = RoundTrip(state, EvictState::kHot, LENGTH, OFFSET);
        EXPECT_EQ(result.state, state) << "state field";
        EXPECT_EQ(result.ref_bit, EvictState::kHot) << "ref_bit field";
        EXPECT_EQ(result.length, LENGTH) << "length field";
        EXPECT_EQ(result.virtual_offset, OFFSET) << "offset field";
    }
}

// -----------------------------------------------------------------------------
// Isolation_RefBitDoesNotCorruptOtherFields
// Toggling the ref_bit must leave state, length, and offset unchanged.
// -----------------------------------------------------------------------------
TEST(PackControlTest, Isolation_RefBitDoesNotCorruptOtherFields) {
    constexpr uint32_t LENGTH = 0x100000;
    constexpr uint64_t OFFSET = 0xABCDEF012ULL;

    for (const EvictState ev : {EvictState::kCold, EvictState::kHot}) {
        const auto result = RoundTrip(NodeState::kPending, ev, LENGTH, OFFSET);
        EXPECT_EQ(result.state, NodeState::kPending) << "state field";
        EXPECT_EQ(result.ref_bit, ev) << "ref_bit field";
        EXPECT_EQ(result.length, LENGTH) << "length field";
        EXPECT_EQ(result.virtual_offset, OFFSET) << "offset field";
    }
}

// -----------------------------------------------------------------------------
// Isolation_LengthDoesNotCorruptOtherFields
// Sweeping through representative length values must not disturb state,
// ref_bit, or offset.
// -----------------------------------------------------------------------------
TEST(PackControlTest, Isolation_LengthDoesNotCorruptOtherFields) {
    constexpr uint64_t OFFSET = 0x1FFFFFFF0ULL;

    const uint32_t lengths[] = {0, 1, 0x800000, kMaxPayloadLength};
    for (const uint32_t len : lengths) {
        const auto result =
            RoundTrip(NodeState::kReady, EvictState::kCold, len, OFFSET);
        EXPECT_EQ(result.state, NodeState::kReady) << "state field";
        EXPECT_EQ(result.ref_bit, EvictState::kCold) << "ref_bit field";
        EXPECT_EQ(result.length, len) << "length field";
        EXPECT_EQ(result.virtual_offset, OFFSET) << "offset field";
    }
}

// -----------------------------------------------------------------------------
// Isolation_OffsetDoesNotCorruptOtherFields
// Sweeping through representative offset values must not disturb state,
// ref_bit, or length.
// -----------------------------------------------------------------------------
TEST(PackControlTest, Isolation_OffsetDoesNotCorruptOtherFields) {
    constexpr uint32_t LENGTH = 0xFFFFFF;

    const uint64_t offsets[] = {0, 1, 0x800000000ULL, kVirtualOffsetMask};
    for (const uint64_t offset : offsets) {
        const auto result =
            RoundTrip(NodeState::kMigrating, EvictState::kHot, LENGTH, offset);
        EXPECT_EQ(result.state, NodeState::kMigrating) << "state field";
        EXPECT_EQ(result.ref_bit, EvictState::kHot) << "ref_bit field";
        EXPECT_EQ(result.length, LENGTH) << "length field";
        EXPECT_EQ(result.virtual_offset, offset) << "offset field";
    }
}

// -----------------------------------------------------------------------------
// LengthOverflowIsTruncated
// PackControl masks length to 24 bits. Passing a value larger than
// MAX_PAYLOAD_LENGTH must silently truncate, not corrupt adjacent fields.
// -----------------------------------------------------------------------------
TEST(PackControlTest, LengthOverflowIsTruncated) {
    // 0x1FFFFFF = 25 bits set; only low 24 bits (0xFFFFFF) should survive.
    constexpr uint32_t OVERFLOW_LEN = 0x1FFFFFFU;
    constexpr uint32_t EXPECTED_LEN = 0xFFFFFFU;
    constexpr uint64_t kOffset      = 0x100ULL;

    const auto result =
        RoundTrip(NodeState::kReady, EvictState::kHot, OVERFLOW_LEN, kOffset);

    EXPECT_EQ(result.length, EXPECTED_LEN) << "overflow bits must be masked";
    EXPECT_EQ(result.state, NodeState::kReady);
    EXPECT_EQ(result.ref_bit, EvictState::kHot);
    EXPECT_EQ(result.virtual_offset, kOffset) << "offset must not be corrupted";
}

// -----------------------------------------------------------------------------
// OffsetOverflowIsTruncated
// PackControl masks offset to 36 bits. Bits above VIRTUAL_OFFSET_MASK
// must be silently discarded without corrupting adjacent fields.
// -----------------------------------------------------------------------------
TEST(PackControlTest, OffsetOverflowIsTruncated) {
    // Set bits 37-38 (above the 36-bit mask) - should be stripped.
    constexpr uint64_t OVERFLOW_OFS = 0x7FFFFFFFFFULL;     // 39 bits set
    constexpr uint64_t EXPECTED_OFS = kVirtualOffsetMask;  // 36 bits

    const auto result =
        RoundTrip(NodeState::kPending, EvictState::kCold, 0, OVERFLOW_OFS);

    EXPECT_EQ(result.virtual_offset, EXPECTED_OFS)
        << "overflow bits must be masked";
    EXPECT_EQ(result.state, NodeState::kPending);
    EXPECT_EQ(result.ref_bit, EvictState::kCold);
    EXPECT_EQ(result.length, 0U);
}

// -----------------------------------------------------------------------------
// KnownBitPattern
// Verifies a manually computed packed value to catch any shift constant
// regression. If Pack/Unpack shifts change, this test fails immediately.
//
// Input:
//   state   = READY     = 0b011 -> bits 63-61 = 0x1800000000000000
//   ref_bit = HOT       = 0b1   -> bit  60    = 0x1000000000000000
//   length  = 0x000001  = 1     -> bits 59-36 = 0x0000010000000000
//   offset  = 0x000001  = 1     -> bits 35-0  = 0x0000000000000001
//
// Calculation:
//   READY    = 3 = 0b011 → (3ULL << 61) = 0x6000000000000000
//   HOT      = 1         → (1ULL << 60) = 0x1000000000000000
//   length=1             → (1ULL << 36) = 0x0000001000000000
//   offset=1             →  1ULL        = 0x0000000000000001
//
// Result:
//   packed = 0x6000000000000000 | 0x1000000000000000
//          | 0x0000001000000000 | 0x0000000000000001
//          = 0x7000001000000001
// -----------------------------------------------------------------------------
TEST(PackControlTest, KnownBitPattern) {
    constexpr uint64_t EXPECTED_PACKED = 0x7000001000000001ULL;

    const uint64_t packed =
        PackControl(NodeState::kReady, EvictState::kHot, 1, 1);

    EXPECT_EQ(packed, EXPECTED_PACKED)
        << "bit pattern regression: shift constants may have changed";

    const auto result = UnpackControl(packed);
    EXPECT_EQ(result.state, NodeState::kReady);
    EXPECT_EQ(result.ref_bit, EvictState::kHot);
    EXPECT_EQ(result.length, 1U);
    EXPECT_EQ(result.virtual_offset, 1ULL);
}
