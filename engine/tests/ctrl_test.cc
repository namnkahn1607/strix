// Author: namnkahn1607
//
// Unit tests for PackControl() and UnpackControl() - the bit-manip
// operations that encode/decode the 64-bit control block of MetaNode.

#include <gtest/gtest.h>

#include "meta_node.h"

namespace {

// RoundTrip(): pack and unpack immediately, return the UnpackedControl.
Control RoundTrip(const NodeState state, const EvictState ref_bit,
                  const uint8_t version, const uint32_t length,
                  const uint64_t v_offset) {
    return UnpackControl(
        PackControl(state, ref_bit, version, length, v_offset));
}

}  // namespace

// -----------------------------------------------------------------------------
// ZeroInputProducesZero
// PackControl with all-zero fields must produce a zero word.
// DEAD=0, COLD=0, length=0, offset=0 -> 0x0.
// -----------------------------------------------------------------------------

TEST(PackControlTest, ZeroInputProducesZero) {
    const auto packed =
        PackControl(NodeState::kDead, EvictState::kCold, 0, 0, 0);
    EXPECT_EQ(packed, 0x0ULL);
}

// -----------------------------------------------------------------------------
// RoundTrip_AllStates
// Every NodeState value round-trips correctly through Pack/Unpack.
// Verifies the 3-bit state field does not overflow into adjacent bits.
// -----------------------------------------------------------------------------

TEST(PackControlTest, RoundTrip_AllStates) {
    const NodeState states[] = {
        NodeState::kDead,
        NodeState::kPending,
        NodeState::kReady,
    };
    for (const auto state : states) {
        const auto result = RoundTrip(state, EvictState::kCold, 0, 0, 0);
        EXPECT_EQ(result.state, state)
            << "state=" << static_cast<uint32_t>(state);
    }
}

// -----------------------------------------------------------------------------
// RoundTrip_BothEvictStates
// Both EvictState values (COLD=0, HOT=1) round-trip correctly.
// -----------------------------------------------------------------------------

TEST(PackControlTest, RoundTrip_BothEvictStates) {
    const auto cold = RoundTrip(NodeState::kReady, EvictState::kCold, 0, 0, 0);
    EXPECT_EQ(cold.ref_bit, EvictState::kCold);

    const auto hot = RoundTrip(NodeState::kDead, EvictState::kHot, 0, 0, 0);
    EXPECT_EQ(hot.ref_bit, EvictState::kHot);
}

// -----------------------------------------------------------------------------
// RoundTrip_VersionBoundaries
// Length field is 4 bits: valid range [0, 0xF].
// Tests zero, one, midpoint (0x8), and max.
// -----------------------------------------------------------------------------

TEST(PackControlTest, RoundTrip_VersionBoundaries) {
    const uint8_t versions[] = {0, 1, 0x8, 0xF};
    for (const auto ver : versions) {
        const auto result =
            RoundTrip(NodeState::kDead, EvictState::kCold, ver, 0, 0);
        EXPECT_EQ(result.version, ver) << "version=" << ver;
    }
}

// -----------------------------------------------------------------------------
// RoundTrip_LengthBoundaries
// Length field is 21 bits: valid range [0, 0x1FFFFF].
// Tests zero, one, midpoint (0xFFFFF), and max.
// -----------------------------------------------------------------------------

TEST(PackControlTest, RoundTrip_LengthBoundaries) {
    const uint32_t lengths[] = {0, 1, 0xFFFFF, kMaxPayloadLength};
    for (const auto len : lengths) {
        const auto result =
            RoundTrip(NodeState::kReady, EvictState::kHot, 0, len, 0);
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
    for (const auto offset : offsets) {
        const auto result =
            RoundTrip(NodeState::kPending, EvictState::kCold, 0, 0, offset);
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
    const auto result = RoundTrip(NodeState::kReady,  // 0b10 = max 2-bit state
                                  EvictState::kHot,   // 1
                                  kVersionMask,       // 0xF
                                  kMaxPayloadLength,  // 0xFFFFFF
                                  kVirtualOffsetMask);  // 0xFFFFFFFFF

    EXPECT_EQ(result.state, NodeState::kReady);
    EXPECT_EQ(result.ref_bit, EvictState::kHot);
    EXPECT_EQ(result.version, kVersionMask);
    EXPECT_EQ(result.length, kMaxPayloadLength);
    EXPECT_EQ(result.virtual_offset, kVirtualOffsetMask);
}

// -----------------------------------------------------------------------------
// Isolation_StateDoesNotCorruptOtherFields
// Changing only the state field must leave all other fields unchanged.
// -----------------------------------------------------------------------------

TEST(PackControlTest, Isolation_StateDoesNotCorruptOtherFields) {
    constexpr uint8_t  kVersion = 0xA;
    constexpr uint32_t kLength  = 0xABCDE;
    constexpr uint64_t kOffset  = 0x123456789ULL;

    const NodeState states[] = {
        NodeState::kDead,
        NodeState::kPending,
        NodeState::kReady,
    };
    for (const auto state : states) {
        const auto result =
            RoundTrip(state, EvictState::kHot, kVersion, kLength, kOffset);
        EXPECT_EQ(result.state, state) << "state field";
        EXPECT_EQ(result.ref_bit, EvictState::kHot) << "ref_bit field";
        EXPECT_EQ(result.version, kVersion) << "version field";
        EXPECT_EQ(result.length, kLength) << "length field";
        EXPECT_EQ(result.virtual_offset, kOffset) << "offset field";
    }
}

// -----------------------------------------------------------------------------
// Isolation_RefBitDoesNotCorruptOtherFields
// Toggling the ref_bit must leave all other fields unchanged.
// -----------------------------------------------------------------------------

TEST(PackControlTest, Isolation_RefBitDoesNotCorruptOtherFields) {
    constexpr uint8_t  kVersion = 0xB;
    constexpr uint32_t kLength  = 0x10000;
    constexpr uint64_t kOffset  = 0xABCDEF012ULL;

    for (const auto ev : {EvictState::kCold, EvictState::kHot}) {
        const auto result =
            RoundTrip(NodeState::kPending, ev, kVersion, kLength, kOffset);
        EXPECT_EQ(result.state, NodeState::kPending) << "state field";
        EXPECT_EQ(result.ref_bit, ev) << "ref_bit field";
        EXPECT_EQ(result.version, kVersion) << "version field";
        EXPECT_EQ(result.length, kLength) << "length field";
        EXPECT_EQ(result.virtual_offset, kOffset) << "offset field";
    }
}

// -----------------------------------------------------------------------------
// Isolation_VersionDoesNotCorruptOtherFields
// Sweeping through representative version values must leave all other fields
// unchanged.
// -----------------------------------------------------------------------------

TEST(PackControlTest, Isolation_VersionDoesNotCorruptOtherFields) {
    constexpr uint32_t kLength = 0x12345;
    constexpr uint64_t kOffset = 0x123ABC456ULL;

    const uint8_t versions[] = {0, 1, 0x6, 0x7, 0x8, 0xF};
    for (const auto ver : versions) {
        const auto result = RoundTrip(NodeState::kDead, EvictState::kHot, ver,
                                      kLength, kOffset);
        EXPECT_EQ(result.state, NodeState::kDead) << "state field";
        EXPECT_EQ(result.ref_bit, EvictState::kHot) << "ref_bit field";
        EXPECT_EQ(result.version, ver) << "version field";
        EXPECT_EQ(result.length, kLength) << "length field";
        EXPECT_EQ(result.virtual_offset, kOffset) << "offset field";
    }
}

// -----------------------------------------------------------------------------
// Isolation_LengthDoesNotCorruptOtherFields
// Sweeping through representative length values must leave all other fields
// unchanged.
// -----------------------------------------------------------------------------

TEST(PackControlTest, Isolation_LengthDoesNotCorruptOtherFields) {
    constexpr uint8_t  kVersion = 0xD;
    constexpr uint64_t kOffset  = 0x1FFFFFFF0ULL;

    const uint32_t lengths[] = {0, 1, 0x80000, kMaxPayloadLength};
    for (const uint32_t len : lengths) {
        const auto result = RoundTrip(NodeState::kReady, EvictState::kCold,
                                      kVersion, len, kOffset);
        EXPECT_EQ(result.state, NodeState::kReady) << "state field";
        EXPECT_EQ(result.ref_bit, EvictState::kCold) << "ref_bit field";
        EXPECT_EQ(result.version, kVersion) << "version field";
        EXPECT_EQ(result.length, len) << "length field";
        EXPECT_EQ(result.virtual_offset, kOffset) << "offset field";
    }
}

// -----------------------------------------------------------------------------
// Isolation_OffsetDoesNotCorruptOtherFields
// Sweeping through representative offset values must leave all other fields
// unchanged.
// -----------------------------------------------------------------------------

TEST(PackControlTest, Isolation_OffsetDoesNotCorruptOtherFields) {
    constexpr uint8_t  kVersion = 0xE;
    constexpr uint32_t kLength  = 0xFFFFF;

    const uint64_t offsets[] = {0, 1, 0x800000000ULL, kVirtualOffsetMask};
    for (const uint64_t offset : offsets) {
        const auto result = RoundTrip(NodeState::kReady, EvictState::kHot,
                                      kVersion, kLength, offset);
        EXPECT_EQ(result.state, NodeState::kReady) << "state field";
        EXPECT_EQ(result.ref_bit, EvictState::kHot) << "ref_bit field";
        EXPECT_EQ(result.version, kVersion) << "version field";
        EXPECT_EQ(result.length, kLength) << "length field";
        EXPECT_EQ(result.virtual_offset, offset) << "offset field";
    }
}

// -----------------------------------------------------------------------------
// LengthOverflowIsTruncated
// PackControl masks length to 21 bits. Passing a value larger than
// MAX_PAYLOAD_LENGTH must silently truncate, not corrupt adjacent fields.
// -----------------------------------------------------------------------------

TEST(PackControlTest, LengthOverflowIsTruncated) {
    // 0x3FFFFF = 22 bits set; only low 21 bits (0x1FFFFF) should survive.
    constexpr uint32_t kOverflowLength = 0x3FFFFFU;
    constexpr uint32_t kExpectedLength = 0x1FFFFFU;
    constexpr uint8_t  kVersion        = 0x3;
    constexpr uint64_t kOffset         = 0x100ULL;

    const auto result = RoundTrip(NodeState::kReady, EvictState::kHot, kVersion,
                                  kOverflowLength, kOffset);

    EXPECT_EQ(result.length, kExpectedLength) << "overflow bits must be masked";
    EXPECT_EQ(result.state, NodeState::kReady);
    EXPECT_EQ(result.ref_bit, EvictState::kHot);
    EXPECT_EQ(result.version, kVersion);
    EXPECT_EQ(result.virtual_offset, kOffset) << "offset must not be corrupted";
}

// -----------------------------------------------------------------------------
// OffsetOverflowIsTruncated
// PackControl masks offset to 36 bits. Bits above VIRTUAL_OFFSET_MASK
// must be silently discarded without corrupting adjacent fields.
// -----------------------------------------------------------------------------

TEST(PackControlTest, OffsetOverflowIsTruncated) {
    // Set bits 37-38 (above the 36-bit mask) - should be stripped.
    constexpr uint64_t kOverflowOfs = 0x7FFFFFFFFFULL;     // 39 bits set
    constexpr uint64_t kExpectedOfs = kVirtualOffsetMask;  // 36 bits

    const auto result =
        RoundTrip(NodeState::kPending, EvictState::kCold, 0, 0, kOverflowOfs);

    EXPECT_EQ(result.virtual_offset, kExpectedOfs)
        << "overflow bits must be masked";
    EXPECT_EQ(result.state, NodeState::kPending);
    EXPECT_EQ(result.ref_bit, EvictState::kCold);
    EXPECT_EQ(result.version, 0);
    EXPECT_EQ(result.length, 0U);
}
