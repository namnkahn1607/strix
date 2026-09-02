// Unit tests for 64-bit control block encoder and decoder,
// version advancing and atomic control block accessors.

#include "memory/control_block.h"

#include <gtest/gtest.h>

#include <atomic>

#include "memory/meta_node.h"
#include "memory/state.h"

using namespace strix::memory;

// =============================================================================
// Section 1: Raw-bit anchoring.
//
// Compare the packed 64-bit word against a hand-computed literal derived from
// the documented bit layout.
// Can detect a wrong shift value same way in both encoder and decoder.
// =============================================================================

TEST(RawBitLayoutTest, ZeroInputProducesZero) {
    const auto packed =
        ControlBlock::Pack(NodeState::kDead, EvictState::kCold, 0, 0, 0);
    EXPECT_EQ(packed, 0x0);
}

TEST(RawBitLayoutTest, NodeStateOccupiesBits63To62) {
    const auto packed =
        ControlBlock::Pack(NodeState::kReady, EvictState::kCold, 0, 0, 0);
    EXPECT_EQ(packed, uint64_t{0b10} << 62);
}

TEST(RawBitLayoutTest, EvictOccupiesBit61) {
    const auto packed =
        ControlBlock::Pack(NodeState::kDead, EvictState::kHot, 0, 0, 0);
    EXPECT_EQ(packed, uint64_t{0x1} << 61);
}

TEST(RawBitLayoutTest, VersionOccupiesBits60To57) {
    const auto packed =
        ControlBlock::Pack(NodeState::kDead, EvictState::kCold, 0xF, 0, 0);
    EXPECT_EQ(packed, uint64_t{kVersionMask} << 57);
}

TEST(RawBitLayoutTest, LengthOccupiesBits56To36) {
    const auto packed = ControlBlock::Pack(
        NodeState::kDead, EvictState::kCold, 0, kMaxPayloadLength, 0
    );
    EXPECT_EQ(packed, uint64_t{kMaxPayloadLength} << 36);
}

TEST(RawBitLayoutTest, OffsetOccupiesBits35To0) {
    const auto packed = ControlBlock::Pack(
        NodeState::kDead, EvictState::kCold, 0, 0, kVirtualOffsetMask
    );
    EXPECT_EQ(packed, uint64_t{kVirtualOffsetMask});
}

// =============================================================================
// Section 2: Round-trip correctness per field.
//
// These confirm encode -> decode recovers the original value without creating
// mask errors, sign-extension bugs, and enum cast issues.
// =============================================================================

namespace {

ControlBlock RoundTrip(
    NodeState state, EvictState ref, uint8_t version, uint32_t length,
    uint64_t v_offset
) {
    return ControlBlock::Unpack(
        ControlBlock::Pack(state, ref, version, length, v_offset)
    );
}

}  // namespace

TEST(RoundTripTest, AllNodeStates) {
    const NodeState states[] = {
        NodeState::kDead,
        NodeState::kPending,
        NodeState::kReady,
    };
    for (const auto state : states) {
        const auto result = RoundTrip(state, EvictState::kCold, 0, 0, 0);
        EXPECT_EQ(result.state, state) << "state=" << state;
    }
}

TEST(RoundTripTest, BothEvictStates) {
    const auto cold = RoundTrip(NodeState::kReady, EvictState::kCold, 0, 0, 0);
    EXPECT_EQ(cold.ref, EvictState::kCold);

    const auto hot = RoundTrip(NodeState::kDead, EvictState::kHot, 0, 0, 0);
    EXPECT_EQ(hot.ref, EvictState::kHot);
}

TEST(RoundTripTest, VersionBoundaries) {
    // Version field is 4 bits wide. Valid range: [0, 0xF].
    const uint8_t versions[] = {0, 1, 0x8, 0xF};
    for (const auto ver : versions) {
        const auto result =
            RoundTrip(NodeState::kDead, EvictState::kCold, ver, 0, 0);
        EXPECT_EQ(result.version, ver) << "version=" << ver;
    }
}

TEST(RoundTripTest, LengthBoundaries) {
    // Length field is 21 bits wide. Valid range: [0, 0x1FFFFF].
    const uint32_t lengths[] = {0, 1, 0xFFFFF, kMaxPayloadLength};
    for (const auto len : lengths) {
        const auto result =
            RoundTrip(NodeState::kReady, EvictState::kHot, 0, len, 0);
        EXPECT_EQ(result.length, len) << "length=0x" << std::hex << len;
    }
}

TEST(RoundTripTest, OffsetBoundaries) {
    // Offset field is 36 bits. Valid range: [0, 0xFFFFFFFFF].
    const uint64_t offsets[] = {0, 1, 0x7FFFFFFFFull, kVirtualOffsetMask};
    for (const auto offset : offsets) {
        const auto result =
            RoundTrip(NodeState::kPending, EvictState::kCold, 0, 0, offset);
        EXPECT_EQ(result.virtual_offset, offset)
            << "offset=0x" << std::hex << offset;
    }
}

TEST(RoundTripTest, AllFieldsMaxedSimultaneously) {
    const auto result = RoundTrip(
        NodeState::kReady, EvictState::kHot, kVersionMask, kMaxPayloadLength,
        kVirtualOffsetMask
    );

    EXPECT_EQ(result.state, NodeState::kReady);
    EXPECT_EQ(result.ref, EvictState::kHot);
    EXPECT_EQ(result.version, kVersionMask);
    EXPECT_EQ(result.length, kMaxPayloadLength);
    EXPECT_EQ(result.virtual_offset, kVirtualOffsetMask);
}

// =============================================================================
// Section 3: Field isolation.
//
// Iterates one field across its full value range must never corrupt others.
// Proves that no field bleeds into its adjacent neighbors.
// =============================================================================

TEST(IsolationTest, NodeStateDoesNotCorruptOtherFields) {
    constexpr uint8_t  kVersion = 0xA;
    constexpr uint32_t kLength  = 0xABCDEu;
    constexpr uint64_t kOffset  = 0x123456789ull;

    const NodeState states[] = {
        NodeState::kDead,
        NodeState::kPending,
        NodeState::kReady,
    };
    for (const auto state : states) {
        const auto result =
            RoundTrip(state, EvictState::kHot, kVersion, kLength, kOffset);
        EXPECT_EQ(result.state, state) << "state field";
        EXPECT_EQ(result.ref, EvictState::kHot) << "ref field";
        EXPECT_EQ(result.version, kVersion) << "version field";
        EXPECT_EQ(result.length, kLength) << "length field";
        EXPECT_EQ(result.virtual_offset, kOffset) << "offset field";
    }
}

TEST(IsolationTest, RefBitDoesNotCorruptOtherFields) {
    constexpr uint8_t  kVersion = 0xB;
    constexpr uint32_t kLength  = 0x10000u;
    constexpr uint64_t kOffset  = 0xABCDEF012ull;

    for (const auto ev : {EvictState::kCold, EvictState::kHot}) {
        const auto result =
            RoundTrip(NodeState::kPending, ev, kVersion, kLength, kOffset);
        EXPECT_EQ(result.state, NodeState::kPending) << "state field";
        EXPECT_EQ(result.ref, ev) << "ref field";
        EXPECT_EQ(result.version, kVersion) << "version field";
        EXPECT_EQ(result.length, kLength) << "length field";
        EXPECT_EQ(result.virtual_offset, kOffset) << "offset field";
    }
}

TEST(IsolationTest, VersionDoesNotCorruptOtherFields) {
    constexpr uint32_t kLength = 0x12345u;
    constexpr uint64_t kOffset = 0x123ABC456ull;

    const uint8_t versions[] = {0, 1, 0x6, 0x7, 0x8, 0xF};
    for (const auto ver : versions) {
        const auto result = RoundTrip(
            NodeState::kDead, EvictState::kHot, ver, kLength, kOffset
        );
        EXPECT_EQ(result.state, NodeState::kDead) << "state field";
        EXPECT_EQ(result.ref, EvictState::kHot) << "ref field";
        EXPECT_EQ(result.version, ver) << "version field";
        EXPECT_EQ(result.length, kLength) << "length field";
        EXPECT_EQ(result.virtual_offset, kOffset) << "offset field";
    }
}

TEST(IsolationTest, LengthDoesNotCorruptOtherFields) {
    constexpr uint8_t  kVersion = 0xD;
    constexpr uint64_t kOffset  = 0x1FFFFFFF0ull;

    const uint32_t lengths[] = {0, 1, 0x80000u, kMaxPayloadLength};
    for (const uint32_t len : lengths) {
        const auto result = RoundTrip(
            NodeState::kReady, EvictState::kCold, kVersion, len, kOffset
        );
        EXPECT_EQ(result.state, NodeState::kReady) << "state field";
        EXPECT_EQ(result.ref, EvictState::kCold) << "ref field";
        EXPECT_EQ(result.version, kVersion) << "version field";
        EXPECT_EQ(result.length, len) << "length field";
        EXPECT_EQ(result.virtual_offset, kOffset) << "offset field";
    }
}

TEST(IsolationTest, OffsetDoesNotCorruptOtherFields) {
    constexpr uint8_t  kVersion = 0xE;
    constexpr uint32_t kLength  = 0xFFFFFu;

    const uint64_t offsets[] = {0, 1, 0x800000000ull, kVirtualOffsetMask};
    for (const uint64_t ofs : offsets) {
        const auto result = RoundTrip(
            NodeState::kReady, EvictState::kHot, kVersion, kLength, ofs
        );
        EXPECT_EQ(result.state, NodeState::kReady) << "state field";
        EXPECT_EQ(result.ref, EvictState::kHot) << "ref field";
        EXPECT_EQ(result.version, kVersion) << "version field";
        EXPECT_EQ(result.length, kLength) << "length field";
        EXPECT_EQ(result.virtual_offset, ofs) << "offset field";
    }
}

// =============================================================================
// Section 4: Overflow masking.
//
// Every field with a mask (version, length, offset) must truncate on overflow
// input rather than bleeding into the adjacent field.
//
// NOTE: state and ref are omitted on purpose: state is a 2-bit field fed from
// a 3-value enum (max encoding 0b10) while ref from a 1-bit enum (max 0b1), so
// neither can overflow its own field width.
// =============================================================================

TEST(OverflowTest, VersionOverflowIsTruncated) {
    // 0xFF = 8 bits set; only the low 4 bits (0xF) should survive.
    constexpr uint8_t  kOverflowVersion = 0xFF;
    constexpr uint8_t  kExpectedVersion = 0xF;
    constexpr uint32_t kLength          = 0x555u;
    constexpr uint64_t kOffset          = 0x200ull;

    const auto result = RoundTrip(
        NodeState::kReady, EvictState::kHot, kOverflowVersion, kLength, kOffset
    );

    EXPECT_EQ(result.version, kExpectedVersion)
        << "overflow bits must be masked";
    EXPECT_EQ(result.state, NodeState::kReady);
    EXPECT_EQ(result.ref, EvictState::kHot);
    EXPECT_EQ(result.length, kLength) << "length must not be corrupted";
    EXPECT_EQ(result.virtual_offset, kOffset) << "offset must not be corrupted";
}

TEST(OverflowTest, LengthOverflowIsTruncated) {
    // 0x3FFFFF = 22 bits set; only low 21 bits (0x1FFFFF) should survive.
    constexpr uint32_t kOverflowLength = 0x3FFFFFu;
    constexpr uint32_t kExpectedLength = 0x1FFFFFu;
    constexpr uint8_t  kVersion        = 0x3;
    constexpr uint64_t kOffset         = 0x100ull;

    const auto result = RoundTrip(
        NodeState::kReady, EvictState::kHot, kVersion, kOverflowLength, kOffset
    );

    EXPECT_EQ(result.length, kExpectedLength) << "overflow bits must be masked";
    EXPECT_EQ(result.state, NodeState::kReady);
    EXPECT_EQ(result.ref, EvictState::kHot);
    EXPECT_EQ(result.version, kVersion) << "version must not be corrupted";
    EXPECT_EQ(result.virtual_offset, kOffset) << "offset must not be corrupted";
}

TEST(OverflowTest, OffsetOverflowIsTruncated) {
    // Set bits 37-38 (above the 36-bit mask) - should be stripped.
    constexpr uint64_t kOverflowOfs = 0x7FFFFFFFFFull;     // 39 bits set
    constexpr uint64_t kExpectedOfs = kVirtualOffsetMask;  // 36 bits

    const auto result =
        RoundTrip(NodeState::kPending, EvictState::kCold, 0, 0, kOverflowOfs);

    EXPECT_EQ(result.virtual_offset, kExpectedOfs)
        << "overflow bits must be masked";
    EXPECT_EQ(result.state, NodeState::kPending);
    EXPECT_EQ(result.ref, EvictState::kCold);
    EXPECT_EQ(result.version, 0) << "version must not be corrupted";
    EXPECT_EQ(result.length, 0) << "length must not be corrupted";
}

// =============================================================================
// Section 5: Next version wrap-around.
//
// The wraparound at the 4-bit boundary 0xF must fold back to 0x0 under masking,
// not saturate or overflow silently in any other ways.
// =============================================================================

TEST(NextVersionTest, IncrementNormally) {
    EXPECT_EQ(NextVersion(0x0), 0x1);
    EXPECT_EQ(NextVersion(0x1), 0x2);
    EXPECT_EQ(NextVersion(0xE), 0xF);
}

TEST(NextVersionTest, WrapsAroundAtMax) { EXPECT_EQ(NextVersion(0xF), 0x0); }

TEST(NextVersionTest, IgnoresBitsAboveMask) {
    EXPECT_EQ(NextVersion(0xFF) & ~kVersionMask, 0);
}

// =============================================================================
// Section 6: Control block atomic accessors.
//
// Independent code paths that both extract fields from the same packed word.
// =============================================================================

class MetaNodeAccessorsTest : public ::testing::Test {
protected:
    MetaNode node;
};

TEST_F(MetaNodeAccessorsTest, LoadControlReflectsStoredWord) {
    constexpr uint8_t  kVersion = 0x7;
    constexpr uint32_t kLength  = 0x1234u;
    constexpr uint64_t kOffset  = 0xABCDEull;

    const auto packed = ControlBlock::Pack(
        NodeState::kReady, EvictState::kHot, kVersion, kLength, kOffset
    );
    node.control_block.store(packed, std::memory_order_relaxed);

    const auto [state, ref, version, length, v_offset] = node.LoadControl();
    EXPECT_EQ(state, NodeState::kReady);
    EXPECT_EQ(ref, EvictState::kHot);
    EXPECT_EQ(version, kVersion);
    EXPECT_EQ(length, kLength);
    EXPECT_EQ(v_offset, kOffset);
}

TEST_F(MetaNodeAccessorsTest, LoadVersionAgreesWithLoadControl) {
    for (uint8_t ver = 0; ver <= kVersionMask; ++ver) {
        const auto packed = ControlBlock::Pack(
            NodeState::kPending, EvictState::kCold, ver, 0, 0
        );
        node.control_block.store(packed, std::memory_order_relaxed);

        EXPECT_EQ(node.LoadVersion(), ver);
        EXPECT_EQ(node.LoadControl().version, ver)
            << "LoadVersion/LoadControl disagree at version=" << ver;
    }
}

TEST_F(MetaNodeAccessorsTest, LoadOffsetAgreesWithLoadControl) {
    for (uint64_t ofs = 0x0; ofs <= kVirtualOffsetMask; ++ofs) {
        const auto packed =
            ControlBlock::Pack(NodeState::kReady, EvictState::kHot, 0, 0, ofs);
        node.control_block.store(packed, std::memory_order_relaxed);

        EXPECT_EQ(node.LoadOffset(), ofs);
        EXPECT_EQ(node.LoadControl().virtual_offset, ofs)
            << "LoadOffset/LoadControl disagree at offset=" << ofs;
    }
}
