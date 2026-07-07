// Author: namnkahn1607
//
// Unit tests for MemoryArena::WritePayload() and ReadPayload().
//
// These tests target the ring buffer chunking logic - specifically the 3
// distinct code paths in WritePayload and 2 in ReadPayload:
//
//   WritePayload paths:
//     [1] Sequential : header + data fit in one contiguous region.
//     [2] Data wrap  : header fits, but data straddles the buffer boundary
//                      (split into two memcpy calls).
//     [3] Header wrap: fewer than sizeof(PayloadHeader)=12 bytes remain before
//                      the boundary - the system is expected to insert padding
//                      and places the entire header at index 0.
//
//   ReadPayload paths:
//     [1] Sequential : data fits in one contiguous region.
//     [2] Wrap       : data straddles the boundary (two memcpy calls).
//
// All configs use lazy_mapping = true (no MAP_POPULATE).

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "arena.h"

namespace {

inline constexpr size_t kSlots      = 4;
inline constexpr size_t kBuf        = 256;                    // 256 bytes
inline constexpr size_t kHeaderSize = sizeof(PayloadHeader);  // 12 bytes
inline constexpr size_t kNode       = 0;

// PayloadTestConfig(): 4 slots, 256 bytes payload buffer, lazy mapping.
ArenaConfig PayloadTestConfig(const uint64_t start_point = 0) {
    return ArenaConfig{kSlots, kBuf, true, start_point};
}

// GenPayload(): generates a deterministic payload of specified byte-size.
std::string GenPayload(const size_t len) {
    std::string str(len, '\0');
    for (size_t i = 0; i < len; ++i) {
        // Prime generators, avoid repetition
        str[i] = static_cast<char>((i * 37 + 13) % 251);
    }

    return str;
}

}  // namespace

// -----------------------------------------------------------------------------
// SequentialWriteRead
// No payload wrapping involved
//
// Layout after WritePayload(len=100) at write_head=0:
//   [0..11]   = PayloadHeader
//   [12..111] = 100 bytes of data
//   write_head = 112
//
// ReadPayload reads from text_index=12, length=100.
// BUF - text_index = 244 >= 100 -> NON-WRAP.
// -----------------------------------------------------------------------------

TEST(MemoryArenaTest, SequentialWriteRead) {
    MemoryArena arena{PayloadTestConfig(0)};

    const std::string in         = GenPayload(100);
    const uint32_t    length     = static_cast<uint32_t>(in.size());
    const auto        offset_opt = arena.WritePayload(
        kNode, reinterpret_cast<const uint8_t*>(in.data()), length);

    EXPECT_EQ(arena.GetWriteHead(), kHeaderSize + 100);

    std::string out;
    arena.ReadPayload(*offset_opt, length, &out);

    EXPECT_EQ(out, in);
}

// -----------------------------------------------------------------------------
// DataWrapAround
// Place write_head = read_tail = BUF - 50 = 206.
//
// WritePayload(len=100):
//   AllocatePayload: actual_index = 206 & 255 = 206.
//   BUF - 206 = 50 >= HEADER(12) -> no padding, header fits at 206.
//   header_index = 206.
//   text_index   = (206 + 12) & 255 = 218.
//   BUF - text_index = 256 - 218 = 38 < 100 -> WRAP.
//     chunk1 = 38 bytes written at [218..255]
//     chunk2 = 62 bytes written at [0..61]
//
// ReadPayload(offset, 100):
//   text_index = (206 + 12) & 255 = 218.
//   BUF - 218 = 38 < 100 -> WRAP.
//     chunk1 = 38 bytes from [218..255]
//     chunk2 = 62 bytes from [0..61]
// -----------------------------------------------------------------------------

TEST(MemoryArenaTest, DataWrapAround) {
    constexpr uint64_t START = kBuf - 50;
    MemoryArena        arena{PayloadTestConfig(START)};

    const std::string in         = GenPayload(100);
    const uint32_t    length     = static_cast<uint32_t>(in.size());
    const auto        offset_opt = arena.WritePayload(
        kNode, reinterpret_cast<const uint8_t*>(in.data()), length);

    std::string out;
    arena.ReadPayload(*offset_opt, length, &out);

    EXPECT_EQ(out, in)
        << "data split across ring buffer boundary must reassemble correctly";
}

// -----------------------------------------------------------------------------
// HeaderWrapPaddingInserted
// Place write_head = read_tail = BUF - 8 = 248.
//
// WritePayload(len=100):
//   AllocatePayload: actual_index = 248 & 255 = 248.
//   BUF - actual_index = 256 - 248 = 8 < HEADER(12) -> INSERT PADDING.
//   header_index = 256 & 255 = 0  (wraps to start of buffer).
//   PayloadHeader written at [0..11].
//   text_index = (0 + 12) & 255 = 12.
//   BUF - 12 = 244 >= 100 -> NON-WRAP.
//   Data written at [12..111].
//
// ReadPayload(offset=256, 100):
//   text_index = (256 + 12) & 255 = 12.
//   BUF - 12 = 244 >= 100 -> NON-WRAP.
// -----------------------------------------------------------------------------

TEST(MemoryArenaTest, HeaderWrapPaddingInserted) {
    constexpr uint64_t START = kBuf - 8;
    MemoryArena        arena{PayloadTestConfig(START)};

    const std::string in         = GenPayload(100);
    const uint32_t    length     = static_cast<uint32_t>(in.size());
    const auto        offset_opt = arena.WritePayload(
        kNode, reinterpret_cast<const uint8_t*>(in.data()), length);

    EXPECT_EQ(*offset_opt & (kBuf - 1), 0ULL)
        << "header must start at physical index 0 after padding";

    std::string out;
    arena.ReadPayload(*offset_opt, length, &out);

    EXPECT_EQ(out, in) << "header-wrapped payload must read back correctly";
}

// -----------------------------------------------------------------------------
// ExhaustionThrows
// GC never runs, so read_tail is pinned at start_point.
// Write until used_space >= BUF.
//
// Each WritePayload(len=50) consumes HEADER(12) + 50 = 62 bytes.
// With 4 writes consumes total of 4 * 62 = 248 bytes.
// 5th write would need 62 more bytes: 248 + 62 = 310 > 256 -> THROW.
// -----------------------------------------------------------------------------

TEST(MemoryArenaTest, ExhaustionThrows) {
    constexpr uint32_t LEN = 50;
    MemoryArena        arena{PayloadTestConfig(0)};

    const std::string in   = GenPayload(LEN);
    const auto*       data = reinterpret_cast<const uint8_t*>(in.data());

    for (int32_t i = 0; i < 4; ++i) {
        ASSERT_NO_THROW(arena.WritePayload(kNode, data, LEN))
            << "write " << i << " should succeed";
    }

    EXPECT_THROW(arena.WritePayload(kNode, data, LEN), std::runtime_error)
        << "5th write must throw when buffer is full";
}

// -----------------------------------------------------------------------------
// MultipleSequentialWrites
// Multiple sequential writes, all read back correctly.
// Verifies write_head accounting across consecutive calls.
// -----------------------------------------------------------------------------

TEST(MemoryArenaTest, MultipleSequentialWrites) {
    MemoryArena arena{PayloadTestConfig(0)};

    const std::string in1 = GenPayload(20);
    const std::string in2 = GenPayload(30);
    const std::string in3 = GenPayload(10);

    const auto offset_opt1 =
        arena.WritePayload(0, reinterpret_cast<const uint8_t*>(in1.data()), 20);
    const auto offset_opt2 =
        arena.WritePayload(1, reinterpret_cast<const uint8_t*>(in2.data()), 30);
    const auto offset_opt3 =
        arena.WritePayload(2, reinterpret_cast<const uint8_t*>(in3.data()), 10);

    std::string out;

    arena.ReadPayload(*offset_opt1, 20, &out);
    EXPECT_EQ(out, in1);

    arena.ReadPayload(*offset_opt2, 30, &out);
    EXPECT_EQ(out, in2);

    arena.ReadPayload(*offset_opt3, 10, &out);
    EXPECT_EQ(out, in3);
}
