// Unit tests for payload read/write operation in Memory Arena.
//
// Target the payload buffer chunking logic - specifically the distinct code
// paths in payload reader and writer:
//
//   Write paths:
//     [1] Sequential : header + data fit in one contiguous region.
//     [2] Data wrap  : header fits, but data straddles the buffer boundary
//                      (split into two memcpy calls).
//     [3] Header wrap: fewer than sizeof(PayloadHeader) bytes remain before
//                      the boundary - the expected behavior is to insert
//                      padding and places the entire header at index 0.
//
//   Read paths:
//     [1] Sequential : data fits in one contiguous region.
//     [2] Wrap       : data straddles the boundary (two memcpy calls).

#include "memory/arena.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "payload_header.h"

using namespace strix::memory;

namespace {

constexpr uint32_t kMaxSlots       = 4;
constexpr uint32_t kNode           = 0;
constexpr size_t   kPayloadBufSize = 256;

ArenaConfig Config(uint64_t start_point = 0) {
    return {kMaxSlots, kPayloadBufSize, /*prefault=*/false, start_point};
}

// Generates a deterministic payload of specified byte-size.
std::string GenPayload(size_t len) {
    std::string str(len, '\0');
    for (size_t i = 0; i < len; ++i) {
        str[i] = static_cast<char>((i * 37 + 13) % 251);  // Prime generators.
    }
    return str;
}

}  // namespace

// -----------------------------------------------------------------------------
// SequentialWriteRead
// No payload wrapping involved.
//
// Layout after WritePayload(len=100) at write_head=0:
//   [0..11]    = PayloadHeader
//   [12..111]  = 100 bytes of data
//   write_head = 112
//
// ReadPayload reads from text_index=12, length=100.
// BUF - text_index = 244 >= 100 -> NO data wrap.
// -----------------------------------------------------------------------------
TEST(MemoryArenaTest, SequentialWriteRead) {
    Arena arena{Config()};

    const std::string in     = GenPayload(100);
    const uint32_t    length = static_cast<uint32_t>(in.size());

    const auto opt_offset = arena.WritePayload(
        kNode, reinterpret_cast<const uint8_t*>(in.data()), length
    );
    EXPECT_TRUE(opt_offset.has_value());
    EXPECT_EQ(arena.GetWriteHead(), sizeof(PayloadHeader) + 100);

    std::string out;
    ArenaPrivateAccess::ReadPayload(arena, opt_offset.value(), length, &out);
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
    constexpr uint64_t kStart = kPayloadBufSize - 50;
    Arena              arena{Config(kStart)};

    const std::string in     = GenPayload(100);
    const uint32_t    length = static_cast<uint32_t>(in.size());

    const auto opt_offset = arena.WritePayload(
        kNode, reinterpret_cast<const uint8_t*>(in.data()), length
    );
    EXPECT_TRUE(opt_offset.has_value());

    std::string out;
    ArenaPrivateAccess::ReadPayload(arena, opt_offset.value(), length, &out);
    EXPECT_EQ(out, in) << "Wrapped payload must reassemble correctly";
}

// -----------------------------------------------------------------------------
// HeaderWrapPaddingInserted
// Place write_head = read_tail = kBuf - 8 = 248.
//
// WritePayload(len=100):
//   AllocatePayload: actual_index = 248 & 255 = 248.
//   kBuf - actual_index = 256 - 248 = 8 < HEADER(12) -> INSERT PADDING.
//   header_index = 256 & 255 = 0  (wraps to start of buffer).
//   PayloadHeader written at [0..11].
//   text_index = (0 + 12) & 255 = 12.
//   kBuf - 12 = 244 >= 100 -> NON-WRAP.
//   Data written at [12..111].
//
// ReadPayload(offset=256, 100):
//   text_index = (256 + 12) & 255 = 12.
//   kBuf - 12 = 244 >= 100 -> NON-WRAP.
// -----------------------------------------------------------------------------
TEST(MemoryArenaTest, HeaderWrapPaddingInserted) {
    constexpr uint64_t kStart = kPayloadBufSize - 8;
    Arena              arena{Config(kStart)};

    const std::string in     = GenPayload(100);
    const uint32_t    length = static_cast<uint32_t>(in.size());

    const auto opt_offset = arena.WritePayload(
        kNode, reinterpret_cast<const uint8_t*>(in.data()), length
    );
    EXPECT_TRUE(opt_offset.has_value());
    EXPECT_EQ(opt_offset.value() & (kPayloadBufSize - 1), 0)
        << "Header must start at physical index 0 after padding";

    std::string out;
    ArenaPrivateAccess::ReadPayload(arena, opt_offset.value(), length, &out);
    EXPECT_EQ(out, in) << "Header-wrapped payload must read back correctly";
}

// -----------------------------------------------------------------------------
// ExhaustionWrite
// GC never runs, so read_tail is pinned at start_point.
// Write until used_space >= BUF.
//
// Each WritePayload(len=50) consumes HEADER(12) + 50 = 62 bytes.
// With 4 writes consumes total of 4 * 62 = 248 bytes.
// 5th write would need 62 more bytes: 248 + 62 = 310 > 256 -> THROW.
// -----------------------------------------------------------------------------
TEST(MemoryArenaTest, ExhaustionThrows) {
    constexpr uint32_t kLen = 50;
    Arena              arena{Config()};

    const std::string in   = GenPayload(kLen);
    const auto*       data = reinterpret_cast<const uint8_t*>(in.data());

    for (int32_t i = 0; i < 4; ++i) {
        ASSERT_NO_THROW(arena.WritePayload(kNode, data, kLen))
            << "write " << i << " should succeed";
    }

    EXPECT_THROW(arena.WritePayload(kNode, data, kLen), std::runtime_error)
        << "5th write must throw when buffer is full";
}

// -----------------------------------------------------------------------------
// MultipleSequentialWrites
// Multiple sequential writes, all read back correctly.
// Verifies write_head accounting across consecutive calls.
// -----------------------------------------------------------------------------
TEST(MemoryArenaTest, MultipleSequentialWrites) {
    Arena arena{Config(0)};

    const std::string in1 = GenPayload(20);
    const std::string in2 = GenPayload(30);
    const std::string in3 = GenPayload(10);

    const auto opt_offset1 =
        arena.WritePayload(0, reinterpret_cast<const uint8_t*>(in1.data()), 20);
    EXPECT_TRUE(opt_offset1.has_value());

    const auto opt_offset2 =
        arena.WritePayload(1, reinterpret_cast<const uint8_t*>(in2.data()), 30);
    EXPECT_TRUE(opt_offset2.has_value());

    const auto opt_offset3 =
        arena.WritePayload(2, reinterpret_cast<const uint8_t*>(in3.data()), 10);
    EXPECT_TRUE(opt_offset3.has_value());

    std::string out;

    ArenaPrivateAccess::ReadPayload(arena, opt_offset1.value(), 20, &out);
    EXPECT_EQ(out, in1);

    ArenaPrivateAccess::ReadPayload(arena, opt_offset2.value(), 30, &out);
    EXPECT_EQ(out, in2);

    ArenaPrivateAccess::ReadPayload(arena, opt_offset3.value(), 10, &out);
    EXPECT_EQ(out, in3);
}
