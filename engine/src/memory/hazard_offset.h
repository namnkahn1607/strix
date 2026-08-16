// Hazard Offsets: range & table definition, publish/detach methods.

#pragma once

#include <array>
#include <atomic>
#include <optional>

#include "memory/control_block.h"

// HazardRange defines a "hazard zone": `[offset, offset + length)`.
struct HazardRange {
    uint64_t offset;
    uint32_t length;
};

// HazardOffsetTable is a way for payload reader to establish temporary
// protection against GC eviction.
// Note: `N` should equal the number of workers in gRPC thread pool.
template <size_t N>
class HazardTable final {
public:
    // Publishes a new "hazard zone" in payload buffer.
    // Used by `MemoryArena::ReadPayload()`, paired with a `Clear()` call
    // afterward.
    void Publish(uint32_t wid, uint64_t offset, uint32_t length) noexcept {
        slots_[wid].value.store(
            PackHazard(offset, length), std::memory_order_release
        );
    }

    // Detaches an existing "hazard zone" in payload buffer.
    // Used by `MemoryArena::ReadPayload()`, paired with a previous `Publish`
    // call.
    void Clear(uint32_t wid) noexcept {
        slots_[wid].value.store(0, std::memory_order_release);
    }

    // Checks if the specified range `[offset, offset + length)` intersects with
    // any published hazard zones.
    bool Overlaps(uint64_t offset, uint32_t length) const noexcept {
        for (auto& s : slots_) {
            auto h = UnpackHazard(s.value.load(std::memory_order_acquire));
            if (h.has_value() &&
                Intersect(offset, length, h.value().offset, h.value().length)) {
                return true;
            }
        }

        return false;
    }

private:
    static constexpr uint32_t kHazardActiveShift = 63u;
    static constexpr uint32_t kHazardLengthShift = 42u;
    static constexpr uint64_t kHazardLengthMask  = 0x1FFFFFull;

    // 8-byte representation of `HazardRange` in table.
    struct alignas(64) Slot {
        std::atomic<uint64_t> value{0};
    };

    // Checks if 2 ranges `[ao, ao + al)` and `[bo, bo + bl)` intersect.
    static bool Intersect(
        uint64_t ao, uint32_t al, uint64_t bo, uint32_t bl
    ) noexcept {
        return ao < bo + bl && bo < ao + al;
    }

    // Packs `offset` and `length` into a 64-bit word.
    static uint64_t PackHazard(uint64_t offset, uint32_t length) noexcept {
        return (1ull << kHazardActiveShift) |
               ((static_cast<uint64_t>(length) & kHazardLengthMask)
                << kHazardLengthShift) |
               (offset & kVirtualOffsetMask);
    }

    // Unpacks a a 64-bit word into an optional `HazardRange`.
    static std::optional<HazardRange> UnpackHazard(uint64_t word) noexcept {
        if ((word >> kHazardActiveShift) == 0) {
            return std::nullopt;
        }

        return HazardRange{
            word & kVirtualOffsetMask,
            static_cast<uint32_t>(
                (word >> kHazardLengthShift) & kHazardLengthMask
            )
        };
    }

    // The slots array, containing exactly `N` entries.
    std::array<Slot, N> slots_{};
};
