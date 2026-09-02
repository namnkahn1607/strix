// Memory arena node slot.

#pragma once

#include <atomic>
#include <chrono>

#include "control_block.h"

namespace strix::memory {

using Clock = std::chrono::steady_clock;

// The metadata node slot.
struct alignas(64) MetaNode {
    // Atomically loads and decodes the control block.
    // By default, an `std::memory_order_acquire` load is performed.
    ControlBlock LoadControl(
        std::memory_order order = std::memory_order_acquire
    ) const noexcept {
        return ControlBlock::Unpack(control_block.load(order));
    }

    // Extracts only the version field of the control block word.
    // Used on both sides of a seqlock-style version check.
    uint8_t LoadVersion(std::memory_order order = std::memory_order_acquire)
        const noexcept {
        return static_cast<uint8_t>(
            (control_block.load(order) >> kVersionShift) & kVersionMask
        );
    }

    // Extracts only the virtual offset field of the control block word.
    uint64_t LoadOffset(std::memory_order order = std::memory_order_acquire)
        const noexcept {
        return static_cast<uint64_t>(
            (control_block.load(order) & kVirtualOffsetMask)
        );
    }

    // Monotonic timestamp at which this node was acquired.
    // Stamped by the acquiring worker before publishing; used to identify
    // expired PENDING nodes.
    std::atomic<Clock::time_point> created_at;

    // The 64-bit control block word.
    // Any modification attempts must be done via compare-and-swap. Direct store
    // can only be viable once ownership is guaranteed.
    std::atomic<uint64_t> control_block;
};

static_assert(std::atomic<Clock::time_point>::is_always_lock_free);

}  // namespace strix::memory
