// Metadata struct for Memory Arena node slot.

#pragma once

#include <atomic>

#include "control_block.h"

// MetaNode defines the metadata for one node slot.
//
// Lock-free and not thread-safe.
struct alignas(64) MetaNode {  // Avoid false sharing under concurrent access.
    // Monotonic timestamp (seconds) at which this slot was acquired.
    //
    // Used by the GC to identify expired `kPending` nodes.
    // Always written by the acquiring worker before publishing, so any read
    // observing `kPending` is valid.
    std::atomic<uint64_t> created_at;

    // The 64-bit control block word.
    //
    // Not thread-safe. Any modification attempt must be done via
    // compare-and-swap, otherwise a direct store only if ownership is
    // guaranteed.
    std::atomic<uint64_t> control_block;

    // LoadControl atomically loads and decodes the control block.
    //
    // By default, a `std::memory_order_acquire` load is performed.
    ControlBlock LoadControl(
        const std::memory_order order = std::memory_order_acquire
    ) const noexcept {
        return UnpackControl(control_block.load(order));
    }

    // LoadVersion extracts only the version field without decoding the
    // rest of the control block word.
    //
    // Used on both sides of a seqlock-style version check.
    uint8_t LoadVersion(
        const std::memory_order order = std::memory_order_acquire
    ) const noexcept {
        return static_cast<uint8_t>(
            (control_block.load(order) >> kVersionShift) & kVersionMask
        );
    }
};
