// Author: namnkahn1607
//
// Global cross-module utility subroutines.
// Module-private ones belong in their own translation units.

#pragma once

#include <sys/mman.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace common {

// `AllocMMap()` allocates a page-aligned (trivially 32-byte alignment 4KB page)
// anonymous private mapping of `size` bytes with RW permission via `mmap`.
// When `lazy = false`, the kernel is instructed to pre-fault all pages during
// syscall, eliminating first-touch latency caused by page-faults.
//
// Contract:
//   1. `size` must be > 0. `nullptr` won't be returned on `size == 0`.
//   2. `std::runtime_error` is thrown on allocation failure. Callers must be
//      construction paths, not steady-state/hot paths.
//   3. Caller owns the object memory lifetime and MUST pair it with a
//      `DeallocMMap(ptr, size)` using the same `size`.
inline void* AllocMMap(const size_t size, bool lazy) {
    assert(size > 0 && "AllocMMap: size must be > 0");

    const int flags = MAP_ANONYMOUS | MAP_PRIVATE | (lazy ? 0 : MAP_POPULATE);

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(
            "AllocMMap: mmap failed (errno=" + std::to_string(errno) +
            ", size=" + std::to_string(size) + ")");
    }

    return ptr;
}

// `DeallocMMap()` unmaps a region previously allocated using `AllocMMap()`.
// The `size` MUST match exactly with the one passed onto `AllocMMap()` call.
inline void DeallocMMap(void* ptr, size_t size) noexcept {
    if (ptr == nullptr || size == 0) {
        assert(false && "DeallocMMap: invalid ptr/size pair");
        return;
    }

    [[maybe_unused]] const int result = munmap(ptr, size);
    assert(result == 0 &&
           "DeallocMMap: munmap failed - check ptr/size pairing");
}

// `MonotonicNow()` returns a monotonic stopwatch value unaffected by NTP
// corrections & adjustments. Use this to measure elapsed time such as
// peroids, intervals, TTLs and countdowns.
// Default unit is seconds if `Duration` is not specified.
template <typename Duration = std::chrono::seconds>
inline uint64_t MonotonicNow() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Duration>(now).count());
}

// `WallUnixNow()` returns the current Unix epoch timestamp from the system
// clock. Use for loggings, cross-process/machine timestamp comparisons.
// Default unit is seconds if `Duration` is not specified.
template <typename Duration = std::chrono::seconds>
inline uint64_t WallUnixNow() {
    const auto wall = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Duration>(wall).count());
}

}  // namespace common
