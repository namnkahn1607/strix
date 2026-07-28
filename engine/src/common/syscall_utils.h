// Syscalls covered as global cross-module subroutines.
// Module-private ones belong in their own translation units.

#pragma once

#include <sys/mman.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace common {

// AllocMMap allocates an anonymous private RW mapping of `size` bytes via
// `mmap`, aligned to the system page size (typically 4KB on Linux x86-64).
//
// When `lazy = false`, the kernel pre-faults all pages during the syscall,
// thus eliminating first-touch latency caused by page-faults.
//
// This is a construction-time ONLY utility, never called on a hot path.
// Throws on `mmap` failure, as this is an unrecoverable error caused by
// programmers.
//
// Contract:
//   1. Asserts `size > 0`. `nullptr` is never returned otherwise.
//   2. Caller owns returned memory lifetime and MUST pair it with
//      `DeallocMMap(ptr, size)` using the exact same `size`.
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

// DeallocMMap unmaps a region previously allocated using `AllocMMap()`.
// `size` MUST match exactly the one passed to the corresponding `AllocMMap()`.
//
// Thread-safety: not synchronized. Caller must guarantee no concurrent
// access to the region for the duration of this call.
inline void DeallocMMap(void* ptr, size_t size) {
    if (ptr == nullptr || size == 0) {
        throw std::invalid_argument(
            "DeallocMMap: invalid ptr/size pair (ptr=" +
            std::to_string(reinterpret_cast<uintptr_t>(ptr)) +
            ", size=" + std::to_string(size) + ")");
    }

    if (munmap(ptr, size) != 0) {
        throw std::runtime_error(
            "DeallocMMap: munmap failed (errno=" + std::to_string(errno) + ")");
    }
}

// MonotonicNow returns a monotonic stopwatch value unaffected by NTP
// corrections & adjustments. Use to measure elapsed time such as periods,
// intervals, TTLs and countdowns.
//
// Default unit is seconds if `Duration` is not specified.
template <typename Duration = std::chrono::seconds>
inline uint64_t MonotonicNow() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Duration>(now).count());
}

// WallUnixNow returns the current Unix epoch timestamp from the system clock.
// Use for loggings, cross-process/machine timestamp comparisons.
//
// Default unit is seconds if `Duration` is not specified.
template <typename Duration = std::chrono::seconds>
inline uint64_t WallUnixNow() {
    const auto wall = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Duration>(wall).count());
}

}  // namespace common
