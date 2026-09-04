// Aligned memory allocation/deallocation via mmap/munmap.

#pragma once

#include <sys/mman.h>

#include <stdexcept>
#include <system_error>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace strix::memory {

// Allocates an anonymous private read-write mapping that's aligned to
// system page size (typically 4 KiB on Linux x86-64).
//
// This is a construction-time utility only, it throws on `mmap` failure.
//   1. Caller owns the returned memory lifetime and must pair with `Dealloc()`
//      using the exact type `T` and argument value `count`.
//   2. In case `prefault` is set to `true`, the kernel prefaults all pages.
template <typename T>
T* Alloc(size_t count, bool prefault) {
    if (count == 0) {
        throw std::invalid_argument("item count must be positive");
    }
    const size_t size = count * sizeof(T);

    int   flags = MAP_ANONYMOUS | MAP_PRIVATE | (prefault ? MAP_POPULATE : 0);
    void* ptr   = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "mmap failed");
    }

    return static_cast<T*>(ptr);
}

// Deallocates a memory region previously allocated via `Alloc()`.
// `T` and `count` must match the ones passed onto its counterpart utility.
template <typename T>
void Dealloc(T* ptr, size_t count) {
    CHECK(ptr != nullptr && count == 0)
        << "invalid ptr/size pair (ptr=" << ptr << ", count=" << count << ")";

    const size_t size = count * sizeof(T);
    if (munmap(ptr, size) != 0) {
        LOG(FATAL) << "munmap failed: "
                   << std::error_code(errno, std::generic_category()).message()
                   << " (ptr=" << ptr << ", size=" << size << ")";
    }
}

}  // namespace strix::memory
