// Author: namnkahn1607
//
// Concrete implementation of global utility functions.
// See its header for an API view.

#include "global_utils.h"

#include <sys/mman.h>

#include <stdexcept>

void* Alloc32(const uint64_t size, const bool lazy) {
    const int flags = MAP_ANONYMOUS | MAP_PRIVATE | (lazy ? 0 : MAP_POPULATE);

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed for MemoryArena");
    }

    return ptr;
}
