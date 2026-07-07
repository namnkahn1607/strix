// Author: namnkahn1607
//
// Global cross-module utility subroutines.
// Module-private ones belong in their own translation units.

#pragma once

#include <cstdlib>

// `Alloc32` utilizes `mmap` underneath, allocates a private anonymous mapping
// of `size` bytes with read / write permissions. When `lazy` is false,
// instruction to the kernel is offered to pre-fault all pages during the mmap
// syscall, thus eliminating first-touch latency at the cost of longer
// construction time.
void* Alloc32(size_t size, bool lazy);
