// Compile-time assertions that guarantee lock-free primitives
// on Linux x86-64.
//
// In case any fails, CAS-based code would silently degrade to
// mutex-emulated atomics. Fail the build instead.

#pragma once

#include <atomic>
#include <cstdint>

// std::atomic on these unsigned integers MUST be hardware lock-free.
// No libatomic fallback.
static_assert(std::atomic<uint8_t>::is_always_lock_free);
static_assert(std::atomic<uint16_t>::is_always_lock_free);
static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(std::atomic<uint64_t>::is_always_lock_free);

// No hidden padding/lock overhead.
// sizeof(atomic<T>) == sizeof(T) is required for correct memory-layout.
static_assert(sizeof(std::atomic<uint8_t>) == sizeof(uint8_t));
static_assert(sizeof(std::atomic<uint16_t>) == sizeof(uint16_t));
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t));
