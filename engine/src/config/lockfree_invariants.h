// Compile-time assertions guaranteeing lock-free primitives assumed on this
// Linux x86-64 platform.
// If any assertion fails, CAS-based code would silently degrade to
// mutex-emulated atomics. Fail the build instead.

#pragma once

#include <atomic>

// std::atomic on these unsigned integers MUST be hardware lock-free (no
// libatomic fallback).
static_assert(std::atomic<uint8_t>::is_always_lock_free);
static_assert(std::atomic<uint16_t>::is_always_lock_free);
static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(std::atomic<uint64_t>::is_always_lock_free);

// No hidden padding/lock overhead: sizeof(atomic<T>) == sizeof(T) is required
// for correct memory-layout/packing assumptions in hot-path structs.
static_assert(sizeof(std::atomic<uint8_t>) == sizeof(uint8_t));
static_assert(sizeof(std::atomic<uint16_t>) == sizeof(uint16_t));
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t));
