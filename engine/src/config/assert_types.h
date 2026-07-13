// Author: namnkahn1607
//
// Compile-time assertions to achieve expected lock-free features.

#pragma once

#include <atomic>

// Assert that std::atomic on unsigned integers are hardware-level lock-free.
static_assert(std::atomic<uint8_t>::is_always_lock_free);
static_assert(std::atomic<uint16_t>::is_always_lock_free);
static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(std::atomic<uint64_t>::is_always_lock_free);

// Assert that std::atomic on unsigned integers don't cause any memory overhead.
static_assert(sizeof(std::atomic<uint8_t>) == sizeof(uint8_t));
static_assert(sizeof(std::atomic<uint16_t>) == sizeof(uint16_t));
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t));
