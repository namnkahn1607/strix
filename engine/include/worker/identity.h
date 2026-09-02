// Per-thread worker identity for the gRPC thread pool.
// Current affinity split ratio: 1 Control - 3 Data.

#pragma once

#include <cstdint>

namespace strix::worker {

// Maximum number of RPC handling threads.
inline constexpr uint32_t kNumRPCWorkers = 3;

// Registers worker ID for the calling thread if it hasn't.
// Idempotent - safe to call multiple times.
void Register() noexcept;

// Returns the calling thread's worker ID.
// Aborts if the thread does not own any ID.
uint32_t ThreadID() noexcept;

}  // namespace strix::worker
