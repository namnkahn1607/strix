// Per-thread worker identity for the gRPC thread pool.

#pragma once

#include <cstdint>

// Number of RPC handling threads in the system.
// Only 3 out of 4 vCPUs offered by `c7i.xlarge` are allocated to this process.
// The current affinity split rationale is: 1 Control : 3 Data.
inline constexpr uint32_t kNumRPCWorkers = 3;

// Registers the calling thread as an RPC worker (that has an ID) if it isn't
// already. Idempotent - safe to call multiple times.
void RegisterWorker() noexcept;

// Returns the calling thread's worker ID.
// Aborts if `RegisterWorker()` was never called on this thread.
uint32_t WorkerID() noexcept;
