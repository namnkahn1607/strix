// Per-thread worker identity for the gRPC thread pool.

#pragma once

#include <cstdint>

// Registers the calling thread as an RPC worker (that has an ID) if it isn't
// already. Idempotent - safe to call multiple times.
void RegisterWorker() noexcept;

// Returns the calling thread's worker ID.
// Aborts if `RegisterWorker()` was never called on this thread.
uint32_t WorkerID() noexcept;
