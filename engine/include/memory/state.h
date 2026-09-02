// Node state machine and Approximate LRU reference bit.

#pragma once

#include <chrono>

namespace strix::memory {

// Maximum lifetime of a PENDING node.
// Crossing this deadline makes the node treated as expired/stale.
inline constexpr std::chrono::seconds kPendingLifespan{30};

// State machine lifecycle of a node slot.
//
// Valid transitions:
//   - `kDead` -> `kPending`  : A worker acquiring this node has finished
//                              committing its vector data; the slot is now
//                              searchable but owns no payload (yet).
//   - `kPending` -> `kReady` : Payload committed and can now be read.
//   - `kReady` -> `kDead`    : A COLD node being evicted by the GC; the slot
//                              will then be released back to node ID Freelist.
enum class NodeState : uint8_t {
    kDead    = 0,
    kPending = 1,
    kReady   = 2,
};

// CLOCK reference bit of a node slot.
enum class EvictState : uint8_t {
    kCold = 0,
    kHot  = 1,
};

}  // namespace strix::memory
