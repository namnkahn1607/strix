// Node state machine and Approximate LRU reference bit.

#pragma once

// NodeState represents state machine lifecycle for a single node slot.
// A state transition can only be achieved using compare-and-swap operation.
//
// Valid transitions:
//   - `kDead` -> `kPending`  : A worker acquired a `node_id` finished writing
//                              its vector data; the slot is now searchable but
//                              owns no payload (yet).
//   - `kPending` -> `kReady` : payload committed; payload can now be
//                              read/extract from this slot.
//   - `kReady` -> `kDead`    : a `kCold` node being evicted by the GC; the slot
//                              is then released back to `FreeList`.
enum class NodeState : uint8_t {
    kDead    = 0,
    kPending = 1,
    kReady   = 2,
};

// EvictState represents CLOCK reference bit of a node slot.
enum class EvictState : uint8_t {
    kCold = 0,  // Evict.
    kHot  = 1,  // Grant a "second chance".
};
