// Cache state enumeration.

#pragma once

namespace strix {

// Part of the cache state contract returned by Data plane to Control plane.
enum class CacheState : uint8_t { kMiss, kPendingHit, kHit };

}  // namespace strix
