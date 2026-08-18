// Cache state enumeration.

#pragma once

// CacheOutcome represents part of the cache state returned by Data plane to
// Control plane.
enum class CacheOutcome : uint8_t { kMiss, kPendingHit, kHit };
