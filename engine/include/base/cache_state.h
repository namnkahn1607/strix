#pragma once

#include <cstdint>

namespace strix {

enum class CacheState : uint8_t { kMiss, kPendingHit, kHit };

}  // namespace strix
