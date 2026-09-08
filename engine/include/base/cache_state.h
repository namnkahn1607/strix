#pragma once

#include <absl/strings/cord.h>

#include <cstdint>
#include <variant>

namespace strix {

enum class MissReason : uint8_t { kMiss, kPendingHit };

using CacheLookUpResult = std::variant<MissReason, absl::Cord>;

}  // namespace strix
