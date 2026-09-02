// Alignment 32 bytes float buffer.

#pragma once

#include <array>
#include <span>

#include "info.h"

namespace strix::inference {

// Abstract of 384-dimensional vector produced by `SentenceEncoder`.
class alignas(32) SimdFloatBuf {
public:
    std::span<float, kVectorDim>       view() noexcept { return buffer_; }
    std::span<const float, kVectorDim> view() const noexcept { return buffer_; }

    float*       data() noexcept { return buffer_.data(); }
    const float* data() const noexcept { return buffer_.data(); }

private:
    std::array<float, kVectorDim> buffer_;
};

}  // namespace strix::inference
