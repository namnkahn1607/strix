// Aligned (32-byte) float buffer.
// AVX2 intrinsics _mm256_load_ps enforces 32-byte alignment.

#pragma once

#include <array>
#include <memory>

#include "info.h"

// SimdFloatBuf represents a 32-byte aligned buffer of `kVectorDim` floats.
class alignas(32) SimdFloatBuf {
public:
    float*       data() noexcept { return buffer_.data(); }
    const float* data() const noexcept { return buffer_.data(); }

private:
    std::array<float, kVectorDim> buffer_;
};

// SimdFloatVec captures `SimdFloatBuf` within a `std::unique_ptr`.
using SimdFloatVec = std::unique_ptr<SimdFloatBuf>;
