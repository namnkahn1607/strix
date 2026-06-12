//
// inference/aligned_vec.hh
//
// RAII wrapper for 32-byte aligned float arrays.
// Required by AVX2 intrinsics (_mm256_load_ps demands 32-byte alignment).
//

#pragma once

#include <cassert>
#include <cstdlib>
#include <memory>
#include <new>

struct AlignedFree {
    void operator()(void* ptr) const noexcept { std::free(ptr); }
};

using AlignedVec = std::unique_ptr<float[], AlignedFree>;

// An AVX2 register holds 8 floats, so any dimensional arguments that is not a
// multiple of 8 cannot fill a full register and will cause out-of-bounds reads
// in DotProductL0_Batch4.
inline AlignedVec CreateAlignedVector(const size_t dim) {
    assert(dim % 8 == 0 &&
           "dimension must be a mutltiple of 8 for AVX2 alignment");

    void* ptr = std::aligned_alloc(32, dim * sizeof(float));
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }

    return AlignedVec{static_cast<float*>(ptr)};
}
