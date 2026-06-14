//
// common/config.hh
//
// Cross-module compile constants.
// Only constants needed by MORE THAN ONE module live here.
// Module-private constants belong in their own headers.
//

#pragma once

#include <cstddef>

// --- Vector dimension ---
inline constexpr size_t VECTOR_DIM = 384;
inline constexpr size_t VECTOR_MEMSIZE = VECTOR_DIM * sizeof(float);

// --- Cache lookup threshold ---
inline constexpr float SIMILARITY_THRESHOLD = 0.85f;

// --- Node slot limits & capacities --
inline constexpr size_t L0_MAX_SLOTS = 1'024;
inline constexpr size_t L1_MAX_SLOTS = 524'288;
inline constexpr size_t TOTAL_MAX_SLOTS = L0_MAX_SLOTS + L1_MAX_SLOTS;
