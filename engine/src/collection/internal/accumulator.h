#pragma once

#include <cstdint>
#include <optional>

#include "collection/search.h"

namespace strix::collection {

template <uint32_t K>
struct TopKAccumulator {
    float    scores[K];
    uint32_t ids[K];
    uint8_t  vers[K];

    TopKAccumulator() noexcept {
        for (uint32_t i = 0; i < K; ++i) {
            scores[i] = -1.0f;
        }
    }

    void Consider(uint32_t id, uint8_t ver, float scr) noexcept {
        if (scr <= scores[K - 1]) {
            return;
        }

        uint32_t pos = K - 1;
        while (pos > 0 && scores[pos - 1] < scr) {
            scores[pos] = scores[pos - 1];
            ids[pos]    = ids[pos - 1];
            vers[pos]   = vers[pos - 1];
            --pos;
        }
        scores[pos] = scr;
        ids[pos]    = id;
        vers[pos]   = ver;
    }

    std::optional<TopKResult<K>> Finalize() const noexcept {
        if (scores[0] < kSimilarityThreshold) {
            return std::nullopt;
        }

        TopKResult<K> res;
        for (uint32_t i = 0; i < K; ++i) {
            if (scores[i] < kSimilarityThreshold) {
                break;
            }
            res.records[res.count++] = {ids[i], vers[i]};
        }
        return res;
    }
};

}  // namespace strix::collection
