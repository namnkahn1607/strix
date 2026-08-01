// Vector search result recorder.

#pragma once

#include <optional>

// `SearchOutcome` describes result of a vector search against either tier.
// Only record information of a node whose similarity score exceeds the
// pre-defined `kSimiarityThreshold`.
struct SearchOutcome {
    uint32_t node_id;
    uint8_t  version;
};

// `SearchResult` tracks not just the champion but also the runner-up.
// The runner-up entry is optional.
struct SearchResult {
    SearchOutcome                primary;
    std::optional<SearchOutcome> secondary;
};
