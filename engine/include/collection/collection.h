#pragma once

#include <collection/config.h>
#include <memory/arena.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "base/cache_state.h"
#include "collection/search.h"

namespace strix::collection {

template <uint32_t K>
struct TopKResult;

template <uint32_t K>
struct TopKAccumulator;

class Collection {
public:
    static constexpr uint32_t kUnclustered = 0xFFFFFFFFu;

    explicit Collection(const Config& config, memory::Arena& arena);
    ~Collection();

    Collection(const Collection&)            = delete;
    Collection& operator=(const Collection&) = delete;
    Collection(Collection&&)                 = delete;
    Collection& operator=(Collection&&)      = delete;

    CacheState FetchCache(
        uint32_t node_id, uint8_t exp_ver, TimePoint now, std::string* out
    ) const noexcept;

    bool CommitEntry(
        uint32_t node_id, const uint8_t* in, uint32_t length
    ) noexcept;

    std::optional<TopKResult<kTopK>> Search(const float* query) const noexcept;

    // Nothing is returned upon exhausted node Freelist or saturated L0-tier
    // node buffer. In those cases, node would still reside DEAD in Freelist.
    std::optional<uint32_t> AcquireSlotFor(
        const float* query, TimePoint now
    ) noexcept;

    // Wired up as `Arena::NodeFreedCallback` during system init.
    void ReleaseSlot(uint32_t node_id) noexcept;

    // Starts a background worker that builds IVF from scratch then schedules
    // and performs Compaction, Recalibration and Reassignment in a work-loop
    // until `shutdown_req` is set to `false`.
    void StartCoordinator(const std::atomic<bool>& shutdown_req) noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;

    Impl*       Inner() noexcept { return pimpl_.get(); }
    const Impl* Inner() const noexcept { return pimpl_.get(); }
};

}  // namespace strix::collection
