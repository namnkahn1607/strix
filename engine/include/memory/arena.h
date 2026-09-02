// Memory arena.

#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "common/cache_state.h"
#include "inference/info.h"
#include "memory/arena_config.h"
#include "memory/meta_node.h"
#include "worker/identity.h"

namespace strix::memory {

class ArenaPrivateAccess;
template <size_t N>
class HazardTable;

// Primary memory regions: metadata & vector array, and a payload buffer.
// Thread-safe as each API defines its own synchronization guarantees.
class Arena final {
public:
    using NodeFreedCallback = std::function<void(uint32_t)>;

    explicit Arena(const ArenaConfig& config);
    ~Arena();

    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&)                 = delete;
    Arena& operator=(Arena&&)      = delete;

    // Reads payload of a node slot and returns corresponding `CacheOutcome`
    // status. Asserts non-null payload buffer.
    // Only invoke after successfully establishing "hazard zone".
    CacheState ReadPayload(
        uint32_t node_id, uint8_t exp_ver, Clock::time_point curr_time,
        std::string* out
    ) const noexcept;

    // Writes a `PayloadHeader` followed by a byte sequence into the buffer.
    // The virtual offset of the written byte series is returned on success.
    // Asserts non-null payload buffer.
    std::optional<uint64_t> WritePayload(
        uint32_t node_id, const uint8_t* in, uint32_t length
    ) noexcept;

    // Triggers a worker that finds and evicts COLD READY nodes.
    // Operates until `shutdown_req` is set to `true`; must be launched on ONE
    // dedicated thread.
    void RunGarbageCollector(const std::atomic<bool>& shutdown_req);

    // Registers the node ID eviction notification hook used by GC.
    // Wired ONCE during system init, before spawning the GC thread.
    void SetNodeFreedCallback(NodeFreedCallback cb) {
        on_node_freed_ = std::move(cb);
    }

    // Returns a reference to the `MetaNode` at `node_id`.
    // Caller must ensure `node_id < max_slots`.
    MetaNode& GetNode(uint32_t node_id) const noexcept {
        return metadata_[static_cast<size_t>(node_id)];
    }

    // Returns a pointer to the first float of the vector at `node_id`.
    // Caller must ensure `node_id < max_slots`.
    float* GetVector(uint32_t node_id) const noexcept {
        return vectors_ + static_cast<size_t>(node_id) * inference::kVectorDim;
    }

    uint64_t GetWriteHead() const noexcept {
        return write_head_.load(std::memory_order_acquire);
    }

    uint64_t GetReadTail() const noexcept {
        return read_tail_.load(std::memory_order_acquire);
    }

    const uint32_t max_slots;
    const size_t   payload_buf_size;

private:
    friend class ArenaPrivateAccess;

    size_t ActualIndex(uint64_t offset) const noexcept {
        return offset & (payload_buf_size - 1);
    }

    // Byte fetching kernel.
    void Read(uint64_t offset, uint32_t length, std::string* out)
        const noexcept;

    std::optional<uint64_t> TryAllocateSpace(uint32_t length) noexcept;

    void TryReclaimSpace(
        uint32_t node_id, uint64_t tail, uint32_t total_len, bool release_node
    ) noexcept;

    void SweepStalePending(Clock::time_point curr_time) noexcept;

    MetaNode* metadata_;  // Node slot array
    float*    vectors_;   // Vector array
    uint8_t*  payload_buf_ = nullptr;

    std::atomic<uint64_t> write_head_;
    std::atomic<uint64_t> read_tail_;

    // Callback hook invoked by the GC when a node is ready for reuse.
    NodeFreedCallback on_node_freed_;

    // Managing table of published hazard zones.
    std::unique_ptr<HazardTable<worker::kNumRPCWorkers>> hazard_table_;
};

// Grants user code (non-const) access to private fields of `Arena`.
class ArenaPrivateAccess final {
public:
    // Writes bytes into the payload buffer to trigger pagefault. Asserts
    // non-null payload buffer.
    // Only use when `Arena` was constructed with page prefault disabled.
    static void PrefaultBuffer(Arena& arena) noexcept {
        CHECK(arena.payload_buf_ != nullptr)
            << "A non-null payload buffer is required";

        arena.payload_buf_[arena.ActualIndex(arena.write_head_)] = 0;
        arena.payload_buf_[0]                                    = 0;
    }

    // Mirrors the byte fetching kernel used by `Arena::ReadPayload()`.
    // Asserts non-null payload buffer.
    static void ReadPayload(
        Arena& arena, uint64_t offset, uint32_t length, std::string* out
    ) noexcept {
        CHECK(arena.payload_buf_ != nullptr)
            << "A non-null payload buffer is required";

        arena.Read(offset, length, out);
    }
};

}  // namespace strix::memory
