//
// rpc/service.cc
//

#include "service.hh"

#include <atomic>
#include <chrono>
#include <exception>
#include <stdexcept>

#include "constants.hh"
#include "meta_node.hh"
#include "search.hh"
#include "strix.pb.h"

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

CacheServiceImpl::CacheServiceImpl(const Embedder& embedder, MemoryArena& arena)
    : embedder(embedder), arena(arena) {}

// ------------------------------------------------------------
// CheckCache
// ------------------------------------------------------------

grpc::Status CacheServiceImpl::CheckCache(
    [[maybe_unused]] grpc::ServerContext* context,
    const proto::CheckCacheRequest*       request,
    proto::CheckCacheResponse*            response) {
    try {
        if (request->prompt().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Prompt is empty"};
        }

        // --- Encode ---
        auto encode_result = embedder.Encode(request->prompt());
        if (!encode_result.ok()) {
            switch (encode_result.error()) {
                case EncodeError::TOKEN_LIMIT_EXCEEDED:
                    response->set_check_state(proto::CACHE_STATE_EXCEEDED);
                    return grpc::Status::OK;

                case EncodeError::DEGENERATED_VECTOR:
                    return {grpc::StatusCode::INTERNAL,
                            "Degenerate vector from model"};
            }
        }

        const float* query = encode_result.value().get();

        // --- Request's timestamp ---
        const auto wall = std::chrono::system_clock::now().time_since_epoch();
        const uint64_t curr_time = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(wall).count());

        // --- Search in L0 Buffer ---
        const auto [best_node_id, best_score, reusable_node_id] =
            SearchL0(arena, query, curr_time);

        // --- HIT path ---
        if (best_score >= SIMILARITY_THRESHOLD && best_node_id != -1) {
            MetaNode& best_node =
                arena.GetNode(static_cast<size_t>(best_node_id));

            const uint64_t best_ctrl =
                best_node.control_block.load(std::memory_order_acquire);
            const auto [state, ref_bit, length, offset] =
                UnpackControl(best_ctrl);

            switch (state) {
                case NodeState::READY:
                case NodeState::MIGRATING: {
                    // Mark HOT via CAS. If CAS fails, another thread raced us
                    // -> Acceptable, Reader is still served.
                    const uint64_t desired =
                        PackControl(state, EvictState::HOT, length, offset);
                    uint64_t expected = best_ctrl;
                    best_node.control_block.compare_exchange_strong(
                        expected, desired, std::memory_order_release,
                        std::memory_order_relaxed);

                    // TODO: Implement Hazard Offsets before reading payload
                    arena.ReadPayload(offset, length,
                                      response->mutable_cached_payload());

                    // Verify node was not evicted during payload reading
                    const auto [verify_state, vb, vl, vo] =
                        UnpackControl(best_node.control_block.load(
                            std::memory_order_acquire));
                    if (verify_state != NodeState::READY &&
                        verify_state != NodeState::MIGRATING) {
                        response->clear_cached_payload();
                        break;
                    }

                    response->set_check_state(proto::CACHE_STATE_HIT);
                    response->set_node_id(-1);
                    return grpc::Status::OK;
                }

                case NodeState::PENDING:
                    response->set_check_state(proto::CACHE_STATE_PENDING);
                    response->set_node_id(best_node_id);
                    return grpc::Status::OK;

                case NodeState::DEAD:
                case NodeState::CLAIMED:
                    // Node is already DEAD or evicted between L0 Search and
                    // here, or it's acquired by another thread.
                    break;
            }
        }

        // --- MISS path ---

        // No reusable slot -> Return MISS with NodeID = -1
        // Skip the follow-up SetCache and serve request uncached
        if (reusable_node_id == -1) {
            response->set_check_state(proto::CACHE_STATE_MISS);
            response->set_node_id(-1);
            return grpc::Status::OK;
        }

        // Claim the slot by CAS from DEAD -> CLAIMED
        MetaNode& slot = arena.GetNode(static_cast<size_t>(reusable_node_id));

        uint64_t expected_dead =
            slot.control_block.load(std::memory_order_relaxed);

        // Slot was taken by another concurrent request between L0 Buffer
        // Searching and here.
        if (UnpackControl(expected_dead).state != NodeState::DEAD) {
            response->set_check_state(proto::CACHE_STATE_MISS);
            response->set_node_id(-1);
            return grpc::Status::OK;
        }

        const uint64_t claimed_ctrl =
            PackControl(NodeState::CLAIMED, EvictState::HOT, 0, 0);

        // Lost the CAS race -> Return MISS + NodeID = -1
        // Skip the follow-up SetCache and serve request uncached
        if (!slot.control_block.compare_exchange_strong(
                expected_dead, claimed_ctrl, std::memory_order_release,
                std::memory_order_relaxed)) {
            response->set_check_state(proto::CACHE_STATE_MISS);
            response->set_node_id(-1);
            return grpc::Status::OK;
        }

        // Won the CAS race -> Copy the vector data to the slot
        // Only the current thread owns the slot -> Brute store
        std::memcpy(arena.GetVector(static_cast<size_t>(reusable_node_id)),
                    query, VECTOR_MEMSIZE);
        const uint64_t pending_ctrl =
            PackControl(NodeState::PENDING, EvictState::HOT, 0, 0);
        slot.control_block.store(pending_ctrl, std::memory_order_release);
        slot.created_at.store(curr_time, std::memory_order_release);

        response->set_check_state(proto::CACHE_STATE_MISS);
        response->set_node_id(reusable_node_id);
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        return {grpc::StatusCode::INTERNAL,
                std::string("Internal error: ") + e.what()};
    } catch (...) {
        return {grpc::StatusCode::INTERNAL, "Unknown fatal error"};
    }
}

// ------------------------------------------------------------
// SetCache
// ------------------------------------------------------------

grpc::Status CacheServiceImpl::SetCache(
    [[maybe_unused]] grpc::ServerContext* context,
    const proto::SetCacheRequest* request, proto::SetCacheResponse* response) {
    try {
        if (request->node_id() < 0) {
            return {grpc::StatusCode::INVALID_ARGUMENT,
                    "Negative NodeID is invalid"};
        }

        if (request->uncached_payload().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Payload is empty"};
        }

        const std::string& payload = request->uncached_payload();
        const auto payload_len = static_cast<uint32_t>(payload.length());
        if (payload_len > MAX_PAYLOAD_LENGTH) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Oversized payload"};
        }

        const auto node_id = static_cast<uint32_t>(request->node_id());
        MetaNode&  node = arena.GetNode(node_id);

        uint64_t new_offset = 0;
        try {
            new_offset = arena.WritePayload(
                node_id, reinterpret_cast<const uint8_t*>(payload.data()),
                payload_len);

        } catch (const std::runtime_error&) {
            // Ring buffer full. Try rolling back node to DEAD state.
            uint64_t expected =
                node.control_block.load(std::memory_order_relaxed);
            const uint64_t dead_ctrl =
                PackControl(NodeState::DEAD, EvictState::COLD, 0, 0);

            node.control_block.compare_exchange_strong(
                expected, dead_ctrl, std::memory_order_release,
                std::memory_order_relaxed);
            response->set_success(false);
            return grpc::Status::OK;
        }

        const uint64_t desired = PackControl(NodeState::READY, EvictState::HOT,
                                             payload_len, new_offset);
        uint64_t expected = node.control_block.load(std::memory_order_relaxed);

        while (true) {
            if (static_cast<uint8_t>(expected >> 62) !=
                static_cast<uint8_t>(NodeState::PENDING)) {
                // Node was evicted (stale PENDING swept by GC) between
                // CheckCache and SetCache. Payload is written but orphaned,
                // GC will reclaim the ring buffer space naturally.
                response->set_success(false);
                break;
            }

            if (node.control_block.compare_exchange_weak(
                    expected, desired, std::memory_order_release,
                    std::memory_order_relaxed)) {
                response->set_success(true);
                break;
            }
        }

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        return {grpc::StatusCode::INTERNAL,
                std::string("Encounter error: ") + e.what()};
    } catch (...) {
        return {grpc::StatusCode::INTERNAL, "Unknown fatal error"};
    }
}
