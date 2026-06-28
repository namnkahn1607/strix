// Author: namnkahn1607
//
// CacheServiceImpl method definitions: CheckCache and SetCache RPC handlers.

#include "service.h"

#include <chrono>
#include <exception>
#include <stdexcept>

#include "constants.h"
#include "meta_node.h"
#include "search.h"
#include "strix.pb.h"

CacheServiceImpl::CacheServiceImpl(const Embedder& embedder, MemoryArena& arena)
    : embedder_(embedder), arena_(arena) {
}

grpc::Status CacheServiceImpl::CheckCache(
    [[maybe_unused]] grpc::ServerContext* context,
    const proto::CheckCacheRequest*       request,
    proto::CheckCacheResponse*            response) {
    try {
        if (request->prompt().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Prompt is empty"};
        }

        auto encode_result = embedder_.Encode(request->prompt());
        if (!encode_result.ok()) {
            switch (encode_result.error()) {
                case EncodeError::kTokenLimitExceeded:
                    response->set_check_state(proto::CACHE_STATE_EXCEEDED);
                    return grpc::Status::OK;

                case EncodeError::kDegeneratedVector:
                    return {grpc::StatusCode::INTERNAL,
                            "Degenerate vector from model"};
            }
        }

        const float* query = encode_result.value().get();

        const auto wall = std::chrono::system_clock::now().time_since_epoch();
        const uint64_t curr_time = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(wall).count());

        const auto [best_node_id, best_score, reusable_node_id] =
            SearchL0(arena_, query, curr_time);

        // HIT path
        if (best_score >= kSimilarityThreshold && best_node_id != -1) {
            MetaNode& best_node =
                arena_.GetNode(static_cast<size_t>(best_node_id));

            const uint64_t best_ctrl =
                best_node.control_block.load(std::memory_order_acquire);
            const auto [state, ref_bit, length, offset] =
                UnpackControl(best_ctrl);

            switch (state) {
                case NodeState::kReady:
                case NodeState::kMigrating: {
                    // Mark HOT via CAS. If CAS fails, another thread raced
                    // us - acceptable, Reader is still served correctly.
                    uint64_t       expected = best_ctrl;
                    const uint64_t desired =
                        PackControl(state, EvictState::kHot, length, offset);
                    best_node.control_block.compare_exchange_strong(
                        expected, desired, std::memory_order_release,
                        std::memory_order_relaxed);

                    // TODO: implement Hazard Offsets before reading payload.
                    arena_.ReadPayload(offset, length,
                                       response->mutable_cached_payload());

                    // Verify node was not evicted during payload read.
                    const auto [verify_state, vb, vl, vo] =
                        best_node.LoadControl();
                    if (verify_state != NodeState::kReady &&
                        verify_state != NodeState::kMigrating) {
                        response->clear_cached_payload();
                        break;
                    }

                    response->set_check_state(proto::CACHE_STATE_HIT);
                    response->set_node_id(-1);
                    return grpc::Status::OK;
                }

                case NodeState::kPending:
                    response->set_check_state(proto::CACHE_STATE_PENDING);
                    response->set_node_id(best_node_id);
                    return grpc::Status::OK;

                case NodeState::kDead:
                case NodeState::kClaimed:
                    // Node was evicted or claimed by another thread between
                    // SearchL0() and here. Fall through to MISS path.
                    break;
            }
        }

        // MISS path

        // No reusable slot: return MISS with node_id = -1.
        // Caller skips the follow-up SetCache and serves the request uncached.
        if (reusable_node_id == -1) {
            response->set_check_state(proto::CACHE_STATE_MISS);
            response->set_node_id(-1);
            return grpc::Status::OK;
        }

        // Attempt to claim the reusable slot via DEAD -> CLAIMED CAS.
        MetaNode& slot = arena_.GetNode(static_cast<size_t>(reusable_node_id));

        uint64_t expected_dead =
            slot.control_block.load(std::memory_order_relaxed);

        // Slot was claimed by a concurrent request between SearchL0() and here.
        if (UnpackControl(expected_dead).state != NodeState::kDead) {
            response->set_check_state(proto::CACHE_STATE_MISS);
            response->set_node_id(-1);
            return grpc::Status::OK;
        }

        const uint64_t claimed_ctrl =
            PackControl(NodeState::kClaimed, EvictState::kHot, 0, 0);

        // Lost the CAS race to another concurrent request.
        if (!slot.control_block.compare_exchange_strong(
                expected_dead, claimed_ctrl, std::memory_order_release,
                std::memory_order_relaxed)) {
            response->set_check_state(proto::CACHE_STATE_MISS);
            response->set_node_id(-1);
            return grpc::Status::OK;
        }

        // Won the CAS race: this thread exclusively owns the slot.
        // Brute-copy the vector; no other thread can observe this slot
        // until the subsequent store to kPending.
        std::memcpy(arena_.GetVector(static_cast<size_t>(reusable_node_id)),
                    query, kVectorMemsize);

        const uint64_t pending_ctrl =
            PackControl(NodeState::kPending, EvictState::kHot, 0, 0);
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

grpc::Status CacheServiceImpl::SetCache(
    [[maybe_unused]] grpc::ServerContext* context,
    const proto::SetCacheRequest* request, proto::SetCacheResponse* response) {
    try {
        if (request->node_id() < 0) {
            return {grpc::StatusCode::INVALID_ARGUMENT,
                    "Negative NodeID is invalid"};
        }

        if (request->uncached_payload().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Empty payload"};
        }

        const std::string& payload = request->uncached_payload();
        const auto payload_len     = static_cast<uint32_t>(payload.length());
        
        if (payload_len > kMaxPayloadLength) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Oversized payload"};
        }

        const auto node_id = static_cast<uint32_t>(request->node_id());
        MetaNode&  node    = arena_.GetNode(node_id);

        uint64_t new_offset = 0;
        try {
            new_offset = arena_.WritePayload(
                node_id, reinterpret_cast<const uint8_t*>(payload.data()),
                payload_len);

        } catch (const std::runtime_error&) {
            // Ring buffer exhausted. Roll back the node to DEAD so the slot
            // can be reclaimed.
            uint64_t expected =
                node.control_block.load(std::memory_order_relaxed);
            const uint64_t dead_ctrl =
                PackControl(NodeState::kDead, EvictState::kCold, 0, 0);

            node.control_block.compare_exchange_strong(
                expected, dead_ctrl, std::memory_order_release,
                std::memory_order_relaxed);

            response->set_success(false);
            return grpc::Status::OK;
        }

        uint64_t expected = node.control_block.load(std::memory_order_relaxed);
        const uint64_t desired = PackControl(
            NodeState::kReady, EvictState::kHot, payload_len, new_offset);

        while (true) {
            if (UnpackControl(expected).state != NodeState::kPending) {
                // Node was swept to kDead by the GC (stale PENDING timeout)
                // between CheckCache and SetCache. The written payload is
                // orphaned; GC will reclaim the ring buffer space naturally.
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
                std::string("Internal error: ") + e.what()};
    } catch (...) {
        return {grpc::StatusCode::INTERNAL, "Unknown fatal error"};
    }
}
