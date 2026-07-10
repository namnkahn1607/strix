// Author: namnkahn1607
//
// CacheServiceImpl method definitions: CheckCache and SetCache RPC handlers.

#include "service.h"

#include <grpcpp/support/status.h>

#include <chrono>
#include <exception>
#include <optional>

#include "strix.pb.h"

CacheServiceImpl::CacheServiceImpl(const Embedder& embedder, VectorIndex& index)
    : embedder_(embedder), index_(index) {
}

grpc::Status CacheServiceImpl::CheckCache(
    [[maybe_unused]] grpc::ServerContext* context,
    const proto::v1::CheckCacheRequest*   request,
    proto::v1::CheckCacheResponse*        response) {
    try {
        if (request->prompt().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Prompt is empty"};
        }

        // ------------------------------------------------------------------
        // Vectorization
        // ------------------------------------------------------------------

        auto encode_result = embedder_.Encode(request->prompt());
        if (!encode_result.ok()) {
            switch (encode_result.error()) {
                case EncodeError::kTokenLimitExceeded:
                    response->set_check_state(proto::v1::CACHE_STATE_REJECTED);
                    return grpc::Status::OK;

                case EncodeError::kDegeneratedVector:
                    response->set_check_state(proto::v1::CACHE_STATE_REJECTED);
                    return {grpc::StatusCode::INTERNAL,
                            "Degenerate vector from model"};
            }
        }

        const float* query = encode_result.value().get();

        // ------------------------------------------------------------------
        // Vector Searching: HIT path
        // ------------------------------------------------------------------

        const auto wall = std::chrono::system_clock::now().time_since_epoch();
        const uint64_t curr_time = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(wall).count());

        const auto l0_result = index_.SearchL0(query);
        if (l0_result.has_value()) {
            if (ProcessCandidate(l0_result->primary, curr_time, response)) {
                return grpc::Status::OK;
            }

            if (l0_result->secondary.has_value() &&
                ProcessCandidate(*l0_result->secondary, curr_time, response)) {
                return grpc::Status::OK;
            }
        }

        const auto l1_result = index_.SearchL1(query);
        if (l1_result.has_value()) {
            if (ProcessCandidate(l1_result->primary, curr_time, response)) {
                return grpc::Status::OK;
            }

            if (l1_result->secondary.has_value() &&
                ProcessCandidate(*l1_result->secondary, curr_time, response)) {
                return grpc::Status::OK;
            }
        }

        // ------------------------------------------------------------------
        // Vector Searching: MISS path
        // ------------------------------------------------------------------

        const auto node_id_opt = index_.AcquireNode(query, curr_time);
        if (!node_id_opt.has_value()) {
            response->set_check_state(proto::v1::CACHE_STATE_REJECTED);
            return grpc::Status::OK;
        }

        response->set_check_state(proto::v1::CACHE_STATE_MISS);
        response->set_node_id(*node_id_opt);
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
    const proto::v1::SetCacheRequest*     request,
    proto::v1::SetCacheResponse*          response) {
    try {
        const uint32_t     node_id = request->node_id();
        const std::string& payload = request->uncached_payload();

        if (request->uncached_payload().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Empty payload"};
        }

        const uint32_t payload_len = static_cast<uint32_t>(payload.length());
        if (payload_len > kMaxPayloadLength) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Oversized payload"};
        }

        response->set_success(index_.CommitPayload(
            node_id, reinterpret_cast<const uint8_t*>(payload.data()),
            payload_len));

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        return {grpc::StatusCode::INTERNAL,
                std::string("Internal error: ") + e.what()};
    } catch (...) {
        return {grpc::StatusCode::INTERNAL, "Unknown fatal error"};
    }
}

bool CacheServiceImpl::ProcessCandidate(
    const SearchOutcome& candidate, const uint64_t timestamp,
    proto::v1::CheckCacheResponse* response) const {
    const auto outcome =
        index_.FetchPayload(candidate.node_id, candidate.version, timestamp,
                            response->mutable_cached_payload());

    switch (outcome) {
        case CacheOutcome::kHit:
            response->set_check_state(proto::v1::CACHE_STATE_HIT);
            return true;

        case CacheOutcome::kPendingHit:
            response->set_check_state(proto::v1::CACHE_STATE_PENDING);
            response->set_node_id(candidate.node_id);
            return true;

        case CacheOutcome::kMiss:
            return false;
    }

    return false;
}
