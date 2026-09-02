// RPC service handler.

#include "rpc/cache_service.h"

#include <grpcpp/support/status.h>

#include <exception>
#include <optional>

#include "cache.pb.h"
#include "inference/sentence_encoder.h"
#include "inference/simd_float_buf.h"
#include "memory/control_block.h"
#include "worker/identity.h"

namespace strix::rpc {

namespace mem = strix::memory;

CacheServiceImpl::CacheServiceImpl(
    const inf::SentenceEncoder& encoder, coll::VectorIndex& index
)
    : encoder_(encoder), index_(index) {}

grpc::Status CacheServiceImpl::CheckCache(
    [[maybe_unused]] grpc::ServerContext* context,
    const strix::v1::CheckCacheRequest*   request,
    strix::v1::CheckCacheResponse*        response
) {
    worker::Register();

    try {
        if (request->prompt().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Prompt is empty"};
        }

        inf::SimdFloatBuf query_buf;
        if (auto encode_err = encoder_.Encode(request->prompt(), query_buf)) {
            switch (encode_err.value()) {
                case inf::EncodeError::kTokenLimitExceeded:
                    response->set_check_state(strix::v1::CACHE_STATE_REJECTED);
                    return grpc::Status::OK;

                case inf::EncodeError::kDegeneratedVector:
                    response->set_check_state(strix::v1::CACHE_STATE_REJECTED);
                    return {
                        grpc::StatusCode::INTERNAL,
                        "Degenerate vector from model"
                    };
            }
        }

        const float*      query     = query_buf.data();
        Clock::time_point curr_time = Clock::now();

        const auto l0_result = index_.SearchL0(query);
        if (l0_result.has_value()) {
            if (Process(l0_result->primary, curr_time, response)) {
                return grpc::Status::OK;
            }

            if (l0_result->secondary.has_value() &&
                Process(*l0_result->secondary, curr_time, response)) {
                return grpc::Status::OK;
            }
        }

        const auto l1_result = index_.SearchL1(query);
        if (l1_result.has_value()) {
            if (Process(l1_result->primary, curr_time, response)) {
                return grpc::Status::OK;
            }

            if (l1_result->secondary.has_value() &&
                Process(*l1_result->secondary, curr_time, response)) {
                return grpc::Status::OK;
            }
        }

        const auto opt_node_id = index_.AcquireNode(query, curr_time);
        if (!opt_node_id.has_value()) {
            response->set_check_state(strix::v1::CACHE_STATE_REJECTED);
            return grpc::Status::OK;
        }

        response->set_check_state(strix::v1::CACHE_STATE_MISS);
        response->set_node_id(opt_node_id.value());
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        return {
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what()
        };
    } catch (...) {
        return {grpc::StatusCode::INTERNAL, "Unknown fatal error"};
    }
}

grpc::Status CacheServiceImpl::SetCache(
    [[maybe_unused]] grpc::ServerContext* context,
    const strix::v1::SetCacheRequest*     request,
    strix::v1::SetCacheResponse*          response
) {
    try {
        const auto  node_id = request->node_id();
        const auto& payload = request->uncached_payload();

        if (request->uncached_payload().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Empty payload"};
        }

        const auto payload_len = static_cast<uint32_t>(payload.length());
        if (payload_len > mem::kMaxPayloadLength) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Oversized payload"};
        }

        response->set_success(index_.Commit(
            node_id, reinterpret_cast<const uint8_t*>(payload.data()),
            payload_len
        ));
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        return {
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what()
        };
    } catch (...) {
        return {grpc::StatusCode::INTERNAL, "Unknown fatal error"};
    }
}

bool CacheServiceImpl::Process(
    coll::SearchOutcome candidate, const Clock::time_point curr_time,
    strix::v1::CheckCacheResponse* response
) const {
    const auto outcome = index_.Fetch(
        candidate.node_id, candidate.version, curr_time,
        response->mutable_cached_payload()
    );

    switch (outcome) {
        case strix::CacheState::kHit:
            response->set_check_state(strix::v1::CACHE_STATE_HIT);
            return true;

        case strix::CacheState::kPendingHit:
            response->set_check_state(strix::v1::CACHE_STATE_PENDING);
            response->set_node_id(candidate.node_id);
            return true;

        case strix::CacheState::kMiss: return false;
    }

    return false;
}

}  // namespace strix::rpc
