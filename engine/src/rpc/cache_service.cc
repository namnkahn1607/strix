// RPC service handler.

#include "rpc/cache_service.h"

#include <absl/strings/cord.h>
#include <grpcpp/support/status.h>

#include <exception>
#include <optional>
#include <variant>

#include "base/cache_state.h"
#include "cache.pb.h"
#include "collection/search.h"
#include "inference/sentence_encoder.h"
#include "inference/simd_float_buf.h"
#include "memory/control_block.h"
#include "worker/identity.h"

namespace strix::rpc {

CacheServiceImpl::CacheServiceImpl(
    const inference::SentenceEncoder& encoder, collection::Collection& collector
)
    : encoder_(encoder), collector_(collector) {}

grpc::Status CacheServiceImpl::CheckCache(
    [[maybe_unused]] grpc::ServerContext* context,
    const strix::v1::CheckCacheRequest*   request,
    strix::v1::CheckCacheResponse*        response
) {
    worker::Register();

    try {
        if (request->prompt().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Empty prompt"};
        }

        inference::SimdFloatBuf query_buf;
        if (auto encode_err = encoder_.Encode(request->prompt(), query_buf)) {
            switch (encode_err.value()) {
                case inference::EncodeError::kTokenLimitExceeded:
                    response->set_check_state(strix::v1::REJECTED);
                    return grpc::Status::OK;

                case inference::EncodeError::kDegeneratedVector:
                    response->set_check_state(strix::v1::REJECTED);
                    return {grpc::StatusCode::INTERNAL, "Degenerated vector"};
            }
        }

        const float* query = query_buf.data();
        TimePoint    now   = Clock::now();

        const auto opt_search_res = collector_.Search(query);
        if (opt_search_res.has_value() &&
            EvalSearchResult(opt_search_res.value(), now, response)) {
            return grpc::Status::OK;
        }

        const auto opt_node_id = collector_.AcquireSlotFor(query, now);
        if (!opt_node_id.has_value()) {
            response->set_check_state(strix::v1::REJECTED);
            return grpc::Status::OK;
        }

        response->set_check_state(strix::v1::MISS);
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
        if (payload_len > memory::kMaxPayloadLength) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "Oversized payload"};
        }

        response->set_success(collector_.CommitEntry(
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

// template <class... Ts> struct overloaded : Ts... {
//     using Ts::operator()...;
// };
// template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

bool CacheServiceImpl::EvalSearchResult(
    const collection::TopKResult<collection::kTopK>& search_res, TimePoint now,
    strix::v1::CheckCacheResponse* response
) const {
    for (uint32_t k = 0; k < search_res.count; ++k) {
        const auto [node_id, ver] = search_res.records[k];

        auto result = collector_.FetchCache(node_id, ver, now);
        if (auto* res = std::get_if<absl::Cord>(&result)) {
            response->set_check_state(strix::v1::HIT);
            response->set_cached_payload(*res);
            return true;
        } else {
            switch (std::get<MissReason>(result)) {
                case MissReason::kMiss: continue;
                case MissReason::kPendingHit:
                    response->set_check_state(strix::v1::PENDING);
                    response->set_node_id(node_id);
                    return true;
            }
        }

        // bool should_return = std::visit(overloaded{
        //     [&](absl::Cord&& cord) {
        //         response->set_check_state(strix::v1::HIT);
        //         response->set_cached_payload(cord);
        //         return true;
        //     },
        //     [&](MissReason reason) {
        //         if (reason == MissReason::kPendingHit) {
        //             response->set_check_state(strix::v1::PENDING);
        //             response->set_node_id(node_id);
        //             return true;
        //         }

        //         return false;
        //     }
        // }, std::move(result));

        // if (should_return) {
        //     return true;
        // }
    }

    return false;
}

}  // namespace strix::rpc
