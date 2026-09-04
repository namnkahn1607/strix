// RPC service handler.

#pragma once

#include <grpcpp/grpcpp.h>

#include <chrono>

#include "cache.grpc.pb.h"
#include "cache.pb.h"
#include "collection/collection.h"
#include "collection/search.h"
#include "inference/sentence_encoder.h"

namespace strix::rpc {

// Strix semantic cache service.
class CacheServiceImpl final : public strix::v1::CacheService::Service {
public:
    explicit CacheServiceImpl(
        const inference::SentenceEncoder& encoder,
        collection::Collection&           collector
    );

    CacheServiceImpl(const CacheServiceImpl&)            = delete;
    CacheServiceImpl& operator=(const CacheServiceImpl&) = delete;
    CacheServiceImpl(CacheServiceImpl&&)                 = delete;
    CacheServiceImpl& operator=(CacheServiceImpl&&)      = delete;

    // Checks for cached payload for the requested prompt.
    grpc::Status CheckCache(
        grpc::ServerContext*                context,
        const strix::v1::CheckCacheRequest* request,
        strix::v1::CheckCacheResponse*      response
    ) override;

    // Sets new payload for a new entry in cache.
    grpc::Status SetCache(
        grpc::ServerContext* context, const strix::v1::SetCacheRequest* request,
        strix::v1::SetCacheResponse* response
    ) override;

private:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    // Populates cache search result.
    // Returns `true` and presets `response` if the evaluation produces either
    // HIT or PENDING_HIT; otherwise leaves `response` untouched.
    bool EvalSearchResult(
        const collection::TopKResult<collection::kTopK>& search_res,
        TimePoint now, strix::v1::CheckCacheResponse* response
    ) const;

    const inference::SentenceEncoder& encoder_;
    collection::Collection&           collector_;
};

}  // namespace strix::rpc
