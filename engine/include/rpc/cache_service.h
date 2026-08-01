// Author: namnkahn1607
//
// CacheServiceImpl: gRPC service handler for CheckCache and SetCache RPCs.
// Orchestrates Embedder and MemoryArena to serve the semantic cache protocol.

#pragma once

#include <grpcpp/grpcpp.h>

#include "index/vector_index.h"
#include "inference/inference_model.h"
#include "strix.grpc.pb.h"
#include "strix.pb.h"

// `CacheServiceImpl`, gRPC service implementation for the Strix semantic cache.
//
// Responsibilities:
//   - Validate incoming RPC fields.
//   - Orchestrate Vectorization -> Vector search -> Payload read/write.
//   - Translate results and errors into gRPC Status codes and response protos.
class CacheServiceImpl final : public proto::v1::CacheService::Service {
public:
    // `CacheServiceImpl` holds references to `Embedder` and `MemoryArena`.
    explicit CacheServiceImpl(const Embedder& embedder, VectorIndex& index);

    CacheServiceImpl(const CacheServiceImpl&)            = delete;
    CacheServiceImpl& operator=(const CacheServiceImpl&) = delete;
    CacheServiceImpl(CacheServiceImpl&&)                 = delete;
    CacheServiceImpl& operator=(CacheServiceImpl&&)      = delete;

    // `CheckCache()` encodes the request prompt, perform searching for a
    // similar vector, and returns the cached payload if a match above
    // the `kSimilarityThreshold` is found.
    grpc::Status CheckCache(
        grpc::ServerContext*                context,
        const proto::v1::CheckCacheRequest* request,
        proto::v1::CheckCacheResponse*      response
    ) override;

    // `SetCache()` encodes the request prompt, writes the vector into a free
    // slot, and commits the payload to the ring buffer.
    grpc::Status SetCache(
        grpc::ServerContext* context, const proto::v1::SetCacheRequest* request,
        proto::v1::SetCacheResponse* response
    ) override;

private:
    const Embedder& embedder_;
    VectorIndex&    index_;

    // `ProcessOutcome()` is used by `CheckCache` to populate from a raw search
    // result `candidate` into the `response`.
    // Returns true and fully presets `response` if `candidate` resolves to
    // either `kHit` or `kPendingHit`; otherwise false and leaves `response` be
    // untouched, delegating further decisions to the caller.
    bool ProcessCandidate(
        const SearchOutcome& candidate, uint64_t timestamp,
        proto::v1::CheckCacheResponse* response
    ) const;
};
