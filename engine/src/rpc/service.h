// Author: namnkahn1607
//
// CacheServiceImpl: gRPC service handler for CheckCache and SetCache RPCs.
// Orchestrates Embedder and MemoryArena to serve the semantic cache protocol.

#pragma once

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#include "arena.h"
#include "inference.h"
#include "strix.grpc.pb.h"
#include "strix.pb.h"

// CacheServiceImpl
//
// gRPC service implementation for the Strix semantic cache.
// Responsibilities:
//   - Validate incoming RPC fields.
//   - Orchestrate Vectorization -> Vector search -> Payload read/write.
//   - Translate results and errors into gRPC Status codes and response protos.
//
// Holds non-owning references to `Embedder` and `MemoryArena`.
class CacheServiceImpl final : public proto::CacheService::Service {
public:
    explicit CacheServiceImpl(const Embedder& embedder, MemoryArena& arena);

    CacheServiceImpl(const CacheServiceImpl&)            = delete;
    CacheServiceImpl& operator=(const CacheServiceImpl&) = delete;

    // Encodes the request prompt, perform searching for a similar vector,
    // and returns the cached payload if a match above `kSimilarityThreshold` is
    // found.
    grpc::Status CheckCache(grpc::ServerContext*            context,
                            const proto::CheckCacheRequest* request,
                            proto::CheckCacheResponse*      response) override;

    // Encodes the request prompt, writes the vector into a free slot, and
    // commits the payload to the ring buffer.
    grpc::Status SetCache(grpc::ServerContext*          context,
                          const proto::SetCacheRequest* request,
                          proto::SetCacheResponse*      response) override;

private:
    const Embedder& embedder_;
    MemoryArena&    arena_;
};
