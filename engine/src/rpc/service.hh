//
// rpc/service.hh
//

#pragma once

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#include "arena.hh"
#include "inference.hh"
#include "strix.grpc.pb.h"
#include "strix.pb.h"

// --- CacheServiceImpl ---
// Responsibilities: validate gRPC input, orchestrate Embedder + MemoryArena,
// and translate results into gRPC responses.
class CacheServiceImpl final : public proto::CacheService::Service {
public:
    explicit CacheServiceImpl(const Embedder& embedder, MemoryArena& arena);

    // No Copy/Move semantics
    CacheServiceImpl(const CacheServiceImpl&) = delete;
    CacheServiceImpl& operator=(const CacheServiceImpl&) = delete;

    grpc::Status CheckCache(grpc::ServerContext*            context,
                            const proto::CheckCacheRequest* request,
                            proto::CheckCacheResponse*      response) override;

    grpc::Status SetCache(grpc::ServerContext*          context,
                          const proto::SetCacheRequest* request,
                          proto::SetCacheResponse*      response) override;

private:
    const Embedder& embedder;
    MemoryArena&    arena;
};
