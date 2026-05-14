//
// Created by nlnk on Apr 17, 26.
//

#ifndef STRIX_ENGINE_SERVICE_HH
#define STRIX_ENGINE_SERVICE_HH

#include "embedder.hh"
#include "proto/strix.grpc.pb.h"

class MemoryArena;

class SemanticServiceImpl final : public proto::SemanticService::Service {
public:
    explicit SemanticServiceImpl(const Embedder& embedder, MemoryArena& arena);

    // No Copy/Move constructor
    SemanticServiceImpl(const SemanticServiceImpl&) = delete;

    // The READ gRPC method
    grpc::Status CheckCache(grpc::ServerContext* context,
                            const proto::CheckCacheRequest* request,
                            proto::CheckCacheResponse* response) override;

    // The WRITE gRPC method
    grpc::Status SetCache(grpc::ServerContext* context,
                          const proto::SetCacheRequest* request,
                          proto::SetCacheResponse* response) override;

private:
    const Embedder& embedder_;
    MemoryArena& memory_arena;
};

#endif  // STRIX_ENGINE_SERVICE_HH
