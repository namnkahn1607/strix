package server

import (
	"context"
	"errors"
	"fmt"
	"gateway/internal/core"
	"gateway/internal/transport"
	pb "gateway/pb/proto"
	"log"
	"net/http"
	"time"

	"github.com/VictoriaMetrics/fastcache"
)

const (
	gatewayEndpoint = "/v1/cache/strix"
	serverPort      = ":8080"
)

type strixServer struct {
	sv *http.Server
}

func newServer(
	stub pb.SemanticServiceClient, cache *fastcache.Cache,
	fatalChan chan error, pool *core.WorkerPool,
	llmAPIKey, llmEndpoint string,
) *strixServer {
	mux := http.NewServeMux()
	mainHandler := transport.StrixService(stub, cache, fatalChan, pool, llmAPIKey, llmEndpoint)
	mux.HandleFunc(gatewayEndpoint, mainHandler)

	return &strixServer{
		sv: &http.Server{
			Addr:    serverPort,
			Handler: mux,
			// Mitigate Slowloris attack
			ReadHeaderTimeout: 3 * time.Second,
			ReadTimeout:       5 * time.Second,
			WriteTimeout:      10 * time.Second,
			IdleTimeout:       60 * time.Second,
		},
	}
}

func (server *strixServer) start() <-chan error {
	serverErrChan := make(chan error, 1)

	go func() {
		log.Printf("[strix serve] HTTP Server listening on port: %s\n", serverPort)
		if serverErr := server.sv.ListenAndServe(); serverErr != nil && !errors.Is(
			serverErr, http.ErrServerClosed,
		) {
			serverErrChan <- serverErr
		}
	}()

	return serverErrChan
}

func (server *strixServer) stop(ctx context.Context) error {
	if shutdownErr := server.sv.Shutdown(ctx); shutdownErr != nil {
		return fmt.Errorf("server shutdown failed: %w", shutdownErr)
	}

	return nil
}
