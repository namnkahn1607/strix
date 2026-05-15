package server

import (
	"context"
	"errors"
	"fmt"
	"gateway/internal/config"
	"gateway/internal/core"
	system "gateway/internal/sys"
	pb "gateway/pb/proto"
	"io"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/VictoriaMetrics/fastcache"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

const (
	socketAddress = "unix:///tmp/strix.sock"

	l0CacheSize = 256 * 1024 * 1024

	pollInterval    = 100 * time.Millisecond
	pollTimeout     = 10 * time.Second
	shutdownTimeout = 3 * time.Second
)

func RunGateway(cfg config.GatewayConfig) error {
	// 1. Apply GOMAXPROCS based on pinned CPU mask.
	system.ApplyGoMaxProcs(cfg.Cores)
	log.Printf("[Gateway] GOMAXPROCS = %d\n", cfg.NumWorkers)

	// 2. Watch the Death Pipe (FD 3).
	deathChan := waitDeathPipe()

	// 3. Connect to Vector Engine over gRPC Unix socket.
	conn, connErr := grpc.NewClient(
		socketAddress,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if connErr != nil {
		return connErr
	}

	defer func() {
		if closeErr := conn.Close(); closeErr != nil {
			log.Printf("[Gateway] gRPC connection close error: %v\n", closeErr)
		} else {
			log.Println("[Gateway] gRPC connection closed")
		}
	}()

	clientStub := pb.NewSemanticServiceClient(conn)

	// 4. Poll until Vector Engine is ready.
	log.Printf("[Gateway] Waiting for Vector Engine to become ready...")
	if pollErr := pollEngine(context.Background(), clientStub); pollErr != nil {
		return pollErr
	}

	// 5. Initialize L0 exact-match cache (off-heap).
	log.Printf(
		"[gateway] Allocating %d MB off-heap memory for L0 Fast Cache...\n",
		l0CacheSize/(1024*1024),
	)
	l0Cache := fastcache.New(l0CacheSize)
	defer l0Cache.Reset()

	// 6. Build and start the HTTP server.
	fatalErrChan := make(chan error, 1)
	pool := core.NewWorkerPool(clientStub, 2*cfg.NumWorkers)
	sv := newServer(clientStub, l0Cache, fatalErrChan, pool, cfg.APIKey, cfg.Endpoint)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	serverErrChan := sv.start()

	// 7. Block until one of the four shutdown triggers fires.
	var runErr error

	select {
	case <-deathChan:
		log.Println(
			"[Gateway] Death Pipe EOF - Supervisor or sibling process died. Initiating shutdown...",
		)

	case serverErr := <-serverErrChan:
		runErr = serverErr
		log.Printf("[Gateway] HTTP server crashed: %v\n", runErr)

	case fatalErr := <-fatalErrChan:
		log.Printf("[Gateway] Fatal handler error: %v - initiating shutdown...\n", fatalErr)

	case sig := <-sigChan:
		log.Printf("[Gateway] Received signal %v - initiating shutdown...\n", sig)
	}

	// 8. HTTP Gateway graceful shutdown.
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer shutdownCancel()

	log.Println("[Gateway] Stopping HTTP server...")
	if stopErr := sv.stop(shutdownCtx); stopErr != nil {
		log.Printf("[Gateway] HTTP server stop error: %v\n", stopErr)
	}

	defer func() {
		if poolErr := pool.Stop(shutdownCtx); poolErr != nil {
			log.Printf("[Gateway] Worker Pool stop error: %v\n", poolErr)
		} else {
			log.Println("[Gateway] Worker Pool stopped.")
		}
	}()

	return runErr
}

func waitDeathPipe() <-chan struct{} {
	ch := make(chan struct{})

	go func() {
		defer close(ch)

		deathPipe := os.NewFile(3, "death_pipe")
		if deathPipe == nil {
			log.Println("[Gateway] WARNING: FD 3 is not a valid Death Pipe")
			return
		}

		defer func() {
			_ = deathPipe.Close()
		}()

		buf := make([]byte, 1)
		_, readErr := deathPipe.Read(buf)
		if readErr == nil || errors.Is(readErr, io.EOF) {
			return
		}

		log.Printf("[Gateway] Death Pipe read error: %v\n", readErr)
	}()

	return ch
}

func pollEngine(ctx context.Context, stub pb.SemanticServiceClient) error {
	deadline := time.Now().Add(pollTimeout)

	for attempt := 1; time.Now().Before(deadline); attempt++ {
		pingCtx, pingCancel := context.WithTimeout(ctx, pollInterval)
		_, pingErr := stub.CheckCache(pingCtx, &pb.CheckCacheRequest{Prompt: []byte("health_check")})
		pingCancel()

		if pingErr == nil {
			log.Printf("[Gateway] Vector Engine ready after %d poll(s).\n", attempt)
			return nil
		}

		if status.Code(pingErr) != codes.Unavailable {
			return fmt.Errorf("unexpected error in Vector Engine: %w", pingErr)
		}

		log.Printf(
			"[Gateway] Vector Engine not ready yet (attempt %d). Retrying in %s...\n",
			attempt, pollInterval,
		)
		time.Sleep(pollInterval)
	}

	return fmt.Errorf("FATAL: Vector Engine unresponsive after %s", pollTimeout)
}
