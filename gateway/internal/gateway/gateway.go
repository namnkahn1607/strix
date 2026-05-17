package gateway

import (
	"context"
	"errors"
	"fmt"
	"gateway/internal/concurrent"
	"gateway/internal/config"
	"gateway/internal/limit"
	"gateway/internal/llm"
	"gateway/internal/rpc"
	system "gateway/internal/sys"
	pb "gateway/pb/proto"
	"io"
	"log"
	"os"
	"time"

	"github.com/VictoriaMetrics/fastcache"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	l0CacheSize = 256 * 1024 * 1024 // 256MB

	pollInterval = 100 * time.Millisecond
	pollTimeout  = 10 * time.Second

	llmReqTimeout = 90 * time.Second

	defWindowSize = 1000
	defRateLimit  = 200
	defTTLMs      = 5 * 60 * 1000
)

func Execute() {
	// 1.1. Watch the Death Pipe
	deadChan := waitDeathPipe()

	// 1.2. Load HTTP Gateway's user configuration.
	cfg, loadErr := config.Load()
	if loadErr != nil {
		log.Printf("[Gateway] cannot load configuration: %v", loadErr)
	}

	// 2. Apply GOMAXPROCS based on pinned CPU mask.
	system.ApplyGoMaxProcs(cfg.Cores)
	log.Printf("[Gateway] GOMAXPROCS = %d\n", cfg.NumWorkers)

	// 3. Connect to Vector Engine over gRPC UNIX Socket.
	stub, conn, createErr := rpc.CreateRPCStub()
	if createErr != nil {
		log.Printf("[Gateway] %v\n", createErr)
	}

	defer func() {
		closeErr := conn.Close()
		if closeErr != nil {
			log.Printf("[Gateway] gRPC connection close error: %v\n", closeErr)
		} else {
			log.Println("[Gateway] gRPC connection closed")
		}
	}()

	// 4. Poll until Vector Engine is ready.
	log.Printf("[Gateway] Waiting for Vector Engine to become ready...")
	if pollErr := pollEngine(context.Background(), stub); pollErr != nil {
		log.Printf("[Gateway] %v\n", pollErr)
	}

	// 5. Build and start the HTTP Server.
	l0Cache := fastcache.New(l0CacheSize)
	defer l0Cache.Reset()

	gatewayErr := CreateGateway(Dependencies{
		Stub:      stub,
		L0Cache:   l0Cache,
		Pool:      concurrent.NewWorkerPool(stub, 2*cfg.NumWorkers),
		HerdCtrl:  NewHerdController(),
		IPLimiter: limit.NewLimiter(defWindowSize, defRateLimit, defTTLMs),
		DeadChan:  deadChan,
		LLMClient: llm.CreateClient(cfg.APIKey, cfg.Endpoint, llmReqTimeout),
	}).Run()
	if gatewayErr != nil {
		log.Printf("Gateway error: %v\n", gatewayErr)
	}
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
		_, pingErr := stub.CheckCache(
			pingCtx, &pb.CheckCacheRequest{Prompt: []byte("health_check")},
		)
		pingCancel()

		if pingErr == nil {
			log.Printf(
				"[Gateway] Vector Engine ready after %d poll(s).\n", attempt,
			)
			return nil
		}

		if status.Code(pingErr) != codes.Unavailable {
			return fmt.Errorf("unexpected error in Vector Engine: %w", pingErr)
		}

		log.Printf(
			"[Gateway] Vector Engine not ready yet (attempt %d). "+
				"Retrying in %s...\n", attempt, pollInterval,
		)

		time.Sleep(pollInterval)
	}

	return fmt.Errorf("FATAL: Vector Engine unresponsive after %s", pollTimeout)
}
