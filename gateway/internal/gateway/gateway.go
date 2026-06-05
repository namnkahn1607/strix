package gateway

import (
	"context"
	"errors"
	"gateway/internal/concurrent"
	"gateway/internal/config"
	"gateway/internal/limit"
	"gateway/internal/llm"
	"gateway/internal/rpc"
	system "gateway/internal/sys"
	"io"
	"log/slog"
	"os"
	"time"

	"github.com/VictoriaMetrics/fastcache"
)

const (
	l0CacheSize = 256 * 1024 * 1024 // 256MB

	llmReqTimeout = 30 * time.Second

	defWindowSize = 1000
	defRateLimit  = 200
	defTTLMs      = 5 * 60 * 1000
)

func Execute() {
	// 1.1. Watch the Death Pipe
	deadChan := waitDeathPipe()

	// 1.2. Load HTTP Gateway's user configuration.
	cfg, err := config.Load()
	if err != nil {
		slog.Error("Failed to load configuration.", slog.Any("error", err))
		return
	}

	slog.Info("Gateway configuration loaded.", slog.String("config", cfg.String()))

	// 2. Apply GOMAXPROCS based on pinned CPU mask.
	system.ApplyGoMaxProcs(cfg.Cores)
	slog.Info("Applied GOMAXPROCS.", slog.Int("GOMAXPROCS", cfg.NumWorkers))

	// 3. Connect to Vector Engine over gRPC UNIX Socket.
	stub, conn, err := rpc.CreateStub()
	if err != nil {
		slog.Error("Failed to create gRPC stub.", slog.Any("error", err))
		return
	}

	defer func() {
		closeErr := conn.Close()
		if closeErr != nil {
			slog.Error("Failed to close gRPC connection.", slog.Any("error", closeErr))
		} else {
			slog.Info("Closed gRPC connection.")
		}
	}()

	// 4. Poll until Vector Engine is ready.
	slog.Info("Waiting for Vector Engine to become ready...")

	err = rpc.PollStub(context.Background(), stub)
	if err != nil {
		slog.Error("Vector Engine unresponsive after polling.", slog.Any("error", err))
		return
	}

	// 5. Build and start the HTTP Server.
	l0Cache := fastcache.New(l0CacheSize)
	defer l0Cache.Reset()

	err = CreateGateway(Dependencies{
		Stub:       stub,
		L0Cache:    l0Cache,
		Pool:       concurrent.NewWorkerPool(stub, cfg.NumWorkers),
		HerdCtrl:   NewHerdController(),
		IPLimiter:  limit.NewLimiter(defWindowSize, defRateLimit, defTTLMs),
		DeadChan:   deadChan,
		LLMClient:  llm.CreateClient(cfg.APIKey, cfg.Endpoint, llmReqTimeout),
		PromptPath: cfg.PromptPath,
	}).Run()
	if err != nil {
		slog.Error("A fatal error occurred in HTTP Gateway.",
			slog.Any("error", err),
		)
	}
}

func waitDeathPipe() <-chan struct{} {
	ch := make(chan struct{})

	go func() {
		defer close(ch)

		deathPipe := os.NewFile(3, "death_pipe")
		if deathPipe == nil {
			slog.Error("FD 3 is not a valid Death Pipe")
			return
		}

		defer func() {
			_ = deathPipe.Close()
		}()

		buf := make([]byte, 1)
		_, err := deathPipe.Read(buf)
		if err == nil || errors.Is(err, io.EOF) {
			return
		}

		slog.Error("Failed to read Death Pipe.", slog.Any("error", err))
	}()

	return ch
}
