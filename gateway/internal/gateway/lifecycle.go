package gateway

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

const shutdownTimeout = 3 * time.Second

func (g *Gateway) Run() error {
	serverErrChan := make(chan error, 1)

	go func() {
		slog.Info("HTTP Server is listening", slog.String("address", g.httpServer.Addr))

		serveErr := g.httpServer.ListenAndServe()
		if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			serverErrChan <- serveErr
		}
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	var runErr error

	select {
	case <-g.DeadChan:
		slog.Info("Death Pipe EOF. HTTP Gateway shutting down...")

	case serverErr := <-serverErrChan:
		runErr = serverErr
		slog.Warn("HTTP Server crashed.", slog.Any("error", serverErr))

	case sig := <-sigChan:
		slog.Info("Received OS signal. HTTP Gateway shutting down...",
			slog.Any("os_signal", sig),
		)
	}

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer shutdownCancel()

	err := g.httpServer.Shutdown(shutdownCtx)
	if err != nil {
		slog.Warn("Failed to stop HTTP Server.", slog.Any("error", err))
	}

	err = g.Pool.Stop(shutdownCtx)
	if err != nil {
		slog.Warn("Failed to stop Worker Pool.", slog.Any("error", err))
	}

	g.IPLimiter.Stop()

	return runErr
}
