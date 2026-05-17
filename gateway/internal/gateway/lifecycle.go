package gateway

import (
	"context"
	"errors"
	"log"
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
		log.Printf(
			"[Gateway] HTTP Server listening on %s\n", g.httpServer.Addr,
		)

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
		log.Println("[Gateway] Death Pipe EOF. Initiating shutdown...")

	case serverErr := <-serverErrChan:
		runErr = serverErr
		log.Printf("[Gateway] HTTP Server crashed: %v\n", serverErr)

	case sig := <-sigChan:
		log.Printf(
			"[Gateway] Received signal %v. Initiating shutdown...\n", sig,
		)
	}

	shutdownCtx, shutdownCancel := context.WithTimeout(
		context.Background(), shutdownTimeout,
	)
	defer shutdownCancel()

	log.Println("[Gateway] Stopping HTTP Server...")

	stopErr := g.httpServer.Shutdown(shutdownCtx)
	if stopErr != nil {
		log.Printf("[Gateway] HTTP Server stop error: %v\n", stopErr)
	}

	poolErr := g.Pool.Stop(shutdownCtx)
	if poolErr != nil {
		log.Printf("[Gateway] Worker Pool stop error: %v\n", poolErr)
	}

	g.IPLimiter.Stop()

	return runErr
}
