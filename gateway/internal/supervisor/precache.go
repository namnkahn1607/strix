package supervisor

import (
	"context"
	"fmt"
	"gateway/internal/rpc"
	"gateway/internal/supervisor/precache"
	"log/slog"
)

type ControllerOptions struct {
	PrecacheFiles  []string
	PrecacheStrict bool
}

// runPrecache creates a gRPC stub, polls until the Vector Engine is ready,
// then runs the precache pipeline.
func (c *Controller) runPrecache(ctx context.Context) error {
	stub, conn, err := rpc.CreateStub()
	if err != nil {
		return fmt.Errorf("cannot create gRPC stub for precache: %w", err)
	}
	defer func() {
		_ = conn.Close()
	}()

	err = rpc.PollStub(context.Background(), stub)
	if err != nil {
		return err
	}

	slog.Info("Starting precache",
		slog.Int("file_count", len(c.opts.PrecacheFiles)),
		slog.Bool("strict", c.opts.PrecacheStrict),
	)

	pipeline := precache.NewPipeLine(stub, c.opts.PrecacheFiles, c.opts.PrecacheStrict)
	return pipeline.Run(ctx)
}
