package rpc

import (
	"context"
	"fmt"
	pb "gateway/pb/proto"
	"log/slog"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

const (
	socketAddress = "unix:///tmp/strix.sock"

	pollInterval = 100 * time.Millisecond
	pollTimeout  = 10 * time.Second
)

func CreateStub() (pb.CacheServiceClient, *grpc.ClientConn, error) {
	conn, connErr := grpc.NewClient(
		socketAddress, grpc.WithTransportCredentials(insecure.NewCredentials()),
	)

	if connErr != nil {
		return nil, nil, fmt.Errorf("gRPC connection error: %w", connErr)
	}

	stub := pb.NewCacheServiceClient(conn)
	return stub, conn, nil
}

func PollStub(ctx context.Context, stub pb.CacheServiceClient) error {
	deadline := time.Now().Add(pollTimeout)

	for attempt := 1; time.Now().Before(deadline); attempt++ {
		pingCtx, pingCancel := context.WithTimeout(ctx, pollInterval)
		_, pingErr := stub.CheckCache(
			pingCtx, &pb.CheckCacheRequest{Prompt: []byte("health_check")},
		)
		pingCancel()

		if pingErr == nil {
			slog.Info(
				fmt.Sprintf("Vector Engine ready after %d poll(s).", attempt),
			)
			return nil
		}

		if status.Code(pingErr) != codes.Unavailable {
			return fmt.Errorf("unexpected error in Vector Engine: %w", pingErr)
		}

		slog.Info(
			fmt.Sprintf("Vector Engine not ready yet (attempt %d)", attempt),
		)
		time.Sleep(pollInterval)
	}

	return fmt.Errorf("unresponsive Vector Engine after %s", pollTimeout)
}
