package rpc

import (
	"fmt"
	pb "gateway/pb/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const socketAddress = "unix:///tmp/strix.sock"

func CreateRPCStub() (pb.CacheServiceClient, *grpc.ClientConn, error) {
	conn, connErr := grpc.NewClient(
		socketAddress, grpc.WithTransportCredentials(insecure.NewCredentials()),
	)

	if connErr != nil {
		return nil, nil, fmt.Errorf("gRPC connection error: %w", connErr)
	}

	stub := pb.NewCacheServiceClient(conn)
	return stub, conn, nil
}
