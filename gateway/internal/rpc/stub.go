package rpc

import (
	"fmt"
	pb "gateway/pb/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	SocketAddress = "unix:///tmp/strix.sock"
)

func CreateRPCStub() (pb.SemanticServiceClient, *grpc.ClientConn, error) {
	conn, connErr := grpc.NewClient(
		SocketAddress, grpc.WithTransportCredentials(insecure.NewCredentials()),
	)

	if connErr != nil {
		return nil, nil, fmt.Errorf("gRPC connection error: %w", connErr)
	}

	stub := pb.NewSemanticServiceClient(conn)
	return stub, conn, nil
}
