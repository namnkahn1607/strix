package main

import (
	"gateway/internal/cli"
	"gateway/internal/server"
	"log"
	"os"
)

func main() {
	if os.Getenv("STRIX_WORKER") == "1" {
		// Process B: HTTP Gateway
		// forked by Supervisor in Process A.
		gatewayCores := os.Getenv("GATEWAY_CORES")
		if gatewayCores == "" {
			log.Fatal("[Gateway] FATAL: GATEWAY_CORES is not set")
		}

		gatewayErr := server.RunGateway(gatewayCores)
		if gatewayErr != nil {
			log.Fatalf("[Gateway] Exited with error: %v\n", gatewayErr)
		}

		return
	}

	// Process A: Supervisor + CLI
	// Default mode, handle all CLI commands.
	cli.Execute()
}
