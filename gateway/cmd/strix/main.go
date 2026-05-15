package main

import (
	"gateway/internal/cli"
	"gateway/internal/config"
	"gateway/internal/server"
	"log"
	"os"
)

func main() {
	if os.Getenv("STRIX_WORKER") == "1" {
		// Process B: HTTP Gateway
		// forked by Supervisor in Process A.
		cfg, configErr := config.Load()
		if configErr != nil {
			log.Printf("[Gateway] Configuration error: %v\n", configErr)
		}

		gatewayErr := server.RunGateway(cfg)
		if gatewayErr != nil {
			log.Printf("[Gateway] Exited with error: %v\n", gatewayErr)
		}

		return
	}

	// Process A: Supervisor + CLI
	// Default mode, handle all CLI commands.
	cli.Execute()
}
