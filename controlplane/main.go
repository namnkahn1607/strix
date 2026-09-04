package main

import (
	"os"
	"strix/cli"
	"strix/gateway"
)

func main() {
	if os.Getenv("STRIX_GATEWAY") == "1" {
		// Process B: HTTP Gateway
		// Forked by Supervisor in Process A.
		gateway.Execute()
	} else {
		// Process A: Supervisor
		// Default mode, handle all CLI commands.
		cli.Execute()
	}
}
