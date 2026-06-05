package cli

import (
	"fmt"
	"os"
	"syscall"
	"time"

	"github.com/spf13/cobra"
)

const (
	stopPollInterval = 200 * time.Millisecond
	stopPollTimeout  = 10 * time.Second
)

var stopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Send SIGTERM to the running Strix instance",
	Long: `Reads ~/.strix/strix.pid and sends SIGTERM to the recorded process.
 
	The Go process catches SIGTERM, shuts down the HTTP server gracefully,
	closes the Death Pipe - causing the Vector Engine to exit cleanly.
 
	If no PID file exists, Strix is not running and this command is a no-op.`,
	RunE: runStop,
}

func runStop(_ *cobra.Command, _ []string) error {
	pid, running, checkErr := isInstanceRunning()
	if checkErr != nil {
		return fmt.Errorf("cannot read PID file: %w", checkErr)
	}

	if !running {
		fmt.Println("[strix stop] Strix is not running.")
		return nil
	}

	proc, _ := os.FindProcess(pid)
	if sigErr := proc.Signal(syscall.SIGTERM); sigErr != nil {
		return fmt.Errorf("cannot send SIGTERM to PID %d: %w", pid, sigErr)
	}

	fmt.Printf(
		"[strix stop] Sent SIGTERM to Strix (PID %d). Waiting for shutdown...\n",
		pid,
	)

	deadline := time.Now().Add(stopPollTimeout)
	for time.Now().Before(deadline) {
		time.Sleep(stopPollInterval)
		if sigErr := proc.Signal(syscall.Signal(0)); sigErr != nil {
			fmt.Printf("[strix stop] Strix (PID %d) has stopped.\n", pid)
			return nil
		}
	}

	return fmt.Errorf(
		"the service (PID %d) did not stop within %s - it may still be shutting down. "+
			"Check with: kill -0 %d",
		pid, stopPollTimeout, pid,
	)
}
