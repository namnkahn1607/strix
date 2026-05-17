package cli

import (
	"errors"
	"fmt"
	"gateway/internal/supervisor"
	"gateway/internal/sys"
	"log"
	"os"
	"strconv"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the HTTP Gateway as supervisor and Vector Engine",
	Long: `Loads ~/.strix/.env, forks Vector Engine, and starts HTTP Gateway
	in the current process. Acts as a supervisor: on SIGTERM or SIGINT
	it shuts down the HTTP server gracefully, then the Death Pipe EOF signal
	causes the Vector Engine to exit cleanly on its own.`,
	RunE: runServe,
}

func runServe(_ *cobra.Command, _ []string) error {
	// 1. Permission & RAM checks and FD expansion.
	if permErr := AssertEnvPermissions(); permErr != nil {
		return permErr
	}

	if pid, running, pidErr := IsInstanceRunning(); pidErr != nil {
		return fmt.Errorf("cannot check for existing instance: %w", pidErr)
	} else if running {
		return fmt.Errorf("another Strix instance is already running (PID: %d)", pid)
	}

	if ramErr := system.CheckRAM(); ramErr != nil {
		return ramErr
	}

	// 2. Calculate CPU affinity ratio for Process B and C.
	mask, maskErr := system.GetAffineCPUs()
	if maskErr != nil {
		return fmt.Errorf("cannot determine CPU affinity: %w", maskErr)
	}

	log.Printf("[strix serve] Gateway cores : %s\n", mask.GatewayCores)
	log.Printf("[strix serve] Engine cores: %s\n", mask.EngineCores)

	// 3. Resolve artifact paths.
	paths, pathErr := supervisor.ResolvePaths()
	if pathErr != nil {
		return pathErr
	}

	log.Printf("[strix serve] Strix Binary: %s\n", paths.MainBin)
	log.Printf("[strix serve] Strix Engine binary: %s\n", paths.EngineBin)
	log.Printf("[strix serve] Inference model: %s\n", paths.ModelPath)

	// 4. Read ~/.strix/.env to get API key and endpoint.
	envPath, envErr := EnvFilePath()
	if envErr != nil {
		return envErr
	}

	configEnv, readErr := godotenv.Read(envPath)
	if readErr != nil {
		return fmt.Errorf("cannot read %s: %w", envPath, readErr)
	}

	// 5. Write Process A's PID to a file.
	writePIDErr := writePIDFile()
	if writePIDErr != nil {
		return writePIDErr
	}

	defer removePIDFile()

	return supervisor.NewController(mask, paths, configEnv).Run()
}

func writePIDFile() error {
	pidPath, pidPathErr := PIDFilePath()
	if pidPathErr != nil {
		return pidPathErr
	}

	pidContent := strconv.Itoa(os.Getpid())
	if writeErr := os.WriteFile(pidPath, []byte(pidContent), 0600); writeErr != nil {
		return fmt.Errorf("cannot write PID file %s: %w", pidPath, writeErr)
	}

	log.Printf("[strix serve] PID file written: %s (PID %d)\n", pidPath, os.Getpid())
	return nil
}

func removePIDFile() {
	pidPath, pidPathErr := PIDFilePath()
	if pidPathErr != nil {
		return
	}

	if rmErr := os.Remove(pidPath); rmErr != nil && !errors.Is(rmErr, os.ErrNotExist) {
		log.Printf(
			"[strix serve] WARNING: cannot remove PID file %s: %v\n",
			pidPath, rmErr,
		)
	}
}
