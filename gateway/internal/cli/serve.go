package cli

import (
	"errors"
	"fmt"
	"gateway/internal/supervisor"
	"gateway/internal/sys"
	"log"
	"os"
	"strconv"
	"syscall"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the HTTP Gateway as supervisor and Vector Engine",
	Long: `Loads ~/.strix/.env, forks HTTP Gateway and Vector Engine as 
	child processes. Acts as a Supervisor: on SIGTERM or SIGINT, it closes both
	Death Pipes, sending EOF to each child process, causing HTTP Gateway and
	Vector Engine to exit gracefully on its own.`,
	RunE: runServe,
}

func runServe(_ *cobra.Command, _ []string) error {
	// 1. Avoid another 'serve' by initiating lock file.
	lockPath, pathErr := LockFilePath()
	if pathErr != nil {
		return pathErr
	}

	lockFile, fileErr := os.OpenFile(
		lockPath, os.O_CREATE|syscall.O_CLOEXEC, ownPermission,
	)
	if fileErr != nil {
		return fmt.Errorf("cannot create lock file: %w", fileErr)
	}

	defer func() {
		_ = lockFile.Close()
	}()

	lockErr := syscall.Flock(
		int(lockFile.Fd()), syscall.LOCK_EX|syscall.LOCK_NB,
	)
	if lockErr != nil {
		if errors.Is(lockErr, syscall.EWOULDBLOCK) ||
			errors.Is(lockErr, syscall.EAGAIN) {
			return fmt.Errorf(
				"FATAL: Another Strix instance is already running",
			)
		}

		return fmt.Errorf("cannot acquire OS lock: %w", lockErr)
	}

	// 2. Perform permission & RAM checking.
	permErr := AssertEnvPermissions()
	if permErr != nil {
		return permErr
	}

	ramErr := system.CheckRAM()
	if ramErr != nil {
		return ramErr
	}

	// 3. Calculate CPU affinity ratio for Process B and C.
	mask, maskErr := system.GetAffineCPUs()
	if maskErr != nil {
		return fmt.Errorf("cannot determine CPU affinity: %w", maskErr)
	}

	log.Printf("[strix serve] Gateway cores : %s\n", mask.GatewayCores)
	log.Printf("[strix serve] Engine cores: %s\n", mask.EngineCores)

	// 4. Resolve artifact paths.
	artPaths, pathErr := supervisor.ResolvePaths()
	if pathErr != nil {
		return pathErr
	}

	log.Printf("[strix serve] Strix binary: %s\n", artPaths.MainBin)
	log.Printf("[strix serve] Strix Engine binary: %s\n", artPaths.EngineBin)
	log.Printf("[strix serve] Tokenizer: %s\n", artPaths.TokPath)
	log.Printf("[strix serve] Transformer model: %s\n", artPaths.BertPath)
	log.Printf("[strix serve] Dictionary: %s\n", artPaths.DictPath)

	// 5. Read ~/.strix/.env to get API key and endpoint.
	envPath, pathErr := EnvFilePath()
	if pathErr != nil {
		return pathErr
	}

	configEnv, readErr := godotenv.Read(envPath)
	if readErr != nil {
		return fmt.Errorf("cannot read %s: %w", envPath, readErr)
	}

	// 6. Write Process A's PID to a file.
	writePIDErr := writePIDFile()
	if writePIDErr != nil {
		return writePIDErr
	}

	defer removePIDFile()

	return supervisor.NewController(mask, artPaths, configEnv).Run()
}

func writePIDFile() error {
	pidPath, pidPathErr := PIDFilePath()
	if pidPathErr != nil {
		return pidPathErr
	}

	pidContent := strconv.Itoa(os.Getpid())

	writeErr := os.WriteFile(pidPath, []byte(pidContent), 0600)
	if writeErr != nil {
		return fmt.Errorf("cannot write PID file %s: %w", pidPath, writeErr)
	}

	log.Printf(
		"[strix serve] PID file written: %s (PID %d)\n", pidPath, os.Getpid(),
	)
	return nil
}

func removePIDFile() {
	pidPath, pidPathErr := PIDFilePath()
	if pidPathErr != nil {
		return
	}

	rmErr := os.Remove(pidPath)
	if rmErr != nil && !errors.Is(rmErr, os.ErrNotExist) {
		log.Printf(
			"[strix serve] WARNING: cannot remove PID file %s: %v\n",
			pidPath, rmErr,
		)
	}
}
