package cli

import (
	"errors"
	"fmt"
	"gateway/internal/supervisor"
	"gateway/internal/sys"
	"log/slog"
	"os"
	"strconv"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

var (
	flagPrecache bool
	flagStrict   bool
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the HTTP Gateway as supervisor and Vector Engine",
	Long: `Loads ~/.strix/.env, forks HTTP Gateway and Vector Engine as 
	child processes. Acts as a Supervisor: on SIGTERM or SIGINT, it closes both
	Death Pipes, sending EOF to each child process, causing HTTP Gateway and
	Vector Engine to exit gracefully on its own.

	Use --precache (-p), positional arguments are treated as JSONL file paths.
	Each line must be {"prompt": "...", "payload": "..."}.
 
	Use --strict (-s) to abort on any malformed JSONL line.`,
	Args: func(cmd *cobra.Command, args []string) error {
		if flagPrecache && len(args) == 0 {
			return fmt.Errorf("--precache requires at least one JSONL file")
		}

		if !flagPrecache && len(args) > 0 {
			return fmt.Errorf(
				"unexpected arguments: %v (did you forget --precache?)", args,
			)
		}

		return nil
	},
	RunE: runServe,
}

func init() {
	serveCmd.Flags().BoolVarP(&flagPrecache, "precache", "p", false,
		"precache entries from JSONL files",
	)

	serveCmd.Flags().BoolVarP(&flagStrict, "strict", "s", false,
		"abort precaching procedure on malformed JSON line (default: skip)",
	)
}

func runServe(_ *cobra.Command, args []string) error {
	// 1. Permission check, file lock, RAM check.
	if err := assertEnvPermissions(); err != nil {
		return err
	}

	lockFile, err := acquireLockFile()
	if err != nil {
		return err
	}
	defer func() {
		_ = lockFile.Close()
	}()

	err = system.CheckRAM()
	if err != nil {
		return err
	}

	// 2. Calculate CPU affinity ratio for Process B and C.
	cpuMask, err := system.GetAffineCPUs()
	if err != nil {
		return fmt.Errorf("cannot determine CPU affinity: %w", err)
	}

	slog.Info("CPU Affinity assigned",
		slog.String("gateway_cores", cpuMask.GatewayCores),
		slog.String("engine_cores", cpuMask.EngineCores),
	)

	// 3. Resolve artifact paths.
	artPaths, err := supervisor.ResolvePaths()
	if err != nil {
		return err
	}

	slog.Info("Artifact path resolved",
		slog.String("main_binary", artPaths.MainBin),
		slog.String("engine_binary", artPaths.EngineBin),
		slog.String("tokenizer", artPaths.TokPath),
		slog.String("transformer", artPaths.BertPath),
	)

	// 4. Read ~/.strix/.env to get API key and endpoint.
	envPath, err := envFilePath()
	if err != nil {
		return err
	}

	configEnv, err := godotenv.Read(envPath)
	if err != nil {
		return fmt.Errorf("cannot read %s: %w", envPath, err)
	}

	// 5. Write Process A's PID to a file.
	err = writePIDFile()
	if err != nil {
		return err
	}
	defer removePIDFile()

	opts := supervisor.ControllerOptions{
		PrecacheFiles:  args,
		PrecacheStrict: flagStrict,
	}

	return supervisor.NewController(cpuMask, artPaths, configEnv, opts).Run()
}

func writePIDFile() error {
	pidPath, err := pidFilePath()
	if err != nil {
		return err
	}

	pidContent := strconv.Itoa(os.Getpid())

	err = os.WriteFile(pidPath, []byte(pidContent), 0600)
	if err != nil {
		return fmt.Errorf("cannot write PID file %s: %w", pidPath, err)
	}

	slog.Info("PID file written",
		slog.String("path", pidPath),
		slog.Int("pid", os.Getpid()),
	)
	return nil
}

func removePIDFile() {
	pidPath, err := pidFilePath()
	if err != nil {
		return
	}

	err = os.Remove(pidPath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		slog.Warn("Cannot remove PID file",
			slog.String("path", pidPath),
			slog.Any("error", err),
		)
	}
}
