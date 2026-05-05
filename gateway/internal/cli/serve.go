package cli

import (
	"context"
	"errors"
	"fmt"
	"gateway/internal/proxy"
	"gateway/internal/server"
	"gateway/internal/sys"
	pb "gateway/pb/proto"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/VictoriaMetrics/fastcache"
	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

const (
	socketAddress = "unix:///tmp/strix.sock"

	maxFD       = 65536
	l0CacheSize = 256 * 1024 * 1024

	pollInterval    = 100 * time.Millisecond
	pollTimeout     = 10 * time.Second
	shutdownTimeout = 3 * time.Second
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

// strix/
// ├── bin/
// │   ├── strix_gateway  <- os.Executable()
// │   └── strix_engine
// ├── engine/
// │   └── model/
// │       ├── strix-minilm-with-tokenizer.onnx
// │	   └── libortextensions.so
// └── gateway/           <- Go Gateway source code

type projectPaths struct {
	engineBinary string // strix/bin/strix_engine
	modelPath    string // strix/engine/model/strix-minilm-with-tokenizer.onnx
	ortLibPath   string // strix/engine/model/libortextensions.so
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

	if fdErr := openMoreFD(); fdErr != nil {
		return fdErr
	}

	// 2. Calculate CPU affinity ratio for each process.
	goCores := system.ApplyGoLimits()
	log.Printf("[strix serve] GOMAXPROCS = %d\n", goCores)
	cppCoresStr := system.GenCppLimits(goCores)

	// 2. Resolve artifact paths.
	paths, pathErr := resolvePaths()
	if pathErr != nil {
		return pathErr
	}

	log.Printf("[strix serve] Vector Engine binary: %s\n", paths.engineBinary)
	log.Printf("[strix serve] Inference model: %s\n", paths.modelPath)
	log.Printf("[strix serve] ORT extensions library: %s\n", paths.ortLibPath)

	// 3. Load ~/.strix/.env into the process environment.
	envPath, pathErr := EnvFilePath()
	if pathErr != nil {
		return pathErr
	}

	if loadErr := godotenv.Load(envPath); loadErr != nil {
		return fmt.Errorf("cannot load %s: %w", envPath, loadErr)
	}

	log.Println("[strix serve] Configuration loaded from .env")

	// 4. Fork a C++ process running Vector Engine.
	reader, writer, pipeErr := os.Pipe()
	if pipeErr != nil {
		return fmt.Errorf("cannot create Death Pipe: %w", pipeErr)
	}

	engineProc := exec.Command("taskset", "-c", cppCoresStr, paths.engineBinary)
	fmt.Printf("Pinned Vector Engine to cores (via taskset): %s\n", cppCoresStr)

	engineProc.ExtraFiles = []*os.File{reader}
	engineProc.Stdout = os.Stdout
	engineProc.Stderr = os.Stderr
	engineProc.Env = buildEngineEnv(paths)

	if startErr := engineProc.Start(); startErr != nil {
		_ = reader.Close()
		_ = writer.Close()

		return fmt.Errorf("cannot start Vector Engine at %q: %w",
			paths.engineBinary, startErr,
		)
	}

	log.Printf("[strix serve] Vector Engine started (PID %d)", engineProc.Process.Pid)

	defer func() {
		if closeErr := writer.Close(); closeErr != nil {
			log.Printf(
				"[strix serve] CRITICAL: Death Pipe Write-end close failed: %v. "+
					"Sending SIGTERM as fallback...\n",
				closeErr,
			)

			_ = engineProc.Process.Signal(syscall.SIGTERM)
		}
	}()

	if closeErr := reader.Close(); closeErr != nil {
		return fmt.Errorf("cannot close Read-end of the Death Pipe: %v", closeErr)
	}

	go func() {
		if waitErr := engineProc.Wait(); waitErr != nil {
			log.Printf("[strix serve] Vector Engine exited: %v", waitErr)
		} else {
			log.Println("[strix serve] Vector Engine exited cleanly")
		}
	}()

	// 5. Initialize L1 Exact-match Cache.
	log.Printf(
		"[strix serve] Allocating %dMB off-heap memory for L1 Fast Cache...\n",
		l0CacheSize/(1024*1024),
	)
	l0Cache := fastcache.New(l0CacheSize)
	defer l0Cache.Reset()

	// 6.1. Open a single gRPC connection to Vector Engine
	conn, connErr := grpc.NewClient(
		socketAddress, grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if connErr != nil {
		return connErr
	}

	defer func() {
		if closeErr := conn.Close(); closeErr != nil {
			log.Printf("[strix serve] gRPC connection close error: %v\n", closeErr)
		} else {
			log.Println("[strix serve] gRPC connection closed.")
		}
	}()

	// 6.2. Create only ONE Vector Engine stub.
	clientStub := pb.NewSemanticServiceClient(conn)

	// 7. Perform polling (ping request) on Vector Engine.
	log.Println("[strix serve] Waiting for Vector Engine to become ready...")
	if pollErr := waitForEngine(context.Background(), clientStub); pollErr != nil {
		return pollErr
	}

	pidPath, pidPathErr := PIDFilePath()
	if pidPathErr != nil {
		return pidPathErr
	}

	pidContent := strconv.Itoa(os.Getpid())
	if writeErr := os.WriteFile(pidPath, []byte(pidContent), 0600); writeErr != nil {
		return fmt.Errorf("cannot write PID file %s: %w", pidPath, writeErr)
	}

	defer func() {
		if rmErr := os.Remove(pidPath); rmErr != nil && !errors.Is(rmErr, os.ErrNotExist) {
			log.Printf(
				"[strix serve] WARNING: cannot remove PID file %s: %v\n",
				pidPath, rmErr,
			)
		}
	}()

	log.Printf("[strix serve] PID file written: %s (PID %d)\n", pidPath, os.Getpid())

	// 8. Initialize HTTP Server.
	fatalErrChan := make(chan error, 1)
	pool := proxy.NewWorkerPool(clientStub)
	sv := server.NewServer(clientStub, l0Cache, fatalErrChan, pool)

	// 9. Create OS Signal listener.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	serverErrChan := sv.Start()
	var runErr error

	select {
	case serverErr := <-serverErrChan:
		runErr = serverErr
		log.Printf("[strix serve] Gateway server crashed due to: %v\n", runErr)

	case fatalErr := <-fatalErrChan:
		log.Printf(
			"[strix serve] Fatal error from handler: %v - Initiating shutdown...\n",
			fatalErr,
		)

	case sysSig := <-sigChan:
		log.Printf("[strix serve] Received SIGTERM %v. Initiating shutdown...\n",
			sysSig,
		)
	}

	// 10. HTTP Gateway graceful shutdown.
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer shutdownCancel()

	defer func() {
		if poolErr := pool.Stop(shutdownCtx); poolErr != nil {
			log.Printf("[strix serve] Worker Pool stop error: %v\n", poolErr)
		} else {
			log.Println("[strix server] Worker Pool stopped")
		}
	}()

	log.Println("[strix serve] Stopping HTTP Server...")
	if stopErr := sv.Stop(shutdownCtx); stopErr != nil {
		log.Printf("[strix serve] HTTP Server stop error: %v\n", stopErr)
	}

	return runErr
}

func openMoreFD() error {
	var rLimit syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rLimit); err != nil {
		return fmt.Errorf("cannot read ulimit: %w", err)
	}

	if rLimit.Cur < maxFD {
		rLimit.Cur = maxFD
		rLimit.Max = max(rLimit.Max, maxFD)
		if err := syscall.Setrlimit(syscall.RLIMIT_NOFILE, &rLimit); err != nil {
			log.Printf(
				"[strix serve] OS refused to raise ulimit - may crash under high load: %v\n",
				err,
			)
		} else {
			log.Printf("[strix serve] FD limit raised to %d\n", maxFD)
		}
	}

	return nil
}

func resolvePaths() (projectPaths, error) {
	execDir, dirErr := os.Executable()
	if dirErr != nil {
		return projectPaths{}, fmt.Errorf("cannot resolve executable path: %w", dirErr)
	}

	execDir, symErr := filepath.EvalSymlinks(execDir)
	if symErr != nil {
		return projectPaths{}, fmt.Errorf(
			"cannot evaluate symlink on executable path: %w", symErr,
		)
	}

	// Move one level up from bin/ to strix/
	projectRoot := filepath.Join(filepath.Dir(execDir), "..")

	return projectPaths{
		engineBinary: filepath.Join(projectRoot, "bin", "strix_engine"),
		modelPath: filepath.Join(
			projectRoot, "engine", "model", "strix-minilm-with-tokenizer.onnx",
		),
		ortLibPath: filepath.Join(
			projectRoot, "engine", "model", "libortextensions.so",
		),
	}, nil
}

func buildEngineEnv(paths projectPaths) []string {
	base := os.Environ()
	env := make([]string, 0, len(base)+2)

	for _, kv := range base {
		if strings.HasPrefix(kv, "API_KEY=") {
			continue
		}

		env = append(env, kv)
	}

	env = append(env,
		"INFERENCE_MODEL_PATH="+paths.modelPath,
		"ORT_EXTENSIONS_PATH="+paths.ortLibPath,
	)

	return env
}

func waitForEngine(ctx context.Context, stub pb.SemanticServiceClient) error {
	deadline := time.Now().Add(pollTimeout)

	for attempt := 1; time.Now().Before(deadline); attempt++ {
		pingCtx, pingCancel := context.WithTimeout(ctx, pollInterval)
		_, pingErr := stub.CheckCache(pingCtx, &pb.CheckCacheRequest{Prompt: []byte("health_check")})
		pingCancel()

		if pingErr == nil {
			log.Printf("[strix serve] Vector Engine ready after %d poll(s).\n", attempt)
			return nil
		}

		if status.Code(pingErr) != codes.Unavailable {
			return fmt.Errorf("unexpected error in Vector Engine: %w", pingErr)
		}

		log.Printf(
			"[strix serve] Vector Engine not ready yet (attempt %d). Retrying in %s...\n",
			attempt, pollInterval,
		)
		time.Sleep(pollInterval)
	}

	return fmt.Errorf("FATAL: unready Vector Engine after %s second(s)", pollTimeout)
}
