package cli

import (
	"errors"
	"fmt"
	"gateway/internal/sys"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

const (
	supervisorDrainTimeout = 6 * time.Second
	gatewayShutdown        = 3 * time.Second
	engineShutdown         = 5 * time.Second
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
// │   ├── strix          <- os.Executable() — Process A and B share this binary
// │   └── strix_engine   <- Process C
// ├── gateway/
// └── engine/
//     └── model/
//         └── strix-minilm-with-tokenizer.onnx

type projectPaths struct {
	mainBinary   string // strix/bin/strix         - reused to fork Process B
	engineBinary string // strix/bin/strix_engine  - forked as Process C
	modelPath    string // strix/engine/model/strix-minilm-with-tokenizer.onnx
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
	paths, pathErr := resolvePaths()
	if pathErr != nil {
		return pathErr
	}

	log.Printf("[strix serve] Strix Binary: %s\n", paths.mainBinary)
	log.Printf("[strix serve] Strix Engine binary: %s\n", paths.engineBinary)
	log.Printf("[strix serve] Inference model: %s\n", paths.modelPath)

	// 4. Read ~/.strix/.env to get API key and endpoint.
	envPath, envErr := EnvFilePath()
	if envErr != nil {
		return envErr
	}

	configEnv, readErr := godotenv.Read(envPath)
	if readErr != nil {
		return fmt.Errorf("cannot read %s: %w", envPath, readErr)
	}

	// 5. Create 2 Death Pipes for Process B and C.
	readB, writeB, pipeErr := os.Pipe()
	if pipeErr != nil {
		return fmt.Errorf("cannot create Gateway Death Pipe: %w", pipeErr)
	}

	readC, writeC, pipeErr := os.Pipe()
	if pipeErr != nil {
		_ = readB.Close()
		_ = writeB.Close()
		return fmt.Errorf("cannot create Engine Death Pipe: %w", pipeErr)
	}

	defer func() {
		_ = readB.Close()
		_ = writeB.Close()
		_ = readC.Close()
		_ = writeC.Close()
	}()

	// 6. Fork Process C - running Vector Engine.
	engineProc, engineErr := forkEngine(paths, mask.EngineCores, readC, configEnv)
	if engineErr != nil {
		_ = readB.Close()
		_ = writeB.Close()
		_ = readC.Close()
		_ = writeC.Close()
		return engineErr
	}

	log.Printf("[strix serve] Vector Engine started (PID %d)\n", engineProc.Process.Pid)
	_ = readC.Close()

	// 7. Fork Process B - running HTTP Gateway.
	gatewayProc, gatewayErr := forkGateway(paths, mask.GatewayCores, readB, configEnv)
	if gatewayErr != nil {
		_ = readB.Close()
		_ = writeB.Close()
		_ = writeC.Close()
		_ = engineProc.Process.Signal(syscall.SIGTERM)
		return gatewayErr
	}

	log.Printf("[strix serve] HTTP Gateway started (PID %d)\n", gatewayProc.Process.Pid)
	_ = readB.Close()

	// 8. Write Process A's PID to a file.
	if writePIDErr := writePIDFile(); writePIDErr != nil {
		return writePIDErr
	}

	defer removePIDFile()

	// 9. Process A now waits for any shutdown of its children.
	deadChan := make(chan *exec.Cmd, 2)

	go func() {
		waitErr := engineProc.Wait()
		if waitErr != nil {
			log.Printf("[Supervisor] Vector Engine exited: %v\n", waitErr)
		} else {
			log.Println("[Supervisor] Vector Engine exited cleanly")
		}

		deadChan <- engineProc
	}()

	go func() {
		waitErr := gatewayProc.Wait()
		if waitErr != nil {
			log.Printf("[Supervisor] HTTP Gateway exited: %v\n", waitErr)
		} else {
			log.Println("[Supervisor] HTTP Gateway exited cleanly")
		}

		deadChan <- gatewayProc
	}()

	// 10. Trap SIGTERM / SIGKILL for parent Process A.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigChan)

	// 11. React to whatever comes first: OS signal or child death.
	select {
	case sig := <-sigChan:
		// Process A received SIGTERM/SIGINT/SIGKILL from OS or 'strix stop'
		log.Printf(
			"[Supervisor] Received signal %v. Closing both child processes...\n",
			sig,
		)

		_ = writeB.Close()
		_ = writeC.Close()

		drainTimeout := time.After(supervisorDrainTimeout)
		for i := range 2 {
			select {
			case <-deadChan:
				// One child confirmed dead, continue to next

			case <-drainTimeout:
				log.Println(
					"[Supervisor] Drain timeout exceeded. Force-killing remaining children...",
				)
				_ = engineProc.Process.Signal(syscall.SIGKILL)
				_ = gatewayProc.Process.Signal(syscall.SIGKILL)

				for ; i < 2; i++ {
					<-deadChan
				}

				return nil
			}
		}

	case first := <-deadChan:
		switch first {
		case engineProc:
			// Process C died → close Gateway Death Pipe → Process B shuts down.
			log.Printf(
				"[Supervisor] Vector Engine died. Closing HTTP Gateway in %d secs...\n",
				gatewayShutdown,
			)

			_ = writeB.Close()

			select {
			case <-deadChan:
				log.Println("[Supervisor] HTTP Gateway finally exited.")
			case <-time.After(gatewayShutdown):
				log.Println(
					"[Supervisor] HTTP Gateway is taking longer to exit - forcing shutdown...",
				)

				_ = gatewayProc.Process.Signal(syscall.SIGKILL)
			}

		case gatewayProc:
			// Process B died → close Engine Death Pipe → Process C shuts down.
			log.Printf(
				"[Supervisor] HTTP Gateway died. Closing Vector Engine in %d secs...\n",
				engineShutdown,
			)

			_ = writeC.Close()

			select {
			case <-deadChan:
				log.Printf("[Supervisor] Vector Engine finally exited")
			case <-time.After(engineShutdown):
				log.Println(
					"[Supervisor] Vector Engine is taking longer to exit - forcing shutdown...",
				)

				_ = engineProc.Process.Signal(syscall.SIGKILL)
			}
		}
	}

	log.Println("All processes exited. Supervisor shutting down...")
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
		mainBinary:   execDir,
		engineBinary: filepath.Join(projectRoot, "bin", "strix_engine"),
		modelPath: filepath.Join(
			projectRoot, "engine", "model", "strix-minilm-with-tokenizer.onnx",
		),
	}, nil
}

func forkGateway(
	paths projectPaths, coreMask string, readerB *os.File, configEnv map[string]string,
) (*exec.Cmd, error) {
	cmd := exec.Command("taskset", "-c", coreMask, paths.mainBinary)
	cmd.ExtraFiles = []*os.File{readerB}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = buildGatewayEnv(coreMask, configEnv)

	if startErr := cmd.Start(); startErr != nil {
		return nil, fmt.Errorf(
			"cannot start HTTP Gateway at %q: %w", paths.mainBinary, startErr,
		)
	}

	return cmd, nil
}

func forkEngine(
	paths projectPaths, coreMask string, readerC *os.File, configEnv map[string]string,
) (*exec.Cmd, error) {
	cmd := exec.Command("taskset", "-c", coreMask, paths.engineBinary)
	cmd.ExtraFiles = []*os.File{readerC}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = buildEngineEnv(paths, coreMask, configEnv)

	if startErr := cmd.Start(); startErr != nil {
		return nil, fmt.Errorf(
			"cannot start Vector Engine at %q: %w", paths.engineBinary, startErr,
		)
	}

	return cmd, nil
}

func buildGatewayEnv(coreMask string, configEnv map[string]string) []string {
	env := []string{
		"STRIX_WORKER=1",
		"GATEWAY_CORES=" + coreMask,
	}

	for key, value := range configEnv {
		if strings.HasPrefix(key, "GATEWAY_") {
			env = append(env, key+"="+value)
		}
	}

	return env
}

func buildEngineEnv(paths projectPaths, coreMask string, configEnv map[string]string) []string {
	env := []string{
		"INFERENCE_MODEL_PATH=" + paths.modelPath,
		"ENGINE_CORES=" + coreMask,
	}

	for key, value := range configEnv {
		if strings.HasPrefix(key, "ENGINE_") {
			env = append(env, key+"="+value)
		}
	}

	return env
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
