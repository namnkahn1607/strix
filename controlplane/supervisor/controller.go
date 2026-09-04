package supervisor

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	system "strix/sys"
	"syscall"
	"time"
)

const (
	drainTimeout   = 6 * time.Second
	gatewayTimeout = 3 * time.Second
	engineTimeout  = 5 * time.Second
)

type Controller struct {
	mask      system.MaskCPU
	paths     ProjectPath
	configEnv map[string]string
	opts      ControllerOptions
}

func NewController(
	mask system.MaskCPU,
	paths ProjectPath,
	configEnv map[string]string,
	opts ControllerOptions,
) *Controller {
	return &Controller{
		mask:      mask,
		paths:     paths,
		configEnv: configEnv,
		opts:      opts,
	}
}

func (c *Controller) Run() error {
	// 1. Trap SIGINT / SIGTERM from OS
	ctx, cancelCtx := context.WithCancel(context.Background())
	defer cancelCtx()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigChan)

	go func() {
		select {
		case <-sigChan:
			cancelCtx()

		case <-ctx.Done():
			// Run() is returning normally - exit cleanly
		}
	}()

	deadChan := make(chan *exec.Cmd, 2)

	// 2. Create 2 Death Pipes for HTTP Gateway and Vector Engine.
	readB, writeB, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("cannot create Gateway Death Pipe: %w", err)
	}

	readC, writeC, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("cannot create Engine Death Pipe: %w", err)
	}

	// 3. Fork process C - running Vector Engine.
	engineProc, err := forkEngine(c.paths, c.mask.EngineCores, readC, c.configEnv)
	if err != nil {
		closePipeEnd(readB, writeB, readC, writeC)
		return err
	}

	// Spawn death-waiting goroutine on successful fork
	go func() {
		waitErr := engineProc.Wait()
		if waitErr != nil {
			slog.Error("Vector Engine exited with error.", slog.Any("error", waitErr))
		} else {
			slog.Info("Vector Engine shutdown gracefully.")
		}

		deadChan <- engineProc
	}()

	// readC now belongs to the child process, the parent one no longer need it
	closePipeEnd(readC)
	slog.Info("Vector Engine has started", slog.Int("pid", engineProc.Process.Pid))

	// 4. Optional precaching procedure
	if len(c.opts.PrecacheFiles) > 0 {
		err = c.runPrecache(ctx)
		if err != nil {
			closePipeEnd(readB, writeB, writeC)
			waitThenKill(engineProc, deadChan, engineTimeout)

			return fmt.Errorf("precaching failed: %w", err)
		}
	}

	// 5. Fork process B - running HTTP Gateway.
	gatewayProc, err := forkGateway(c.paths, c.mask.GatewayCores, readB, c.configEnv)
	if err != nil {
		closePipeEnd(readB, writeB, writeC)
		waitThenKill(engineProc, deadChan, engineTimeout)

		return err
	}

	// Spawn death-waiting goroutine on successful fork
	go func() {
		waitErr := gatewayProc.Wait()
		if waitErr != nil {
			slog.Error("HTTP Gateway exited with error.", slog.Any("error", waitErr))
		} else {
			slog.Info("HTTP Gateway shutdown gracefully.")
		}

		deadChan <- gatewayProc
	}()

	// readB now belongs to the child process, the parent one no longer need it
	closePipeEnd(readB)
	slog.Info("HTTP Gateway has started", slog.Int("pid", gatewayProc.Process.Pid))

	// 6. React to first event: OS signal or child death.
	select {
	case <-ctx.Done():
		// Process A received SIGTERM / SIGINT from OS or 'strix stop'
		slog.Info("Received OS signal. Initiating graceful shutdown...")

		closePipeEnd(writeB, writeC)
		drainChildProcs(deadChan, engineProc, gatewayProc)

	case first := <-deadChan:
		switch first {
		case engineProc:
			// Vector Engine died -> Init HTTP Gateway shutdown
			slog.Warn("Vector Engine exited unexpectedly. Shutting down HTTP Gateway...",
				slog.Duration("timeout", gatewayTimeout),
			)

			closePipeEnd(writeB)
			waitThenKill(gatewayProc, deadChan, gatewayTimeout)

		case gatewayProc:
			// HTTP Gateway died -> Init Vector Engine shutdown
			slog.Warn("HTTP Gateway exited unexpectedly. Shutting down Vector Engine...",
				slog.Duration("timeout", engineTimeout),
			)

			closePipeEnd(writeC)
			waitThenKill(engineProc, deadChan, engineTimeout)
		}
	}

	slog.Info("All processes exited. Supervisor shutting down...")
	return nil
}

func closePipeEnd(pipeEnds ...*os.File) {
	for _, end := range pipeEnds {
		_ = end.Close()
	}
}

func forkEngine(
	paths ProjectPath,
	coreMask string,
	reader *os.File,
	configEnv map[string]string,
) (*exec.Cmd, error) {
	cmd := exec.Command("taskset", "-c", coreMask, paths.EngineBin)
	cmd.ExtraFiles = []*os.File{reader}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = buildEngineEnv(paths, coreMask, configEnv)

	startErr := cmd.Start()
	if startErr != nil {
		return nil, fmt.Errorf(
			"cannot start Vector Engine at %q: %w", paths.EngineBin, startErr,
		)
	}

	return cmd, nil
}

func forkGateway(
	paths ProjectPath,
	coreMask string,
	reader *os.File,
	configEnv map[string]string,
) (*exec.Cmd, error) {
	cmd := exec.Command("taskset", "-c", coreMask, paths.MainBin)
	cmd.ExtraFiles = []*os.File{reader}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = buildGatewayEnv(coreMask, configEnv)

	startErr := cmd.Start()
	if startErr != nil {
		return nil, fmt.Errorf(
			"cannot start HTTP Gateway at %q: %w", paths.MainBin, startErr,
		)
	}

	return cmd, nil
}

func buildEngineEnv(
	paths ProjectPath,
	coreMask string,
	configEnv map[string]string,
) []string {
	env := os.Environ()
	env = append(env,
		"TOKENIZER_PATH="+paths.TokPath,
		"TRANSFORMER_PATH="+paths.BertPath,
		"ENGINE_CORES="+coreMask,
	)

	for key, value := range configEnv {
		if strings.HasPrefix(key, "ENGINE_") {
			env = append(env, key+"="+value)
		}
	}

	return env
}

func buildGatewayEnv(
	coreMask string,
	configEnv map[string]string,
) []string {
	env := os.Environ()
	env = append(env,
		"STRIX_GATEWAY=1",
		"GATEWAY_CORES="+coreMask,
	)

	for key, value := range configEnv {
		if strings.HasPrefix(key, "GATEWAY_") {
			env = append(env, key+"="+value)
		}
	}

	return env
}

// drainChildren waits for both children to exit after a graceful shutdown
// signal, force-killing if drainTimeout is exceeded.
func drainChildProcs(deadChan <-chan *exec.Cmd, engineProc, gatewayProc *exec.Cmd) {
	confirmed := 0

	for confirmed < 2 {
		select {
		case <-deadChan:
			confirmed++

		case <-time.After(drainTimeout):
			slog.Warn("Drain timeout exceeded. Force killing remaining processes...",
				slog.Duration("timeout", drainTimeout),
			)

			_ = engineProc.Process.Signal(syscall.SIGKILL)
			_ = gatewayProc.Process.Signal(syscall.SIGKILL)

			for ; confirmed < 2; confirmed++ {
				<-deadChan
			}

			return
		}
	}
}

func waitThenKill(proc *exec.Cmd, deadChan <-chan *exec.Cmd, timeout time.Duration) {
	select {
	case <-deadChan:

	case <-time.After(timeout):
		slog.Warn("Graceful shutdown timeout exceeded. Force killing...",
			slog.Int("pid", proc.Process.Pid),
			slog.Duration("timeout", timeout),
		)

		_ = proc.Process.Signal(syscall.SIGKILL)
		<-deadChan
	}
}
