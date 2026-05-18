package supervisor

import (
	"fmt"
	system "gateway/internal/sys"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"strings"
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
}

func NewController(
	mask system.MaskCPU, paths ProjectPath, env map[string]string,
) *Controller {
	return &Controller{
		mask:      mask,
		paths:     paths,
		configEnv: env,
	}
}

func (c *Controller) Run() error {
	// 1. Create 2 Death Pipes for HTTP Gateway and Vector Engine.
	readB, writeB, pipeErr := os.Pipe()
	if pipeErr != nil {
		return fmt.Errorf("cannot create Gateway Death Pipe: %w", pipeErr)
	}

	readC, writeC, pipeErr := os.Pipe()
	if pipeErr != nil {
		return fmt.Errorf("cannot create Engine Death Pipe: %w", pipeErr)
	}

	// 2. Fork process C - running Vector Engine.
	engineProc, engineErr := forkEngine(
		c.paths.EngineBin, c.paths.ModelPath, c.mask.EngineCores,
		readC, c.configEnv,
	)
	if engineErr != nil {
		closePipeEnd(readB, writeB, readC, writeC)
		return engineErr
	}

	closePipeEnd(readC)
	log.Printf("[Supervisor] Vector Engine started (PID %d)\n",
		engineProc.Process.Pid,
	)

	// 3. Fork process B - running HTTP Gateway.
	gatewayProc, gatewayErr := forkGateway(
		c.paths.MainBin, c.mask.GatewayCores,
		readB, c.configEnv,
	)
	if gatewayErr != nil {
		closePipeEnd(readB, writeB, readC, writeC)
		_ = engineProc.Wait()
		return gatewayErr
	}

	closePipeEnd(readB)
	log.Printf("[Supervisor] HTTP Gateway started (PID %d)\n",
		gatewayProc.Process.Pid,
	)

	// 4. Wait for any shutdown of child processes.
	deadChan := make(chan *exec.Cmd, 2)
	defer func() {
		for len(deadChan) > 0 {
			<-deadChan
		}
	}()

	go func() {
		waitErr := engineProc.Wait()
		if waitErr != nil {
			log.Printf("[Supervisor] Vector Engine exited: %v\n", waitErr)
		} else {
			log.Println("[Supervisor] Vector Engine shutdown gracefully")
		}

		deadChan <- engineProc
	}()

	go func() {
		waitErr := gatewayProc.Wait()
		if waitErr != nil {
			log.Printf("[Supervisor] HTTP Gateway exited: %v\n", waitErr)
		} else {
			log.Println("[Supervisor] HTTP Gateway shutdown gracefully")
		}

		deadChan <- gatewayProc
	}()

	// 5. Trap SIGINT / SIGTERM from OS.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigChan)

	// 6. React to first event: OS signal or child death.
	select {
	case sig := <-sigChan:
		// Process A received SIGTERM / SIGINT from OS or 'strix stop'
		log.Printf(
			"[Supervisor] Received %v. Closing both child processes..\n", sig,
		)

		closePipeEnd(writeB, writeC)

		deathConfirmed := 0
		for range 2 {
			select {
			case <-deadChan:
				deathConfirmed++

			case <-time.After(drainTimeout):
				log.Println("[Supervisor] Drain timeout. Force killing...")
				_ = engineProc.Process.Signal(syscall.SIGKILL)
				_ = gatewayProc.Process.Signal(syscall.SIGKILL)

				remaining := 2 - deathConfirmed
				for range remaining {
					<-deadChan
				}

				return nil
			}
		}

	case first := <-deadChan:
		switch first {
		case engineProc:
			// Vector Engine died -> Init HTTP Gateway shutdown
			log.Printf(
				"[Supervisor] Vector Engine shut down. "+
					"Closing HTTP Gateway in %d...\n", gatewayTimeout,
			)

			closePipeEnd(writeB)

			select {
			case <-deadChan:
				log.Println("[Supervisor] HTTP Gateway finally exited")

			case <-time.After(gatewayTimeout):
				log.Println(
					"[Supervisor] HTTP Gateway is taking longer to exit. " +
						"Forcing shutdown...",
				)

				_ = gatewayProc.Process.Signal(syscall.SIGKILL)
			}

		case gatewayProc:
			// HTTP Gateway died -> Init Vector Engine shutdown
			log.Printf(
				"[Supervisor] HTTP Gateway shut down. "+
					"Closing Vector Engine in %d...\n", engineTimeout,
			)

			closePipeEnd(writeC)

			select {
			case <-deadChan:
				log.Println("[Supervisor] Vector Engine finally exited")

			case <-time.After(engineTimeout):
				log.Println(
					"[Supervisor] Vector Engine is taking longer to exit. " +
						"Forcing shutdown...",
				)

				_ = engineProc.Process.Signal(syscall.SIGKILL)
			}
		}
	}

	log.Println("All processes exited. Supervisor shutting down...")
	return nil
}

func forkEngine(
	binPath, modelPath, coreMask string,
	reader *os.File, configEnv map[string]string,
) (*exec.Cmd, error) {
	cmd := exec.Command("taskset", "-c", coreMask, binPath)
	cmd.ExtraFiles = []*os.File{reader}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = buildEngineEnv(modelPath, coreMask, configEnv)

	startErr := cmd.Start()
	if startErr != nil {
		return nil, fmt.Errorf(
			"cannot start Vector Engine at %q: %w", binPath, startErr,
		)
	}

	return cmd, nil
}

func closePipeEnd(pipeEnds ...*os.File) {
	for _, end := range pipeEnds {
		_ = end.Close()
	}
}

func forkGateway(
	binPath string, coreMask string,
	reader *os.File, configEnv map[string]string,
) (*exec.Cmd, error) {
	cmd := exec.Command("taskset", "-c", coreMask, binPath)
	cmd.ExtraFiles = []*os.File{reader}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = buildGatewayEnv(coreMask, configEnv)

	startErr := cmd.Start()
	if startErr != nil {
		return nil, fmt.Errorf(
			"cannot start HTTP Gateway at %q: %w", binPath, startErr,
		)
	}

	return cmd, nil
}

func buildEngineEnv(
	modelPath, coreMask string, configEnv map[string]string,
) []string {
	env := []string{
		"INFERENCE_MODEL_PATH=" + modelPath,
		"ENGINE_CORES=" + coreMask,
	}

	for key, value := range configEnv {
		if strings.HasPrefix(key, "ENGINE_") {
			env = append(env, key+"="+value)
		}
	}

	return env
}

func buildGatewayEnv(coreMask string, configEnv map[string]string) []string {
	env := []string{
		"STRIX_GATEWAY=1",
		"GATEWAY_CORES=" + coreMask,
	}

	for key, value := range configEnv {
		if strings.HasPrefix(key, "GATEWAY_") {
			env = append(env, key+"="+value)
		}
	}

	return env
}
