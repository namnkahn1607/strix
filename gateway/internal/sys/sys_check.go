package system

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

const (
	minRAMBytes    = 8 * 1024 * 1024 * 1024
	maxGatewayCPUs = 4
)

type MaskCPU struct {
	GatewayCores string
	EngineCores  string
}

func GetAffineCPUs() (MaskCPU, error) {
	var cpuSet unix.CPUSet
	if getErr := unix.SchedGetaffinity(os.Getpid(), &cpuSet); getErr != nil {
		return MaskCPU{}, fmt.Errorf("sched_setaffinity failed: %w", getErr)
	}

	cpuCount := cpuSet.Count()
	if cpuCount <= 0 {
		return MaskCPU{}, fmt.Errorf("FATAL: no CPU allocated for process")
	}

	available := make([]int, 0, cpuCount)
	found := 0
	for i := range 1024 {
		if cpuSet.IsSet(i) {
			available = append(available, i)
			found++

			if found == cpuCount {
				break
			}
		}
	}

	switch {
	case cpuCount == 1:
		_, _ = fmt.Fprintln(os.Stderr,
			"[strix] WARNING: only 1 CPU available. "+
				"HTTP Gateway and Vector Engine will share it. "+
				"Expect degraded performance.",
		)

		shared := joinCPUs(available)
		return MaskCPU{
			GatewayCores: shared,
			EngineCores:  shared,
		}, nil

	case cpuCount <= 8:
		half := cpuCount / 2
		return MaskCPU{
			GatewayCores: joinCPUs(available[:half]),
			EngineCores:  joinCPUs(available[half:]),
		}, nil

	default:
		return MaskCPU{
			GatewayCores: joinCPUs(available[:maxGatewayCPUs]),
			EngineCores:  joinCPUs(available[maxGatewayCPUs:]),
		}, nil
	}
}

// ApplyGoMaxProcs sets GOMAXPROCS to the number of CPUs in the provided
// taskset -c string, capped at 4.
// Call this from Process B after it has identified itself via STRIX_WORKER=1.
func ApplyGoMaxProcs(gatewayCores []int) {
	runtime.GOMAXPROCS(min(maxGatewayCPUs, len(gatewayCores)))
}

// CheckRAM checks if the system has sufficient amount of RAM (>= 8GB).
func CheckRAM() error {
	memData, readErr := os.ReadFile("/proc/meminfo")
	if readErr != nil {
		return fmt.Errorf("cannot read /proc/meminfo: %w", readErr)
	}

	for _, line := range strings.Split(string(memData), "\n") {
		if !strings.HasPrefix(line, "MemTotal:") {
			continue
		}

		fields := strings.Fields(line)
		if len(fields) < 2 {
			break
		}

		kb, parseErr := strconv.ParseInt(fields[1], 10, 64)
		if parseErr != nil {
			return fmt.Errorf("cannot parse RAM size: %w", parseErr)
		}

		if totalBytes := kb * 1024; totalBytes < minRAMBytes {
			return fmt.Errorf(
				"insufficient RAM. Require at least %dMB", minRAMBytes/(1024*1024),
			)
		}

		return nil
	}

	return fmt.Errorf("MemTotal not found in /proc/meminfo")
}

func joinCPUs(cpus []int) string {
	parts := make([]string, len(cpus))

	for i, c := range cpus {
		parts[i] = strconv.Itoa(c)
	}

	return strings.Join(parts, ",")
}
