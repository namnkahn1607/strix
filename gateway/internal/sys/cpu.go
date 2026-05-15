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

// ApplyGoMaxProcs sets GOMAXPROCS to the number of CPUs, capped at 4.
// Call this from Process B after it has identified itself via STRIX_WORKER=1.
func ApplyGoMaxProcs(gatewayCores []int) {
	runtime.GOMAXPROCS(min(maxGatewayCPUs, len(gatewayCores)))
}

type MaskCPU struct {
	GatewayCores string
	EngineCores  string
}

// GetAffineCPUs reads the CPU set visible to this process via
// unix.SchedGetAffinity (respects container cpuset), then
// partitions them according to the following policy:
//   - 1 CPU            : a performance warning is printed as both processes share the same CPU.
//   - From 2 to 8 CPUs : 50/50 split; the odd CPU (if any) goes to Engine.
//   - More than 8 CPUs : Gateway gets the first 4 CPUs; Engine gets the rest.
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

func joinCPUs(cpus []int) string {
	parts := make([]string, len(cpus))

	for i, c := range cpus {
		parts[i] = strconv.Itoa(c)
	}

	return strings.Join(parts, ",")
}
