package system

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// CheckRAM verifies if the system meets the sufficient amount of RAM (>= 8GB)
// required for Strix to operate. If not an error is returned.
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
