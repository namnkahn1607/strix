package cli

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

const (
	pidFileName  = "strix.pid"
	lockFileName = "strix.lock"
)

// AssertEnvPermissions returns an error if ~/.strix/.env has permissions
// wider than 0600. Called by 'strix serve' as a boot-time security check.
func AssertEnvPermissions() error {
	envPath, dirErr := EnvFilePath()
	if dirErr != nil {
		return dirErr
	}

	info, statErr := os.Stat(envPath)
	if statErr != nil {
		return fmt.Errorf("cannot stat %s (run 'strix init' first): %w", envPath, statErr)
	}

	if perm := info.Mode().Perm(); perm&0077 != 0 {
		return fmt.Errorf(
			"SECURITY: %s has permissions %04o - group/other bits must be zero. Run 'chmod 0600 %s' to fix",
			envPath, perm, envPath,
		)
	}

	return nil
}

// IsInstanceRunning reads ~/.strix/strix.pid and checks whether the
// recorded process is still alive via Signal(0).
func IsInstanceRunning() (int, bool, error) {
	pidPath, dirErr := PIDFilePath()
	if dirErr != nil {
		return 0, false, dirErr
	}

	data, readErr := os.ReadFile(pidPath)

	if errors.Is(readErr, os.ErrNotExist) {
		return 0, false, nil
	}

	if readErr != nil {
		return 0, false, fmt.Errorf("cannot read %s: %w", pidPath, readErr)
	}

	pid, parseErr := strconv.Atoi(strings.TrimSpace(string(data)))
	if parseErr != nil {
		return 0, false, fmt.Errorf("malformed PID file %s: %w", pidPath, parseErr)
	}

	proc, _ := os.FindProcess(pid)
	if sigErr := proc.Signal(syscall.Signal(0)); sigErr != nil {
		return 0, false, nil
	}

	return pid, true, nil
}

// EnvFilePath returns the canonical path to ~/.strix/.env.
// Used by other commands to locate the configuration file.
func EnvFilePath() (string, error) {
	home, dirErr := os.UserHomeDir()
	if dirErr != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", dirErr)
	}

	return filepath.Join(home, strixDir, envFileName), nil
}

// LockFilePath returns the canonical path to ~/.strix/strix.lock.
// Called by 'strix serve' to prevent another (accident) fork procedure.
func LockFilePath() (string, error) {
	home, dirErr := os.UserHomeDir()
	if dirErr != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", dirErr)
	}

	return filepath.Join(home, strixDir, lockFileName), nil
}

// PIDFilePath returns the canonical path to ~/.strix/strix.pid.
func PIDFilePath() (string, error) {
	home, dirErr := os.UserHomeDir()
	if dirErr != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", dirErr)
	}

	return filepath.Join(home, strixDir, pidFileName), nil
}
