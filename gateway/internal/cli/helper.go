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

var ErrIsServing = errors.New("another strix serve is running")

// assertEnvPermissions returns an error if ~/.strix/.env has permissions
// wider than 0600. Called by 'strix serve' as a boot-time security check.
func assertEnvPermissions() error {
	envPath, dirErr := envFilePath()
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

// acquireLockFile opens ~/.strix/strix.lock and attempts to acquire an
// exclusive non-blocking flock on it. Returns:
// * (file, nil) on success - caller must close the file to release
// the lock (typically via defer).
// * (nil, ErrInstanceRunning) if the lock is already held by another
// 'strix serve' process (EWOULDBLOCK / EAGAIN).
// * (nil, err) for any other I/O failure.
func acquireLockFile() (*os.File, error) {
	lockPath, err := lockFilePath()
	if err != nil {
		return nil, err
	}

	file, err := os.OpenFile(lockPath, os.O_CREATE|syscall.O_CLOEXEC, ownPermission)
	if err != nil {
		return nil, fmt.Errorf("cannot open/create lock file %s: %w", lockPath, err)
	}

	err = syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if err != nil {
		_ = file.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN) {
			return nil, ErrIsServing
		}
	}

	return file, nil
}

// isInstanceRunning reads ~/.strix/strix.pid and checks whether the
// recorded process is still alive via Signal(0).
func isInstanceRunning() (int, bool, error) {
	pidPath, dirErr := pidFilePath()
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

// envFilePath returns the canonical path to ~/.strix/.env.
// Used by other commands to locate the configuration file.
func envFilePath() (string, error) {
	home, dirErr := os.UserHomeDir()
	if dirErr != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", dirErr)
	}

	return filepath.Join(home, strixDir, envFileName), nil
}

// lockFilePath returns the canonical path to ~/.strix/strix.lock.
// Called by 'strix serve' to prevent another (accident) fork procedure.
func lockFilePath() (string, error) {
	home, dirErr := os.UserHomeDir()
	if dirErr != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", dirErr)
	}

	return filepath.Join(home, strixDir, lockFileName), nil
}

// pidFilePath returns the canonical path to ~/.strix/strix.pid.
func pidFilePath() (string, error) {
	home, dirErr := os.UserHomeDir()
	if dirErr != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", dirErr)
	}

	return filepath.Join(home, strixDir, pidFileName), nil
}
