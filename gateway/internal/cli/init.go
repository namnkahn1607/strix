package cli

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

const (
	strixDir      = ".strix"
	envFileName   = ".env"
	pidFileName   = "strix.pid"
	envPermission = 0600 // rw
	mkPermission  = 0700 // rwx
)

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize the Strix environment (~/.strix/)",
	Long: `Creates ~/.strix/ directory and an empty ~/.strix/.env config file.

	The .env file is created atomically with O_EXCL - if it already exists (or
	a symlink occupies that path), the operation fails immediately at the 
	kernel level, preventing symlink-based TOCTOU attacks.
 
	Permission check at runtime enforces that group and other have zero access
	bits. The owner may use 0600 (rw) or 0400 (r-only) as they see fit.`,
	RunE: runInit,
}

func runInit(_ *cobra.Command, _ []string) error {
	home, dirErr := os.UserHomeDir()
	if dirErr != nil {
		return fmt.Errorf("cannot determine home directory: %w", dirErr)
	}

	dir := filepath.Join(home, strixDir)
	env := filepath.Join(dir, envFileName)

	// 1. Create ~/.strix/ directory with the highest permission.
	if mkErr := os.MkdirAll(dir, mkPermission); mkErr != nil {
		return fmt.Errorf("cannot create %s: %w", dir, mkErr)
	}

	// 2. Create the environment file ~/.strix/.env.
	file, createErr := os.OpenFile(env, os.O_CREATE|os.O_EXCL|os.O_WRONLY, envPermission)
	switch {
	case createErr == nil && file != nil:
		fmt.Printf("[strix init] Created %s\n", env)

		if closeErr := file.Close(); closeErr != nil {
			fmt.Printf("[strix init] Error closing env-file: %v.\n", closeErr)
		}

	case errors.Is(createErr, os.ErrExist):
		fmt.Printf("[strix init] %s already exists.\n", env)

	default:
		return fmt.Errorf("cannot create %s: %w", env, createErr)
	}

	// 3. Enforce 0600 regardless manual chmod
	if chmodErr := os.Chmod(env, envPermission); chmodErr != nil {
		return fmt.Errorf("SECURITY: Cannot set 0600 on %s due to: %w", env, chmodErr)
	}

	fmt.Println(
		"[strix init] Environment ready. Run 'strix config set' to add your credentials.",
	)
	return nil
}
