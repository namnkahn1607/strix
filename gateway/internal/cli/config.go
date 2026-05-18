package cli

import (
	"fmt"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

var flagEndpoint string

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Manage Strix configuration",
}

var configSetCmd = &cobra.Command{
	Use:   "set",
	Short: "Set credentials in ~/.strix/.env",
	Long: `Writes (or overwrites) the environment variables values into
	~/.strix/.env. The file must exist (run 'strix init' first) and
	must have permission 0600.`,
	RunE: runConfigSet,
}

func init() {
	configSetCmd.Flags().StringVar(
		&flagEndpoint, "endpoint", "", "base URL (required)",
	)
	_ = configSetCmd.MarkFlagRequired("endpoint")

	configCmd.AddCommand(configSetCmd)
}

func runConfigSet(_ *cobra.Command, _ []string) error {
	if pid, running, checkErr := IsInstanceRunning(); checkErr != nil {
		return fmt.Errorf("cannot check server status: %w", checkErr)
	} else if running {
		return fmt.Errorf(
			"[strix config] ERROR: Cannot mutate config while Strix is "+
				"running (PID: %d). "+
				"Please stop the server first with 'strix stop'", pid,
		)
	}

	if permErr := AssertEnvPermissions(); permErr != nil {
		return permErr
	}

	fmt.Print("[strix config] Enter API Key: ")
	rawKey, readErr := term.ReadPassword(int(os.Stdin.Fd()))
	fmt.Println()
	if readErr != nil {
		return fmt.Errorf("cannot read API key from terminal: %w", readErr)
	}

	apiKey := strings.TrimSpace(string(rawKey))
	emptyAPIKey := len(apiKey) == 0
	emptyEndpoint := len(flagEndpoint) == 0

	if emptyAPIKey && emptyEndpoint {
		fmt.Println("[strix config] No credentials are updated.")
		return nil
	}

	if emptyAPIKey || emptyEndpoint {
		return fmt.Errorf("one of the credentials is empty")
	}

	envPath, pathErr := EnvFilePath()
	if pathErr != nil {
		return pathErr
	}

	currEnv, readErr := godotenv.Read(envPath)
	if readErr != nil {
		return fmt.Errorf("cannot parse %s: %w", envPath, readErr)
	}

	currEnv["GATEWAY_APIKEY"] = apiKey
	currEnv["GATEWAY_ENDPOINT"] = flagEndpoint

	if writeErr := godotenv.Write(currEnv, envPath); writeErr != nil {
		return fmt.Errorf("cannot write to %s: %w", envPath, writeErr)
	}

	if chmodErr := os.Chmod(envPath, ownPermission); chmodErr != nil {
		return fmt.Errorf("SECURITY: Cannot enforce 0600 after write: %w", chmodErr)
	}

	fmt.Printf("[strix config] Credentials saved to %s\n", envPath)
	return nil
}
