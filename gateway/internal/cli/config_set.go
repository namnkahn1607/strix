package cli

import (
	"fmt"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

var configSetCmd = &cobra.Command{
	Use:   "set",
	Short: "Set one or more config values in ~/.strix/.env",
	Long: `Writes (or overwrites) environment variable values in ~/.strix/.env.
	The file must exist (run 'strix init' first) and must have permission 0600.
 
	Flags control which fields are updated; at least one flag is required.
  	--endpoint <url>          overwrite GATEWAY_ENDPOINT
  	--apikey                  prompt securely for GATEWAY_APIKEY
  	--endpoint <url> --apikey update both in a single run`,
	RunE: runConfigSet,
}

func runConfigSet(cmd *cobra.Command, _ []string) error {
	// 1. Check for modifying configurations.
	wantAPIKey := cmd.Flags().Changed("apikey")
	wantEndpoint := cmd.Flags().Changed("endpoint")

	if !wantAPIKey && !wantEndpoint {
		return fmt.Errorf("no flags provided - please specify")
	}

	// 2. Check if service is running or not.
	if pid, running, checkErr := IsInstanceRunning(); checkErr != nil {
		return fmt.Errorf("cannot check status: %w", checkErr)
	} else if running {
		return fmt.Errorf(
			"[strix config] ERROR: Cannot mutate config while Strix is "+
				"running (PID: %d). "+
				"Please stop the server first with 'strix stop'", pid,
		)
	}

	// 3. Read env-var map.
	envPath, pathErr := EnvFilePath()
	if pathErr != nil {
		return pathErr
	}

	currEnv, readErr := godotenv.Read(envPath)
	if readErr != nil {
		return fmt.Errorf("cannot parse %s: %w", envPath, readErr)
	}

	// 4. Modify specified configurations.
	if wantAPIKey {
		apiKey, promptErr := promptSecret("API Key")
		if promptErr != nil {
			return promptErr
		}

		if apiKey == "" {
			return fmt.Errorf("API Key must not be empty")
		}

		currEnv["GATEWAY_APIKEY"] = apiKey
	}

	if wantEndpoint {
		if flagSetEndpoint == "" {
			return fmt.Errorf("--endpoint requires a non-empty URL")
		}

		currEnv["GATEWAY_ENDPOINT"] = flagSetEndpoint
	}

	// 5. Write specified configurations to disk.
	if writeErr := godotenv.Write(currEnv, envPath); writeErr != nil {
		return fmt.Errorf("cannot write to %s: %w", envPath, writeErr)
	}

	if chmodErr := os.Chmod(envPath, ownPermission); chmodErr != nil {
		return fmt.Errorf("SECURITY: cannot enforce 0600 after write: %w", chmodErr)
	}

	fmt.Printf("[strix config] Changes saved to %s\n", envPath)
	return nil
}

func promptSecret(fieldName string) (string, error) {
	fmt.Printf("Enter %s:", fieldName)

	rawKey, readErr := term.ReadPassword(int(os.Stdin.Fd()))
	fmt.Println()
	if readErr != nil {
		return "", fmt.Errorf(
			"cannot read %s from terminal: %w", fieldName, readErr,
		)
	}

	return strings.TrimSpace(string(rawKey)), nil
}
