package cli

import (
	"fmt"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

const (
	apiKeyVar     = "GATEWAY_UPSTREAM_APIKEY"
	endpointVar   = "GATEWAY_UPSTREAM_ENDPOINT"
	promptPathVar = "GATEWAY_PROMPT_PATH"
)

var configSetCmd = &cobra.Command{
	Use:   "set",
	Short: "Set one or more config values in ~/.strix/.env",
	Long: `Writes (or overwrites) environment variable values in ~/.strix/.env.
	The file must exist (run 'strix init' first) and must have permission 0600.
 
	Flags control which fields are updated; at least one flag is required.
	--apikey                  prompt securely for upstream API Key
  	--endpoint <url>          overwrite upstream endpoint
	--prompt-path <path>      comma-path to prompt field in request body`,
	RunE: runConfigSet,
}

func runConfigSet(cmd *cobra.Command, _ []string) error {
	// 1. Check for modifying configurations.
	wantAPIKey := cmd.Flags().Changed("apikey")
	wantEndpoint := cmd.Flags().Changed("endpoint")
	wantPromptPath := cmd.Flags().Changed("prompt-path")

	if !wantAPIKey && !wantEndpoint && !wantPromptPath {
		return fmt.Errorf("no flags provided - please specify")
	}

	// 2. Check if service is running or not.
	if pid, running, err := IsInstanceRunning(); err != nil {
		return fmt.Errorf("cannot check status: %w", err)
	} else if running {
		return fmt.Errorf(
			"cannot mutate config while Strix is running (PID: %d). "+
				"Please stop the server first with 'strix stop'", pid,
		)
	}

	// 3. Read env-var map.
	envPath, err := EnvFilePath()
	if err != nil {
		return err
	}

	currEnv, err := godotenv.Read(envPath)
	if err != nil {
		return fmt.Errorf("cannot parse %s: %w", envPath, err)
	}

	// 4. Modify specified configurations.
	if wantAPIKey {
		apiKey, promptErr := promptSecret("API Key")
		if promptErr != nil {
			return promptErr
		}

		if apiKey == "" {
			return fmt.Errorf("--apikey requires a non-empty key")
		}

		currEnv[apiKeyVar] = apiKey
	}

	if wantEndpoint {
		if flagSetEndpoint == "" {
			return fmt.Errorf("--endpoint requires a non-empty URL")
		}

		currEnv[endpointVar] = flagSetEndpoint
	}

	if wantPromptPath {
		promptErr := validatePromptPath(flagSetPromptPath)
		if promptErr != nil {
			return fmt.Errorf("invalid Prompt Path: %w", promptErr)
		}

		currEnv[promptPathVar] = flagSetPromptPath
	}

	// 5. Write specified configurations to disk.
	if writeErr := godotenv.Write(currEnv, envPath); writeErr != nil {
		return fmt.Errorf("cannot write to %s: %w", envPath, writeErr)
	}

	if chmodErr := os.Chmod(envPath, ownPermission); chmodErr != nil {
		return fmt.Errorf("cannot enforce 0600 after write: %w", chmodErr)
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

func validatePromptPath(s string) error {
	if s == "" {
		return fmt.Errorf("path must not be empty")
	}

	for _, part := range strings.Split(s, ",") {
		if strings.TrimSpace(part) == "" {
			return fmt.Errorf("empty segment in path %q", s)
		}
	}

	return nil
}
