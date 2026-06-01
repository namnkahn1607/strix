package cli

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

const (
	apiKeyVar       = "GATEWAY_UPSTREAM_APIKEY"
	endpointVar     = "GATEWAY_UPSTREAM_ENDPOINT"
	promptPathVar   = "GATEWAY_PROMPT_PATH"
	templatePathVar = "ASK_TEMPLATE_PATH"
)

var configSetCmd = &cobra.Command{
	Use:   "set",
	Short: "Set one or more config values in ~/.strix/.env",
	Long: `Writes (or overwrites) environment variable values in ~/.strix/.env.
	The file must exist (run 'strix init' first) and must have permission 0600.
 
	Flags control which fields are updated; at least one flag is required.
	--apikey                  prompt securely for upstream API Key
  	--endpoint <url>          overwrite upstream endpoint
	--prompt-path <path>      comma-path to prompt field in request body
	--template-path <path>    path to LLM request template file for 'strix ask -l'
							  template must contain {{STRIX_PROMPT}} placeholder`,
	RunE: runConfigSet,
}

func runConfigSet(cmd *cobra.Command, _ []string) error {
	// 1. Check for modifying configurations.
	wantAPIKey := cmd.Flags().Changed(apiKeyFlag)
	wantEndpoint := cmd.Flags().Changed(endpointFlag)
	wantPromptPath := cmd.Flags().Changed("prompt-path")
	wantTemplatePath := cmd.Flags().Changed("template-path")

	if !wantAPIKey && !wantEndpoint && !wantPromptPath && !wantTemplatePath {
		return fmt.Errorf("no flags provided - please specify")
	}

	// 2. Check if Strix is serving or not.
	lockFile, err := acquireLockFile()
	if err != nil {
		if errors.Is(err, ErrIsServing) {
			return fmt.Errorf(
				"cannot mutate config while Strix is running - " +
					"stop it first with 'strix stop'",
			)
		}

		return fmt.Errorf("cannot check server status: %w", err)
	}

	_ = lockFile.Close()

	// 3. Read env-var map.
	envPath, err := envFilePath()
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

	if wantTemplatePath {
		templateErr := validateTemplatePath(flagSetTemplatePath)
		if templateErr != nil {
			return fmt.Errorf("invalid Template Path: %w", templateErr)
		}

		currEnv[templatePathVar] = flagSetTemplatePath
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

func validatePromptPath(path string) error {
	if path == "" {
		return fmt.Errorf("--prompt-path requires a non-empty path")
	}

	for _, part := range strings.Split(path, ",") {
		if strings.TrimSpace(part) == "" {
			return fmt.Errorf("empty segment in path %q", path)
		}
	}

	return nil
}

func validateTemplatePath(path string) error {
	if path == "" {
		return fmt.Errorf("--template-path requires a non-empty path")
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("cannot read %q: %w", path, err)
	}

	if !bytes.Contains(content, []byte(promptPlaceholder)) {
		return fmt.Errorf("%q does not contain placeholder %s", path, promptPlaceholder)
	}

	return nil
}
