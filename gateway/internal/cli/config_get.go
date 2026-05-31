package cli

import (
	"fmt"
	"strings"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

type configField struct {
	envKey   string
	label    string
	flagName string
	isSecret bool
}

var registry = []configField{
	{
		envKey:   apiKeyVar,
		label:    "API Key",
		flagName: "apikey",
		isSecret: true,
	},
	{
		envKey:   endpointVar,
		label:    "Endpoint",
		flagName: "endpoint",
		isSecret: false,
	},
	{
		envKey:   promptPathVar,
		label:    "Prompt Path",
		flagName: "prompt-path",
		isSecret: false,
	},
}

var configGetCmd = &cobra.Command{
	Use:   "get",
	Short: "Print current config values from ~/.strix/.env",
	Long: `Reads ~/.strix/.env and prints configuration values.
 
  	(no flags)        print all fields; secrets are redacted
	--apikey          print upstream API Key  (value is always redacted)
  	--endpoint        print upstream endpoint
	--prompt-path     print path to prompt field in request body`,
	RunE: runConfigGet,
}

func runConfigGet(cmd *cobra.Command, _ []string) error {
	envPath, err := EnvFilePath()
	if err != nil {
		return err
	}

	currEnv, err := godotenv.Read(envPath)
	if err != nil {
		return fmt.Errorf("cannot parse %s: %w", envPath, err)
	}

	var selected []configField
	anyFlagPassed := false

	for _, field := range registry {
		if field.flagName != "" && cmd.Flags().Changed(field.flagName) {
			selected = append(selected, field)
			anyFlagPassed = true
		}
	}

	if !anyFlagPassed {
		selected = registry
	}

	for _, field := range selected {
		printField(field, currEnv[field.envKey])
	}

	return nil
}

func printField(f configField, value string) {
	if value == "" {
		fmt.Printf("%-20s (not set)\n", f.label+":")
		return
	}

	if f.isSecret {
		fmt.Printf("%-20s %s\n", f.label+":", redact(value))
		return
	}

	fmt.Printf("%-20s %s\n", f.label+":", value)
}

func redact(s string) string {
	const (
		visibleTail        = 4
		minLenToRevealTail = 8
	)

	if len(s) < minLenToRevealTail {
		return strings.Repeat("*", len(s))
	}

	return strings.Repeat("*", len(s)-visibleTail) + s[len(s)-visibleTail:]
}
