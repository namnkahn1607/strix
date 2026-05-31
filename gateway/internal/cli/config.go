package cli

import "github.com/spf13/cobra"

var (
	flagSetAPIKey     bool
	flagSetEndpoint   string
	flagSetPromptPath string

	flagGetAPIKey     bool
	flagGetEndpoint   bool
	flagGetPromptPath bool
)

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Manage Strix configuration",
}

func init() {
	// set flags
	configSetCmd.Flags().BoolVar(&flagSetAPIKey, "apikey", false,
		"prompt securely for upstream API Key",
	)
	configSetCmd.Flags().StringVar(&flagSetEndpoint, "endpoint", "",
		"upstream LLM endpoint URL",
	)
	configSetCmd.Flags().StringVar(&flagSetPromptPath, "prompt-path", "",
		`comma-separated path to prompt field (e.g. "messages,1,content")`,
	)

	// get flags
	configGetCmd.Flags().BoolVar(&flagGetAPIKey, "apikey", false,
		"show redacted API Key of this HTTP Gateway",
	)
	configGetCmd.Flags().BoolVar(&flagGetEndpoint, "endpoint", false,
		"show configured Endpoint of the LLM Provider",
	)
	configGetCmd.Flags().BoolVar(&flagGetPromptPath, "prompt-path", false,
		"show configured path to prompt of user request body",
	)

	configCmd.AddCommand(configSetCmd)
	configCmd.AddCommand(configGetCmd)
}
