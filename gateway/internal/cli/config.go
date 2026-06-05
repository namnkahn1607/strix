package cli

import "github.com/spf13/cobra"

const (
	apiKeyFlag       = "apikey"
	endpointFlag     = "endpoint"
	promptPathFlag   = "prompt-flag"
	templatePathFlag = "template-flag"
)

var (
	flagSetAPIKey       bool
	flagSetEndpoint     string
	flagSetPromptPath   string
	flagSetTemplatePath string

	flagGetAPIKey       bool
	flagGetEndpoint     bool
	flagGetPromptPath   bool
	flagGetTemplatePath bool
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
	configSetCmd.Flags().StringVar(&flagSetTemplatePath, "template-path", "",
		"path to LLM request template file used by 'strix ask -l'",
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
	configGetCmd.Flags().BoolVar(&flagGetTemplatePath, "template-path", false,
		"show configured path to LLM request template",
	)

	configCmd.AddCommand(configSetCmd)
	configCmd.AddCommand(configGetCmd)
}
