package cli

import "github.com/spf13/cobra"

var (
	flagSetAPIKey   bool
	flagSetEndpoint string

	flagGetAPIKey   bool
	flagGetEndpoint bool
)

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Manage Strix configuration",
}

func init() {
	// set flags
	configSetCmd.Flags().BoolVar(
		&flagSetAPIKey, "apikey", false,
		"prompt for the LLM Provider's API Key",
	)
	configSetCmd.Flags().StringVar(
		&flagSetEndpoint, "endpoint", "",
		"base LLM Provider's forwarding URL",
	)

	// get flags
	configGetCmd.Flags().BoolVar(
		&flagGetAPIKey, "apikey", false,
		"show redacted API Key of this HTTP Gateway",
	)
	configGetCmd.Flags().BoolVar(
		&flagGetEndpoint, "endpoint", false,
		"show configured Endpoint of the LLM Provider",
	)

	configCmd.AddCommand(configSetCmd)
	configCmd.AddCommand(configGetCmd)
}
