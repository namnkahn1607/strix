package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "strix",
	Short: "Strix - Semantic Cache Proxy CLI",
	Long: `Strix is a high-performance semantic cache gateway.

	Use 'strix init' to set up your environment, then 'strix serve' to start.`,
}

// Execute is the single entry point of Supervisor.
func Execute() {
	if execErr := rootCmd.Execute(); execErr != nil {
		fmt.Printf("[strix] Entry point error: %v\n", execErr)
		os.Exit(1)
	}
}

func init() {
	rootCmd.AddCommand(askCmd)
	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(configCmd)
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(stopCmd)
}
