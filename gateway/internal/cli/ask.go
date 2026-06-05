package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

const (
	gatewayEndpoint   = "http://localhost:8080/v1/cache/strix"
	cacheOnlyHeader   = "X-Strix-Cache-Only"
	promptPlaceholder = "{{STRIX_PROMPT}}"
)

var (
	flagLLMAns bool
	flagRTT    bool
)

var askCmd = &cobra.Command{
	Use:   "ask [flags] \"<prompt>\"",
	Short: "Query the Strix cache and optionally forward to LLM",
	Long: `Sends a prompt to the running HTTP Gateway and prints the cached payload.
 
	Default mode: returns cached payload on HIT, or an error message on MISS.
	--llmans (-l)      forward to LLM Provider on MISS and cache the response
	--rtt (-r)         display the total round-trip time`,
	Args: cobra.ExactArgs(1),
	RunE: runAsk,
}

func init() {
	askCmd.Flags().BoolVarP(&flagLLMAns, "llmans", "l", false,
		"forward to LLM Provider on MISS and cache the response",
	)
	askCmd.Flags().BoolVarP(&flagRTT, "rtt", "r", false,
		"display the total round-trip time",
	)
}

func runAsk(_ *cobra.Command, args []string) error {
	lockFile, err := acquireLockFile()
	if err == nil {
		_ = lockFile.Close()
		return fmt.Errorf(
			"daemon process is not running - " +
				"please run it using strix serve",
		)
	} else if !errors.Is(err, ErrIsServing) {
		return fmt.Errorf("cannot check daemon status: %w", err)
	}

	prompt := args[0]

	reqBody, err := buildRequestBody(prompt)
	if err != nil {
		return err
	}

	req, err := http.NewRequest(http.MethodPost, gatewayEndpoint, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("cannot build HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	if !flagLLMAns {
		req.Header.Set(cacheOnlyHeader, "true")
	}

	askClient := &http.Client{
		// Must be longer than Gateway to LLM timeout (currently 30 secs)
		Timeout: 40 * time.Second,
		Transport: &http.Transport{
			// No need to keep TCP connection alive
			DisableKeepAlives: true,

			// Dialing localhost, TCP handshake should take less than 1 sec
			DialContext: (&net.Dialer{
				Timeout: 2 * time.Second,
			}).DialContext,

			// Avoid I/O hanging from HTTP Gateway
			ResponseHeaderTimeout: 5 * time.Second,
		},
	}

	start := time.Now()

	resp, err := askClient.Do(req)
	if err != nil {
		return fmt.Errorf("cannot reach HTTP Gateway at %s: %w", gatewayEndpoint, err)
	}

	rtt := time.Since(start)

	defer func() {
		_ = resp.Body.Close()
	}()

	err = printResponse(resp)
	if err != nil {
		return err
	}

	if flagRTT {
		fmt.Printf("RTT: %s\n", rtt)
	}

	return nil
}

func buildRequestBody(prompt string) ([]byte, error) {
	templatePath, err := loadTemplatePath()
	if err != nil {
		return nil, err
	}

	templateBytes, err := os.ReadFile(templatePath)
	if err != nil {
		return nil, fmt.Errorf("cannot read LLM request at %q: %w", templatePath, err)
	}

	if !bytes.Contains(templateBytes, []byte(promptPlaceholder)) {
		return nil, fmt.Errorf(
			"LLM template %q no longer contains placeholder %s. "+
				"Re-run 'strix config set --template-path' to re-validate",
			templatePath, promptPlaceholder,
		)
	}

	escapedBytes, err := json.Marshal(prompt)
	if err != nil {
		return nil, fmt.Errorf("")
	}

	escapedBytes = escapedBytes[1 : len(escapedBytes)-1]
	result := bytes.Replace(templateBytes, []byte(promptPlaceholder), []byte(escapedBytes), 1)

	return result, nil
}

func loadTemplatePath() (string, error) {
	envPath, err := envFilePath()
	if err != nil {
		return "", err
	}

	envMap, err := godotenv.Read(envPath)
	if err != nil {
		return "", fmt.Errorf("cannot read %s: %w", envPath, err)
	}

	templatePath, ok := envMap[templatePathVar]
	if !ok || templatePath == "" {
		return "", fmt.Errorf(
			"template path not configured - run 'strix config set --template-path <path>'",
		)
	}

	return templatePath, nil
}

func printResponse(resp *http.Response) error {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("cannot read response body: %w", err)
	}

	switch resp.StatusCode {
	case http.StatusOK:
		fmt.Println(string(body))

	case http.StatusNotFound:
		fmt.Println("Strix was unable to respond due to: uncached response.")

	default:
		return fmt.Errorf("unexpected response from Gateway (HTTP %d): %s",
			resp.StatusCode, string(body),
		)
	}

	return nil
}
