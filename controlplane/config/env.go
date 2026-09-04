package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

const (
	apiKeyVar     = "GATEWAY_UPSTREAM_APIKEY"
	endpointVar   = "GATEWAY_UPSTREAM_ENDPOINT"
	promptPathVar = "GATEWAY_PROMPT_PATH"
)

type GatewayConfig struct {
	APIKey     string
	Endpoint   string
	Cores      []int
	NumWorkers int
	PromptPath []string
}

func Load() (GatewayConfig, error) {
	coreStr := os.Getenv("GATEWAY_CORES")
	if len(coreStr) == 0 {
		panic("FATAL: no CPU cores allocating for HTTP Gateway")
	}

	cores, err := parseCoreList(coreStr)
	if err != nil {
		return GatewayConfig{}, fmt.Errorf("invalid Gateway cores %q: %w", coreStr, err)
	}

	apiKey := os.Getenv(apiKeyVar)
	endpoint := os.Getenv(endpointVar)

	if len(apiKey) == 0 || len(endpoint) == 0 {
		panic("FATAL: misconfigured API Key or Endpoint")
	}

	promptPath, err := parsePromptPath(os.Getenv(promptPathVar))
	if err != nil {
		return GatewayConfig{}, fmt.Errorf("invalid Prompt path %s: %w", promptPath, err)
	}

	return GatewayConfig{
		APIKey:     apiKey,
		Endpoint:   endpoint,
		Cores:      cores,
		NumWorkers: len(cores),
		PromptPath: promptPath,
	}, nil
}

func (cfg GatewayConfig) String() string {
	return fmt.Sprintf(
		`{
			APIKey: [REDACTED], 
			Endpoint: %s, 
			Cores: %v, 
			NumWorkers: %d, 
			PromptPath: %v
		}`,
		cfg.Endpoint, cfg.Cores, cfg.NumWorkers, cfg.PromptPath,
	)
}

func parseCoreList(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	cores := make([]int, 0, len(parts))

	for _, p := range parts {
		core, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return nil, fmt.Errorf("cannot parse core index %q: %w", p, err)
		}

		if core < 0 {
			return nil, fmt.Errorf("negative core found %q", core)
		}

		cores = append(cores, core)
	}

	if len(cores) == 0 {
		return nil, fmt.Errorf("empty core list")
	}

	return cores, nil
}

func parsePromptPath(s string) ([]string, error) {
	if s == "" {
		return nil, nil
	}

	parts := strings.Split(s, ",")
	for _, part := range parts {
		if strings.TrimSpace(part) == "" {
			return nil, fmt.Errorf("empty segment in prompt path %q", s)
		}
	}

	return parts, nil
}
