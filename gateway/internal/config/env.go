package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

const (
	apiKeyVar   = "GATEWAY_APIKEY"
	endpointVar = "GATEWAY_ENDPOINT"
)

type GatewayConfig struct {
	APIKey     string
	Endpoint   string
	Cores      []int
	NumWorkers int
}

func Load() (GatewayConfig, error) {
	coreStr := os.Getenv("GATEWAY_CORES")
	if len(coreStr) == 0 {
		panic("[Gateway] FATAL: no core allocating for HTTP Gateway")
	}

	cores, parseErr := parseCoreList(coreStr)
	if parseErr != nil {
		return GatewayConfig{}, fmt.Errorf(
			"invalid GATEWAY_CORES %q: %w", coreStr, parseErr,
		)
	}

	apiKey := os.Getenv(apiKeyVar)
	endpoint := os.Getenv(endpointVar)

	if len(apiKey) == 0 || len(endpoint) == 0 {
		panic("[Gateway] FATAL: misconfigured API Key or Endpoint")
	}

	defer func() {
		_ = os.Unsetenv(apiKeyVar)
		_ = os.Unsetenv(endpointVar)
	}()

	return GatewayConfig{
		APIKey:     apiKey,
		Endpoint:   endpoint,
		Cores:      cores,
		NumWorkers: len(cores),
	}, nil
}

func (cfg GatewayConfig) String() string {
	return fmt.Sprintf(
		"GatewayConfig{APIKey: [REDACTED], Endpoint: %s, Cores: %v, NumWorkers: %d}",
		cfg.Endpoint, cfg.Cores, cfg.NumWorkers,
	)
}

func parseCoreList(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	cores := make([]int, 0, len(parts))

	for _, p := range parts {
		core, convErr := strconv.Atoi(strings.TrimSpace(p))
		if convErr != nil {
			return nil, fmt.Errorf("cannot parse core index %q: %w", p, convErr)
		}

		if core < 0 {
			return nil, fmt.Errorf("negative core index %q", core)
		}

		cores = append(cores, core)
	}

	if len(cores) == 0 {
		return nil, fmt.Errorf("empty core list")
	}

	return cores, nil
}
