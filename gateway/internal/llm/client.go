package llm

import (
	"net/http"
	"time"
)

type Client struct {
	http     *http.Client
	apiKey   string
	endpoint string
}

func CreateClient(apiKey, endpoint string, timeout time.Duration) *Client {
	return &Client{
		apiKey:   apiKey,
		endpoint: endpoint,
		http: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}
