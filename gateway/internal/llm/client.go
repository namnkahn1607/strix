package llm

import (
	"net"
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
				IdleConnTimeout:     60 * time.Second,

				DialContext: (&net.Dialer{
					Timeout:   10 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,

				TLSHandshakeTimeout:   10 * time.Second,
				ResponseHeaderTimeout: timeout,
			},
		},
	}
}
