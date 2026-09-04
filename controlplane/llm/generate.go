package llm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
)

func (c *Client) Generate(ctx context.Context, body []byte) ([]byte, error) {
	req, reqErr := http.NewRequestWithContext(
		ctx, http.MethodPost, c.endpoint, bytes.NewReader(body),
	)
	if reqErr != nil {
		return nil, fmt.Errorf("cannot create request to LLM: %w", reqErr)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	res, doErr := c.http.Do(req)
	if doErr != nil {
		return nil, fmt.Errorf("call to LLM failed: %w", doErr)
	}

	defer func() {
		_ = res.Body.Close()
	}()

	if res.StatusCode != http.StatusOK {
		// Drain response body to reuse connection
		_, _ = io.Copy(io.Discard, res.Body)
		return nil, fmt.Errorf("LLM returned HTTP %d", res.StatusCode)
	}

	payload, readErr := io.ReadAll(res.Body)
	if readErr != nil {
		return nil, fmt.Errorf("cannot read LLM response body: %w", readErr)
	}

	return payload, nil
}
