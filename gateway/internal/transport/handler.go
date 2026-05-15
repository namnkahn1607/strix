package transport

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"gateway/internal/core"
	pb "gateway/pb/proto"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/VictoriaMetrics/fastcache"
)

const (
	ServiceTimeout = 50 * time.Millisecond

	LLMForwardTimeout = 10 * time.Second

	maxReaderSize = 1024 * 1024
	maxPromptLen  = 512

	maxL0PayloadSize = 64 * 1024
)

var (
	errMisConfiguredCredential = errors.New(
		"API Key or Endpoint is not configured - run 'strix config set'",
	)
)

type CheckCacheAPIRequest struct {
	Prompt  []byte `json:"prompt"`
	LLMBody []byte `json:"llm_body"`
}

// StrixService returns an http.HandlerFunc that:
//  1. Validates the request method to be POST.
//  2. Decodes the JSON body.
//  3. Computes SHA-256 hash of the prompt and queries the L0 exact-match cache.
//  4. Falls through to the LLM provider for long prompts (> 512 bytes).
//  5. Falls through to the Vector Engine for short prompts.
func StrixService(
	stub pb.SemanticServiceClient, l0Cache *fastcache.Cache,
	fatalErrChan chan<- error, pool *core.WorkerPool,
	apiKey, endpoint string,
) http.HandlerFunc {
	if len(apiKey) == 0 || len(endpoint) == 0 {
		go func() {
			fatalErrChan <- errMisConfiguredCredential
		}()
	}

	llmTransport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
	}

	llmClient := &http.Client{
		Timeout:   LLMForwardTimeout,
		Transport: llmTransport,
	}

	return func(w http.ResponseWriter, r *http.Request) {
		// 1. Validates the request method to be POST.
		if r.Method != http.MethodPost {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}

		// 2. Decodes the JSON body.
		var apiReq CheckCacheAPIRequest
		r.Body = http.MaxBytesReader(w, r.Body, maxReaderSize)
		if decErr := json.NewDecoder(r.Body).Decode(&apiReq); decErr != nil {
			log.Printf("[Handler] Decoding error: %v\n", decErr)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}

		// 3.1. Computes SHA-256 hash of the prompt
		hash := sha256.Sum256(apiReq.Prompt)
		hashKey := hash[:]

		// 3.2. Check on L0 Fast Cache using hashcode. If hit, return immediately.
		if matchedPayload := l0Cache.Get(nil, hashKey); matchedPayload != nil {
			writePayload(w, matchedPayload)
			return
		}

		// 4. Prompts longer than 512 bytes are forward to LLM Provider.
		if len(apiReq.Prompt) > maxPromptLen {
			llmPayload, llmErr := forwardToLLM(
				r.Context(), llmClient, endpoint, apiKey, apiReq.LLMBody,
			)
			if llmErr != nil {
				handleLLMError(w, llmErr, fatalErrChan)
				return
			}

			writePayload(w, llmPayload)
			trySetL0(l0Cache, hashKey, llmPayload)
			return
		}

		// 5. Short prompts handled to Vector Engine via gRPC.
		ctx, cancel := context.WithTimeout(r.Context(), ServiceTimeout)
		defer cancel()

		grpcRes, rpcErr := stub.CheckCache(ctx, &pb.CheckCacheRequest{Prompt: apiReq.Prompt})
		if rpcErr != nil {
			log.Printf("[Handler] Read RPC error: %v\n", rpcErr)
			llmPayload, llmErr := forwardToLLM(
				r.Context(), llmClient, endpoint, apiKey, apiReq.LLMBody,
			)
			if llmErr != nil {
				handleLLMError(w, llmErr, fatalErrChan)
				return
			}

			writePayload(w, llmPayload)
			trySetL0(l0Cache, hashKey, llmPayload)
			return
		}

		// 6. Investigate CheckCache state and act correspondingly.
		nodeID := grpcRes.GetNodeId()

		switch grpcRes.GetCheckState() {
		case pb.CacheState_CACHE_STATE_HIT:
			writePayload(w, grpcRes.GetCachedPayload())

		case pb.CacheState_CACHE_STATE_MISS:
			var (
				llmPayload []byte
				llmErr     error
			)

			if nodeID == -1 {
				llmPayload, llmErr = forwardToLLM(
					r.Context(), llmClient, endpoint, apiKey, apiReq.LLMBody,
				)
				if llmErr != nil {
					handleLLMError(w, llmErr, fatalErrChan)
					return
				}

				writePayload(w, llmPayload)
				trySetL0(l0Cache, hashKey, llmPayload)
				return
			}

			promise := core.PioneerRegister(nodeID)

			// Guarantee pioneerFulfill getting called exactly ONCE,
			// hence avoiding leaked herd goroutines.
			defer func() {
				core.PioneerFulfill(nodeID, promise, llmPayload, llmErr)
			}()

			llmPayload, llmErr = forwardToLLM(
				r.Context(), llmClient, endpoint, apiKey, apiReq.LLMBody,
			)
			if llmErr != nil {
				handleLLMError(w, llmErr, fatalErrChan)
				return
			}

			writePayload(w, llmPayload)
			trySetL0(l0Cache, hashKey, llmPayload)
			pool.TryEnqueue(nodeID, llmPayload)

		case pb.CacheState_CACHE_STATE_PENDING:
			payload, pioneerErr, selfCancelled, found := core.HerdAwait(
				r.Context(), grpcRes.GetNodeId(),
			)

			if selfCancelled {
				return
			}

			if !found {
				llmPayload, llmErr := forwardToLLM(
					r.Context(), llmClient, endpoint, apiKey, apiReq.LLMBody,
				)
				if llmErr != nil {
					handleLLMError(w, llmErr, fatalErrChan)
					return
				}

				writePayload(w, llmPayload)
				trySetL0(l0Cache, hashKey, llmPayload)
				pool.TryEnqueue(nodeID, llmPayload)
				return
			}

			if pioneerErr != nil {
				llmPayload, llmErr := forwardToLLM(
					r.Context(), llmClient, endpoint, apiKey, apiReq.LLMBody,
				)
				if llmErr != nil {
					handleLLMError(w, llmErr, fatalErrChan)
					return
				}

				writePayload(w, llmPayload)
				trySetL0(l0Cache, hashKey, llmPayload)
				pool.TryEnqueue(nodeID, llmPayload)
				return
			}

			writePayload(w, payload)
			trySetL0(l0Cache, hashKey, payload)

		default:
			log.Printf(
				"Unexpected Check Cache state %v. Falling back to LLM",
				grpcRes.GetCheckState(),
			)

			llmPayload, llmErr := forwardToLLM(
				r.Context(), llmClient, endpoint, apiKey, apiReq.LLMBody,
			)
			if llmErr != nil {
				handleLLMError(w, llmErr, fatalErrChan)
				return
			}

			writePayload(w, llmPayload)
			trySetL0(l0Cache, hashKey, llmPayload)
		}
	}
}

func writePayload(w http.ResponseWriter, payload []byte) {
	w.Header().Set("Content-Type", "application/json")
	if _, writeErr := w.Write(payload); writeErr != nil {
		log.Printf("[Handler] Write-response error: %v\n", writeErr)
	}
}

func trySetL0(cache *fastcache.Cache, key, payload []byte) {
	if len(payload) < maxL0PayloadSize {
		// fastcache.Set() is lock-free and uses memcopy at the OS level.
		cache.Set(key, payload)
	}
}

func forwardToLLM(
	ctx context.Context, client *http.Client, endpoint, apiKey string, llmBody []byte,
) ([]byte, error) {
	req, reqErr := http.NewRequestWithContext(
		ctx, http.MethodPost, endpoint, bytes.NewReader(llmBody),
	)
	if reqErr != nil {
		return nil, fmt.Errorf("cannot build request to LLM: %w", reqErr)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	res, doErr := client.Do(req)
	if doErr != nil {
		return nil, fmt.Errorf("call to LLM failed: %w", doErr)
	}

	defer func() {
		_ = res.Body.Close()
	}()

	if res.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, res.Body)
		return nil, fmt.Errorf("LLM returned HTTP %d", res.StatusCode)
	}

	llmPayload, readErr := io.ReadAll(res.Body)
	if readErr != nil {
		return nil, fmt.Errorf("cannot read LLM response body: %w", readErr)
	}

	return llmPayload, nil
}

func handleLLMError(w http.ResponseWriter, err error, fatalErrChan chan<- error) {
	log.Printf("[Handler] LLM encounter error: %v\n", err)

	if errors.Is(err, errMisConfiguredCredential) {
		select {
		case fatalErrChan <- err:
		default:
		}

		http.Error(w, "HTTP Gateway Misconfigured", http.StatusInternalServerError)
		return
	}

	http.Error(w, "Bad Gateway", http.StatusBadGateway)
}
