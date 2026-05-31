package gateway

import (
	"context"
	"crypto/sha256"
	pb "gateway/pb/proto"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/VictoriaMetrics/fastcache"
	"github.com/buger/jsonparser"
)

const (
	cacheOnlyHeader = "X-Strix-Cache-Only"

	maxReaderSize    = 1024 * 1024 // 1MB
	maxPromptLen     = 1536        // 1536B
	maxL0PayloadSize = 64 * 1024   // 64KB

	serviceTimeout = 50 * time.Millisecond
)

func (g *Gateway) Handle(w http.ResponseWriter, r *http.Request) {
	// 1. Validate the request method to be POST.
	cacheOnly := r.Header.Get(cacheOnlyHeader) == "true"

	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	// 2. Read request body into memory.
	r.Body = http.MaxBytesReader(w, r.Body, maxReaderSize)

	reqBody, err := io.ReadAll(r.Body)
	if err != nil {
		slog.Warn("Failed to read request body.", slog.Any("error", err))
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	// 3. Extract user prompt (zero allocation).
	promptPath := g.PromptPath
	if len(promptPath) == 0 {
		promptPath = []string{"prompt"}
	}

	prompt, _, _, err := jsonparser.Get(reqBody, promptPath...)
	if err != nil || len(prompt) == 0 {
		slog.Warn("Missing or empty prompt field.",
			slog.Any("path", promptPath),
			slog.Any("error", err),
		)
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	// 4.1. Compute SHA-256 hashcode of the prompt.
	hash := sha256.Sum256(prompt)
	hashKey := hash[:]

	// 5. Exact-match cache check - fastest, no gRPC involved.
	matchedPayload := g.L0Cache.Get(nil, hashKey)
	if matchedPayload != nil {
		writePayload(w, matchedPayload)
		return
	}

	fallbackToLLM := func() ([]byte, error) {
		if cacheOnly {
			http.Error(w, "Not Found", http.StatusNotFound)
			return nil, nil
		}

		payload, llmErr := g.LLMClient.Generate(r.Context(), reqBody)
		if llmErr != nil {
			slog.Warn("Failed to dial LLM Provider.", slog.Any("error", llmErr))
			http.Error(w, "Bad Gateway", http.StatusBadGateway)
			return nil, llmErr
		}

		writePayload(w, payload)
		trySetL0(g.L0Cache, hashKey, payload)
		return payload, nil
	}

	// 5. Prompts exceeding maxPromptLen bypass Vector Engine entirely.
	if len(prompt) > maxPromptLen {
		_, _ = fallbackToLLM()
		return
	}

	// 6. Short prompts: semantic cache lookup via gRPC to Vector Engine.
	ctx, cancel := context.WithTimeout(r.Context(), serviceTimeout)
	defer cancel()

	grpcRes, err := g.Stub.CheckCache(ctx, &pb.CheckCacheRequest{Prompt: prompt})
	if err != nil {
		slog.Warn("Failed to read cached payload.", slog.Any("error", err))
		_, _ = fallbackToLLM()
		return
	}

	// 7. Investigate returned state and act correspondingly.
	nodeID := grpcRes.GetNodeId()

	switch grpcRes.GetCheckState() {
	case pb.CacheState_CACHE_STATE_HIT:
		writePayload(w, grpcRes.GetCachedPayload())

	case pb.CacheState_CACHE_STATE_MISS:
		// Load shedding from Vector Engine - bypass cache write
		if nodeID == -1 {
			_, _ = fallbackToLLM()
			return
		}

		promise := g.HerdCtrl.Register(nodeID)
		var llmPayload []byte
		var llmErr error

		// Guarantee Fulfill is called exactly once to avoid herd goroutine leaks
		defer func() {
			g.HerdCtrl.Fulfill(nodeID, promise, llmPayload, llmErr)
		}()

		llmPayload, llmErr = fallbackToLLM()
		if llmErr == nil {
			g.Pool.TryEnqueue(nodeID, llmPayload)
		}

	case pb.CacheState_CACHE_STATE_PENDING:
		payload, pioneerErr, selfCancelled, found := g.HerdCtrl.Await(r.Context(), nodeID)

		if selfCancelled {
			return
		}

		if !found || pioneerErr != nil {
			llmPayload, llmErr := fallbackToLLM()
			if llmErr == nil {
				g.Pool.TryEnqueue(nodeID, llmPayload)
			}

			return
		}

		writePayload(w, payload)
		trySetL0(g.L0Cache, hashKey, payload)

	default:
		_, _ = fallbackToLLM()
	}
}

func writePayload(w http.ResponseWriter, payload []byte) {
	w.Header().Set("Content-Type", "application/json")

	_, err := w.Write(payload)
	if err != nil {
		slog.Warn("An error occurred upon writing HTTP response.",
			slog.Any("error", err),
		)
	}
}

func trySetL0(cache *fastcache.Cache, key, payload []byte) {
	if len(payload) < maxL0PayloadSize {
		// Cache.Set() is lock-free and uses memcopy at the OS level.
		cache.Set(key, payload)
	}
}
