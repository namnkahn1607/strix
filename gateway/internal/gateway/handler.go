package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	pb "gateway/pb/proto"
	"log"
	"net/http"
	"time"

	"github.com/VictoriaMetrics/fastcache"
)

const (
	serviceTimeout = 50 * time.Millisecond

	maxReaderSize    = 1024 * 1024 // 1MB
	maxPromptLen     = 512         // 512B
	maxL0PayloadSize = 64 * 1024   // 64KB
)

type CheckCacheAPIRequest struct {
	Prompt  []byte `json:"prompt"`
	LLMBody []byte `json:"llm_body"`
}

func (g *Gateway) Handle(w http.ResponseWriter, r *http.Request) {
	// 1. Validates the request method to be POST.
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	// 2. Decode the JSON body
	var apiReq CheckCacheAPIRequest
	r.Body = http.MaxBytesReader(w, r.Body, maxReaderSize)

	decErr := json.NewDecoder(r.Body).Decode(&apiReq)
	if decErr != nil {
		log.Printf("[Gateway] Decoding error: %v\n", decErr)
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	// 3.1. Compute SHA-256 hashcode of the prompt.
	hash := sha256.Sum256(apiReq.Prompt)
	hashKey := hash[:]

	// 3.2. Check on Exact-match Cache using hashcode.
	// If hit, return immediately.
	matchedPayload := g.L0Cache.Get(nil, hashKey)
	if matchedPayload != nil {
		writePayload(w, matchedPayload)
		return
	}

	fallbackToLLM := func() ([]byte, error) {
		payload, llmErr := g.LLMClient.Generate(
			r.Context(), apiReq.LLMBody,
		)
		if llmErr != nil {
			log.Printf("[Gateway] LLM error: %v\n", llmErr)
			http.Error(w, "Bad Gateway", http.StatusBadGateway)
			return nil, llmErr
		}

		writePayload(w, payload)
		trySetL0(g.L0Cache, hashKey, payload)
		return payload, nil
	}

	// 4. Prompts longer than 512 bytes are forward to LLM Provider.
	if len(apiReq.Prompt) > maxPromptLen {
		_, _ = fallbackToLLM()
		return
	}

	// 5. Short prompts are handled to Vector Engine via gRPC.
	ctx, cancel := context.WithTimeout(r.Context(), serviceTimeout)
	defer cancel()

	grpcRes, rpcErr := g.Stub.CheckCache(
		ctx, &pb.CheckCacheRequest{Prompt: apiReq.Prompt},
	)
	if rpcErr != nil {
		log.Printf("[Gateway] Read RPC error: %v\n", rpcErr)
		_, _ = fallbackToLLM()
		return
	}

	// 6. Investigate returned state and act correspondingly.
	nodeID := grpcRes.GetNodeId()

	switch grpcRes.GetCheckState() {
	case pb.CacheState_CACHE_STATE_HIT:
		writePayload(w, grpcRes.GetCachedPayload())

	case pb.CacheState_CACHE_STATE_MISS:
		if nodeID == -1 {
			_, _ = fallbackToLLM()
			return
		}

		promise := g.HerdCtrl.Register(nodeID)
		var llmPayload []byte
		var llmErr error

		// Guarantee pioneerFulfill getting called exactly ONCE,
		// hence avoiding leaked herd goroutines.
		defer func() {
			g.HerdCtrl.Fulfill(nodeID, promise, llmPayload, llmErr)
		}()

		llmPayload, llmErr = fallbackToLLM()
		if llmErr == nil {
			g.Pool.TryEnqueue(nodeID, llmPayload)
		}

	case pb.CacheState_CACHE_STATE_PENDING:
		payload, pioneerErr, selfCancelled, found := g.HerdCtrl.Await(
			r.Context(), nodeID,
		)

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

	_, writeErr := w.Write(payload)
	if writeErr != nil {
		log.Printf("[Gateway] Response writing error: %v\n", writeErr)
	}
}

func trySetL0(cache *fastcache.Cache, key, payload []byte) {
	if len(payload) < maxL0PayloadSize {
		// Cache.Set() is lock-free and uses memcopy at the OS level.
		cache.Set(key, payload)
	}
}
