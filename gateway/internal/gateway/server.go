package gateway

import (
	"gateway/internal/concurrent"
	"gateway/internal/limit"
	"gateway/internal/llm"
	"gateway/internal/pb"
	"net/http"
	"time"

	"github.com/VictoriaMetrics/fastcache"
)

const (
	gatewayEndpoint = "/v1/cache/strix"
	gatewayPort     = ":8080"
)

type Dependencies struct {
	Stub       pb.CacheServiceClient
	L0Cache    *fastcache.Cache
	Pool       *concurrent.WorkerPool
	HerdCtrl   *HerdController
	IPLimiter  *limit.RateLimiter
	DeadChan   <-chan struct{}
	LLMClient  *llm.Client
	PromptPath []string
}

type Gateway struct {
	Dependencies
	httpServer *http.Server
}

func CreateGateway(deps Dependencies) *Gateway {
	gw := &Gateway{Dependencies: deps}
	mux := http.NewServeMux()

	handler := NewMiddleware(deps.IPLimiter).Wrap(http.HandlerFunc(gw.Handle))
	mux.Handle(gatewayEndpoint, handler)

	gw.httpServer = &http.Server{
		Addr:              gatewayPort,
		Handler:           mux,
		ReadHeaderTimeout: 3 * time.Second,
		ReadTimeout:       5 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	return gw
}
