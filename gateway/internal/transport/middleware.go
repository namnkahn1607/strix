package transport

import (
	"gateway/internal/transport/middleware"
	"net"
	"net/http"
	"strings"
)

const (
	defaultWindowSize = 1000
	defaultLimit      = 200
	defaultTTLMs      = 5 * 60 * 1000
)

var (
	errTooManyIPReqs = []byte("429 Too Many Requests (IP Limit)\n")
)

type Middleware struct {
	ipLimiter *middleware.RateLimiter
}

func NewMiddleware() *Middleware {
	return &Middleware{
		ipLimiter: middleware.NewLimiter(
			defaultWindowSize, defaultLimit, defaultTTLMs,
		),
	}
}

func (m *Middleware) Wrap(next http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		clientIP := extractIP(r)
		if !m.ipLimiter.Allow(clientIP) {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write(errTooManyIPReqs)
			return
		}

		next.ServeHTTP(w, r)
	}
}

func extractIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		if idx := strings.IndexByte(xff, ','); idx > 0 {
			return xff[:idx]
		}

		return xff
	}

	ip, _, parseErr := net.SplitHostPort(r.RemoteAddr)
	if parseErr != nil {
		return r.RemoteAddr
	}

	return ip
}
