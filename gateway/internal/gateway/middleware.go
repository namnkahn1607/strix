package gateway

import (
	"gateway/internal/limit"
	"net"
	"net/http"
	"strings"
)

var errTooManyIPReqs = []byte("429 Too Many Requests (IP Limit)\n")

type Middleware struct {
	ipLimiter *limit.RateLimiter
}

func NewMiddleware(ipLimiter *limit.RateLimiter) *Middleware {
	return &Middleware{ipLimiter: ipLimiter}
}

func (mdw *Middleware) Wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		clientIP := extractIP(r)
		if !mdw.ipLimiter.Allow(clientIP) {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write(errTooManyIPReqs)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (mdw *Middleware) Stop() {
	mdw.ipLimiter.Stop()
}

// extractIP returns the client IP for rate limiting.
// X-Forwarded-For is trusted unconditionally - this is correct ONLY when
// Strix runs behind a trusted reverse proxy (Nginx, Cloudflare, etc.)
// that strips/overwrites XFF before forwarding.
// If Strix is exposed directly to the Internet, XFF can be spoofed.
func extractIP(r *http.Request) string {
	// Prioritize request from another Proxy or Load Balancer
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		if idx := strings.IndexByte(xff, ','); idx > 0 {
			return strings.TrimSpace(xff[:idx])
		}

		return strings.TrimSpace(xff)
	}

	// Otherwise fallback to trivial client connection
	ip, _, parseErr := net.SplitHostPort(r.RemoteAddr)
	if parseErr != nil {
		return r.RemoteAddr
	}

	return ip
}
