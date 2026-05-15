package middleware

import (
	"net/http"

	"golang.org/x/time/rate"
)

func RateLimiter(limiter *rate.Limiter, next http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !limiter.Allow() {
			http.Error(w, "Too Many Request", http.StatusTooManyRequests)
			return
		}

		next.ServeHTTP(w, r)
	}
}
