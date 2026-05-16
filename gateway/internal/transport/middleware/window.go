package middleware

import "sync"

type RateWindow struct {
	mu          sync.Mutex
	prevCount   int64
	currCount   int64
	totalCount  int64
	windowStart int64
}
