package middleware

import "sync"

// Lock ordering (MUST follow to prevent deadlock):
//   Shard.RWMutex → RateWindow.Mutex
// Never acquire shard lock while holding RateWindow.mu

type RateWindow struct {
	mu          sync.Mutex
	prevCount   int64
	currCount   int64
	totalCount  int64
	windowStart int64
}
