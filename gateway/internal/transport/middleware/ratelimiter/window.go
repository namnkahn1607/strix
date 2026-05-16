package ratelimiter

import "sync"

type RateWindow struct {
	mu          sync.Mutex
	prevCount   int64
	currCount   int64
	windowStart int64
}
