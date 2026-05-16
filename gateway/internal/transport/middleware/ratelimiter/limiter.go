package ratelimiter

import "time"

const batchSize = 1000

type RateLimiter struct {
	shards      [256]*Shard
	windowMs    int64 // Time window size
	limit       int64 // Request limit per window
	ttlMs       int64 // Maximal idle time
	stopSweeper chan struct{}
}

func NewLimiter(windowMs, limit, ttlMs int64) *RateLimiter {
	rl := &RateLimiter{
		windowMs:    windowMs,
		limit:       limit,
		ttlMs:       ttlMs,
		stopSweeper: make(chan struct{}),
	}

	for i := range rl.shards {
		rl.shards[i] = &Shard{
			clients: make(map[string]*RateWindow),
		}
	}

	go rl.sweepLoop()
	return rl
}

func (rl *RateLimiter) Allow(key string) bool {
	rw := rl.getOrCreate(key)
	now := monoMs()

	rw.mu.Lock()
	defer rw.mu.Unlock()

	elapsed := now - rw.windowStart
	windowPassed := elapsed / rl.windowMs
	if windowPassed > 0 {
		if windowPassed >= 2 {
			// Reset if IP remains silent more than 2 windows
			rw.prevCount = 0
		} else {
			rw.prevCount = rw.currCount
		}

		rw.currCount = 0
		rw.windowStart += windowPassed * rl.windowMs
		elapsed = now - rw.windowStart
	}

	weight := float64(rl.windowMs-elapsed) / float64(rl.windowMs)
	estimate := float64(rw.currCount) + float64(rw.prevCount)*weight

	if int64(estimate) >= rl.limit {
		return false
	}

	rw.currCount++
	return true
}

func (rl *RateLimiter) Stop() {
	close(rl.stopSweeper)
}

func (rl *RateLimiter) getOrCreate(key string) *RateWindow {
	shard := rl.getShard(key)

	shard.RLock()
	rw, fstExists := shard.clients[key]
	shard.RUnlock()

	if fstExists {
		return rw
	}

	shard.Lock()
	rw, secExists := shard.clients[key]
	if !secExists {
		rw = &RateWindow{
			windowStart: monoMs(),
		}

		shard.clients[key] = rw
	}
	shard.Unlock()

	return rw
}

func (rl *RateLimiter) getShard(key string) *Shard {
	return rl.shards[fnv1a(key)&0xFF]
}

func (rl *RateLimiter) sweepLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			rl.sweepAll()
		case <-rl.stopSweeper:
			return
		}
	}
}

func (rl *RateLimiter) sweepAll() {
	now := monoMs()
	for _, shard := range rl.shards {
		rl.sweepShard(shard, now)
	}
}

func (rl *RateLimiter) sweepShard(shard *Shard, now int64) {
	for {
		expired := make([]string, 0, batchSize)
		shard.RLock()
		for key, rw := range shard.clients {
			if len(expired) >= batchSize {
				break
			}

			rw.mu.Lock()
			idle := now - rw.windowStart
			rw.mu.Unlock()
			if idle > rl.ttlMs {
				expired = append(expired, key)
			}
		}
		shard.RUnlock()

		if len(expired) == 0 {
			return
		}

		shard.Lock()
		for _, key := range expired {
			rw, exists := shard.clients[key]
			if !exists {
				continue
			}

			rw.mu.Lock()
			idle := now - rw.windowStart
			rw.mu.Unlock()
			if idle > rl.ttlMs {
				delete(shard.clients, key)
			}
		}
		shard.Unlock()

		time.Sleep(5 * time.Millisecond)
	}
}

func fnv1a(s string) uint32 {
	var hash uint32 = 2166136261

	for i := 0; i < len(s); i++ {
		hash ^= uint32(s[i])
		hash *= 16777619
	}

	return hash
}
