package middleware

import "sync"

type Shard struct {
	sync.RWMutex
	clients map[string]*RateWindow
}
