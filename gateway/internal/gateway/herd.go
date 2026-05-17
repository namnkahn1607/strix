package gateway

import (
	"context"
	"sync"

	"golang.org/x/sys/cpu"
)

const numShards = 256

type Promise struct {
	Payload []byte
	Err     error
	ready   chan struct{}
}

type promiseShard struct {
	mu       sync.RWMutex
	registry map[int32]*Promise
	_        cpu.CacheLinePad
}

type HerdController struct {
	shards [numShards]promiseShard
}

func NewHerdController() *HerdController {
	hc := &HerdController{}

	for i := range numShards {
		hc.shards[i].registry = make(map[int32]*Promise)
	}

	return hc
}

// Register is called by pioneer LLM request, which will create a broadcasting
// channel inside Promise, then Register it to the HerdController.
func (hc *HerdController) Register(nodeID int32) *Promise {
	p := &Promise{
		ready: make(chan struct{}),
	}

	shard := hc.getShard(nodeID)
	shard.mu.Lock()
	shard.registry[nodeID] = p
	shard.mu.Unlock()
	return p
}

// Await called by herd, which will look up the Promise of the
// corresponding Node ID, then blocks until:
// a. The pioneer fulfills the Promise.
// b. Its context is canceled (client disconnect/timeout).
func (hc *HerdController) Await(
	ctx context.Context, nodeID int32,
) (payload []byte, pioneerErr error, selfCancelled bool, found bool) {
	shard := hc.getShard(nodeID)
	shard.mu.RLock()
	p := shard.registry[nodeID]
	shard.mu.RUnlock()

	if p == nil {
		// Promise not in registry, pioneer ended its duty.
		// Fall back to direct LLM call immediately
		return nil, nil, false, false
	}

	select {
	case <-p.ready:
		// Successfully fulfilled, use payload
		return p.Payload, p.Err, false, true

	case <-ctx.Done():
		// Herd's client disconnect or timer runs out
		return nil, ctx.Err(), true, true
	}
}

// Fulfill injects payload, (potential) error, hence broadcasting
// the result to its herd by closing the channel.
// The registered Promise would also be removed afterward.
func (hc *HerdController) Fulfill(
	nodeID int32, p *Promise, payload []byte, err error,
) {
	p.Payload = payload
	p.Err = err

	close(p.ready)

	shard := hc.getShard(nodeID)
	shard.mu.Lock()
	delete(shard.registry, nodeID)
	shard.mu.Unlock()
}

func (hc *HerdController) getShard(nodeID int32) *promiseShard {
	return &hc.shards[uint32(nodeID)&(numShards-1)]
}
