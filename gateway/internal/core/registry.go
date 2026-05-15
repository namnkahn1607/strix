package core

import (
	"context"
	"sync"
)

const NumShards = 256

type Promise struct {
	Payload []byte
	Err     error
	ready   chan struct{}
}

// Lock Striping won't lock the whole Shard-array.
type promiseShard struct {
	mu       sync.RWMutex
	registry map[int32]*Promise
	// Avoid False Sharing between Shards by padding,
	// fitting each of them well in a CPU Cache line.
	_ [32]byte
}

var shards [NumShards]promiseShard

func init() {
	for i := range shards {
		shards[i].registry = make(map[int32]*Promise)
	}
}

func getShard(nodeID int32) *promiseShard {
	return &shards[uint32(nodeID)&(NumShards-1)]
}

// PioneerRegister is called by pioneer LLM request, which will create a
// broadcasting channel inside Promise, then register it to the Shard-array.
func PioneerRegister(nodeID int32) *Promise {
	p := &Promise{
		ready: make(chan struct{}),
	}

	shard := getShard(nodeID)
	shard.mu.Lock()
	shard.registry[nodeID] = p
	shard.mu.Unlock()
	return p
}

// PioneerFulfill injects payload, (potential) error, hence broadcasting
// the result to its herd by closing the channel.
// The registered Promise would also be removed.
func PioneerFulfill(nodeID int32, p *Promise, payload []byte, err error) {
	p.Payload = payload
	p.Err = err

	close(p.ready)

	shard := getShard(nodeID)
	shard.mu.Lock()
	delete(shard.registry, nodeID)
	shard.mu.Unlock()
}

// HerdAwait called by herd, which will look up the Promise of the
// corresponding Node ID, then blocks until:
// a. The pioneer fulfills the Promise.
// b. Its context is canceled (client disconnect/timeout).
func HerdAwait(
	ctx context.Context, nodeID int32,
) (payload []byte, pioneerErr error, selfCancelled bool, found bool) {
	shard := getShard(nodeID)
	shard.mu.RLock()
	p := shard.registry[nodeID]
	shard.mu.RUnlock()

	if p == nil {
		// Promise not in registry, pioneer closed it duty.
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
