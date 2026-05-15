package core

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestThunderingHerd_SingleLLMCall(t *testing.T) {
	const (
		numRequests = 100
		nodeID      = int32(42)
		wantPayload = `{"choices":[{"message":{"content":"mocked response"}}]}`
		llmLatency  = 30 * time.Millisecond
	)

	promise := PioneerRegister(nodeID)

	var (
		llmCallCount atomic.Int32

		mu         sync.Mutex
		gotPayload = make([]string, 0, numRequests-1)

		wg sync.WaitGroup
	)

	for i := 1; i < numRequests; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			payload, err, _, found := HerdAwait(context.Background(), nodeID)

			if !found {
				t.Errorf("herd goroutine: promise not found — registry invariant violated")
				return
			}

			if err != nil {
				t.Errorf("herd goroutine: unexpected error: %v", err)
				return
			}

			mu.Lock()
			gotPayload = append(gotPayload, string(payload))
			mu.Unlock()
		}()
	}

	llmCallCount.Add(1)
	time.Sleep(llmLatency)
	PioneerFulfill(nodeID, promise, []byte(wantPayload), nil)

	mu.Lock()
	gotPayload = append(gotPayload, wantPayload)
	mu.Unlock()

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:

	case <-time.After(5 * time.Second):
		t.Fatal("test timed out — goroutines leaked or deadlocked in herdAwait")
	}

	if got := llmCallCount.Load(); got != 1 {
		t.Errorf("LLM call count = %d, want 1: thundering herd not prevented", got)
	}

	mu.Lock()
	totalResults := len(gotPayload)
	mu.Unlock()

	if totalResults != numRequests {
		t.Errorf("received %d payloads, want %d", totalResults, numRequests)
	}

	for i, p := range gotPayload {
		if p != wantPayload {
			t.Errorf("payload[%d] = %q, want %q", i, p, wantPayload)
		}
	}
}

func TestHerdAwait_FinishedPioneer(t *testing.T) {
	const nodeID = int32(99)

	promise := PioneerRegister(nodeID)
	PioneerFulfill(nodeID, promise, []byte("data"), nil)

	payload, err, _, found := HerdAwait(context.Background(), nodeID)
	if found {
		t.Errorf(
			"expected found=false (Promise already removed), got true - payload=%q err=%v",
			payload, err,
		)
	}
}

func TestHerdAwait_ContextCancellation(t *testing.T) {
	const nodeID = int32(55)

	promise := PioneerRegister(nodeID)
	defer PioneerFulfill(nodeID, promise, nil, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err, _, found := HerdAwait(ctx, nodeID)

	if !found {
		t.Error("expected found=true (Promise exists in registry), got false")
	}

	if err == nil {
		t.Error(
			"expected ctx.Err() to propagate as err, got nil - " +
				"goroutines might not respect context cancellation",
		)
	}
}

func TestLockStriping_SameShardDifferentKeys(t *testing.T) {
	const (
		nodeA = int32(10)
		nodeB = int32(10 + NumShards)
	)

	pA := PioneerRegister(nodeA)
	pB := PioneerRegister(nodeB)

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		payload, err, _, found := HerdAwait(context.Background(), nodeA)
		if !found || err != nil || string(payload) != "payload-A" {
			t.Errorf("nodeA: got payload=%q found=%v err=%v", payload, found, err)
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		payload, err, _, found := HerdAwait(context.Background(), nodeB)
		if !found || err != nil || string(payload) != "payload-B" {
			t.Errorf("nodeB: got payload=%q found=%v err=%v", payload, found, err)
		}
	}()

	time.Sleep(10 * time.Millisecond)

	PioneerFulfill(nodeA, pA, []byte("payload-A"), nil)
	PioneerFulfill(nodeB, pB, []byte("payload-B"), nil)

	wg.Wait()
}
