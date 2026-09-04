package concurrent

import (
	"context"
	"log/slog"
	"strix/pb"
	"sync"
	"time"
)

const (
	jobQueueCap = 5000
	jobTimeout  = 50 * time.Millisecond
)

type Job struct {
	NodeID  int32
	Payload []byte
}

type WorkerPool struct {
	queue chan Job
	wg    sync.WaitGroup
}

func NewWorkerPool(stub pb.CacheServiceClient, numWorkers int) *WorkerPool {
	wp := &WorkerPool{
		queue: make(chan Job, jobQueueCap),
	}

	wp.wg.Add(numWorkers)
	for range numWorkers {
		go wp.runWorker(stub)
	}

	return wp
}

func (wp *WorkerPool) TryEnqueue(nodeID int32, payload []byte) {
	select {
	case wp.queue <- Job{NodeID: nodeID, Payload: payload}:

	default:
		// Silent drop in case cannot schedule more job
	}
}

func (wp *WorkerPool) Stop(ctx context.Context) error {
	close(wp.queue)

	drained := make(chan struct{})
	go func() {
		wp.wg.Wait()
		close(drained)
	}()

	select {
	case <-drained:
		return nil

	case <-ctx.Done():
		slog.Warn("Stop deadline exceeded, abandoning remaining jobs in queue",
			slog.Any("error", ctx.Err()),
		)

		return ctx.Err()
	}
}

func (wp *WorkerPool) runWorker(stub pb.CacheServiceClient) {
	defer wp.wg.Done()
	for job := range wp.queue {
		ctx, cancel := context.WithTimeout(context.Background(), jobTimeout)

		_, err := stub.SetCache(ctx, &pb.SetCacheRequest{
			NodeId:          job.NodeID,
			UncachedPayload: job.Payload,
		})

		cancel()

		if err != nil {
			slog.Warn("Failed to write uncached payload",
				slog.Int("node_id", int(job.NodeID)),
				slog.Any("error", err),
			)
		}
	}
}
