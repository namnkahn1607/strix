package concurrent

import (
	"context"
	pb "gateway/pb/proto"
	"log"
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

func NewWorkerPool(stub pb.SemanticServiceClient, numWorkers int) *WorkerPool {
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
		log.Printf(
			"[WorkerPool] Stop deadline exceeded (%v): "+
				"abandoning remaining jobs in queue\n", ctx.Err(),
		)

		return ctx.Err()
	}
}

func (wp *WorkerPool) runWorker(stub pb.SemanticServiceClient) {
	for job := range wp.queue {
		ctx, cancel := context.WithTimeout(context.Background(), jobTimeout)

		_, rpcErr := stub.SetCache(ctx, &pb.SetCacheRequest{
			NodeId:          job.NodeID,
			UncachedPayload: job.Payload,
		})

		cancel()

		if rpcErr != nil {
			log.Printf("[WorkerPool] RPC Write failed for nodeID = %d: %v\n",
				job.NodeID, rpcErr,
			)
		}
	}

	wp.wg.Done()
}
