package precache

import (
	"bufio"
	"context"
	"fmt"
	"gateway/internal/pb"
	"log/slog"
	"os"
	"sync"
	"time"
)

const (
	numWorkers = 8
	rpcTimeout = 120 * time.Millisecond
)

type Pipeline struct {
	stub   pb.CacheServiceClient
	files  []string
	strict bool
}

func NewPipeLine(
	stub pb.CacheServiceClient, files []string, strict bool,
) *Pipeline {
	return &Pipeline{stub: stub, files: files, strict: strict}
}

// Run orchestrates the full precache pipeline:
//  1. Stream JSONL records into a jobs channel (no full load into memory).
//  2. Fan-out to numWorkers goroutines executing CheckCache -> SetCache.
//  3. Collect outcomes, report live progress, enforce shed threshold.
func (p *Pipeline) Run(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	total, err := p.countRecords()
	if err != nil {
		return err
	}

	if total == 0 {
		slog.Warn("No valid records found. Nothing to cache.")
		return nil
	}

	jobs := make(chan record, 2*numWorkers)
	outcomes := make(chan outcome, 2*numWorkers)

	var feedErr error
	// Stream records into buffered channel
	go func() {
		feedErr = p.streamRecords(ctx, jobs)
		close(jobs)
	}()

	// Fan-out to worker goroutines
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for range numWorkers {
		go func() {
			defer wg.Done()
			p.runWorker(ctx, jobs, outcomes)
		}()
	}

	go func() {
		wg.Wait()
		close(outcomes)
	}()

	// Collect outcomes, drive progress, enforce threshold
	collectErr := p.collect(outcomes, cancel, total)
	if collectErr != nil {
		return collectErr
	}

	if feedErr != nil {
		return fmt.Errorf("error reading JSONL input: %w", feedErr)
	}

	return nil
}

// countRecords does a fast pre-scan to get the total line count.
// Used solely for progress percentage; does not validate JSON.
func (p *Pipeline) countRecords() (int, error) {
	total := 0

	for _, filePath := range p.files {
		file, err := os.Open(filePath)
		if err != nil {
			return 0, fmt.Errorf("cannot open %s: %w", filePath, err)
		}

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			total++
		}

		_ = file.Close()

		err = scanner.Err()
		if err != nil {
			return 0, fmt.Errorf("error scanning %s: %w", filePath, err)
		}
	}

	return total, nil
}

func (p *Pipeline) runWorker(
	ctx context.Context, jobs <-chan record, outcomes chan<- outcome,
) {
	for rec := range jobs {
		outcomes <- p.processRecord(ctx, rec)
	}
}

func (p *Pipeline) processRecord(ctx context.Context, rec record) outcome {
	checkCtx, checkCancel := context.WithTimeout(ctx, rpcTimeout)
	defer checkCancel()

	checkResp, err := p.stub.CheckCache(checkCtx, &pb.CheckCacheRequest{
		Prompt: rec.Prompt,
	})
	if err != nil {
		return outcome{
			err: fmt.Errorf("CheckCache RPC failed: %w", err),
		}
	}

	switch checkResp.GetCheckState() {
	case pb.CacheState_CACHE_STATE_HIT, pb.CacheState_CACHE_STATE_PENDING:
		return outcome{}

	case pb.CacheState_CACHE_STATE_MISS:
		nodeID := checkResp.GetNodeId()

		if nodeID == -1 {
			return outcome{shed: true}
		}

		setCtx, setCancel := context.WithTimeout(ctx, rpcTimeout)
		defer setCancel()

		_, err = p.stub.SetCache(setCtx, &pb.SetCacheRequest{
			NodeId:          nodeID,
			UncachedPayload: rec.Payload,
		})
		if err != nil {
			return outcome{
				err: fmt.Errorf("SetCache RPC failed: %w", err),
			}
		}

		return outcome{}

	case pb.CacheState_CACHE_STATE_EXCEEDED:
		return outcome{
			err: fmt.Errorf("prompt exceeds token limit"),
		}

	default:
		return outcome{
			fatal: true,
			err: fmt.Errorf(
				"unspecified return state - Vector Engine maybe corrupted",
			),
		}
	}
}
