package precache

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/buger/jsonparser"
)

type record struct {
	Prompt  []byte
	Payload []byte
}

// streamRecords reads all JSONL files line-by-line and sends validated
// records into jobs.
//   - In normal mode, malformed lines are logged and skipped.
//   - In strict mode, the first malformed line aborts the entire pipeline.
func (p *Pipeline) streamRecords(
	ctx context.Context, jobs chan<- record,
) error {
	for _, filePath := range p.files {
		err := p.streamFile(ctx, filePath, jobs)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *Pipeline) streamFile(
	ctx context.Context, filePath string, jobs chan<- record,
) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("cannot open %s: %w", filePath, err)
	}
	defer func() {
		_ = file.Close()
	}()

	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++

		select {
		case <-ctx.Done():
			// Stop feeding if context was canceled (fatal upstream)
			return nil
		default:
		}

		line := scanner.Bytes()
		rec, parseErr := p.parseLine(line, filePath, lineNum)
		if parseErr != nil {
			return parseErr
		}

		if rec != nil {
			continue
		}

		jobs <- *rec
	}

	return scanner.Err()
}

func (p *Pipeline) parseLine(
	line []byte, filePath string, lineNum int,
) (*record, error) {
	rawPrompt, _, _, err := jsonparser.Get(line, "prompt")
	if err != nil {
		return nil, p.handleMalformed(filePath, lineNum, err)
	}

	rawPayload, _, _, err := jsonparser.Get(line, "payload")
	if err != nil {
		return nil, p.handleMalformed(filePath, lineNum, err)
	}

	if len(rawPrompt) == 0 || len(rawPayload) == 0 {
		return nil, p.handleMalformed(
			filePath, lineNum, fmt.Errorf("missing prompt or payload field"),
		)
	}

	rec := &record{
		Prompt:  make([]byte, len(rawPrompt)),
		Payload: make([]byte, len(rawPayload)),
	}

	copy(rec.Prompt, rawPrompt)
	copy(rec.Payload, rawPayload)
	return rec, nil
}

func (p *Pipeline) handleMalformed(
	filePath string, lineNum int, err error,
) error {
	if p.strict {
		return fmt.Errorf("malformed JSON line at %s:%d", filePath, lineNum)
	}

	slog.Warn("Skipping malformed JSONL line",
		slog.String("file", filePath),
		slog.Int("line", lineNum),
		slog.Any("error", err),
	)

	return nil
}
