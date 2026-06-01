package precache

import (
	"context"
	"fmt"
	"log/slog"
)

const shedThreshold = 0.10

type outcome struct {
	shed  bool
	fatal bool
	err   error
}

type stats struct {
	total     int
	completed int
	shed      int
	skipped   int
	failed    int
}

func (p *Pipeline) collect(outcomes <-chan outcome, cancel context.CancelFunc, total int) error {
	s := stats{total: total}
	var finalErr error
	isAborting := false

	for o := range outcomes {
		if isAborting {
			continue
		}

		if o.fatal {
			cancel()
			isAborting = true
			finalErr = o.err

			fmt.Println()
			slog.Error("Fatal error during precache - Tearing down...",
				slog.Any("error", o.err),
			)

			continue
		}

		switch {
		case o.shed:
			s.shed++
			slog.Warn("Vector Engine load-shedding record.",
				slog.Int("shed_count", s.shed),
				slog.Int("total", s.total),
			)

		case o.err != nil:
			s.failed++
			slog.Warn("Non-fatal precache error.",
				slog.Any("error", o.err),
				slog.Int("failed_count", s.failed),
			)

		default:
			s.skipped++
		}

		s.completed++

		if s.completed >= 100 {
			shedRate := float64(s.shed) / float64(s.completed)
			if shedRate > shedThreshold {
				cancel()
				isAborting = true
				finalErr = fmt.Errorf(
					"precache aborted: shed rate %.1f%% exceeds %.0f%% threshold",
					shedRate*100, shedThreshold*100,
				)

				fmt.Println()
				slog.Error("Shedding threshold exceeded - Tearing down...")

				continue
			}
		}

		pct := s.completed / s.total * 100
		fmt.Printf(
			"\r[strix precache] Progress: %d%% (%d/%d, shed: %d, failed: %d)",
			pct, s.completed, s.total, s.shed, s.failed,
		)
	}

	if finalErr != nil {
		return finalErr
	}

	fmt.Println()
	slog.Info("Precache procedure completed.",
		slog.Int("total", s.total),
		slog.Int("completed", s.completed),
		slog.Int("shed", s.shed),
		slog.Int("failed", s.failed),
	)

	return nil
}
