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

func (p *Pipeline) collect(
	outcomes <-chan outcome, cancel context.CancelFunc, total int,
) error {
	s := stats{total: total}

	for o := range outcomes {
		if o.fatal {
			cancel()

			go func() {
				for range outcomes {
				}
			}()

			fmt.Println()
			slog.Error("Fatal error during precache - aborting",
				slog.Any("error", o.err),
			)

			return o.err
		}

		switch {
		case o.shed:
			s.shed++
			slog.Warn("Vector Engine load-shedding record",
				slog.Int("shed_count", s.shed),
				slog.Int("total", s.total),
			)

		case o.err != nil:
			s.failed++
			slog.Error("Non-fatal precache error",
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

				go func() {
					for range outcomes {
					}
				}()

				fmt.Println()
				return fmt.Errorf(
					"precache aborted: shed rate %.1f%% exceeds %.0f%% threshold "+
						"(%d/%d records shed) - Vector Engine may be overloaded",
					shedRate*100, shedThreshold*100, s.shed, s.completed,
				)
			}
		}

		pct := s.completed / s.total * 100
		fmt.Printf(
			"\r[strix precache] Progress: %d%% (%d/%d, shed: %d, failed: %d)",
			pct, s.completed, s.total, s.shed, s.failed,
		)
	}

	fmt.Println()
	slog.Info("Precache complete",
		slog.Int("total", s.total),
		slog.Int("completed", s.completed),
		slog.Int("shed", s.shed),
		slog.Int("failed", s.failed),
	)

	return nil
}
