package limits

import "sync"

const lfuSampleSize = 5

// Lock ordering (MUST follow to prevent deadlock):
//   Shard.RWMutex → RateWindow.Mutex
// Never acquire shard lock while holding RateWindow.mu

type Shard struct {
	sync.RWMutex
	clients map[string]*RateWindow
}

func (s *Shard) evictOneLFU() {
	type candidate struct {
		key    string
		weight int64
	}

	var (
		samples [lfuSampleSize]candidate
		count   int
	)

	for key, rw := range s.clients {
		weight := rw.totalCount
		samples[count] = candidate{key, weight}

		count++
		if count >= lfuSampleSize {
			break
		}
	}

	if count == 0 {
		return
	}

	for i := range count {
		if samples[i].weight == 1 {
			delete(s.clients, samples[i].key)
			return
		}
	}

	minIdx := 0
	for i := 1; i < count; i++ {
		if samples[i].weight < samples[minIdx].weight {
			minIdx = i
		}
	}

	delete(s.clients, samples[minIdx].key)
}
