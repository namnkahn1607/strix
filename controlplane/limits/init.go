package limits

import "time"

var serverStartTime = time.Now()

func monoMs() int64 {
	return time.Since(serverStartTime).Milliseconds()
}
