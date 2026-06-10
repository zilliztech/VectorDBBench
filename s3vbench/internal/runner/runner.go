package runner

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"s3vbench/internal/synth"
	"s3vbench/internal/workload"
)

type Config struct {
	Workload workload.Config
	Client   Client
}

type Runner struct {
	cfg Config
}

type Result struct {
	RunID      string                    `json:"run_id"`
	Tool       ToolMetadata              `json:"tool"`
	StartTime  time.Time                 `json:"start_time"`
	EndTime    time.Time                 `json:"end_time"`
	Partial    bool                      `json:"partial"`
	Workload   workload.Config           `json:"workload"`
	Summary    Summary                   `json:"summary"`
	TimeSeries []TimeSample              `json:"time_series,omitempty"`
	Errors     []ErrorSample             `json:"errors,omitempty"`
	Host       HostMetadata              `json:"host"`
	BySecond   map[string]map[string]any `json:"-"`
}

type ToolMetadata struct {
	Version       string `json:"version"`
	GitCommit     string `json:"git_commit,omitempty"`
	AWSSDKVersion string `json:"aws_sdk_version,omitempty"`
	GoVersion     string `json:"go_version"`
}

type HostMetadata struct {
	Hostname string `json:"hostname,omitempty"`
}

type Summary struct {
	TotalRequests      uint64                                `json:"total_requests"`
	SuccessfulRequests uint64                                `json:"successful_requests"`
	FailedRequests     uint64                                `json:"failed_requests"`
	RetryAttempts      uint64                                `json:"retry_attempts"`
	ThrottleRequests   uint64                                `json:"throttle_requests"`
	TimeoutRequests    uint64                                `json:"timeout_requests"`
	BytesSent          int64                                 `json:"bytes_sent"`
	BytesReceived      int64                                 `json:"bytes_received"`
	QPS                float64                               `json:"qps"`
	ErrorRate          float64                               `json:"error_rate"`
	Latency            LatencySummary                        `json:"latency"`
	ByOperation        map[workload.Operation]OperationStats `json:"by_operation"`
	ErrorCounts        map[string]uint64                     `json:"error_counts,omitempty"`
}

type OperationStats struct {
	TotalRequests      uint64         `json:"total_requests"`
	SuccessfulRequests uint64         `json:"successful_requests"`
	FailedRequests     uint64         `json:"failed_requests"`
	RetryAttempts      uint64         `json:"retry_attempts"`
	ThrottleRequests   uint64         `json:"throttle_requests"`
	TimeoutRequests    uint64         `json:"timeout_requests"`
	BytesSent          int64          `json:"bytes_sent"`
	BytesReceived      int64          `json:"bytes_received"`
	QPS                float64        `json:"qps"`
	ErrorRate          float64        `json:"error_rate"`
	Latency            LatencySummary `json:"latency"`
}

type LatencySummary struct {
	MinMillis  float64 `json:"min_ms"`
	MaxMillis  float64 `json:"max_ms"`
	P50Millis  float64 `json:"p50_ms"`
	P90Millis  float64 `json:"p90_ms"`
	P95Millis  float64 `json:"p95_ms"`
	P99Millis  float64 `json:"p99_ms"`
	P999Millis float64 `json:"p999_ms"`
}

type TimeSample struct {
	Second             time.Time `json:"second"`
	TotalRequests      uint64    `json:"total_requests"`
	SuccessfulRequests uint64    `json:"successful_requests"`
	FailedRequests     uint64    `json:"failed_requests"`
	ThrottleRequests   uint64    `json:"throttle_requests"`
	BytesSent          int64     `json:"bytes_sent"`
	BytesReceived      int64     `json:"bytes_received"`
	ConfiguredWorkers  int       `json:"configured_workers"`
	ActiveWorkers      int       `json:"active_workers"`
	TargetQPS          float64   `json:"target_qps,omitempty"`
}

type ErrorSample struct {
	Time      time.Time          `json:"time"`
	Operation workload.Operation `json:"operation"`
	Code      string             `json:"code"`
	Message   string             `json:"message"`
}

type event struct {
	at            time.Time
	op            workload.Operation
	latency       time.Duration
	response      Response
	activeWorkers int
	err           error
}

func New(cfg Config) *Runner {
	if cfg.Client == nil {
		cfg.Client = NewMemoryClient()
	}
	return &Runner{cfg: cfg}
}

func (r *Runner) Run(ctx context.Context) (Result, error) {
	cfg := r.cfg.Workload
	start := time.Now().UTC()
	result := Result{
		RunID:     newRunID(start),
		Tool:      ToolMetadata{Version: "dev", GoVersion: runtime.Version()},
		StartTime: start,
		Workload:  cfg,
		Summary: Summary{
			ByOperation: make(map[workload.Operation]OperationStats),
			ErrorCounts: make(map[string]uint64),
		},
	}

	if err := r.cfg.Client.EnsureIndex(ctx, IndexSpec{
		VectorBucket:    cfg.VectorBucket,
		Index:           cfg.Index,
		Dimension:       cfg.Dimension,
		DistanceMetric:  cfg.DistanceMetric,
		CreateIfMissing: cfg.CreateIndex,
	}); err != nil {
		result.EndTime = time.Now().UTC()
		result.Partial = true
		return result, err
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	if cfg.Duration > 0 {
		var timeoutCancel context.CancelFunc
		runCtx, timeoutCancel = context.WithTimeout(runCtx, cfg.Duration)
		defer timeoutCancel()
	}

	events := make(chan event, cfg.Concurrency*2)
	var issued atomic.Uint64
	var active atomic.Int64
	var wg sync.WaitGroup
	limiter := newRateLimiter(cfg.TargetQPS)

	for workerID := 0; workerID < cfg.Concurrency; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			gen := synth.NewGenerator(synth.Config{Seed: cfg.Seed, Dimension: cfg.Dimension, KeyPrefix: cfg.KeyPrefix}, id)
			for {
				seq := issued.Add(1) - 1
				if cfg.Requests > 0 && seq >= cfg.Requests {
					return
				}
				if err := limiter.Wait(runCtx); err != nil {
					return
				}
				op := cfg.Operation
				if op == workload.OpMixed {
					op = cfg.Mix.Choose(seq)
				}
				activeWorkers := int(active.Add(1))
				ev := r.execute(runCtx, cfg, gen, op, seq)
				ev.activeWorkers = activeWorkers
				active.Add(-1)
				select {
				case events <- ev:
				case <-runCtx.Done():
					return
				}
			}
		}(workerID)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(events)
		close(done)
	}()

	var collected []event
	for ev := range events {
		collected = append(collected, ev)
	}
	<-done
	result.EndTime = time.Now().UTC()
	result.Summary, result.TimeSeries, result.Errors = summarize(collected, cfg, result.StartTime, result.EndTime)
	if err := ctx.Err(); err != nil {
		result.Partial = true
		return result, err
	}
	if cfg.Duration > 0 && errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		return result, nil
	}
	if runCtx.Err() != nil && cfg.Requests == 0 {
		result.Partial = true
		return result, runCtx.Err()
	}
	return result, nil
}

func (r *Runner) execute(parent context.Context, cfg workload.Config, gen synth.Generator, op workload.Operation, seq uint64) event {
	ctx, cancel := context.WithTimeout(parent, cfg.RequestTimeout)
	defer cancel()
	start := time.Now()
	var resp Response
	var err error
	switch op {
	case workload.OpPut:
		batchSize := batchSizeFor(cfg, workload.OpPut)
		vectors := make([]synth.Vector, batchSize)
		base := seq * uint64(batchSize)
		for i := range vectors {
			vectors[i] = gen.Vector(base + uint64(i))
		}
		resp, err = r.cfg.Client.PutVectors(ctx, PutRequest{VectorBucket: cfg.VectorBucket, Index: cfg.Index, Vectors: vectors})
	case workload.OpQuery, workload.OpQueryFilter:
		req := QueryRequest{VectorBucket: cfg.VectorBucket, Index: cfg.Index, Vector: gen.QueryVector(seq).Values, TopK: cfg.TopK}
		if op == workload.OpQueryFilter {
			if len(cfg.Filters) > 0 {
				req.Filter = cfg.Filters[int(seq)%len(cfg.Filters)]
			} else {
				err = OperationError{Kind: ErrorKindClient, Code: "MissingFilter", Err: fmt.Errorf("query-filter requires loaded filters")}
				break
			}
		}
		resp, err = r.cfg.Client.QueryVectors(ctx, req)
	case workload.OpGet:
		resp, err = r.cfg.Client.GetVectors(ctx, GetRequest{VectorBucket: cfg.VectorBucket, Index: cfg.Index, Keys: keysForBatch(gen, seq, batchSizeFor(cfg, workload.OpGet))})
	case workload.OpList:
		resp, err = r.cfg.Client.ListVectors(ctx, ListRequest{VectorBucket: cfg.VectorBucket, Index: cfg.Index, Limit: batchSizeFor(cfg, workload.OpList)})
	case workload.OpDelete:
		resp, err = r.cfg.Client.DeleteVectors(ctx, DeleteRequest{VectorBucket: cfg.VectorBucket, Index: cfg.Index, Keys: keysForBatch(gen, seq, batchSizeFor(cfg, workload.OpDelete))})
	default:
		err = OperationError{Kind: ErrorKindClient, Code: "UnsupportedOperation", Err: fmt.Errorf("unsupported operation %s", op)}
	}
	return event{at: time.Now().UTC(), op: op, latency: time.Since(start), response: resp, err: err}
}

func batchSizeFor(cfg workload.Config, op workload.Operation) int {
	if cfg.OperationBatchSizes != nil && cfg.OperationBatchSizes[op] > 0 {
		return cfg.OperationBatchSizes[op]
	}
	if cfg.BatchSize > 0 {
		return cfg.BatchSize
	}
	return 1
}

func keysForBatch(gen synth.Generator, seq uint64, batchSize int) []string {
	keys := make([]string, batchSize)
	base := seq * uint64(batchSize)
	for i := range keys {
		keys[i] = gen.Vector(base + uint64(i)).Key
	}
	return keys
}

func summarize(events []event, cfg workload.Config, start, end time.Time) (Summary, []TimeSample, []ErrorSample) {
	summary := Summary{ByOperation: make(map[workload.Operation]OperationStats), ErrorCounts: make(map[string]uint64)}
	latencies := make(map[workload.Operation][]time.Duration)
	var allLatencies []time.Duration
	secondBuckets := make(map[int64]*TimeSample)
	var errorsOut []ErrorSample
	for _, ev := range events {
		stats := summary.ByOperation[ev.op]
		stats.TotalRequests++
		summary.TotalRequests++
		stats.RetryAttempts += ev.response.RetryAttempts
		summary.RetryAttempts += ev.response.RetryAttempts
		stats.BytesSent += ev.response.BytesSent
		stats.BytesReceived += ev.response.BytesReceived
		summary.BytesSent += ev.response.BytesSent
		summary.BytesReceived += ev.response.BytesReceived
		if ev.err == nil {
			stats.SuccessfulRequests++
			summary.SuccessfulRequests++
		} else {
			kind, code := operationFromError(ev.err)
			stats.FailedRequests++
			summary.FailedRequests++
			summary.ErrorCounts[code]++
			if kind == ErrorKindThrottle {
				stats.ThrottleRequests++
				summary.ThrottleRequests++
			}
			if kind == ErrorKindTimeout {
				stats.TimeoutRequests++
				summary.TimeoutRequests++
			}
			if len(errorsOut) < 100 {
				errorsOut = append(errorsOut, ErrorSample{Time: ev.at, Operation: ev.op, Code: code, Message: ev.err.Error()})
			}
		}
		latencies[ev.op] = append(latencies[ev.op], ev.latency)
		allLatencies = append(allLatencies, ev.latency)
		bucket := ev.at.Truncate(time.Second).Unix()
		sample := secondBuckets[bucket]
		if sample == nil {
			sample = &TimeSample{Second: ev.at.Truncate(time.Second), ConfiguredWorkers: cfg.Concurrency, TargetQPS: cfg.TargetQPS}
			secondBuckets[bucket] = sample
		}
		if ev.activeWorkers > sample.ActiveWorkers {
			sample.ActiveWorkers = ev.activeWorkers
		}
		sample.TotalRequests++
		sample.BytesSent += ev.response.BytesSent
		sample.BytesReceived += ev.response.BytesReceived
		if ev.err == nil {
			sample.SuccessfulRequests++
		} else {
			sample.FailedRequests++
			kind, _ := operationFromError(ev.err)
			if kind == ErrorKindThrottle {
				sample.ThrottleRequests++
			}
		}
		summary.ByOperation[ev.op] = stats
	}
	duration := end.Sub(start).Seconds()
	if duration <= 0 {
		duration = 1
	}
	summary.QPS = float64(summary.SuccessfulRequests) / duration
	if summary.TotalRequests > 0 {
		summary.ErrorRate = float64(summary.FailedRequests) / float64(summary.TotalRequests)
	}
	summary.Latency = latencySummary(allLatencies)
	for op, stats := range summary.ByOperation {
		stats.QPS = float64(stats.SuccessfulRequests) / duration
		if stats.TotalRequests > 0 {
			stats.ErrorRate = float64(stats.FailedRequests) / float64(stats.TotalRequests)
		}
		stats.Latency = latencySummary(latencies[op])
		summary.ByOperation[op] = stats
	}
	samples := make([]TimeSample, 0, len(secondBuckets))
	for _, sample := range secondBuckets {
		samples = append(samples, *sample)
	}
	sort.Slice(samples, func(i, j int) bool { return samples[i].Second.Before(samples[j].Second) })
	return summary, samples, errorsOut
}

func latencySummary(values []time.Duration) LatencySummary {
	if len(values) == 0 {
		return LatencySummary{}
	}
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	return LatencySummary{
		MinMillis:  millis(values[0]),
		MaxMillis:  millis(values[len(values)-1]),
		P50Millis:  percentile(values, 0.50),
		P90Millis:  percentile(values, 0.90),
		P95Millis:  percentile(values, 0.95),
		P99Millis:  percentile(values, 0.99),
		P999Millis: percentile(values, 0.999),
	}
}

func percentile(values []time.Duration, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	idx := int(float64(len(values)-1) * p)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(values) {
		idx = len(values) - 1
	}
	return millis(values[idx])
}

func millis(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000
}

type rateLimiter struct {
	interval time.Duration
	next     time.Time
	mu       sync.Mutex
}

func newRateLimiter(qps float64) *rateLimiter {
	if qps <= 0 {
		return &rateLimiter{}
	}
	return &rateLimiter{interval: time.Duration(float64(time.Second) / qps)}
}

func (l *rateLimiter) Wait(ctx context.Context) error {
	if l.interval <= 0 {
		return ctx.Err()
	}
	l.mu.Lock()
	now := time.Now()
	if l.next.IsZero() || l.next.Before(now) {
		l.next = now
	}
	waitUntil := l.next
	l.next = l.next.Add(l.interval)
	l.mu.Unlock()
	timer := time.NewTimer(time.Until(waitUntil))
	defer timer.Stop()
	select {
	case <-timer.C:
		return ctx.Err()
	case <-ctx.Done():
		return ctx.Err()
	}
}

func newRunID(start time.Time) string {
	var buf [4]byte
	_, _ = rand.Read(buf[:])
	return start.Format("20060102T150405Z") + "-" + hex.EncodeToString(buf[:])
}
