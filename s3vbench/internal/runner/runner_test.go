package runner

import (
	"context"
	"sync"
	"testing"
	"time"

	"s3vbench/internal/synth"
	"s3vbench/internal/workload"
)

func TestRunnerExecutesCountBasedMixedWorkload(t *testing.T) {
	cfg, err := workload.Config{
		Operation:      workload.OpMixed,
		VectorBucket:   "bucket",
		Index:          "index",
		Dimension:      4,
		Requests:       10,
		Concurrency:    2,
		Mix:            mustParseMix(t, "query=7,put=2,get=1"),
		RequestTimeout: time.Second,
	}.Resolve()
	if err != nil {
		t.Fatal(err)
	}

	result, err := New(Config{Workload: cfg, Client: NewMemoryClient()}).Run(context.Background())
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	if result.Summary.TotalRequests != 10 {
		t.Fatalf("total requests = %d, want 10", result.Summary.TotalRequests)
	}
	assertOpCount(t, result, workload.OpQuery, 7)
	assertOpCount(t, result, workload.OpPut, 2)
	assertOpCount(t, result, workload.OpGet, 1)
	if len(result.TimeSeries) == 0 {
		t.Fatalf("expected time series samples")
	}
	if result.TimeSeries[0].ActiveWorkers == 0 {
		t.Fatalf("active workers should be sampled during execution")
	}
}

func TestRunnerUsesBatchSizeForVectorKeyOperations(t *testing.T) {
	for _, tc := range []struct {
		name string
		op   workload.Operation
	}{
		{name: "put", op: workload.OpPut},
		{name: "get", op: workload.OpGet},
		{name: "delete", op: workload.OpDelete},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &recordingClient{MemoryClient: NewMemoryClient()}
			cfg, err := workload.Config{
				Operation:           tc.op,
				VectorBucket:        "bucket",
				Index:               "index",
				Dimension:           4,
				Requests:            1,
				Concurrency:         1,
				BatchSize:           2,
				OperationBatchSizes: map[workload.Operation]int{tc.op: 3},
				RequestTimeout:      time.Second,
			}.Resolve()
			if err != nil {
				t.Fatal(err)
			}
			if _, err := New(Config{Workload: cfg, Client: client}).Run(context.Background()); err != nil {
				t.Fatalf("Run returned error: %v", err)
			}
			if got := client.batchSize(tc.op); got != 3 {
				t.Fatalf("%s batch size = %d, want 3", tc.op, got)
			}
		})
	}
}

func mustParseMix(t *testing.T, raw string) workload.Mix {
	t.Helper()
	mix, err := workload.ParseMix(raw)
	if err != nil {
		t.Fatal(err)
	}
	return mix
}

type recordingClient struct {
	*MemoryClient
	mu      sync.Mutex
	batches map[workload.Operation]int
}

func (c *recordingClient) PutVectors(ctx context.Context, req PutRequest) (Response, error) {
	c.record(workload.OpPut, len(req.Vectors))
	return c.MemoryClient.PutVectors(ctx, req)
}

func (c *recordingClient) GetVectors(ctx context.Context, req GetRequest) (Response, error) {
	c.record(workload.OpGet, len(req.Keys))
	return c.MemoryClient.GetVectors(ctx, req)
}

func (c *recordingClient) DeleteVectors(ctx context.Context, req DeleteRequest) (Response, error) {
	c.record(workload.OpDelete, len(req.Keys))
	return c.MemoryClient.DeleteVectors(ctx, req)
}

func (c *recordingClient) record(op workload.Operation, size int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.batches == nil {
		c.batches = make(map[workload.Operation]int)
	}
	c.batches[op] = size
}

func (c *recordingClient) batchSize(op workload.Operation) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.batches[op]
}

var _ = synth.Vector{}

func TestRunnerReturnsPartialMetricsOnContextCancel(t *testing.T) {
	cfg, err := workload.Config{
		Operation:      workload.OpQuery,
		VectorBucket:   "bucket",
		Index:          "index",
		Dimension:      4,
		Requests:       1000,
		Concurrency:    1,
		RequestTimeout: time.Second,
	}.Resolve()
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := New(Config{Workload: cfg, Client: NewMemoryClient()}).Run(ctx)
	if err == nil {
		t.Fatalf("expected cancellation error")
	}
	if !result.Partial {
		t.Fatalf("expected partial result")
	}
}

func assertOpCount(t *testing.T, result Result, op workload.Operation, want uint64) {
	t.Helper()
	stats := result.Summary.ByOperation[op]
	if stats.TotalRequests != want {
		t.Fatalf("%s total requests = %d, want %d", op, stats.TotalRequests, want)
	}
	if stats.SuccessfulRequests != want {
		t.Fatalf("%s successes = %d, want %d", op, stats.SuccessfulRequests, want)
	}
}
