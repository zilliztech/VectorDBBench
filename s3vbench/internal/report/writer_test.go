package report

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"s3vbench/internal/runner"
	"s3vbench/internal/workload"
)

func TestWriterCreatesStableArtifacts(t *testing.T) {
	dir := t.TempDir()
	result := runner.Result{
		RunID:     "run-test",
		StartTime: time.Unix(10, 0).UTC(),
		EndTime:   time.Unix(11, 0).UTC(),
		Summary: runner.Summary{
			TotalRequests:      1,
			SuccessfulRequests: 1,
			ByOperation: map[workload.Operation]runner.OperationStats{
				workload.OpQuery: {TotalRequests: 1, SuccessfulRequests: 1},
			},
		},
		TimeSeries: []runner.TimeSample{{Second: time.Unix(10, 0).UTC(), TotalRequests: 1, SuccessfulRequests: 1}},
	}

	paths, err := Write(dir, result)
	if err != nil {
		t.Fatalf("Write returned error: %v", err)
	}

	for _, path := range []string{paths.SummaryJSON, paths.TimeSeriesCSV, paths.TimeSeriesNDJSON} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected artifact %s: %v", path, err)
		}
	}

	data, err := os.ReadFile(filepath.Join(dir, "summary.json"))
	if err != nil {
		t.Fatal(err)
	}
	var decoded runner.Result
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("summary is not valid json: %v", err)
	}
	if decoded.Summary.TotalRequests != 1 {
		t.Fatalf("summary total = %d", decoded.Summary.TotalRequests)
	}
}

func TestWriterRemovesStaleErrorsWhenRunHasNoErrors(t *testing.T) {
	dir := t.TempDir()
	stale := filepath.Join(dir, "errors.jsonl")
	if err := os.WriteFile(stale, []byte("stale\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Write(dir, runner.Result{
		RunID:     "run-clean",
		StartTime: time.Unix(10, 0).UTC(),
		EndTime:   time.Unix(11, 0).UTC(),
		Summary:   runner.Summary{ByOperation: map[workload.Operation]runner.OperationStats{}},
	})
	if err != nil {
		t.Fatalf("Write returned error: %v", err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("expected stale errors.jsonl to be removed, got %v", err)
	}
}
