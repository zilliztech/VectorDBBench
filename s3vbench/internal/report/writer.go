package report

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"s3vbench/internal/runner"
)

type Paths struct {
	SummaryJSON      string
	TimeSeriesCSV    string
	TimeSeriesNDJSON string
	ErrorsJSONL      string
}

func Write(dir string, result runner.Result) (Paths, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Paths{}, err
	}
	paths := Paths{
		SummaryJSON:      filepath.Join(dir, "summary.json"),
		TimeSeriesCSV:    filepath.Join(dir, "timeseries.csv"),
		TimeSeriesNDJSON: filepath.Join(dir, "timeseries.ndjson"),
		ErrorsJSONL:      filepath.Join(dir, "errors.jsonl"),
	}
	if err := writeJSON(paths.SummaryJSON, result); err != nil {
		return Paths{}, err
	}
	if err := writeCSV(paths.TimeSeriesCSV, result.TimeSeries); err != nil {
		return Paths{}, err
	}
	if err := writeNDJSON(paths.TimeSeriesNDJSON, result.TimeSeries); err != nil {
		return Paths{}, err
	}
	if len(result.Errors) > 0 {
		if err := writeNDJSON(paths.ErrorsJSONL, result.Errors); err != nil {
			return Paths{}, err
		}
	} else if err := os.Remove(paths.ErrorsJSONL); err != nil && !os.IsNotExist(err) {
		return Paths{}, err
	}
	return paths, nil
}

func writeJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func writeNDJSON[T any](path string, values []T) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	w := bufio.NewWriter(file)
	for _, value := range values {
		data, err := json.Marshal(value)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintln(w, string(data)); err != nil {
			return err
		}
	}
	return w.Flush()
}

func writeCSV(path string, samples []runner.TimeSample) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	w := csv.NewWriter(file)
	defer w.Flush()
	if err := w.Write([]string{"second", "total_requests", "successful_requests", "failed_requests", "throttle_requests", "bytes_sent", "bytes_received", "configured_workers", "active_workers", "target_qps"}); err != nil {
		return err
	}
	for _, sample := range samples {
		if err := w.Write([]string{
			sample.Second.Format("2006-01-02T15:04:05Z07:00"),
			strconv.FormatUint(sample.TotalRequests, 10),
			strconv.FormatUint(sample.SuccessfulRequests, 10),
			strconv.FormatUint(sample.FailedRequests, 10),
			strconv.FormatUint(sample.ThrottleRequests, 10),
			strconv.FormatInt(sample.BytesSent, 10),
			strconv.FormatInt(sample.BytesReceived, 10),
			strconv.Itoa(sample.ConfiguredWorkers),
			strconv.Itoa(sample.ActiveWorkers),
			strconv.FormatFloat(sample.TargetQPS, 'f', -1, 64),
		}); err != nil {
			return err
		}
	}
	return w.Error()
}
