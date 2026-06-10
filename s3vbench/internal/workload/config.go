package workload

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

type Operation string

const (
	OpPut         Operation = "put"
	OpQuery       Operation = "query"
	OpQueryFilter Operation = "query-filter"
	OpGet         Operation = "get"
	OpList        Operation = "list"
	OpDelete      Operation = "delete"
	OpMixed       Operation = "mixed"
)

var singleOperations = map[Operation]struct{}{
	OpPut:         {},
	OpQuery:       {},
	OpQueryFilter: {},
	OpGet:         {},
	OpList:        {},
	OpDelete:      {},
}

type Config struct {
	Region              string            `json:"region"`
	VectorBucket        string            `json:"vector_bucket"`
	Index               string            `json:"index"`
	Operation           Operation         `json:"operation"`
	Dimension           int               `json:"dimension"`
	DistanceMetric      string            `json:"distance_metric"`
	Requests            uint64            `json:"requests"`
	Duration            time.Duration     `json:"duration"`
	Concurrency         int               `json:"concurrency"`
	TargetQPS           float64           `json:"target_qps,omitempty"`
	BatchSize           int               `json:"batch_size"`
	OperationBatchSizes map[Operation]int `json:"operation_batch_sizes,omitempty"`
	TopK                int               `json:"top_k"`
	Seed                int64             `json:"seed"`
	KeyPrefix           string            `json:"key_prefix"`
	OutputDir           string            `json:"output_dir"`
	FixedOutputDir      bool              `json:"fixed_output_dir"`
	RequestTimeout      time.Duration     `json:"request_timeout"`
	RetryMode           string            `json:"retry_mode"`
	ClientMode          string            `json:"client_mode"`
	Endpoint            string            `json:"endpoint,omitempty"`
	AccessKey           string            `json:"-"`
	SecretKey           string            `json:"-"`
	SessionToken        string            `json:"-"`
	FilterFile          string            `json:"filter_file,omitempty"`
	Filters             []map[string]any  `json:"-"`
	Mix                 Mix               `json:"mix,omitempty"`
	CreateIndex         bool              `json:"create_index"`
}

func (c Config) Resolve() (Config, error) {
	if c.Operation == "" {
		return c, errors.New("operation is required")
	}
	if c.Operation != OpMixed {
		if _, ok := singleOperations[c.Operation]; !ok {
			return c, fmt.Errorf("unsupported operation %q", c.Operation)
		}
	}
	if c.VectorBucket == "" {
		return c, errors.New("vector bucket is required")
	}
	if c.Index == "" {
		return c, errors.New("index is required")
	}
	if c.Dimension <= 0 {
		return c, errors.New("dimension must be positive")
	}
	if c.Requests == 0 && c.Duration <= 0 {
		c.Requests = 1
	}
	if c.Concurrency <= 0 {
		c.Concurrency = 1
	}
	if c.BatchSize <= 0 {
		c.BatchSize = 1
	}
	if c.OperationBatchSizes == nil {
		c.OperationBatchSizes = make(map[Operation]int)
	}
	for _, op := range []Operation{OpPut, OpGet, OpList, OpDelete} {
		if c.OperationBatchSizes[op] <= 0 {
			c.OperationBatchSizes[op] = c.BatchSize
		}
	}
	if c.TopK <= 0 {
		c.TopK = 10
	}
	if c.OutputDir == "" {
		c.OutputDir = "result"
	}
	if c.RequestTimeout <= 0 {
		c.RequestTimeout = 30 * time.Second
	}
	if c.DistanceMetric == "" {
		c.DistanceMetric = "cosine"
	}
	if c.RetryMode == "" {
		c.RetryMode = "sdk-default"
	}
	if c.ClientMode == "" {
		c.ClientMode = "mock"
	}
	if c.KeyPrefix == "" {
		c.KeyPrefix = "vec"
	}
	if c.Operation == OpMixed && c.Mix.TotalWeight() == 0 {
		return c, errors.New("mixed workload requires --mix")
	}
	if (c.Operation == OpQueryFilter || c.Mix.Weight(OpQueryFilter) > 0) && c.FilterFile == "" {
		return c, errors.New("query-filter requires --filter-file")
	}
	if c.TargetQPS < 0 {
		return c, errors.New("target qps must not be negative")
	}
	return c, nil
}

type Mix struct {
	weights  map[Operation]int
	order    []Operation
	schedule []Operation
}

func ParseMix(raw string) (Mix, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Mix{}, nil
	}
	weights := make(map[Operation]int)
	var order []Operation
	for _, part := range strings.Split(raw, ",") {
		pair := strings.SplitN(strings.TrimSpace(part), "=", 2)
		if len(pair) != 2 {
			return Mix{}, fmt.Errorf("invalid mix component %q", part)
		}
		op := Operation(strings.TrimSpace(pair[0]))
		if _, ok := singleOperations[op]; !ok {
			return Mix{}, fmt.Errorf("unsupported mixed operation %q", op)
		}
		weight, err := strconv.Atoi(strings.TrimSpace(pair[1]))
		if err != nil || weight <= 0 {
			return Mix{}, fmt.Errorf("invalid weight for %s", op)
		}
		if _, exists := weights[op]; !exists {
			order = append(order, op)
		}
		weights[op] += weight
	}
	return Mix{weights: weights, order: order, schedule: buildSmoothSchedule(weights, order)}, nil
}

func (m Mix) Weight(op Operation) int {
	return m.weights[op]
}

func (m Mix) TotalWeight() int {
	total := 0
	for _, weight := range m.weights {
		total += weight
	}
	return total
}

func (m Mix) Operations() []Operation {
	out := append([]Operation(nil), m.order...)
	return out
}

func (m Mix) Choose(sequence uint64) Operation {
	if len(m.schedule) == 0 {
		return OpQuery
	}
	return m.schedule[int(sequence%uint64(len(m.schedule)))]
}

func (m Mix) MarshalJSON() ([]byte, error) {
	if len(m.weights) == 0 {
		return []byte("{}"), nil
	}
	parts := make([]string, 0, len(m.weights))
	ops := m.Operations()
	sort.SliceStable(ops, func(i, j int) bool { return ops[i] < ops[j] })
	for _, op := range ops {
		parts = append(parts, fmt.Sprintf("%q:%d", op, m.weights[op]))
	}
	return []byte("{" + strings.Join(parts, ",") + "}"), nil
}

func buildSmoothSchedule(weights map[Operation]int, order []Operation) []Operation {
	total := 0
	for _, weight := range weights {
		total += weight
	}
	if total == 0 {
		return nil
	}
	current := make(map[Operation]int, len(weights))
	schedule := make([]Operation, 0, total)
	for i := 0; i < total; i++ {
		var selected Operation
		selectedSet := false
		for _, op := range order {
			current[op] += weights[op]
			if !selectedSet || current[op] > current[selected] {
				selected = op
				selectedSet = true
			}
		}
		current[selected] -= total
		schedule = append(schedule, selected)
	}
	return schedule
}
