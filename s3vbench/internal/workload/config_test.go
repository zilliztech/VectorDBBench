package workload

import (
	"reflect"
	"testing"
	"time"
)

func TestParseMixNormalizesConfiguredOperations(t *testing.T) {
	mix, err := ParseMix("query=90,put=5,get=3,delete=1,list=1")
	if err != nil {
		t.Fatalf("ParseMix returned error: %v", err)
	}

	if got, want := mix.TotalWeight(), 100; got != want {
		t.Fatalf("total weight = %d, want %d", got, want)
	}
	if got, want := mix.Operations(), []Operation{OpQuery, OpPut, OpGet, OpDelete, OpList}; !reflect.DeepEqual(got, want) {
		t.Fatalf("operations = %v, want %v", got, want)
	}
}

func TestMixChooseInterleavesWeightedOperations(t *testing.T) {
	mix, err := ParseMix("query=90,put=5,get=3,delete=1,list=1")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[Operation]bool{}
	for i := uint64(0); i < 20; i++ {
		seen[mix.Choose(i)] = true
	}
	if !seen[OpPut] {
		t.Fatalf("first 20 choices did not include put: %#v", seen)
	}
}

func TestResolveAppliesDefaultsAndValidates(t *testing.T) {
	cfg := Config{
		Operation:    OpMixed,
		VectorBucket: "bucket",
		Index:        "index",
		Dimension:    8,
		Requests:     10,
		Concurrency:  0,
		Mix:          MustParseMixForTest("query=2,put=1"),
	}

	resolved, err := cfg.Resolve()
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}

	if resolved.Concurrency != 1 {
		t.Fatalf("default concurrency = %d, want 1", resolved.Concurrency)
	}
	if resolved.OutputDir != "result" {
		t.Fatalf("default output dir = %q, want result", resolved.OutputDir)
	}
	if resolved.RequestTimeout != 30*time.Second {
		t.Fatalf("default timeout = %s, want 30s", resolved.RequestTimeout)
	}
	if resolved.OperationBatchSizes[OpPut] != 1 {
		t.Fatalf("default put batch size = %d, want 1", resolved.OperationBatchSizes[OpPut])
	}
}

func TestResolveRequiresFilterFileWhenMixedContainsQueryFilter(t *testing.T) {
	_, err := Config{
		Operation:    OpMixed,
		VectorBucket: "bucket",
		Index:        "index",
		Dimension:    8,
		Requests:     10,
		Mix:          MustParseMixForTest("query=1,query-filter=1"),
	}.Resolve()
	if err == nil {
		t.Fatalf("expected query-filter in mixed workload to require filter file")
	}
}

func MustParseMixForTest(raw string) Mix {
	mix, err := ParseMix(raw)
	if err != nil {
		panic(err)
	}
	return mix
}
