package workload

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFiltersJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "filters.jsonl")
	if err := os.WriteFile(path, []byte("{\"tenant\":\"tenant-01\"}\n\n{\"price\":{\"$lt\":10}}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	filters, err := LoadFiltersJSONL(path)
	if err != nil {
		t.Fatalf("LoadFiltersJSONL returned error: %v", err)
	}
	if len(filters) != 2 {
		t.Fatalf("filters = %d, want 2", len(filters))
	}
	if filters[0]["tenant"] != "tenant-01" {
		t.Fatalf("first filter = %#v", filters[0])
	}
}
