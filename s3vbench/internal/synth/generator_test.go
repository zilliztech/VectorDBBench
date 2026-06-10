package synth

import "testing"

func TestGeneratorIsDeterministicBySeedAndShard(t *testing.T) {
	a := NewGenerator(Config{Seed: 42, Dimension: 4, KeyPrefix: "vec"}, 3)
	b := NewGenerator(Config{Seed: 42, Dimension: 4, KeyPrefix: "vec"}, 3)

	va := a.Vector(7)
	vb := b.Vector(7)

	if va.Key != "vec-000000000007" {
		t.Fatalf("key = %q", va.Key)
	}
	for i := range va.Values {
		if va.Values[i] != vb.Values[i] {
			t.Fatalf("value[%d] differs: %f != %f", i, va.Values[i], vb.Values[i])
		}
	}
	if va.Metadata["tenant"] == "" || va.Metadata["category"] == "" {
		t.Fatalf("expected generated low-cardinality metadata, got %#v", va.Metadata)
	}
}

func TestQueryVectorDoesNotAllocateMetadata(t *testing.T) {
	g := NewGenerator(Config{Seed: 9, Dimension: 3}, 0)
	q := g.QueryVector(2)
	if len(q.Values) != 3 {
		t.Fatalf("dimension = %d, want 3", len(q.Values))
	}
	if q.Key != "" {
		t.Fatalf("query key = %q, want empty", q.Key)
	}
	if q.Metadata != nil {
		t.Fatalf("query metadata = %#v, want nil", q.Metadata)
	}
}
