package synth

import (
	"fmt"
	"math"
	"math/rand"
)

type Config struct {
	Seed      int64
	Dimension int
	KeyPrefix string
}

type Vector struct {
	Key      string            `json:"key,omitempty"`
	Values   []float32         `json:"values"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type Generator struct {
	cfg   Config
	shard int
}

func NewGenerator(cfg Config, shard int) Generator {
	if cfg.KeyPrefix == "" {
		cfg.KeyPrefix = "vec"
	}
	return Generator{cfg: cfg, shard: shard}
}

func (g Generator) Vector(id uint64) Vector {
	return Vector{
		Key:      fmt.Sprintf("%s-%012d", g.cfg.KeyPrefix, id),
		Values:   g.values(id),
		Metadata: g.metadata(id),
	}
}

func (g Generator) QueryVector(id uint64) Vector {
	return Vector{Values: g.values(id + 1_000_000_000)}
}

func (g Generator) values(id uint64) []float32 {
	r := rand.New(rand.NewSource(g.cfg.Seed + int64(g.shard)*1_000_003 + int64(id)*97))
	values := make([]float32, g.cfg.Dimension)
	var norm float64
	for i := range values {
		v := r.Float64()*2 - 1
		values[i] = float32(v)
		norm += v * v
	}
	if norm == 0 {
		return values
	}
	scale := float32(1 / math.Sqrt(norm))
	for i := range values {
		values[i] *= scale
	}
	return values
}

func (g Generator) metadata(id uint64) map[string]string {
	return map[string]string{
		"tenant":   fmt.Sprintf("tenant-%02d", id%16),
		"category": fmt.Sprintf("cat-%02d", id%8),
		"lang":     []string{"en", "zh", "es", "fr"}[id%4],
	}
}
