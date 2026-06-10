package workload

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

func LoadFiltersJSONL(path string) ([]map[string]any, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var filters []map[string]any
	scanner := bufio.NewScanner(file)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var filter map[string]any
		if err := json.Unmarshal([]byte(line), &filter); err != nil {
			return nil, fmt.Errorf("invalid filter JSONL line %d: %w", lineNo, err)
		}
		if len(filter) == 0 {
			return nil, fmt.Errorf("invalid filter JSONL line %d: empty filter", lineNo)
		}
		filters = append(filters, filter)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(filters) == 0 {
		return nil, fmt.Errorf("filter file %s did not contain filters", path)
	}
	return filters, nil
}
