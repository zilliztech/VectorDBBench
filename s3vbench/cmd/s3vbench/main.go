package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"s3vbench/internal/report"
	"s3vbench/internal/runner"
	"s3vbench/internal/workload"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "s3vbench:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		usage()
		return errors.New("command is required")
	}
	cmd := args[0]
	switch workload.Operation(cmd) {
	case workload.OpPut, workload.OpQuery, workload.OpQueryFilter, workload.OpGet, workload.OpList, workload.OpDelete, workload.OpMixed:
		return runBenchmark(workload.Operation(cmd), args[1:])
	case "prepare", "index", "report":
		fmt.Printf("%s helper is reserved for Phase 1 workflow integration; use operation commands for benchmark execution.\n", cmd)
		return nil
	case "help", "-h", "--help":
		usage()
		return nil
	default:
		usage()
		return fmt.Errorf("unknown command %q", cmd)
	}
}

func runBenchmark(op workload.Operation, args []string) error {
	fs := flag.NewFlagSet(string(op), flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	cfg := workload.Config{Operation: op}
	var durationRaw string
	var timeoutRaw string
	var mixRaw string
	var putBatchSize, getBatchSize, listBatchSize, deleteBatchSize int
	fs.StringVar(&cfg.Region, "region", "us-east-1", "AWS region")
	fs.StringVar(&cfg.VectorBucket, "vector-bucket", "", "S3 vector bucket")
	fs.StringVar(&cfg.Index, "index", "", "S3 vector index")
	fs.IntVar(&cfg.Dimension, "dimension", 0, "vector dimension")
	fs.StringVar(&cfg.DistanceMetric, "distance-metric", "cosine", "index distance metric")
	fs.Uint64Var(&cfg.Requests, "requests", 0, "number of logical requests")
	fs.StringVar(&durationRaw, "duration", "", "duration limit such as 10s or 5m")
	fs.IntVar(&cfg.Concurrency, "concurrency", 1, "worker count")
	fs.Float64Var(&cfg.TargetQPS, "target-qps", 0, "optional global target QPS")
	fs.IntVar(&cfg.BatchSize, "batch-size", 1, "default operation batch size")
	fs.IntVar(&putBatchSize, "put-batch-size", 0, "put batch size override")
	fs.IntVar(&getBatchSize, "get-batch-size", 0, "get batch size override")
	fs.IntVar(&listBatchSize, "list-batch-size", 0, "list batch size override")
	fs.IntVar(&deleteBatchSize, "delete-batch-size", 0, "delete batch size override")
	fs.IntVar(&cfg.TopK, "top-k", 10, "query top-k")
	fs.Int64Var(&cfg.Seed, "seed", 1, "synthetic data seed")
	fs.StringVar(&cfg.OutputDir, "output", "result", "output directory")
	fs.BoolVar(&cfg.FixedOutputDir, "fixed-output", false, "write artifacts directly to output directory")
	fs.StringVar(&timeoutRaw, "request-timeout", "30s", "per-request timeout")
	fs.StringVar(&cfg.RetryMode, "retry-mode", "sdk-default", "retry policy label")
	fs.StringVar(&cfg.ClientMode, "client", "mock", "client mode: mock or aws")
	fs.StringVar(&cfg.Endpoint, "endpoint", getenv("S3VB_ENDPOINT", ""), "S3 Vectors endpoint URL")
	fs.StringVar(&cfg.AccessKey, "access-key", getenv("AWS_ACCESS_KEY_ID", ""), "AWS access key id")
	fs.StringVar(&cfg.SecretKey, "secret-key", getenv("AWS_SECRET_ACCESS_KEY", ""), "AWS secret access key")
	fs.StringVar(&cfg.SessionToken, "session-token", getenv("AWS_SESSION_TOKEN", ""), "AWS session token")
	fs.StringVar(&cfg.FilterFile, "filter-file", "", "JSONL filter file for query-filter")
	fs.StringVar(&mixRaw, "mix", "", "mixed operation ratios, for example query=90,put=5")
	fs.BoolVar(&cfg.CreateIndex, "create-index", true, "create target index if missing")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if durationRaw != "" {
		duration, err := time.ParseDuration(durationRaw)
		if err != nil {
			return err
		}
		cfg.Duration = duration
	}
	timeout, err := time.ParseDuration(timeoutRaw)
	if err != nil {
		return err
	}
	cfg.RequestTimeout = timeout
	if mixRaw != "" {
		mix, err := workload.ParseMix(mixRaw)
		if err != nil {
			return err
		}
		cfg.Mix = mix
	}
	cfg.OperationBatchSizes = map[workload.Operation]int{
		workload.OpPut:    putBatchSize,
		workload.OpGet:    getBatchSize,
		workload.OpList:   listBatchSize,
		workload.OpDelete: deleteBatchSize,
	}
	cfg, err = cfg.Resolve()
	if err != nil {
		return err
	}
	if cfg.FilterFile != "" {
		filters, err := workload.LoadFiltersJSONL(cfg.FilterFile)
		if err != nil {
			return err
		}
		cfg.Filters = filters
	}
	client := runner.Client(runner.NewMemoryClient())
	if cfg.ClientMode == "aws" {
		if cfg.Endpoint == "" || cfg.AccessKey == "" || cfg.SecretKey == "" {
			return fmt.Errorf("--client aws requires --endpoint, --access-key, and --secret-key")
		}
		awsClient, err := runner.NewSignedHTTPClient(runner.SignedHTTPConfig{
			Endpoint:     cfg.Endpoint,
			Region:       cfg.Region,
			AccessKey:    cfg.AccessKey,
			SecretKey:    cfg.SecretKey,
			SessionToken: cfg.SessionToken,
		})
		if err != nil {
			return err
		}
		client = awsClient
	} else if cfg.ClientMode != "mock" {
		return fmt.Errorf("unsupported client mode %q", cfg.ClientMode)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	result, runErr := runner.New(runner.Config{Workload: cfg, Client: client}).Run(ctx)
	outputDir := cfg.OutputDir
	if !cfg.FixedOutputDir {
		outputDir = filepath.Join(cfg.OutputDir, result.RunID)
	}
	paths, writeErr := report.Write(outputDir, result)
	printSummary(result, paths)
	if writeErr != nil {
		return writeErr
	}
	return runErr
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: s3vbench <put|query|query-filter|get|list|delete|mixed> [flags]")
}

func printSummary(result runner.Result, paths report.Paths) {
	fmt.Printf("Run ID: %s\n", result.RunID)
	fmt.Printf("Requests: total=%d success=%d failed=%d qps=%.2f\n", result.Summary.TotalRequests, result.Summary.SuccessfulRequests, result.Summary.FailedRequests, result.Summary.QPS)
	fmt.Printf("Latency: p50=%.3fms p95=%.3fms p99=%.3fms\n", result.Summary.Latency.P50Millis, result.Summary.Latency.P95Millis, result.Summary.Latency.P99Millis)
	fmt.Printf("Artifacts: %s\n", paths.SummaryJSON)
}
