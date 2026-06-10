package runner

import (
	"context"
	"errors"
	"sync"

	"s3vbench/internal/synth"
	"s3vbench/internal/workload"
)

type Client interface {
	EnsureIndex(context.Context, IndexSpec) error
	PutVectors(context.Context, PutRequest) (Response, error)
	QueryVectors(context.Context, QueryRequest) (Response, error)
	GetVectors(context.Context, GetRequest) (Response, error)
	ListVectors(context.Context, ListRequest) (Response, error)
	DeleteVectors(context.Context, DeleteRequest) (Response, error)
}

type IndexSpec struct {
	VectorBucket    string
	Index           string
	Dimension       int
	DistanceMetric  string
	CreateIfMissing bool
}

type PutRequest struct {
	VectorBucket string
	Index        string
	Vectors      []synth.Vector
}

type QueryRequest struct {
	VectorBucket string
	Index        string
	Vector       []float32
	TopK         int
	Filter       map[string]any
}

type GetRequest struct {
	VectorBucket string
	Index        string
	Keys         []string
}

type ListRequest struct {
	VectorBucket string
	Index        string
	Limit        int
}

type DeleteRequest struct {
	VectorBucket string
	Index        string
	Keys         []string
}

type Response struct {
	BytesSent     int64
	BytesReceived int64
	RetryAttempts uint64
}

type ErrorKind string

const (
	ErrorKindThrottle ErrorKind = "throttle"
	ErrorKindTimeout  ErrorKind = "timeout"
	ErrorKindServer   ErrorKind = "server"
	ErrorKindClient   ErrorKind = "client"
)

type OperationError struct {
	Kind ErrorKind
	Code string
	Err  error
}

func (e OperationError) Error() string {
	if e.Err == nil {
		return e.Code
	}
	return e.Err.Error()
}

func (e OperationError) Unwrap() error {
	return e.Err
}

type MemoryClient struct {
	mu      sync.Mutex
	vectors map[string]synth.Vector
}

func NewMemoryClient() *MemoryClient {
	return &MemoryClient{vectors: make(map[string]synth.Vector)}
}

func (c *MemoryClient) EnsureIndex(context.Context, IndexSpec) error {
	return nil
}

func (c *MemoryClient) PutVectors(_ context.Context, req PutRequest) (Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	var bytes int64
	for _, vector := range req.Vectors {
		c.vectors[vector.Key] = vector
		bytes += int64(len(vector.Values) * 4)
	}
	return Response{BytesSent: bytes, BytesReceived: 128}, nil
}

func (c *MemoryClient) QueryVectors(_ context.Context, req QueryRequest) (Response, error) {
	if len(req.Vector) == 0 {
		return Response{}, OperationError{Kind: ErrorKindClient, Code: "EmptyQuery", Err: errors.New("query vector is empty")}
	}
	return Response{BytesSent: int64(len(req.Vector) * 4), BytesReceived: int64(req.TopK * 64)}, nil
}

func (c *MemoryClient) GetVectors(_ context.Context, req GetRequest) (Response, error) {
	return Response{BytesSent: int64(len(req.Keys) * 16), BytesReceived: int64(len(req.Keys) * 128)}, nil
}

func (c *MemoryClient) ListVectors(_ context.Context, req ListRequest) (Response, error) {
	if req.Limit <= 0 {
		req.Limit = 100
	}
	return Response{BytesSent: 64, BytesReceived: int64(req.Limit * 64)}, nil
}

func (c *MemoryClient) DeleteVectors(_ context.Context, req DeleteRequest) (Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, key := range req.Keys {
		delete(c.vectors, key)
	}
	return Response{BytesSent: int64(len(req.Keys) * 16), BytesReceived: 64}, nil
}

func operationFromError(err error) (ErrorKind, string) {
	if err == nil {
		return "", ""
	}
	var opErr OperationError
	if errors.As(err, &opErr) {
		return opErr.Kind, opErr.Code
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return ErrorKindTimeout, "Timeout"
	}
	if errors.Is(err, context.Canceled) {
		return ErrorKindClient, "Canceled"
	}
	return ErrorKindClient, "ClientError"
}

var _ = workload.OpPut
