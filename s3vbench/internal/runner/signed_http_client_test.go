package runner

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"s3vbench/internal/synth"
)

func TestSignedHTTPClientSendsSignedPutVectorsRequest(t *testing.T) {
	var path string
	var auth string
	var body map[string]any
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		path = r.URL.Path
		auth = r.Header.Get("authorization")
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}")), Header: make(http.Header)}, nil
	})

	client, err := NewSignedHTTPClient(SignedHTTPConfig{
		Endpoint:   "https://s3vectors.example.test",
		Region:     "us-east-1",
		AccessKey:  "ak",
		SecretKey:  "sk",
		HTTPClient: &http.Client{Transport: transport},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.PutVectors(context.Background(), PutRequest{
		VectorBucket: "bucket",
		Index:        "idx",
		Vectors: []synth.Vector{{
			Key:    "vec-1",
			Values: []float32{0.1, 0.2},
			Metadata: map[string]string{
				"tenant": "tenant-01",
			},
		}},
	})
	if err != nil {
		t.Fatalf("PutVectors returned error: %v", err)
	}
	if path != "/PutVectors" {
		t.Fatalf("path = %q, want /PutVectors", path)
	}
	if auth == "" {
		t.Fatalf("expected authorization header")
	}
	if body["vectorBucketName"] != "bucket" || body["indexName"] != "idx" {
		t.Fatalf("unexpected body: %#v", body)
	}
}

func TestSignedHTTPClientMapsThrottleError(t *testing.T) {
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusTooManyRequests, Body: io.NopCloser(strings.NewReader(`{"code":"ThrottlingException"}`)), Header: make(http.Header)}, nil
	})

	client, err := NewSignedHTTPClient(SignedHTTPConfig{
		Endpoint:   "https://s3vectors.example.test",
		Region:     "us-east-1",
		AccessKey:  "ak",
		SecretKey:  "sk",
		HTTPClient: &http.Client{Transport: transport},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.QueryVectors(context.Background(), QueryRequest{VectorBucket: "bucket", Index: "idx", Vector: []float32{1}, TopK: 1})
	if err == nil {
		t.Fatalf("expected error")
	}
	kind, code := operationFromError(err)
	if kind != ErrorKindThrottle || code != "ThrottlingException" {
		t.Fatalf("kind/code = %s/%s", kind, code)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}
