package runner

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type SignedHTTPConfig struct {
	Endpoint     string
	Region       string
	AccessKey    string
	SecretKey    string
	SessionToken string
	HTTPClient   *http.Client
}

type SignedHTTPClient struct {
	cfg      SignedHTTPConfig
	endpoint *url.URL
	client   *http.Client
}

func NewSignedHTTPClient(cfg SignedHTTPConfig) (*SignedHTTPClient, error) {
	if cfg.Region == "" {
		cfg.Region = "us-east-1"
	}
	endpoint, err := url.Parse(cfg.Endpoint)
	if err != nil {
		return nil, err
	}
	if endpoint.Scheme == "" || endpoint.Host == "" {
		return nil, fmt.Errorf("endpoint must include scheme and host")
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 30 * time.Second}
	}
	return &SignedHTTPClient{cfg: cfg, endpoint: endpoint, client: httpClient}, nil
}

func (c *SignedHTTPClient) EnsureIndex(ctx context.Context, spec IndexSpec) error {
	body := map[string]any{
		"vectorBucketName": spec.VectorBucket,
		"indexName":        spec.Index,
	}
	if err := c.call(ctx, "GetIndex", body, nil); err == nil {
		return nil
	} else if !isNotFound(err) || !spec.CreateIfMissing {
		return err
	}
	createBody := map[string]any{
		"vectorBucketName": spec.VectorBucket,
		"indexName":        spec.Index,
		"dimension":        spec.Dimension,
		"distanceMetric":   strings.ToLower(spec.DistanceMetric),
		"dataType":         "float32",
	}
	return c.call(ctx, "CreateIndex", createBody, nil)
}

func (c *SignedHTTPClient) PutVectors(ctx context.Context, req PutRequest) (Response, error) {
	vectors := make([]map[string]any, 0, len(req.Vectors))
	for _, vector := range req.Vectors {
		vectors = append(vectors, map[string]any{
			"key":      vector.Key,
			"data":     map[string]any{"float32": vector.Values},
			"metadata": vector.Metadata,
		})
	}
	return c.callResponse(ctx, "PutVectors", map[string]any{
		"vectorBucketName": req.VectorBucket,
		"indexName":        req.Index,
		"vectors":          vectors,
	})
}

func (c *SignedHTTPClient) QueryVectors(ctx context.Context, req QueryRequest) (Response, error) {
	body := map[string]any{
		"vectorBucketName": req.VectorBucket,
		"indexName":        req.Index,
		"queryVector":      map[string]any{"float32": req.Vector},
		"topK":             req.TopK,
	}
	if req.Filter != nil {
		body["filter"] = req.Filter
	}
	return c.callResponse(ctx, "QueryVectors", body)
}

func (c *SignedHTTPClient) GetVectors(ctx context.Context, req GetRequest) (Response, error) {
	return c.callResponse(ctx, "GetVectors", map[string]any{
		"vectorBucketName": req.VectorBucket,
		"indexName":        req.Index,
		"keys":             req.Keys,
		"returnData":       false,
		"returnMetadata":   true,
	})
}

func (c *SignedHTTPClient) ListVectors(ctx context.Context, req ListRequest) (Response, error) {
	return c.callResponse(ctx, "ListVectors", map[string]any{
		"vectorBucketName": req.VectorBucket,
		"indexName":        req.Index,
		"maxResults":       req.Limit,
	})
}

func (c *SignedHTTPClient) DeleteVectors(ctx context.Context, req DeleteRequest) (Response, error) {
	return c.callResponse(ctx, "DeleteVectors", map[string]any{
		"vectorBucketName": req.VectorBucket,
		"indexName":        req.Index,
		"keys":             req.Keys,
	})
}

func (c *SignedHTTPClient) callResponse(ctx context.Context, operation string, body any) (Response, error) {
	var payload bytes.Buffer
	if err := c.call(ctx, operation, body, &payload); err != nil {
		return Response{BytesSent: int64(payload.Len())}, err
	}
	return Response{BytesSent: int64(payload.Len())}, nil
}

func (c *SignedHTTPClient) call(ctx context.Context, operation string, body any, capturedPayload *bytes.Buffer) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	if capturedPayload != nil {
		capturedPayload.Write(payload)
	}
	u := *c.endpoint
	u.Path = strings.TrimRight(c.endpoint.Path, "/") + "/" + operation
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(payload))
	if err != nil {
		return err
	}
	now := time.Now().UTC()
	payloadHash := sha256Hex(payload)
	req.Header.Set("content-type", "application/json")
	req.Header.Set("x-amz-date", now.Format("20060102T150405Z"))
	req.Header.Set("x-amz-content-sha256", payloadHash)
	if c.cfg.SessionToken != "" {
		req.Header.Set("x-amz-security-token", c.cfg.SessionToken)
	}
	c.sign(req, payloadHash, now)
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	return statusError(resp.StatusCode, respBody)
}

func (c *SignedHTTPClient) sign(req *http.Request, payloadHash string, now time.Time) {
	date := now.Format("20060102")
	scope := date + "/" + c.cfg.Region + "/s3vectors/aws4_request"
	canonicalURI := req.URL.EscapedPath()
	if canonicalURI == "" {
		canonicalURI = "/"
	}
	signedHeaders := "content-type;host;x-amz-content-sha256;x-amz-date"
	canonicalHeaders := "content-type:" + req.Header.Get("content-type") + "\n" +
		"host:" + req.URL.Host + "\n" +
		"x-amz-content-sha256:" + payloadHash + "\n" +
		"x-amz-date:" + req.Header.Get("x-amz-date") + "\n"
	if c.cfg.SessionToken != "" {
		signedHeaders += ";x-amz-security-token"
		canonicalHeaders += "x-amz-security-token:" + req.Header.Get("x-amz-security-token") + "\n"
	}
	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		req.URL.RawQuery,
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")
	stringToSign := strings.Join([]string{
		"AWS4-HMAC-SHA256",
		req.Header.Get("x-amz-date"),
		scope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")
	signature := hex.EncodeToString(hmacSHA256(deriveSigningKey(c.cfg.SecretKey, date, c.cfg.Region, "s3vectors"), []byte(stringToSign)))
	req.Header.Set("authorization", fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s", c.cfg.AccessKey, scope, signedHeaders, signature))
}

func statusError(status int, body []byte) error {
	code := fmt.Sprintf("HTTP%d", status)
	var decoded map[string]any
	if json.Unmarshal(body, &decoded) == nil {
		for _, key := range []string{"code", "Code", "__type"} {
			if value, ok := decoded[key].(string); ok && value != "" {
				code = value
				break
			}
		}
	}
	kind := ErrorKindClient
	if status == http.StatusTooManyRequests {
		kind = ErrorKindThrottle
	} else if status >= 500 {
		kind = ErrorKindServer
	}
	return OperationError{Kind: kind, Code: code, Err: fmt.Errorf("s3vectors request failed: status=%d body=%s", status, strings.TrimSpace(string(body)))}
}

func isNotFound(err error) bool {
	var opErr OperationError
	if !asOperationError(err, &opErr) {
		return false
	}
	return strings.Contains(strings.ToLower(opErr.Code), "notfound") || opErr.Code == "HTTP404"
}

func asOperationError(err error, target *OperationError) bool {
	return errors.As(err, target)
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func deriveSigningKey(secret, date, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secret), []byte(date))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(service))
	return hmacSHA256(kService, []byte("aws4_request"))
}

func hmacSHA256(key, data []byte) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write(data)
	return mac.Sum(nil)
}
