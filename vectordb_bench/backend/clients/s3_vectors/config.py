from typing import Literal

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class S3VectorsConfig(DBConfig):
    region_name: str = "us-west-2"
    access_key_id: SecretStr
    secret_access_key: SecretStr
    bucket_name: str
    index_name: str = "vdbbench-index"

    endpoint_url: str | None = None
    """Custom endpoint URL (e.g. http://192.168.1.100:8080) for testing against
    a local or on-prem S3 Vectors deployment. When None, uses AWS default."""

    insert_batch_size: int = 100
    """PutVectors per-call batch size. AWS hard limit: 500. Larger means fewer API
    calls (cheaper) but higher per-call latency and memory; smaller means more API
    calls (more throttling risk) but more even latency. Recommended 100-500."""

    max_pool_connections: int = 50
    """urllib3 connection pool size for the boto3 client. Should be >= 2 * the
    ConcurrentInsertRunner worker count to avoid pool starvation. boto3 default
    is 10, which is too low for benchmark workloads."""

    retry_mode: Literal["legacy", "standard", "adaptive"] = "adaptive"
    """boto3 retry mode. 'adaptive' uses a token bucket to slow down on throttling;
    'standard' is fixed-attempt; 'legacy' is the boto3 v1 default. Adaptive
    recommended for S3 Vectors due to AWS service-level rate limits.
    Literal-typed so pydantic rejects invalid values at config-construction time."""

    retry_max_attempts: int = 10
    """Total attempts including the first call. boto3 default is 3-5; raised to
    10 for benchmark stability under temporary throttling."""

    def to_dict(self) -> dict:
        return {
            "region_name": self.region_name,
            "access_key_id": self.access_key_id.get_secret_value() if self.access_key_id else "",
            "secret_access_key": self.secret_access_key.get_secret_value() if self.secret_access_key else "",
            "bucket_name": self.bucket_name,
            "index_name": self.index_name,
            "endpoint_url": self.endpoint_url,
            "insert_batch_size": self.insert_batch_size,
            "max_pool_connections": self.max_pool_connections,
            "retry_mode": self.retry_mode,
            "retry_max_attempts": self.retry_max_attempts,
        }


class S3VectorsIndexConfig(DBCaseConfig, BaseModel):
    """Base config for s3-vectors"""

    metric_type: MetricType | None = None
    data_type: str = "float32"

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type == MetricType.L2:
            return "euclidean"
        msg = f"Unsupported metric type: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
