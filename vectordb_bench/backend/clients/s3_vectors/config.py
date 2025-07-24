from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class S3VectorsConfig(DBConfig):
    region_name: str = "us-west-2"
    access_key_id: SecretStr
    secret_access_key: SecretStr
    bucket_name: str
    index_name: str = "vdbbench-index"

    def to_dict(self) -> dict:
        return {
            "region_name": self.region_name,
            "access_key_id": self.access_key_id.get_secret_value() if self.access_key_id else "",
            "secret_access_key": self.secret_access_key.get_secret_value() if self.secret_access_key else "",
            "bucket_name": self.bucket_name,
            "index_name": self.index_name,
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
