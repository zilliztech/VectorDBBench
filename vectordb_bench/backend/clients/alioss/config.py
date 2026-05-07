from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class AliOSSConfig(DBConfig):
    access_key_id: SecretStr
    access_key_secret: SecretStr
    region: str = "cn-shenzhen"
    account_id: str
    bucket_name: str
    index_name: str = "vdbbench-index"
    insert_batch_size: int = 100

    def to_dict(self) -> dict:
        return {
            "access_key_id": self.access_key_id.get_secret_value(),
            "access_key_secret": self.access_key_secret.get_secret_value(),
            "region": self.region,
            "account_id": self.account_id,
            "bucket_name": self.bucket_name,
            "index_name": self.index_name,
            "insert_batch_size": self.insert_batch_size,
        }


class AliOSSIndexConfig(DBCaseConfig, BaseModel):
    metric_type: MetricType | None = None
    data_type: str = "float32"

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type == MetricType.L2:
            return "euclidean"
        raise ValueError(f"Unsupported metric type: {self.metric_type}")

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
