from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class TurboPufferConfig(DBConfig):
    api_key: SecretStr
    api_base_url: str
    namespace: str = "vdbbench_test"

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key.get_secret_value(),
            "api_base_url": self.api_base_url,
            "namespace": self.namespace,
        }


class TurboPufferIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    use_multi_ns_for_filter: bool = False
    time_wait_warmup: int = 60 * 1  # 1min

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine_distance"
        if self.metric_type == MetricType.L2:
            return "euclidean_squared"

        msg = f"Not Support {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
