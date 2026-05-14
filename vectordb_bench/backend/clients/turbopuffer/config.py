from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class TurboPufferConfig(DBConfig):
    api_key: SecretStr
    region: str
    api_base_url: str | None = None
    namespace: str = "vdbbench_test"
    multitenant_namespace_prefix: str = "vdbbench_mt_"
    scalar_payload_label_field: str = "label"
    pin_namespace: bool = False
    pin_namespace_requested: bool = False
    pin_replicas: int = 1
    pin_timeout: int = 45 * 60
    pin_target_namespace_count: int = 0

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key.get_secret_value(),
            "region": self.region,
            "api_base_url": self.api_base_url,
            "namespace": self.namespace,
            "multitenant_namespace_prefix": self.multitenant_namespace_prefix,
            "scalar_payload_label_field": self.scalar_payload_label_field,
            "pin_namespace": self.pin_namespace,
            "pin_namespace_requested": self.pin_namespace_requested,
            "pin_replicas": self.pin_replicas,
            "pin_timeout": self.pin_timeout,
            "pin_target_namespace_count": self.pin_target_namespace_count,
        }


class TurboPufferIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    use_multi_ns_for_filter: bool = False
    time_wait_warmup: int = 60 * 1  # 1min
    disable_backpressure: bool = False

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
