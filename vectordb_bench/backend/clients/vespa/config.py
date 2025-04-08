from typing import Literal, TypeAlias

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType

VespaMetric: TypeAlias = Literal["euclidean", "angular", "dotproduct", "prenormalized-angular", "hamming", "geodegrees"]

VespaQuantizationType: TypeAlias = Literal["none", "binary"]


class VespaConfig(DBConfig):
    url: SecretStr = "http://127.0.0.1"
    port: int = 8080

    def to_dict(self):
        return {
            "url": self.url.get_secret_value(),
            "port": self.port,
        }


class VespaHNSWConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.COSINE
    quantization_type: VespaQuantizationType = "none"
    M: int = 16
    efConstruction: int = 200
    ef: int = 100

    def index_param(self) -> dict:
        return {
            "distance_metric": self.parse_metric(self.metric_type),
            "max_links_per_node": self.M,
            "neighbors_to_explore_at_insert": self.efConstruction,
        }

    def search_param(self) -> dict:
        return {}

    def parse_metric(self, metric_type: MetricType) -> VespaMetric:
        match metric_type:
            case MetricType.COSINE:
                return "angular"
            case MetricType.L2:
                return "euclidean"
            case MetricType.DP | MetricType.IP:
                return "dotproduct"
            case MetricType.HAMMING:
                return "hamming"
            case _:
                raise NotImplementedError
