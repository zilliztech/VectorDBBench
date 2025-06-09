from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class QdrantLocalConfig(DBConfig):
    url: SecretStr

    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
        }


class QdrantLocalIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    m: int
    ef_construct: int
    hnsw_ef: int | None = 0
    on_disk: bool | None = False

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "Euclid"

        if self.metric_type == MetricType.IP:
            return "Dot"

        return "Cosine"

    def index_param(self) -> dict:
        return {
            "distance": self.parse_metric(),
            "m": self.m,
            "ef_construct": self.ef_construct,
            "on_disk": self.on_disk,
        }

    def search_param(self) -> dict:
        search_params = {
            "exact": False,  # Force to use ANNs
        }

        if self.hnsw_ef != 0:
            search_params["hnsw_ef"] = self.hnsw_ef

        return search_params
