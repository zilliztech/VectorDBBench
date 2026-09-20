from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, MetricType


class LogosDBConfig(DBConfig):
    uri: str = "/tmp/vectordbbench_logosdb"

    def to_dict(self) -> dict:
        return {"uri": self.uri}


class LogosDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> int:
        import logosdb

        if self.metric_type == MetricType.L2:
            return logosdb.DIST_L2
        if self.metric_type == MetricType.IP:
            return logosdb.DIST_IP
        return logosdb.DIST_COSINE

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
