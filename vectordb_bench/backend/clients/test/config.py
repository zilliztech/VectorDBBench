from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class TestConfig(DBConfig):
    def to_dict(self) -> dict:
        return {"db_label": self.db_label}


class TestIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
