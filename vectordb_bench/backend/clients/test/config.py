from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, MetricType


class TestConfig(DBConfig):
    def to_dict(self) -> dict:
        return {"db_label": self.db_label}


class TestIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
