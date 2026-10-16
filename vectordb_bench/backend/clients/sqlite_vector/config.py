from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class SQLiteVectorConfig(DBConfig):
    db_path: str

    def to_dict(self) -> dict:
        return {"db_path": self.db_path}


class SQLiteVectorIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.Flat
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
