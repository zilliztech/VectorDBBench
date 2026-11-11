from pydantic import BaseModel, Field

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class DuckDBConfig(DBConfig):
    db_path: str
    threads: int = Field(default=1, ge=1)

    def to_dict(self) -> dict:
        return {"db_path": self.db_path, "threads": self.threads}


class DuckDBIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.Flat
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
