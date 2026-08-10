from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class TursoConfig(DBConfig):
    db_path: str
    experimental_multiprocess_wal: bool = True

    def to_dict(self) -> dict:
        return {
            "db_path": self.db_path,
            "experimental_multiprocess_wal": self.experimental_multiprocess_wal,
        }


class TursoIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.NONE
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
