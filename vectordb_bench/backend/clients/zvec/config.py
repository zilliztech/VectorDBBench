from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, MetricType


class ZvecConfig(DBConfig):
    """Zvec connection configuration."""

    db_label: str
    path: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
        }


class ZvecIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}


class ZvecHNSWIndexConfig(ZvecIndexConfig):
    M: int | None = 50
    ef_construction: int | None = 500

    ef_search: int | None = 300

    quantize_type: str = ""

    is_using_refiner: bool = False
