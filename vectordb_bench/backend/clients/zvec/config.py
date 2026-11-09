from typing import Literal

from pydantic import BaseModel, Field

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


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


class ZvecDiskANNIndexConfig(ZvecIndexConfig):
    index: Literal[IndexType.DISKANN] = IndexType.DISKANN
    max_degree: int = Field(default=64, ge=1, le=100)
    build_list_size: int = Field(default=100, ge=10, le=100)
    search_list_size: int = Field(default=300, ge=1)
    pq_chunk_num: int = Field(default=0, ge=0, le=1024)

    def index_param(self) -> dict:
        return {
            "max_degree": self.max_degree,
            "list_size": self.build_list_size,
            "pq_chunk_num": self.pq_chunk_num,
        }

    def search_param(self) -> dict:
        return {"index": self.index, "list_size": self.search_list_size}
