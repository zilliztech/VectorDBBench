from pydantic import BaseModel
from ..api import DBConfig, DBCaseConfig, MetricType, TestType
import json


class KnowhereConfig(DBConfig):
    test_type: str = TestType.LIBRARY.value
    index_type: str = "HNSW"
    build_threads: int = 4
    search_threads: int = 4
    config: str = (
        '"M": 30, "efConstruction": 360, "ef": 100, "nlist": 1024, "nprobe": 64'
    )

    def to_dict(self) -> dict:
        return {
            "index_type": self.index_type,
            "config": self.config,
            "search_threads": self.search_threads,
            "build_threads": self.build_threads,
        }

    @property
    def config_json(self):
        config = json.loads(f"{{{self.config}}}")
        config["index_type"] = self.index_type
        return config


class KnowhereIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    nprobe: int | None = None
    ef: int | None = None
    search_list_size: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return MetricType.L2.value
        return self.metric_type.value

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric()
        }
        
    def search_param(self) -> str:
        params = {"metric_type": self.parse_metric()}
        if self.nprobe != None:
            params['nprobe'] = self.nprobe
        if self.ef != None:
            params['ef'] = self.ef
        if self.search_list_size != None:
            params['search_list_size'] = self.search_list_size
        return params
