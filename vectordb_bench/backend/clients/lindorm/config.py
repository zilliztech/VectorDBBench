import click
from pydantic import SecretStr, validator
from vectordb_bench.base import BaseModel
from ..api import DBConfig, DBCaseConfig, IndexType, MetricType


class LindormConfig(DBConfig):
    # Implement the required configuration fields for the database connection
    # ...
    host: str = ""
    port: int = 30070
    user: str = ""
    password: SecretStr = ""
    index_name: str = ""

    def to_dict(self) -> dict:
        return {
            "hosts": [{'host': self.host, 'port': self.port}],
            "http_auth": (self.user, self.password.get_secret_value()),
            "use_ssl": False,
            "http_compress": False,
            "verify_certs": False,
            "ssl_assert_hostname": False,
            "ssl_show_warn": False,
            "timeout": 600,
            "index_name": self.index_name
        }


class LindormIndexConfig(BaseModel):
    index: IndexType
    metric_type: MetricType | None = MetricType.L2

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.IP:
            return "innerproduct"
        elif self.metric_type == MetricType.COSINE:
            return "cosinesimil"
        return "l2"


class HNSWConfig(LindormIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.HNSW
    M: int | None
    efConstruction: int | None
    efSearch: int | None
    filter_type: str | None = "efficient_filter"
    k_expand_scope: int | None = 1000

    def index_param(self, dim: int|None = None) -> dict:
        return {
            "engine": "lvector",
            "name": "hnswq",
            "space_type": self.parse_metric(),
            "parameters": {
                "m": dim if dim is not None else self.M,
                "ef_construction": self.efConstruction,
            }
        }

    def search_param(self, do_filter: bool = False) -> dict:
        search_ext_param = {
            "lvector": {
                "ef_search": str(self.efSearch)
            }
        }
        if do_filter:
            search_ext_param["lvector"]["filter_type"] = self.filter_type
            if self.filter_type == "efficient_filter":
                search_ext_param["lvector"]["k_expand_scope"] = str(self.k_expand_scope)
        return search_ext_param

# first layer searching for cluster centroids is hnsw
class IVFPQConfig(LindormIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.IVFPQ
    nlist: int | None
    nprobe: int | None
    # search parameters
    centroids_hnsw_M: int | None
    centroids_hnsw_efConstruction: int | None
    centroids_hnsw_efSearch: int | None
    filter_type: str | None = "efficient_filter"

    reorder_factor: int | None = 10
    client_refactor: bool = False
    k_expand_scope: int | None = 1000

    def index_param(self) -> dict:
        return {
            "engine": "lvector",
            "name": "ivfpq",
            "space_type": self.parse_metric(),
            "parameters": {
                "nlist": self.nlist,
                "centroids_use_hnsw": True,
                "centroids_hnsw_m": self.centroids_hnsw_M,
                "centroids_hnsw_ef_construct": self.centroids_hnsw_efConstruction,
                "centroids_hnsw_ef_search": self.centroids_hnsw_efSearch
            }
        }

    def search_param(self, do_filter: bool = False) -> dict:
        search_ext_param = {
            "lvector": {
                "nprobe": str(self.nprobe),
                "reorder_factor": str(self.reorder_factor),
                "client_refactor": str(self.client_refactor),
                "ef_search": str(self.centroids_hnsw_efSearch),
            }
        }
        if do_filter:
            search_ext_param["lvector"]["filter_type"] = self.filter_type
            if self.filter_type == "efficient_filter":
                search_ext_param["lvector"]["k_expand_scope"] = str(self.k_expand_scope)
        return search_ext_param


class IVFBQConfig(LindormIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.IVFBQ
    nlist: int | None
    exbits: int | None
    nprobe: int | None
    # search parameters
    centroids_hnsw_M: int | None
    centroids_hnsw_efConstruction: int | None
    centroids_hnsw_efSearch: int | None
    filter_type: str | None = "efficient_filter"

    reorder_factor: int | None = 10
    client_refactor: bool = False
    k_expand_scope: int | None = 1000

    def index_param(self) -> dict:
        return {
            "engine": "lvector",
            "name": "ivfbq",
            "space_type": self.parse_metric(),
            "parameters": {
                "nlist": self.nlist,
                "exbits": self.exbits,
                "centroids_use_hnsw": True,
                "centroids_hnsw_m": self.centroids_hnsw_M,
                "centroids_hnsw_ef_construct": self.centroids_hnsw_efConstruction,
                "centroids_hnsw_ef_search": self.centroids_hnsw_efSearch
            }
        }

    def search_param(self, do_filter: bool = False) -> dict:
        search_ext_param = {
            "lvector": {
                "nprobe": str(self.nprobe),
                "reorder_factor": str(self.reorder_factor),
                "client_refactor": str(self.client_refactor),
            }
        }
        if do_filter:
            search_ext_param["lvector"]["filter_type"] = self.filter_type
            if self.filter_type == "efficient_filter":
                search_ext_param["lvector"]["k_expand_scope"] = str(self.k_expand_scope)
        return search_ext_param

_lindorm_vector_case_config = {
    IndexType.HNSW: HNSWConfig,
    IndexType.IVFPQ: IVFPQConfig,
    IndexType.IVFBQ: IVFBQConfig
}
