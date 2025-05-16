import logging
from enum import Enum

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType

log = logging.getLogger(__name__)


class AWSOpenSearchConfig(DBConfig, BaseModel):
    host: str = ""
    port: int = 443
    user: str = ""
    password: SecretStr = ""

    def to_dict(self) -> dict:
        return {
            "hosts": [{"host": self.host, "port": self.port}],
            "http_auth": (self.user, self.password.get_secret_value()),
            "use_ssl": True,
            "http_compress": True,
            "verify_certs": True,
            "ssl_assert_hostname": False,
            "ssl_show_warn": False,
            "timeout": 600,
        }


class AWSOS_Engine(Enum):
    nmslib = "nmslib"
    faiss = "faiss"
    lucene = "Lucene"


class AWSOpenSearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.L2
    engine: AWSOS_Engine = AWSOS_Engine.faiss
    engine_name: str | None = None  # 添加与前端一致的参数名
    metric_type_name: str | None = None  # 添加与前端一致的参数名
    efConstruction: int = 256
    efSearch: int = 256
    ef_search: int | None = None  # 添加与前端一致的参数名
    M: int = 16
    index_thread_qty: int | None = 4
    number_of_shards: int | None = 1
    number_of_replicas: int | None = 0
    number_of_segments: int | None = 1
    refresh_interval: str | None = "60s"
    force_merge_enabled: bool | None = True
    flush_threshold_size: str | None = "5120mb"
    number_of_indexing_clients: int | None = 1
    index_thread_qty_during_force_merge: int
    cb_threshold: str | None = "50%"

    def parse_metric(self) -> str:
        # 记录输入参数
        log.info(f"parse_metric called with engine={self.engine}, engine_name={self.engine_name}, metric_type={self.metric_type}, metric_type_name={self.metric_type_name}")
        
        # 确定实际使用的引擎
        engine_value = self.engine
        if self.engine_name is not None:
            try:
                engine_value = AWSOS_Engine[self.engine_name.lower()]
                log.info(f"Using engine from frontend: {engine_value}")
            except (KeyError, ValueError):
                log.warning(f"Invalid engine name: {self.engine_name}, using default: {self.engine}")
        
        # 如果前端传入了明确的度量类型名称，优先使用它
        if self.metric_type_name is not None:
            metric_type_name = self.metric_type_name.lower()
            log.info(f"Using metric_type from frontend: {metric_type_name}")
            
            if metric_type_name == "cosine":
                if engine_value == AWSOS_Engine.faiss:
                    log.info("Using innerproduct because faiss doesn't support cosine as metric type for Opensearch")
                    return "innerproduct"
                log.info("Using cosinesimil for nmslib/lucene engine with cosine metric")
                return "cosinesimil"
            elif metric_type_name == "l2" or metric_type_name == "euclidean":
                log.info("Using l2 metric type")
                return "l2"
            elif metric_type_name == "ip" or metric_type_name == "innerproduct":
                log.info("Using innerproduct metric type")
                return "innerproduct"
            
        # 否则使用原有的逻辑
        if self.metric_type == MetricType.IP:
            log.info("Using innerproduct based on MetricType.IP")
            return "innerproduct"
        if self.metric_type == MetricType.COSINE:
            if engine_value == AWSOS_Engine.faiss:
                log.info(
                    "Using innerproduct because faiss doesn't support cosine as metric type for Opensearch",
                )
                return "innerproduct"
            log.info("Using cosinesimil based on MetricType.COSINE")
            return "cosinesimil"
        log.info("Using l2 as default metric type")
        return "l2"

    def index_param(self) -> dict:
        # 使用 ef_search 参数（如果设置了），否则使用 efSearch
        ef_search_value = self.ef_search if self.ef_search is not None else self.efSearch
        log.info(f"Using ef_search value: {ef_search_value} for index creation")
        
        # 如果前端传入了明确的引擎名称，优先使用它
        engine_value = self.engine
        if self.engine_name is not None:
            try:
                engine_value = AWSOS_Engine[self.engine_name.lower()]
                log.info(f"Using engine from frontend: {engine_value}")
            except (KeyError, ValueError):
                log.warning(f"Invalid engine name: {self.engine_name}, using default: {self.engine}")
        
        # 获取度量类型
        space_type = self.parse_metric()
        log.info(f"Final space_type for index creation: {space_type}")
        
        result = {
            "name": "hnsw",
            "space_type": space_type,
            "engine": engine_value.value,
            "parameters": {
                "ef_construction": self.efConstruction,
                "m": self.M,
                # 从参数中移除 ef_search，它不应该在这里
            },
        }
        
        log.info(f"Final index_param result: {result}")
        return result

    def search_param(self) -> dict:
        return {}
