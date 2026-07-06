from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class LanceDBConfig(DBConfig):
    """LanceDB connection configuration."""

    db_label: str = ""
    uri: str = "/tmp/lancedb"
    token: SecretStr | None = None
    storage_options: dict[str, str] | None = None

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "token": self.token.get_secret_value() if self.token else None,
            "storage_options": self.storage_options,
        }


class LanceDBIndexConfig(BaseModel, DBCaseConfig):
    """Default IVF_PQ index configuration."""

    index: IndexType = IndexType.IVFPQ
    metric_type: MetricType = MetricType.L2
    num_partitions: int = 0
    num_sub_vectors: int = 0
    nbits: int = 8  # Must be 4 or 8
    sample_rate: int = 256
    max_iterations: int = 50
    nprobes: int = 0
    refine_factor: int = 0

    def parse_metric(self) -> str:
        if self.metric_type in (MetricType.L2, MetricType.COSINE):
            return self.metric_type.value.lower()
        if self.metric_type in (MetricType.IP, MetricType.DP):
            return "dot"
        msg = f"Metric type {self.metric_type} is not supported for LanceDB!"
        raise ValueError(msg)

    def index_param(self) -> dict:
        params = {
            "metric": self.parse_metric(),
            "num_bits": self.nbits,
            "sample_rate": self.sample_rate,
            "max_iterations": self.max_iterations,
        }
        if self.num_partitions > 0:
            params["num_partitions"] = self.num_partitions
        if self.num_sub_vectors > 0:
            params["num_sub_vectors"] = self.num_sub_vectors
        return params

    def search_param(self) -> dict:
        params = {}
        if self.nprobes > 0:
            params["nprobes"] = self.nprobes
        if self.refine_factor > 0:
            params["refine_factor"] = self.refine_factor
        return params


class LanceDBNoIndexConfig(LanceDBIndexConfig):
    """No index — brute-force scan."""

    index: IndexType = IndexType.NONE

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        params = {}
        if self.refine_factor > 0:
            params["refine_factor"] = self.refine_factor
        return params


class LanceDBAutoIndexConfig(LanceDBIndexConfig):
    """AutoIndex — let LanceDB decide."""

    index: IndexType = IndexType.AUTOINDEX

    def index_param(self) -> dict:
        return {"metric": self.parse_metric()}

    def search_param(self) -> dict:
        params = {}
        if self.nprobes > 0:
            params["nprobes"] = self.nprobes
        if self.refine_factor > 0:
            params["refine_factor"] = self.refine_factor
        return params


class LanceDBIVFHNSWSQConfig(BaseModel, DBCaseConfig):
    """IVF_HNSW_SQ index — IVF partitioning + HNSW graph + scalar quantization."""

    index: IndexType = IndexType.IVF_HNSW_SQ
    metric_type: MetricType = MetricType.L2
    num_partitions: int = 0
    m: int = 0
    ef_construction: int = 0
    ef: int = 0
    nprobes: int = 0
    refine_factor: int = 0

    def parse_metric(self) -> str:
        if self.metric_type in (MetricType.L2, MetricType.COSINE):
            return self.metric_type.value.lower()
        if self.metric_type in (MetricType.IP, MetricType.DP):
            return "dot"
        msg = f"Metric type {self.metric_type} is not supported for LanceDB!"
        raise ValueError(msg)

    def index_param(self) -> dict:
        params = {
            "metric": self.parse_metric(),
            "index_type": "IVF_HNSW_SQ",
        }
        if self.num_partitions > 0:
            params["num_partitions"] = self.num_partitions
        if self.m > 0:
            params["m"] = self.m
        if self.ef_construction > 0:
            params["ef_construction"] = self.ef_construction
        return params

    def search_param(self) -> dict:
        params = {}
        if self.ef > 0:
            params["ef"] = self.ef
        if self.nprobes > 0:
            params["nprobes"] = self.nprobes
        if self.refine_factor > 0:
            params["refine_factor"] = self.refine_factor
        return params


class LanceDBIVFHNSWPQConfig(BaseModel, DBCaseConfig):
    """IVF_HNSW_PQ index — IVF partitioning + HNSW graph + product quantization."""

    index: IndexType = IndexType.IVF_HNSW_PQ
    metric_type: MetricType = MetricType.L2
    num_partitions: int = 0
    num_sub_vectors: int = 0
    m: int = 0
    ef_construction: int = 0
    ef: int = 0
    nprobes: int = 0
    refine_factor: int = 0

    def parse_metric(self) -> str:
        if self.metric_type in (MetricType.L2, MetricType.COSINE):
            return self.metric_type.value.lower()
        if self.metric_type in (MetricType.IP, MetricType.DP):
            return "dot"
        msg = f"Metric type {self.metric_type} is not supported for LanceDB!"
        raise ValueError(msg)

    def index_param(self) -> dict:
        params = {
            "metric": self.parse_metric(),
            "index_type": "IVF_HNSW_PQ",
        }
        if self.num_partitions > 0:
            params["num_partitions"] = self.num_partitions
        if self.num_sub_vectors > 0:
            params["num_sub_vectors"] = self.num_sub_vectors
        if self.m > 0:
            params["m"] = self.m
        if self.ef_construction > 0:
            params["ef_construction"] = self.ef_construction
        return params

    def search_param(self) -> dict:
        params = {}
        if self.ef > 0:
            params["ef"] = self.ef
        if self.nprobes > 0:
            params["nprobes"] = self.nprobes
        if self.refine_factor > 0:
            params["refine_factor"] = self.refine_factor
        return params


_lancedb_case_config = {
    IndexType.IVFPQ: LanceDBIndexConfig,
    IndexType.AUTOINDEX: LanceDBAutoIndexConfig,
    IndexType.IVF_HNSW_SQ: LanceDBIVFHNSWSQConfig,
    IndexType.IVF_HNSW_PQ: LanceDBIVFHNSWPQConfig,
    IndexType.HNSW: LanceDBIVFHNSWSQConfig,  # backward compat: HNSW maps to IVF_HNSW_SQ
    IndexType.NONE: LanceDBNoIndexConfig,
}
