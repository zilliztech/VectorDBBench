from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class LanceDBConfig(DBConfig):
    """LanceDB connection configuration."""

    db_label: str
    uri: str
    token: SecretStr | None = None

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "token": self.token.get_secret_value() if self.token else None,
        }


class LanceDBIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.IVFPQ
    metric_type: MetricType = MetricType.L2
    num_partitions: int = 0
    num_sub_vectors: int = 0
    nbits: int = 8  # Must be 4 or 8
    sample_rate: int = 256
    max_iterations: int = 50

    def index_param(self) -> dict:
        if self.index not in [
            IndexType.IVFPQ,
            IndexType.HNSW,
            IndexType.AUTOINDEX,
            IndexType.NONE,
        ]:
            msg = f"Index type {self.index} is not supported for LanceDB!"
            raise ValueError(msg)

        # See https://lancedb.github.io/lancedb/python/python/#lancedb.table.Table.create_index
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
        pass

    def parse_metric(self) -> str:
        if self.metric_type in [MetricType.L2, MetricType.COSINE]:
            return self.metric_type.value.lower()
        if self.metric_type in [MetricType.IP, MetricType.DP]:
            return "dot"
        msg = f"Metric type {self.metric_type} is not supported for LanceDB!"
        raise ValueError(msg)


class LanceDBNoIndexConfig(LanceDBIndexConfig):
    index: IndexType = IndexType.NONE

    def index_param(self) -> dict:
        return {}


class LanceDBAutoIndexConfig(LanceDBIndexConfig):
    index: IndexType = IndexType.AUTOINDEX

    def index_param(self) -> dict:
        return {}


class LanceDBHNSWIndexConfig(LanceDBIndexConfig):
    index: IndexType = IndexType.HNSW
    m: int = 0
    ef_construction: int = 0

    def index_param(self) -> dict:
        params = LanceDBIndexConfig.index_param(self)

        # See https://lancedb.github.io/lancedb/python/python/#lancedb.index.HnswSq
        params["index_type"] = "IVF_HNSW_SQ"
        if self.m > 0:
            params["m"] = self.m
        if self.ef_construction > 0:
            params["ef_construction"] = self.ef_construction

        return params


_lancedb_case_config = {
    IndexType.IVFPQ: LanceDBIndexConfig,
    IndexType.AUTOINDEX: LanceDBAutoIndexConfig,
    IndexType.HNSW: LanceDBHNSWIndexConfig,
    IndexType.NONE: LanceDBNoIndexConfig,
}
