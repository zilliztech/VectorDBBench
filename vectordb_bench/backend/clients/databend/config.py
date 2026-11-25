from typing import ClassVar, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class DatabendConfigDict(TypedDict):
    user: str
    password: str
    host: str
    port: int
    database: str
    secure: bool


class DatabendConfig(DBConfig):
    # A local Databend has no password for `root`.
    _extra_empty_skip: ClassVar[frozenset[str]] = frozenset({"password"})

    user: str = "root"
    password: SecretStr = SecretStr("")
    host: str = "localhost"
    port: int = 8000
    db_name: str = "default"
    secure: bool = False

    def to_dict(self) -> DatabendConfigDict:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.db_name,
            "user": self.user,
            "password": self.password.get_secret_value(),
            "secure": self.secure,
        }


# The metrics Databend indexes, by their `distance` option.
_METRICS = {MetricType.L2: "l2", MetricType.COSINE: "cosine"}


class DatabendIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    # When the HNSW index is created: before the load (the index is built while
    # blocks are written) or in `optimize`, after the load.
    create_index_before_load: bool = True
    create_index_after_load: bool = False
    m: int | None = None
    ef_construct: int | None = None
    # Stores FP16 data (`refine = true`) and queries with this `vector_refine_factor`;
    # unset: no refine.
    refine_factor: int | None = None
    # Query-time HNSW search width (`vector_search_ef`); unset: the server default.
    ef_search: int | None = None
    # Clusters the table by the vector column in `optimize`: `CLUSTER BY (embedding)`
    # and `RECLUSTER FINAL`, so that each block holds one k-means cluster.
    recluster: bool = False

    def parse_metric_str(self) -> str:
        """The `distance` of the index; its distance function is `<metric>_distance`."""
        metric = _METRICS.get(self.metric_type)
        if metric is None:
            msg = f"Databend does not support metric type {self.metric_type}, only {', '.join(_METRICS)}"
            raise ValueError(msg)
        return metric

    def index_param(self) -> dict:
        return {
            "m": self.m,
            "metric_type": self.parse_metric_str(),
            "ef_construct": self.ef_construct,
            "refine_factor": self.refine_factor,
        }

    def search_param(self) -> dict:
        return {
            "refine_factor": self.refine_factor,
            "ef_search": self.ef_search,
        }
