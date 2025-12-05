import logging

from pydantic import BaseModel, SecretStr, validator

from ..api import DBCaseConfig, DBConfig, MetricType

log = logging.getLogger(__name__)


class DorisConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 9030
    # Doris FE HTTP port for stream load. Default 8030 (8040 for HTTPS if enabled).
    http_port: int = 8030
    db_name: str = "test"
    ssl: bool = False

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        return v

    def to_dict(self) -> dict:
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "http_port": self.http_port,
            "user": self.user_name,
            "password": pwd_str,
            "database": self.db_name,
        }


class DorisCaseConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    # Optional explicit HNSW/IVF params for convenience
    index_type: str | None = None
    m: int | None = None
    ef_construction: int | None = None
    nlist: int | None = None
    # Arbitrary index properties and session variables
    index_properties: dict[str, str] | None = None
    session_vars: dict[str, str] | None = None
    # Control rows per single stream load request
    stream_load_rows_per_batch: int | None = None
    # Create table without ANN index
    no_index: bool = False

    def get_metric_fn(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_distance_approximate"
        if self.metric_type == MetricType.IP:
            return "inner_product_approximate"
        if self.metric_type == MetricType.COSINE:
            log.debug("Using inner_product_approximate because doris doesn't support cosine as metric type")
            return "inner_product_approximate"
        msg = f"Unsupported metric type: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        # Use exact metric function name for index creation by removing '_approximate' suffix
        metric_type = self.get_metric_fn()
        if metric_type.endswith("_approximate"):
            metric_type = metric_type[: -len("_approximate")]
        props = {"metric_type": metric_type}

        if self.index_type is not None:
            props.setdefault("index_type", self.index_type)
        else:
            props.setdefault("index_type", "hnsw")

        # Merge optional HNSW params
        props["index_type"] = str.lower(props["index_type"])
        if props["index_type"] == "hnsw":
            if self.m is not None:
                props.setdefault("max_degree", str(self.m))
            if self.ef_construction is not None:
                props.setdefault("ef_construction", str(self.ef_construction))
        elif props["index_type"] == "ivf":
            if self.nlist is not None:
                props.setdefault("nlist", str(self.nlist))
        else:
            msg = f"Unsupported index type: {props['index_type']}"
            raise ValueError(msg)

        # Merge user provided index_properties
        if self.index_properties:
            props.update(self.index_properties)
        return props

    def search_param(self) -> dict:
        return {
            "metric_fn": self.get_metric_fn(),
        }

    def session_param(self) -> dict:
        return self.session_vars or {}
