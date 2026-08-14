from collections.abc import Sequence
from typing import Any, LiteralString, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class LakebaseVectorConfigDict(TypedDict):
    connect_config: dict[str, Any]
    table_name: str


class LakebaseVectorParam(TypedDict):
    metric_fun_op: LiteralString


class LakebaseSessionCommands(TypedDict):
    session_options: Sequence[dict[str, Any]]


class LakebaseVectorConfig(DBConfig):
    user_name: SecretStr = SecretStr("postgres")
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str = "databricks_postgres"
    table_name: str = "vdbbench_table_test"

    def to_dict(self) -> LakebaseVectorConfigDict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        return {
            "connect_config": {
                "host": self.host,
                "port": self.port,
                "dbname": self.db_name,
                "user": user_str,
                "password": pwd_str,
            },
            "table_name": self.table_name,
        }


class LakebaseANNConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True
    index: IndexType = IndexType.LAKEBASE_ANN
    probes: str | None = None
    epsilon: float | None = None
    max_parallel_workers: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vector_l2_ops"
        if self.metric_type == MetricType.IP:
            return "vector_ip_ops"
        if self.metric_type == MetricType.COSINE:
            return "vector_cosine_ops"
        return None

    def parse_metric_fun_op(self) -> LiteralString:
        if self.metric_type == MetricType.L2:
            return "<->"
        if self.metric_type == MetricType.IP:
            return "<#>"
        return "<=>"

    def index_param(self) -> dict[str, Any]:
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "max_parallel_workers": self.max_parallel_workers,
        }

    def search_param(self) -> LakebaseVectorParam:
        return {"metric_fun_op": self.parse_metric_fun_op()}

    def session_param(self) -> LakebaseSessionCommands:
        session_options = []
        if self.probes is not None and self.probes.strip():
            session_options.append(
                {
                    "parameter": {
                        "setting_name": "lakebase_ann.probes",
                        "val": self.probes,
                    },
                },
            )
        if self.epsilon is not None:
            session_options.append(
                {
                    "parameter": {
                        "setting_name": "lakebase_ann.epsilon",
                        "val": str(self.epsilon),
                    },
                },
            )
        return {"session_options": session_options}


_lakebase_search_case_config = {
    IndexType.LAKEBASE_ANN: LakebaseANNConfig,
}
