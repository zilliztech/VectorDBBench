from pydantic import BaseModel, SecretStr

from vectordb_bench.backend.clients import MetricType

from ..api import DBCaseConfig, DBConfig


class GarnetDBConfig(DBConfig):
    username: SecretStr | None = None
    password: SecretStr | None = None
    host: SecretStr
    port: int

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "username": self.username.get_secret_value() if self.username is not None else None,
            "password": self.password.get_secret_value() if self.password is not None else None,
        }


class GarnetDBCaseConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    max_degree: int
    l_search: int
    l_build: int

    def index_param(self) -> dict:
        return {
            "metric_type": self.metric_type.value if self.metric_type is not None else "",
            "params": {"max_degree": self.max_degree, "l_build": self.l_build},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.metric_type.value if self.metric_type is not None else "",
            "params": {"l_search": self.l_search},
        }
