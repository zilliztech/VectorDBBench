from pydantic import SecretStr, BaseModel, validator
from ..api import DBConfig, DBCaseConfig, MetricType


class TiDBConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 4000
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
            "user": self.user_name,
            "password": pwd_str,
            "database": self.db_name,
            "ssl_verify_cert": self.ssl,
            "ssl_verify_identity": self.ssl,
        }


class TiDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def get_metric_fn(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vec_l2_distance"
        elif self.metric_type == MetricType.COSINE:
            return "vec_cosine_distance"
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

    def index_param(self) -> dict:
        return {
            "metric_fn": self.get_metric_fn(),
        }

    def search_param(self) -> dict:
        return {
            "metric_fn": self.get_metric_fn(),
        }
