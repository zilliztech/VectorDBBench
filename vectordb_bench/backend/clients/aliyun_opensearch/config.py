import logging

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType

log = logging.getLogger(__name__)


class AliyunOpenSearchConfig(DBConfig, BaseModel):
    host: str = ""
    user: str = ""
    password: SecretStr = ""

    ak: str = ""
    sk: SecretStr = ""
    control_host: str = "searchengine.cn-hangzhou.aliyuncs.com"

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "user": self.user,
            "password": self.password.get_secret_value(),
            "ak": self.ak,
            "sk": self.sk.get_secret_value(),
            "control_host": self.control_host,
        }


class AliyunOpenSearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.L2
    ef_construction: int = 500
    M: int = 100
    ef_search: int = 40

    def distance_type(self) -> str:
        if self.metric_type == MetricType.L2:
            return "SquaredEuclidean"
        if self.metric_type in (MetricType.IP, MetricType.COSINE):
            return "InnerProduct"
        return "SquaredEuclidean"

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
