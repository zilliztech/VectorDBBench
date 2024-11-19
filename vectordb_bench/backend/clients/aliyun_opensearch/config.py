import logging
from enum import Enum
from pydantic import SecretStr, BaseModel

from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

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
    efConstruction: int = 500
    M: int = 100
    ef_search: int = 40

    def distance_type(self) -> str:
        if self.metric_type == MetricType.L2:
            return "SquaredEuclidean"
        elif self.metric_type == MetricType.IP:
            return "InnerProduct"
        elif self.metric_type == MetricType.COSINE:
            return "InnerProduct"
        return "SquaredEuclidean"

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
