from pydantic import SecretStr

from ..api import DBCaseConfig, DBConfig
from ..milvus.config import MilvusIndexConfig, IndexType


class ZillizCloudConfig(DBConfig):
    uri: SecretStr
    user: str
    password: SecretStr

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "user": self.user,
            "password": self.password.get_secret_value(),
        }


class AutoIndexConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.AUTOINDEX

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }


