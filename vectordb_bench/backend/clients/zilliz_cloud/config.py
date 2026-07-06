from pydantic import SecretStr

from ..api import DBCaseConfig, DBConfig
from ..milvus.config import IndexType, MilvusFtsConfig, MilvusIndexConfig


class ZillizCloudConfig(DBConfig):
    uri: SecretStr
    user: str = ""
    password: SecretStr = SecretStr("")
    token: SecretStr = SecretStr("")
    num_shards: int = 1
    collection_name: str = "ZillizCloudVDBBench"

    @staticmethod
    def common_long_configs() -> list[str]:
        return [*DBConfig.common_long_configs(), "user", "password", "token"]

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "user": self.user,
            "password": self.password.get_secret_value(),
            "token": self.token.get_secret_value(),
            "num_shards": self.num_shards,
            "collection_name": self.collection_name,
        }


class AutoIndexConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.AUTOINDEX
    level: int = 1
    num_shards: int = 1

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"shardsNum": self.num_shards},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {
                "level": self.level,
            },
        }


class ZillizCloudFtsConfig(MilvusFtsConfig):
    index_type: str = IndexType.AUTOINDEX.value
    level: int = 1

    def sparse_index_param(self) -> dict:
        params = {}
        if self.bm25_k1 is not None:
            params["bm25_k1"] = self.bm25_k1
        if self.bm25_b is not None:
            params["bm25_b"] = self.bm25_b

        return {
            "index_type": self.index_type,
            "metric_type": self.metric_type.value,
            "params": params,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.metric_type.value,
            "params": {"level": self.level},
        }
