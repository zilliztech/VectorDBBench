from pydantic import SecretStr

from ..api import DBConfig


class PineconeConfig(DBConfig):
    api_key: SecretStr
    index_name: str
    multitenant_namespace_prefix: str = "vdbbench_mt_"

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key.get_secret_value(),
            "index_name": self.index_name,
            "multitenant_namespace_prefix": self.multitenant_namespace_prefix,
        }
